"""Guarded subprocess helpers for benchmark driver scripts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import Sequence

import psutil


MB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class GuardedSubprocessCsvResult:
    row: dict[str, str]
    runner_status: str
    peak_rss_mb: float
    wall_time_s: float


def default_guarded_rss_limit_mb() -> float:
    total_mb = psutil.virtual_memory().total / MB
    return min(3072.0, max(1536.0, 0.20 * total_mb))


def cleanup_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return

    procs = [parent, *parent.children(recursive=True)]
    for proc in procs:
        try:
            proc.terminate()
        except psutil.Error:
            pass

    _gone, alive = psutil.wait_procs(procs, timeout=5.0)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=5.0)


def read_single_csv_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {csv_path}, got {len(rows)}.")
    return rows[0]


def _process_tree_rss_bytes(process: psutil.Process) -> int:
    rss = 0
    try:
        rss += process.memory_info().rss
        for child in process.children(recursive=True):
            rss += child.memory_info().rss
    except psutil.Error:
        pass
    return rss


def run_guarded_subprocess_csv(
    cmd: Sequence[str],
    *,
    cwd: Path,
    temp_csv: Path,
    missing_row: dict[str, str],
    rss_limit_mb: float,
    timeout_s: float,
    stdout_path: Path | None = None,
    suppress_output: bool = False,
    poll_interval_s: float = 0.2,
) -> GuardedSubprocessCsvResult:
    if temp_csv.exists():
        temp_csv.unlink()
    if stdout_path is not None and stdout_path.exists():
        stdout_path.unlink()

    start = time.perf_counter()
    peak_rss_mb = 0.0
    runner_status = "ok"
    stdout_target = None
    stderr_target = None
    log_handle = None

    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = stdout_path.open("w", encoding="utf-8")
        stdout_target = log_handle
        stderr_target = subprocess.STDOUT
    elif suppress_output:
        stdout_target = subprocess.DEVNULL
        stderr_target = subprocess.DEVNULL

    try:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=stdout_target,
            stderr=stderr_target,
        )
        try:
            ps_proc = psutil.Process(proc.pid)
            while proc.poll() is None:
                peak_rss_mb = max(peak_rss_mb, _process_tree_rss_bytes(ps_proc) / MB)
                if peak_rss_mb > rss_limit_mb:
                    runner_status = "killed_rss_guard"
                    cleanup_process_tree(proc.pid)
                    break
                if time.perf_counter() - start > timeout_s:
                    runner_status = "killed_timeout_guard"
                    cleanup_process_tree(proc.pid)
                    break
                time.sleep(poll_interval_s)
        finally:
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                cleanup_process_tree(proc.pid)
                proc.wait(timeout=5.0)
    finally:
        if log_handle is not None:
            log_handle.close()

    row = read_single_csv_row(temp_csv) if temp_csv.exists() else dict(missing_row)
    return GuardedSubprocessCsvResult(
        row=row,
        runner_status=runner_status if runner_status != "ok" else f"exit_{proc.returncode}",
        peak_rss_mb=peak_rss_mb,
        wall_time_s=time.perf_counter() - start,
    )
