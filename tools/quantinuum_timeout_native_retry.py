"""Retry Quantinuum challenge timeout cases with the q3-free native batch path."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quantinuum_challenge_terket_graphs import _input_gate_count, _int_field, _load_pytket_json


SNAP_MAX_LEVEL = 19
SNAP_MAX_TOTAL_ERROR = 0.1
CASE_MODES = {
    "unified8192": (8192, "unified", 0),
    "unified32": (32, "unified", 0),
    "strat4_8192": (8192, "stratified", 4),
    "balanced32": (32, "balanced", 0),
}
MODE_CHOICES = tuple(CASE_MODES)


def _case_mode(mode: str) -> dict[str, object]:
    samples, sample_mode, stratified_vars = CASE_MODES[mode]
    return {
        "approx_tensor_residue_forest_samples": samples,
        "approx_tensor_residue_sample_mode": sample_mode,
        "approx_tensor_residue_stratified_vars": stratified_vars,
    }


def _child_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--mode", choices=MODE_CHOICES, required=True)
    args = parser.parse_args(argv)

    from terket import SolverConfig, compute_circuit_amplitude_scaled, snap_arbitrary_angles

    start = time.perf_counter()
    spec, counts = _load_pytket_json(args.json_path)
    spec = snap_arbitrary_angles(
        spec,
        max_level=SNAP_MAX_LEVEL,
        max_total_error=SNAP_MAX_TOTAL_ERROR,
    )
    config_kwargs = _case_mode(args.mode)
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="residue_forest",
        approx_tensor_residue_level=16,
        approx_tensor_reliability_repeats=3,
        **config_kwargs,
    )
    amp, info = compute_circuit_amplitude_scaled(
        spec,
        [0] * spec.n_qubits,
        [0] * spec.n_qubits,
        allow_tensor_contraction=True,
        solver_config=config,
    )
    elapsed_s = time.perf_counter() - start
    result = {
        "status": "ok",
        "mode": args.mode,
        "elapsed_s": elapsed_s,
        "qubits": spec.n_qubits,
        "terket_gates": len(spec.gates),
        "input_gates": sum(counts.values()),
        "log2_abs": amp.log2_abs(),
        "mantissa_real": float(amp.mantissa.real),
        "mantissa_imag": float(amp.mantissa.imag),
        "half_pow2_exp": int(amp.half_pow2_exp),
        "phase3_backend": info.get("phase3_backend") or "",
        "is_zero": bool(info.get("is_zero", False)),
        "initial_free": info.get("initial_free", ""),
        "remaining_free": info.get("remaining_free", ""),
        "branches": info.get("branches", ""),
        "cost_model_r": info.get("cost_model_r", ""),
        "cubic_obstruction": info.get("cubic_obstruction", ""),
        "gauss_obstruction": info.get("gauss_obstruction", ""),
        "snap_phase_count": spec.metadata.get("approximation_phase_count", 0),
        "snap_total_error": spec.metadata.get("approximation_total_error", 0.0),
        "snap_max_error": spec.metadata.get("approximation_max_angle_error", 0.0),
        "snap_basis_size": spec.metadata.get("approximation_basis_size", 0),
    }
    for key in (
        "approx_q3_free_method",
        "approx_q3_free_reliable",
        "approx_q3_free_rejection_reason",
        "approx_q3_free_repeats",
        "approx_q3_free_level",
        "approx_q3_free_samples",
        "approx_q3_free_log2_abs",
        "approx_q3_free_error_log2_abs",
        "approx_q3_free_rel_stderr",
        "approx_q3_free_log2_spread",
        "approx_q3_free_bound_violation_log2",
    ):
        if key in info:
            result[key] = info[key]
    print(json.dumps(result), flush=True)
    return 0


def _load_metadata(challenge_dir: Path) -> dict[str, dict[str, str]]:
    metadata_path = challenge_dir / "metadata.csv"
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        return {row["circuit_name"]: row for row in csv.DictReader(handle)}


def _needs_original_angles(row: dict[str, str]) -> bool:
    return any(_int_field(row, key) for key in ("Rz", "Rx", "ZZPhase", "XXPhase", "PauliExpBox"))


def _timeout_cases(bounded_path: Path, backend_path: Path) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    bounded = json.loads(bounded_path.read_text(encoding="utf-8"))
    for case in bounded.get("cases", []):
        if case.get("status") != "timeout":
            continue
        key = (str(case["circuit_name"]), "unified32")
        if key not in seen:
            seen.add(key)
            cases.append({"circuit_name": str(case["circuit_name"]), "mode": "unified32"})

    backend = json.loads(backend_path.read_text(encoding="utf-8"))
    for case in backend.get("cases", []):
        for run in case.get("runs", []):
            if run.get("status") == "timeout" and run.get("mode") == "strat4_8192":
                key = (str(case["circuit_name"]), "unified8192")
                if key not in seen:
                    seen.add(key)
                    cases.append({"circuit_name": str(case["circuit_name"]), "mode": "unified8192"})
    return cases


def _load_existing(out_path: Path) -> tuple[list[dict[str, object]], set[tuple[str, str]]]:
    if not out_path.exists():
        return [], set()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    rows = list(data.get("cases", []))
    done = {(str(row.get("circuit_name")), str(row.get("mode"))) for row in rows}
    return rows, done


def _write_results(out_path: Path, rows: list[dict[str, object]], timeout_s: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", ""))
        summary[status] = summary.get(status, 0) + 1
    out_path.write_text(
        json.dumps(
            {
                "snap_max_total_error": SNAP_MAX_TOTAL_ERROR,
                "snap_max_level": SNAP_MAX_LEVEL,
                "timeout_s": timeout_s,
                "summary": summary,
                "cases": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _run_child(json_path: Path, *, mode: str, timeout_s: float) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        str(json_path),
        "--mode",
        mode,
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "mode": mode, "elapsed_s": timeout_s}
    elapsed_s = time.perf_counter() - start
    if proc.returncode != 0:
        child_output = proc.stderr.strip() or proc.stdout.strip()
        output_lines = child_output.splitlines()
        return {
            "status": "error",
            "mode": mode,
            "elapsed_s": elapsed_s,
            "error": output_lines[-1][:240] if output_lines else f"child exited {proc.returncode} without output",
            "traceback": proc.stderr.strip(),
            "returncode": proc.returncode,
        }
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "mode": mode,
            "elapsed_s": elapsed_s,
            "error": "child emitted non-json output",
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-1000:],
            "returncode": proc.returncode,
        }
    result["elapsed_s"] = float(result.get("elapsed_s", elapsed_s))
    result["returncode"] = proc.returncode
    return result


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-dir", type=Path, default=Path("tmp/tn_sim_challenge/challenge_files/attachments"))
    parser.add_argument("--out", type=Path, default=Path("results/quantinuum_timeout_native_retry_10min_20260616.json"))
    parser.add_argument("--bounded", type=Path, default=Path("results/quantinuum_challenge_bounded_run_20260616.json"))
    parser.add_argument("--backend", type=Path, default=Path("results/quantinuum_challenge_backend_run_20260616.json"))
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--mode", choices=MODE_CHOICES, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args(argv)

    metadata = _load_metadata(args.challenge_dir)
    cases = _timeout_cases(args.bounded, args.backend)
    if args.family is not None:
        cases = [case for case in cases if metadata[case["circuit_name"]]["family"] == args.family]
    if args.mode is not None:
        cases = [case for case in cases if case["mode"] == args.mode]
    if args.name is not None:
        cases = [case for case in cases if case["circuit_name"] == args.name]
    if args.max_cases is not None:
        cases = cases[: max(0, int(args.max_cases))]
    rows, done = _load_existing(args.out)
    extracted_dir = args.out.parent / "quantinuum_timeout_native_retry_json"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    circuit_zip_path = args.challenge_dir / "circuit_suite.zip"

    try:
        with zipfile.ZipFile(circuit_zip_path) as archive:
            for case in cases:
                name = case["circuit_name"]
                mode = case["mode"]
                if (name, mode) in done:
                    print(f"{name} [{mode}]: skipped existing", flush=True)
                    continue
                row = metadata[name]
                source_dir = "pytket_orig" if _needs_original_angles(row) else "pytket_decomp"
                json_path = extracted_dir / f"{name}.json"
                json_path.write_bytes(archive.read(f"circuit_suite/{source_dir}/{name}.json"))
                result = _run_child(json_path, mode=mode, timeout_s=float(args.timeout_s))
                result.update(
                    {
                        "circuit_name": name,
                        "family": row["family"],
                        "hardness": row["hardness"],
                        "support": "angle_path_sum" if _needs_original_angles(row) else "terket_exact",
                        "input_gates_meta": _input_gate_count(row),
                    }
                )
                rows.append(result)
                done.add((name, mode))
                _write_results(args.out, rows, float(args.timeout_s))
                print(f"{name} [{mode}]: {result['status']} in {result['elapsed_s']:.3g}s", flush=True)
    except Exception:
        _write_results(args.out, rows, float(args.timeout_s))
        traceback.print_exc()
        return 1

    _write_results(args.out, rows, float(args.timeout_s))
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        raise SystemExit(_child_main(sys.argv[2:]))
    raise SystemExit(_main(sys.argv[1:]))
