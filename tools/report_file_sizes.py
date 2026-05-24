"""Report repo code-file sizes against bloat refactor thresholds."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = Path(__file__).with_name("file_size_allowlist.json")


@dataclass(frozen=True)
class FileSizePolicy:
    roots: tuple[str, ...]
    extensions: tuple[str, ...]
    report_above: int
    fail_above: int
    allow_over_fail: dict[str, str]


@dataclass(frozen=True)
class FileStat:
    path: str
    lines: int


@dataclass(frozen=True)
class FileSizeSummary:
    total_files: int
    over_report: tuple[FileStat, ...]
    over_fail: tuple[FileStat, ...]
    unexpected_over_fail: tuple[FileStat, ...]
    stale_allowlist: tuple[str, ...]


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> FileSizePolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return FileSizePolicy(
        roots=tuple(str(root) for root in payload["roots"]),
        extensions=tuple(str(ext) for ext in payload["extensions"]),
        report_above=int(payload["report_above"]),
        fail_above=int(payload["fail_above"]),
        allow_over_fail={str(key): str(value) for key, value in payload["allow_over_fail"].items()},
    )


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _iter_code_files(repo_root: Path, policy: FileSizePolicy):
    for root_name in policy.roots:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in policy.extensions:
                yield path


def scan_repo(repo_root: Path, policy: FileSizePolicy) -> tuple[FileStat, ...]:
    stats = [
        FileStat(path=path.relative_to(repo_root).as_posix(), lines=_count_lines(path))
        for path in _iter_code_files(repo_root, policy)
    ]
    return tuple(sorted(stats, key=lambda item: (-item.lines, item.path)))


def summarize_scan(stats: tuple[FileStat, ...], policy: FileSizePolicy) -> FileSizeSummary:
    over_report = tuple(item for item in stats if item.lines > policy.report_above)
    over_fail = tuple(item for item in stats if item.lines > policy.fail_above)
    allowlisted = set(policy.allow_over_fail)
    unexpected_over_fail = tuple(item for item in over_fail if item.path not in allowlisted)
    stale_allowlist = tuple(sorted(path for path in allowlisted if path not in {item.path for item in over_fail}))
    return FileSizeSummary(
        total_files=len(stats),
        over_report=over_report,
        over_fail=over_fail,
        unexpected_over_fail=unexpected_over_fail,
        stale_allowlist=stale_allowlist,
    )


def _rows_to_json(rows: tuple[FileStat, ...]) -> list[dict[str, int | str]]:
    return [asdict(row) for row in rows]


def _render_text(summary: FileSizeSummary, policy: FileSizePolicy, *, top: int) -> str:
    lines = [
        f"scope_roots={','.join(policy.roots)}",
        f"extensions={','.join(policy.extensions)}",
        f"report_above={policy.report_above}",
        f"fail_above={policy.fail_above}",
        f"total_files={summary.total_files}",
        f"over_report={len(summary.over_report)}",
        f"over_fail={len(summary.over_fail)}",
    ]
    if summary.over_fail:
        lines.append("-- over fail threshold --")
        for row in summary.over_fail:
            note = policy.allow_over_fail.get(row.path)
            if note is None:
                lines.append(f"{row.lines}\t{row.path}\tUNEXPECTED")
            else:
                lines.append(f"{row.lines}\t{row.path}\tALLOW\t{note}")
    report_only = [row for row in summary.over_report if row.lines <= policy.fail_above]
    if report_only:
        lines.append("-- over report threshold --")
        for row in report_only[:top]:
            lines.append(f"{row.lines}\t{row.path}")
    if summary.unexpected_over_fail:
        lines.append("-- unexpected over fail threshold --")
        for row in summary.unexpected_over_fail:
            lines.append(f"{row.lines}\t{row.path}")
    if summary.stale_allowlist:
        lines.append("-- stale allowlist entries --")
        lines.extend(summary.stale_allowlist)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    policy = load_policy(args.policy)
    stats = scan_repo(args.root, policy)
    summary = summarize_scan(stats, policy)

    if args.json:
        payload = {
            "policy": {
                "roots": list(policy.roots),
                "extensions": list(policy.extensions),
                "report_above": policy.report_above,
                "fail_above": policy.fail_above,
                "allow_over_fail": dict(policy.allow_over_fail),
            },
            "summary": {
                "total_files": summary.total_files,
                "over_report": _rows_to_json(summary.over_report),
                "over_fail": _rows_to_json(summary.over_fail),
                "unexpected_over_fail": _rows_to_json(summary.unexpected_over_fail),
                "stale_allowlist": list(summary.stale_allowlist),
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_render_text(summary, policy, top=max(1, args.top)))

    if args.check and (summary.unexpected_over_fail or summary.stale_allowlist):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
