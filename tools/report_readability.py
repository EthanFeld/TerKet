"""Report repo readability hotspots against ratcheted policy thresholds."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = Path(__file__).with_name("readability_policy.json")
MODULE_REPLACE_ASSIGN_RE = re.compile(r"^\s*_sys\.modules\[__name__\]\s*=\s*_[A-Za-z0-9_]+", re.MULTILINE)
MODULE_REPLACE_SETATTR_RE = re.compile(
    r'^\s*setattr\(_sys\.modules\[__package__\],\s*__name__\.rpartition\("\."\)\[2\],\s*_[A-Za-z0-9_]+\)',
    re.MULTILINE,
)


@dataclass(frozen=True)
class ReadabilityPolicy:
    roots: tuple[str, ...]
    extensions: tuple[str, ...]
    report_file_above: int
    fail_file_above: int
    report_function_above: int
    fail_function_above: int
    allow_over_file: dict[str, str]
    allow_over_function: dict[str, str]
    allow_missing_module_docstring: dict[str, str]
    allow_module_replace_shims: dict[str, str]


@dataclass(frozen=True)
class FileStat:
    path: str
    lines: int


@dataclass(frozen=True)
class FunctionStat:
    path: str
    qualname: str
    lineno: int
    lines: int

    @property
    def key(self) -> str:
        return f"{self.path}::{self.qualname}:{self.lineno}"


@dataclass(frozen=True)
class ModuleScan:
    file_stat: FileStat
    function_stats: tuple[FunctionStat, ...]
    has_module_docstring: bool
    is_module_replace_shim: bool


@dataclass(frozen=True)
class ReadabilitySummary:
    total_files: int
    total_functions: int
    over_report_files: tuple[FileStat, ...]
    over_fail_files: tuple[FileStat, ...]
    unexpected_over_fail_files: tuple[FileStat, ...]
    stale_allow_over_file: tuple[str, ...]
    over_report_functions: tuple[FunctionStat, ...]
    over_fail_functions: tuple[FunctionStat, ...]
    unexpected_over_fail_functions: tuple[FunctionStat, ...]
    stale_allow_over_function: tuple[str, ...]
    missing_module_docstrings: tuple[str, ...]
    unexpected_missing_module_docstrings: tuple[str, ...]
    stale_allow_missing_module_docstring: tuple[str, ...]
    module_replace_shims: tuple[str, ...]
    unexpected_module_replace_shims: tuple[str, ...]
    stale_allow_module_replace_shim: tuple[str, ...]


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self._stack: list[str] = []
        self.rows: list[FunctionStat] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualname = ".".join([*self._stack, node.name]) if self._stack else node.name
        end_lineno = getattr(node, "end_lineno", node.lineno)
        self.rows.append(
            FunctionStat(
                path="",
                qualname=qualname,
                lineno=node.lineno,
                lines=end_lineno - node.lineno + 1,
            )
        )
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()


def load_policy(path: Path = DEFAULT_POLICY_PATH) -> ReadabilityPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ReadabilityPolicy(
        roots=tuple(str(root) for root in payload["roots"]),
        extensions=tuple(str(ext) for ext in payload["extensions"]),
        report_file_above=int(payload["report_file_above"]),
        fail_file_above=int(payload["fail_file_above"]),
        report_function_above=int(payload["report_function_above"]),
        fail_function_above=int(payload["fail_function_above"]),
        allow_over_file={str(key): str(value) for key, value in payload["allow_over_file"].items()},
        allow_over_function={str(key): str(value) for key, value in payload["allow_over_function"].items()},
        allow_missing_module_docstring={
            str(key): str(value) for key, value in payload["allow_missing_module_docstring"].items()
        },
        allow_module_replace_shims={
            str(key): str(value) for key, value in payload["allow_module_replace_shims"].items()
        },
    )


def _iter_code_files(repo_root: Path, policy: ReadabilityPolicy):
    for root_name in policy.roots:
        root = repo_root / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in policy.extensions:
                yield path


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _is_module_replace_shim(text: str) -> bool:
    return bool(MODULE_REPLACE_ASSIGN_RE.search(text) and MODULE_REPLACE_SETATTR_RE.search(text))


def _scan_module(repo_root: Path, path: Path) -> ModuleScan:
    rel = path.relative_to(repo_root).as_posix()
    text = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    visitor = _FunctionCollector()
    visitor.visit(tree)
    functions = tuple(
        FunctionStat(path=rel, qualname=row.qualname, lineno=row.lineno, lines=row.lines) for row in visitor.rows
    )
    return ModuleScan(
        file_stat=FileStat(path=rel, lines=_count_lines(path)),
        function_stats=functions,
        has_module_docstring=ast.get_docstring(tree) is not None,
        is_module_replace_shim=_is_module_replace_shim(text),
    )


def scan_repo(repo_root: Path, policy: ReadabilityPolicy) -> tuple[ModuleScan, ...]:
    scans = [_scan_module(repo_root, path) for path in _iter_code_files(repo_root, policy)]
    return tuple(sorted(scans, key=lambda item: item.file_stat.path))


def summarize_scan(scans: tuple[ModuleScan, ...], policy: ReadabilityPolicy) -> ReadabilitySummary:
    files = tuple(sorted((scan.file_stat for scan in scans), key=lambda item: (-item.lines, item.path)))
    functions = tuple(
        sorted(
            (fn for scan in scans for fn in scan.function_stats),
            key=lambda item: (-item.lines, item.path, item.lineno, item.qualname),
        )
    )
    missing_module_docstrings = tuple(
        sorted(scan.file_stat.path for scan in scans if not scan.has_module_docstring)
    )
    module_replace_shims = tuple(sorted(scan.file_stat.path for scan in scans if scan.is_module_replace_shim))

    over_report_files = tuple(item for item in files if item.lines > policy.report_file_above)
    over_fail_files = tuple(item for item in files if item.lines > policy.fail_file_above)
    over_report_functions = tuple(item for item in functions if item.lines > policy.report_function_above)
    over_fail_functions = tuple(item for item in functions if item.lines > policy.fail_function_above)

    allow_over_file = set(policy.allow_over_file)
    allow_over_function = set(policy.allow_over_function)
    allow_missing_docstring = set(policy.allow_missing_module_docstring)
    allow_module_replace_shim = set(policy.allow_module_replace_shims)

    over_fail_file_paths = {item.path for item in over_fail_files}
    over_fail_function_keys = {item.key for item in over_fail_functions}
    missing_docstring_paths = set(missing_module_docstrings)
    module_replace_shim_paths = set(module_replace_shims)

    unexpected_over_fail_files = tuple(item for item in over_fail_files if item.path not in allow_over_file)
    unexpected_over_fail_functions = tuple(item for item in over_fail_functions if item.key not in allow_over_function)
    unexpected_missing_module_docstrings = tuple(
        sorted(path for path in missing_module_docstrings if path not in allow_missing_docstring)
    )
    unexpected_module_replace_shims = tuple(
        sorted(path for path in module_replace_shims if path not in allow_module_replace_shim)
    )

    stale_allow_over_file = tuple(sorted(path for path in allow_over_file if path not in over_fail_file_paths))
    stale_allow_over_function = tuple(sorted(key for key in allow_over_function if key not in over_fail_function_keys))
    stale_allow_missing_module_docstring = tuple(
        sorted(path for path in allow_missing_docstring if path not in missing_docstring_paths)
    )
    stale_allow_module_replace_shim = tuple(
        sorted(path for path in allow_module_replace_shim if path not in module_replace_shim_paths)
    )

    return ReadabilitySummary(
        total_files=len(files),
        total_functions=len(functions),
        over_report_files=over_report_files,
        over_fail_files=over_fail_files,
        unexpected_over_fail_files=unexpected_over_fail_files,
        stale_allow_over_file=stale_allow_over_file,
        over_report_functions=over_report_functions,
        over_fail_functions=over_fail_functions,
        unexpected_over_fail_functions=unexpected_over_fail_functions,
        stale_allow_over_function=stale_allow_over_function,
        missing_module_docstrings=missing_module_docstrings,
        unexpected_missing_module_docstrings=unexpected_missing_module_docstrings,
        stale_allow_missing_module_docstring=stale_allow_missing_module_docstring,
        module_replace_shims=module_replace_shims,
        unexpected_module_replace_shims=unexpected_module_replace_shims,
        stale_allow_module_replace_shim=stale_allow_module_replace_shim,
    )


def _rows_to_json(rows: tuple[FileStat, ...] | tuple[FunctionStat, ...]):
    return [asdict(row) for row in rows]


def _render_text(summary: ReadabilitySummary, policy: ReadabilityPolicy, *, top: int) -> str:
    lines = [
        f"scope_roots={','.join(policy.roots)}",
        f"extensions={','.join(policy.extensions)}",
        f"report_file_above={policy.report_file_above}",
        f"fail_file_above={policy.fail_file_above}",
        f"report_function_above={policy.report_function_above}",
        f"fail_function_above={policy.fail_function_above}",
        f"total_files={summary.total_files}",
        f"total_functions={summary.total_functions}",
        f"over_report_files={len(summary.over_report_files)}",
        f"over_fail_files={len(summary.over_fail_files)}",
        f"over_report_functions={len(summary.over_report_functions)}",
        f"over_fail_functions={len(summary.over_fail_functions)}",
        f"missing_module_docstrings={len(summary.missing_module_docstrings)}",
        f"module_replace_shims={len(summary.module_replace_shims)}",
    ]
    if summary.over_fail_files:
        lines.append("-- over fail file threshold --")
        for row in summary.over_fail_files:
            note = policy.allow_over_file.get(row.path)
            suffix = "UNEXPECTED" if note is None else f"ALLOW\t{note}"
            lines.append(f"{row.lines}\t{row.path}\t{suffix}")
    report_only_files = [row for row in summary.over_report_files if row.lines <= policy.fail_file_above]
    if report_only_files:
        lines.append("-- over report file threshold --")
        for row in report_only_files[:top]:
            lines.append(f"{row.lines}\t{row.path}")
    if summary.over_fail_functions:
        lines.append("-- over fail function threshold --")
        for row in summary.over_fail_functions:
            note = policy.allow_over_function.get(row.key)
            suffix = "UNEXPECTED" if note is None else f"ALLOW\t{note}"
            lines.append(f"{row.lines}\t{row.path}\t{row.qualname}:{row.lineno}\t{suffix}")
    report_only_functions = [row for row in summary.over_report_functions if row.lines <= policy.fail_function_above]
    if report_only_functions:
        lines.append("-- over report function threshold --")
        for row in report_only_functions[:top]:
            lines.append(f"{row.lines}\t{row.path}\t{row.qualname}:{row.lineno}")
    if summary.missing_module_docstrings:
        lines.append("-- missing module docstrings --")
        for path in summary.missing_module_docstrings:
            note = policy.allow_missing_module_docstring.get(path)
            suffix = "UNEXPECTED" if note is None else f"ALLOW\t{note}"
            lines.append(f"{path}\t{suffix}")
    if summary.module_replace_shims:
        lines.append("-- module replacement shims --")
        for path in summary.module_replace_shims:
            note = policy.allow_module_replace_shims.get(path)
            suffix = "UNEXPECTED" if note is None else f"ALLOW\t{note}"
            lines.append(f"{path}\t{suffix}")
    if summary.unexpected_over_fail_files:
        lines.append("-- unexpected over fail file threshold --")
        for row in summary.unexpected_over_fail_files:
            lines.append(f"{row.lines}\t{row.path}")
    if summary.unexpected_over_fail_functions:
        lines.append("-- unexpected over fail function threshold --")
        for row in summary.unexpected_over_fail_functions:
            lines.append(f"{row.lines}\t{row.key}")
    if summary.unexpected_missing_module_docstrings:
        lines.append("-- unexpected missing module docstrings --")
        lines.extend(summary.unexpected_missing_module_docstrings)
    if summary.unexpected_module_replace_shims:
        lines.append("-- unexpected module replacement shims --")
        lines.extend(summary.unexpected_module_replace_shims)
    if summary.stale_allow_over_file:
        lines.append("-- stale allow_over_file entries --")
        lines.extend(summary.stale_allow_over_file)
    if summary.stale_allow_over_function:
        lines.append("-- stale allow_over_function entries --")
        lines.extend(summary.stale_allow_over_function)
    if summary.stale_allow_missing_module_docstring:
        lines.append("-- stale allow_missing_module_docstring entries --")
        lines.extend(summary.stale_allow_missing_module_docstring)
    if summary.stale_allow_module_replace_shim:
        lines.append("-- stale allow_module_replace_shim entries --")
        lines.extend(summary.stale_allow_module_replace_shim)
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
    scans = scan_repo(args.root, policy)
    summary = summarize_scan(scans, policy)

    if args.json:
        payload = {
            "policy": {
                "roots": list(policy.roots),
                "extensions": list(policy.extensions),
                "report_file_above": policy.report_file_above,
                "fail_file_above": policy.fail_file_above,
                "report_function_above": policy.report_function_above,
                "fail_function_above": policy.fail_function_above,
                "allow_over_file": dict(policy.allow_over_file),
                "allow_over_function": dict(policy.allow_over_function),
                "allow_missing_module_docstring": dict(policy.allow_missing_module_docstring),
                "allow_module_replace_shims": dict(policy.allow_module_replace_shims),
            },
            "summary": {
                "total_files": summary.total_files,
                "total_functions": summary.total_functions,
                "over_report_files": _rows_to_json(summary.over_report_files),
                "over_fail_files": _rows_to_json(summary.over_fail_files),
                "unexpected_over_fail_files": _rows_to_json(summary.unexpected_over_fail_files),
                "stale_allow_over_file": list(summary.stale_allow_over_file),
                "over_report_functions": _rows_to_json(summary.over_report_functions),
                "over_fail_functions": _rows_to_json(summary.over_fail_functions),
                "unexpected_over_fail_functions": _rows_to_json(summary.unexpected_over_fail_functions),
                "stale_allow_over_function": list(summary.stale_allow_over_function),
                "missing_module_docstrings": list(summary.missing_module_docstrings),
                "unexpected_missing_module_docstrings": list(summary.unexpected_missing_module_docstrings),
                "stale_allow_missing_module_docstring": list(summary.stale_allow_missing_module_docstring),
                "module_replace_shims": list(summary.module_replace_shims),
                "unexpected_module_replace_shims": list(summary.unexpected_module_replace_shims),
                "stale_allow_module_replace_shim": list(summary.stale_allow_module_replace_shim),
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_render_text(summary, policy, top=max(1, args.top)))

    if args.check and (
        summary.unexpected_over_fail_files
        or summary.stale_allow_over_file
        or summary.unexpected_over_fail_functions
        or summary.stale_allow_over_function
        or summary.unexpected_missing_module_docstrings
        or summary.stale_allow_missing_module_docstring
        or summary.unexpected_module_replace_shims
        or summary.stale_allow_module_replace_shim
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
