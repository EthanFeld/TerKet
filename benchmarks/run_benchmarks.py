"""Unified benchmark entrypoint for all TerKet benchmark scripts."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BENCHMARKS: dict[str, tuple[str, str]] = {
    "curated": (
        "benchmarks.curated_benchmark",
        "Curated benchmark with both impressive showcase cases and fair head-to-head cases.",
    ),
    "head-to-head": (
        "benchmarks.quimb_head_to_head",
        "Direct TerKet versus quimb runtime comparison on fixed benchmark suites.",
    ),
    "structured-showcase": (
        "benchmarks.structured_showcase",
        "Structural showcase cases that surface solver diagnostics and residual width.",
    ),
    "depth-scaling": (
        "benchmarks.targeted.depth_scaling_head_to_head",
        "Depth-parameter sweeps for representative TerKet versus quimb cases.",
    ),
    "amplitude-post-elimination-tensor-rcs": (
        "benchmarks.targeted.rcs.amplitude_post_elimination_tensor_rcs",
        "Probe tensor-network viability on post-elimination q3-free RCS residuals.",
    ),
    "rcs-import-strategy-probe": (
        "benchmarks.targeted.rcs.rcs_import_strategy_probe",
        "Compare structural impact of alternate RCS import strategies.",
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: python benchmarks/run_benchmarks.py curated --repeats 1\n\n"
            "Available benchmark families:\n"
            + "\n".join(f"  {name}: {description}" for name, (_, description) in BENCHMARKS.items())
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("benchmark", choices=sorted(BENCHMARKS), help="Benchmark family to run.")
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the selected benchmark.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    module_name, _description = BENCHMARKS[args.benchmark]
    module = importlib.import_module(module_name)

    old_argv = sys.argv
    forwarded = list(args.benchmark_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    try:
        sys.argv = [f"{Path(old_argv[0]).name} {args.benchmark}", *forwarded]
        module.main()
    finally:
        sys.argv = old_argv

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
