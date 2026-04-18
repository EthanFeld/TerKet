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
    "head-to-head": ("benchmarks.quimb_head_to_head", "TerKet versus quimb benchmark."),
    "structured-showcase": ("benchmarks.structured_showcase", "Structured hidden-shift showcase."),
    "depth-scaling": ("benchmarks.depth_scaling_head_to_head", "Depth-scaling TerKet versus quimb benchmark."),
    "probability-native-rcs": ("benchmarks.probability_native_rcs", "Probability-native RCS benchmark."),
    "amplitude-post-elimination-tensor-rcs": (
        "benchmarks.amplitude_post_elimination_tensor_rcs",
        "Post-elimination tensor-network probe for q3-free amplitude residuals.",
    ),
    "rcs-import-strategy-probe": ("benchmarks.rcs_import_strategy_probe", "Structural probe for RCS import strategies."),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Example: python benchmarks/run_benchmarks.py head-to-head --suite smoke --repeats 1\n\n"
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
