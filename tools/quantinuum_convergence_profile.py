"""Compare q3-free residue convergence modes on difficult Quantinuum circuits."""

from __future__ import annotations

import argparse
import cProfile
import csv
import json
import math
from pathlib import Path
import pstats
import subprocess
import sys
import time
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quantinuum_challenge_terket_graphs import _int_field, _load_pytket_json


DEFAULT_CASES = (
    "TFIM_square_PBC_Jz=1.0_hx=2.0_dt=0.5_n_trotter_steps=4_Lx=8_Ly=7",
    "TFIM_honeycomb_PBC_Jz=1.0_hx=3.0_dt=0.3_n_trotter_steps=18_Lx=4_Ly=7",
    "TFIM_honeycomb_PBC_Jz=1.0_hx=6.0_dt=0.5_n_trotter_steps=16_Lx=4_Ly=7",
)
DEFAULT_SAMPLES = (32, 128, 512, 2048)
MODES = ("balanced", "unified")
CHILD_MODES = (*MODES, "unified_random", "unified_dual", "antithetic", "stratified", "boundary_mps")


def _child_run(
    json_path: Path, *, mode: str, samples: int, level: int, max_bond: int = 64
) -> dict[str, object]:
    from terket import SolverConfig, compute_circuit_amplitude_scaled, snap_arbitrary_angles

    start = time.perf_counter()
    spec, counts = _load_pytket_json(json_path)
    spec = snap_arbitrary_angles(spec, max_level=19, max_total_error=0.1)
    config = SolverConfig(
        approx_q3_free_tensor=True,
        approx_tensor_method="boundary_mps" if mode == "boundary_mps" else "residue_forest",
        approx_tensor_max_bond=int(max_bond),
        approx_tensor_residue_level=int(level),
        approx_tensor_residue_forest_samples=int(samples),
        approx_tensor_residue_sample_mode=mode,
        approx_tensor_residue_stratified_vars=4 if mode == "stratified" else 0,
        approx_tensor_reliability_repeats=3,
        approx_tensor_reliability_reject=False,
        approx_tensor_raise_on_unreliable=False,
        approx_tensor_mps_fallback=False,
    )
    amp, info = compute_circuit_amplitude_scaled(
        spec,
        [0] * spec.n_qubits,
        [0] * spec.n_qubits,
        allow_tensor_contraction=True,
        solver_config=config,
    )
    result: dict[str, object] = {
        "status": "ok",
        "mode": mode,
        "samples": samples,
        "level": level,
        "max_bond": max_bond,
        "elapsed_s": time.perf_counter() - start,
        "qubits": spec.n_qubits,
        "input_gates": sum(counts.values()),
        "log2_abs": amp.log2_abs(),
        "mantissa_real": float(amp.mantissa.real),
        "mantissa_imag": float(amp.mantissa.imag),
        "half_pow2_exp": int(amp.half_pow2_exp),
        "snap_total_error": spec.metadata.get("approximation_total_error", 0.0),
    }
    for key, value in info.items():
        if key.startswith("approx_q3_free_"):
            result[key] = value
    return result


def _child_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--mode", choices=CHILD_MODES, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--level", type=int, default=16)
    parser.add_argument("--max-bond", type=int, default=64)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--profile-text", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.profile is None and not args.profile_text:
            result = _child_run(
                args.json_path,
                mode=args.mode,
                samples=args.samples,
                level=args.level,
                max_bond=args.max_bond,
            )
        else:
            profiler = cProfile.Profile()
            result = profiler.runcall(
                _child_run,
                args.json_path,
                mode=args.mode,
                samples=args.samples,
                level=args.level,
                max_bond=args.max_bond,
            )
            if args.profile is not None:
                args.profile.parent.mkdir(parents=True, exist_ok=True)
                profiler.dump_stats(args.profile)
                result["profile_path"] = str(args.profile)
            if args.profile_text:
                pstats.Stats(profiler, stream=sys.stderr).sort_stats("cumtime").print_stats(40)
    except Exception as exc:
        result = {
            "status": "error",
            "mode": args.mode,
            "samples": args.samples,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    print(json.dumps(result), flush=True)
    return 0


def _metadata(challenge_dir: Path) -> dict[str, dict[str, str]]:
    with (challenge_dir / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        return {row["circuit_name"]: row for row in csv.DictReader(handle)}


def _write(path: Path, rows: list[dict[str, object]], timeout_s: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"timeout_s": timeout_s, "samples": DEFAULT_SAMPLES, "cases": rows}, indent=2),
        encoding="utf-8",
    )


def _run_one(
    json_path: Path,
    *,
    mode: str,
    samples: int,
    timeout_s: float,
    profile_path: Path | None,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        str(json_path),
        "--mode",
        mode,
        "--samples",
        str(samples),
    ]
    if profile_path is not None:
        command.extend(("--profile", str(profile_path)))
    start = time.perf_counter()
    try:
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "mode": mode, "samples": samples, "elapsed_s": timeout_s}
    elapsed = time.perf_counter() - start
    if proc.returncode:
        return {
            "status": "error",
            "mode": mode,
            "samples": samples,
            "elapsed_s": elapsed,
            "error": proc.stderr.strip()[-500:],
        }
    result = json.loads(proc.stdout)
    result.setdefault("elapsed_s", elapsed)
    return result


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-dir", type=Path, default=Path("tmp/tn_sim_challenge/challenge_files/attachments"))
    parser.add_argument("--out", type=Path, default=Path("results/quantinuum_convergence_unified_20260619.json"))
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--samples", type=int, nargs="+", default=DEFAULT_SAMPLES)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--json-dir", type=Path, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    metadata = _metadata(args.challenge_dir)
    extracted = args.json_dir or args.out.parent / "quantinuum_convergence_json"
    if args.json_dir is None:
        extracted.mkdir(parents=True, exist_ok=True)
    profiles = args.out.parent / "profiles" / "quantinuum_convergence_20260619"
    rows: list[dict[str, object]] = []
    profile_samples = min(512, max(args.samples))
    profile_case = args.cases[min(1, len(args.cases) - 1)]
    with zipfile.ZipFile(args.challenge_dir / "circuit_suite.zip") as archive:
        for name in args.cases:
            row = metadata[name]
            source = "pytket_orig" if any(
                _int_field(row, key) for key in ("Rz", "Rx", "ZZPhase", "XXPhase", "PauliExpBox")
            ) else "pytket_decomp"
            json_path = extracted / f"{name}.json"
            if args.json_dir is None:
                json_path.write_bytes(archive.read(f"circuit_suite/{source}/{name}.json"))
            for mode in MODES:
                for samples in args.samples:
                    profile_path = (
                        profiles / f"{name}_{mode}_{samples}.pstats"
                        if not args.no_write and name == profile_case and samples == profile_samples
                        else None
                    )
                    result = _run_one(
                        json_path,
                        mode=mode,
                        samples=samples,
                        timeout_s=float(args.timeout_s),
                        profile_path=profile_path,
                    )
                    result.update({"circuit_name": name, "family": row["family"], "hardness": row["hardness"]})
                    rows.append(result)
                    if not args.no_write:
                        _write(args.out, rows, float(args.timeout_s))
                    rel = result.get("approx_q3_free_rel_stderr", math.nan)
                    print(f"{name} {mode} n={samples}: {result['status']} {result.get('elapsed_s', 0):.3g}s rel={rel}", flush=True)
    if args.no_write:
        print("RESULT_JSON=" + json.dumps({"cases": rows}), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        raise SystemExit(_child_main(sys.argv[2:]))
    raise SystemExit(_main(sys.argv[1:]))
