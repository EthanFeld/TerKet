"""Curated benchmark runner with both showcase and fair comparison cases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys

from mqt.bench import get_benchmark_alg


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import analyze_circuit, compute_circuit_amplitude, compute_circuit_amplitude_scaled, normalize_circuit
from terket.benchmarking import (
    count_t_gates,
    measure_callable,
    quimb_amplitude,
    runtime_versions,
    scaled_amplitude_fields,
    warm_up_quimb,
    warm_up_terket,
    write_rows,
)
from terket.benchmarking.head_to_head_cases import resolve_cases as resolve_head_to_head_cases
from terket.benchmarking.structured_cases import resolve_cases as resolve_structured_cases
from terket.circuits import bits_to_big_endian_string


DEFAULT_SHOWCASE_CASES = ("mm_hidden_shift192", "mm_hidden_shift384")
DEFAULT_FAIR_CASES = (
    "mqt:ghz:40",
    "mqt:graphstate:40",
    "mqt:bv:40",
    "mqt:dj:40",
    "mqt:qftentangled:24",
    "mqt:qpeexact:12",
    "mqt:vqe_real_amp:20",
    "mqt:qnn:20",
    "mqt:randomcircuit:8",
)
DEFAULT_TERKET_FRONTIER_CASES = (
    "mqt:qaoa:24",
    "mqt:vqe_two_local:18",
)


@dataclass(frozen=True, slots=True)
class MqtFairCase:
    case: str
    family: str
    size: int
    benchmark: str
    circuit_size: int

    @property
    def name(self) -> str:
        return self.case

    def build_query(self):
        circuit = _bind_deterministic_parameters(
            get_benchmark_alg(self.benchmark, circuit_size=self.circuit_size, random_parameters=False),
            self.benchmark,
            self.circuit_size,
        )
        spec = normalize_circuit(circuit)
        input_bits = (0,) * spec.n_qubits
        output_bits = _hash_bits(f"{self.benchmark}:{self.circuit_size}:output", spec.n_qubits)
        return _SimpleAmplitudeQuery(circuit=spec, input_bits=input_bits, output_bits=output_bits)


@dataclass(frozen=True, slots=True)
class _SimpleAmplitudeQuery:
    circuit: object
    input_bits: tuple[int, ...]
    output_bits: tuple[int, ...]


def _hash_bits(label: str, width: int) -> tuple[int, ...]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    bits = tuple((digest[idx % len(digest)] >> (idx % 8)) & 1 for idx in range(width))
    if width and not any(bits):
        bits = (1,) + bits[1:]
    return bits


def _bind_deterministic_parameters(circuit, benchmark: str, circuit_size: int):
    if not circuit.parameters:
        return circuit

    assignments = {}
    ordered = sorted(circuit.parameters, key=lambda param: param.name)
    for idx, param in enumerate(ordered):
        digest = hashlib.sha256(f"{benchmark}:{circuit_size}:{param.name}:{idx}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="little", signed=False) % 256
        assignments[param] = (2.0 * math.pi * bucket) / 256.0
    return circuit.assign_parameters(assignments, inplace=False)


def resolve_fair_cases(case_names: list[str]):
    standard_names = [name for name in case_names if not name.startswith("mqt:")]
    resolved = list(resolve_head_to_head_cases(standard_names)) if standard_names else []
    for name in case_names:
        if not name.startswith("mqt:"):
            continue
        parts = name.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected MQT fair case in mqt:<benchmark>:<size> form, got {name!r}.")
        _prefix, benchmark, raw_size = parts
        size = int(raw_size)
        resolved.append(
            MqtFairCase(
                case=f"mqt_{benchmark}_{size}",
                family=f"mqt_{benchmark}",
                size=size,
                benchmark=benchmark,
                circuit_size=size,
            )
        )
    return resolved


@dataclass(slots=True)
class CuratedBenchmarkRow:
    section: str
    mode: str
    case: str
    family: str
    size: int
    n_qubits: int
    gate_count: int
    t_count: int
    cubic_obstruction: int
    gauss_obstruction: int | None
    cost_model_r: int
    terket_phase3_backend: str
    terket_wall_time_s: float
    terket_peak_rss_mb: float | None
    quimb_wall_time_s: float | None
    quimb_peak_rss_mb: float | None
    quimb_over_terket_time_ratio: float | None
    abs_error: float | None
    relative_error: float | None
    target_log2_abs: float | None
    wrong_abs: float | None
    target_bits: str | None
    wrong_bits: str | None
    python_version: str
    numpy_version: str
    qiskit_version: str
    quimb_version: str
    cotengra_version: str


def run_showcase_case(case) -> CuratedBenchmarkRow:
    query = case.build_query()
    spec = normalize_circuit(query.circuit)
    analysis = analyze_circuit(spec, query.input_bits, query.output_bits)
    terket_wall_time_s, terket_peak_rss_mb, result = measure_callable(
        lambda: compute_circuit_amplitude_scaled(spec, query.input_bits, query.output_bits),
        1,
    )
    scaled_amp, _ = result
    wrong_amp, _ = compute_circuit_amplitude(spec, query.input_bits, query.wrong_output_bits, as_complex=True)
    _target_real, _target_imag, target_log2_abs, _mantissa_real, _mantissa_imag, _half_pow2_exp = scaled_amplitude_fields(
        scaled_amp
    )
    versions = runtime_versions()

    return CuratedBenchmarkRow(
        section="showcase",
        mode="terket_only",
        case=case.name,
        family=case.family,
        size=case.size,
        n_qubits=spec.n_qubits,
        gate_count=len(spec.gates),
        t_count=count_t_gates(spec),
        cubic_obstruction=int(analysis["cubic_obstruction"]),
        gauss_obstruction=int(analysis["gauss_obstruction"]),
        cost_model_r=int(analysis["cost_model_r"]),
        terket_phase3_backend=str(analysis.get("phase3_backend") or ""),
        terket_wall_time_s=float(terket_wall_time_s),
        terket_peak_rss_mb=float(terket_peak_rss_mb),
        quimb_wall_time_s=None,
        quimb_peak_rss_mb=None,
        quimb_over_terket_time_ratio=None,
        abs_error=None,
        relative_error=None,
        target_log2_abs=float(target_log2_abs),
        wrong_abs=float(abs(complex(wrong_amp))),
        target_bits=bits_to_big_endian_string(query.output_bits),
        wrong_bits=bits_to_big_endian_string(query.wrong_output_bits),
        python_version=versions["python_version"],
        numpy_version=versions["numpy_version"],
        qiskit_version=versions["qiskit_version"],
        quimb_version=versions["quimb_version"],
        cotengra_version=versions["cotengra_version"],
    )


def run_fair_case(case, *, repeats: int, quimb_optimize: str) -> CuratedBenchmarkRow:
    query = case.build_query()
    spec = normalize_circuit(query.circuit)
    analysis = analyze_circuit(spec, query.input_bits, query.output_bits)
    t_count = count_t_gates(spec)

    terket_wall_time_s, terket_peak_rss_mb, terket_result = measure_callable(
        lambda: compute_circuit_amplitude(spec, query.input_bits, query.output_bits, as_complex=True),
        repeats,
    )
    quimb_wall_time_s, quimb_peak_rss_mb, quimb_result = measure_callable(
        lambda: quimb_amplitude(spec, query.input_bits, query.output_bits, optimize=quimb_optimize),
        repeats,
    )
    terket_amp, _ = terket_result
    quimb_amp = quimb_result
    terket_amp = complex(terket_amp)
    quimb_amp = complex(quimb_amp)
    abs_error = abs(terket_amp - quimb_amp)
    relative_error = abs_error / max(abs(terket_amp), abs(quimb_amp), 1e-300)
    versions = runtime_versions()

    return CuratedBenchmarkRow(
        section="fair",
        mode="head_to_head",
        case=case.name,
        family=case.family,
        size=case.size,
        n_qubits=spec.n_qubits,
        gate_count=len(spec.gates),
        t_count=t_count,
        cubic_obstruction=int(analysis["cubic_obstruction"]),
        gauss_obstruction=int(analysis["gauss_obstruction"]),
        cost_model_r=int(analysis["cost_model_r"]),
        terket_phase3_backend=str(analysis.get("phase3_backend") or ""),
        terket_wall_time_s=float(terket_wall_time_s),
        terket_peak_rss_mb=float(terket_peak_rss_mb),
        quimb_wall_time_s=float(quimb_wall_time_s),
        quimb_peak_rss_mb=float(quimb_peak_rss_mb),
        quimb_over_terket_time_ratio=float(quimb_wall_time_s / max(terket_wall_time_s, 1e-12)),
        abs_error=float(abs_error),
        relative_error=float(relative_error),
        target_log2_abs=None,
        wrong_abs=None,
        target_bits=None,
        wrong_bits=None,
        python_version=versions["python_version"],
        numpy_version=versions["numpy_version"],
        qiskit_version=versions["qiskit_version"],
        quimb_version=versions["quimb_version"],
        cotengra_version=versions["cotengra_version"],
    )


def run_terket_frontier_case(case, *, repeats: int) -> CuratedBenchmarkRow:
    query = case.build_query()
    spec = normalize_circuit(query.circuit)
    analysis = analyze_circuit(spec, query.input_bits, query.output_bits)
    t_count = count_t_gates(spec)

    terket_wall_time_s, terket_peak_rss_mb, _terket_result = measure_callable(
        lambda: compute_circuit_amplitude(spec, query.input_bits, query.output_bits, as_complex=True),
        repeats,
    )
    versions = runtime_versions()

    return CuratedBenchmarkRow(
        section="terket_frontier",
        mode="terket_only",
        case=case.name,
        family=case.family,
        size=case.size,
        n_qubits=spec.n_qubits,
        gate_count=len(spec.gates),
        t_count=t_count,
        cubic_obstruction=int(analysis["cubic_obstruction"]),
        gauss_obstruction=int(analysis["gauss_obstruction"]),
        cost_model_r=int(analysis["cost_model_r"]),
        terket_phase3_backend=str(analysis.get("phase3_backend") or ""),
        terket_wall_time_s=float(terket_wall_time_s),
        terket_peak_rss_mb=float(terket_peak_rss_mb),
        quimb_wall_time_s=None,
        quimb_peak_rss_mb=None,
        quimb_over_terket_time_ratio=None,
        abs_error=None,
        relative_error=None,
        target_log2_abs=None,
        wrong_abs=None,
        target_bits=None,
        wrong_bits=None,
        python_version=versions["python_version"],
        numpy_version=versions["numpy_version"],
        qiskit_version=versions["qiskit_version"],
        quimb_version=versions["quimb_version"],
        cotengra_version=versions["cotengra_version"],
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showcase-case", nargs="+", default=list(DEFAULT_SHOWCASE_CASES))
    parser.add_argument("--fair-case", nargs="+", default=list(DEFAULT_FAIR_CASES))
    parser.add_argument("--terket-case", nargs="+", default=list(DEFAULT_TERKET_FRONTIER_CASES))
    parser.add_argument("--repeats", type=int, default=1, help="Timing repeats for fair head-to-head cases.")
    parser.add_argument("--quimb-optimize", default="auto-hq")
    parser.add_argument(
        "--csv",
        type=Path,
        default=RESULTS_ROOT / "curated_benchmark.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    showcase_cases = resolve_structured_cases(args.showcase_case)
    fair_cases = resolve_fair_cases(args.fair_case)
    terket_frontier_cases = resolve_fair_cases(args.terket_case)

    warm_up_terket()
    if fair_cases:
        warm_up_quimb(args.quimb_optimize)

    rows: list[CuratedBenchmarkRow] = []
    for case in showcase_cases:
        row = run_showcase_case(case)
        rows.append(row)
        print(
            f"{row.case}: showcase TerKet={row.terket_wall_time_s:.6f}s, "
            f"log2|amp|={row.target_log2_abs:.3f}, "
            f"wrong_abs={row.wrong_abs:.3e}, "
            f"backend={row.terket_phase3_backend or 'q3_free'}"
        )

    for case in fair_cases:
        row = run_fair_case(case, repeats=args.repeats, quimb_optimize=args.quimb_optimize)
        rows.append(row)
        print(
            f"{row.case}: fair TerKet={row.terket_wall_time_s:.6f}s, "
            f"quimb={row.quimb_wall_time_s:.6f}s, "
            f"ratio={row.quimb_over_terket_time_ratio:.3f}x, "
            f"backend={row.terket_phase3_backend or 'q3_free'}"
        )

    for case in terket_frontier_cases:
        row = run_terket_frontier_case(case, repeats=args.repeats)
        rows.append(row)
        print(
            f"{row.case}: terket-frontier TerKet={row.terket_wall_time_s:.6f}s, "
            f"rss={row.terket_peak_rss_mb:.3f}MB, "
            f"backend={row.terket_phase3_backend or 'q3_free'}"
        )

    write_rows(rows, args.csv)
    print(f"Wrote {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
