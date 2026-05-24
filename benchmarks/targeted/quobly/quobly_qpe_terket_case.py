"""Single guarded case for Quobly-style Heisenberg Trotter-QPE in TerKet."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import compute_circuit_amplitude, normalize_circuit
from terket.benchmarking import measure_callable, runtime_versions, write_rows
from benchmarks.targeted.quobly.quobly_qpe_probe import build_quobly_qpe_heisenberg


@dataclass(frozen=True)
class QuoblyQpeCaseRow:
    n_data: int
    n_phase: int
    n_steps0: int
    trotter_order: int
    n_total: int | None
    qiskit_gate_count: int | None
    qiskit_depth: int | None
    terket_gate_count: int | None
    terket_initial_free: int | None
    terket_quad_eliminated: int | None
    terket_constraint_eliminated: int | None
    terket_remaining_free: int | None
    terket_branches: int | None
    terket_cubic_obstruction: int | None
    terket_gauss_obstruction: int | None
    terket_phase3_backend: str | None
    terket_wall_time_s: float | None
    terket_peak_rss_mb: float | None
    amplitude_abs: float | None
    amplitude_real: float | None
    amplitude_imag: float | None
    python_version: str
    numpy_version: str
    qiskit_version: str
    quimb_version: str
    cotengra_version: str
    status: str
    error_type: str | None
    error_message: str | None

def run_case(n_data: int, n_phase: int, n_steps0: int, trotter_order: int) -> QuoblyQpeCaseRow:
    versions = runtime_versions()
    try:
        qc = build_quobly_qpe_heisenberg(
            n_data,
            n_phase,
            n_steps0=n_steps0,
            trotter_order=trotter_order,
        )
        spec = normalize_circuit(qc)
        bits = (0,) * spec.n_qubits
        wall_time_s, peak_rss_mb, result = measure_callable(
            lambda: compute_circuit_amplitude(
                spec,
                bits,
                bits,
                as_complex=True,
                allow_tensor_contraction=False,
            ),
            repeats=1,
        )
        amplitude, info = result
        amplitude = complex(amplitude)
        return QuoblyQpeCaseRow(
            n_data=n_data,
            n_phase=n_phase,
            n_steps0=n_steps0,
            trotter_order=trotter_order,
            n_total=spec.n_qubits,
            qiskit_gate_count=int(qc.size()),
            qiskit_depth=int(qc.depth()),
            terket_gate_count=int(len(spec.gates)),
            terket_initial_free=int(info["initial_free"]),
            terket_quad_eliminated=int(info["quad_eliminated"]),
            terket_constraint_eliminated=int(info["constraint_eliminated"]),
            terket_remaining_free=int(info["remaining_free"]),
            terket_branches=int(info["branches"]),
            terket_cubic_obstruction=int(info["cubic_obstruction"]),
            terket_gauss_obstruction=int(info["gauss_obstruction"]),
            terket_phase3_backend=str(info.get("phase3_backend") or ""),
            terket_wall_time_s=float(wall_time_s),
            terket_peak_rss_mb=float(peak_rss_mb),
            amplitude_abs=float(abs(amplitude)),
            amplitude_real=float(amplitude.real),
            amplitude_imag=float(amplitude.imag),
            python_version=versions["python_version"],
            numpy_version=versions["numpy_version"],
            qiskit_version=versions["qiskit_version"],
            quimb_version=versions["quimb_version"],
            cotengra_version=versions["cotengra_version"],
            status="ok",
            error_type=None,
            error_message=None,
        )
    except Exception as exc:
        return QuoblyQpeCaseRow(
            n_data=n_data,
            n_phase=n_phase,
            n_steps0=n_steps0,
            trotter_order=trotter_order,
            n_total=None,
            qiskit_gate_count=None,
            qiskit_depth=None,
            terket_gate_count=None,
            terket_initial_free=None,
            terket_quad_eliminated=None,
            terket_constraint_eliminated=None,
            terket_remaining_free=None,
            terket_branches=None,
            terket_cubic_obstruction=None,
            terket_gauss_obstruction=None,
            terket_phase3_backend=None,
            terket_wall_time_s=None,
            terket_peak_rss_mb=None,
            amplitude_abs=None,
            amplitude_real=None,
            amplitude_imag=None,
            python_version=versions["python_version"],
            numpy_version=versions["numpy_version"],
            qiskit_version=versions["qiskit_version"],
            quimb_version=versions["quimb_version"],
            cotengra_version=versions["cotengra_version"],
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-data", type=int, required=True)
    parser.add_argument("--n-phase", type=int, required=True)
    parser.add_argument("--n-steps0", type=int, default=4)
    parser.add_argument("--trotter-order", type=int, default=2)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    row = run_case(args.n_data, args.n_phase, args.n_steps0, args.trotter_order)
    write_rows([row], args.csv)
    print(
        f"n_data={row.n_data} n_phase={row.n_phase} status={row.status} "
        f"backend={row.terket_phase3_backend} remaining={row.terket_remaining_free} "
        f"time={row.terket_wall_time_s} rss={row.terket_peak_rss_mb}"
    )


if __name__ == "__main__":
    main()
