"""Probe Quobly and TerKet performance on QPE benchmark circuits."""

from __future__ import annotations

import argparse
import math
import sys
import time

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT


def _quobly_controlled_pair_mode(n_data: int, n_phase: int) -> str:
    """
    Pick the cheaper exact representation for the controlled two-qubit terms.

    The low-gate ``cx; crz; cx`` form is decisively better on the higher-phase
    frontier (`4:5`, `6:5`) and on smaller `*:3` cases, but `8:3` still solves
    faster with the denser `ccx; crz; ccx` encoding because it leaves a more
    elimination-friendly residual for TerKet's exact reducer.
    """
    if n_phase >= 5 or n_data <= 6:
        return "cx_crz_cx"
    return "ccx_crz_ccx"


def apply_controlled_rotation_term(
    qc: QuantumCircuit,
    control: int,
    q0: int,
    q1: int,
    pauli: str,
    angle: float,
    *,
    pair_mode: str,
) -> None:
    if pauli == "xx":
        qc.h(q0)
        qc.h(q1)
    elif pauli == "yy":
        qc.sx(q0)
        qc.sx(q1)

    if pair_mode == "cx_crz_cx":
        qc.cx(q0, q1)
    else:
        qc.ccx(control, q0, q1)
    qc.crz(angle, control, q1)
    if pair_mode == "cx_crz_cx":
        qc.cx(q0, q1)
    else:
        qc.ccx(control, q0, q1)

    if pauli == "xx":
        qc.h(q0)
        qc.h(q1)
    elif pauli == "yy":
        qc.sxdg(q0)
        qc.sxdg(q1)


def build_quobly_qpe_heisenberg(
    n_data: int,
    n_phase: int,
    *,
    n_steps0: int = 4,
    trotter_order: int = 2,
    e_target: float = -0.5,
    size_interval: float = 2.0,
    pair_mode: str = "auto",
) -> QuantumCircuit:
    qc = QuantumCircuit(n_phase + n_data)
    data = list(range(n_phase, n_phase + n_data))
    phase = list(range(n_phase))
    if pair_mode == "auto":
        pair_mode = _quobly_controlled_pair_mode(n_data, n_phase)
    if pair_mode not in {"cx_crz_cx", "ccx_crz_ccx"}:
        raise ValueError(f"Unsupported pair_mode {pair_mode!r}.")

    evolution_time = 2 * math.pi / size_interval
    emax = e_target + size_interval / 2
    global_phase = emax * evolution_time

    for q in phase:
        qc.h(q)

    dt = evolution_time / n_steps0
    terms: list[tuple[float, str, int, int]] = []
    for i in range(n_data - 1):
        for pauli in ("xx", "yy", "zz"):
            terms.append((0.25, pauli, i, i + 1))

    if trotter_order == 1:
        trotter_terms = terms
        coeff_scale = 1.0
    elif trotter_order == 2:
        trotter_terms = terms + list(reversed(terms))
        coeff_scale = 0.5
    else:
        raise ValueError(f"Unsupported trotter order {trotter_order}.")

    for k, ctrl in enumerate(phase):
        qc.p(global_phase * (2**k), ctrl)
        reps = (2**k) * n_steps0
        for _ in range(reps):
            for theta, pauli, i, j in trotter_terms:
                angle = 2 * theta * dt * coeff_scale
                if pauli == "zz":
                    if pair_mode == "cx_crz_cx":
                        qc.cx(data[i], data[j])
                    else:
                        qc.ccx(ctrl, data[i], data[j])
                    qc.crz(angle, ctrl, data[j])
                    if pair_mode == "cx_crz_cx":
                        qc.cx(data[i], data[j])
                    else:
                        qc.ccx(ctrl, data[i], data[j])
                else:
                    apply_controlled_rotation_term(
                        qc,
                        ctrl,
                        data[i],
                        data[j],
                        pauli,
                        angle,
                        pair_mode=pair_mode,
                    )

    qc.append(QFT(n_phase, inverse=True, do_swaps=True).to_instruction(), phase)
    return qc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-data", type=int, default=4)
    parser.add_argument("--n-phase", type=int, default=2)
    parser.add_argument("--n-steps0", type=int, default=4)
    parser.add_argument("--trotter-order", type=int, default=2)
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    qc = build_quobly_qpe_heisenberg(
        args.n_data,
        args.n_phase,
        n_steps0=args.n_steps0,
        trotter_order=args.trotter_order,
    )
    print("built", time.perf_counter() - t0, qc.num_qubits, qc.size(), qc.depth(), flush=True)

    t1 = time.perf_counter()
    tqc = transpile(qc, basis_gates=["h", "sx", "x", "rz", "cx", "cz"], optimization_level=0)
    print("transpiled", time.perf_counter() - t1, tqc.num_qubits, tqc.size(), tqc.depth(), flush=True)

    if args.analyze:
        sys.path.insert(0, "src")
        from terket import analyze_circuit, compute_circuit_amplitude, normalize_circuit

        t2 = time.perf_counter()
        spec = normalize_circuit(tqc)
        print("normalized", time.perf_counter() - t2, spec.n_qubits, len(spec.gates), flush=True)

        input_bits = (0,) * spec.n_qubits
        output_bits = (0,) * spec.n_qubits

        t3 = time.perf_counter()
        info = analyze_circuit(spec, input_bits, output_bits, allow_tensor_contraction=False)
        print("analysis", time.perf_counter() - t3, info, flush=True)

        t4 = time.perf_counter()
        amp, info2 = compute_circuit_amplitude(
            spec,
            input_bits,
            output_bits,
            as_complex=True,
            allow_tensor_contraction=False,
        )
        print("amplitude", time.perf_counter() - t4, amp, flush=True)
        print("amp_info", info2, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
