"""Compute the amplitude <x|U|0> for a QASM circuit U and observable bit string x."""

import argparse
import sys
from pathlib import Path

import terket


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the amplitude <x|U|0> for a QASM circuit.",
    )
    parser.add_argument("circuit", help="Path to an OpenQASM 2.0 file")
    parser.add_argument(
        "observable",
        help="Output bit string x (little-endian: q0 first), e.g. '101'",
    )
    parser.add_argument(
        "--rz-compile-mode",
        default="approx_dyadic",
        choices=("dyadic", "approx_dyadic", "clifford_t"),
        help="How to lower non-dyadic single-qubit phases before simulation.",
    )
    parser.add_argument(
        "--rz-tolerance",
        type=float,
        default=1e-5,
        help="Approximation / synthesis tolerance for non-dyadic single-qubit phases.",
    )
    args = parser.parse_args()

    source = Path(args.circuit).read_text()
    circuit = terket.normalize_circuit(
        source,
        rz_compile_mode=args.rz_compile_mode,
        rz_tolerance=args.rz_tolerance,
    )
    n = circuit.n_qubits

    obs = args.observable
    if len(obs) != n:
        print(
            f"Error: observable length {len(obs)} does not match circuit qubit count {n}",
            file=sys.stderr,
        )
        sys.exit(1)
    if not all(c in "01" for c in obs):
        print("Error: observable must contain only '0' and '1'", file=sys.stderr)
        sys.exit(1)

    input_bits = tuple(0 for _ in range(n))
    output_bits = tuple(int(c) for c in obs)

    amp, _ = terket.compute_circuit_amplitude(
        circuit, input_bits, output_bits, as_complex=True
    )
    amp = complex(amp)

    input_str = "0" * n
    print(f"Circuit:    {Path(args.circuit).name}  ({n} qubits, {len(circuit.gates)} gates)")
    print(f"Input:      |{input_str}>")
    print(f"Output:     |{obs}>")
    print(f"<{obs}|U|{input_str}> = {amp}")
    print(f"|amplitude|^2 = {abs(amp) ** 2}")
    approximation_mode = circuit.metadata.get("approximation_mode")
    if approximation_mode:
        print("Approximation:")
        print(f"  mode={approximation_mode}")
        print(f"  tolerance={circuit.metadata.get('approximation_tolerance', 0.0)}")
        print(f"  basis_size={circuit.metadata.get('approximation_basis_size', 0)}")
        print(f"  approximated_phase_count={circuit.metadata.get('approximation_phase_count', 0)}")
        print(f"  approximated_run_count={circuit.metadata.get('approximation_run_count', 0)}")
        print(f"  total_angle_error={circuit.metadata.get('approximation_total_angle_error', 0.0)}")
        print(f"  max_angle_error={circuit.metadata.get('approximation_max_angle_error', 0.0)}")
        print(f"  total_run_fro_error={circuit.metadata.get('approximation_total_run_fro_error', 0.0)}")
        print(f"  max_run_fro_error={circuit.metadata.get('approximation_max_run_fro_error', 0.0)}")


if __name__ == "__main__":
    main()
