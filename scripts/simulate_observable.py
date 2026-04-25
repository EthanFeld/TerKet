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
    args = parser.parse_args()

    source = Path(args.circuit).read_text()
    circuit = terket.normalize_circuit(source)
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


if __name__ == "__main__":
    main()
