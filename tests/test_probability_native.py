from __future__ import annotations

import sys
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import compute_circuit_amplitude, make_circuit
from terket.engine import (
    ScaledProbability,
    compute_circuit_probability,
    compute_circuit_probability_scaled,
)


class ProbabilityNativeTests(unittest.TestCase):
    def assertProbabilityMatchesAmplitude(self, circuit, input_bits, output_bits):
        amplitude, _ = compute_circuit_amplitude(
            circuit,
            input_bits,
            output_bits,
            as_complex=True,
            allow_tensor_contraction=False,
        )
        probability, info = compute_circuit_probability(
            circuit,
            input_bits,
            output_bits,
            allow_tensor_contraction=False,
        )
        self.assertAlmostEqual(probability, abs(amplitude) ** 2, places=12)
        self.assertIn("method", info)
        self.assertIn("phase3_backend", info)

    def test_probability_matches_amplitude_on_small_q3_free_cases(self):
        cases = [
            (make_circuit(1, [("h", 0)]), [0], [0]),
            (make_circuit(1, [("h", 0), ("t", 0)]), [0], [1]),
            (make_circuit(2, [("h", 0), ("cnot", 0, 1)]), [0, 0], [0, 0]),
            (make_circuit(2, [("h", 0), ("s", 0), ("sx", 1), ("cz", 0, 1), ("t", 1)]), [0, 0], [1, 0]),
            (
                make_circuit(
                    3,
                    [("h", 0), ("h", 1), ("cnot", 0, 1), ("cz", 1, 2), ("t", 2), ("sx", 0)],
                ),
                [0, 0, 0],
                [1, 0, 1],
            ),
        ]
        for circuit, input_bits, output_bits in cases:
            with self.subTest(circuit=circuit, output_bits=tuple(output_bits)):
                self.assertProbabilityMatchesAmplitude(circuit, input_bits, output_bits)

    def test_scaled_probability_round_trips(self):
        circuit = make_circuit(2, [("h", 0), ("cnot", 0, 1)])
        probability, info = compute_circuit_probability_scaled(
            circuit,
            [0, 0],
            [0, 0],
            allow_tensor_contraction=False,
        )
        self.assertIsInstance(probability, ScaledProbability)
        self.assertAlmostEqual(probability.to_float(), 0.5, places=12)
        self.assertEqual(info["method"], "amplitude_square")

    def test_scaled_probability_matches_exact_amplitude_square(self):
        circuit = make_circuit(2, [("h", 0), ("t", 0), ("cnot", 0, 1), ("s", 1)])
        amplitude, _ = compute_circuit_amplitude(
            circuit,
            [0, 0],
            [1, 1],
            allow_tensor_contraction=False,
        )
        probability, _info = compute_circuit_probability_scaled(
            circuit,
            [0, 0],
            [1, 1],
            allow_tensor_contraction=False,
        )
        self.assertAlmostEqual(probability.to_float(), abs(amplitude.to_complex()) ** 2, places=12)

    def test_deterministic_output_probability(self):
        circuit = make_circuit(2, [])
        probability_ok, info_ok = compute_circuit_probability(
            circuit,
            [1, 0],
            [1, 0],
            allow_tensor_contraction=False,
        )
        probability_bad, info_bad = compute_circuit_probability(
            circuit,
            [1, 0],
            [0, 1],
            allow_tensor_contraction=False,
        )
        self.assertEqual(probability_ok, 1.0)
        self.assertEqual(probability_bad, 0.0)
        self.assertFalse(info_ok["is_zero"])
        self.assertTrue(info_bad["is_zero"])

    def test_arbitrary_angle_phase_matches_amplitude(self):
        circuit = make_circuit(1, [("h", 0), ("rz_arbitrary", 0, 0.123), ("h", 0)])
        amplitude, _ = compute_circuit_amplitude(
            circuit,
            [0],
            [0],
            as_complex=True,
            allow_tensor_contraction=False,
        )
        probability, info = compute_circuit_probability(
            circuit,
            [0],
            [0],
            allow_tensor_contraction=False,
        )
        self.assertAlmostEqual(probability, abs(amplitude) ** 2, places=12)
        self.assertEqual(info["method"], "amplitude_square")


if __name__ == "__main__":
    unittest.main()
