from __future__ import annotations

import sys
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import compute_circuit_amplitude, make_circuit
from terket.circuits import _circuit_global_phase_radians
from terket.engine import _factor_scope_order, _sum_factor_tables_scaled, build_state
from terket.probability_native import (
    ScaledProbability,
    _build_half_phase_probability_factors_scaled,
    _build_raw_output_constrained_q3_free_phase,
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
        self.assertIn("transformed_var_count", info)

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
        self.assertIn(info["method"], {"half_phase_parity_factors", "q3_free_raw_difference"})

    def test_half_phase_factor_model_matches_probability(self):
        circuit = make_circuit(2, [("h", 0), ("cnot", 0, 1)])
        state = build_state(
            circuit.n_qubits,
            circuit.gates,
            [0, 0],
            global_phase_radians=_circuit_global_phase_radians(circuit),
        )
        raw_q = _build_raw_output_constrained_q3_free_phase(state, [0, 0])
        factor_model = _build_half_phase_probability_factors_scaled(raw_q)
        self.assertIsNotNone(factor_model)
        scalar, factors = factor_model
        order, _width = _factor_scope_order(raw_q.n, tuple(factors))
        reduced_total, _max_scope = _sum_factor_tables_scaled(raw_q.n, factors, order, scalar=scalar)
        probability, _info = compute_circuit_probability_scaled(
            circuit,
            [0, 0],
            [0, 0],
            allow_tensor_contraction=False,
        )
        reduced_total = (
            reduced_total[0],
            reduced_total[1] + (2 * int(state.scalar_half_pow2)) - (4 * circuit.n_qubits),
        )
        self.assertAlmostEqual(ScaledProbability.from_tuple(reduced_total).to_float(), probability.to_float(), places=12)

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

    def test_arbitrary_angle_phase_is_not_supported(self):
        circuit = make_circuit(1, [("h", 0), ("rz_arbitrary", 0, 0.123), ("h", 0)])
        with self.assertRaises(NotImplementedError):
            compute_circuit_probability(
                circuit,
                [0],
                [0],
                allow_tensor_contraction=False,
            )


if __name__ == "__main__":
    unittest.main()
