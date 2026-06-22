"""Public doubled-sum approximation facade."""

from __future__ import annotations

from typing import Literal, Sequence

from ._amplitude_api import _build_cached_state
from ._doubled_arbitrary import sum_doubled_arbitrary_phase
from ._doubled_core import DoubledSumResult, sum_coupled_doubled_phase, sum_doubled_phase
from ._doubled_factors import DoubledFactorProblem, sum_doubled_factor_problem
from ._engine_runtime_state import _aff_compose_cached
from ._reduction_support import _ReductionContext
from .scaling import ScaledAmplitude, _normalize_scaled_complex
from .spec import _circuit_global_phase_radians, normalize_circuit
from .state import BitSequence, CircuitInput, ExtendedReductionMode, SolverConfig
from ._state_config import _reset_solver_config, _set_solver_config


def _constant_probability_result(state, matches: bool) -> DoubledSumResult:
    scaled = _normalize_scaled_complex(
        abs(complex(state.scalar)) ** 2 if matches else 0j,
        2 * int(state.scalar_half_pow2),
    )
    return DoubledSumResult(
        estimate=ScaledAmplitude.from_tuple(scaled),
        path_variables=0,
        max_difference_weight=0,
        sectors_evaluated=1,
        sectors_total=1,
        exact=True,
        max_reducer_remaining=0,
        phase3_backends=(),
    )


def _zero_probability_result(path_variables: int, max_difference_weight: int) -> DoubledSumResult:
    return DoubledSumResult(
        estimate=ScaledAmplitude(0j),
        path_variables=path_variables,
        max_difference_weight=min(max(0, int(max_difference_weight)), path_variables),
        sectors_evaluated=0,
        sectors_total=1 << path_variables,
        exact=True,
        max_reducer_remaining=0,
        phase3_backends=(),
    )


def _scale_circuit_result(result: DoubledSumResult, state) -> DoubledSumResult:
    scalar_magnitude = abs(complex(state.scalar)) ** 2
    scalar_half_pow2 = 2 * int(state.scalar_half_pow2)

    def scale_amplitude(amplitude: ScaledAmplitude) -> ScaledAmplitude:
        return ScaledAmplitude.from_tuple(_normalize_scaled_complex(
            amplitude.mantissa * scalar_magnitude,
            amplitude.half_pow2_exp + scalar_half_pow2,
        ))

    return DoubledSumResult(
        estimate=scale_amplitude(result.estimate),
        path_variables=result.path_variables,
        max_difference_weight=result.max_difference_weight,
        sectors_evaluated=result.sectors_evaluated,
        sectors_total=result.sectors_total,
        exact=result.exact,
        max_reducer_remaining=result.max_reducer_remaining,
        phase3_backends=result.phase3_backends,
        omitted_magnitude_bound=(
            None
            if result.omitted_magnitude_bound is None
            else scale_amplitude(result.omitted_magnitude_bound)
        ),
    )


def compute_circuit_probability_doubled(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    max_difference_weight: int,
    sector_batch_size: int = 128,
    max_sectors: int | None = None,
    difference_strategy: Literal["hamming", "factor_bound", "general_bound", "subspace"] = "hamming",
    omitted_magnitude_tolerance: float | None = None,
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> DoubledSumResult:
    """Estimate one output probability by truncating doubled-sum difference sectors."""
    spec = normalize_circuit(circuit)
    state = _build_cached_state(
        spec.n_qubits,
        spec.gates,
        input_bits,
        global_phase_radians=_circuit_global_phase_radians(spec),
        extended_reductions=extended_reductions,
    )
    return _compute_state_probability_doubled(
        state,
        output_bits,
        max_difference_weight=max_difference_weight,
        sector_batch_size=sector_batch_size,
        max_sectors=max_sectors,
        difference_strategy=difference_strategy,
        omitted_magnitude_tolerance=omitted_magnitude_tolerance,
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
        solver_config=solver_config,
    )


def _compute_state_probability_doubled(
    state,
    output_bits: BitSequence,
    *,
    max_difference_weight: int,
    sector_batch_size: int,
    max_sectors: int | None,
    difference_strategy: Literal["hamming", "factor_bound", "general_bound", "subspace"],
    omitted_magnitude_tolerance: float | None,
    allow_tensor_contraction: bool,
    extended_reductions: ExtendedReductionMode | str,
    solver_config: SolverConfig | None,
) -> DoubledSumResult:
    if len(output_bits) != state.n:
        raise ValueError(f"Expected {state.n} output bits, received {len(output_bits)}.")
    if int(max_difference_weight) < 0:
        raise ValueError("max_difference_weight must be nonnegative.")
    if sector_batch_size <= 0:
        raise ValueError("sector_batch_size must be positive.")

    token = _set_solver_config(solver_config)
    try:
        if state.m == 0:
            matches = all(state.eps0[idx] == output_bits[idx] for idx in range(state.n))
            return _constant_probability_result(state, matches)

        cache = state._prepare_echelon()
        solved = state._solve_for_output(cache, output_bits)
        if solved is None:
            return _zero_probability_result(cache.n_free, max_difference_weight)

        shift_mask, _, gamma, path_variables = solved
        _, arbitrary_terms = state._transform_arbitrary_phases(shift_mask, gamma)
        context = _ReductionContext(
            preserve_scale=True,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
        )
        q_free = _aff_compose_cached(state.q, shift_mask, gamma, path_variables, context=context)
        if arbitrary_terms:
            if difference_strategy == "factor_bound":
                raise ValueError("factor_bound is unavailable for compact arbitrary phases.")
            if omitted_magnitude_tolerance is not None:
                raise ValueError("Flat arbitrary-phase bounds cannot use omitted_magnitude_tolerance.")
            result = sum_doubled_arbitrary_phase(
                q_free,
                arbitrary_terms,
                max_difference_weight=max_difference_weight,
                sector_batch_size=sector_batch_size,
                max_sectors=max_sectors,
                difference_strategy=difference_strategy,
            )
        else:
            if (
                max_sectors is not None
                or difference_strategy != "hamming"
                or omitted_magnitude_tolerance is not None
            ):
                raise ValueError("Bound sector options require arbitrary-angle circuit factors.")
            result = sum_doubled_phase(
                q_free,
                max_difference_weight=max_difference_weight,
                sector_batch_size=sector_batch_size,
                allow_tensor_contraction=allow_tensor_contraction,
                extended_reductions=extended_reductions,
            )
        return _scale_circuit_result(result, state)
    finally:
        _reset_solver_config(token)


def compute_circuit_pauli_expectation_probabilities_doubled(
    circuit: CircuitInput,
    observables: Sequence[str],
    *,
    input_bits: BitSequence | None = None,
    max_difference_weight: int,
    sector_batch_size: int = 128,
    max_sectors: int | None = None,
    difference_strategy: Literal["hamming", "general_bound", "subspace"] = "hamming",
    allow_tensor_contraction: bool = True,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> list[DoubledSumResult]:
    """Estimate squared Pauli expectations using observable-aware state replay."""
    from ._pauli_api import _build_pauli_expectation_base_state, _prepare_pauli_expectation_request
    from ._state_direct import _build_post_replay_state

    request = _prepare_pauli_expectation_request(circuit, observables, input_bits)
    base_state = _build_pauli_expectation_base_state(
        request,
        extended_reductions=extended_reductions,
    )
    return [
        _compute_state_probability_doubled(
            _build_post_replay_state(base_state, gates, request.inverse_gates),
            request.input_bits,
            max_difference_weight=max_difference_weight,
            sector_batch_size=sector_batch_size,
            max_sectors=max_sectors,
            difference_strategy=difference_strategy,
            omitted_magnitude_tolerance=None,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
            solver_config=solver_config,
        )
        for gates in request.observable_gate_sets
    ]


__all__ = [
    "DoubledFactorProblem",
    "DoubledSumResult",
    "compute_circuit_pauli_expectation_probabilities_doubled",
    "compute_circuit_probability_doubled",
    "sum_coupled_doubled_phase",
    "sum_doubled_factor_problem",
    "sum_doubled_phase",
]
