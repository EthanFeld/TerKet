"""Standalone exact probability queries built on TerKet phase reductions.

This module keeps probability-oriented logic separate from the amplitude-native
paths in ``engine.py``. The current implementation targets dyadic q3-free
Schur states with no deferred arbitrary-angle phases.

For a fixed output bitstring ``y`` and the raw q3-free constrained phase

    A_y = 2^{-n} sum_z omega^{q_y(z)},

the exact probability satisfies

    |A_y|^2 = 2^{-2n} sum_{u, v} omega^{q_y(u xor v) - q_y(v)}.

The transformed phase ``q_y(u xor v) - q_y(v)`` is built directly as a new
degree-3 ``PhaseFunction`` over doubled variables, then reduced by the existing
exact TerKet reducers. This stays separate from the amplitude-native solver and
does not route dense q3-free tails to quimb unless the caller explicitly opts
into tensor contraction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, TypedDict, overload

from .circuits import _circuit_global_phase_radians, normalize_circuit
from .cubic_arithmetic import PhaseFunction
from .engine import (
    BitSequence,
    CircuitInput,
    ExtendedReductionMode,
    ReductionInfo,
    SolverConfig,
    _SOLVER_CONFIG_VAR,
    _ONE_SCALED,
    _ReductionContext,
    _add_scaled_complex,
    _apply_affine_bit_in_place,
    _apply_bilinear_phase_in_place,
    _build_cubic_factors_scaled,
    _combine_factor_scaled,
    _factor_scope_order,
    _gauss_obstruction,
    _gauss_sum_q3_free_scaled,
    _info,
    _is_half_phase_q2,
    _mul_scaled_complex,
    _normalize_scaled_complex,
    _omega_scaled_table,
    _phase_function_from_parts,
    _q3_free_graph,
    _q3_free_phase3_backend_name,
    _q3_free_spanning_data,
    _q3_free_treewidth_order,
    _reduce_and_sum_scaled,
    _scale_scaled_complex,
    _select_feedback_vertices,
    _scaled_to_complex,
    _sum_factor_tables_scaled,
    _treewidth_order_width,
    _estimate_treewidth_dp_work,
    build_state,
)


ScaledComplex = tuple[complex, int]


class ProbabilityInfo(ReductionInfo, total=False):
    """Public metadata for the exact probability-native path."""

    method: str
    raw_var_count: int
    raw_q2_terms: int
    output_constraint_count: int
    transformed_var_count: int
    transformed_q2_terms: int
    transformed_q3_terms: int
    effective_factor_count: int
    effective_order_width: int
    effective_max_scope: int


@dataclass(frozen=True)
class ScaledProbability:
    """Probability represented as ``mantissa * 2 ** (half_pow2_exp / 2)``."""

    mantissa: complex
    half_pow2_exp: int = 0

    def __post_init__(self) -> None:
        value, half_pow2_exp = _normalize_scaled_complex(self.mantissa, self.half_pow2_exp)
        object.__setattr__(self, "mantissa", value)
        object.__setattr__(self, "half_pow2_exp", half_pow2_exp)

    @classmethod
    def from_tuple(cls, scaled: ScaledComplex) -> "ScaledProbability":
        value, half_pow2_exp = scaled
        return cls(value, half_pow2_exp)

    def as_tuple(self) -> ScaledComplex:
        return self.mantissa, self.half_pow2_exp

    def to_complex(self) -> complex:
        return _scaled_to_complex(self.as_tuple())

    def to_float(self, *, atol: float = 1e-12) -> float:
        value = self.to_complex()
        scale = max(1.0, abs(value))
        if abs(value.imag) > atol * scale:
            raise ValueError(
                "Probability result retained a non-negligible imaginary part: "
                f"{value.imag!r}."
            )
        real = float(value.real)
        if real < 0.0 and abs(real) <= atol * scale:
            return 0.0
        return real

    def log2(self) -> float:
        if self.mantissa == 0j:
            return -math.inf
        return math.log2(abs(self.mantissa)) + self.half_pow2_exp / 2.0


def _iter_mask_bits(mask: int):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _build_raw_output_constrained_q3_free_phase(
    state,
    output_bits: BitSequence,
) -> PhaseFunction:
    """Construct the exact raw-output q3-free constraint phase for one output."""
    if len(output_bits) != state.n:
        raise ValueError(f"Expected {state.n} output bits, received {len(output_bits)}.")
    if state._arbitrary_phases:
        raise NotImplementedError(
            "Probability-native queries currently do not support deferred arbitrary-angle phases."
        )
    if state.q.q3:
        raise NotImplementedError(
            "Probability-native queries currently require a q3-free Schur kernel."
        )

    lambda_offset = state.q.n
    augmented_q2 = dict(state.q.q2)
    bilinear_half_phase = state.q.mod_q2 // 2

    for lambda_idx, row_mask in enumerate(state.eps):
        dual_var = lambda_offset + lambda_idx
        for var in _iter_mask_bits(int(row_mask)):
            key = (var, dual_var) if var < dual_var else (dual_var, var)
            value = (augmented_q2.get(key, 0) + bilinear_half_phase) % state.q.mod_q2
            if value:
                augmented_q2[key] = value
            elif key in augmented_q2:
                del augmented_q2[key]

    q1 = list(state.q.q1) + ([0] * state.n)
    rhs_linear_coeff = state.q.mod_q1 // 2
    for idx, bit in enumerate(output_bits):
        if (int(bit) ^ int(state.eps0[idx])) & 1:
            q1[lambda_offset + idx] = rhs_linear_coeff

    return _phase_function_from_parts(
        state.q.n + state.n,
        level=state.q.level,
        q0=0,
        q1=q1,
        q2=augmented_q2,
        q3={},
    )


def _build_probability_difference_phase(raw_q: PhaseFunction) -> PhaseFunction:
    """Return ``raw_q(u xor v) - raw_q(v)`` as a degree-3 phase function."""
    if raw_q.q3:
        raise ValueError("The probability-native difference transform expects a q3-free raw phase.")

    base_n = raw_q.n
    total_n = 2 * base_n
    transformed = _phase_function_from_parts(
        total_n,
        level=raw_q.level,
        q0=0,
        q1=[0] * total_n,
        q2={},
        q3={},
    )
    plus_masks = tuple((1 << idx) | (1 << (base_n + idx)) for idx in range(base_n))
    base_masks = tuple(1 << (base_n + idx) for idx in range(base_n))

    for idx, coeff in enumerate(raw_q.q1):
        coeff = int(coeff) % raw_q.mod_q1
        if not coeff:
            continue
        _apply_affine_bit_in_place(transformed, plus_masks[idx], 0, coeff)
        _apply_affine_bit_in_place(transformed, base_masks[idx], 0, (-coeff) % raw_q.mod_q1)

    for (left, right), coeff in raw_q.q2.items():
        coeff = int(coeff) % raw_q.mod_q2
        if not coeff:
            continue
        _apply_bilinear_phase_in_place(
            transformed,
            plus_masks[left],
            0,
            plus_masks[right],
            0,
            coeff,
        )
        _apply_bilinear_phase_in_place(
            transformed,
            base_masks[left],
            0,
            base_masks[right],
            0,
            (-coeff) % raw_q.mod_q2,
        )

    return transformed


def _build_half_phase_probability_factors_scaled(
    raw_q: PhaseFunction,
) -> tuple[ScaledComplex, dict[tuple[int, ...], list[ScaledComplex]]] | None:
    """Return the exact ``u``-only factor model for half-phase raw q3-free kernels."""
    if not _is_half_phase_q2(raw_q):
        return None

    scalar, factors = _build_cubic_factors_scaled(raw_q)
    adjacency = [set() for _ in range(raw_q.n)]
    for (left, right), coeff in raw_q.q2.items():
        if coeff % raw_q.mod_q2:
            adjacency[left].add(right)
            adjacency[right].add(left)

    omega_scaled = _omega_scaled_table(raw_q.level)
    half_q1 = raw_q.mod_q1 // 2

    for var in range(raw_q.n):
        neighbors = tuple(sorted(adjacency[var]))
        unary_shift = (-2 * (raw_q.q1[var] % raw_q.mod_q1)) % raw_q.mod_q1
        include_var = unary_shift != 0
        scope_vars = list(neighbors)
        if include_var:
            scope_vars.append(var)
        scope = tuple(sorted(scope_vars))
        scope_positions = {vertex: idx for idx, vertex in enumerate(scope)}
        table: list[ScaledComplex] = []

        for assignment in range(1 << len(scope)):
            beta = 0
            if include_var and ((assignment >> scope_positions[var]) & 1):
                beta = unary_shift
            for neighbor in neighbors:
                if (assignment >> scope_positions[neighbor]) & 1:
                    beta = (beta + half_q1) % raw_q.mod_q1
            table.append(_add_scaled_complex(_ONE_SCALED, omega_scaled[beta]))

        scalar = _mul_scaled_complex(
            scalar,
            _combine_factor_scaled(factors, scope, table),
        )

    return scalar, factors


def _estimate_factor_table_elimination_work(
    factor_scopes: tuple[tuple[int, ...], ...],
    order: list[int],
) -> int:
    """Cheap work proxy mirroring TerKet's generic factor-table elimination."""
    factors = {tuple(scope) for scope in factor_scopes if scope}
    work = 0

    for var in order:
        bucket_scopes = [scope for scope in factors if var in scope]
        if not bucket_scopes:
            work += 1
            continue

        for scope in bucket_scopes:
            factors.remove(scope)
        union_scope = tuple(sorted({vertex for scope in bucket_scopes for vertex in scope}))
        new_scope = tuple(vertex for vertex in union_scope if vertex != var)
        work += len(bucket_scopes) * (1 << len(union_scope))
        if new_scope:
            factors.add(new_scope)

    return work


def _estimate_transformed_q3_free_work(
    transformed_q: PhaseFunction,
) -> tuple[int, int] | None:
    """Return ``(width, work)`` for the direct q3-free difference kernel when viable."""
    if transformed_q.q3:
        return None

    adjacency, edges = _q3_free_graph(transformed_q)
    depth, chords = _q3_free_spanning_data(adjacency, edges)
    feedback_vars = _select_feedback_vertices(transformed_q.n, chords, depth)
    max_degree = max((len(neighbors) for neighbors in adjacency), default=0)
    order = _q3_free_treewidth_order(
        transformed_q,
        len(feedback_vars),
        max_degree=max_degree,
    )
    if order is None:
        return None
    width = _treewidth_order_width(transformed_q, order)
    work = _estimate_treewidth_dp_work(transformed_q, order)
    return int(width), int(work)


def _probability_info_from_reducer(
    *,
    raw_q: PhaseFunction,
    transformed_q: PhaseFunction,
    output_constraint_count: int,
    reducer_info: dict[str, int | str | None],
    zero: bool,
) -> ProbabilityInfo:
    info = _info(
        transformed_q.n,
        int(reducer_info["quad"]),
        int(reducer_info["constraint"]),
        int(reducer_info["branched"]),
        int(reducer_info["remaining"]),
        structural_obstruction=int(reducer_info.get("structural_obstruction", reducer_info["remaining"])),
        gauss_obstruction=int(
            reducer_info.get(
                "gauss_obstruction",
                reducer_info.get("structural_obstruction", reducer_info["remaining"]),
            )
        ),
        phase_states=int(reducer_info.get("phase_states", 0)),
        phase_splits=int(reducer_info.get("phase_splits", 0)),
        cost_model_r=int(reducer_info.get("cost_r", reducer_info["remaining"])),
        phase3_backend=(
            None
            if reducer_info.get("phase3_backend") is None
            else str(reducer_info.get("phase3_backend"))
        ),
        zero=zero,
    )
    probability_info: ProbabilityInfo = dict(info)
    probability_info.update(
        {
            "method": "q3_free_raw_difference",
            "raw_var_count": raw_q.n,
            "raw_q2_terms": len(raw_q.q2),
            "output_constraint_count": output_constraint_count,
            "transformed_var_count": transformed_q.n,
            "transformed_q2_terms": len(transformed_q.q2),
            "transformed_q3_terms": len(transformed_q.q3),
        }
    )
    return probability_info


def _probability_info_from_factor_model(
    *,
    raw_q: PhaseFunction,
    output_constraint_count: int,
    factor_count: int,
    order_width: int,
    max_scope: int,
    zero: bool,
) -> ProbabilityInfo:
    info = _info(
        raw_q.n,
        0,
        0,
        0,
        0,
        structural_obstruction=0,
        gauss_obstruction=0,
        cost_model_r=order_width,
        phase3_backend="half_phase_probability_factors",
        zero=zero,
    )
    probability_info: ProbabilityInfo = dict(info)
    probability_info.update(
        {
            "method": "half_phase_parity_factors",
            "raw_var_count": raw_q.n,
            "raw_q2_terms": len(raw_q.q2),
            "output_constraint_count": output_constraint_count,
            "transformed_var_count": raw_q.n,
            "transformed_q2_terms": len(raw_q.q2),
            "transformed_q3_terms": 0,
            "effective_factor_count": factor_count,
            "effective_order_width": order_width,
            "effective_max_scope": max_scope,
        }
    )
    return probability_info


def _compute_state_probability_scaled(
    state,
    output_bits: BitSequence,
    *,
    allow_tensor_contraction: bool,
    extended_reductions: ExtendedReductionMode | str,
) -> tuple[ScaledProbability, ProbabilityInfo]:
    if len(output_bits) != state.n:
        raise ValueError(f"Expected {state.n} output bits, received {len(output_bits)}.")

    if state.m == 0:
        ok = all(int(state.eps0[idx]) == int(output_bits[idx]) for idx in range(state.n))
        if ok:
            scaled = _normalize_scaled_complex(abs(complex(state.scalar)) ** 2, 2 * int(state.scalar_half_pow2))
            info: ProbabilityInfo = {
                "method": "deterministic",
                "raw_var_count": 0,
                "raw_q2_terms": 0,
                "output_constraint_count": state.n,
                "transformed_var_count": 0,
                "transformed_q2_terms": 0,
                "transformed_q3_terms": 0,
                **_info(0, 0, 0, 0, 0, zero=False, cost_model_r=0, phase3_backend=None),
            }
            return ScaledProbability.from_tuple(scaled), info
        zero = ScaledProbability(0j, 0)
        info = {
            "method": "deterministic",
            "raw_var_count": 0,
            "raw_q2_terms": 0,
            "output_constraint_count": state.n,
            "transformed_var_count": 0,
            "transformed_q2_terms": 0,
            "transformed_q3_terms": 0,
            **_info(0, 0, 0, 0, 0, zero=True, cost_model_r=0, phase3_backend=None),
        }
        return zero, info

    raw_q = _build_raw_output_constrained_q3_free_phase(state, output_bits)
    factor_model = _build_half_phase_probability_factors_scaled(raw_q)
    transformed_q = None
    use_factor_model = False
    if factor_model is not None:
        scalar, factors = factor_model
        factor_count = len(factors)
        factor_scopes = tuple(factors)
        order, order_width = _factor_scope_order(raw_q.n, factor_scopes)
        factor_work = _estimate_factor_table_elimination_work(factor_scopes, order)
        transformed_q = _build_probability_difference_phase(raw_q)
        transformed_work = _estimate_transformed_q3_free_work(transformed_q)
        if transformed_q.q3:
            use_factor_model = True
        elif transformed_work is not None and factor_work < transformed_work[1]:
            use_factor_model = True
        else:
            use_factor_model = False

    if factor_model is not None and use_factor_model:
        reduced_total, max_scope = _sum_factor_tables_scaled(
            raw_q.n,
            factors,
            order,
            scalar=scalar,
        )
        probability_info = _probability_info_from_factor_model(
            raw_q=raw_q,
            output_constraint_count=state.n,
            factor_count=factor_count,
            order_width=order_width,
            max_scope=max_scope,
            zero=reduced_total[0] == 0j,
        )
    else:
        if transformed_q is None:
            transformed_q = _build_probability_difference_phase(raw_q)
        if transformed_q.q3:
            context = _ReductionContext(
                preserve_scale=True,
                allow_tensor_contraction=allow_tensor_contraction,
                extended_reductions=extended_reductions,
            )
            reduced_total, reducer_info = _reduce_and_sum_scaled(transformed_q, context=context)
        else:
            reduced_total, phase_info = _gauss_sum_q3_free_scaled(
                transformed_q,
                allow_tensor_contraction=allow_tensor_contraction,
            )
            reducer_info = {
                "quad": 0,
                "constraint": 0,
                "branched": 0,
                "remaining": 0,
                "structural_obstruction": 0,
                "gauss_obstruction": _gauss_obstruction(transformed_q, 0),
                "cost_r": 0,
                "phase_states": int(phase_info.get("phase_states", 0)),
                "phase_splits": int(phase_info.get("phase_splits", 0)),
                "phase3_backend": _q3_free_phase3_backend_name(transformed_q),
            }
        probability_info = _probability_info_from_reducer(
            raw_q=raw_q,
            transformed_q=transformed_q,
            output_constraint_count=state.n,
            reducer_info=reducer_info,
            zero=reduced_total[0] == 0j,
        )
    reduced_total = _scale_scaled_complex(
        reduced_total,
        (2 * int(state.scalar_half_pow2)) - (4 * state.n),
    )
    scalar_abs_sq = abs(complex(state.scalar)) ** 2
    if scalar_abs_sq != 1.0:
        reduced_total = _normalize_scaled_complex(reduced_total[0] * scalar_abs_sq, reduced_total[1])

    probability = ScaledProbability.from_tuple(reduced_total)
    probability_info["is_zero"] = probability.mantissa == 0j
    return probability, probability_info


@overload
def compute_circuit_probability(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    as_float: Literal[True] = True,
    allow_tensor_contraction: bool = False,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> tuple[float, ProbabilityInfo]:
    ...


@overload
def compute_circuit_probability(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    as_float: Literal[False],
    allow_tensor_contraction: bool = False,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> tuple[ScaledProbability, ProbabilityInfo]:
    ...


def compute_circuit_probability(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    as_float: bool = True,
    allow_tensor_contraction: bool = False,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> tuple[float | ScaledProbability, ProbabilityInfo]:
    """Compute one exact output probability using the standalone native path.

    The current implementation supports q3-free dyadic circuits with no
    deferred arbitrary-angle phases. By default tensor contraction is disabled
    so the query stays on TerKet backends only.
    Pass a ``SolverConfig`` to tune cutset and tensor-hint preferences.
    """
    spec = normalize_circuit(circuit)
    state = build_state(
        spec.n_qubits,
        spec.gates,
        input_bits,
        global_phase_radians=_circuit_global_phase_radians(spec),
        extended_reductions=extended_reductions,
    )
    _token = _SOLVER_CONFIG_VAR.set(solver_config) if solver_config is not None else None
    try:
        probability, info = _compute_state_probability_scaled(
            state,
            output_bits,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
        )
    finally:
        if _token is not None:
            _SOLVER_CONFIG_VAR.reset(_token)
    return (probability.to_float(), info) if as_float else (probability, info)


def compute_circuit_probability_scaled(
    circuit: CircuitInput,
    input_bits: BitSequence,
    output_bits: BitSequence,
    *,
    allow_tensor_contraction: bool = False,
    extended_reductions: ExtendedReductionMode | str = "auto",
    solver_config: SolverConfig | None = None,
) -> tuple[ScaledProbability, ProbabilityInfo]:
    """Compute one exact probability without collapsing tiny values to float."""
    probability, info = compute_circuit_probability(
        circuit,
        input_bits,
        output_bits,
        as_float=False,
        allow_tensor_contraction=allow_tensor_contraction,
        extended_reductions=extended_reductions,
        solver_config=solver_config,
    )
    return probability, info
