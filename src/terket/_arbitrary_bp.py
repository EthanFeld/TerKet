"""Extracted arbitrary-angle BP helpers."""

from __future__ import annotations

import cmath
from typing import Mapping, Sequence

from ._engine_runtime_core import _configure_extracted_module
from .scaling import ScaledComplex
from .state import ReductionInfo

_LOCAL_NAMES = {
    '_arbitrary_bp_backend',
    '_arbitrary_exact_metadata',
    '_arbitrary_approx_metadata',
    '_mark_invalid_arbitrary_bp_info',
    '_raise_if_invalid_arbitrary_bp_amplitude',
    '_sum_pairwise_factor_graph_bethe_scaled',
    '_sum_factor_graph_bethe_scaled',
    '_sum_factor_graph_with_sparse_parity_bethe_scaled'
}
_LOCAL_IMPLS = {}
_configure_extracted_module(globals(), local_names=_LOCAL_NAMES, local_impls=_LOCAL_IMPLS)


def _arbitrary_bp_backend(backend: str | None) -> bool:
    if backend is None:
        return False
    name = str(backend)
    name = name.removesuffix("_invalid_scale")
    name = name.removesuffix("_heuristic")
    return name in {
        "arbitrary_bethe_bp",
        "arbitrary_factor_bethe_bp",
        "arbitrary_sparse_parity_bethe_bp",
        "arbitrary_bethe_bp_normalized",
    }


def _arbitrary_exact_metadata() -> dict[str, object]:
    return {
        "is_approximate": False,
        "approx_backend": None,
        "approx_validation": "exact",
    }


def _arbitrary_approx_metadata(
    backend: str,
    validation: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "is_approximate": True,
        "approx_backend": backend,
        "approx_validation": validation,
    }
    if extra:
        metadata.update(extra)
    return metadata


def _mark_invalid_arbitrary_bp_info(info: ReductionInfo, scaled_amp: ScaledComplex) -> None:
    backend = str(info.get("phase3_backend"))
    if not backend.endswith("_invalid_scale"):
        info["phase3_backend"] = f"{backend}_invalid_scale"
    info["bp_invalid_reason"] = "implied_probability_exceeds_one"  # type: ignore[typeddict-unknown-key]
    info["bp_log2_probability"] = _scaled_probability_log2(scaled_amp)  # type: ignore[typeddict-unknown-key]


def _raise_if_invalid_arbitrary_bp_amplitude(info: ReductionInfo, scaled_amp: ScaledComplex) -> None:
    if not _arbitrary_bp_backend(info.get("phase3_backend")):
        return
    log2_probability = _scaled_probability_log2(scaled_amp)
    if log2_probability <= _ARBITRARY_BP_DIRECT_PROB_LOG2_TOL:
        return
    _mark_invalid_arbitrary_bp_info(info, scaled_amp)
    raise RuntimeError(
        "Unreliable arbitrary-angle BP estimate: implied output probability "
        f"2^{log2_probability:.3g} exceeds the quantum bound. Use an exact path, "
        "a normalized observable estimator, or a fidelity-validated approximate backend."
    )


def _sum_pairwise_factor_graph_bethe_scaled(
    n_vars: int,
    factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]],
    *,
    scalar: ScaledComplex,
    max_iters: int | None = None,
    damping: float | None = None,
    require_forest: bool = True,
) -> tuple[ScaledComplex, int] | None:
    """Approximate complex pairwise factor graph partition fn via loopy BP."""
    unary = [[1.0 + 0.0j, 1.0 + 0.0j] for _ in range(n_vars)]
    pair_tables: dict[tuple[int, int], list[complex]] = {}
    scalar_scaled = scalar

    for scope, table in factors.items():
        scope = tuple(int(var) for var in scope)
        if len(scope) == 0:
            scalar_scaled = _mul_scaled_complex(scalar_scaled, table[0])
        elif len(scope) == 1:
            var = scope[0]
            unary[var][0] *= _scaled_to_plain_complex(table[0])
            unary[var][1] *= _scaled_to_plain_complex(table[1])
        elif len(scope) == 2:
            left, right = scope
            key = (left, right) if left < right else (right, left)
            table_complex = [_scaled_to_plain_complex(entry) for entry in table]
            if left > right:
                table_complex = [table_complex[0], table_complex[2], table_complex[1], table_complex[3]]
            existing = pair_tables.get(key)
            if existing is None:
                pair_tables[key] = table_complex
            else:
                pair_tables[key] = [a * b for a, b in zip(existing, table_complex)]
        else:
            return None

    if not pair_tables:
        total = scalar_scaled
        for phi0, phi1 in unary:
            total = _mul_scaled_complex(total, _make_scaled_complex(phi0 + phi1))
        return total, 0

    neighbors = [set() for _ in range(n_vars)]
    for left, right in pair_tables:
        neighbors[left].add(right)
        neighbors[right].add(left)
    if require_forest and not _factor_graph_is_forest(n_vars, tuple(pair_tables)):
        return None
    neighbor_lists = [tuple(sorted(neighbor_set)) for neighbor_set in neighbors]
    messages: dict[tuple[int, int], list[complex]] = {}
    for left, right in pair_tables:
        messages[(left, right)] = [1.0 + 0.0j, 1.0 + 0.0j]
        messages[(right, left)] = [1.0 + 0.0j, 1.0 + 0.0j]

    damping_value = float(_ARBITRARY_BP_DAMPING if damping is None else damping)
    iter_limit = int(_ARBITRARY_BP_MAX_ITERS if max_iters is None else max_iters)
    for _iter in range(iter_limit):
        max_delta = 0.0
        new_messages: dict[tuple[int, int], list[complex]] = {}
        for src, src_neighbors in enumerate(neighbor_lists):
            count = len(src_neighbors)
            if not count:
                continue
            prefix0 = [unary[src][0]] * (count + 1)
            prefix1 = [unary[src][1]] * (count + 1)
            for idx, nbr in enumerate(src_neighbors):
                incoming = messages[(nbr, src)]
                prefix0[idx + 1] = prefix0[idx] * incoming[0]
                prefix1[idx + 1] = prefix1[idx] * incoming[1]
            suffix0 = [1.0 + 0.0j] * (count + 1)
            suffix1 = [1.0 + 0.0j] * (count + 1)
            for idx in range(count - 1, -1, -1):
                incoming = messages[(src_neighbors[idx], src)]
                suffix0[idx] = suffix0[idx + 1] * incoming[0]
                suffix1[idx] = suffix1[idx + 1] * incoming[1]
            for idx, dst in enumerate(src_neighbors):
                prod0 = prefix0[idx] * suffix0[idx + 1]
                prod1 = prefix1[idx] * suffix1[idx + 1]
                key = (src, dst) if src < dst else (dst, src)
                psi = pair_tables[key]
                if src < dst:
                    raw0 = prod0 * psi[0] + prod1 * psi[1]
                    raw1 = prod0 * psi[2] + prod1 * psi[3]
                else:
                    raw0 = prod0 * psi[0] + prod1 * psi[2]
                    raw1 = prod0 * psi[1] + prod1 * psi[3]
                scale = max(abs(raw0), abs(raw1), 1e-300)
                msg = [raw0 / scale, raw1 / scale]
                old = messages[(src, dst)]
                damped = [
                    (1.0 - damping_value) * msg[0] + damping_value * old[0],
                    (1.0 - damping_value) * msg[1] + damping_value * old[1],
                ]
                norm = max(abs(damped[0]), abs(damped[1]), 1e-300)
                damped = [damped[0] / norm, damped[1] / norm]
                max_delta = max(max_delta, abs(damped[0] - old[0]), abs(damped[1] - old[1]))
                new_messages[(src, dst)] = damped
        messages = new_messages
        if max_delta <= _ARBITRARY_BP_TOL:
            break

    scalar_log = _scaled_complex_log(scalar_scaled)
    if scalar_log is None:
        return _ZERO_SCALED, max(1, max(len(neighbors[var]) for var in range(n_vars)))
    log_z = scalar_log
    log_z_vars: list[complex] = []
    for var in range(n_vars):
        b0 = unary[var][0]
        b1 = unary[var][1]
        for nbr in neighbors[var]:
            incoming = messages[(nbr, var)]
            b0 *= incoming[0]
            b1 *= incoming[1]
        log_z_vars.append(_complex_logsum((b0, b1)))

    for (left, right), psi in pair_tables.items():
        edge_terms = []
        for left_bit in (0, 1):
            left_weight = unary[left][left_bit]
            for nbr in neighbors[left]:
                if nbr != right:
                    left_weight *= messages[(nbr, left)][left_bit]
            for right_bit in (0, 1):
                right_weight = unary[right][right_bit]
                for nbr in neighbors[right]:
                    if nbr != left:
                        right_weight *= messages[(nbr, right)][right_bit]
                edge_terms.append(left_weight * right_weight * psi[left_bit | (right_bit << 1)])
        log_z += _complex_logsum(edge_terms)

    for var in range(n_vars):
        log_z += (1 - len(neighbors[var])) * log_z_vars[var]

    max_degree = max((len(neighbor_set) for neighbor_set in neighbors), default=0)
    return _scaled_from_complex_log(log_z), max(1, max_degree + 1)


def _sum_factor_graph_bethe_scaled(
    n_vars: int,
    factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]],
    *,
    scalar: ScaledComplex,
    max_iters: int | None = None,
    damping: float | None = None,
    require_forest: bool = True,
) -> tuple[ScaledComplex, int] | None:
    """Approximate a bounded-scope complex factor graph partition fn via BP."""
    factor_items: list[tuple[tuple[int, ...], list[complex], dict[int, int]]] = []
    scalar_scaled = scalar
    unary = [[1.0 + 0.0j, 1.0 + 0.0j] for _ in range(n_vars)]
    max_scope = 0
    for scope, table in factors.items():
        scope = tuple(int(var) for var in scope)
        if not scope:
            scalar_scaled = _mul_scaled_complex(scalar_scaled, table[0])
            continue
        if len(scope) > 8:
            return None
        max_scope = max(max_scope, len(scope))
        if len(scope) == 1:
            var = scope[0]
            unary[var][0] *= _scaled_to_plain_complex(table[0])
            unary[var][1] *= _scaled_to_plain_complex(table[1])
            continue
        factor_items.append((
            scope,
            [_scaled_to_plain_complex(entry) for entry in table],
            {var: pos for pos, var in enumerate(scope)},
        ))

    scalar_log = _scaled_complex_log(scalar_scaled)
    if scalar_log is None:
        return _ZERO_SCALED, max(1, max_scope)
    if not factor_items:
        log_z = scalar_log
        for prior in unary:
            log_z += _complex_logsum((prior[0], prior[1]))
        return _scaled_from_complex_log(log_z), 1 if n_vars else 0
    if require_forest and not _factor_graph_is_forest(n_vars, tuple(scope for scope, _table, _positions in factor_items)):
        return None

    var_factors: list[list[int]] = [[] for _ in range(n_vars)]
    for factor_idx, (scope, _table, _positions) in enumerate(factor_items):
        for var in scope:
            var_factors[var].append(factor_idx)

    msg_vf: dict[tuple[int, int], list[complex]] = {}
    msg_fv: dict[tuple[int, int], list[complex]] = {}
    message_edge_count = 0
    for factor_idx, (scope, _table, _positions) in enumerate(factor_items):
        for var in scope:
            message_edge_count += 1
            msg_vf[(var, factor_idx)] = [1.0 + 0.0j, 1.0 + 0.0j]
            msg_fv[(factor_idx, var)] = [1.0 + 0.0j, 1.0 + 0.0j]

    damping_value = float(_ARBITRARY_BP_DAMPING if damping is None else damping)
    max_iters = (
        int(_ARBITRARY_FACTOR_BP_LARGE_MAX_ITERS if max_iters is None else max_iters)
        if message_edge_count >= _ARBITRARY_FACTOR_BP_LARGE_EDGE_THRESHOLD
        else int(_ARBITRARY_FACTOR_BP_MAX_ITERS if max_iters is None else max_iters)
    )
    for _iter in range(max_iters):
        max_delta = 0.0
        new_msg_fv: dict[tuple[int, int], list[complex]] = {}
        for factor_idx, (scope, table, positions) in enumerate(factor_items):
            incoming = [msg_vf[(var, factor_idx)] for var in scope]
            for var in scope:
                var_pos = positions[var]
                raw = [0.0 + 0.0j, 0.0 + 0.0j]
                for assignment, table_value in enumerate(table):
                    bit = (assignment >> var_pos) & 1
                    weight = table_value
                    for other_pos, other_msg in enumerate(incoming):
                        if other_pos == var_pos:
                            continue
                        weight *= other_msg[(assignment >> other_pos) & 1]
                    raw[bit] += weight
                scale = max(abs(raw[0]), abs(raw[1]), 1e-300)
                msg = [raw[0] / scale, raw[1] / scale]
                old = msg_fv[(factor_idx, var)]
                damped = [
                    (1.0 - damping_value) * msg[0] + damping_value * old[0],
                    (1.0 - damping_value) * msg[1] + damping_value * old[1],
                ]
                norm = max(abs(damped[0]), abs(damped[1]), 1e-300)
                damped = [damped[0] / norm, damped[1] / norm]
                max_delta = max(max_delta, abs(damped[0] - old[0]), abs(damped[1] - old[1]))
                new_msg_fv[(factor_idx, var)] = damped

        new_msg_vf: dict[tuple[int, int], list[complex]] = {}
        for var, incident in enumerate(var_factors):
            count = len(incident)
            prefix0 = [unary[var][0]] * (count + 1)
            prefix1 = [unary[var][1]] * (count + 1)
            for idx, other_factor in enumerate(incident):
                incoming = new_msg_fv[(other_factor, var)]
                prefix0[idx + 1] = prefix0[idx] * incoming[0]
                prefix1[idx + 1] = prefix1[idx] * incoming[1]
            suffix0 = [1.0 + 0.0j] * (count + 1)
            suffix1 = [1.0 + 0.0j] * (count + 1)
            for idx in range(count - 1, -1, -1):
                incoming = new_msg_fv[(incident[idx], var)]
                suffix0[idx] = suffix0[idx + 1] * incoming[0]
                suffix1[idx] = suffix1[idx + 1] * incoming[1]
            for idx, factor_idx in enumerate(incident):
                prod0 = prefix0[idx] * suffix0[idx + 1]
                prod1 = prefix1[idx] * suffix1[idx + 1]
                scale = max(abs(prod0), abs(prod1), 1e-300)
                msg = [prod0 / scale, prod1 / scale]
                old = msg_vf[(var, factor_idx)]
                damped = [
                    (1.0 - damping_value) * msg[0] + damping_value * old[0],
                    (1.0 - damping_value) * msg[1] + damping_value * old[1],
                ]
                norm = max(abs(damped[0]), abs(damped[1]), 1e-300)
                damped = [damped[0] / norm, damped[1] / norm]
                max_delta = max(max_delta, abs(damped[0] - old[0]), abs(damped[1] - old[1]))
                new_msg_vf[(var, factor_idx)] = damped
        msg_fv = new_msg_fv
        msg_vf = new_msg_vf
        if max_delta <= _ARBITRARY_BP_TOL:
            break

    log_z = scalar_log
    for factor_idx, (scope, table, _positions) in enumerate(factor_items):
        terms = []
        for assignment, table_value in enumerate(table):
            weight = table_value
            for pos, var in enumerate(scope):
                weight *= msg_vf[(var, factor_idx)][(assignment >> pos) & 1]
            terms.append(weight)
        log_z += _complex_logsum(terms)

    for var, incident in enumerate(var_factors):
        b0 = unary[var][0]
        b1 = unary[var][1]
        for factor_idx in incident:
            incoming = msg_fv[(factor_idx, var)]
            b0 *= incoming[0]
            b1 *= incoming[1]
        log_z += _complex_logsum((b0, b1))

    for factor_idx, (scope, _table, _positions) in enumerate(factor_items):
        for var in scope:
            log_z -= _complex_logsum((
                msg_vf[(var, factor_idx)][0] * msg_fv[(factor_idx, var)][0],
                msg_vf[(var, factor_idx)][1] * msg_fv[(factor_idx, var)][1],
            ))

    return _scaled_from_complex_log(log_z), max(1, max_scope)


def _sum_factor_graph_with_sparse_parity_bethe_scaled(
    n_vars: int,
    factors: Mapping[tuple[int, ...], Sequence[ScaledComplex]],
    terms: Sequence[_ArbitraryPhaseTerm],
    *,
    scalar: ScaledComplex,
    max_iters: int | None = None,
    damping: float | None = None,
    require_forest: bool = True,
) -> tuple[ScaledComplex, int] | None:
    """Approximate factor graph BP with wide affine-parity phase factors kept sparse."""
    dense_items: list[tuple[tuple[int, ...], list[complex], dict[int, int]]] = []
    parity_items: list[tuple[tuple[int, ...], complex, int]] = []
    scalar_scaled = scalar
    unary = [[1.0 + 0.0j, 1.0 + 0.0j] for _ in range(n_vars)]
    max_scope = 0

    for scope, table in factors.items():
        scope = tuple(int(var) for var in scope)
        if not scope:
            scalar_scaled = _mul_scaled_complex(scalar_scaled, table[0])
            continue
        if len(scope) > 8:
            return None
        max_scope = max(max_scope, len(scope))
        if len(scope) == 1:
            var = scope[0]
            unary[var][0] *= _scaled_to_plain_complex(table[0])
            unary[var][1] *= _scaled_to_plain_complex(table[1])
            continue
        dense_items.append((
            scope,
            [_scaled_to_plain_complex(entry) for entry in table],
            {var: pos for pos, var in enumerate(scope)},
        ))

    for term in terms:
        row_mask = int(term.row_mask)
        phase = cmath.exp(1j * float(term.angle))
        offset = int(term.offset) & 1
        if row_mask == 0:
            if offset:
                scalar_scaled = _mul_scaled_complex(scalar_scaled, _make_scaled_complex(phase))
            continue
        scope = _support_from_mask(row_mask)
        max_scope = max(max_scope, len(scope))
        parity_items.append((scope, phase, offset))

    scalar_log = _scaled_complex_log(scalar_scaled)
    if scalar_log is None:
        return _ZERO_SCALED, max(1, max_scope)
    sparse_scopes = tuple(scope for scope, _table, _positions in dense_items) + tuple(
        scope for scope, _phase, _offset in parity_items
    )
    if require_forest and not _factor_graph_is_forest(n_vars, sparse_scopes):
        return None

    var_factors: list[list[tuple[str, int]]] = [[] for _ in range(n_vars)]
    for factor_idx, (scope, _table, _positions) in enumerate(dense_items):
        for var in scope:
            var_factors[var].append(("d", factor_idx))
    for factor_idx, (scope, _phase, _offset) in enumerate(parity_items):
        for var in scope:
            var_factors[var].append(("p", factor_idx))

    message_edge_count = sum(len(scope) for scope, _table, _positions in dense_items)
    message_edge_count += sum(len(scope) for scope, _phase, _offset in parity_items)
    if message_edge_count == 0:
        log_z = scalar_log
        for prior in unary:
            log_z += _complex_logsum((prior[0], prior[1]))
        return _scaled_from_complex_log(log_z), 1 if n_vars else 0

    msg_vf: dict[tuple[int, str, int], list[complex]] = {}
    msg_fv: dict[tuple[str, int, int], list[complex]] = {}
    for factor_idx, (scope, _table, _positions) in enumerate(dense_items):
        for var in scope:
            msg_vf[(var, "d", factor_idx)] = [1.0 + 0.0j, 1.0 + 0.0j]
            msg_fv[("d", factor_idx, var)] = [1.0 + 0.0j, 1.0 + 0.0j]
    for factor_idx, (scope, _phase, _offset) in enumerate(parity_items):
        for var in scope:
            msg_vf[(var, "p", factor_idx)] = [1.0 + 0.0j, 1.0 + 0.0j]
            msg_fv[("p", factor_idx, var)] = [1.0 + 0.0j, 1.0 + 0.0j]

    damping_value = float(_ARBITRARY_BP_DAMPING if damping is None else damping)
    max_iters = (
        int(_ARBITRARY_FACTOR_BP_LARGE_MAX_ITERS if max_iters is None else max_iters)
        if message_edge_count >= _ARBITRARY_FACTOR_BP_LARGE_EDGE_THRESHOLD
        else int(_ARBITRARY_FACTOR_BP_MAX_ITERS if max_iters is None else max_iters)
    )
    for _iter in range(max_iters):
        max_delta = 0.0
        new_msg_fv: dict[tuple[str, int, int], list[complex]] = {}

        for factor_idx, (scope, table, positions) in enumerate(dense_items):
            incoming = [msg_vf[(var, "d", factor_idx)] for var in scope]
            for var in scope:
                var_pos = positions[var]
                raw = [0.0 + 0.0j, 0.0 + 0.0j]
                for assignment, table_value in enumerate(table):
                    bit = (assignment >> var_pos) & 1
                    weight = table_value
                    for other_pos, other_msg in enumerate(incoming):
                        if other_pos != var_pos:
                            weight *= other_msg[(assignment >> other_pos) & 1]
                    raw[bit] += weight
                scale = max(abs(raw[0]), abs(raw[1]), 1e-300)
                msg = [raw[0] / scale, raw[1] / scale]
                old = msg_fv[("d", factor_idx, var)]
                damped = [
                    (1.0 - damping_value) * msg[0] + damping_value * old[0],
                    (1.0 - damping_value) * msg[1] + damping_value * old[1],
                ]
                norm = max(abs(damped[0]), abs(damped[1]), 1e-300)
                damped = [damped[0] / norm, damped[1] / norm]
                max_delta = max(max_delta, abs(damped[0] - old[0]), abs(damped[1] - old[1]))
                new_msg_fv[("d", factor_idx, var)] = damped

        for factor_idx, (scope, phase, offset) in enumerate(parity_items):
            incoming = [msg_vf[(var, "p", factor_idx)] for var in scope]
            count = len(scope)
            prefix_sum = [1.0 + 0.0j] * (count + 1)
            prefix_diff = [1.0 + 0.0j] * (count + 1)
            for idx, msg in enumerate(incoming):
                prefix_sum[idx + 1] = prefix_sum[idx] * (msg[0] + msg[1])
                prefix_diff[idx + 1] = prefix_diff[idx] * (msg[0] - msg[1])
            suffix_sum = [1.0 + 0.0j] * (count + 1)
            suffix_diff = [1.0 + 0.0j] * (count + 1)
            for idx in range(count - 1, -1, -1):
                msg = incoming[idx]
                suffix_sum[idx] = suffix_sum[idx + 1] * (msg[0] + msg[1])
                suffix_diff[idx] = suffix_diff[idx + 1] * (msg[0] - msg[1])
            phase_delta = phase - 1.0
            for idx, var in enumerate(scope):
                other_sum = prefix_sum[idx] * suffix_sum[idx + 1]
                other_diff = prefix_diff[idx] * suffix_diff[idx + 1]
                even = 0.5 * (other_sum + other_diff)
                odd = 0.5 * (other_sum - other_diff)
                raw = [0.0 + 0.0j, 0.0 + 0.0j]
                for bit in (0, 1):
                    trigger_even = ((1 ^ bit ^ offset) == 0)
                    phase_sum = even if trigger_even else odd
                    raw[bit] = other_sum + phase_delta * phase_sum
                scale = max(abs(raw[0]), abs(raw[1]), 1e-300)
                msg = [raw[0] / scale, raw[1] / scale]
                old = msg_fv[("p", factor_idx, var)]
                damped = [
                    (1.0 - damping_value) * msg[0] + damping_value * old[0],
                    (1.0 - damping_value) * msg[1] + damping_value * old[1],
                ]
                norm = max(abs(damped[0]), abs(damped[1]), 1e-300)
                damped = [damped[0] / norm, damped[1] / norm]
                max_delta = max(max_delta, abs(damped[0] - old[0]), abs(damped[1] - old[1]))
                new_msg_fv[("p", factor_idx, var)] = damped

        new_msg_vf: dict[tuple[int, str, int], list[complex]] = {}
        for var, incident in enumerate(var_factors):
            count = len(incident)
            prefix0 = [unary[var][0]] * (count + 1)
            prefix1 = [unary[var][1]] * (count + 1)
            for idx, (kind, factor_idx) in enumerate(incident):
                incoming = new_msg_fv[(kind, factor_idx, var)]
                prefix0[idx + 1] = prefix0[idx] * incoming[0]
                prefix1[idx + 1] = prefix1[idx] * incoming[1]
            suffix0 = [1.0 + 0.0j] * (count + 1)
            suffix1 = [1.0 + 0.0j] * (count + 1)
            for idx in range(count - 1, -1, -1):
                kind, factor_idx = incident[idx]
                incoming = new_msg_fv[(kind, factor_idx, var)]
                suffix0[idx] = suffix0[idx + 1] * incoming[0]
                suffix1[idx] = suffix1[idx + 1] * incoming[1]
            for idx, (kind, factor_idx) in enumerate(incident):
                prod0 = prefix0[idx] * suffix0[idx + 1]
                prod1 = prefix1[idx] * suffix1[idx + 1]
                scale = max(abs(prod0), abs(prod1), 1e-300)
                msg = [prod0 / scale, prod1 / scale]
                old = msg_vf[(var, kind, factor_idx)]
                damped = [
                    (1.0 - damping_value) * msg[0] + damping_value * old[0],
                    (1.0 - damping_value) * msg[1] + damping_value * old[1],
                ]
                norm = max(abs(damped[0]), abs(damped[1]), 1e-300)
                damped = [damped[0] / norm, damped[1] / norm]
                max_delta = max(max_delta, abs(damped[0] - old[0]), abs(damped[1] - old[1]))
                new_msg_vf[(var, kind, factor_idx)] = damped

        msg_fv = new_msg_fv
        msg_vf = new_msg_vf
        if max_delta <= _ARBITRARY_BP_TOL:
            break

    log_z = scalar_log
    for factor_idx, (scope, table, _positions) in enumerate(dense_items):
        terms_for_log = []
        for assignment, table_value in enumerate(table):
            weight = table_value
            for pos, var in enumerate(scope):
                weight *= msg_vf[(var, "d", factor_idx)][(assignment >> pos) & 1]
            terms_for_log.append(weight)
        log_z += _complex_logsum(terms_for_log)

    for factor_idx, (scope, phase, offset) in enumerate(parity_items):
        prod_sum = 1.0 + 0.0j
        prod_diff = 1.0 + 0.0j
        for var in scope:
            msg = msg_vf[(var, "p", factor_idx)]
            prod_sum *= msg[0] + msg[1]
            prod_diff *= msg[0] - msg[1]
        even = 0.5 * (prod_sum + prod_diff)
        odd = 0.5 * (prod_sum - prod_diff)
        phase_sum = even if offset else odd
        value = prod_sum + (phase - 1.0) * phase_sum
        if value == 0j:
            return _ZERO_SCALED, max(1, max_scope)
        log_z += cmath.log(value)

    for var, incident in enumerate(var_factors):
        b0 = unary[var][0]
        b1 = unary[var][1]
        for kind, factor_idx in incident:
            incoming = msg_fv[(kind, factor_idx, var)]
            b0 *= incoming[0]
            b1 *= incoming[1]
        log_z += _complex_logsum((b0, b1))

    for factor_idx, (scope, _table, _positions) in enumerate(dense_items):
        for var in scope:
            log_z -= _complex_logsum((
                msg_vf[(var, "d", factor_idx)][0] * msg_fv[("d", factor_idx, var)][0],
                msg_vf[(var, "d", factor_idx)][1] * msg_fv[("d", factor_idx, var)][1],
            ))
    for factor_idx, (scope, _phase, _offset) in enumerate(parity_items):
        for var in scope:
            log_z -= _complex_logsum((
                msg_vf[(var, "p", factor_idx)][0] * msg_fv[("p", factor_idx, var)][0],
                msg_vf[(var, "p", factor_idx)][1] * msg_fv[("p", factor_idx, var)][1],
            ))

    return _scaled_from_complex_log(log_z), max(1, max_scope)

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
