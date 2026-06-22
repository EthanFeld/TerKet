"""Bounded-bond boundary-MPS contraction for q3-free partition sums."""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np

from ..scaling import ScaledComplex
from .approx_mps_state import _BoundaryMPS


def _cuthill_mckee_order(neighbors: list[set[int]]) -> tuple[int, ...]:
    unseen = set(range(len(neighbors)))
    order: list[int] = []
    while unseen:
        root = min(unseen, key=lambda var: (len(neighbors[var]), var))
        queue = [root]
        unseen.remove(root)
        for var in queue:
            order.append(var)
            fresh = sorted(
                (neighbor for neighbor in neighbors[var] if neighbor in unseen),
                key=lambda neighbor: (len(neighbors[neighbor]), neighbor),
            )
            for neighbor in fresh:
                unseen.remove(neighbor)
                queue.append(neighbor)
    return tuple(order)


def _sweep_route_cost(order: tuple[int, ...], neighbors: list[set[int]]) -> tuple[int, int]:
    position = {var: idx for idx, var in enumerate(order)}
    final_use = [position[var] for var in range(len(order))]
    for var, adjacent in enumerate(neighbors):
        for neighbor in adjacent:
            final_use[var] = max(final_use[var], position[neighbor])
    active: list[int] = []
    route = 0
    peak = 0
    for step, var in enumerate(order):
        active.append(var)
        peak = max(peak, len(active))
        var_index = active.index(var)
        route += sum(
            max(0, var_index - active.index(neighbor) - 1)
            for neighbor in neighbors[var]
            if position[neighbor] < step
        )
        active = [label for label in active if final_use[label] > step]
    return route, peak


def _choose_sweep_order(q) -> tuple[tuple[int, ...], str, tuple[int, int], int]:
    from .treewidth import _min_fill_cubic_order

    elimination, width = _min_fill_cubic_order(q)
    elimination = tuple(int(var) for var in elimination)
    neighbors = [set() for _ in range(q.n)]
    for left, right in q.q2:
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))
    cuthill = _cuthill_mckee_order(neighbors)
    candidates = {
        "natural": tuple(range(q.n)),
        "natural_reverse": tuple(range(q.n - 1, -1, -1)),
        "min_fill": elimination,
        "min_fill_reverse": tuple(reversed(elimination)),
        "cuthill_mckee": cuthill,
        "cuthill_mckee_reverse": tuple(reversed(cuthill)),
    }
    metrics = {name: _sweep_route_cost(order, neighbors) for name, order in candidates.items()}
    name = min(metrics, key=lambda key: (metrics[key][0] * metrics[key][1], metrics[key]))
    return candidates[name], name, metrics[name], int(width)


def _sweep_graph_data(q, order: tuple[int, ...]):
    position = {var: idx for idx, var in enumerate(order)}
    previous: list[list[int]] = [[] for _ in range(q.n)]
    last_use = list(range(q.n))
    edge_phase: dict[tuple[int, int], complex] = {}
    for (left, right), coeff in q.q2.items():
        left, right = int(left), int(right)
        edge_phase[(min(left, right), max(left, right))] = cmath.exp(
            2j * math.pi * (int(coeff) % q.mod_q2) / float(q.mod_q2)
        )
        early, late = (left, right) if position[left] < position[right] else (right, left)
        previous[late].append(early)
        last_use[early] = max(last_use[early], position[late])
    return previous, last_use, edge_phase


def _sum_q3_free_boundary_mps_scaled(
    q,
    *,
    max_bond: int,
    cutoff: float,
) -> tuple[ScaledComplex, dict[str, Any]] | None:
    """Sweep q2 graph; truncate active separator with canonical MPS SVDs."""
    if q.q3:
        return None
    order, order_name, route, width = _choose_sweep_order(q)
    previous, last_use, edge_phase = _sweep_graph_data(q, order)
    state = _BoundaryMPS(
        cmath.exp(2j * math.pi * float(q.q0)), max_bond=max_bond, cutoff=cutoff
    )
    for step, var in enumerate(order):
        unary = cmath.exp(2j * math.pi * (int(q.q1[var]) % q.mod_q1) / float(q.mod_q1))
        state.append(var, unary)
        for neighbor in sorted(previous[var], key=state.labels.index, reverse=True):
            left_index = state.labels.index(neighbor)
            right_index = state.labels.index(var)
            phase = edge_phase[(min(var, neighbor), max(var, neighbor))]
            gate = np.asarray(
                [[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, phase]], dtype=np.complex128
            )
            if not state.apply_gate(left_index, right_index, gate):
                return None
        for label in [label for label in state.labels if last_use[label] <= step]:
            if not state.remove(state.labels.index(label)):
                return None
    value = state.finish()
    if value is None:
        return None
    return value, {
        "order": order_name,
        "route_swaps": int(route[0]),
        "width": width,
        "peak_active": int(state.peak_active),
        "peak_bond": int(state.peak_bond),
        "discarded_rss": float(math.sqrt(state.discarded_sq)),
        "max_discarded": float(state.max_discarded),
    }


def _sum_q3_free_boundary_mps_configured_scaled(q, config) -> ScaledComplex | None:
    result = _sum_q3_free_boundary_mps_scaled(
        q,
        max_bond=int(config.approx_tensor_max_bond),
        cutoff=float(config.approx_tensor_cutoff),
    )
    if result is None:
        return None
    value, diagnostics = result
    from .approx_guard import _set_q3_free_approx_diagnostics

    _set_q3_free_approx_diagnostics(
        {
            "approx_q3_free_method": "boundary_mps",
            "approx_q3_free_reliable": False,
            "approx_q3_free_mps_bond": int(config.approx_tensor_max_bond),
            **{f"approx_q3_free_mps_{key}": item for key, item in diagnostics.items()},
        }
    )
    return value
