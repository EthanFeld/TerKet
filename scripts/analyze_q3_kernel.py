"""Inspect the reduced q3-free kernel structure behind a circuit amplitude query."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path

import terket
from terket.circuits import _circuit_global_phase_radians
from terket.engine import (
    _cubic_order_width,
    _is_half_phase_q2,
    _min_degree_cubic_order_uncached,
    _min_fill_cubic_order,
    _phase_function_from_parts,
    _q3_free_edge_density,
    _q3_free_spanning_data,
    _select_feedback_vertices,
    build_state,
)


def _component_sizes(n_vars: int, q2: dict[tuple[int, int], int]) -> list[int]:
    adjacency = [[] for _ in range(n_vars)]
    for left, right in q2:
        adjacency[left].append(right)
        adjacency[right].append(left)

    seen = [False] * n_vars
    sizes: list[int] = []
    for start in range(n_vars):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        sizes.append(size)
    sizes.sort(reverse=True)
    return sizes


def _series_reduced_core(n_vars: int, q2: dict[tuple[int, int], int]) -> tuple[int, int, dict[int, int]]:
    adjacency = [set() for _ in range(n_vars)]
    for left, right in q2:
        adjacency[left].add(right)
        adjacency[right].add(left)

    active = [True] * n_vars
    queue = deque(idx for idx in range(n_vars) if len(adjacency[idx]) <= 2)
    while queue:
        node = queue.popleft()
        if not active[node]:
            continue
        degree = len(adjacency[node])
        if degree > 2:
            continue
        neighbors = sorted(adjacency[node])
        active[node] = False
        if degree == 1:
            neighbor = neighbors[0]
            adjacency[neighbor].discard(node)
            if len(adjacency[neighbor]) <= 2:
                queue.append(neighbor)
        elif degree == 2:
            left, right = neighbors
            adjacency[left].discard(node)
            adjacency[right].discard(node)
            if left != right:
                adjacency[left].add(right)
                adjacency[right].add(left)
            if len(adjacency[left]) <= 2:
                queue.append(left)
            if len(adjacency[right]) <= 2:
                queue.append(right)
        adjacency[node].clear()

    core_nodes = [idx for idx, keep in enumerate(active) if keep]
    degree_hist = Counter(len(adjacency[idx]) for idx in core_nodes)
    core_edges = sum(len(adjacency[idx]) for idx in core_nodes) // 2
    return len(core_nodes), core_edges, dict(sorted(degree_hist.items()))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("circuit", help="Path to an OpenQASM 2.0 file")
    parser.add_argument(
        "--rz-compile-mode",
        default="approx_dyadic",
        choices=("dyadic", "approx_dyadic", "clifford_t"),
        help="How to lower non-dyadic single-qubit phases before kernel analysis.",
    )
    parser.add_argument(
        "--rz-tolerance",
        type=float,
        default=1e-5,
        help="Approximation / synthesis tolerance for non-dyadic single-qubit phases.",
    )
    parser.add_argument(
        "--skip-min-fill",
        action="store_true",
        help="Skip the more expensive min-fill width computation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = Path(args.circuit).read_text()
    spec = terket.normalize_circuit(
        source,
        rz_compile_mode=args.rz_compile_mode,
        rz_tolerance=args.rz_tolerance,
    )
    state = build_state(
        spec.n_qubits,
        spec.gates,
        [0] * spec.n_qubits,
        global_phase_radians=_circuit_global_phase_radians(spec),
    )
    q = state.q

    print(f"Circuit: {Path(args.circuit).name}")
    print(f"Normalized gates: {len(spec.gates)}")
    print(f"Kernel vars: {q.n}")
    print(f"q2 edges: {len(q.q2)}")
    print(f"q3 terms: {len(q.q3)}")
    print(f"half_phase_q2: {_is_half_phase_q2(q)}")

    if q.q3:
        print("Kernel is not q3-free. Use cubic profiling instead.")
        return

    components = _component_sizes(q.n, q.q2)
    degrees = Counter()
    adjacency = [[] for _ in range(q.n)]
    for left, right in q.q2:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for neighbors in adjacency:
        degrees[len(neighbors)] += 1

    q_struct = _phase_function_from_parts(
        q.n,
        level=q.level,
        q0=q.q0,
        q1=[0] * q.n,
        q2=q.q2,
        q3={},
    )
    forward_order = list(range(q.n))
    reverse_order = list(range(q.n - 1, -1, -1))
    forward_width = _cubic_order_width(q_struct, forward_order)
    reverse_width = _cubic_order_width(q_struct, reverse_order)
    min_degree_order, min_degree_width = _min_degree_cubic_order_uncached(q_struct)
    del min_degree_order
    min_fill_width = None
    if not args.skip_min_fill:
        _min_fill_order, min_fill_width = _min_fill_cubic_order(q_struct)

    depth, chords = _q3_free_spanning_data(
        adjacency,
        [(left, right, coeff) for (left, right), coeff in q.q2.items()],
    )
    feedback_size = len(_select_feedback_vertices(q.n, chords, depth))
    core_vars, core_edges, core_degree_hist = _series_reduced_core(q.n, q.q2)

    print(f"components: {len(components)}")
    print(f"largest_component: {components[0] if components else 0}")
    print(f"degree_hist: {dict(sorted(degrees.items()))}")
    print(f"edge_density: {_q3_free_edge_density(q):.9f}")
    print(f"feedback_size: {feedback_size}")
    print(f"series_reduced_core_vars: {core_vars}")
    print(f"series_reduced_core_edges: {core_edges}")
    print(f"series_reduced_core_degree_hist: {core_degree_hist}")
    print(f"forward_width: {forward_width}")
    print(f"reverse_width: {reverse_width}")
    print(f"min_degree_width: {min_degree_width}")
    if min_fill_width is not None:
        print(f"min_fill_width: {min_fill_width}")

    print("Implications:")
    if min_fill_width is not None and min(forward_width, reverse_width) + 16 < min_fill_width:
        print("  chronological order much better than blind min-fill; order-aware slicing likely useful")
    if core_vars and core_vars < q.n:
        print("  iterative degree-1/2 elimination can shrink kernel before heavy planning")
    if max(degrees, default=0) <= 4:
        print("  low-degree kernel; small-separator or order-guided cutset search is natural")
    if feedback_size > 512:
        print("  feedback-set heuristics likely overestimate difficulty on this family")


if __name__ == "__main__":
    main()
