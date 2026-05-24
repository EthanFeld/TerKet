"""Helpers for dense q2-core heuristic experiments on MQT QAOA."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import sys
from typing import Callable, Iterable

from mqt.bench import get_benchmark_alg
import networkx as nx
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import normalize_circuit
from terket.benchmarking.mqt import bind_deterministic_parameters, hash_bits
from terket.circuits import _circuit_global_phase_radians
from terket.engine import (
    _aff_compose_cached,
    _evaluate_q3_free_cutset_candidate,
    _evaluate_q3_free_cutset_conditioning_plan_scaled,
    _gauss_sum_q3_free_scaled,
    _minimum_bad_q2_vertex_cover,
    _min_fill_cubic_order,
    _pair_graph_separator_order,
    _q3_free_cutset_conditioning_plan,
    _row_masks_from_gamma,
    _scaled_to_complex,
    build_state,
)


QAOA_SEED = 10
DEFAULT_SIZES = (24, 25)
DEFAULT_BUDGETS = (4, 6, 8, 10, 12)
DEEP_BUDGETS = (12, 14, 16)
TARGET_REMAINING_WIDTH = 22


@dataclass(frozen=True)
class HeuristicSpec:
    category: str
    name: str
    description: str
    ranker: Callable[["DenseCoreCase", int], list[int]]


@dataclass(frozen=True)
class DenseCoreCase:
    size: int
    circuit: object
    q: object
    original_var_count: int
    free_var_count: int
    q2_graph: nx.Graph
    original_graph: nx.Graph
    min_fill_order: tuple[int, ...]
    min_fill_width: int
    separator_order: tuple[int, ...]
    separator_width: int | None
    row_masks: tuple[int, ...]
    qubit_support: tuple[tuple[int, ...], ...]
    slice_support: tuple[tuple[int, ...], ...]
    output_support: tuple[int, ...]
    original_degree_lift: tuple[float, ...]
    original_betweenness_lift: tuple[float, ...]
    original_boundary_lift: tuple[float, ...]
    slice_span: tuple[float, ...]
    support_entropy: tuple[float, ...]
    graph_degree: tuple[float, ...]
    graph_betweenness: tuple[float, ...]
    graph_edge_betweenness_incident: tuple[float, ...]
    graph_core: tuple[float, ...]
    graph_cycle: tuple[float, ...]
    graph_boundary: tuple[float, ...]
    spectral_boundary: tuple[float, ...]
    local_field: tuple[float, ...]
    cavity_pressure: tuple[float, ...]
    spin_glass_core: tuple[float, ...]
    bad_edge_incidence: tuple[float, ...]
    bad_edge_cover_rank: tuple[int, ...]
    builtin_cutset: tuple[int, ...]


@dataclass(frozen=True)
class CandidateRow:
    size: int
    category: str
    heuristic: str
    description: str
    budget: int
    cutset_vars: tuple[int, ...]
    viable: bool
    remaining_backend: str
    remaining_width: int
    estimated_total_work: int
    estimated_total_work_log2: float
    generic_penalty: int
    improvement_vs_min_fill: int
    builtin_cutset_overlap: int


def _build_original_qaoa_graph(num_qubits: int, seed: int = QAOA_SEED) -> nx.Graph:
    rng = np.random.default_rng(seed)
    adjacency_matrix = rng.integers(0, 2, size=(num_qubits, num_qubits))
    adjacency_matrix = np.triu(adjacency_matrix, 1)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    for left in range(num_qubits):
        for right in range(left + 1, num_qubits):
            if int(adjacency_matrix[left, right]) != 0:
                graph.add_edge(left, right)
    return graph


def _score_sort_primary(scores: Iterable[float], tie_break: Iterable[float] | None = None) -> list[int]:
    primary = list(scores)
    secondary = list(tie_break) if tie_break is not None else [0.0] * len(primary)
    return [
        idx
        for idx, _score, _tie in sorted(
            ((idx, primary[idx], secondary[idx]) for idx in range(len(primary))),
            key=lambda item: (item[1], item[2], -item[0]),
            reverse=True,
        )
    ]


def _reverse_order(order: Iterable[int], n_vars: int) -> list[int]:
    order_list = list(order)
    seen = set(order_list)
    remainder = [var for var in range(n_vars) if var not in seen]
    return list(reversed(order_list)) + remainder


def _normalize_map(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    ordered_values = list(values.values())
    mean_value = statistics.fmean(ordered_values)
    if len(ordered_values) < 2:
        return {key: 0.0 for key in values}
    std_value = statistics.pstdev(ordered_values)
    if std_value <= 1e-12:
        return {key: 0.0 for key in values}
    return {key: (value - mean_value) / std_value for key, value in values.items()}


def _fundamental_cycle_participation(graph: nx.Graph) -> dict[int, float]:
    if graph.number_of_nodes() == 0:
        return {}
    forest = nx.minimum_spanning_tree(graph)
    tree_edges = {tuple(sorted(edge)) for edge in forest.edges()}
    score = Counter()
    for left, right in graph.edges():
        edge = tuple(sorted((left, right)))
        if edge in tree_edges:
            continue
        path = nx.shortest_path(forest, left, right)
        for vertex in path:
            score[vertex] += 1
    return {vertex: float(score.get(vertex, 0)) for vertex in graph.nodes()}


def _community_boundary_score(graph: nx.Graph) -> dict[int, float]:
    if graph.number_of_edges() == 0:
        return {vertex: 0.0 for vertex in graph.nodes()}
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
    community_of = {}
    for idx, community in enumerate(communities):
        for vertex in community:
            community_of[int(vertex)] = idx
    scores = {}
    for vertex in graph.nodes():
        scores[int(vertex)] = float(
            sum(1 for neighbor in graph.neighbors(vertex) if community_of[int(neighbor)] != community_of[int(vertex)])
        )
    return scores


def _spectral_boundary_score(graph: nx.Graph) -> dict[int, float]:
    if graph.number_of_nodes() <= 2 or graph.number_of_edges() == 0:
        return {vertex: 0.0 for vertex in graph.nodes()}
    try:
        laplacian = nx.laplacian_matrix(graph).astype(float).toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        fiedler = eigenvectors[:, 1]
    except Exception:
        return {vertex: 0.0 for vertex in graph.nodes()}
    threshold = float(np.median(fiedler))
    signs = {int(vertex): int(fiedler[idx] >= threshold) for idx, vertex in enumerate(graph.nodes())}
    scores = {}
    for idx, vertex in enumerate(graph.nodes()):
        cross_edges = sum(1 for neighbor in graph.neighbors(vertex) if signs[int(neighbor)] != signs[int(vertex)])
        closeness = 1.0 / (abs(float(fiedler[idx]) - threshold) + 1e-9)
        scores[int(vertex)] = float(cross_edges) + closeness
    return scores


def _phase_distance(residue: int, modulus: int) -> float:
    residue_mod = int(residue) % int(modulus)
    return float(min(residue_mod, (int(modulus) - residue_mod) % int(modulus)))


def _compose_output_masks_free(output_masks: list[int], row_masks: tuple[int, ...], n_free: int) -> tuple[int, ...]:
    counts = [0] * n_free
    for output_mask in output_masks:
        free_mask = 0
        tmp = int(output_mask)
        while tmp:
            bit = tmp & -tmp
            idx = bit.bit_length() - 1
            free_mask ^= int(row_masks[idx])
            tmp ^= bit
        tmp_free = free_mask
        while tmp_free:
            bit = tmp_free & -tmp_free
            counts[bit.bit_length() - 1] += 1
            tmp_free ^= bit
    return tuple(counts)


def _support_arrays(row_masks: tuple[int, ...], n_qubits: int, n_free: int) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    if n_qubits <= 0:
        empty = tuple(tuple() for _ in range(n_free))
        return empty, empty
    slice_count = max(1, len(row_masks) // n_qubits)
    qubit_support = [[0] * n_qubits for _ in range(n_free)]
    slice_support = [[0] * slice_count for _ in range(n_free)]
    for row_idx, mask in enumerate(row_masks):
        qubit = row_idx % n_qubits
        slice_idx = min(slice_count - 1, row_idx // n_qubits)
        tmp = int(mask)
        while tmp:
            bit = tmp & -tmp
            var = bit.bit_length() - 1
            qubit_support[var][qubit] += 1
            slice_support[var][slice_idx] += 1
            tmp ^= bit
    return tuple(tuple(row) for row in qubit_support), tuple(tuple(row) for row in slice_support)


def _entropy(values: Iterable[int]) -> float:
    positive = [float(value) for value in values if value > 0]
    total = sum(positive)
    if total <= 0:
        return 0.0
    probs = [value / total for value in positive]
    return float(-sum(prob * math.log(prob + 1e-300) for prob in probs))


def _weighted_graph_lift(qubit_support: tuple[tuple[int, ...], ...], qubit_scores: dict[int, float]) -> tuple[float, ...]:
    lifted = []
    for support_row in qubit_support:
        score = 0.0
        for qubit, count in enumerate(support_row):
            if count:
                score += float(count) * float(qubit_scores.get(qubit, 0.0))
        lifted.append(score)
    return tuple(lifted)
