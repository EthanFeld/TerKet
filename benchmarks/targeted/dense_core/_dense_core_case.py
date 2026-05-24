"""Case extraction helpers for dense-core QAOA experiments."""

from __future__ import annotations

from benchmarks.targeted.dense_core._dense_core_types import *  # noqa: F403


def extract_qaoa_case(size: int) -> DenseCoreCase:
    circuit = bind_deterministic_parameters(
        get_benchmark_alg("qaoa", circuit_size=size, random_parameters=False),
        "qaoa",
        size,
    )
    spec = normalize_circuit(circuit)
    input_bits = (0,) * spec.n_qubits
    output_bits = hash_bits(f"qaoa:{size}:output", spec.n_qubits)
    state = build_state(
        spec.n_qubits,
        spec.gates,
        input_bits,
        global_phase_radians=_circuit_global_phase_radians(spec),
    )
    cache = state._prepare_echelon()
    solved = state._solve_for_output(cache, output_bits)
    if solved is None:
        raise RuntimeError(f"Output restriction unsatisfiable for qaoa:{size}.")
    shift_mask, _eps, gamma, k = solved
    row_masks = tuple(int(mask) for mask in _row_masks_from_gamma(gamma))
    q_free = _aff_compose_cached(state.q, shift_mask, gamma, k)
    if q_free.q3:
        raise RuntimeError(f"Expected q3-free restricted kernel for qaoa:{size}.")

    q2_graph = nx.Graph()
    q2_graph.add_nodes_from(range(q_free.n))
    for left, right in q_free.q2:
        q2_graph.add_edge(int(left), int(right))

    original_graph = _build_original_qaoa_graph(size)
    min_fill_order, min_fill_width = _min_fill_cubic_order(q_free)
    separator_info = _pair_graph_separator_order(q_free)
    separator_order = tuple(separator_info[0]) if separator_info is not None else ()
    separator_width = int(separator_info[1]) if separator_info is not None else None

    graph_degree_map = {int(vertex): float(q2_graph.degree(vertex)) for vertex in q2_graph.nodes()}
    graph_betweenness_map = {int(key): float(value) for key, value in nx.betweenness_centrality(q2_graph).items()}
    graph_core_map = {int(key): float(value) for key, value in nx.core_number(q2_graph).items()}
    graph_cycle_map = _fundamental_cycle_participation(q2_graph)
    graph_boundary_map = _community_boundary_score(q2_graph)
    spectral_boundary_map = _spectral_boundary_score(q2_graph)
    edge_betweenness = nx.edge_betweenness_centrality(q2_graph)
    graph_edge_betweenness_incident_map = defaultdict(float)
    for (left, right), value in edge_betweenness.items():
        graph_edge_betweenness_incident_map[int(left)] += float(value)
        graph_edge_betweenness_incident_map[int(right)] += float(value)

    original_degree_map = {int(vertex): float(original_graph.degree(vertex)) for vertex in original_graph.nodes()}
    original_betweenness_map = {
        int(key): float(value)
        for key, value in nx.betweenness_centrality(original_graph).items()
    }
    original_boundary_map = _community_boundary_score(original_graph)

    qubit_support, slice_support = _support_arrays(row_masks, size, q_free.n)
    output_support = _compose_output_masks_free(state.eps, row_masks, q_free.n)
    original_degree_lift = _weighted_graph_lift(qubit_support, original_degree_map)
    original_betweenness_lift = _weighted_graph_lift(qubit_support, original_betweenness_map)
    original_boundary_lift = _weighted_graph_lift(qubit_support, original_boundary_map)
    slice_span = tuple(
        float(sum(1 for value in support_row if value > 0) * max(1, sum(1 for value in qubit_row if value > 0)))
        for support_row, qubit_row in zip(slice_support, qubit_support)
    )
    support_entropy = tuple(
        _entropy(qubit_row) + _entropy(slice_row)
        for qubit_row, slice_row in zip(qubit_support, slice_support)
    )

    bad_edge_cover = tuple(int(var) for var in _minimum_bad_q2_vertex_cover(q_free))
    bad_edge_cover_set = set(bad_edge_cover)
    bad_edge_incidence = [0.0] * q_free.n
    for left, right in q_free.q2:
        bad_edge_incidence[int(left)] += 1.0
        bad_edge_incidence[int(right)] += 1.0

    local_field = []
    cavity_pressure = []
    spin_glass_core = []
    for var in range(q_free.n):
        unary_term = _phase_distance(int(q_free.q1[var]), q_free.mod_q1)
        edge_term = sum(
            _phase_distance(int(q_free.q2[(min(var, neighbor), max(var, neighbor))]), q_free.mod_q2)
            for neighbor in q2_graph.neighbors(var)
        )
        local_field.append(unary_term + edge_term)
        cavity_pressure.append(
            float(q2_graph.degree(var))
            + sum(float(q2_graph.degree(neighbor)) for neighbor in q2_graph.neighbors(var))
        )
        spin_glass_core.append(graph_core_map.get(var, 0.0) * (1.0 + bad_edge_incidence[var]))

    builtin_plan = _q3_free_cutset_conditioning_plan(q_free)
    builtin_cutset = tuple(int(var) for var in builtin_plan.cutset_vars) if builtin_plan is not None else ()

    return DenseCoreCase(
        size=size,
        circuit=spec,
        q=q_free,
        original_var_count=int(state.q.n),
        free_var_count=int(q_free.n),
        q2_graph=q2_graph,
        original_graph=original_graph,
        min_fill_order=tuple(int(var) for var in min_fill_order),
        min_fill_width=int(min_fill_width),
        separator_order=separator_order,
        separator_width=separator_width,
        row_masks=row_masks,
        qubit_support=qubit_support,
        slice_support=slice_support,
        output_support=tuple(int(value) for value in output_support),
        original_degree_lift=tuple(float(value) for value in original_degree_lift),
        original_betweenness_lift=tuple(float(value) for value in original_betweenness_lift),
        original_boundary_lift=tuple(float(value) for value in original_boundary_lift),
        slice_span=tuple(float(value) for value in slice_span),
        support_entropy=tuple(float(value) for value in support_entropy),
        graph_degree=tuple(float(graph_degree_map.get(var, 0.0)) for var in range(q_free.n)),
        graph_betweenness=tuple(float(graph_betweenness_map.get(var, 0.0)) for var in range(q_free.n)),
        graph_edge_betweenness_incident=tuple(
            float(graph_edge_betweenness_incident_map.get(var, 0.0)) for var in range(q_free.n)
        ),
        graph_core=tuple(float(graph_core_map.get(var, 0.0)) for var in range(q_free.n)),
        graph_cycle=tuple(float(graph_cycle_map.get(var, 0.0)) for var in range(q_free.n)),
        graph_boundary=tuple(float(graph_boundary_map.get(var, 0.0)) for var in range(q_free.n)),
        spectral_boundary=tuple(float(spectral_boundary_map.get(var, 0.0)) for var in range(q_free.n)),
        local_field=tuple(float(value) for value in local_field),
        cavity_pressure=tuple(float(value) for value in cavity_pressure),
        spin_glass_core=tuple(float(value) for value in spin_glass_core),
        bad_edge_incidence=tuple(float(value) for value in bad_edge_incidence),
        bad_edge_cover_rank=bad_edge_cover,
        builtin_cutset=builtin_cutset,
    )
