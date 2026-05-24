"""Heuristic catalogs for dense-core QAOA experiments."""

from __future__ import annotations

from benchmarks.targeted.dense_core._dense_core_types import *  # noqa: F403


def heuristic_specs() -> list[HeuristicSpec]:
    def quantum_tensor_hotspot(case: DenseCoreCase, _max_budget: int) -> list[int]:
        return _score_sort_primary(case.graph_edge_betweenness_incident, case.graph_betweenness)

    def quantum_gatecut_lift(case: DenseCoreCase, _max_budget: int) -> list[int]:
        combined = [
            0.60 * case.original_boundary_lift[var]
            + 0.25 * case.original_betweenness_lift[var]
            + 0.15 * case.output_support[var]
            for var in range(case.free_var_count)
        ]
        return _score_sort_primary(combined, case.slice_span)

    def quantum_problem_degree_lift(case: DenseCoreCase, _max_budget: int) -> list[int]:
        return _score_sort_primary(case.original_degree_lift, case.slice_span)

    def quantum_problem_betweenness_lift(case: DenseCoreCase, _max_budget: int) -> list[int]:
        return _score_sort_primary(case.original_betweenness_lift, case.original_degree_lift)

    def quantum_slice_span(case: DenseCoreCase, _max_budget: int) -> list[int]:
        combined = [case.slice_span[var] + 0.50 * case.output_support[var] for var in range(case.free_var_count)]
        return _score_sort_primary(combined, case.support_entropy)

    def ising_bad_edge_cover(case: DenseCoreCase, _max_budget: int) -> list[int]:
        cover_first = list(case.bad_edge_cover_rank)
        remainder = [var for var in _score_sort_primary(case.bad_edge_incidence, case.graph_degree) if var not in set(cover_first)]
        return cover_first + remainder

    def ising_local_field(case: DenseCoreCase, _max_budget: int) -> list[int]:
        return _score_sort_primary(case.local_field, case.bad_edge_incidence)

    def ising_loop_hotspot(case: DenseCoreCase, _max_budget: int) -> list[int]:
        combined = [case.graph_cycle[var] + 0.50 * case.graph_betweenness[var] for var in range(case.free_var_count)]
        return _score_sort_primary(combined, case.graph_degree)

    def ising_cavity_pressure(case: DenseCoreCase, _max_budget: int) -> list[int]:
        return _score_sort_primary(case.cavity_pressure, case.local_field)

    def ising_spin_glass_core(case: DenseCoreCase, _max_budget: int) -> list[int]:
        return _score_sort_primary(case.spin_glass_core, case.bad_edge_incidence)

    def graph_degree(case: DenseCoreCase, _max_budget: int) -> list[int]:
        return _score_sort_primary(case.graph_degree, case.graph_betweenness)

    def graph_betweenness(case: DenseCoreCase, _max_budget: int) -> list[int]:
        return _score_sort_primary(case.graph_betweenness, case.graph_degree)

    def graph_minfill_tail(case: DenseCoreCase, _max_budget: int) -> list[int]:
        return _reverse_order(case.min_fill_order, case.free_var_count)

    def graph_nested_separator(case: DenseCoreCase, _max_budget: int) -> list[int]:
        if case.separator_order:
            return _reverse_order(case.separator_order, case.free_var_count)
        return _reverse_order(case.min_fill_order, case.free_var_count)

    def graph_spectral_boundary(case: DenseCoreCase, _max_budget: int) -> list[int]:
        return _score_sort_primary(case.spectral_boundary, case.graph_boundary)

    def own_bridge_lift_hybrid(case: DenseCoreCase, _max_budget: int) -> list[int]:
        maps = [
            _normalize_map({idx: value for idx, value in enumerate(case.graph_betweenness)}),
            _normalize_map({idx: value for idx, value in enumerate(case.original_degree_lift)}),
            _normalize_map({idx: value for idx, value in enumerate(case.slice_span)}),
        ]
        combined = [
            0.45 * maps[0].get(var, 0.0) + 0.35 * maps[1].get(var, 0.0) + 0.20 * maps[2].get(var, 0.0)
            for var in range(case.free_var_count)
        ]
        return _score_sort_primary(combined, case.graph_degree)

    def own_community_bridge(case: DenseCoreCase, _max_budget: int) -> list[int]:
        combined = [
            case.graph_boundary[var] + 0.75 * case.original_boundary_lift[var] + 0.25 * case.graph_cycle[var]
            for var in range(case.free_var_count)
        ]
        return _score_sort_primary(combined, case.graph_betweenness)

    def own_gamma_entropy(case: DenseCoreCase, _max_budget: int) -> list[int]:
        combined = [
            case.support_entropy[var] + 0.25 * case.output_support[var] + 0.10 * case.slice_span[var]
            for var in range(case.free_var_count)
        ]
        return _score_sort_primary(combined, case.original_degree_lift)

    def greedy_rank(case: DenseCoreCase, max_budget: int, *, mode: str) -> list[int]:
        selected: list[int] = []
        remaining = set(range(case.free_var_count))
        for _step in range(min(max_budget, case.free_var_count)):
            best_var = None
            best_score = None
            for candidate in remaining:
                cutset = tuple(selected + [candidate])
                evaluation = _evaluate_q3_free_cutset_candidate(
                    case.q,
                    cutset,
                    allow_generic_remaining=True,
                    prefer_one_shot_slicing=True,
                    target_remaining_width=TARGET_REMAINING_WIDTH,
                )
                if evaluation is None or evaluation.plan is None:
                    score = (10**9, 10**18, candidate)
                else:
                    plan = evaluation.plan
                    generic_penalty = 1 if plan.remaining_backend == "generic" else 0
                    if mode == "width":
                        score = (
                            int(plan.remaining_width),
                            int(generic_penalty),
                            int(plan.estimated_total_work),
                            candidate,
                        )
                    else:
                        score = (
                            int(plan.estimated_total_work),
                            int(plan.remaining_width),
                            int(generic_penalty),
                            candidate,
                        )
                if best_score is None or score < best_score:
                    best_var = int(candidate)
                    best_score = score
            if best_var is None:
                break
            selected.append(best_var)
            remaining.remove(best_var)
        remainder = [
            var
            for var in own_bridge_lift_hybrid(case, max_budget)
            if var not in set(selected)
        ]
        return selected + remainder

    def own_greedy_width_drop(case: DenseCoreCase, max_budget: int) -> list[int]:
        return greedy_rank(case, max_budget, mode="width")

    def own_greedy_work_drop(case: DenseCoreCase, max_budget: int) -> list[int]:
        return greedy_rank(case, max_budget, mode="work")

    return [
        HeuristicSpec("quantum", "quantum_tensor_hotspot", "TN/circuit-cut hotspot on q2 factor graph.", quantum_tensor_hotspot),
        HeuristicSpec("quantum", "quantum_gatecut_lift", "Original-graph gate-cut boundary lifted through gamma support.", quantum_gatecut_lift),
        HeuristicSpec("quantum", "quantum_problem_degree_lift", "Original QAOA problem-graph degree lifted to free vars.", quantum_problem_degree_lift),
        HeuristicSpec("quantum", "quantum_problem_betweenness_lift", "Original QAOA problem-graph betweenness lifted to free vars.", quantum_problem_betweenness_lift),
        HeuristicSpec("quantum", "quantum_slice_span", "Free vars touching many qubits and internal QAOA slices.", quantum_slice_span),
        HeuristicSpec("ising", "ising_bad_edge_cover", "Spin-glass bad-edge cover first, then remaining frustration mass.", ising_bad_edge_cover),
        HeuristicSpec("ising", "ising_local_field", "Large unary plus incident phase load first.", ising_local_field),
        HeuristicSpec("ising", "ising_loop_hotspot", "Cycle/loop-heavy variables first.", ising_loop_hotspot),
        HeuristicSpec("ising", "ising_cavity_pressure", "High cavity pressure from local neighborhood first.", ising_cavity_pressure),
        HeuristicSpec("ising", "ising_spin_glass_core", "Deep core plus frustration incidence first.", ising_spin_glass_core),
        HeuristicSpec("graph", "graph_degree", "Highest q2 degree first.", graph_degree),
        HeuristicSpec("graph", "graph_betweenness", "Highest q2 node betweenness first.", graph_betweenness),
        HeuristicSpec("graph", "graph_minfill_tail", "Tail of min-fill elimination order.", graph_minfill_tail),
        HeuristicSpec("graph", "graph_nested_separator", "Tail of nested-dissection separator order.", graph_nested_separator),
        HeuristicSpec("graph", "graph_spectral_boundary", "Spectral bisection boundary first.", graph_spectral_boundary),
        HeuristicSpec("own", "own_greedy_width_drop", "Greedy add vars that minimize remaining cutset width proxy.", own_greedy_width_drop),
        HeuristicSpec("own", "own_greedy_work_drop", "Greedy add vars that minimize cutset work proxy.", own_greedy_work_drop),
        HeuristicSpec("own", "own_bridge_lift_hybrid", "Blend q2 bridge score with original-graph lift.", own_bridge_lift_hybrid),
        HeuristicSpec("own", "own_community_bridge", "Cross-community bridge vars on reduced and original graphs.", own_community_bridge),
        HeuristicSpec("own", "own_gamma_entropy", "High-support-entropy vars across qubits and slices.", own_gamma_entropy),
    ]
