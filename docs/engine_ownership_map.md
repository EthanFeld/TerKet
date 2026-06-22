# TerKet Engine Ownership Map

Date: 2026-05-21

Scope: current monolithic implementation in `src/terket/_engine_impl.py`.
`src/terket/engine.py` is compatibility alias only.

Rule: this map describes ownership. It does not approve behavior changes,
backend deletion, or performance tradeoffs.

## Section Map

| Lines | Current part | Owns | Target owner | Extract risk |
|---:|---|---|---|---|
| 1-53 | imports | shared deps | keep local per module | high circular risk if copied blindly |
| 55-97 | native load + core aliases | `_schur_native`, `BitSequence`, `ScaledComplex` | `native.py`, `state.py`, `scaling.py` | load order, optional native import |
| 100-145 | protocol/info types | `SupportsQiskitCircuit`, `ReducerInfo`, `ReductionInfo` | `state.py`, `reduction.py` | low |
| 147-340 | global knobs | backend and rewrite thresholds | split by owner module | high: constants drive perf selection |
| 341-443 | solver config/env | `SolverConfig`, default config, env helpers | `state.py` or `config.py` later | medium: contextvars/default config |
| 449-613 | optional deps/native/mask helpers | quimb import, native flags, row/mask primitives, rewrite gates | `native.py`, `state.py`, `reduction.py` | medium: helpers shared widely |
| 616-980 | metadata/cache/plan dataclasses/scaled amp | `_ReductionContext`, `_BoundedMemoCache`, q3-free plans, arbitrary plans, `ScaledAmplitude` | `cache.py`, `scaling.py`, `q3free.py`, `arbitrary.py`, `reduction.py` | high: plan classes cross backend borders |
| 981-1120 | affine/output solve primitives | affine bit/phase mutation, echelon cache, output solve | `state.py`, `reduction.py` | medium: native output solver coupling |
| 1121-2489 | arbitrary phases | arbitrary term coalesce, unary factor tables, cutset factor tables | `arbitrary.py` | high: exact factor paths |
| 2492-3385 | `SchurState` | symbolic state, gate application methods, pending arbitrary phases | `state.py` | high: central object, many call sites |
| 3386-4246 | reducer loop + early helpers | pre-exact Phase-3 escape, reducer recursion, batch reduce, Pauli prep, exact eliminations, omega/product sums | `reduction.py`, `phase3.py`, `pauli.py`, `scaling.py` | high: main semantics |
| 4247-7057 | q3-free primitives/preprocessors | scaled arithmetic arrays, binary/half-phase checks, mediators, clusters, factor cutsets, cluster eval | `q3free.py`, `scaling.py`, `arbitrary.py` | very high: backend selection + perf |
| 7058-8003 | shared caches/build helpers | scaled conversion, early elim heuristics, phase keys, copy/build phase functions, direct replay, native output solve | `scaling.py`, `cache.py`, `phase_function.py`, `pauli.py`, `state.py` | high: many hidden deps |
| 8004-10058 | q3-free planning/execution | q3-free component plans, reusable plans, normal-form rewrite, raw constraints | `q3free.py` | very high: benchmark-sensitive |
| 10059-14353 | q3-free backend families | factorized components, brute force, forest transfer, dense Schur, BL26, bad-q2, nonquadratic support, treewidth, cutset | `q3free.py` | very high: backend bloat target |
| 14354-17464 | Phase-3 planning/execution | treewidth order, structure opt, separator/cutset, backend choice, native plans, tensor stubs, q3 cover | `phase3.py` | very high: backend bloat target |
| 17465-18539 | classification/eliminations/affine compose | BL26 classify, quadratic elim, constraint elim, affine compose, reducer info | `reduction.py` | high: exact semantics |
| 18540-19091 | public amplitude API | `affine_compose`, `reduce_and_sum`, `build_state`, gate replay, batch query, amplitude APIs | public facades + `state.py` | medium: API compat |
| 19092-20073 | Pauli expectations | PauliExpBox controls and exact expectation API | `pauli.py` | high: central observable path |
| 20074-20220 | analysis/compat wrappers | `analyze_*`, old `compute_amplitude*` overloads | public facades | low-medium: public API compat |

## Domain Owners

`native.py`
: native module loading, native symbol lookup, quimb import, native feature flags.

`cache.py`
: `_BoundedMemoCache`; later named registry, cache stats, clear hooks.

`scaling.py`
: `ScaledComplex`, `ScaledAmplitude`, scaled multiply/add/normalize, omega tables, `_scaled_to_complex`.

`phase_function.py`
: `PhaseFunction`/`CubicFunction` re-export now; later phase copy/build/key helpers if they stop needing engine state.

`state.py`
: `SolverConfig`, `SchurState`, gate replay, output echelon solve, `build_state`, batch query state.

`reduction.py`
: `_ReductionContext`, classification, exact eliminations, affine compose, reducer loop, `reduce_and_sum`.

`q3free.py`
: all q3-free plan types, q3-free preprocessing, q3-free backend selection, q3-free evaluators.

`phase3.py`
: cubic residual planning, treewidth DP, q3 separator/cover, cubic/tensor/native residual backends.

`arbitrary.py`
: arbitrary-angle exact factor paths.

`pauli.py`
: PauliExpBox lowering and exact expectation flow.

`interop/`
: QASM/Qiskit import/export and rewrite; engine should import only normalized `CircuitSpec`/`Gate`.

## Extraction Order

1. Extract leaf primitives first: `cache.py`, `scaling.py`, `native.py`.
2. Extract config/types: `SolverConfig`, typed dicts, `BitSequence`, `CircuitInput`.
3. Extract output solve primitives: `EchelonCache`, echelon cache prep, RHS solve.
4. Extract `SchurState` only after scaling/native/output solve are stable.
5. Keep arbitrary factor-table code isolated behind its exact facade.
6. Extract q3-free plan classes before q3-free evaluators.
7. Extract q3-free shared factor-table math before treewidth/cutset backends.
8. Extract Phase-3 order/factor-table helpers before backend selector.
9. Extract classification/eliminations after q3-free and Phase-3 helper imports stop cycling.
10. Move public wrappers last; keep old `terket.engine` alias until private imports gone.

## Circular Import Rules

1. Leaf modules must not import `terket.engine` or `_engine_impl`.
2. `scaling.py`, `cache.py`, `native.py` should depend only on stdlib/numpy/optional deps.
3. `phase_function.py` depends on `cubic_arithmetic.py`; no engine dependency.
4. `state.py` may depend on `circuits.py`, `scaling.py`, `native.py`; avoid `q3free.py` and `phase3.py`.
5. `reduction.py` may call q3-free/Phase-3 through narrow solver interfaces only.
6. `q3free.py` and `phase3.py` may share a future `factor_tables.py`; neither should import the other except through interfaces.
7. Public facades may import modules; implementation modules must not import public facades.
8. Tests for internals should import owner modules, not `terket.engine`, after each extraction lands.

## Private Import Inventory

| Importer | Current private engine symbols | Future import owner |
|---|---|---|
| `tests/test_q3free_normal_form_rewrite.py` | `_build_q3_free_execution_plan`, `_evaluate_q3_free_execution_plan_scaled`, `_q3_free_execution_plan_runtime_score`, `_rewrite_q3_free_phase_to_normal_form`, `_scaled_to_complex` | `q3free.py`, `scaling.py` |
| `scripts/analyze_q3_kernel.py` | `_cubic_order_width`, `_is_half_phase_q2`, `_min_degree_cubic_order_uncached`, `_min_fill_cubic_order`, `_phase_function_from_parts`, `_q3_free_edge_density`, `_q3_free_spanning_data`, `_select_feedback_vertices`, `build_state` | `phase3.py`, `q3free.py`, `phase_function.py`, `state.py` |
| `benchmarks/targeted/dense_core/dense_core_common.py` | `_aff_compose_cached`, `_evaluate_q3_free_cutset_candidate`, `_evaluate_q3_free_cutset_conditioning_plan_scaled`, `_gauss_sum_q3_free_scaled`, `_minimum_bad_q2_vertex_cover`, `_min_fill_cubic_order`, `_pair_graph_separator_order`, `_q3_free_cutset_conditioning_plan`, `_row_masks_from_gamma`, `_scaled_to_complex`, `build_state` | `reduction.py`, `q3free.py`, `phase3.py`, `state.py`, `scaling.py` |
| `benchmarks/targeted/mqt/mqt_bench_head_to_head.py` | `_aff_compose_cached`, `_min_fill_cubic_order`, `build_state` | `reduction.py`, `phase3.py`, `state.py` |
| `benchmarks/targeted/rcs/amplitude_post_elimination_tensor_rcs.py` | `_build_cubic_factors`, `_build_q3_free_raw_constraint_plan`, `_dense_q2_matrix`, `_min_fill_cubic_order`, `_phase_function_from_parts`, `_restrict_q3_free_raw_constraint_plan`, `_scaled_to_complex`, `_schur_complement_q3_free_sum_scaled_dense`, `_sum_q3_free_component_scaled`, `_sum_via_tensor_contraction`, `_treewidth_order_width`, `build_state` | `phase3.py`, `q3free.py`, `phase_function.py`, `scaling.py`, `state.py` |
| `benchmarks/targeted/rcs/rcs_import_strategy_probe.py` | `_build_q3_free_raw_constraint_plan`, `_min_fill_cubic_order`, `_phase_function_from_parts`, `_restrict_q3_free_raw_constraint_plan`, `build_state` | `q3free.py`, `phase3.py`, `phase_function.py`, `state.py` |
| `benchmarks/targeted/dense_core/dense_core_plan_eval.py` | `_scaled_to_complex` | `scaling.py` |
| `tools/profile_rcs_old_vs_new.py` | `import terket.engine as engine` dynamic profiling | keep compat until profiling script rewritten to owner modules |
| `tools/quantinuum_challenge_terket_graphs.py` | `SolverConfig` | public `terket.SolverConfig` or `state.py` |

## Move Notes

1. Do not split q3-free and Phase-3 in same patch.
2. Do not move constants without the scoring/evaluator that uses them.
3. Preserve backend metadata names until performance baseline proves alias change safe.
4. Keep `_engine_impl.py` as compatibility implementation during extraction.
5. After each module extraction: run compileall, full pytest, then baseline compare if backend code moved.
