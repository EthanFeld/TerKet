# TerKet Backend Inventory

Date: 2026-05-21

Scope: inventory only. No backend removal approved here.

Classification:
- `CORE EXACT`: required default exact path
- `PERF EXACT`: exact specialization kept only if benchmark value holds
- `PREPROCESS`: transform feeding another exact backend, not independent backend
- `OPT-IN APPROX`: approximation gated by `SolverConfig.allow_approximate`
- `REMOVED/DEAD`: code path removed or selector always false
- `COMPAT`: metadata/import compatibility only

## Current Metadata Problem

`phase3_backend` is overloaded. It reports:
- q3-free labels: `q3_free`, `quadratic_tensor`
- Phase-3 labels: `treewidth_dp`, `treewidth_dp_peeled`, `q3_cover`, `q3_separator`, `q3_treewidth_cutset`, `cubic_contraction`
- approximate labels: `arbitrary_bethe_bp`, `arbitrary_factor_bethe_bp`, `arbitrary_sparse_parity_bethe_bp`, `*_heuristic`, `pauli_beam_approx`, `native_mps_approx`
- branch aggregate label: `mixed`

Target: keep compatibility, but later split metadata into:
- `backend_family`
- `backend_name`
- `is_approximate`
- `approx_backend`
- `approx_validation`
- `backend_mode`

## Q3-Free Backends

| Family | Current symbols/labels | Class | Keep now | Bloat issue | Step-6 direction |
|---|---|---|---|---|---|
| constant | `_Q3FreeConstraintComponentPlan.backend == "constant"` | CORE EXACT | yes | precomputed component is clear | keep |
| product | `_product_q1_sum_scaled`, isolated vars, cutset `remaining_backend == "product"` | CORE EXACT | yes | tiny, useful | keep |
| forest | `_forest_transfer_sum_scaled`, backend `"forest"` | CORE EXACT | yes | clear q2 forest path | keep |
| treewidth | `_q3_free_treewidth_order`, `_sum_q3_free_treewidth_dp_scaled_batch`, backend `"treewidth"` | CORE EXACT | yes | main exact q3-free DP | keep one planner/evaluator |
| generic factor-table fallback | backend `"generic"`, `_evaluate_q3_free_component_plan_scaled`, recursive component fallback | CORE EXACT | yes | label hides many subpaths | keep one generic evaluator |
| binary-phase quadratic plan | `_build_binary_phase_quadratic_plan`, `_evaluate_binary_phase_quadratic_plan_scaled_batch` | PERF EXACT | yes pending data | tucked inside generic plan | keep only if benchmark win |
| half-phase unary expansion | `_sum_half_phase_q2_unary_expansion_*` | PERF EXACT | yes pending data | overlaps binary-phase plan/generic fallback | coalesce into q3-free factor evaluator |
| bad-q2 cover branch | `_minimum_bad_q2_vertex_cover`, `_sum_q3_free_via_bad_q2_cover_scaled` | PERF EXACT | yes pending data | separate branch path for q2 obstruction | treat as preprocessing/branch mode |
| mediator reduction | `_build_half_phase_mediator_plan`, `_evaluate_half_phase_mediator_plan_scaled` | PREPROCESS | yes pending data | acts like backend field, but is transform | convert to preprocessing transform |
| generic q2 mediator reduction | `_build_generic_q2_mediator_plan`, `_evaluate_generic_q2_mediator_plan_scaled` | PREPROCESS | yes pending data | duplicates mediator concept | merge with mediator interface |
| half-phase cluster reduction | `_build_half_phase_cluster_plan`, `_build_q1_cluster_plan`, `_evaluate_half_phase_cluster_plan_scaled` | PREPROCESS | yes pending data | several cluster flavors hidden in generic path | fold into preprocessing strategy |
| cutset conditioning | `_q3_free_cutset_conditioning_plan`, `_sum_q3_free_via_cutset_conditioning_scaled` | PERF EXACT | yes | main high-width escape | keep one evaluator |
| one-shot cutset conditioning | `_q3_free_one_shot_cutset_conditioning_plan`, `_sum_q3_free_via_one_shot_cutset_scaled` | PERF EXACT | yes pending data | duplicates regular cutset evaluator/search | merge with cutset mode flag |
| raw-constraint reusable plan | `_Q3FreeRawConstraintPlan`, `_build_q3_free_raw_constraint_plan`, `_evaluate_q3_free_raw_constraint_plan_scaled_batch` | PERF EXACT | yes pending batch data | batch-only path, large surface | keep only if batch benchmark win |
| dense Schur complement | `_schur_complement_q3_free_sum_scaled`, `_schur_complement_q3_free_sum_scaled_dense` | PERF EXACT | yes pending data | separate dense direct path | compare vs treewidth/generic; maybe tiny dense mode |
| native q3-free treewidth | `_build_native_q3_free_treewidth_plan`, native preplanned batch calls | PERF EXACT | yes | adapter mixed with planner | keep as adapter behind treewidth backend |
| BL26 quadratic tensor | `_sum_bl26_quadratic_tensor_component_scaled`, metadata `quadratic_tensor` | CORE EXACT/COMPAT | yes | reported as backend family, but mostly q3-free subtype | keep metadata alias; internal as q3-free fast path |
| tensor hint slicing | `_q3_free_tensor_slice_hint`, kahypar-assisted cutset search | PERF EXACT | yes pending data | optional-dep hint mixed into q3-free selector | keep as cutset candidate hint only |
| mixed | metadata aggregate `mixed` | COMPAT | yes | hides per-branch backend detail | keep alias, later add branch backend list |

## Phase-3 Backends

| Family | Current symbols/labels | Class | Keep now | Bloat issue | Step-7 direction |
|---|---|---|---|---|---|
| treewidth DP | `_sum_via_treewidth_dp_scaled`, label `treewidth_dp` | CORE EXACT | yes | main low-width cubic path | keep |
| peeled treewidth DP | `_Phase3BackendCandidate(backend="treewidth_dp", peeled=True)`, label `treewidth_dp_peeled` | PERF EXACT | yes pending data | metadata alias for same treewidth selector/evaluator family | keep compat label; internal as `treewidth_dp` plan flag |
| q3 cover | `_sum_via_q3_cover`, label `q3_cover` | CORE EXACT | yes | general fallback | keep |
| q3 separator | `_find_small_q3_separator`, `_sum_via_q3_separator`, label `q3_separator` | PERF EXACT | yes pending data | tiny separator branch competes with cover | keep only if sparse separator benchmarks win |
| q3 treewidth cutset | `_find_q3_treewidth_cutset`, `_sum_via_q3_treewidth_cutset`, label `q3_treewidth_cutset` | PERF EXACT | yes pending data | another branch-to-DP strategy | compare against q3 cover/separator |
| cubic contraction | `.cubic_contraction` import, `_prefer_cubic_contraction_phase3`, label `cubic_contraction` | PERF EXACT | yes if optional module exists and data wins | optional exact backend mixed into core selector | isolate optional module; benchmark gate |
| tensor contraction | `_sum_via_tensor_contraction`, label path disabled | REMOVED/DEAD | no | internal compat stub only; selector/preference/export removed | delete compat stub after benchmark harness stops importing it |
| hybrid contraction | `_build_reduced_tensor_network`, `_contract_reduced_network` | REMOVED/DEAD | no | internal compat stub only; selector/preference/export removed | delete compat stub after benchmark harness stops importing it |
| native level3 treewidth | `_sum_native_level3_phase3_treewidth_preplanned`, native `sum_treewidth_dp_level3` | PERF EXACT | yes | native detail inside DP evaluator | keep as treewidth adapter |
| native phase-function batch treewidth | `_sum_native_phase_function_treewidth_batch_shared_support` | PERF EXACT | yes | batch native detail inside Python selector | keep as treewidth batch adapter |
| direct Phase-3 escape before eliminations | `_pre_exact_phase3_treewidth_escape` | PERF EXACT | yes pending data | selector bypasses normal reduction order | benchmark gate; document as pre-elim mode |

## Arbitrary-Angle Paths

| Family | Current symbols/labels | Class | Keep now | Bloat issue | Step-8 direction |
|---|---|---|---|---|---|
| unary arbitrary factors exact | `_sum_q3_free_with_unary_arbitrary_phases_scaled` | CORE EXACT | yes | narrow useful fast path | keep exact path |
| dense/general factor path sum exact | `solve_arbitrary_exact`, `_sum_with_arbitrary_phases_exact_scaled`, `_sum_factor_tables_scaled`, label `arbitrary_path_sum` | CORE EXACT | yes | exact arbitrary default | keep |
| cutset arbitrary path sum exact | `_find_arbitrary_factor_cutset_plan`, `_sum_factor_tables_with_cutset_scaled`, label `arbitrary_path_sum_cutset` | PERF EXACT | yes | separate from generic factor table evaluator | keep as exact fallback mode |
| Bethe/BP approximate | `solve_arbitrary_approx`, `_sum_pairwise_factor_graph_bethe_scaled`, `_sum_factor_graph_bethe_scaled`, labels `arbitrary_bethe_bp`, `arbitrary_factor_bethe_bp` | OPT-IN APPROX | yes, opt-in only | implementation still in `_engine_impl`, facade in `approx.py` | keep isolated from exact wrapper |
| sparse parity Bethe/BP approximate | `solve_arbitrary_approx`, `_sum_factor_graph_with_sparse_parity_bethe_scaled`, label `arbitrary_sparse_parity_bethe_bp` | OPT-IN APPROX | yes, opt-in only | extra BP variant | keep in `approx.py`; compare to generic BP |
| BP heuristic ensemble | `_sum_arbitrary_bp_heuristic_ensemble_scaled`, labels `*_heuristic` | OPT-IN APPROX | yes, opt-in only | experimental acceptance heuristics isolated behind opt-in wrapper | keep metadata-gated |
| invalid BP labels | `*_invalid_scale`, `arbitrary_bethe_bp_invalid`, `arbitrary_bethe_bp_normalized` | COMPAT/OPT-IN APPROX | yes | status encoded in backend string | replace with structured fields later |

## Pauli/Observable Approx Paths

| Family | Current symbols/labels | Class | Keep now | Bloat issue | Direction |
|---|---|---|---|---|---|
| exact replay Pauli expectation | `compute_circuit_pauli_expectations`, `_prepare_pauli_expectation_request`, `_build_pauli_expectation_base_state`, state replay | CORE EXACT | yes | now split into named phases; impl still in `_engine_impl` | keep exact facade in `pauli.py` |
| direct post replay | `_DirectPostReplayTemplate`, `_select_pauli_direct_replay_template`, `_build_direct_post_replay_template` | PERF EXACT | yes pending data | specialized many-observable shortcut | benchmark many-observable cases |
| Pauli beam approximate | `compute_circuit_pauli_expectations_approx`, `_compute_pauli_beam_approx_fast_path`, `_pauli_beam_approx_pauli_expectations`, label `pauli_beam_approx` | OPT-IN APPROX | yes, opt-in only | moved out of exact facade | exposed in `pauli_approx.py` and `approx.py` |
| native MPS approximate | `compute_circuit_pauli_expectations_approx`, `_compute_native_mps_approx_pauli_expectations`, `_NativeApproxMPS`, label `native_mps_approx` | OPT-IN APPROX | yes, opt-in only | moved out of exact facade | exposed in `pauli_approx.py` and `approx.py` |

## Backend Labels To Preserve For Compatibility

Keep these labels stable until a metadata migration lands:
- `q3_free`
- `quadratic_tensor`
- `mixed`
- `treewidth_dp`
- `treewidth_dp_peeled`
- `q3_cover`
- `q3_separator`
- `q3_treewidth_cutset`
- `cubic_contraction`
- `arbitrary_path_sum`
- `arbitrary_path_sum_cutset`
- `arbitrary_bethe_bp`
- `arbitrary_factor_bethe_bp`
- `arbitrary_sparse_parity_bethe_bp`
- `arbitrary_bethe_bp_heuristic`
- `arbitrary_factor_bethe_bp_heuristic`
- `arbitrary_sparse_parity_bethe_bp_heuristic`
- `pauli_beam_approx`
- `native_mps_approx`

Removed/dead labels/functions not worth preserving in public docs:
- `tensor_contraction`
- `hybrid_contraction`
- `_sum_via_tensor_contraction`
- `_build_reduced_tensor_network`
- `_contract_reduced_network`

## Bloat Hotspots

1. Q3-free has one `backend` field but many hidden subbackend fields:
   `binary_phase_plan`, `mediator_plan`, `generic_mediator_plan`,
   `cluster_plan`, `cutset_plan`, `direct_schur_ok`, `native_treewidth_plan`.
2. Cutset exists as regular, one-shot, reusable execution, raw-constraint,
   and arbitrary-factor cutset. Evaluators should converge before deletion.
3. Phase-3 still keeps removed tensor/hybrid compat stubs for benchmark harness imports.
4. Approximate arbitrary/Pauli paths now expose `is_approximate`, `approx_backend`,
   and `approx_validation`; backend-string compatibility remains.
5. Native adapters are embedded in planner/evaluator code instead of wrapper
   modules.
6. `mixed` loses branch-level backend detail; good for compatibility, weak for
   backend decisions.

## Inventory Decisions

Immediate keep:
- q3-free `constant`, `product`, `forest`, `treewidth`, `generic`
- Phase-3 `treewidth_dp`, `q3_cover`
- arbitrary exact `arbitrary_path_sum`, `arbitrary_path_sum_cutset`
- native exact adapters whenever a native plan can be built for the same
  treewidth problem

Keep behind benchmark gate:
- binary-phase quadratic
- half-phase unary expansion
- mediator/generic mediator/cluster preprocessing
- bad-q2 cover
- cutset vs one-shot cutset vs raw-constraint reusable
- dense Schur complement
- cubic contraction
- q3 separator
- q3 treewidth cutset
- direct post replay

Move/isolate, not delete:
- arbitrary BP variants
- Pauli beam approximate now isolated in `pauli_approx.py`
- native MPS approximate now isolated in `pauli_approx.py`
- native q3-free/Phase-3 adapters

Delete/unexport candidates:
- removed tensor contraction stubs
- removed hybrid contraction stubs

Next required work before coalescing:
1. Add backend disable flags for A/B.
2. Run baseline compare with each candidate disabled.
3. Record KEEP/COALESCE/DELETE/OPT-IN in this doc with data.
4. Delete at most one backend family per patch.
