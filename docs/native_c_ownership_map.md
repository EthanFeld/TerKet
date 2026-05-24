# Native C Ownership Map

Native C is optional acceleration. Python remains source of truth for behavior.

## Current Files

1. `src/terket/_schur_native.c`
   - Owns Python module definition.
   - Owns exported method table.
   - Owns module lifecycle cleanup.
   - Must stay small.
   - New native kernels must not be added here without updating ABI guard tests.

2. `src/terket/_schur_native_internal.h`
   - Owns shared structs, constants, and cross-file declarations.
   - Owns capsule names and shared globals declarations.
   - Must not accumulate algorithm bodies.

3. `src/terket/_schur_native_support.c`
   - Owns support-mask cache.
   - Owns bitset helpers.
   - Owns complex/scaled-complex arithmetic.
   - Owns parsing helpers for C extension inputs.
   - Owns plan/factor memory cleanup and capsule destructors.
   - Owns small comparison and key parsing helpers.

4. `src/terket/_schur_native_graph.c`
   - Owns graph/order helpers.
   - Owns min-fill/min-degree order.
   - Owns cubic order width.
   - Owns q3-free cutset extension ranking.
   - Must not grow DP evaluation code.

5. `src/terket/native_phase_eval.c`
   - Owns `evaluate_q_mask_terms_native`.
   - Keeps direct q1/q2/q3 residue evaluation separate from elimination/build code.

6. `src/terket/native_constraint_elim.c`
   - Owns single/two-partner constraint elimination kernels.
   - Owns shared packed-dict algebra helpers through `native_algebra_helpers.inc`.

7. `src/terket/native_affine_compose.c`
   - Owns `aff_compose_terms_native`.
   - Shares affine helper bodies through `native_algebra_helpers.inc`.

8. `src/terket/native_classification.c`
   - Owns classification-data and classification-lookup builders.
   - Owns structure-key serialization.
   - Shares binary loaders through `native_binary_loaders.inc`.

9. `src/terket/native_output_solve.c`
   - Owns 64-bit output shift-mask solvers.
   - Shares binary loaders through `native_binary_loaders.inc`.

10. `src/terket/native_level3_dp.c`
    - Owns level-3-only treewidth plan/eval entrypoints.
    - Includes `native_level3_dp_core.inc`.

11. `src/terket/native_phase_function_dp.c`
    - Owns generic phase-function and scaled-factor treewidth plan/eval entrypoints.
    - Includes `native_phase_function_dp_core.inc`.

12. `src/terket/native_q3_free_dp.c`
    - Owns q3-free treewidth plan/eval entrypoints.
    - Owns q3-free work estimator and batch eval entrypoints.
    - Includes `native_q3_free_dp_core.inc`.

13. Private `.inc` fragments
    - `native_algebra_helpers.inc`
    - `native_binary_loaders.inc`
    - `native_level3_dp_core.inc`
    - `native_phase_function_dp_core.inc`
    - `native_q3_free_dp_core.inc`
    - Not compiled directly.
    - Exist only to keep family translation units small and ownership-separated.

## Retired Monolith Stubs

1. `src/terket/_schur_native_algebra.c`
   - Breadcrumb stub only.
   - Not compiled by `setup.py`.

2. `src/terket/_schur_native_dp.c`
   - Breadcrumb stub only.
   - Not compiled by `setup.py`.

## Split Rules

1. Do not change Python-visible method names during file split.
2. Do not change capsule names during file split.
3. Keep `setup.py` source order explicit.
4. Run native enabled and `TERKET_DISABLE_NATIVE=1` build checks after edits.
5. Run native/Python parity tests after edits.
6. Do not add new native kernels until backend list is smaller.
