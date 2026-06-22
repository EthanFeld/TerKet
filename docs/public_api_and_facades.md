# Public API And Facades

TerKet's stable public import surface is intentionally small.

## Facade Kinds

Use these labels consistently in module docstrings and review discussion:

- stable public API:
  modules intended for normal callers, primarily `terket`
- focused internal facade:
  grouped domain helpers with stable owner intent but no public compatibility
  promise, e.g. `terket.q3free`, `terket.phase3`, `terket.state`
- compatibility facade:
  old import path preserved while callers migrate, e.g. `terket.engine`
- module-replacement shim:
  file whose only job is aliasing old module path to new owner module

Readability rule: compat facades and shims must stay narrow, documented, and
easy to delete later.

## Stable Public Imports

Use `import terket` for normal callers:

- `CircuitSpec`
- `ScaledAmplitude`
- `SchurState`
- `SolverConfig`
- `CubicFunction`
- `PhaseFunction`
- `make_circuit`
- `normalize_circuit`
- `from_qiskit`
- `compute_circuit_amplitude`
- `compute_circuit_amplitude_scaled`
- `compute_circuit_probability_doubled`
- `compute_circuit_pauli_expectation_probabilities_doubled`
  - estimates squared Pauli expectations through observable-aware state replay;
    it does not preserve expectation sign.
- `DoubledFactorProblem`
- `compute_amplitude`
- `compute_amplitude_scaled`
- `compute_amplitudes`
- `analyze_circuit`
- `analyze_amplitudes`
- `compute_circuit_pauli_expectations`
- `reduce_and_sum`
- `sum_doubled_phase`
- `sum_doubled_factor_problem`
- `cache_stats`
- `clear_caches`
- bit-string helpers

## Compatibility Facades

These stay for current internal/test users and one release of compatibility:

- `terket.engine`: aliases `_engine_impl` so old private monkeypatches still work.
- `terket.schur_engine`: compatibility facade for historical engine imports.
- `terket.circuits`: circuit normalization/interoperability facade.

## Module-Replacement Shims

Current module-replacement shims:

- `terket._engine_impl`
- `terket._phase3_cover`
- `terket._phase3_exec`
- `terket._phase3_factors`
- `terket._phase3_order`
- `terket._phase3_select`
- `terket._phase3_structure`
- `terket._q3free_clusters`
- `terket._q3free_factor_plans`
- `terket._q3free_primitives`

These should not grow behavior. They exist only to preserve old import paths
while owner modules move under `_phase3/` and `_q3free/`.

## Focused Internal Facades

These group private helpers by domain. They are not the public API:

- `terket.phase3`
- `terket.q3free`
- `terket.arbitrary`
- `terket.doubled`
- `terket.pauli`
- `terket.phase_function`
- `terket.reduction`
- `terket.scaling`
- `terket.state`
- `terket.native`
- `terket.interop.*`

Tests may import these only when intentionally checking internals.

## Benchmark Helpers

`terket.benchmarking` is benchmark-owned support code. Public package import must not load it.

## Removed Surface

Removed tensor-contraction Phase-3 stubs are not exported. Benchmark probes must use maintained exact treewidth/q3-free paths instead.
