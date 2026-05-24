# Public API And Facades

TerKet's stable public import surface is intentionally small.

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
- `compute_amplitude`
- `compute_amplitude_scaled`
- `compute_amplitudes`
- `analyze_circuit`
- `analyze_amplitudes`
- `compute_circuit_pauli_expectations`
- `compute_circuit_pauli_expectations_approx`
- `reduce_and_sum`
- `cache_stats`
- `clear_caches`
- bit-string helpers

## Compatibility Facades

These stay for current internal/test users and one release of compatibility:

- `terket.engine`: aliases `_engine_impl` so old private monkeypatches still work.
- `terket.schur_engine`: compatibility facade for historical engine imports.
- `terket.circuits`: circuit normalization/interoperability facade.

## Focused Internal Facades

These group private helpers by domain. They are not the public API:

- `terket.phase3`
- `terket.q3free`
- `terket.arbitrary`
- `terket.approx`
- `terket.pauli`
- `terket.pauli_approx`
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
