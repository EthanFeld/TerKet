# Doubled-Sum Approximation

For a restricted dyadic path phase `P(x)`, TerKet writes the probability as

`|sum_x omega^P(x)|^2 = sum_d sum_x omega^(P(x) - P(x xor d))`.

The doubled backend keeps sectors with `|d| <= max_difference_weight`. Every
retained sector is still summed exactly by TerKet's normal reduction machinery.
The approximation comes only from omitted sectors.

## APIs

`compute_circuit_probability_doubled(...)` builds and output-restricts the
normal Schur state, then applies the doubled backend:

```python
result = terket.compute_circuit_probability_doubled(
    circuit,
    input_bits,
    output_bits,
    max_difference_weight=2,
)
probability = result.to_float()
```

`sum_doubled_phase(P, ...)` handles a bare phase function without materializing
`2k` variables. `sum_coupled_doubled_phase(Q, contour_variables=k, ...)`
accepts a genuine coupled phase over variables ordered as `(x..., y...)`.
Mixed contour terms can encode inserted operators, trace constraints, or
averaged noise/twirl factors.

`sum_doubled_factor_problem(problem, ...)` handles general local factors,
including zero-valued projectors, non-unit complex weights, and auxiliary
averaging variables:

```python
problem = terket.DoubledFactorProblem(
    contour_variables=k,
    auxiliary_variables=r,
    phase=phase,
    factors={(0, k): [1, 0, 0, 1]},
    scalar=0.5,
)
result = terket.sum_doubled_factor_problem(problem, max_difference_weight=2)
```

Problem variables are ordered as `(x..., y..., auxiliary...)`. Each factor
scope is strictly increasing and its table has `2 ** len(scope)` entries.

When direct contour-pair factors have strongly unequal diagonal/off-diagonal
magnitudes, a fixed sector budget can use factor-bound ordering:

```python
result = terket.sum_doubled_factor_problem(
    problem,
    max_difference_weight=k,
    max_sectors=256,
    difference_strategy="factor_bound",
)
```

This orders sectors by descending separable magnitude bound from factors with
scope `(x_i, y_i)`. Hard zero pair constraints are exact: impossible sectors
are omitted without approximation. Factor-bound ordering requires
`max_difference_weight=k` and an explicit `max_sectors`; it replaces rather
than combines with Hamming-shell pruning.

For correlated local factors, `general_bound` compiles each factor into a
positive factor over its difference variables, ranks sectors with exact
best-first branch-and-bound, and computes a rigorous omitted-magnitude bound:

```python
result = terket.sum_doubled_factor_problem(
    problem,
    max_difference_weight=k,
    max_sectors=256,
    difference_strategy="general_bound",
    omitted_magnitude_tolerance=1e-6,
)
certificate = result.omitted_magnitude_bound
```

`omitted_magnitude_tolerance` stops after the absolute certificate reaches the
requested tolerance. The certificate includes all omitted sectors, but can be
conservative because it discards phase cancellation. General-bound partition
elimination and branch-and-bound have explicit resource guards.

For arbitrary-angle circuit factors, `difference_strategy="subspace"` contracts
all sectors in a selected coordinate subspace jointly. `max_sectors` must be a
power of two and denotes the number of sectors represented by one exact grouped
contraction. This uses

`sum_(d in H) C(d) = |H| sum_cosets |sum_(x in coset) f(x)|^2`

and can cover exponentially many sectors without enumerating them.

## Execution

- `d=0` for a bare phase is evaluated analytically.
- Full-cutoff bare and coupled phase sums use one direct exact reduction,
  avoiding exponential sector enumeration.
- Full-cutoff general-factor problems use direct exact factor elimination when
  estimated work and table size are within configured limits. Impractical
  exponential sector fallback now raises a clear error.
- Remaining sectors are streamed in `sector_batch_size` chunks, bounding memory.
- Per-sector reduction and affine caches are released after every chunk.
- Bare sectors use direct sparse difference-polynomial construction.
- Coupled sectors use affine substitution `y=x xor d`, using the native affine
  composer when available.
- General-factor sectors restrict all factor tables after that substitution,
  merge duplicated variables, then use exact native generic factor-table
  elimination. Factor tables are scaled once and reused across sectors.
- A restricted zero-valued scalar factor terminates its sector before phase
  composition or elimination.
- Restricted all-zero tables terminate sectors before elimination.
- Repeated reduced factor-scope layouts reuse elimination orders.
- General-bound search caches conditional local-factor maxima and updates only
  factors incident to each newly assigned difference bit.
- Compact arbitrary-angle sectors cancel ket/bra parity phases before exact
  reduction, avoiding dense doubled factor tables.
- Subspace mode adds a small set of difference coordinates and contracts their
  complete sector group in one exact arbitrary path sum.
- Sector sums use the existing exact/native-capable reduction machinery.
- `max_difference_weight >= k` evaluates every sector and is exact.

The circuit-level API converts arbitrary non-dyadic phase factors into local
ket factors and conjugate bra factors, then evaluates them through the general
factor backend.

Truncated sector sums are signed approximations, not guaranteed physical
probabilities. Before full weight they may be negative or exceed one. Callers
should inspect the raw estimate rather than assume positivity.

## Benchmark Findings

Exploratory runs on June 15, 2026 found:

| Case | Exact time | Truncation | Truncated time | Relative error |
|---|---:|---:|---:|---:|
| QAOA ring, 16 qubits | 0.020 s | `w=2`, 529 sectors | 0.042 s | 0.997 |
| QAOA ring, 64 qubits | 0.045 s | `w=1`, 129 sectors | 0.030 s | approximately 1 |
| Grover, 48 qubits | 0.148 s | `w=1`, 136 sectors | 0.033 s | approximately 1 |
| Hidden shift, 192 half-qubits | 0.071 s | `w=1`, 769 sectors | 0.721 s | 1 |

For shallow 9-qubit RCS, `w=3` reached correlation `0.89` across sampled
outputs, but was much slower than exact evaluation and produced negative
estimates. Deeper RCS low-weight sectors frequently gave only the incoherent
baseline.

Random mixed-contour phase tests also showed poor generic convergence:
`w <= 4` had median relative error between `0.69` and `1.01`, while direct
exact reduction was much faster at tested sizes.

Current verdict: not promising as a generic bare-output probability backend.
It remains potentially useful only for coupled problems whose operator,
trace, or averaging structure demonstrably suppresses high-difference sectors.
The general factor-problem API represents those coupled cases; usefulness is
problem dependent and should be checked against increasing difference weight.

Benchmarking exposed and fixed three correctness/resource issues:

- sector caches previously grew with total sector count; peak QAOA16 `w=4`
  traced memory dropped from 23.8 MB to 4.4 MB after chunk cache release;
- level-1 phase reduction could crash or use invalid level-3 elimination rules;
- level-above-3 cubic sectors could use unsafe specialized eliminations and
  return incorrect sums.

Post-fix brute-force fuzz covered 360 bare and 150 coupled phase/cutoff cases.
General-factor fuzz covered another 360 cases with arbitrary complex/zero
factors, auxiliary variables, mixed phases, and every cutoff.
Circuit-level arbitrary-angle full-cutoff fuzz covered 600 outputs against
exact amplitudes.

For structures that exactly suppress off-diagonal sectors, the new factor API
is promising. On local projector and valid auxiliary-twirl models, `w=0`
matched the full sector sum exactly. After zero-sector short-circuiting,
`k=8..12` runs were 113x-2396x faster than full sector enumeration.

Further audit found:

- direct full-cutoff bare reduction was 18x-6500x faster than legacy sector
  enumeration in tested `k=8..14` cases;
- direct low-width general-factor elimination was 51x-476x faster at
  `k=8..12`;
- on phase-plus-anisotropic-damping models, factor-bound ordering reduced
  median relative error from `0.484` to `0.293` at 16 sectors, `0.281` to
  `0.047` at 64 sectors, and `0.056` to `0.007` at 256 sectors;
- uniform random sector sampling had much worse variance than either
  deterministic strategy and is not recommended as default.
- distance-decayed random dropout with unbiased reweighting was also poor for
  bare sums: all tested adaptive runs hit 8192 draws with median relative
  error about 1.0;
- factor-weighted adaptive sampling converged around 80 draws on damped models,
  but equal-work deterministic factor-bound selection had median relative
  error `0.00035` versus `0.0152`, winning 49 of 50 cases. Sampling remains
  useful only when an uncertainty estimate matters more than point accuracy.

General-bound ordering closes the direct-pair-only gap: arbitrary local
couplings now contribute correlated difference-sector scores and a rigorous
omitted-tail certificate.

## Factor-Bound Verdict

Factor-bound is worth pursuing for coupled fidelity/decoherence problems where:

- fixing `d` substantially lowers graph width relative to the original doubled
  graph;
- direct contour-pair factors strongly suppress most difference sectors;
- only a point estimate, not a rigorous error certificate, is required.

On physical `P(x)-P(y)` grid phases with local decoherence factors:

| Case | Exact | Factor-bound | Speedup | Relative error |
|---|---:|---:|---:|---:|
| `7x7`, strong damping, 64 sectors | width 24 | width about 5 | 37x | `3.4e-6` |
| `7x7`, medium damping, 64 sectors | width 24 | width about 5 | 42x | `1.0e-3` |
| `8x8`, strong damping, 64 sectors | 51 s | 0.107 s | 480x | `1.5e-4` |
| `8x8`, strong damping, 256 sectors | 51 s | 0.366 s | 140x | `1.1e-5` |

Main blockers:

- weak damping leaves combinatorially many important sectors, producing large
  error despite speedup;
- fixed budgets degrade with system size unless total off-diagonal bound mass
  remains small;
- magnitude bounds still miss phase cancellation;
- rigorous certificates can remain orders of magnitude looser than actual
  error in weakly suppressed models;
- budget-doubling convergence is useful in strong damping but can falsely
  converge in medium/weak regimes;
- each retained sector still rebuilds and restricts factors.

Correlated grid-factor probes show general-bound can provide real advantage.
At 64 retained sectors, tested `6x6` and `7x7` cases reached about `1.3x` and
`2.2x` speedup over exact evaluation with relative errors below `6e-10`. At 16
sectors, speedups reached `3.7x` and `6.0x`, with relative errors below
`1.4e-7`. Certificate quality varied from useful to very conservative.

Profiling a `7x7`, 64-sector correlated case found native generic elimination
was only about 15% of runtime. A native generic batch ABI therefore has limited
near-term upside. Conditional-max caching reduced total runtime from about
`0.130 s` to `0.064 s`; factor restriction and bound search remain larger
targets.

## Quantinuum TN Challenge Probe

A June 15, 2026 probe used circuits from the local Quantinuum tensor-network
challenge bundle.

- QEC challenge observables tested were exact zero, so they did not provide a
  useful approximation-quality signal.
- On the smallest condensed-matter case, zero-output probability had 280 path
  variables and 784 arbitrary-angle factors. Compact arbitrary-sector
  construction reduced `w=0` from `0.069 s` to `0.015 s` and `w=1` from
  `12.0 s` to `0.303 s`. Both still returned the same `1.3878e-17` estimate.
- `general_bound` evaluated one sector in `0.0027 s`, but its rigorous omitted
  certificate was about `2^224`. Unit-magnitude ket/bra factors give a flat
  difference bound, so correlated ranking has no signal.
- Observable-aware state replay now avoids materializing `U-P-U†`. A nonzero
  depth-14 condensed-matter expectation completed at `w=0` in `0.085 s` and
  `w=1` in `58.9 s`; both remained at the incoherent baseline instead of the
  reference squared expectation `3.91e-9`.
- Compact arbitrary-sector construction also removes the chemistry CO2
  scope-26 dense-factor failure. CO2 zero-output runs completed in `6.5 s` for
  one sector and `132.7 s` for 128 sectors. A nonzero CO2 observable completed
  in `118.5 s` for one sector and `129.0 s` for eight, but both estimates stayed
  near `9.31e-10` versus reference squared expectation `0.7801`.
- Low-disruption phase candidates were also tested. The smallest condensed
  case has arbitrary-parity rank 278 over 280 path variables; its two
  nontrivial arbitrary-phase-nullspace sectors had Hamming weight 56 and
  exponentially tiny contributions. Dyadic snapping of the nonzero
  condensed expectation exceeded 10 minutes. Neither route improves the
  challenge approximation.
- Grouped subspace contraction is computationally effective: on the smallest
  condensed case it covered 4,096 sectors in `0.081 s`, versus `0.303 s` for
  281 individually evaluated sectors. On the nonzero depth-14 observable it
  contracted up to `2^36` selected sectors in `22.7 s`. Low-incidence,
  high-incidence, random, and topology-derived coordinate bases all remained
  at the incoherent baseline.

Challenge verdict: current doubled approximation is not competitive on the
challenge as exposed. Observable-aware replay and compact arbitrary sectors fix
the main execution blockers, but not approximation quality. Useful next work
requires phase-aware sector scoring or a different contour approximation; more
magnitude-bound ranking or native batching cannot fix the flat-bound blocker.
Compact arbitrary sums now require an explicit `max_sectors` budget when a
nonzero Hamming shell contains more than 512 path variables, preventing
accidental combinatorial challenge runs.

The grouped-subspace result sharpens the blocker: challenge coherence is
globally diffuse rather than concentrated in a tractable sector family. Any
method that recovers it must approximate the full contour contraction, not
select individual sectors or small subspaces.
