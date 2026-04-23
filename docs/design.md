# TerKet Design

## Purpose

TerKet is an exact strong-simulation engine for Clifford+T-oriented circuits and related imported circuits. For a fixed input basis state and output basis state it computes the exact amplitude

`<output|U|input>`

The architecture has three layers:

1. compile the circuit into a Schur-style symbolic state
2. reduce the resulting exact phase sum by algebraic eliminations
3. solve the remaining hard core with the cheapest exact backend that matches its structure

This document covers:

- the end-to-end pipeline
- every elimination family used by the reducer
- every backend family used by the solver

## Main Representations

### `CircuitSpec`

Defined in [src/terket/circuit_spec.py](/c:/Users/ethan/github/bee/TerKet/src/terket/circuit_spec.py). This is the normalized frontend circuit representation:

- qubit count
- normalized gate list
- metadata such as name and global phase

All public entrypoints normalize to this form first.

### `SchurState`

Defined in [src/terket/engine.py](/c:/Users/ethan/github/bee/TerKet/src/terket/engine.py). This is the compiled state for one circuit and one input bitstring. It stores:

- affine output map `eps, eps0`
- symbolic phase kernel `q`
- scalar prefactors and `sqrt(2)` scaling
- deferred arbitrary-angle phase terms
- caches for echelon solves, affine compositions, and structural plans

This is the object reused across many output queries.

### `PhaseFunction`

Defined in [src/terket/cubic_arithmetic.py](/c:/Users/ethan/github/bee/TerKet/src/terket/cubic_arithmetic.py). This is the symbolic phase polynomial used by the reducer:

- `q0`: constant phase
- `q1`: unary coefficients
- `q2`: pair couplings
- `q3`: cubic couplings
- `level`: dyadic precision level

The amplitude problem becomes an exact sum of `omega^q(x)` over binary assignments.

### `ScaledAmplitude` and `ScaledProbability`

Defined in [src/terket/engine.py](/c:/Users/ethan/github/bee/TerKet/src/terket/engine.py). Results are stored as:

`mantissa * 2 ** (half_pow2_exp / 2)`

This preserves tiny exact values that would otherwise underflow in plain `complex`.

Probability is not a separate solver path. TerKet computes the exact scaled amplitude and squares it exactly.

## Pipeline

## 1. Input Normalization

Normalization lives mostly in [src/terket/circuits.py](/c:/Users/ethan/github/bee/TerKet/src/terket/circuits.py) and [src/terket/circuit_spec.py](/c:/Users/ethan/github/bee/TerKet/src/terket/circuit_spec.py).

This stage:

- accepts TerKet circuits, Qiskit circuits, and textual circuit inputs
- lowers imported circuits into TerKet gate vocabulary
- preserves dyadic `rz` phases exactly when possible
- leaves arbitrary-angle phases explicit when exact dyadic lowering is not possible
- rewrites gate streams to remove local cancellations before solver work starts

The result is one small gate language and one exact phase convention for the engine.

## 2. Schur-State Compilation

`build_state(...)` walks the normalized gate list and updates a `SchurState`.

Each gate does two things:

- changes the affine output map from path variables to measured qubit values
- contributes exact phase data to the symbolic kernel

At the end of compilation:

- output constraints live in `eps, eps0`
- phase data lives in `q`
- normalization factors live in `scalar` and `scalar_half_pow2`

This is the translation from circuit semantics to exact finite-field phase algebra.

## 3. Output Restriction

For one requested output bitstring, TerKet solves the output affine system.

Possible outcomes:

- inconsistent output constraints: exact zero amplitude
- consistent output constraints: constrained variables become affine functions of the remaining free variables

This stage uses reusable echelon data, so repeated output queries on one `SchurState` do not rebuild the linear solve.

## 4. Exact Reduction

TerKet then shrinks the restricted phase kernel by exact eliminations.

The reducer alternates between:

- variable classification
- one-step eliminations
- exact no-branch q3-free rewrites
- selective branching when a tiny split is cheaper than carrying difficult structure forward

If the kernel becomes q3-free, TerKet switches to q3-free exact summation. If genuine cubic structure survives, it enters Phase 3 backend selection.

## 5. Q3-Free Exact Summation

When `q3` is empty, TerKet:

- optionally optimizes q3-free structure
- builds a q3-free execution plan
- chooses component backends from pair-graph structure
- evaluates the plan exactly

This is still exact symbolic summation, not approximation.

## 6. Phase-3 Residual Handling

When genuine cubic structure remains, TerKet computes structural diagnostics:

- q3 cover
- min-fill order and width estimate
- q3 hypergraph 2-core and peeled order
- optional small separators

It then chooses the cheapest exact Phase-3 backend from those diagnostics.

## Elimination Catalog

This section lists the elimination families used by the reducer.

## A. Output-Constraint Solve

This is the affine solve performed before phase reduction.

How it works:

- row-reduce the output constraint matrix
- test consistency with the requested output bits
- if consistent, build an affine substitution for constrained variables

Why it matters:

- exact zero outputs die immediately
- repeated output queries reuse the echelon form
- the phase reducer sees fewer live variables

This is not counted as `quad_eliminated` or `constraint_eliminated`; it is the setup for the reduction phase.

## B. Decoupled Constraint Elimination

Code path: `_elim_constraint(..., {"type": "decoupled"})`

When it applies:

- a variable contributes only a trivial character constraint and is otherwise decoupled

How it works:

- remove the variable
- preserve the rest of the kernel unchanged
- contribute a factor of `2`

## C. Zero Constraint Elimination

Code path: `_elim_constraint(..., {"type": "zero"})`

When it applies:

- the character contribution of one variable sums exactly to zero

How it works:

- return `None` for the reduced kernel
- caller interprets that as exact zero amplitude for the whole branch

## D. Parity Constraint Elimination

Code path: `_elim_constraint(..., {"type": "parity", ...})`

When it applies:

- a variable is constrained to equal a parity of partner variables, possibly with a shift

How it works:

- if there is one partner, use the dedicated single-partner fast path
- otherwise compose the affine substitution into the kernel with `_aff_compose_cached(...)`
- contribute a factor of `2`

This is the main exact constraint-substitution rule used by the reducer.

### D1. Single-Partner Parity Fast Path

Code path: `_elim_single_partner_constraint(...)`

When it applies:

- parity constraint has exactly one partner

How it works:

- fix the partner to the required target value
- rebuild the kernel with both variables removed
- avoid paying the full generic affine-compose cost

### D2. Two-Partner q3-Free Specialization

Code path: `_elim_two_partner_constraint_q3_free(...)`

When it applies:

- two-partner parity constraint inside a q3-free situation where a specialized reduction preserves better q3-free structure

How it works:

- collapse the parity relation directly
- keep the resulting pair graph smaller and easier for q3-free backends

## E. Quadratic Elimination

Code path: `_elim_quadratic(...)`

When it applies:

- one variable satisfies the one-variable Gaussian elimination condition used by the reducer

How it works:

- compute the exact one-variable Gauss sum
- update constant phase
- add linear corrections to neighbors
- add pair corrections between neighbors
- remove the variable
- add one power of `sqrt(2)` to scaling bookkeeping

This is the main one-variable exact Gauss-sum rule in the engine.

## F. Quadratic Split Elimination

Code path: `_elim_quadratic_via_split(...)`

When it applies:

- the variable is close to quadratically eliminable, but odd bilinear structure blocks the direct rule

How it works:

- branch on the variable value `0/1`
- fix that variable explicitly in each branch
- recurse exactly on both branches
- add the branch totals

This is exact elimination by tiny explicit branching.

## G. Batch Decoupled-Constraint Elimination

Code path: `_elim_decoupled_constraints_batch(...)`

When it applies:

- many decoupled constraints appear at once

How it works:

- remove them in one batched rewrite instead of rebuilding the kernel one-by-one

This is mostly a performance optimization for large structured states.

## H. Safe q3-Free Parity Substitutions

Code path: `_apply_safe_q3_free_parity_substitutions(...)`

When it applies:

- the kernel is q3-free
- parity substitutions can be made without creating unsupported higher-degree terms

How it works:

- apply exact no-branch substitutions only when they are safe
- keep the result inside the representable q3-free class

This exists because not every exact substitution is safe at higher precision levels.

## I. Exact q3-Free Normal-Form Rewrite

Code path: `_rewrite_q3_free_phase_to_normal_form(...)`

When it applies:

- a q3-free kernel can be rewritten into an equivalent but solver-cheaper form

How it works:

- run exact eliminations that do not branch
- build execution plans for the baseline and rewritten kernels
- keep the rewrite only if TerKet's runtime score improves

This is not cosmetic simplification. It is a solver-facing structural rewrite.

## J. q3-Free Structural Optimization

Code path: `_optimize_q3_free_phase(...)`

When it applies:

- a q3-free kernel may admit an equivalent basis or structure change with a better execution plan

How it works:

- call the bounded phase-structure optimizer
- score the candidate by the q3-free execution-plan cost model
- keep it only if the score improves

## K. Arbitrary-Phase Branch Elimination

Code path: `_build_arbitrary_phase_branch_plan(...)` plus branch solve in `SchurState._amplitude_internal(...)`

When it applies:

- deferred arbitrary-angle phase terms survive output restriction
- they cannot be collapsed into the cheap unary-factor path

How it works:

- express arbitrary-phase dependencies in a small independent branch basis
- enumerate assignments to that basis
- apply the corresponding affine shift to the kernel
- solve each branch exactly and sum with the correct complex phase

## L. Variable Fixing

Code paths: `_fix_variable(...)`, `_fix_variables(...)`

When it applies:

- a backend or elimination needs to condition on explicit variable assignments

How it works:

- treat fixed values as affine shifts
- compose them into the remaining kernel exactly
- cache the restricted kernels

This is not a standalone algebraic rule, but many branching backends depend on it.

## M. Factorization Reduction

Code path: `detect_factorization(...)` plus `_sum_factorized_components_scaled(...)`

When it applies:

- the kernel decomposes into disconnected components

How it works:

- restrict to each component independently
- solve each component exactly
- multiply the totals

This turns one global hard problem into smaller exact subproblems.

## Backend Catalog

TerKet uses "backend" in two layers:

- q3-free execution-plan component backends
- Phase-3 backends for genuine cubic residuals

Both layers are exact.

## Q3-Free Backends

These operate only after genuine cubic terms are gone.

## 1. `constant`

Used when:

- one component has no live variables and contributes a precomputed exact total

How it works:

- return that constant contribution directly

## 2. `product`

Used when:

- all surviving variables are isolated unary terms

How it works:

- exact sum factorizes into a product of one-variable sums
- runtime is linear in variable count

## 3. `forest`

Used when:

- the q2 interaction graph is a forest

How it works:

- perform leaf-to-root transfer messages on each tree
- each subtree contributes a two-state transfer function
- multiply component totals

## 4. `treewidth`

Used when:

- the q2 pair graph has low enough treewidth

How it works:

- choose an elimination order
- convert the kernel to local factors
- eliminate variables along that order
- keep exact intermediate factor tables

The effective width controls the cost.

### Native treewidth support

When available, TerKet may preplan and execute some treewidth factor eliminations in the native extension instead of pure Python. The mathematics is the same; only execution changes.

## 5. `generic`

Used when:

- no cheaper q3-free structural backend applies

How it works:

- build local factor tables
- choose a generic elimination order
- perform exact factor elimination directly

This is the q3-free fallback backend.

## 6. Half-Phase Mediator Reduction

Represented by: `_HalfPhaseMediatorPlan`

Used when:

- some variables act like mediators with favorable half-phase structure

How it works:

- exactly sum the mediator
- collapse its effect onto a smaller boundary factor over neighboring variables
- hand the reduced core to the next backend

This is planner-side preprocessing, not a public top-level backend label.

## 7. Generic q2 Mediator Reduction

Represented by: `_GenericQ2MediatorPlan`

Used when:

- a mediator is not in the strict half-phase regime but still has low-degree q2 structure

How it works:

- eliminate the mediator exactly
- emit a boundary factor on the remaining core

## 8. Half-Phase Cluster Reduction

Represented by: `_HalfPhaseClusterPlan`

Used when:

- a small hard-support cluster can be collapsed exactly onto a small boundary

How it works:

- solve the cluster exactly
- replace it by boundary factors
- continue with a smaller core

## 9. Cutset-Conditioned q3-Free Solve

Represented by: `_Q3FreeCutsetConditioningPlan`

Used when:

- the pair graph is too wide directly
- conditioning on a small cutset makes the remainder cheap

How it works:

- choose cutset variables
- enumerate assignments to the cutset
- update the remainder incrementally
- solve the remainder with `product`, `treewidth`, or `generic`
- sum all branch totals exactly

## 10. Unary Arbitrary-Phase q3-Free Solve

Used when:

- arbitrary-angle side terms remain, but each depends on only one free variable after restriction

How it works:

- convert those unary terms to exact two-entry factor tables
- combine them with the q3-free execution plan
- evaluate with the same q3-free backend family

## Phase-3 Backends

These are used only when genuine cubic structure remains.

## 1. `treewidth_dp`

Used when:

- a low-treewidth elimination order is predicted to beat cover branching

How it works:

- run exact dynamic programming along the chosen order
- width controls the runtime exponent

## 2. `treewidth_dp_peeled`

Used when:

- the cubic hypergraph has peeled especially well and the remaining core is friendlier than a generic cubic residual

How it works:

- same dynamic-programming idea as `treewidth_dp`
- allowed in a somewhat wider regime because peeled cores are easier

## 3. `cubic_contraction_cpu`

Used when:

- the residual cubic core is small and dense enough that direct exact contraction is cheaper than branching

How it works:

- convert the cubic kernel to local factors
- plan a contraction
- execute that contraction exactly on CPU

## 4. `q3_separator`

Used when:

- a small separator splits the cubic structure better than a global cover

How it works:

- branch on separator assignments
- each branch disconnects or becomes easier
- solve each branch exactly and sum the totals

## 5. `q3_cover`

Used when:

- no more specialized Phase-3 backend wins the cost model

How it works:

- choose a vertex cover of the cubic hypergraph
- enumerate assignments to the cover variables
- every branch becomes q3-free
- solve each branch with q3-free exact summation
- sum the branch totals

This is the main fully general cubic fallback.

## Backend Selection Logic

## Q3-Free Selection

For q3-free kernels, TerKet builds a `_Q3FreeExecutionPlan`. It prefers, roughly:

- trivial or product structure
- forest structure
- direct treewidth
- cutset-conditioned remainders
- generic factor elimination

The exact decision depends on:

- pair-graph width
- factor density
- mediator opportunities
- cluster opportunities
- one-shot versus reusable planning mode
- native support availability

## Phase-3 Selection

For genuinely cubic kernels, `_choose_phase3_backend(...)` compares runtime scores for:

- `treewidth_dp_peeled`
- `treewidth_dp`
- `cubic_contraction_cpu`
- `q3_separator`
- `q3_cover`

The score depends on:

- q3 cover size
- min-fill width
- structural obstruction
- peeled-core status
- optional separator availability
- whether tensor-like contraction is allowed

The chosen backend determines `phase3_backend` and the meaning of `cost_model_r`.

## Whole-Pipeline View

A single amplitude query works like this:

1. Normalize the frontend circuit into `CircuitSpec`.
2. Compile circuit and input basis state into `SchurState`.
3. Solve output constraints for the requested output bitstring.
4. If inconsistent, return exact zero.
5. Compose the output restriction into the phase kernel.
6. Apply exact reductions:
   - decoupled, zero, and parity constraints
   - quadratic eliminations
   - special fast paths
   - q3-free no-branch rewrites
   - structural optimization when it improves the runtime score
7. If the kernel factorizes, solve components independently and multiply.
8. If the kernel is q3-free, build and evaluate a q3-free execution plan.
9. If cubic structure remains, choose a Phase-3 backend and solve the residual core exactly.
10. Reapply scalar prefactors and `sqrt(2)` scaling.
11. Return the exact scaled amplitude and solver metadata.

Probability query is the same pipeline through step 10, followed by:

12. square the exact scaled amplitude to form `ScaledProbability`

## Reading Metadata

The most important solver diagnostics are:

- `cubic_obstruction`
  Meaning: how much genuine cubic structure survived exact reduction

- `gauss_obstruction`
  Meaning: broader obstruction to the q3-free exact solver family

- `cost_model_r`
  Meaning: backend-specific runtime exponent proxy for the selected hard-core solver

Typical interpretations:

- low `cubic_obstruction`, low `gauss_obstruction`: exact eliminations did most of the work
- zero `cubic_obstruction`, high `gauss_obstruction`: q3-free but still structurally hard
- high `cost_model_r` with backend `q3_cover`: cover branching dominated
- high `cost_model_r` with backend `treewidth_dp*`: width dominated

## Practical Implications

If many outputs are queried for one circuit:

- build one `SchurState`
- reuse echelon solves and reduction caches
- prefer scaled APIs


