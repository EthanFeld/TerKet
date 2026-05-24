# TerKet Backend Coalescing Principles

Date: 2026-05-21

Scope: Step 5 and q3-free Step 6 guardrails.

## Rules Now In Force

1. Keep exact semantics first. No coalescing patch may change exact amplitudes.
2. Keep one generic exact evaluator per backend family, plus small adapters.
3. Keep performance specializations only after benchmark proof:
   >= 10% median win or avoids > 2x memory.
4. Approximate paths stay behind `SolverConfig.allow_approximate`.
5. Approximate results need explicit approximation metadata.
6. Backend selection must have:
   - candidate builder
   - viability predicate
   - score/comparator
   - stable compatibility label
7. One-shot/reusable variants must converge on shared plan/evaluator objects.
8. Factor-table math should live in one evaluator path, reused by adapters.
9. Native exact adapters win over Python implementations whenever a native plan
   can be built for the same exact treewidth problem.

## Q3-Free Shape

Keep:
- `constant`
- `product`
- `forest`
- `treewidth`
- `generic`
- `cutset`

Treat as preprocessing/adapter:
- binary-phase quadratic plan
- half-phase unary expansion
- mediator/generic mediator
- cluster reduction
- dense Schur complement
- native q3-free treewidth

Native preference:
- direct q3-free treewidth component plan uses native plan first
- dense q3-free component tries native treewidth before dense generic fallback
- restricted q3-free component tries native treewidth before reusable cutset
- cutset remaining treewidth rebuilds native plan after order refinement

Deferred until benchmark gates:
- remove dense direct variants
- collapse one-shot/reusable cutset search modes
- decide raw-constraint reusable path
- decide mediator/cluster keep/coalesce/delete
