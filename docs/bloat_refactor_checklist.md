# Bloat Refactor Checklist

Use this checklist for any PR or local patch whose goal is code-size reduction,
module extraction, or backend ownership cleanup.

## Local Pre-Merge Guard

Run these before asking for review:

```powershell
python tools/report_file_sizes.py --check
pytest -q tests/test_repo_bloat_guards.py
python -m compileall -q src tests benchmarks scripts tools
pytest -q
```

What this does:

1. `report_file_sizes.py --check`
   - fails if a new code file crosses 1000 lines
   - fails if allowlist still names a file already pulled below 1000
   - reports current 300+ line files for continued cleanup
2. `tests/test_repo_bloat_guards.py`
   - freezes current direct-import surface from `terket._engine_impl`
   - freezes current direct-import surface from `terket.circuit_spec`
3. `compileall`
   - catches syntax and import-time parse errors across repo code
4. full `pytest`
   - required for all refactor waves

## Backend-Sensitive Moves

If patch moves q3-free, Phase-3, reduction, state-build, or public amplitude code,
compare benchmark CSVs against baseline in `results/bloat_baseline_20260521/`.

Examples:

```powershell
python tools/compare_bloat_baseline.py results/bloat_baseline_20260521/head_to_head_smoke.csv <candidate_head_to_head.csv>
python tools/compare_bloat_baseline.py results/bloat_baseline_20260521/structured_showcase_smoke.csv <candidate_structured_showcase.csv>
python tools/compare_bloat_baseline.py results/bloat_baseline_20260521/curated_smoke.csv <candidate_curated.csv>
```

## Native-Code Moves

If patch changes `_schur_native*.c` or `_schur_native_internal.h`, also run:

```powershell
python tools/check_native_build_matrix.py
pytest -q tests/test_native_bloat_guards.py
```

This keeps native-enabled and `TERKET_DISABLE_NATIVE=1` behavior aligned while
native files are being split.
