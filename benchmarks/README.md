# Benchmarks

TerKet benchmark workflows run through one entrypoint:

```powershell
python benchmarks/run_benchmarks.py <family> [family args]
```

Installed environments can also use:

```powershell
terket-bench <family> [family args]
```

## Families

- `head-to-head`: compare TerKet amplitude runtime against quimb on named case suites
- `structured-showcase`: large hidden-shift showcase focused on scaled amplitudes
- `depth-scaling`: sweep circuit depth on controlled families
- `probability-native-rcs`: benchmark probability-native path on grid-style random circuit sampling cases
- `amplitude-post-elimination-tensor-rcs`: inspect tensor viability after TerKet elimination
- `rcs-import-strategy-probe`: compare import and transpilation strategies structurally

## Common Commands

```powershell
python benchmarks/run_benchmarks.py head-to-head --suite smoke --repeats 1
python benchmarks/run_benchmarks.py structured-showcase --suite smoke
python benchmarks/run_benchmarks.py depth-scaling --depths 1 2 4 8
python benchmarks/run_benchmarks.py probability-native-rcs --case smoke --cycles 2
```

## Output

- CSV files default to `results/`
- Scripts print brief row summaries to stdout
- Head-to-head outputs include error checks and runtime ratios
- Structural probes expose obstruction counts, widths, and chosen backend families
