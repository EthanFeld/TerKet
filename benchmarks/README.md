# Benchmarks

Use one maintained entrypoint:

```powershell
python benchmarks/run_benchmarks.py curated [args]
```

The curated run keeps two buckets in one script:

- `showcase`: impressive TerKet-only structured cases
- `fair`: shared head-to-head TerKet vs quimb cases, defaulting to larger MQT Bench circuits beyond statevector scale when they complete cleanly
- `terket_frontier`: larger cases kept separate when TerKet works but quimb head-to-head is not the right comparison

Default command:

```powershell
python benchmarks/run_benchmarks.py curated --repeats 1
```

Useful overrides:

```powershell
python benchmarks/run_benchmarks.py curated --showcase-case mm_hidden_shift192 mm_hidden_shift384
python benchmarks/run_benchmarks.py curated --fair-case mqt:ghz:40 mqt:graphstate:40 mqt:qftentangled:24
python benchmarks/run_benchmarks.py curated --terket-case mqt:qaoa:24 mqt:vqe_two_local:18
python benchmarks/run_benchmarks.py curated --csv results/curated_benchmark.csv
```

Notes:

- CSV files default to `results/`
- Exploratory one-off benchmark scripts live under `benchmarks/targeted/`
- `benchmarks/targeted/` is gitignored so targeted experiments do not clutter the maintained benchmark surface
