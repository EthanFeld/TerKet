# Benchmarks

Use one maintained entrypoint:

```powershell
python benchmarks/run_benchmarks.py <benchmark> [args]
```

## Maintained Benchmark Matrix

| Benchmark | Purpose | Expected Runtime Scale | Required Optional Deps |
| --- | --- | --- | --- |
| `curated` | Mixed default suite: TerKet showcase, fair TerKet-vs-quimb cases, and TerKet frontier cases. | Seconds to minutes by selected cases/repeats. | Qiskit, quimb, cotengra, psutil, mqt-bench for MQT cases. |
| `head-to-head` | Direct TerKet-vs-quimb comparison on fixed exact-strong cases. | Seconds for `smoke`; minutes for `expanded`. | Qiskit, quimb, cotengra, psutil. |
| `structured-showcase` | Structured TerKet-only hidden-shift showcase cases with solver diagnostics. | Seconds for `smoke`; seconds to minutes for large/xlarge. | Qiskit only for version reporting; core cases are TerKet-native. |
| `depth-scaling` | Depth sweeps for representative TerKet-vs-quimb cases. | Minutes. | Qiskit, quimb, cotengra, psutil. |
| `amplitude-post-elimination-tensor-rcs` | RCS post-elimination residual planning probe; exact residual eval only, no removed tensor contraction path. | Seconds for `smoke`; minutes+ for Sycamore-like cases. | Qiskit, rustworkx, networkx, cotengra. |
| `rcs-import-strategy-probe` | Compare RCS import strategies and structural impact. | Minutes on Sycamore-like cases. | Qiskit, rustworkx, networkx. |

## Common Commands

```powershell
python benchmarks/run_benchmarks.py curated --repeats 1
python benchmarks/run_benchmarks.py head-to-head --suite smoke --repeats 1
python benchmarks/run_benchmarks.py structured-showcase --suite smoke
python benchmarks/run_benchmarks.py curated --csv results/curated_benchmark.csv
```

## Benchmark Code Ownership

- CLI runners live under `benchmarks/`.
- Shared case builders live under `src/terket/benchmarking/`.
- Local maintenance scripts live under `tools/`.
- Generated CSV/log/profile output goes under `results/` or `benchmarks/results/` and stays ignored.

## Curated Buckets

- `showcase`: TerKet-only structured cases.
- `fair`: shared TerKet-vs-quimb cases.
- `terket_frontier`: larger cases where TerKet works but quimb is not the right comparison.
