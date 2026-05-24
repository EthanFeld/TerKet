# Tools

Tools are repo-maintenance scripts, not TerKet public API.

| Tool | Purpose | Runtime Scale | Optional Deps |
| --- | --- | --- | --- |
| `report_file_sizes.py` | Report code-file line counts, enforce 1000-line cap allowlist, and print current 300+ line offenders. | Seconds. | None. |
| `compare_bloat_baseline.py` | Compare benchmark CSVs for backend/output/runtime regressions. | Seconds. | None beyond benchmark CSV deps. |
| `check_native_build_matrix.py` | Verify native-enabled and `TERKET_DISABLE_NATIVE=1` build modes. | Seconds to minutes; compiler-dependent. | C compiler, setuptools. |
| `profile_rcs_old_vs_new.py` | Profile RCS import/solver behavior during refactors. | Minutes on larger cases. | Qiskit, profiling tooling. |
| `quantinuum_challenge_terket_graphs.py` | Local Quantinuum challenge graph analysis/probe. | Case-dependent. | Challenge inputs, Qiskit. |
| `quantinuum_compare_terket_public.py` | Compare TerKet public challenge behavior. | Case-dependent. | Challenge inputs, Qiskit. |
| `try_cirq_to_qasm.py` | Temporary Cirq-to-QASM import probe. | Seconds to minutes. | Cirq, Qiskit. |

Ignored local probes may exist in `tools/`; keep new durable benchmark logic under `benchmarks/` or `src/terket/benchmarking/`.
