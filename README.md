# TerKet

TerKet is exact strong-simulation toolkit for Clifford+T circuits. Repo combines library code, backend diagnostics, and reproducible benchmark entrypoints in one place.

## Why TerKet

- Exact amplitude and probability queries for Clifford+T workloads
- Qiskit import path for practical circuit ingestion
- Native accelerator support when local toolchain is available
- Backend diagnostics that expose `phase3_backend`, obstruction metrics, and solver cost models
- Unified benchmark CLI for smoke runs, head-to-head studies, and structural probes

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
```

Run tests:

```powershell
pytest
```

If local compiler is available, `pip install -e .` can rebuild native accelerator from `src/terket/_schur_native.c`. Otherwise bundled platform artifacts are used when they match active interpreter. Set `TERKET_DISABLE_NATIVE=1` to force pure-Python path.

## Quick Example

```python
from terket import compute_circuit_amplitude, make_circuit

circuit = make_circuit(1, [("h", 0)])
amplitude, info = compute_circuit_amplitude(circuit, [0], [0], as_complex=True)

print(complex(amplitude))
print(info["phase3_backend"])
```

## Benchmarks

Unified benchmark entrypoint:

```powershell
python benchmarks/run_benchmarks.py head-to-head
python benchmarks/run_benchmarks.py structured-showcase
python benchmarks/run_benchmarks.py probability-native-rcs --case smoke --cycles 2
```

Benchmark families and usage notes live in [benchmarks/README.md](benchmarks/README.md).

## Notebook

Interactive repo walkthrough lives in [notebooks/terket_demo.ipynb](notebooks/terket_demo.ipynb). Notebook covers:

- basic amplitude queries
- Qiskit import flow
- backend detection and metadata
- unified benchmark commands

## Repo Layout

- `src/terket/`: package source, case builders, solver logic, native accelerator sources
- `benchmarks/`: benchmark families plus unified benchmark CLI
- `notebooks/`: demo notebook and notebook notes
- `tests/`: smoke and regression coverage
- `results/`: generated CSV outputs, created on demand
- `tools/`: profiling and local investigation scripts
