# TerKet

TerKet is exact strong-simulation toolkit for Clifford+T-style quantum circuits. It ingests normalized gate lists, Qiskit circuits, or QASM-like inputs, rewrites them into Schur-state phase functions, and answers exact amplitude queries with explicit solver metadata.

Repo bundles three things:

- public Python API for exact amplitude queries
- native and pure-Python solver backends
- reproducible benchmark and profiling entrypoints

## What It Does

- Computes exact amplitudes without collapsing tiny nonzero values to `0j`
- Imports practical circuits from Qiskit
- Reports reduction diagnostics such as `cubic_obstruction`, `gauss_obstruction`, `cost_model_r`, and `phase3_backend`
- Includes benchmark families for head-to-head runtime studies and structural probes

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

If you want notebook and benchmark dependencies separated from base install:

```powershell
pip install -r requirements.txt
```

Run test suite:

```powershell
pytest
```

## Native Accelerator

TerKet can use bundled native Schur helpers when platform/interpreter match available artifact. If local compiler toolchain is available, `pip install -e .` can rebuild native extension from [src/terket/_schur_native.c](TerKet/src/terket/_schur_native.c). Set `TERKET_DISABLE_NATIVE=1` to force pure-Python execution.

Native code accelerates selected q3-free exact summation kernels. Core algorithm and public behavior stay same with or without it.

## Quick Start

```python
from terket import (
    compute_circuit_amplitude,
    make_circuit,
)

circuit = make_circuit(2, [("h", 0), ("cnot", 0, 1)])

amplitude, amp_info = compute_circuit_amplitude(
    circuit,
    [0, 0],
    [0, 0],
    as_complex=True,
)

print(amplitude)
print(amp_info["phase3_backend"])
```

Default amplitude return is `ScaledAmplitude`, not `complex`, so tiny exact values do not silently underflow. If you need a probability, square the returned amplitude yourself.

## Public API

Main entrypoints:

- `compute_circuit_amplitude(...)`
- `compute_circuit_amplitude_scaled(...)`
- `analyze_circuit(...)`
- `analyze_amplitudes(...)`
- `make_circuit(...)`
- `normalize_circuit(...)`
- `from_qiskit(...)`

Input circuits may be:

- TerKet `CircuitSpec`
- supported Qiskit circuit objects
- textual circuit/QASM-style input accepted by normalization layer

## Solver Metadata

Every exact query returns metadata describing work active solver paid for. Most useful fields:

- `cubic_obstruction`: residual genuine cubic structure after exact eliminations
- `gauss_obstruction`: broader obstruction to q3-free quadratic-style contraction
- `cost_model_r`: runtime exponent proxy for chosen Phase-3 backend
- `phase3_backend`: backend actually used on residual hard core
- `is_zero`: exact zero detection result

## Benchmarks

Unified entrypoint:

```powershell
python benchmarks/run_benchmarks.py head-to-head --suite smoke --repeats 1
python benchmarks/run_benchmarks.py structured-showcase --suite smoke
python benchmarks/run_benchmarks.py depth-scaling --depths 1 2 4 8
python benchmarks/run_benchmarks.py amplitude-post-elimination-tensor-rcs
python benchmarks/run_benchmarks.py rcs-import-strategy-probe
```

More benchmark detail lives in [benchmarks/README.md](TerKet/benchmarks/README.md).

## Notebook

Interactive walkthrough: [notebooks/terket_demo.ipynb](TerKet/notebooks/terket_demo.ipynb)

## Design

High-level design and solver pipeline: [docs/design.md](TerKet/docs/design.md)

## Repo Layout

- `src/terket/`: package source, solver engine, case builders, native helpers
- `benchmarks/`: benchmark families and benchmark CLI
- `tests/`: regression and smoke coverage
- `notebooks/`: interactive walkthrough material
- `tools/`: local profiling and investigation scripts
- `results/`: generated CSV output, created on demand
