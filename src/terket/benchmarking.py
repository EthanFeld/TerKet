"""Benchmark helpers shared by the TerKet CLI entrypoints."""

from __future__ import annotations

import cmath
import csv
from dataclasses import asdict, dataclass, is_dataclass
import gc
import importlib.metadata
import math
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Callable, Sequence

import psutil

from .circuits import (
    _circuit_global_phase_radians,
    _dyadic_phase_to_angle,
    bits_to_little_endian_string,
    make_circuit,
    normalize_circuit,
)
from .schur_engine import _get_quimb_tensor_module, _quimb_import_reason, compute_circuit_amplitude


MB = 1024 * 1024


@dataclass
class HeadToHeadRow:
    case: str
    family: str
    size: int
    n_qubits: int
    gate_count: int
    t_count: int
    cubic_obstruction: int
    gauss_obstruction: int
    cost_model_r: int
    terket_phase3_backend: str
    obstruction_fraction: float
    repeats: int
    quimb_optimize: str
    terket_wall_time_s: float
    terket_peak_rss_mb: float
    quimb_wall_time_s: float
    quimb_peak_rss_mb: float
    terket_real: float
    terket_imag: float
    quimb_real: float
    quimb_imag: float
    abs_error: float
    relative_error: float
    quimb_over_terket_time_ratio: float
    python_version: str
    numpy_version: str
    qiskit_version: str
    quimb_version: str
    cotengra_version: str


@dataclass
class StructuredShowcaseRow:
    case: str
    family: str
    size: int
    n_qubits: int
    gate_count: int
    t_count: int
    cubic_obstruction: int
    cost_model_r: int
    terket_phase3_backend: str
    wall_time_s: float
    target_bits: str
    target_amplitude_real: float
    target_amplitude_imag: float
    target_log2_abs: float
    target_scaled_mantissa_real: float
    target_scaled_mantissa_imag: float
    target_scaled_half_pow2_exp: float
    wrong_bits: str
    wrong_abs: float
    python_version: str
    numpy_version: str
    qiskit_version: str


def _memory_monitor(stop_event: threading.Event, peak_holder: list[int], process: psutil.Process):
    peak = peak_holder[0]
    while not stop_event.is_set():
        peak = max(peak, process.memory_info().rss)
        time.sleep(0.005)
    peak_holder[0] = max(peak, process.memory_info().rss)


def measure_callable(func: Callable[[], object], repeats: int) -> tuple[float, float, object]:
    process = psutil.Process()
    wall_times: list[float] = []
    peaks: list[float] = []
    value = None

    for _ in range(repeats):
        gc.collect()
        base_rss = process.memory_info().rss
        peak_holder = [base_rss]
        stop_event = threading.Event()
        monitor = threading.Thread(
            target=_memory_monitor,
            args=(stop_event, peak_holder, process),
            daemon=True,
        )
        monitor.start()
        start = time.perf_counter()
        value = func()
        elapsed = time.perf_counter() - start
        stop_event.set()
        monitor.join()

        wall_times.append(elapsed)
        peaks.append(max(0, peak_holder[0] - base_rss) / MB)

    return statistics.median(wall_times), max(peaks), value


def count_t_gates(circuit) -> int:
    spec = normalize_circuit(circuit)
    return sum(1 for gate in spec.gates if gate[0] in {"t", "tdg"})


def quimb_circuit_from_circuit(circuit, input_bits: Sequence[int]):
    qtn = _get_quimb_tensor_module()
    if qtn is None:
        raise RuntimeError(_quimb_import_reason())

    spec = normalize_circuit(circuit)
    if len(input_bits) != spec.n_qubits:
        raise ValueError("Input bit string must match the circuit width.")

    quimb_circuit = qtn.Circuit(spec.n_qubits)
    for qubit, bit in enumerate(input_bits):
        if bit:
            quimb_circuit.x(qubit)

    for gate in spec.gates:
        name, *qubits = gate
        if name == "h":
            quimb_circuit.h(qubits[0])
        elif name == "x":
            quimb_circuit.x(qubits[0])
        elif name == "sx":
            quimb_circuit.sx(qubits[0])
        elif name == "t":
            quimb_circuit.t(qubits[0])
        elif name == "tdg":
            quimb_circuit.tdg(qubits[0])
        elif name == "s":
            quimb_circuit.s(qubits[0])
        elif name == "sdg":
            quimb_circuit.sdg(qubits[0])
        elif name == "z":
            quimb_circuit.z(qubits[0])
        elif name == "cnot":
            quimb_circuit.cx(*qubits)
        elif name == "cz":
            quimb_circuit.cz(*qubits)
        elif name == "rz_pi_16":
            quimb_circuit.phase(_dyadic_phase_to_angle(1, 5), qubits[0])
        elif name == "rz_pi_16_dg":
            quimb_circuit.phase(_dyadic_phase_to_angle(31, 5), qubits[0])
        elif name == "rz_pi_32":
            quimb_circuit.phase(_dyadic_phase_to_angle(1, 6), qubits[0])
        elif name == "rz_pi_32_dg":
            quimb_circuit.phase(_dyadic_phase_to_angle(63, 6), qubits[0])
        elif name == "rz_dyadic":
            quimb_circuit.phase(_dyadic_phase_to_angle(qubits[1], qubits[2]), qubits[0])
        elif name == "rz_arbitrary":
            quimb_circuit.phase(float(qubits[1]), qubits[0])
        else:
            raise ValueError(f"Unsupported quimb gate {name!r}.")

    return quimb_circuit


def quimb_amplitude(circuit, input_bits: Sequence[int], output_bits: Sequence[int], *, optimize: str) -> complex:
    spec = normalize_circuit(circuit)
    if len(input_bits) != spec.n_qubits or len(output_bits) != spec.n_qubits:
        raise ValueError("Input and output bit strings must match the circuit width.")

    quimb_circuit = quimb_circuit_from_circuit(spec, input_bits)
    amplitude = complex(quimb_circuit.amplitude(bits_to_little_endian_string(output_bits), optimize=optimize))
    return amplitude * cmath.exp(1j * _circuit_global_phase_radians(spec))


def warm_up_terket() -> None:
    circuit = make_circuit(1, [("h", 0)])
    compute_circuit_amplitude(circuit, [0], [0], as_complex=True)


def warm_up_quimb(optimize: str) -> None:
    qtn = _get_quimb_tensor_module()
    if qtn is None:
        raise RuntimeError(_quimb_import_reason())

    circuit = qtn.Circuit(1)
    circuit.h(0)
    circuit.amplitude("0", optimize=optimize)


def runtime_versions() -> dict[str, str]:
    def version(dist_name: str) -> str:
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy_version": version("numpy"),
        "qiskit_version": version("qiskit"),
        "quimb_version": version("quimb"),
        "cotengra_version": version("cotengra"),
    }


def scaled_amplitude_fields(amplitude) -> tuple[float, float, float, float, float, float]:
    try:
        amplitude_complex = amplitude.to_complex()
    except OverflowError:
        amplitude_real = math.nan
        amplitude_imag = math.nan
    else:
        amplitude_real = float(amplitude_complex.real)
        amplitude_imag = float(amplitude_complex.imag)

    return (
        amplitude_real,
        amplitude_imag,
        float(amplitude.log2_abs()),
        float(amplitude.mantissa.real),
        float(amplitude.mantissa.imag),
        float(amplitude.half_pow2_exp),
    )


def write_rows(rows: Sequence[object], csv_path: Path) -> None:
    if not rows:
        raise ValueError("At least one row is required to write a CSV.")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    first = rows[0]
    if is_dataclass(first):
        fieldnames = list(asdict(first).keys())
        records = [asdict(row) for row in rows]
    else:
        fieldnames = list(first.keys())
        records = list(rows)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
