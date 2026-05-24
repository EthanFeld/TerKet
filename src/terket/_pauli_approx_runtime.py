"""Extracted approximate Pauli runtime helpers."""

from __future__ import annotations

import cmath
import importlib
import math
from typing import Sequence

import numpy as np

from ._engine_runtime_core import _configure_extracted_module
from .spec import CircuitSpec, Gate

_LOCAL_NAMES = {
    '_native_mps_one_qubit_matrix',
    '_native_mps_rx_matrix',
    '_native_mps_rzz_matrix',
    '_native_mps_apply_pauli_expbox',
    '_native_mps_approx_state',
    '_native_mps_apply_gate',
    '_native_mps_approx_mirror_fidelity',
    '_native_mps_approx_pauli_expectations'
}
_LOCAL_IMPLS = {}
_configure_extracted_module(globals(), local_names=_LOCAL_NAMES, local_impls=_LOCAL_IMPLS)


def _refresh_engine_bindings() -> None:
    _sync_from_engine(importlib.import_module("terket._engine_impl"))


def _native_mps_one_qubit_matrix(name: str, gate: Gate) -> np.ndarray | None:
    if name == "h":
        return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / math.sqrt(2.0)
    if name == "x":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    if name == "z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    if name == "s":
        return np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=np.complex128)
    if name == "sdg":
        return np.array([[1.0, 0.0], [0.0, -1.0j]], dtype=np.complex128)
    if name == "t":
        return np.array([[1.0, 0.0], [0.0, cmath.exp(0.25j * math.pi)]], dtype=np.complex128)
    if name == "tdg":
        return np.array([[1.0, 0.0], [0.0, cmath.exp(-0.25j * math.pi)]], dtype=np.complex128)
    if name == "sx":
        return 0.5 * np.array(
            [[1.0 + 1.0j, 1.0 - 1.0j], [1.0 - 1.0j, 1.0 + 1.0j]],
            dtype=np.complex128,
        )
    if name == "sxdg":
        return 0.5 * np.array(
            [[1.0 - 1.0j, 1.0 + 1.0j], [1.0 + 1.0j, 1.0 - 1.0j]],
            dtype=np.complex128,
        )
    if name == "rz_arbitrary":
        return np.array([[1.0, 0.0], [0.0, cmath.exp(1j * float(gate[2]))]], dtype=np.complex128)
    if name == "rz_dyadic":
        angle = 2.0 * math.pi * (int(gate[2]) % (1 << int(gate[3]))) / float(1 << int(gate[3]))
        return np.array([[1.0, 0.0], [0.0, cmath.exp(1j * angle)]], dtype=np.complex128)
    return None


_NATIVE_MPS_CNOT = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)
_NATIVE_MPS_CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128).reshape(2, 2, 2, 2)
_NATIVE_MPS_SWAP = np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)


class _NativeApproxMPS:
    def __init__(self, n_qubits: int, input_bits: Sequence[int], max_bond: int) -> None:
        self.max_bond = int(max_bond)
        self.position_to_qubit = list(range(n_qubits))
        self.qubit_to_position = list(range(n_qubits))
        self.tensors: list[np.ndarray] = []
        for bit in input_bits:
            tensor = np.zeros((1, 2, 1), dtype=np.complex128)
            tensor[0, int(bit) & 1, 0] = 1.0
            self.tensors.append(tensor)

    def apply_one(self, qubit: int, matrix: np.ndarray) -> None:
        pos = self.qubit_to_position[int(qubit)]
        self.tensors[pos] = np.einsum("ab,lbr->lar", matrix, self.tensors[pos], optimize=True)

    def apply_adjacent_two(self, pos: int, matrix: np.ndarray) -> None:
        left = self.tensors[pos]
        right = self.tensors[pos + 1]
        dl = left.shape[0]
        dr = right.shape[2]
        theta = np.einsum("lar,rbs->labs", left, right, optimize=True)
        theta = np.einsum("abij,lijk->labk", matrix, theta, optimize=True)
        flat = theta.reshape(dl * 2, 2 * dr)
        u, singular, vh = np.linalg.svd(flat, full_matrices=False)
        keep = min(self.max_bond, len(singular))
        while keep > 1 and singular[keep - 1] <= 1e-12:
            keep -= 1
        u = u[:, :keep]
        singular = singular[:keep]
        vh = vh[:keep, :]
        norm = float(np.linalg.norm(singular))
        if norm > 0.0:
            singular = singular / norm
        self.tensors[pos] = u.reshape(dl, 2, keep)
        self.tensors[pos + 1] = (singular[:, None] * vh).reshape(keep, 2, dr)

    def swap_positions(self, pos: int) -> None:
        self.apply_adjacent_two(pos, _NATIVE_MPS_SWAP)
        left_qubit = self.position_to_qubit[pos]
        right_qubit = self.position_to_qubit[pos + 1]
        self.position_to_qubit[pos], self.position_to_qubit[pos + 1] = right_qubit, left_qubit
        self.qubit_to_position[left_qubit], self.qubit_to_position[right_qubit] = pos + 1, pos

    def apply_two(self, left_qubit: int, right_qubit: int, matrix: np.ndarray) -> None:
        left_pos = self.qubit_to_position[int(left_qubit)]
        right_pos = self.qubit_to_position[int(right_qubit)]
        routed_matrix = matrix
        if left_pos > right_pos:
            left_pos, right_pos = right_pos, left_pos
            routed_matrix = np.transpose(matrix, (1, 0, 3, 2))
        for pos in range(right_pos - 1, left_pos, -1):
            self.swap_positions(pos)
        self.apply_adjacent_two(left_pos, routed_matrix)
        for pos in range(left_pos + 1, right_pos):
            self.swap_positions(pos)

    def expectation(self, observable: str) -> complex:
        pauli = {
            "I": np.eye(2, dtype=np.complex128),
            "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
            "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
            "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
        }
        env = np.array([[1.0 + 0.0j]], dtype=np.complex128)
        for pos, tensor in enumerate(self.tensors):
            op = pauli[observable[self.position_to_qubit[pos]]]
            env = np.einsum("ab,ais,ij,bjt->st", env, tensor.conj(), op, tensor, optimize=True)
        return complex(env[0, 0])

    def norm(self) -> complex:
        return self.expectation("I" * len(self.tensors))

    def amplitude(self, bits: Sequence[int]) -> complex:
        env = np.array([1.0 + 0.0j], dtype=np.complex128)
        for pos, tensor in enumerate(self.tensors):
            bit = int(bits[self.position_to_qubit[pos]]) & 1
            env = np.einsum("l,lr->r", env, tensor[:, bit, :], optimize=True)
        return complex(env[0])


def _native_mps_rx_matrix(angle: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(0.5 * angle), -1.0j * math.sin(0.5 * angle)],
            [-1.0j * math.sin(0.5 * angle), math.cos(0.5 * angle)],
        ],
        dtype=np.complex128,
    )


def _native_mps_rzz_matrix(angle: float) -> np.ndarray:
    return np.diag([
        cmath.exp(-0.5j * angle),
        cmath.exp(0.5j * angle),
        cmath.exp(0.5j * angle),
        cmath.exp(-0.5j * angle),
    ]).astype(np.complex128).reshape(2, 2, 2, 2)


def _native_mps_apply_pauli_expbox(
    mps: _NativeApproxMPS,
    paulis: Sequence[str],
    qubits: Sequence[int],
    angle: float,
) -> None:
    active: list[tuple[str, int]] = []
    for pauli, qubit in zip(paulis, qubits):
        pauli_char = str(pauli).upper()
        if pauli_char != "I":
            active.append((pauli_char, int(qubit)))
    if not active:
        return

    for pauli_char, qubit in active:
        if pauli_char == "X":
            mps.apply_one(qubit, _native_mps_one_qubit_matrix("h", ("h", qubit)))
        elif pauli_char == "Y":
            mps.apply_one(qubit, _native_mps_one_qubit_matrix("sdg", ("sdg", qubit)))
            mps.apply_one(qubit, _native_mps_one_qubit_matrix("h", ("h", qubit)))

    ordered_qubits = [
        qubit
        for _pauli, qubit in sorted(active, key=lambda item: mps.qubit_to_position[item[1]])
    ]
    for left, right in zip(ordered_qubits, ordered_qubits[1:]):
        mps.apply_two(left, right, _NATIVE_MPS_CNOT)
    target = ordered_qubits[-1]
    mps.apply_one(target, _native_mps_one_qubit_matrix("rz_arbitrary", ("rz_arbitrary", target, angle)))
    for left, right in reversed(tuple(zip(ordered_qubits, ordered_qubits[1:]))):
        mps.apply_two(left, right, _NATIVE_MPS_CNOT)

    for pauli_char, qubit in reversed(active):
        if pauli_char == "X":
            mps.apply_one(qubit, _native_mps_one_qubit_matrix("h", ("h", qubit)))
        elif pauli_char == "Y":
            mps.apply_one(qubit, _native_mps_one_qubit_matrix("h", ("h", qubit)))
            mps.apply_one(qubit, _native_mps_one_qubit_matrix("s", ("s", qubit)))


def _native_mps_approx_state(
    spec: CircuitSpec,
    input_bits: Sequence[int],
    *,
    max_bond: int | None = None,
    use_rotation_macros: bool = True,
) -> _NativeApproxMPS | None:
    mps = _NativeApproxMPS(spec.n_qubits, input_bits, _native_mps_approx_bond() if max_bond is None else max_bond)
    gate_idx = 0
    while gate_idx < len(spec.gates):
        gate = spec.gates[gate_idx]
        name = str(gate[0])
        if (
            use_rotation_macros
            and
            gate_idx + 2 < len(spec.gates)
            and name == "h"
            and spec.gates[gate_idx + 1][0] == "rz_arbitrary"
            and spec.gates[gate_idx + 2] == gate
            and int(spec.gates[gate_idx + 1][1]) == int(gate[1])
        ):
            mps.apply_one(int(gate[1]), _native_mps_rx_matrix(float(spec.gates[gate_idx + 1][2])))
            gate_idx += 3
            continue
        if (
            use_rotation_macros
            and
            gate_idx + 2 < len(spec.gates)
            and name == "cnot"
            and spec.gates[gate_idx + 1][0] == "rz_arbitrary"
            and spec.gates[gate_idx + 2] == gate
            and int(spec.gates[gate_idx + 1][1]) == int(gate[2])
        ):
            mps.apply_two(
                int(gate[1]),
                int(gate[2]),
                _native_mps_rzz_matrix(float(spec.gates[gate_idx + 1][2])),
            )
            gate_idx += 3
            continue
        one_qubit = _native_mps_one_qubit_matrix(name, gate)
        if one_qubit is not None:
            mps.apply_one(int(gate[1]), one_qubit)
            gate_idx += 1
            continue
        if name == "cnot":
            mps.apply_two(int(gate[1]), int(gate[2]), _NATIVE_MPS_CNOT)
            gate_idx += 1
            continue
        if name == "cz":
            mps.apply_two(int(gate[1]), int(gate[2]), _NATIVE_MPS_CZ)
            gate_idx += 1
            continue
        if name == "rzz_dyadic":
            angle = 2.0 * math.pi * (int(gate[3]) % (1 << int(gate[4]))) / float(1 << int(gate[4]))
            mps.apply_two(int(gate[1]), int(gate[2]), _native_mps_rzz_matrix(angle))
            gate_idx += 1
            continue
        if name == "pauli_expbox":
            _native_mps_apply_pauli_expbox(mps, gate[1], gate[2], float(gate[3]))
            gate_idx += 1
            continue
        return None
    return mps


def _native_mps_apply_gate(mps: _NativeApproxMPS, gate: Gate) -> bool:
    name = str(gate[0])
    one_qubit = _native_mps_one_qubit_matrix(name, gate)
    if one_qubit is not None:
        mps.apply_one(int(gate[1]), one_qubit)
        return True
    if name == "cnot":
        mps.apply_two(int(gate[1]), int(gate[2]), _NATIVE_MPS_CNOT)
        return True
    if name == "cz":
        mps.apply_two(int(gate[1]), int(gate[2]), _NATIVE_MPS_CZ)
        return True
    if name == "rzz_dyadic":
        angle = 2.0 * math.pi * (int(gate[3]) % (1 << int(gate[4]))) / float(1 << int(gate[4]))
        mps.apply_two(int(gate[1]), int(gate[2]), _native_mps_rzz_matrix(angle))
        return True
    if name == "pauli_expbox":
        _native_mps_apply_pauli_expbox(mps, gate[1], gate[2], float(gate[3]))
        return True
    return False


def _native_mps_approx_mirror_fidelity(
    spec: CircuitSpec,
    dagger_spec: CircuitSpec,
    input_bits: Sequence[int],
    *,
    max_bond: int | None = None,
) -> float | None:
    mps = _NativeApproxMPS(spec.n_qubits, input_bits, _native_mps_approx_bond() if max_bond is None else max_bond)
    if dagger_spec.n_qubits != spec.n_qubits:
        return None
    for gate in tuple(spec.gates) + tuple(dagger_spec.gates):
        if not _native_mps_apply_gate(mps, gate):
            return None
    amplitude = mps.amplitude(input_bits)
    return float(abs(amplitude) ** 2)


def _native_mps_approx_pauli_expectations(
    spec: CircuitSpec,
    input_bits: Sequence[int],
    observables: Sequence[str],
    *,
    max_bond: int | None = None,
) -> list[complex] | None:
    _refresh_engine_bindings()
    if not observables:
        return []
    mps = _native_mps_approx_state(spec, input_bits, max_bond=max_bond)
    if mps is None:
        return None
    norm = mps.norm()
    if abs(norm) <= 1e-300:
        return None
    return [mps.expectation(observable) / norm for observable in observables]

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
