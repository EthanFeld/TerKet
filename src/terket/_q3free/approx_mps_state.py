"""Numerical state for bounded-bond q3-free boundary-MPS contraction."""

from __future__ import annotations

import cmath
import math

import numpy as np

from ..scaling import ScaledComplex, _make_scaled_complex, _scaled_from_complex_log

try:
    from scipy.linalg import eigh as _subset_eigh
except Exception:  # pragma: no cover - optional accelerator
    _subset_eigh = None


def _bounded_svd(matrix: np.ndarray, max_bond: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = matrix.shape
    rank = min(rows, columns)
    keep = min(int(max_bond), rank)
    if _subset_eigh is None or keep == rank:
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        return u[:, :keep], singular[:keep], vh[:keep]
    if rows <= columns:
        gram = matrix @ matrix.conj().T
        values, u = _subset_eigh(
            gram, subset_by_index=(rows - keep, rows - 1), check_finite=False, driver="evr"
        )
        order = np.argsort(values)[::-1]
        singular = np.sqrt(np.maximum(values[order], 0.0))
        u = u[:, order]
        vh = (u.conj().T @ matrix) / np.maximum(singular[:, None], np.finfo(float).tiny)
        return u, singular, vh
    gram = matrix.conj().T @ matrix
    values, v = _subset_eigh(
        gram, subset_by_index=(columns - keep, columns - 1), check_finite=False, driver="evr"
    )
    order = np.argsort(values)[::-1]
    singular = np.sqrt(np.maximum(values[order], 0.0))
    v = v[:, order]
    u = (matrix @ v) / np.maximum(singular[None, :], np.finfo(float).tiny)
    return u, singular, v.conj().T


class _BoundaryMPS:
    def __init__(self, scalar: complex, *, max_bond: int, cutoff: float) -> None:
        self.labels: list[int] = []
        self.tensors: list[np.ndarray] = []
        self.phase_scalar = complex(scalar)
        self.log2_scale = 0.0
        self.max_bond = max(1, int(max_bond))
        self.cutoff = max(0.0, float(cutoff))
        self.center = -1
        self.discarded_sq = 0.0
        self.max_discarded = 0.0
        self.peak_active = 0
        self.peak_bond = 1

    def _absorb(self, value: complex) -> bool:
        magnitude = abs(value)
        if magnitude == 0.0 or not math.isfinite(magnitude):
            return False
        self.phase_scalar *= value / magnitude
        self.log2_scale += math.log2(magnitude)
        return True

    def _normalize(self, data: np.ndarray) -> np.ndarray | None:
        magnitude = float(np.max(np.abs(data)))
        if magnitude == 0.0 or not math.isfinite(magnitude) or not self._absorb(magnitude):
            return None
        return data / magnitude

    def append(self, label: int, unary_phase: complex) -> None:
        self.labels.append(int(label))
        self.tensors.append(
            np.asarray([1.0 + 0j, unary_phase], dtype=np.complex128).reshape(1, 2, 1)
        )
        if self.center < 0:
            self.center = 0
        self.peak_active = max(self.peak_active, len(self.labels))

    def move_center(self, target: int) -> bool:
        while self.center < target:
            tensor = self.tensors[self.center]
            left, _, right = tensor.shape
            q_factor, r_factor = np.linalg.qr(tensor.reshape(left * 2, right))
            bond = q_factor.shape[1]
            self.tensors[self.center] = q_factor.reshape(left, 2, bond)
            self.tensors[self.center + 1] = np.einsum(
                "ab,bic->aic", r_factor, self.tensors[self.center + 1], optimize=True
            )
            self.center += 1
        while self.center > target:
            tensor = self.tensors[self.center]
            left, _, right = tensor.shape
            q_factor, r_factor = np.linalg.qr(tensor.reshape(left, 2 * right).T)
            bond = q_factor.shape[1]
            self.tensors[self.center] = q_factor.T.reshape(bond, 2, right)
            self.tensors[self.center - 1] = np.einsum(
                "aib,bc->aic", self.tensors[self.center - 1], r_factor.T, optimize=True
            )
            self.center -= 1
        normalized = self._normalize(self.tensors[self.center])
        if normalized is None:
            return False
        self.tensors[self.center] = normalized
        return True

    def _split(self, index: int, gate: np.ndarray | None = None) -> bool:
        left_tensor, right_tensor = self.tensors[index : index + 2]
        theta = np.einsum("aib,bjc->aijc", left_tensor, right_tensor, optimize=True)
        if gate is not None:
            theta *= gate[None, :, :, None]
        left, _, _, right = theta.shape
        matrix = theta.reshape(left * 2, 2 * right)
        try:
            u, singular, vh = _bounded_svd(matrix, self.max_bond)
        except np.linalg.LinAlgError:
            return False
        if singular.size == 0 or singular[0] == 0.0:
            return False
        keep = int(singular.size)
        if self.cutoff:
            keep = min(keep, max(1, int(np.count_nonzero(singular >= self.cutoff * singular[0]))))
        total_sq = float(np.vdot(matrix, matrix).real)
        kept_sq = float(singular[:keep] @ singular[:keep])
        lost_sq = 0.0 if keep == min(matrix.shape) else max(0.0, total_sq - kept_sq)
        ratio = math.sqrt(lost_sq / total_sq) if total_sq else 0.0
        self.discarded_sq += ratio * ratio
        self.max_discarded = max(self.max_discarded, ratio)
        scale = float(singular[0])
        if not self._absorb(scale):
            return False
        self.tensors[index] = u[:, :keep].reshape(left, 2, keep)
        self.tensors[index + 1] = (
            (singular[:keep] / scale)[:, None] * vh[:keep]
        ).reshape(keep, 2, right)
        self.center = index + 1
        self.peak_bond = max(self.peak_bond, keep)
        return True

    def apply_gate(self, left_index: int, right_index: int, gate: np.ndarray) -> bool:
        if not self.move_center(left_index):
            return False
        if right_index == left_index + 1:
            return self._split(left_index, gate)
        gate_u, gate_s, gate_vh = np.linalg.svd(gate, full_matrices=False)
        rank = int(np.count_nonzero(gate_s > np.finfo(float).eps * gate_s[0]))
        root_s = np.sqrt(gate_s[:rank])
        self._expand_interval(
            left_index,
            right_index,
            gate_u[:, :rank] * root_s[None, :],
            root_s[:, None] * gate_vh[:rank],
        )
        self._right_canonicalize_interval(left_index, right_index)
        for index in range(left_index, right_index):
            if not self._split(index):
                return False
        return True

    def _expand_interval(
        self, left_index: int, right_index: int, left_op: np.ndarray, right_op: np.ndarray
    ) -> None:
        rank = left_op.shape[1]
        tensor = self.tensors[left_index]
        left, _, right = tensor.shape
        self.tensors[left_index] = np.einsum(
            "apb,pr->apbr", tensor, left_op, optimize=True
        ).reshape(left, 2, right * rank)
        for index in range(left_index + 1, right_index):
            tensor = self.tensors[index]
            left, _, right = tensor.shape
            expanded = np.zeros((left, rank, 2, right, rank), dtype=np.complex128)
            for channel in range(rank):
                expanded[:, channel, :, :, channel] = tensor
            self.tensors[index] = expanded.reshape(left * rank, 2, right * rank)
        tensor = self.tensors[right_index]
        left, _, right = tensor.shape
        self.tensors[right_index] = (
            tensor[:, None, :, :] * right_op[None, :, :, None]
        ).reshape(left * rank, 2, right)

    def _right_canonicalize_interval(self, left_index: int, right_index: int) -> None:
        for index in range(right_index, left_index, -1):
            tensor = self.tensors[index]
            left, _, right = tensor.shape
            q_factor, r_factor = np.linalg.qr(tensor.reshape(left, 2 * right).T)
            bond = q_factor.shape[1]
            self.tensors[index] = q_factor.T.reshape(bond, 2, right)
            self.tensors[index - 1] = np.einsum(
                "aib,bc->aic", self.tensors[index - 1], r_factor.T, optimize=True
            )
        self.center = left_index
        normalized = self._normalize(self.tensors[left_index])
        if normalized is not None:
            self.tensors[left_index] = normalized

    def remove(self, index: int) -> bool:
        if not self.move_center(index):
            return False
        matrix = self.tensors[index].sum(axis=1)
        if len(self.tensors) == 1:
            value = complex(matrix[0, 0])
            self.tensors.pop()
            self.labels.pop()
            self.center = -1
            return self._absorb(value)
        if index + 1 < len(self.tensors):
            merged = np.einsum("ab,bic->aic", matrix, self.tensors[index + 1], optimize=True)
            self.center = index
            target = index + 1
        else:
            merged = np.einsum("aib,bc->aic", self.tensors[index - 1], matrix, optimize=True)
            self.center = index - 1
            target = index - 1
        normalized = self._normalize(merged)
        if normalized is None:
            return False
        self.tensors[target] = normalized
        self.tensors.pop(index)
        self.labels.pop(index)
        return True

    def finish(self) -> ScaledComplex | None:
        if self.labels or self.tensors:
            return None
        magnitude = abs(self.phase_scalar)
        if magnitude == 0.0 or not math.isfinite(magnitude):
            return _make_scaled_complex(0j)
        log_value = complex(
            (self.log2_scale + math.log2(magnitude)) * math.log(2.0),
            cmath.phase(self.phase_scalar),
        )
        return _scaled_from_complex_log(log_value)
