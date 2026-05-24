"""Recovered pauli support helpers."""

from __future__ import annotations

import cmath
import math
import os
from typing import Sequence

from ._engine_runtime_core import _configure_extracted_module
from .spec import CircuitSpec
from .state import ReductionInfo

_LOCAL_NAMES = {
    '_normalize_pauli_expbox_terms',
    'apply_pauli_expbox_to_state',
    '_pauli_expbox_dyadic_snap_level',
    '_native_mps_approx_bond',
    '_approx_pauli_expectation_info',
    '_pauli_beam_approx_terms',
    '_pauli_beam_needs_large_default',
    '_pauli_masks_from_string',
    '_pauli_masks_from_sparse',
    '_pauli_code',
    '_pauli_product_phase',
    '_pauli_product_phase_left_parts',
    '_pauli_beam_prune',
    '_pauli_beam_reverse_ops',
    '_pauli_beam_approx_pauli_expectations'
}
_LOCAL_IMPLS = {}
_configure_extracted_module(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)


def _normalize_pauli_expbox_terms(
    paulis: Sequence[str],
    qubits: Sequence[int],
) -> tuple[tuple[str, int], ...]:
    if len(paulis) != len(qubits):
        raise ValueError("PauliExpBox paulis and qubits must have equal length.")

    active: list[tuple[str, int]] = []
    for pauli, qubit in zip(paulis, qubits):
        pauli_char = str(pauli).upper()
        if pauli_char == "I":
            continue
        if pauli_char not in {"X", "Y", "Z"}:
            raise ValueError(f"Unsupported PauliExpBox Pauli {pauli!r}.")
        active.append((pauli_char, int(qubit)))
    return tuple(active)

def apply_pauli_expbox_to_state(
    state: "SchurState",
    paulis: Sequence[str],
    qubits: Sequence[int],
    angle: float,
) -> None:
    """Apply one PauliExpBox to a SchurState without materializing a CNOT ladder."""
    angle_value = float(angle)
    if not math.isfinite(angle_value):
        raise ValueError(f"pauli_expbox angle must be finite, received {angle!r}.")

    active = _normalize_pauli_expbox_terms(paulis, qubits)
    if not active:
        state.scalar *= cmath.exp(-0.5j * angle_value)
        return

    if math.isclose(math.remainder(angle_value, 2.0 * math.pi), 0.0, rel_tol=0.0, abs_tol=1e-15):
        state.scalar *= cmath.exp(-0.5j * angle_value)
        return

    for pauli_char, qubit in active:
        if pauli_char == "X":
            state.h(qubit)
        elif pauli_char == "Y":
            state.sdg(qubit)
            state.h(qubit)

    row_mask = 0
    offset = 0
    for _pauli_char, qubit in active:
        row_mask ^= state.eps[qubit]
        offset ^= state.eps0[qubit] & 1
    state.scalar *= cmath.exp(-0.5j * angle_value)
    snap_level = _pauli_expbox_dyadic_snap_level()
    if snap_level is not None:
        modulus = 1 << snap_level
        coeff = int(round(angle_value * modulus / (2.0 * math.pi)))
        snapped = 2.0 * math.pi * coeff / modulus
        state.scalar *= cmath.exp(0.5j * (angle_value - snapped))
        coeff %= modulus
        if row_mask and coeff:
            state._ensure_phase_precision(snap_level)
            _apply_diag_phase_in_place(
                state.q,
                row_mask,
                offset,
                state._lift_linear_coeff(coeff, snap_level),
            )
            state._invalidate_classification_cache()
        elif offset and coeff:
            state.scalar *= cmath.exp(1j * snapped)
    else:
        if row_mask:
            state._arbitrary_phases.append(_ArbitraryPhaseTerm(row_mask, offset, angle_value))
            state._update_reference_mask(0, row_mask)
        elif offset:
            state.scalar *= cmath.exp(1j * angle_value)

    for pauli_char, qubit in reversed(active):
        if pauli_char == "X":
            state.h(qubit)
        elif pauli_char == "Y":
            state.h(qubit)
            state.s(qubit)

def _pauli_expbox_dyadic_snap_level() -> int | None:
    raw = os.environ.get("TERKET_PAULI_EXPBOX_DYADIC_LEVEL")
    if raw is None or raw == "":
        return None
    try:
        level = int(raw)
    except ValueError:
        return None
    return max(1, level)

def _native_mps_approx_bond() -> int:
    raw = os.environ.get("TERKET_APPROX_MPS_BOND")
    if raw is None:
        return _NATIVE_MPS_APPROX_DEFAULT_BOND
    try:
        return max(1, int(raw))
    except ValueError:
        return _NATIVE_MPS_APPROX_DEFAULT_BOND

def _approx_pauli_expectation_info(
    spec: CircuitSpec,
    backend: str,
) -> ReductionInfo:
    bond = _native_mps_approx_bond()
    info = _info(
        spec.n_qubits,
        0,
        0,
        0,
        bond,
        structural_obstruction=spec.n_qubits,
        gauss_obstruction=spec.n_qubits,
        cost_model_r=bond,
        phase3_backend=backend,
    )
    info["is_approximate"] = True
    info["mps_max_bond"] = bond
    info["pauli_beam_max_terms"] = _pauli_beam_approx_terms(spec)
    return info

def _pauli_beam_approx_terms(spec: CircuitSpec | None = None) -> int:
    raw = os.environ.get("TERKET_APPROX_PAULI_BEAM")
    if raw is None:
        if spec is not None and _pauli_beam_needs_large_default(spec):
            return _PAULI_BEAM_APPROX_LARGE_TERMS
        return _PAULI_BEAM_APPROX_DEFAULT_TERMS
    try:
        return max(1, int(raw))
    except ValueError:
        return _PAULI_BEAM_APPROX_DEFAULT_TERMS

def _pauli_beam_needs_large_default(spec: CircuitSpec) -> bool:
    expbox_count = 0
    has_entangling_expbox = False
    for gate in spec.gates:
        name = str(gate[0])
        if name == "pauli_expbox":
            expbox_count += 1
            if len(tuple(gate[2])) > 1:
                has_entangling_expbox = True
            continue
        if name not in {"x", "z"}:
            return False
    return has_entangling_expbox and expbox_count >= 512

def _pauli_masks_from_string(pauli: str) -> tuple[int, int]:
    x_mask = 0
    z_mask = 0
    for idx, char in enumerate(pauli):
        if char == "I":
            continue
        if char == "X":
            x_mask |= 1 << idx
            continue
        if char == "Z":
            z_mask |= 1 << idx
            continue
        if char == "Y":
            x_mask |= 1 << idx
            z_mask |= 1 << idx
            continue
        raise ValueError(f"Observable must use only I/X/Y/Z characters, received {pauli!r}.")
    return x_mask, z_mask

def _pauli_masks_from_sparse(paulis: Sequence[str], qubits: Sequence[int]) -> tuple[int, int]:
    x_mask = 0
    z_mask = 0
    for pauli, qubit in zip(paulis, qubits):
        bit = 1 << int(qubit)
        pauli_char = str(pauli).upper()
        if pauli_char == "I":
            continue
        if pauli_char == "X":
            x_mask |= bit
            continue
        if pauli_char == "Z":
            z_mask |= bit
            continue
        if pauli_char == "Y":
            x_mask |= bit
            z_mask |= bit
            continue
        raise ValueError(f"Unsupported PauliExpBox Pauli {pauli!r}.")
    return x_mask, z_mask

def _pauli_code(x_mask: int, z_mask: int, bit: int) -> int:
    return (1 if x_mask & bit else 0) | (2 if z_mask & bit else 0)

def _pauli_product_phase(left: tuple[int, int], right: tuple[int, int]) -> complex:
    lx, lz = left
    rx, rz = right
    left_x = lx & ~lz
    left_z = lz & ~lx
    left_y = lx & lz
    right_x = rx & ~rz
    right_z = rz & ~rx
    right_y = rx & rz
    positive = (
        (left_x & right_z).bit_count()
        + (left_z & right_y).bit_count()
        + (left_y & right_x).bit_count()
    )
    negative = (
        (left_z & right_x).bit_count()
        + (left_y & right_z).bit_count()
        + (left_x & right_y).bit_count()
    )
    return (1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j)[(positive - negative) & 3]

def _pauli_product_phase_left_parts(
    left_x: int,
    left_z: int,
    left_y: int,
    right: tuple[int, int],
) -> complex:
    rx, rz = right
    right_x = rx & ~rz
    right_z = rz & ~rx
    right_y = rx & rz
    positive = (
        (left_x & right_z).bit_count()
        + (left_z & right_y).bit_count()
        + (left_y & right_x).bit_count()
    )
    negative = (
        (left_z & right_x).bit_count()
        + (left_y & right_z).bit_count()
        + (left_x & right_y).bit_count()
    )
    return (1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j)[(positive - negative) & 3]

def _pauli_beam_prune(
    terms: dict[tuple[int, int], complex],
    max_terms: int,
) -> dict[tuple[int, int], complex]:
    weighted = [
        (weight, key, value)
        for key, value in terms.items()
        if (weight := abs(value)) > 1e-15
    ]
    if len(weighted) <= max_terms:
        return {key: value for _weight, key, value in weighted}
    weighted.sort(key=lambda item: item[0], reverse=True)
    return {key: value for _weight, key, value in weighted[:max_terms]}

def _pauli_beam_reverse_ops(spec: CircuitSpec) -> list[_PauliBeamOp] | None:
    ops: list[_PauliBeamOp] = []
    for gate in reversed(spec.gates):
        name = str(gate[0])
        if name == "pauli_expbox":
            q_masks = _pauli_masks_from_sparse(gate[1], gate[2])
            qx, qz = q_masks
            if qx == 0 and qz == 0:
                continue
            qx_only = qx & ~qz
            qz_only = qz & ~qx
            qy = qx & qz
            ops.append((
                "pauli_expbox",
                (
                    qx,
                    qz,
                    qx_only,
                    qz_only,
                    qy,
                    math.cos(float(gate[3])),
                    -1.0j * math.sin(float(gate[3])),
                ),
            ))
            continue
        if name == "x":
            ops.append(("x", 1 << int(gate[1])))
            continue
        if name == "z":
            ops.append(("z", 1 << int(gate[1])))
            continue
        if name in {"h", "s", "sdg"}:
            ops.append((name, 1 << int(gate[1])))
            continue
        if name == "cnot":
            ops.append(("cnot", (1 << int(gate[1]), 1 << int(gate[2]))))
            continue
        return None
    return ops

def _pauli_beam_approx_pauli_expectations(
    spec: CircuitSpec,
    input_bits: Sequence[int],
    observables: Sequence[str],
    *,
    max_terms: int | None = None,
) -> list[complex] | None:
    limit = _pauli_beam_approx_terms(spec) if max_terms is None else max(1, int(max_terms))
    input_mask = 0
    for idx, bit in enumerate(input_bits):
        if int(bit) & 1:
            input_mask |= 1 << idx

    reverse_ops = _pauli_beam_reverse_ops(spec)
    if reverse_ops is None:
        return None

    results: list[complex] = []
    for observable in observables:
        terms: dict[tuple[int, int], complex] = {_pauli_masks_from_string(observable): 1.0 + 0.0j}
        for name, payload in reverse_ops:
            if name == "pauli_expbox":
                qx, qz, qx_only, qz_only, qy, cos_angle, minus_i_sin = payload
                updated: dict[tuple[int, int], complex] = {}
                for p_masks, coeff in terms.items():
                    px, pz = p_masks
                    anticommutes = (((qx & pz) ^ (qz & px)).bit_count() & 1) != 0
                    if not anticommutes:
                        updated[p_masks] = updated.get(p_masks, 0j) + coeff
                        continue
                    updated[p_masks] = updated.get(p_masks, 0j) + coeff * cos_angle
                    product_masks = (qx ^ px, qz ^ pz)
                    px_only = px & ~pz
                    pz_only = pz & ~px
                    py = px & pz
                    positive = (
                        (qx_only & pz_only).bit_count()
                        + (qz_only & py).bit_count()
                        + (qy & px_only).bit_count()
                    )
                    negative = (
                        (qz_only & px_only).bit_count()
                        + (qy & pz_only).bit_count()
                        + (qx_only & py).bit_count()
                    )
                    phase = (1.0 + 0.0j, 1.0j, -1.0 + 0.0j, -1.0j)[(positive - negative) & 3]
                    product_coeff = coeff * minus_i_sin * phase
                    updated[product_masks] = updated.get(product_masks, 0j) + product_coeff
                terms = _pauli_beam_prune(updated, limit)
                continue
            if name == "x":
                bit = int(payload)
                terms = {key: (-value if key[1] & bit else value) for key, value in terms.items()}
                continue
            if name == "z":
                bit = int(payload)
                terms = {key: (-value if key[0] & bit else value) for key, value in terms.items()}
                continue
            if name == "h":
                bit = int(payload)
                updated = {}
                for (x_mask, z_mask), coeff in terms.items():
                    x_bit = x_mask & bit
                    z_bit = z_mask & bit
                    next_x = (x_mask & ~bit) | (bit if z_bit else 0)
                    next_z = (z_mask & ~bit) | (bit if x_bit else 0)
                    next_coeff = -coeff if x_bit and z_bit else coeff
                    updated[(next_x, next_z)] = updated.get((next_x, next_z), 0j) + next_coeff
                terms = _pauli_beam_prune(updated, limit)
                continue
            if name == "s":
                bit = int(payload)
                updated = {}
                for (x_mask, z_mask), coeff in terms.items():
                    x_bit = x_mask & bit
                    z_bit = z_mask & bit
                    next_x = x_mask
                    next_z = z_mask ^ (bit if x_bit else 0)
                    next_coeff = -coeff if x_bit and not z_bit else coeff
                    updated[(next_x, next_z)] = updated.get((next_x, next_z), 0j) + next_coeff
                terms = _pauli_beam_prune(updated, limit)
                continue
            if name == "sdg":
                bit = int(payload)
                updated = {}
                for (x_mask, z_mask), coeff in terms.items():
                    x_bit = x_mask & bit
                    z_bit = z_mask & bit
                    next_x = x_mask
                    next_z = z_mask ^ (bit if x_bit else 0)
                    next_coeff = -coeff if x_bit and z_bit else coeff
                    updated[(next_x, next_z)] = updated.get((next_x, next_z), 0j) + next_coeff
                terms = _pauli_beam_prune(updated, limit)
                continue
            if name == "cnot":
                control_bit, target_bit = payload
                updated = {}
                for (x_mask, z_mask), coeff in terms.items():
                    image_x = 0
                    image_z = 0
                    image_coeff = coeff
                    control_code = _pauli_code(x_mask, z_mask, control_bit)
                    target_code = _pauli_code(x_mask, z_mask, target_bit)
                    for factor_x, factor_z in (
                        {
                            0: (0, 0),
                            1: (control_bit | target_bit, 0),
                            2: (0, control_bit),
                            3: (control_bit | target_bit, control_bit),
                        }[control_code],
                        {
                            0: (0, 0),
                            1: (target_bit, 0),
                            2: (0, control_bit | target_bit),
                            3: (target_bit, control_bit | target_bit),
                        }[target_code],
                    ):
                        if factor_x or factor_z:
                            image_coeff *= _pauli_product_phase((image_x, image_z), (factor_x, factor_z))
                            image_x ^= factor_x
                            image_z ^= factor_z
                    next_x = (x_mask & ~(control_bit | target_bit)) | image_x
                    next_z = (z_mask & ~(control_bit | target_bit)) | image_z
                    updated[(next_x, next_z)] = updated.get((next_x, next_z), 0j) + image_coeff
                terms = _pauli_beam_prune(updated, limit)
                continue

        total = 0.0 + 0.0j
        for (x_mask, z_mask), coeff in terms.items():
            if x_mask:
                continue
            sign = -1.0 if ((z_mask & input_mask).bit_count() & 1) else 1.0
            total += coeff * sign
        results.append(total)
    return results

_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
