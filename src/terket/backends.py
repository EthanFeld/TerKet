"""Lazy optional backend imports for the TerKet strong-simulation stack."""

from __future__ import annotations

from .engine import (
    _get_cupy_module,
    _get_quimb_tensor_module,
    _load_schur_native_module,
    _quimb_import_reason,
    _schur_native,
)

__all__ = [
    "_get_cupy_module",
    "_get_quimb_tensor_module",
    "_load_schur_native_module",
    "_quimb_import_reason",
    "_schur_native",
]
