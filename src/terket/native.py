"""Optional native backend loading and quimb helpers."""

from __future__ import annotations

from functools import lru_cache
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import platform
import sys
from typing import Any

from .cache import register_lru_cache


def _load_schur_native_module():
    package_dir = Path(__file__).resolve().parent
    module_name = "terket._schur_native"
    suffixes = tuple(getattr(importlib.machinery, "EXTENSION_SUFFIXES", ()))
    in_tree_candidates: list[Path] = []
    for suffix in suffixes:
        in_tree_candidates.extend(sorted(package_dir.glob(f"_schur_native*{suffix}")))
    in_tree_path = max(in_tree_candidates, key=lambda path: path.stat().st_mtime_ns) if in_tree_candidates else None
    build_candidates = sorted(package_dir.parent.glob("build/lib.*/terket/_schur_native*"))

    candidate_paths: list[Path] = []
    if build_candidates:
        freshest_build = max(build_candidates, key=lambda path: path.stat().st_mtime_ns)
        if in_tree_path is None or freshest_build.stat().st_mtime_ns > in_tree_path.stat().st_mtime_ns:
            candidate_paths.append(freshest_build)
    if in_tree_path is not None:
        candidate_paths.append(in_tree_path)

    for candidate in candidate_paths:
        try:
            spec = importlib.util.spec_from_file_location(module_name, candidate)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except ImportError:
            sys.modules.pop(module_name, None)
            continue
        except OSError:
            sys.modules.pop(module_name, None)
            continue
    return None


_schur_native = _load_schur_native_module()
_QUIMB_TENSOR_MODULE = None
_QUIMB_TENSOR_IMPORT_ERROR = None
_NATIVE_MODULE_SENTINEL = object()


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _quimb_import_enabled() -> bool:
    """Return whether optional quimb imports are allowed in this process."""
    return not _env_flag_enabled("TERKET_DISABLE_QUIMB")


def _quimb_import_reason() -> str:
    """Explain why quimb-backed paths are unavailable."""
    if _env_flag_enabled("TERKET_DISABLE_QUIMB"):
        return "quimb support is disabled by TERKET_DISABLE_QUIMB."
    if _QUIMB_TENSOR_IMPORT_ERROR is not None:
        return f"quimb import failed: {_QUIMB_TENSOR_IMPORT_ERROR}"
    return "quimb is not installed."


def _import_quimb_tensor_module():
    """Import ``quimb.tensor`` while avoiding Python's slow Windows WMI probe."""
    if sys.platform != "win32" or not hasattr(platform, "_wmi_query"):
        import quimb.tensor as qtn

        return qtn

    original_wmi_query = platform._wmi_query
    original_uname_cache = getattr(platform, "_uname_cache", None)

    def _disabled_wmi_query(*args, **kwargs):
        raise OSError("disabled during quimb import")

    try:
        platform._wmi_query = _disabled_wmi_query
        if hasattr(platform, "_uname_cache"):
            platform._uname_cache = None
        import quimb.tensor as qtn

        return qtn
    finally:
        platform._wmi_query = original_wmi_query
        if hasattr(platform, "_uname_cache"):
            platform._uname_cache = original_uname_cache


def _native_level3_enabled(q: Any | None = None, *, native_module=_NATIVE_MODULE_SENTINEL) -> bool:
    module = _schur_native if native_module is _NATIVE_MODULE_SENTINEL else native_module
    return module is not None and (q is None or getattr(q, "level", 3) == 3)


def _native_aff_compose_enabled(*, native_module=_NATIVE_MODULE_SENTINEL) -> bool:
    module = _schur_native if native_module is _NATIVE_MODULE_SENTINEL else native_module
    return module is not None


def _native_symbol(name: str, *, native_module=_NATIVE_MODULE_SENTINEL):
    """Return an optional native helper without assuming a full ABI match."""
    module = _schur_native if native_module is _NATIVE_MODULE_SENTINEL else native_module
    if module is None:
        return None
    return getattr(module, name, None)


@lru_cache(maxsize=1)
def _kahypar_available() -> bool:
    return importlib.util.find_spec("kahypar") is not None


register_lru_cache("engine.kahypar_available", _kahypar_available)


def _get_quimb_tensor_module():
    """Return ``quimb.tensor`` when available, otherwise ``None``."""
    global _QUIMB_TENSOR_IMPORT_ERROR, _QUIMB_TENSOR_MODULE
    if not _quimb_import_enabled():
        return None
    if _QUIMB_TENSOR_MODULE is False:
        return None
    if _QUIMB_TENSOR_MODULE is None:
        try:
            qtn = _import_quimb_tensor_module()
        except Exception as exc:
            _QUIMB_TENSOR_IMPORT_ERROR = exc
            _QUIMB_TENSOR_MODULE = False
        else:
            _QUIMB_TENSOR_IMPORT_ERROR = None
            _QUIMB_TENSOR_MODULE = qtn
    return None if _QUIMB_TENSOR_MODULE is False else _QUIMB_TENSOR_MODULE


__all__ = [
    "_env_flag_enabled",
    "_get_quimb_tensor_module",
    "_import_quimb_tensor_module",
    "_kahypar_available",
    "_load_schur_native_module",
    "_native_aff_compose_enabled",
    "_native_level3_enabled",
    "_native_symbol",
    "_quimb_import_enabled",
    "_quimb_import_reason",
    "_schur_native",
]
