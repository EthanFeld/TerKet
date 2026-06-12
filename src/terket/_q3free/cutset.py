"""Compatibility facade for q3-free cutset planning helpers.

This module keeps historical import paths stable while ownership lives in:
- ``cutset_runtime`` for plan finalization and runtime caches
- ``cutset_search_core`` for reusable candidate search
- ``cutset_search`` for cached and one-shot orchestration
"""

from __future__ import annotations

from .cutset_runtime import (
    _attach_q3_free_cutset_runtime_cache,
    _finalize_q3_free_cutset_conditioning_plan,
)
from .cutset_search import (
    _q3_free_cutset_conditioning_plan,
    _q3_free_one_shot_cutset_conditioning_plan,
)
from .cutset_search_core import _build_q3_free_cutset_conditioning_plan_uncached

__all__ = [
    "_attach_q3_free_cutset_runtime_cache",
    "_build_q3_free_cutset_conditioning_plan_uncached",
    "_finalize_q3_free_cutset_conditioning_plan",
    "_q3_free_cutset_conditioning_plan",
    "_q3_free_one_shot_cutset_conditioning_plan",
]
