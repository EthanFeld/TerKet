"""Compatibility facade for TerKet engine runtime."""

from __future__ import annotations

import sys as _sys

from . import _engine_runtime as _runtime

_sys.modules[__name__] = _runtime
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _runtime)
