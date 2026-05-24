"""Compatibility facade for the TerKet strong-simulation engine.

Implementation currently lives in ``terket._engine_impl`` while the repo is
being mechanically split into focused modules. This module aliases the impl
module so existing imports and private-helper monkeypatches keep working.
"""

from __future__ import annotations

import sys as _sys

from . import _engine_impl as _impl

_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)
