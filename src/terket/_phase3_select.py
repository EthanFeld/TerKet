"""Compatibility facade for ``terket._phase3.select``."""

from __future__ import annotations

import sys as _sys

from ._phase3 import select as _impl

_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)
