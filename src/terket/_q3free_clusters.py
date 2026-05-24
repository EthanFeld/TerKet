"""Compatibility facade for ``terket._q3free.clusters``."""

from __future__ import annotations

import sys as _sys

from ._q3free import clusters as _impl

_sys.modules[__name__] = _impl
setattr(_sys.modules[__package__], __name__.rpartition(".")[2], _impl)
