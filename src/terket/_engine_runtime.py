"""Compatibility barrel that rebuilds historical ``_engine_impl`` runtime surface."""

from __future__ import annotations

import importlib
import sys as _sys

from . import _engine_runtime_core as _core
from . import _engine_runtime_state as _state
from . import _engine_runtime_q3free as _q3free
from . import _engine_runtime_phase3 as _phase3
from . import _engine_runtime_reduction as _reduction
from . import _engine_runtime_public as _public
from . import _engine_runtime_pauli as _pauli

_sys.modules[f"{__package__}._engine_impl"] = _core

from . import _arbitrary_clusters as _arbitrary_clusters_owner
from . import _pauli_api as _pauli_api_owner
from . import _pauli_approx_runtime as _pauli_approx_runtime_owner
from . import _reduction_support as _reduction_support_owner
from . import _state_direct as _state_direct_owner
from . import _state_runtime as _state_runtime_owner

from ._state_runtime import SchurState as SchurState
from ._state_runtime import _ArbitraryPhaseTerm as _ArbitraryPhaseTerm
from ._state_direct import _DirectAffineMaskPattern as _DirectAffineMaskPattern
from ._state_direct import _DirectPostReplayTemplate as _DirectPostReplayTemplate
from ._arbitrary_clusters import _ArbitraryFactorCutsetPlan as _ArbitraryFactorCutsetPlan
from ._arbitrary_clusters import _GenericQ2MediatorPlan as _GenericQ2MediatorPlan
from ._arbitrary_clusters import _GenericQ2MediatorSpec as _GenericQ2MediatorSpec
from ._arbitrary_clusters import _HalfPhaseClusterPlan as _HalfPhaseClusterPlan
from ._arbitrary_clusters import _HalfPhaseClusterSpec as _HalfPhaseClusterSpec
from ._arbitrary_clusters import _HalfPhaseMediatorPlan as _HalfPhaseMediatorPlan
from ._arbitrary_clusters import _HalfPhaseMediatorSpec as _HalfPhaseMediatorSpec
from ._reduction_support import _ReductionContext as _ReductionContext
from ._pauli_approx_runtime import _NativeApproxMPS as _NativeApproxMPS
from ._pauli_api import _PauliExpectationRequest as _PauliExpectationRequest


_MODULES = (
    _core,
    _state,
    _q3free,
    _phase3,
    _reduction,
    _public,
    _pauli,
)

_OWNER_MODULES = (
    _state_runtime_owner,
    _state_direct_owner,
    _arbitrary_clusters_owner,
    _reduction_support_owner,
    _pauli_approx_runtime_owner,
    _pauli_api_owner,
)

for _module in _MODULES:
    globals().update({name: value for name, value in vars(_module).items() if not name.startswith("__")})


def _preload_backend_owner_modules() -> None:
    # Move first-use import cost for extracted backend owners out of timed exact
    # amplitude paths. These imports stay within the TerKet package surface.
    for module_name in (
        "terket._q3free.primitives",
        "terket._q3free.factor_plans",
        "terket._q3free.clusters",
        "terket._q3free.plans",
        "terket._q3free.components",
        "terket._q3free.execution",
        "terket._q3free.raw_constraints",
        "terket._q3free.exact",
        "terket._q3free.fallbacks",
        "terket._q3free.treewidth",
        "terket._q3free.native",
        "terket._q3free.cutset_support",
        "terket._q3free.cutset_residue",
        "terket._q3free.cutset",
        "terket._q3free.cutset_exec",
        "terket._phase3.order",
        "terket._phase3.structure",
        "terket._phase3.select",
        "terket._phase3.factors",
        "terket._phase3.cover",
        "terket._phase3.exec",
    ):
        importlib.import_module(module_name)

def _sync_runtime_globals(module) -> None:
    local_names = set(getattr(module, "_LOCAL_NAMES", ()))
    baselines = getattr(module, "_ENGINE_LOCAL_BASELINES", None)
    if isinstance(baselines, dict):
        for name in local_names:
            if name in globals():
                baselines[name] = globals()[name]

    for name, value in _runtime_globals.items():
        if name in local_names:
            continue
        module.__dict__[name] = value

    local_impls = getattr(module, "_LOCAL_IMPLS", None)
    if isinstance(local_impls, dict):
        for name in local_names:
            if name in local_impls:
                module.__dict__[name] = local_impls[name]

_runtime_globals = {name: value for name, value in globals().items() if not name.startswith("__")}
for _module in _MODULES:
    _sync_runtime_globals(_module)
for _module in _OWNER_MODULES:
    _sync_runtime_globals(_module)

_sys.modules[f"{__package__}._engine_impl"] = _sys.modules[__name__]

_preload_backend_owner_modules()

__all__ = [name for name in globals() if not name.startswith("__")]
