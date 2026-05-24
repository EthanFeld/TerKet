"""Central cache registry for TerKet runtime caches."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable


CacheStatsRow = dict[str, int | str | None]


@dataclass(slots=True)
class _RegisteredCache:
    name: str
    kind: str
    max_entries: int | None
    current_size: Callable[[], int | None]
    clear: Callable[[], None]
    hits: Callable[[], int | None]
    misses: Callable[[], int | None]


_CACHE_REGISTRY: dict[str, _RegisteredCache] = {}


class _BoundedMemoCache(OrderedDict):
    """Small LRU cache keyed by compact digests rather than full phase structures."""

    __slots__ = ("hits", "max_entries", "misses")

    def __init__(self, max_entries: int):
        super().__init__()
        self.max_entries = int(max_entries)
        self.hits = 0
        self.misses = 0

    def get(self, key, default=None):
        try:
            value = super().__getitem__(key)
        except KeyError:
            self.misses += 1
            return default
        self.hits += 1
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if self.max_entries <= 0:
            return
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.max_entries:
            self.popitem(last=False)

    def clear(self) -> None:
        super().clear()
        self.hits = 0
        self.misses = 0


def _register_cache(record: _RegisteredCache) -> None:
    _CACHE_REGISTRY[record.name] = record


def make_bounded_cache(name: str, max_entries: int) -> _BoundedMemoCache:
    cache = _BoundedMemoCache(max_entries)
    _register_cache(
        _RegisteredCache(
            name=name,
            kind="bounded_lru",
            max_entries=cache.max_entries,
            current_size=lambda cache=cache: len(cache),
            clear=cache.clear,
            hits=lambda cache=cache: cache.hits,
            misses=lambda cache=cache: cache.misses,
        )
    )
    return cache


def register_lru_cache(name: str, func: Any) -> Any:
    cache_info = getattr(func, "cache_info", None)
    cache_clear = getattr(func, "cache_clear", None)
    if cache_info is None or cache_clear is None:
        raise TypeError(f"{name!r} is not an lru_cache-wrapped callable.")

    def current_size() -> int | None:
        return int(cache_info().currsize)

    def max_entries() -> int | None:
        value = cache_info().maxsize
        return None if value is None else int(value)

    _register_cache(
        _RegisteredCache(
            name=name,
            kind="functools_lru",
            max_entries=max_entries(),
            current_size=current_size,
            clear=cache_clear,
            hits=lambda: int(cache_info().hits),
            misses=lambda: int(cache_info().misses),
        )
    )
    return func


def cache_stats() -> tuple[CacheStatsRow, ...]:
    rows: list[CacheStatsRow] = []
    for name in sorted(_CACHE_REGISTRY):
        record = _CACHE_REGISTRY[name]
        rows.append(
            {
                "name": record.name,
                "kind": record.kind,
                "max_entries": record.max_entries,
                "current_size": record.current_size(),
                "hits": record.hits(),
                "misses": record.misses(),
            }
        )
    return tuple(rows)


def clear_caches() -> None:
    for record in tuple(_CACHE_REGISTRY.values()):
        record.clear()


__all__ = [
    "CacheStatsRow",
    "_BoundedMemoCache",
    "cache_stats",
    "clear_caches",
    "make_bounded_cache",
    "register_lru_cache",
]
