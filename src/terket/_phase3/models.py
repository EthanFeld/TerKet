"""Owned Phase-3 backend selection models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Phase3BackendCandidate:
    backend: str
    score: tuple[int, int, int, int, int]
    separator: tuple[int, ...] | None = None
    peeled: bool = False

    @property
    def metadata_backend(self) -> str:
        if self.backend == "treewidth_dp" and self.peeled:
            return "treewidth_dp_peeled"
        return self.backend
