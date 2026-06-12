"""Tests for cache registration, stats, and clearing behavior."""

from __future__ import annotations

import terket


def test_public_cache_stats_and_clear_caches() -> None:
    terket.compute_circuit_amplitude(
        terket.make_circuit(1, [("h", 0)]),
        [0],
        [0],
        as_complex=True,
    )

    stats = terket.cache_stats()
    names = {str(row["name"]) for row in stats}
    assert "engine.native_support_from_mask" in names
    assert "engine.structure.classification_data" in names
    assert "engine.subcircuit_macro.plan" in names
    assert all(
        {"name", "kind", "max_entries", "current_size", "hits", "misses"} <= set(row)
        for row in stats
    )

    terket.clear_caches()
    cleared = {str(row["name"]): row for row in terket.cache_stats()}
    assert cleared["engine.native_support_from_mask"]["current_size"] == 0
    assert cleared["engine.structure.classification_data"]["current_size"] == 0
