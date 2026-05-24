"""Shared MQT benchmark helpers."""

from __future__ import annotations

import hashlib
import math
from typing import Any


def hash_bits(label: str, width: int) -> tuple[int, ...]:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    bits = tuple((digest[idx % len(digest)] >> (idx % 8)) & 1 for idx in range(width))
    if width and not any(bits):
        bits = (1,) + bits[1:]
    return bits


def bind_deterministic_parameters(circuit: Any, benchmark: str, circuit_size: int) -> Any:
    if not circuit.parameters:
        return circuit

    assignments = {}
    ordered = sorted(circuit.parameters, key=lambda param: param.name)
    for idx, param in enumerate(ordered):
        digest = hashlib.sha256(f"{benchmark}:{circuit_size}:{param.name}:{idx}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], byteorder="little", signed=False) % 256
        assignments[param] = (2.0 * math.pi * bucket) / 256.0
    return circuit.assign_parameters(assignments, inplace=False)


__all__ = ["bind_deterministic_parameters", "hash_bits"]
