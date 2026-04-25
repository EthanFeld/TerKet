"""
Level-aware cubic phase arithmetic over binary variables for the TerKet engine.

`PhaseFunction` stores a degree-3 phase polynomial with configurable dyadic
precision:

- `q0` in `R/Z`
- `q1` coefficients in `Z_(2^L)`
- `q2` coefficients in `Z_(2^(L-1))`
- `q3` coefficients in `Z_(2^(L-2))`

When `L = 3`, this reduces to the original Clifford+T `CubicFunction`
representation with `Z8 / Z4 / Z2` coefficients.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence


BitVector = Sequence[int]
LinearMap = Sequence[Sequence[int]]


def _modulus_for_degree(level: int, degree: int) -> int:
    shift = level - degree + 1
    if shift <= 0:
        return 1
    return 1 << shift


class PhaseFunction:
    """
    Degree-3 phase function on binary variables with configurable precision.

    Storage layout:
        q0: Fraction mod 1
        q1: list[int mod 2^L], length n
        q2: dict[(k, l)] -> int mod 2^(L-1) for k < l
        q3: dict[(k, l, m)] -> int mod 2^(L-2) for k < l < m
    """

    __slots__ = (
        "n",
        "level",
        "mod_q1",
        "mod_q2",
        "mod_q3",
        "q0",
        "q1",
        "q2",
        "q3",
        "_schur_mutable",
        "_schur_q_key",
        "_schur_q_structure_key",
        "_schur_q_classification_structure_key",
        "_schur_q_phase3_structure_key",
        "_schur_q3_support_key",
        "_schur_q_classification_data",
        "_schur_q_classification_lookup",
    )

    def __init__(
        self,
        n: int,
        level: int = 3,
        q0: Fraction | int = Fraction(0),
        q1: Sequence[int] | None = None,
        q2: dict[tuple[int, int], int] | None = None,
        q3: dict[tuple[int, int, int], int] | None = None,
    ) -> None:
        if level < 1:
            raise ValueError("PhaseFunction level must be positive.")
        self.n = int(n)
        self.level = int(level)
        self.mod_q1 = _modulus_for_degree(self.level, 1)
        self.mod_q2 = _modulus_for_degree(self.level, 2)
        self.mod_q3 = _modulus_for_degree(self.level, 3)

        self.q0 = Fraction(q0) % 1
        self.q1 = [int(x) % self.mod_q1 for x in (q1 or [0] * self.n)]
        assert len(self.q1) == self.n
        self.q2 = {
            key: int(value) % self.mod_q2
            for key, value in (q2 or {}).items()
            if int(value) % self.mod_q2
        }
        self.q3 = {
            key: int(value) % self.mod_q3
            for key, value in (q3 or {}).items()
            if int(value) % self.mod_q3
        }
        # Circuit-construction states mutate these structures in place, while
        # reducer intermediates created through ``_phase_function_from_parts``
        # are treated as immutable and can cache structural digests safely.
        self._schur_mutable = True
        self._schur_q_key = None
        self._schur_q_structure_key = None
        self._schur_q_classification_structure_key = None
        self._schur_q_phase3_structure_key = None
        self._schur_q3_support_key = None
        self._schur_q_classification_data = None
        self._schur_q_classification_lookup = None

    def copy(self) -> PhaseFunction:
        return PhaseFunction(
            self.n,
            level=self.level,
            q0=self.q0,
            q1=list(self.q1),
            q2=dict(self.q2),
            q3=dict(self.q3),
        )

    def promote(self, level: int) -> PhaseFunction:
        """Return an equivalent phase function at precision ``level``."""
        level = int(level)
        if level <= self.level:
            return self.copy()

        factor = 1 << (level - self.level)
        return PhaseFunction(
            self.n,
            level=level,
            q0=self.q0,
            q1=[value * factor for value in self.q1],
            q2={key: value * factor for key, value in self.q2.items()},
            q3={key: value * factor for key, value in self.q3.items()},
        )

    def promote_in_place(self, level: int) -> None:
        """Widen coefficient precision in place."""
        if level <= self.level:
            return
        promoted = self.promote(level)
        self.level = promoted.level
        self.mod_q1 = promoted.mod_q1
        self.mod_q2 = promoted.mod_q2
        self.mod_q3 = promoted.mod_q3
        self.q0 = promoted.q0
        self.q1 = promoted.q1
        self.q2 = promoted.q2
        self.q3 = promoted.q3
        self._schur_q_key = None
        self._schur_q_structure_key = None
        self._schur_q_classification_structure_key = None
        self._schur_q_phase3_structure_key = None
        self._schur_q3_support_key = None
        self._schur_q_classification_data = None
        self._schur_q_classification_lookup = None

    # -- Evaluation ------------------------------------------------

    def _eval_residue(self, g: BitVector) -> int:
        """Return the phase residue modulo ``2^level``."""
        value = sum(self.q1[idx] * g[idx] for idx in range(self.n))
        value += sum((self.mod_q1 // self.mod_q2) * coeff * g[i] * g[j] for (i, j), coeff in self.q2.items())
        value += sum(
            (self.mod_q1 // self.mod_q3) * coeff * g[i] * g[j] * g[k]
            for (i, j, k), coeff in self.q3.items()
        )
        return value % self.mod_q1

    def _eval_z8(self, g: BitVector) -> int:
        """Compatibility alias for the original Clifford+T evaluator name."""
        return self._eval_residue(g)

    def evaluate(self, g: BitVector) -> Fraction:
        return (self.q0 + Fraction(self._eval_residue(g), self.mod_q1)) % 1

    # -- Derivatives ----------------------------------------------

    def deriv2(self, g: BitVector, h: BitVector) -> Fraction:
        gh = [(g[idx] + h[idx]) % 2 for idx in range(self.n)]
        return (self.evaluate(gh) - self.evaluate(g) - self.evaluate(h) + self.q0) % 1

    def deriv3(self, g: BitVector, h: BitVector, k: BitVector) -> Fraction:
        gh = [(g[idx] + h[idx]) % 2 for idx in range(self.n)]
        gk = [(g[idx] + k[idx]) % 2 for idx in range(self.n)]
        hk = [(h[idx] + k[idx]) % 2 for idx in range(self.n)]
        ghk = [(g[idx] + h[idx] + k[idx]) % 2 for idx in range(self.n)]
        return (
            self.evaluate(ghk)
            - self.evaluate(gh)
            - self.evaluate(gk)
            - self.evaluate(hk)
            + self.evaluate(g)
            + self.evaluate(h)
            + self.evaluate(k)
            - self.q0
        ) % 1

    # -- Dressed tensors ------------------------------------------
    # These retain the original parity-style views used by the binary-variable
    # engine. They are still useful for structural checks and remain unchanged.

    def M3(self, i: int, j: int, k: int) -> int:
        if i > j:
            i, j = j, i
        if j > k:
            j, k = k, j
            if i > j:
                i, j = j, i
        if i == j == k:
            return self.q1[i] % 2
        if i == j:
            return self.q2.get((i, k), 0) % 2
        if j == k:
            return self.q2.get((i, j), 0) % 2
        return self.q3.get((i, j, k), 0) % 2

    def M2(self, i: int, j: int) -> int:
        if i == j:
            return (-self.q1[i]) % 4
        if i > j:
            i, j = j, i
        return self.q2.get((i, j), 0) % 4

    # -- Composition ----------------------------------------------

    def compose(self, gamma: LinearMap) -> PhaseFunction:
        """
        Compose with a binary linear map ``gamma : Z2^p -> Z2^n``.

        The result stays degree-3; only the coefficient rings depend on
        ``self.level``.
        """
        n = self.n
        p = len(gamma[0]) if gamma else 0
        assert len(gamma) == n
        cols = [[gamma[row][col] for row in range(n)] for col in range(p)]
        vc = [self._eval_residue(cols[col]) for col in range(p)]
        q1 = [vc[col] % self.mod_q1 for col in range(p)]
        q2: dict[tuple[int, int], int] = {}
        xor_cache: dict[tuple[int, int], list[int]] = {}

        for left in range(p):
            for right in range(left + 1, p):
                image = [(cols[left][idx] + cols[right][idx]) % 2 for idx in range(n)]
                residue = (self._eval_residue(image) - vc[left] - vc[right]) % self.mod_q1
                assert residue % 2 == 0
                coeff = (residue // 2) % self.mod_q2
                if coeff:
                    q2[(left, right)] = coeff
                xor_cache[(left, right)] = image

        q3: dict[tuple[int, int, int], int] = {}
        for a in range(p):
            for b in range(a + 1, p):
                for c in range(b + 1, p):
                    xab = xor_cache.get((a, b), [(cols[a][idx] + cols[b][idx]) % 2 for idx in range(n)])
                    xac = xor_cache.get((a, c), [(cols[a][idx] + cols[c][idx]) % 2 for idx in range(n)])
                    xbc = xor_cache.get((b, c), [(cols[b][idx] + cols[c][idx]) % 2 for idx in range(n)])
                    xabc = [(xab[idx] + cols[c][idx]) % 2 for idx in range(n)]
                    residue = (
                        self._eval_residue(xabc)
                        - self._eval_residue(xab)
                        - self._eval_residue(xac)
                        - self._eval_residue(xbc)
                        + vc[a]
                        + vc[b]
                        + vc[c]
                    ) % self.mod_q1
                    assert residue % 4 == 0
                    coeff = (residue // 4) % self.mod_q3
                    if coeff:
                        q3[(a, b, c)] = coeff

        return PhaseFunction(p, level=self.level, q0=self.q0, q1=q1, q2=q2, q3=q3)

    # -- Group operations -----------------------------------------

    def __add__(self, other: PhaseFunction) -> PhaseFunction:
        if self.n != other.n:
            raise ValueError("Cannot add phase functions over different variable counts.")
        level = max(self.level, other.level)
        left = self.promote(level) if self.level != level else self
        right = other.promote(level) if other.level != level else other
        q2 = {
            key: (left.q2.get(key, 0) + right.q2.get(key, 0)) % left.mod_q2
            for key in set(left.q2) | set(right.q2)
        }
        q3 = {
            key: (left.q3.get(key, 0) + right.q3.get(key, 0)) % left.mod_q3
            for key in set(left.q3) | set(right.q3)
        }
        return PhaseFunction(
            left.n,
            level=level,
            q0=(left.q0 + right.q0) % 1,
            q1=[(left.q1[idx] + right.q1[idx]) % left.mod_q1 for idx in range(left.n)],
            q2={key: value for key, value in q2.items() if value},
            q3={key: value for key, value in q3.items() if value},
        )

    def __neg__(self) -> PhaseFunction:
        return PhaseFunction(
            self.n,
            level=self.level,
            q0=(-self.q0) % 1,
            q1=[(-value) % self.mod_q1 for value in self.q1],
            q2={key: (-value) % self.mod_q2 for key, value in self.q2.items()},
            q3={key: (-value) % self.mod_q3 for key, value in self.q3.items()},
        )

    def __sub__(self, other: PhaseFunction) -> PhaseFunction:
        return self + (-other)

    def is_zero(self) -> bool:
        return self.q0 % 1 == 0 and all(value == 0 for value in self.q1) and not self.q2 and not self.q3

    def is_quadratic(self) -> bool:
        if self.q3:
            return False
        if any(value % 2 for value in self.q2.values()):
            return False
        if any(value % 2 for value in self.q1):
            return False
        return True

    def __repr__(self) -> str:
        parts = [f"n={self.n}"]
        if self.level != 3:
            parts.append(f"level={self.level}")
        if self.q0:
            parts.append(f"q0={self.q0}")
        nonzero_q1 = {idx: value for idx, value in enumerate(self.q1) if value}
        if nonzero_q1:
            parts.append(f"q1={nonzero_q1}")
        if self.q2:
            parts.append(f"q2={self.q2}")
        if self.q3:
            parts.append(f"q3={self.q3}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class CubicFunction(PhaseFunction):
    """Backward-compatible alias for the original Clifford+T phase function."""

    def __init__(
        self,
        n: int,
        q0: Fraction | int = Fraction(0),
        q1: Sequence[int] | None = None,
        q2: dict[tuple[int, int], int] | None = None,
        q3: dict[tuple[int, int, int], int] | None = None,
    ) -> None:
        super().__init__(n, level=3, q0=q0, q1=q1, q2=q2, q3=q3)


def T_state() -> CubicFunction:
    return CubicFunction(1, q1=[1])


def CS_state() -> CubicFunction:
    return CubicFunction(2, q2={(0, 1): 1})


def CCZ_state() -> CubicFunction:
    return CubicFunction(3, q3={(0, 1, 2): 1})


def detect_factorization(q: PhaseFunction) -> list[set[int]]:
    """Return connected components of the coupling graph."""
    adjacency = [set() for _ in range(q.n)]
    for i, j in q.q2:
        adjacency[i].add(j)
        adjacency[j].add(i)
    for i, j, k in q.q3:
        adjacency[i].update([j, k])
        adjacency[j].update([i, k])
        adjacency[k].update([i, j])

    active_vars = {idx for idx, coeff in enumerate(q.q1) if coeff}
    for i, j in q.q2:
        active_vars.update([i, j])
    for i, j, k in q.q3:
        active_vars.update([i, j, k])

    visited: set[int] = set()
    components: list[set[int]] = []
    for start in sorted(active_vars):
        if start in visited:
            continue
        component: set[int] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            stack.extend(adjacency[node] - visited)
        if component:
            components.append(component)
    return components
