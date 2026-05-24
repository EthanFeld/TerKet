"""State-building helpers and `SchurState`."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys

from ._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals

_LOCAL_NAMES = {
    '_ArbitraryPhaseTerm',
    '_support_from_mask',
    '_apply_affine_bit_in_place',
    '_apply_diag_phase_in_place',
    '_apply_bilinear_phase_in_place',
    '_solve_output_from_echelon',
    'SchurState',

}

_MISSING = object()
_ENGINE_LOCAL_BASELINES = {}
_LOCAL_IMPLS = {}
_FORCE_ENGINE_BINDINGS_REFRESH = "pytest" in sys.modules


def _sync_from_engine(engine) -> None:
    _sync_extracted_globals(
        globals(),
        engine,
        local_names=_LOCAL_NAMES,
        local_impls=_LOCAL_IMPLS,
        baselines=_ENGINE_LOCAL_BASELINES,
        missing=_MISSING,
        respect_mock_wraps=True,
    )


_MISSING, _ENGINE_LOCAL_BASELINES = _bootstrap_extracted_globals(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)

@dataclass(frozen=True, slots=True)
class _ArbitraryPhaseTerm:
    """Deferred exact phase on an affine Boolean output of the current state."""

    row_mask: int
    offset: int
    angle: float

def _support_from_mask(mask):
    if _native_level3_enabled():
        return _schur_native.support_from_mask(mask)
    return tuple(_iter_mask_bits(mask))

def _apply_affine_bit_in_place(q, row_mask, offset, alpha):
    """Add alpha/mod_q1 * (offset xor parity(row_mask*f)) minus the constant term."""
    alpha %= q.mod_q1
    if not alpha or not row_mask:
        return

    support = _support_from_mask(row_mask)
    support_len = len(support)
    linear = alpha if not offset else (-alpha) % q.mod_q1
    pair = (-alpha) % q.mod_q2 if not offset else alpha % q.mod_q2
    cubic = alpha % q.mod_q3 if not offset else (-alpha) % q.mod_q3

    if linear:
        for idx in support:
            q.q1[idx] = (q.q1[idx] + linear) % q.mod_q1

    if pair:
        if support_len == 2:
            key = (support[0], support[1])
            value = (q.q2.get(key, 0) + pair) % q.mod_q2
            if value:
                q.q2[key] = value
            elif key in q.q2:
                del q.q2[key]
        elif support_len == 3:
            idx0, idx1, idx2 = support
            for key in ((idx0, idx1), (idx0, idx2), (idx1, idx2)):
                value = (q.q2.get(key, 0) + pair) % q.mod_q2
                if value:
                    q.q2[key] = value
                elif key in q.q2:
                    del q.q2[key]
        else:
            for idx0, idx1 in combinations(support, 2):
                key = (idx0, idx1)
                value = (q.q2.get(key, 0) + pair) % q.mod_q2
                if value:
                    q.q2[key] = value
                elif key in q.q2:
                    del q.q2[key]

    if cubic and support_len >= 3:
        if support_len == 3:
            key = (support[0], support[1], support[2])
            value = (q.q3.get(key, 0) + cubic) % q.mod_q3
            if value:
                q.q3[key] = value
            elif key in q.q3:
                del q.q3[key]
        else:
            for idx0, idx1, idx2 in combinations(support, 3):
                key = (idx0, idx1, idx2)
                value = (q.q3.get(key, 0) + cubic) % q.mod_q3
                if value:
                    q.q3[key] = value
                elif key in q.q3:
                    del q.q3[key]

def _apply_diag_phase_in_place(q, row_mask, shift, alpha):
    """Apply alpha/mod_q1 * g where g is an affine output bit."""
    if shift:
        q.q0 = (q.q0 + Fraction(alpha, q.mod_q1)) % 1
    _apply_affine_bit_in_place(q, row_mask, shift, alpha)

def _apply_bilinear_phase_in_place(q, row_mask0, shift0, row_mask1, shift1, coeff):
    """Apply coeff/mod_q2 * g0 * g1 for affine output bits g0 and g1."""
    coeff %= q.mod_q2
    if not coeff:
        return
    if shift0 and shift1:
        q.q0 = (q.q0 + Fraction(coeff, q.mod_q2)) % 1
    _apply_affine_bit_in_place(q, row_mask0, shift0, coeff)
    _apply_affine_bit_in_place(q, row_mask1, shift1, coeff)
    _apply_affine_bit_in_place(q, row_mask0 ^ row_mask1, shift0 ^ shift1, (-coeff) % q.mod_q1)

def _solve_output_from_echelon(
    eps0: Sequence[int],
    cache: EchelonCache,
    output_bits: BitSequence,
) -> tuple[int, tuple[int, ...], tuple[int, ...], int] | None:
    return _state_solve_output_from_echelon(
        eps0,
        cache,
        output_bits,
        native_solver=_native_solve_for_output,
    )

class SchurState:
    """3rd-order tensor data (E, eps, q) over G=Z2^n [BL26 Def. 33]."""

    def __init__(self, n: int) -> None:
        self.n, self.m = n, 0
        self.eps: list[int] = [0] * n
        self.eps0: list[int] = [0] * n
        self.q: PhaseFunction = CubicFunction(0)
        self.scalar: complex = 1.0 + 0j
        self.scalar_half_pow2: int = 0
        self.output_refcount: list[int] = []
        self._arbitrary_phases: list[_ArbitraryPhaseTerm] = []
        self._pending_dead: set[int] = set()
        self._defer_early_elim = False
        self._cached_classification_data = None
        self._cached_classification_q = None

    def _invalidate_classification_cache(self) -> None:
        self._cached_classification_data = None
        self._cached_classification_q = None

    def _current_classification_data(self):
        if self._cached_classification_q is self.q and self._cached_classification_data is not None:
            return self._cached_classification_data
        classification_data = _build_classification_data(self.q)
        self._cached_classification_data = classification_data
        self._cached_classification_q = self.q
        return classification_data

    def _add_var(self):
        if not getattr(self.q, "_schur_mutable", True):
            self.q = _copy_cubic_function(self.q)
        self._invalidate_classification_cache()
        idx = self.m
        self.m += 1
        self.output_refcount.append(0)
        self.q.n = self.m
        self.q.q1.append(0)
        return idx

    def _update_reference_mask(self, old_mask, new_mask):
        changed = old_mask ^ new_mask
        newly_dead = []
        while changed:
            bit = changed & -changed
            idx = bit.bit_length() - 1
            if new_mask & bit:
                self.output_refcount[idx] += 1
            else:
                self.output_refcount[idx] -= 1
                if self.output_refcount[idx] == 0:
                    newly_dead.append(idx)
            changed ^= bit
        return newly_dead

    def _set_row_mask(self, qubit, new_mask):
        old_mask = self.eps[qubit]
        if old_mask == new_mask:
            return []
        newly_dead = self._update_reference_mask(old_mask, new_mask)
        self.eps[qubit] = new_mask
        return newly_dead

    def _rebuild_output_refcount(self):
        self.output_refcount = [0] * self.m
        for row_mask in self.eps:
            for idx in _iter_mask_bits(row_mask):
                self.output_refcount[idx] += 1
        for term in self._arbitrary_phases:
            for idx in _iter_mask_bits(term.row_mask):
                self.output_refcount[idx] += 1

    def _remap_mask_after_removal(self, mask, removed):
        if not mask or not removed:
            return mask
        new_mask = 0
        for idx in _iter_mask_bits(mask):
            shift = bisect.bisect_left(removed, idx)
            new_mask |= 1 << (idx - shift)
        return new_mask

    def _build_mask_remap_chunks(self, removed: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
        removed = tuple(sorted(int(idx) for idx in removed))
        kept_starts: list[int] = []
        old_starts: list[int] = []
        lengths: list[int] = []
        old = 0
        kept = 0
        for idx in removed:
            if idx > old:
                kept_starts.append(kept)
                old_starts.append(old)
                lengths.append(idx - old)
                kept += idx - old
            old = idx + 1
        if old < self.m + len(removed):
            kept_starts.append(kept)
            old_starts.append(old)
            lengths.append(self.m + len(removed) - old)
        return kept_starts, old_starts, lengths

    @staticmethod
    def _remap_mask_after_removal_chunks(
        mask: int,
        kept_starts: Sequence[int],
        old_starts: Sequence[int],
        lengths: Sequence[int],
    ) -> int:
        if not mask:
            return 0
        new_mask = 0
        for kept_start, old_start, length in zip(kept_starts, old_starts, lengths):
            chunk = (mask >> old_start) & ((1 << length) - 1)
            if chunk:
                new_mask |= chunk << kept_start
        return new_mask

    def _apply_elimination_result(self, new_q, half_pow2, removed):
        old_m = self.m
        old_output_refcount = self.output_refcount
        self.q = new_q
        self.m = new_q.n
        self.scalar_half_pow2 += half_pow2
        self._invalidate_classification_cache()
        removed = sorted(removed)
        if removed:
            kept_starts, old_starts, lengths = self._build_mask_remap_chunks(removed)
            remap = self._remap_mask_after_removal_chunks
            self.eps = [remap(mask, kept_starts, old_starts, lengths) for mask in self.eps]
        else:
            self.eps = list(self.eps)
        remapped_terms: list[_ArbitraryPhaseTerm] = []
        for term in self._arbitrary_phases:
            remapped_mask = (
                self._remap_mask_after_removal_chunks(term.row_mask, kept_starts, old_starts, lengths)
                if removed
                else term.row_mask
            )
            if remapped_mask:
                remapped_terms.append(_ArbitraryPhaseTerm(remapped_mask, term.offset, term.angle))
            elif term.offset:
                self.scalar *= cmath.exp(1j * term.angle)
        self._arbitrary_phases = remapped_terms
        if removed:
            removed_set = set(removed)
            self.output_refcount = [
                count
                for idx, count in enumerate(old_output_refcount[:old_m])
                if idx not in removed_set
            ]
        else:
            self.output_refcount = list(old_output_refcount)

    def _queue_dead_candidates(self, candidates):
        if not candidates:
            return
        self._pending_dead.update(candidates)
        if not self._defer_early_elim and len(self._pending_dead) >= _build_early_elim_batch_size(self.q.level):
            self._flush_pending_dead_variables()

    def _flush_pending_dead_variables(self):
        if not self._pending_dead:
            return
        candidates = tuple(sorted(self._pending_dead))
        self._pending_dead.clear()
        self._early_eliminate_dead_variables(candidates)

    def _early_eliminate_dead_variables(self, candidates=None):
        if self.scalar == 0j or self.m == 0:
            self._pending_dead.clear()
            return

        if candidates is None:
            dead = {idx for idx, count in enumerate(self.output_refcount) if count == 0}
        else:
            dead = {idx for idx in candidates if idx < self.m and self.output_refcount[idx] == 0}
        if not dead:
            return

        changed = True
        while changed and dead:
            changed = False
            classification_data = self._current_classification_data()
            ordered_dead = tuple(sorted(dead))
            classification_entries = {
                var: _classification_entry(self.q, var, classification_data=classification_data)
                for var in ordered_dead
            }

            decoupled = [
                var
                for var in ordered_dead
                if classification_entries[var][0] == _CLASS_CONSTRAINT_DECOUPLED
            ]
            if decoupled:
                new_q, half_pow2 = _elim_decoupled_constraints_batch(self.q, decoupled)
                self._apply_elimination_result(new_q, half_pow2, decoupled)
                dead = {idx for idx, count in enumerate(self.output_refcount) if count == 0}
                changed = True
                continue

            batch_quadratic = [
                var
                for var in ordered_dead
                if (
                    self.q.level == 3
                    and not self.q.q3
                    and classification_entries[var][0] == _CLASS_QUADRATIC
                    and not bool(classification_entries[var][2])
                )
            ]
            if len(batch_quadratic) >= 8:
                new_q, half_pow2, removed = _elim_sparse_dead_quadratics_batch(
                    self.q,
                    batch_quadratic,
                    classification_data=classification_data,
                )
                if removed:
                    self._apply_elimination_result(new_q, half_pow2, removed)
                    dead = {idx for idx, count in enumerate(self.output_refcount) if count == 0}
                    changed = True
                    continue

            for var in ordered_dead:
                entry = classification_entries[var]
                tag = entry[0]
                if tag in {_CLASS_CONSTRAINT_DECOUPLED, _CLASS_CONSTRAINT_ZERO, _CLASS_CONSTRAINT_PARITY}:
                    if tag == _CLASS_CONSTRAINT_ZERO:
                        self.scalar = 0j
                        self.scalar_half_pow2 = 0
                        self.q = PhaseFunction(0, level=self.q.level)
                        self.m = 0
                        self.eps = [0] * self.n
                        self.eps0 = [0] * self.n
                        self.output_refcount = []
                        self._arbitrary_phases = []
                        self._pending_dead.clear()
                        return
                    continue
                if tag != _CLASS_QUADRATIC:
                    continue
                # Above Clifford+T precision the same affine substitutions used
                # by `_elim_quadratic(...)` can leave the q3-free kernel outside
                # the current exact PhaseFunction representation. The runtime
                # reducer already bypasses that elimination regime, so keep the
                # build-time dead-variable pass consistent and defer the exact
                # q3-free work to the final solver.
                if self.q.level > 3:
                    continue
                if bool(entry[2]):
                    continue
                if _should_defer_build_quadratic_elimination(
                    self.q,
                    var,
                    classification_data=classification_data,
                ):
                    continue
                new_q, half_pow2 = _elim_quadratic(
                    self.q,
                    var,
                    classification_data=classification_data,
                )
                self._apply_elimination_result(new_q, half_pow2, [var])
                dead = {idx for idx, count in enumerate(self.output_refcount) if count == 0}
                changed = True
                break

    # -- Gates --

    def _ensure_mutable_phase_function(self) -> None:
        if not getattr(self.q, "_schur_mutable", True):
            self.q = _copy_cubic_function(self.q)
            self._invalidate_classification_cache()

    def _ensure_phase_precision(self, precision_level: int) -> None:
        if precision_level > self.q.level:
            self._ensure_mutable_phase_function()
            self.q.promote_in_place(precision_level)
            self._invalidate_classification_cache()

    def _lift_linear_coeff(self, coeff: int, precision_level: int) -> int:
        self._ensure_phase_precision(precision_level)
        return (int(coeff) * (1 << (self.q.level - precision_level))) % self.q.mod_q1

    def _lift_quadratic_coeff(self, coeff: int, precision_level: int) -> int:
        self._ensure_phase_precision(precision_level)
        return (int(coeff) * (1 << (self.q.level - precision_level))) % self.q.mod_q2

    def cnot(self, c: int, t: int) -> None:                   # [BL26 Sec. VA]
        newly_dead = self._set_row_mask(t, self.eps[t] ^ self.eps[c])
        self.eps0[t] ^= self.eps0[c]
        self._queue_dead_candidates(newly_dead)

    def _diag(self, qubit: int, q1v: int, precision_level: int = 3) -> None:  # [BL26 Eq.290]
        """Apply a unary dyadic phase to the current affine output bit."""
        q1v = self._lift_linear_coeff(q1v, precision_level)
        row_mask, sh = self.eps[qubit], self.eps0[qubit]
        if not row_mask:
            self.scalar *= cmath.exp(2j * cmath.pi * q1v * sh / self.q.mod_q1)
        else:
            self._ensure_mutable_phase_function()
            _apply_diag_phase_in_place(self.q, row_mask, sh, q1v)
            self._invalidate_classification_cache()

    def t(self, q: int) -> None:
        self._diag(q, 1, precision_level=3)

    def tdg(self, q: int) -> None:
        self._diag(q, -1, precision_level=3)

    def s(self, q: int) -> None:
        self._diag(q, 2, precision_level=3)

    def sdg(self, q: int) -> None:
        self._diag(q, -2, precision_level=3)

    def z(self, q: int) -> None:
        self._diag(q, 4, precision_level=3)

    def rz_arbitrary(self, qubit: int, angle: float) -> None:
        """Apply `diag(1, exp(i * angle))` exactly."""
        angle_value = float(angle)
        if not math.isfinite(angle_value):
            raise ValueError(f"rz_arbitrary angle must be finite, received {angle!r}.")
        if math.isclose(math.remainder(angle_value, 2.0 * math.pi), 0.0, rel_tol=0.0, abs_tol=1e-15):
            return
        row_mask = self.eps[qubit]
        offset = self.eps0[qubit] & 1
        if not row_mask:
            if offset:
                self.scalar *= cmath.exp(1j * angle_value)
            return
        self._arbitrary_phases.append(_ArbitraryPhaseTerm(row_mask, offset, angle_value))
        self._update_reference_mask(0, row_mask)

    def rz_dyadic(self, qubit: int, coeff: int, precision_level: int) -> None:
        """Apply diag(1, exp(2*pi*i*coeff / 2^precision_level))."""
        self._diag(qubit, coeff, precision_level=precision_level)

    def _apply_sx_family(self, qubit: int, q0_coeff: int, q1_coeff: int, diag_coeff: int) -> None:
        old_mask = self.eps[qubit]
        old_shift = self.eps0[qubit]
        new_var = self._add_var()

        self.q.q0 = (self.q.q0 + Fraction(self._lift_linear_coeff(q0_coeff, 3), self.q.mod_q1)) % 1
        self.q.q1[new_var] = (self.q.q1[new_var] + self._lift_linear_coeff(q1_coeff, 3)) % self.q.mod_q1

        if old_mask or old_shift:
            _apply_diag_phase_in_place(
                self.q,
                old_mask,
                old_shift,
                self._lift_linear_coeff(diag_coeff, 3),
            )
            _apply_bilinear_phase_in_place(
                self.q,
                old_mask,
                old_shift,
                1 << new_var,
                0,
                self._lift_quadratic_coeff(2, 3),
            )

        self._invalidate_classification_cache()
        newly_dead = self._set_row_mask(qubit, 1 << new_var)
        self.eps0[qubit] = 0
        self.scalar_half_pow2 -= 1
        self._queue_dead_candidates(newly_dead)

    def sx(self, qubit: int) -> None:
        """Apply `sqrt(X)` with one fresh path variable."""
        self._apply_sx_family(qubit, 1, 6, 6)

    def sxdg(self, qubit: int) -> None:
        """Apply `sqrt(X)†` with one fresh path variable."""
        self._apply_sx_family(qubit, 7, 2, 2)

    def rz_pi_2k(self, qubit: int, k: int, dagger: bool = False) -> None:
        coeff = -1 if dagger else 1
        self.rz_dyadic(qubit, coeff, precision_level=k + 1)

    def rz_pi_16(self, qubit: int) -> None:
        self.rz_pi_2k(qubit, 4, dagger=False)

    def rz_pi_16_dg(self, qubit: int) -> None:
        self.rz_pi_2k(qubit, 4, dagger=True)

    def rz_pi_32(self, qubit: int) -> None:
        self.rz_pi_2k(qubit, 5, dagger=False)

    def rz_pi_32_dg(self, qubit: int) -> None:
        self.rz_pi_2k(qubit, 5, dagger=True)

    def x(self, qubit: int) -> None:
        """Apply a Pauli X gate by toggling the affine output offset."""
        self.eps0[qubit] ^= 1

    def cz(self, q0: int, q1: int) -> None:                  # [BL26 Eq.309]
        """Apply the exact `CZ` phase on two affine output bits."""
        r0,r1 = self.eps[q0],self.eps[q1]
        s0,s1 = self.eps0[q0],self.eps0[q1]
        if not r0 and not r1:
            self.scalar *= cmath.exp(2j*cmath.pi*s0*s1/2)
        else:
            self._ensure_mutable_phase_function()
            _apply_bilinear_phase_in_place(
                self.q,
                r0,
                s0,
                r1,
                s1,
                self._lift_quadratic_coeff(2, 3),
            )
            self._invalidate_classification_cache()

    def rzz_dyadic(self, q0: int, q1: int, coeff: int, precision_level: int) -> None:
        """Apply the exact dyadic parity phase on ``(q0, q1)``."""
        modulus = 1 << int(precision_level)
        coeff = int(coeff) % modulus
        if coeff == 0:
            return
        self._diag(q0, coeff, precision_level=precision_level)
        self._diag(q1, coeff, precision_level=precision_level)
        r0, r1 = self.eps[q0], self.eps[q1]
        s0, s1 = self.eps0[q0], self.eps0[q1]
        if not r0 and not r1:
            self.scalar *= cmath.exp(
                2j
                * cmath.pi
                * Fraction(self._lift_quadratic_coeff(-coeff, precision_level), self.q.mod_q2)
                * s0
                * s1
            )
        else:
            self._ensure_mutable_phase_function()
            _apply_bilinear_phase_in_place(
                self.q,
                r0,
                s0,
                r1,
                s1,
                self._lift_quadratic_coeff(-coeff, precision_level),
            )
            self._invalidate_classification_cache()

    def pauli_expbox(self, paulis: Sequence[str], qubits: Sequence[int], angle: float) -> None:
        """Apply an exact Pauli-string phase without materializing a parity ladder."""
        apply_pauli_expbox_to_state(self, paulis, qubits, angle)

    def h(self, qubit: int) -> None:                         # [BL26 Eq.284]
        k=qubit
        old_mask = self.eps[k]
        a=self._add_var()
        for j in _iter_mask_bits(old_mask):
            key=(min(j,a),max(j,a))
            value = (self.q.q2.get(key, 0) + self._lift_quadratic_coeff(2, 3)) % self.q.mod_q2
            if value:
                self.q.q2[key] = value
            elif key in self.q.q2:
                del self.q.q2[key]
        if self.eps0[k]:
            self.q.q1[a] = (self.q.q1[a] + self._lift_linear_coeff(4, 3)) % self.q.mod_q1
        newly_dead = self._set_row_mask(k, 1 << a)
        self.eps0[k]=0
        self.scalar_half_pow2 -= 1
        self._queue_dead_candidates(newly_dead)

    def _prepare_echelon(self) -> EchelonCache:
        """Row-reduce the output constraint matrix once for batch reuse."""
        self._flush_pending_dead_variables()
        return _prepare_affine_constraint_cache(self.n, self.m, self.eps)

    def _prepare_constraint_echelon(self) -> EchelonCache:
        """Row-reduce output constraints without building solution bases."""
        self._flush_pending_dead_variables()
        rows = self.eps[:]
        rows, row_ops, pivot_col, used_mask = _row_reduce_output_constraints(self.n, rows)
        return EchelonCache(
            n=self.n,
            m=self.m,
            echelon_rows=tuple(rows),
            pivot_col=tuple(pivot_col),
            used_mask=used_mask,
            row_ops=tuple(row_ops),
            free_vars=(),
            gamma_masks=(),
            n_free=self.m - used_mask.bit_count(),
        )

    def _solve_for_output(
        self,
        cache: EchelonCache,
        output_bits: BitSequence,
    ) -> tuple[int, tuple[int, ...], tuple[int, ...], int] | None:
        """Solve the output constraints for one output string."""
        return _solve_output_from_echelon(self.eps0, cache, output_bits)

    def _transform_arbitrary_phases(
        self,
        shift_mask: int,
        gamma_masks: Sequence[int],
    ) -> tuple[complex, tuple[_ArbitraryPhaseTerm, ...]]:
        scalar = 1.0 + 0.0j
        transformed: list[_ArbitraryPhaseTerm] = []
        for term in self._arbitrary_phases:
            offset = (int(term.offset) ^ _parity(int(term.row_mask) & shift_mask)) & 1
            row_mask = 0
            for idx in _iter_mask_bits(int(term.row_mask)):
                row_mask ^= int(gamma_masks[idx])
            if row_mask == 0:
                if offset:
                    scalar *= cmath.exp(1j * float(term.angle))
                continue
            transformed.append(_ArbitraryPhaseTerm(row_mask, offset, float(term.angle)))
        return scalar, _coalesce_arbitrary_phase_terms(transformed)

    # -- Amplitude -------------------------------------------------

    def _amplitude_internal(
        self,
        output_bits: BitSequence,
        preserve_scale: bool = False,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        allow_unbounded_bp_result: bool = False,
    ) -> tuple[ScaledAmplitude | complex, ReductionInfo]:
        if _FORCE_ENGINE_BINDINGS_REFRESH:
            _sync_from_engine(importlib.import_module("terket._engine_impl"))
        if len(output_bits) != self.n:
            raise ValueError(f"Expected {self.n} output bits, received {len(output_bits)}.")
        if self.m == 0:
            ok = all(self.eps0[idx] == output_bits[idx] for idx in range(self.n))
            scaled = (
                _normalize_scaled_complex(
                    self.scalar * cmath.exp(2j * cmath.pi * float(self.q.q0)),
                    self.scalar_half_pow2,
                )
                if ok
                else _make_scaled_complex(0j)
            )
            amp = ScaledAmplitude.from_tuple(scaled) if preserve_scale else _scaled_to_complex(scaled)
            return amp, _info(0, 0, 0, 0, 0, zero=not ok)

        cache = self._prepare_echelon()
        solved = self._solve_for_output(cache, output_bits)
        if solved is None:
            zero_scaled = _make_scaled_complex(0j)
            amp = ScaledAmplitude.from_tuple(zero_scaled) if preserve_scale else 0j
            return amp, _info(0, 0, 0, 0, 0, zero=True)

        from ._reduction_support import _ReductionContext

        context = _ReductionContext(
            preserve_scale=preserve_scale,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
        )
        shift_mask, _, gamma, k = solved
        q_free = _aff_compose_cached(self.q, shift_mask, gamma, k, context=context)
        arbitrary_scalar, arbitrary_terms = self._transform_arbitrary_phases(shift_mask, gamma)

        if arbitrary_terms:
            result, max_scope, arbitrary_backend, arbitrary_metadata = _sum_with_arbitrary_phases_scaled(
                q_free,
                arbitrary_terms,
                allow_approximate=bool(_get_solver_config().allow_approximate),
            )
            structural_obstruction = (
                0
                if not q_free.q3
                else _phase3_plan(q_free, allow_tensor_contraction=allow_tensor_contraction)[3]
            )
            elim_info = {
                'quad': 0,
                'constraint': 0,
                'branched': 0,
                'remaining': max_scope,
                'structural_obstruction': structural_obstruction,
                'gauss_obstruction': _gauss_obstruction(q_free, structural_obstruction),
                'cost_r': max_scope,
                'phase_states': 0,
                'phase_splits': 0,
                'phase3_backend': arbitrary_backend,
            }
            elim_info.update(arbitrary_metadata)
        else:
            result, elim_info = _reduce_and_sum_scaled(q_free, context=context)

        scaled_amp = _normalize_scaled_complex(
            complex(self.scalar) * arbitrary_scalar * result[0],
            result[1] + self.scalar_half_pow2,
        )
        info = _info(
            k,
            elim_info['quad'],
            elim_info['constraint'],
            elim_info['branched'],
            elim_info['remaining'],
            structural_obstruction=elim_info.get('structural_obstruction', elim_info['remaining']),
            gauss_obstruction=elim_info.get(
                'gauss_obstruction',
                elim_info.get('structural_obstruction', elim_info['remaining']),
            ),
            phase_states=elim_info.get('phase_states', 0),
            phase_splits=elim_info.get('phase_splits', 0),
            zero=scaled_amp[0] == 0j,
            cost_model_r=elim_info.get('cost_r', elim_info['remaining']),
            phase3_backend=elim_info.get('phase3_backend'),
        )
        for key in (
            "is_approximate",
            "approx_backend",
            "approx_validation",
            "bp_heuristic_ensemble_size",
            "bp_heuristic_log2_abs_spread",
            "bp_heuristic_phase_spread",
            "bp_heuristic_max_log2_probability",
        ):
            if key in elim_info:
                info[key] = elim_info[key]  # type: ignore[typeddict-unknown-key]
        if _arbitrary_bp_backend(info.get("phase3_backend")):
            log2_probability = _scaled_probability_log2(scaled_amp)
            info["bp_log2_probability"] = log2_probability  # type: ignore[typeddict-unknown-key]
            if log2_probability > _ARBITRARY_BP_DIRECT_PROB_LOG2_TOL:
                if allow_unbounded_bp_result:
                    _mark_invalid_arbitrary_bp_info(info, scaled_amp)
                else:
                    _raise_if_invalid_arbitrary_bp_amplitude(info, scaled_amp)
        amp = ScaledAmplitude.from_tuple(scaled_amp) if preserve_scale else _scaled_to_complex(scaled_amp)
        return amp, info

    @overload
    def amplitude(
        self,
        output_bits: BitSequence,
        *,
        as_complex: Literal[False] = False,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> tuple[ScaledAmplitude, ReductionInfo]:
        ...

    @overload
    def amplitude(
        self,
        output_bits: BitSequence,
        *,
        as_complex: Literal[True],
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> tuple[complex, ReductionInfo]:
        ...

    def amplitude(
        self,
        output_bits: BitSequence,
        *,
        as_complex: bool = False,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> tuple[ScaledAmplitude | complex, ReductionInfo]:
        _token = _set_solver_config(solver_config)
        try:
            return self._amplitude_internal(
                output_bits,
                preserve_scale=not as_complex,
                allow_tensor_contraction=allow_tensor_contraction,
                extended_reductions=extended_reductions,
            )
        finally:
            _reset_solver_config(_token)

    def amplitude_scaled(
        self,
        output_bits: BitSequence,
        *,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> tuple[ScaledAmplitude, ReductionInfo]:
        return self.amplitude(
            output_bits,
            as_complex=False,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
            solver_config=solver_config,
        )

    def amplitudes(
        self,
        output_list: Sequence[BitSequence],
        *,
        as_complex: bool = False,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> list[tuple[ScaledAmplitude | complex, ReductionInfo]]:
        _token = _set_solver_config(solver_config)
        try:
            return _batch_query_state(
                self,
                output_list,
                preserve_scale=not as_complex,
                allow_tensor_contraction=allow_tensor_contraction,
                extended_reductions=extended_reductions,
                analyze_only=False,
            )
        finally:
            _reset_solver_config(_token)

    def amplitudes_scaled(
        self,
        output_list: Sequence[BitSequence],
        *,
        allow_tensor_contraction: bool = True,
        extended_reductions: ExtendedReductionMode | str = "auto",
        solver_config: "SolverConfig | None" = None,
    ) -> list[tuple[ScaledAmplitude, ReductionInfo]]:
        return self.amplitudes(
            output_list,
            as_complex=False,
            allow_tensor_contraction=allow_tensor_contraction,
            extended_reductions=extended_reductions,
            solver_config=solver_config,
        )


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
