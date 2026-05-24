
"""Direct post-replay template helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib

from ._engine_runtime_core import _bootstrap_extracted_globals, _sync_extracted_globals

_LOCAL_NAMES = {
    '_DirectPostReplayTemplate',
    '_DirectAffineMaskPattern',
    '_direct_affine_mask_pattern',
    '_build_post_replay_state',
    '_build_direct_post_replay_validation_observable',
    '_build_direct_post_replay_template',
    '_construct_direct_post_replay_payload',
    '_direct_post_replay_payload_matches_state',

}

_MISSING = object()
_ENGINE_LOCAL_BASELINES = {}
_LOCAL_IMPLS = {}


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


_INITIAL_ENGINE = importlib.import_module("terket._engine_impl")
_MISSING, _ENGINE_LOCAL_BASELINES = _bootstrap_extracted_globals(
    globals(),
    local_names=_LOCAL_NAMES,
    local_impls=_LOCAL_IMPLS,
    respect_mock_wraps=True,
)
del _INITIAL_ENGINE

@dataclass(frozen=True, slots=True)
class _DirectPostReplayTemplate:
    """Precomputed fixed-row replay plan for a long Pauli-expectation suffix."""

    base_rows: tuple[int, ...]
    final_rows: tuple[int, ...]
    final_m: int
    suffix_ops: tuple[tuple[int, ...], ...]
    scalar_half_pow2_delta: int
    echelon_cache: EchelonCache
    z_coeff: int
    s_coeff: int
    sdg_coeff: int

@dataclass(frozen=True, slots=True)
class _DirectAffineMaskPattern:
    """Cached support/pair/triple lists for small fixed affine masks."""

    support: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    triples: tuple[tuple[int, int, int], ...]

_DIRECT_LINEAR_GATE_SPECS = {
    "t": (1, 3),
    "tdg": (-1, 3),
    "s": (2, 3),
    "sdg": (-2, 3),
    "z": (4, 3),
    "rz_pi_16": (1, 5),
    "rz_pi_16_dg": (-1, 5),
    "rz_pi_32": (1, 6),
    "rz_pi_32_dg": (-1, 6),
}

def _direct_affine_mask_pattern(mask: int) -> _DirectAffineMaskPattern:
    support = tuple(_support_from_mask(int(mask)))
    pairs = tuple(combinations(support, 2))
    triples = tuple(combinations(support, 3))
    return _DirectAffineMaskPattern(support=support, pairs=pairs, triples=triples)

def _build_post_replay_state(
    base_state: SchurState,
    observable_gates: Sequence[Gate],
    inverse_gates: Sequence[Gate],
) -> SchurState:
    state = _fork_state_for_extension(base_state)
    _apply_gate_sequence_to_state(state, observable_gates + inverse_gates)
    state._flush_pending_dead_variables()
    return state

def _build_direct_post_replay_validation_observable(observables: Sequence[str]) -> str | None:
    if not observables:
        return None
    if len(observables) == 1:
        return observables[0]
    n_qubits = len(observables[0])
    chosen = ["I"] * n_qubits
    saw_non_identity = False
    for qubit in range(n_qubits):
        choice = next(
            (pauli for pauli in ("Y", "X", "Z") if any(observable[qubit] == pauli for observable in observables)),
            "I",
        )
        chosen[qubit] = choice
        saw_non_identity |= choice != "I"
    return "".join(chosen) if saw_non_identity else observables[0]

def _build_direct_post_replay_template(
    base_state: SchurState,
    inverse_gates: Sequence[Gate],
    observable_count: int,
) -> _DirectPostReplayTemplate | None:
    if (
        observable_count < _DIRECT_POST_REPLAY_MIN_OBSERVABLES
        or len(inverse_gates) < _DIRECT_POST_REPLAY_MIN_SUFFIX_GATES
        or base_state._arbitrary_phases
    ):
        return None

    level = int(base_state.q.level)
    if level < 3:
        return None

    diag_coeffs: dict[tuple[int, int], int] = {}
    quad_coeffs: dict[tuple[int, int], int] = {}

    def linear_coeff(coeff: int, precision_level: int) -> int:
        key = (int(coeff), int(precision_level))
        cached = diag_coeffs.get(key)
        if cached is None:
            cached = _lift_direct_linear_coeff(level, coeff, precision_level)
            diag_coeffs[key] = cached
        return cached

    def quadratic_coeff(coeff: int, precision_level: int) -> int:
        key = (int(coeff), int(precision_level))
        cached = quad_coeffs.get(key)
        if cached is None:
            cached = _lift_direct_quadratic_coeff(level, coeff, precision_level)
            quad_coeffs[key] = cached
        return cached

    rows = list(base_state.eps)
    m = int(base_state.m)
    scalar_half_pow2_delta = 0
    suffix_ops: list[tuple[int, ...]] = []

    def append_new_var_op(qubit: int, *payload: int) -> None:
        nonlocal m, scalar_half_pow2_delta
        new_var = m
        suffix_ops.append((payload[0], qubit, rows[qubit], new_var, *payload[1:]))
        rows[qubit] = 1 << new_var
        m += 1
        scalar_half_pow2_delta -= 1

    for gate in inverse_gates:
        name = gate[0]
        if name == "x":
            suffix_ops.append((0, int(gate[1])))
            continue
        if name == "cnot":
            control = int(gate[1])
            target = int(gate[2])
            suffix_ops.append((1, control, target))
            rows[target] ^= rows[control]
            continue
        linear_gate = _DIRECT_LINEAR_GATE_SPECS.get(name)
        if linear_gate is not None:
            qubit = int(gate[1])
            coeff, precision_level = linear_gate
            suffix_ops.append((2, qubit, rows[qubit], linear_coeff(coeff, precision_level)))
            continue
        if name == "rz_dyadic":
            qubit = int(gate[1])
            coeff = linear_coeff(int(gate[2]), int(gate[3]))
            suffix_ops.append((2, qubit, rows[qubit], coeff))
            continue
        if name == "cz":
            q0 = int(gate[1])
            q1 = int(gate[2])
            suffix_ops.append((3, q0, q1, rows[q0], rows[q1], quadratic_coeff(2, 3)))
            continue
        if name == "rzz_dyadic":
            q0 = int(gate[1])
            q1 = int(gate[2])
            coeff = int(gate[3])
            precision_level = int(gate[4])
            lifted_linear = linear_coeff(coeff, precision_level)
            suffix_ops.append((2, q0, rows[q0], lifted_linear))
            suffix_ops.append((2, q1, rows[q1], lifted_linear))
            suffix_ops.append(
                (
                    3,
                    q0,
                    q1,
                    rows[q0],
                    rows[q1],
                    quadratic_coeff(-coeff, precision_level),
                )
            )
            continue
        if name == "h":
            append_new_var_op(int(gate[1]), 4, linear_coeff(4, 3), quadratic_coeff(2, 3))
            continue
        if name == "sx":
            append_new_var_op(
                int(gate[1]),
                5,
                linear_coeff(1, 3),
                linear_coeff(6, 3),
                linear_coeff(6, 3),
                quadratic_coeff(2, 3),
            )
            continue
        if name == "sxdg":
            append_new_var_op(
                int(gate[1]),
                5,
                linear_coeff(7, 3),
                linear_coeff(2, 3),
                linear_coeff(2, 3),
                quadratic_coeff(2, 3),
            )
            continue
        return None

    final_rows = tuple(rows)
    return _DirectPostReplayTemplate(
        base_rows=tuple(base_state.eps),
        final_rows=final_rows,
        final_m=m,
        suffix_ops=tuple(suffix_ops),
        scalar_half_pow2_delta=scalar_half_pow2_delta,
        echelon_cache=_prepare_affine_constraint_cache(base_state.n, m, final_rows),
        z_coeff=linear_coeff(4, 3),
        s_coeff=linear_coeff(2, 3),
        sdg_coeff=linear_coeff(-2, 3),
    )

def _construct_direct_post_replay_payload(
    base_state: SchurState,
    observable: str,
    template: _DirectPostReplayTemplate,
) -> tuple[tuple[int, ...], complex, int, PhaseFunction]:
    q = _copy_cubic_function_extended(base_state.q, template.final_m)
    eps0 = list(base_state.eps0)
    scalar = complex(base_state.scalar)
    scalar_half_pow2 = int(base_state.scalar_half_pow2) + int(template.scalar_half_pow2_delta)
    q1_terms = q.q1
    q2_terms = q.q2
    q3_terms = q.q3
    mod_q1 = int(q.mod_q1)
    mod_q2 = int(q.mod_q2)
    mod_q3 = int(q.mod_q3)
    q0_residue = _phase_fraction_to_residue(q.q0, mod_q1)
    base_rows = template.base_rows

    def apply_affine_pattern(pattern: _DirectAffineMaskPattern, offset: int, alpha: int) -> None:
        alpha %= mod_q1
        if not alpha or not pattern.support:
            return

        linear = alpha if not offset else (-alpha) % mod_q1
        pair = (-alpha) % mod_q2 if not offset else alpha % mod_q2
        cubic = alpha % mod_q3 if not offset else (-alpha) % mod_q3

        if linear:
            for idx in pattern.support:
                q1_terms[idx] = (q1_terms[idx] + linear) % mod_q1

        if pair:
            for key in pattern.pairs:
                value = (q2_terms.get(key, 0) + pair) % mod_q2
                if value:
                    q2_terms[key] = value
                elif key in q2_terms:
                    del q2_terms[key]

        if cubic:
            for key in pattern.triples:
                value = (q3_terms.get(key, 0) + cubic) % mod_q3
                if value:
                    q3_terms[key] = value
                elif key in q3_terms:
                    del q3_terms[key]

    def apply_diag_pattern(pattern: _DirectAffineMaskPattern, shift: int, alpha: int) -> None:
        nonlocal q0_residue
        if shift:
            q0_residue = (q0_residue + int(alpha)) % mod_q1
        apply_affine_pattern(pattern, shift, alpha)

    def apply_bilinear_patterns(
        pattern0: _DirectAffineMaskPattern,
        shift0: int,
        pattern1: _DirectAffineMaskPattern,
        shift1: int,
        xor_pattern: _DirectAffineMaskPattern,
        coeff: int,
    ) -> None:
        nonlocal q0_residue
        coeff %= mod_q2
        if not coeff:
            return
        if shift0 and shift1:
            q0_residue = (q0_residue + int(coeff)) % mod_q1
        apply_affine_pattern(pattern0, shift0, coeff)
        apply_affine_pattern(pattern1, shift1, coeff)
        apply_affine_pattern(xor_pattern, shift0 ^ shift1, (-coeff) % mod_q1)

    for qubit, pauli in enumerate(observable):
        pattern = _direct_affine_mask_pattern(base_rows[qubit])
        if pauli == "I":
            continue
        if pauli == "X":
            eps0[qubit] ^= 1
            continue
        if pauli == "Z":
            apply_diag_pattern(pattern, eps0[qubit], template.z_coeff)
            continue
        apply_diag_pattern(pattern, eps0[qubit], template.sdg_coeff)
        eps0[qubit] ^= 1
        apply_diag_pattern(pattern, eps0[qubit], template.s_coeff)

    for op in template.suffix_ops:
        code = op[0]
        if code == 0:
            eps0[op[1]] ^= 1
            continue
        if code == 1:
            control = op[1]
            target = op[2]
            eps0[target] ^= eps0[control]
            continue
        if code == 2:
            qubit = op[1]
            apply_diag_pattern(_direct_affine_mask_pattern(op[2]), eps0[qubit], op[3])
            continue
        if code == 3:
            q0 = op[1]
            q1_idx = op[2]
            pattern0 = _direct_affine_mask_pattern(op[3])
            pattern1 = _direct_affine_mask_pattern(op[4])
            xor_pattern = _direct_affine_mask_pattern(op[3] ^ op[4])
            apply_bilinear_patterns(pattern0, eps0[q0], pattern1, eps0[q1_idx], xor_pattern, op[5])
            continue
        if code == 4:
            qubit = op[1]
            old_mask = op[2]
            new_var = op[3]
            if old_mask:
                for old_var in _direct_affine_mask_pattern(old_mask).support:
                    key = (old_var, new_var) if old_var < new_var else (new_var, old_var)
                    value = (q2_terms.get(key, 0) + op[5]) % mod_q2
                    if value:
                        q2_terms[key] = value
                    elif key in q2_terms:
                        del q2_terms[key]
            if eps0[qubit]:
                q1_terms[new_var] = (q1_terms[new_var] + op[4]) % mod_q1
            eps0[qubit] = 0
            continue

        qubit = op[1]
        old_mask = op[2]
        new_var = op[3]
        old_pattern = _direct_affine_mask_pattern(old_mask)
        new_pattern = _direct_affine_mask_pattern(1 << new_var)
        xor_pattern = _direct_affine_mask_pattern(old_mask ^ (1 << new_var))
        q0_residue = (q0_residue + op[4]) % mod_q1
        q1_terms[new_var] = (q1_terms[new_var] + op[5]) % mod_q1
        shift = eps0[qubit] & 1
        if old_pattern.support or shift:
            apply_diag_pattern(old_pattern, shift, op[6])
            apply_bilinear_patterns(old_pattern, shift, new_pattern, 0, xor_pattern, op[7])
        eps0[qubit] = 0

    q.q0 = Fraction(q0_residue, mod_q1)
    return tuple(bit & 1 for bit in eps0), scalar, scalar_half_pow2, q

def _direct_post_replay_payload_matches_state(
    payload: tuple[tuple[int, ...], complex, int, PhaseFunction],
    state: SchurState,
    template: _DirectPostReplayTemplate,
) -> bool:
    eps0, scalar, scalar_half_pow2, q = payload
    if state.m != template.final_m:
        return False
    if tuple(state.eps) != template.final_rows:
        return False
    if tuple(bit & 1 for bit in state.eps0) != eps0:
        return False
    if state.scalar_half_pow2 != scalar_half_pow2:
        return False
    if abs(complex(state.scalar) - complex(scalar)) > 1e-12:
        return False
    return (
        int(state.q.n) == int(q.n)
        and int(state.q.level) == int(q.level)
        and state.q.q0 == q.q0
        and state.q.q1 == q.q1
        and state.q.q2 == q.q2
        and state.q.q3 == q.q3
    )


_LOCAL_IMPLS = {name: globals()[name] for name in _LOCAL_NAMES if name in globals()}
