"""Approximate tensor-network contraction for q3-free partition sums."""

from __future__ import annotations

import cmath
from typing import Any
import warnings

import numpy as np

from ..native import _get_quimb_tensor_module
from ..scaling import ScaledComplex, _make_scaled_complex, _omega_table, _scale_scaled_complex, _scaled_from_complex_log
from ..state import SolverConfig, _get_solver_config
from .approx_residue import _sum_q3_free_residue_forest_scaled
from .approx_mps import _sum_q3_free_boundary_mps_configured_scaled


__all__ = ["_q3_free_partition_tensor_network", "_sum_q3_free_residue_sample_scaled", "_sum_q3_free_residue_forest_scaled", "_sum_q3_free_bethe_bp_scaled", "_sum_q3_free_approx_tensor_scaled", "_clear_q3_free_approx_diagnostics", "_get_q3_free_approx_diagnostics"]


def _clear_q3_free_approx_diagnostics() -> None:
    from .approx_guard import _clear_q3_free_approx_diagnostics as clear
    clear()


def _get_q3_free_approx_diagnostics() -> dict[str, Any] | None:
    from .approx_guard import _get_q3_free_approx_diagnostics as get
    return get()


def _q3_free_partition_tensor_network(q, *, config: SolverConfig | None = None) -> tuple[Any, complex] | None:
    """Build a factor-graph tensor network for a q3-free partition sum."""
    if q.q3:
        return None
    cfg = _get_solver_config() if config is None else config
    if q.n > int(cfg.approx_tensor_max_vars):
        return None

    qtn = _get_quimb_tensor_module()
    if qtn is None:
        return None

    incident_inds: list[list[str]] = [[] for _ in range(q.n)]
    edge_specs: list[tuple[int, int, int, str, str]] = []
    q2_lift = q.mod_q1 // q.mod_q2 if q.mod_q2 else 0
    for edge_idx, ((left, right), coeff) in enumerate(sorted(q.q2.items())):
        left = int(left)
        right = int(right)
        left_ind = f"q3f_e{edge_idx}_l"
        right_ind = f"q3f_e{edge_idx}_r"
        incident_inds[left].append(left_ind)
        incident_inds[right].append(right_ind)
        residue = (q2_lift * int(coeff)) % q.mod_q1
        edge_specs.append((left, right, residue, left_ind, right_ind))

    if any(len(inds) > int(cfg.approx_tensor_max_degree) for inds in incident_inds):
        return None

    omega = _omega_table(q.level)
    tensors = []
    scalar = cmath.exp(2j * cmath.pi * float(q.q0))

    for var, inds in enumerate(incident_inds):
        unary = omega[int(q.q1[var]) % q.mod_q1]
        degree = len(inds)
        if degree == 0:
            scalar *= 1.0 + unary
            continue
        if degree == 1:
            data = np.asarray([1.0 + 0j, unary], dtype=np.complex128)
        else:
            data = np.zeros((2,) * degree, dtype=np.complex128)
            data[(0,) * degree] = 1.0 + 0j
            data[(1,) * degree] = unary
        tensors.append(qtn.Tensor(data=data, inds=tuple(inds), tags=(f"VAR{var}",)))

    for left, right, residue, left_ind, right_ind in edge_specs:
        phase = omega[residue]
        data = np.asarray([[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, phase]], dtype=np.complex128)
        tensors.append(qtn.Tensor(data=data, inds=(left_ind, right_ind), tags=(f"EDGE{left}_{right}",)))

    return qtn.TensorNetwork(tensors, virtual=True), scalar


def _sum_q3_free_approx_tensor_scaled(
    q,
    *,
    config: SolverConfig | None = None,
) -> ScaledComplex | None:
    """Approximate a q3-free partition sum via quimb compressed contraction."""
    cfg = _get_solver_config() if config is None else config
    if not bool(cfg.approx_q3_free_tensor):
        return None
    method = str(cfg.approx_tensor_method).strip().lower()
    if method in {"residue_forest", "forest_residue", "rb_residue", "rao_blackwell"}:
        from .approx_guard import _sum_q3_free_residue_forest_checked_scaled

        return _sum_q3_free_residue_forest_checked_scaled(q, config=cfg)
    if method in {"residue", "residue_sample", "histogram_sample", "sample"}:
        return _sum_q3_free_residue_sample_scaled(q, config=cfg)
    if method in {"bp", "bethe", "belief-propagation", "belief_propagation"}:
        return _sum_q3_free_bethe_bp_scaled(q, config=cfg)
    if method in {"boundary_mps", "bounded_bond", "mps"}:
        return _sum_q3_free_boundary_mps_configured_scaled(q, cfg)
    if method not in {"compressed", "contract_compressed", "quimb"}:
        raise ValueError(
            "approx_tensor_method must be one of 'residue_forest', 'residue_sample', 'bp', or 'compressed'; "
            f"received {cfg.approx_tensor_method!r}."
        )

    built = _q3_free_partition_tensor_network(q, config=cfg)
    if built is None:
        return None
    tn, scalar = built
    if len(tn.tensor_map) == 0:
        return _make_scaled_complex(scalar)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The contraction tree is not a compressed one.*",
                category=UserWarning,
            )
            value = tn.contract_compressed(str(cfg.approx_tensor_optimize), max_bond=int(cfg.approx_tensor_max_bond), cutoff=float(cfg.approx_tensor_cutoff))
    except Exception:
        return None
    return _make_scaled_complex(scalar * complex(value))


def _sum_q3_free_residue_sample_scaled(
    q,
    *,
    config: SolverConfig | None = None,
) -> ScaledComplex | None:
    """Estimate q3-free sum by sampling nonnegative residue counts."""
    if q.q3:
        return None
    cfg = _get_solver_config() if config is None else config
    if q.n > int(cfg.approx_tensor_max_vars):
        return None
    if any(len_neighbors > int(cfg.approx_tensor_max_degree) for len_neighbors in _q3_free_degrees(q)):
        return None

    sample_count = max(1, int(cfg.approx_tensor_residue_samples))
    batch_size = max(1, int(cfg.approx_tensor_residue_batch))
    modulus = int(q.mod_q1)
    omega = np.asarray(_omega_table(q.level), dtype=np.complex128)
    q1 = np.asarray(q.q1, dtype=np.int64) % modulus
    q2_lift = q.mod_q1 // q.mod_q2 if q.mod_q2 else 0

    if q.q2:
        edge_left = np.fromiter((int(left) for left, _right in q.q2), dtype=np.int64, count=len(q.q2))
        edge_right = np.fromiter((int(right) for _left, right in q.q2), dtype=np.int64, count=len(q.q2))
        edge_residue = np.fromiter(
            ((q2_lift * int(coeff)) % modulus for coeff in q.q2.values()),
            dtype=np.int64,
            count=len(q.q2),
        )
    else:
        edge_left = edge_right = edge_residue = np.asarray([], dtype=np.int64)

    rng = np.random.default_rng(int(cfg.approx_tensor_residue_seed))
    phase_sum = 0.0 + 0.0j
    remaining = sample_count
    while remaining:
        size = min(batch_size, remaining)
        bits = rng.integers(0, 2, size=(size, q.n), dtype=np.uint8)
        residues = (bits.astype(np.int64) @ q1) % modulus
        if edge_residue.size:
            active_edges = (bits[:, edge_left] & bits[:, edge_right]).astype(np.int64, copy=False)
            residues = (residues + active_edges @ edge_residue) % modulus
        phase_sum += complex(np.sum(omega[residues]))
        remaining -= size

    mean_phase = phase_sum / float(sample_count)
    scalar = cmath.exp(2j * cmath.pi * float(q.q0))
    return _scale_scaled_complex(_make_scaled_complex(scalar * mean_phase), 2 * int(q.n))


def _q3_free_degrees(q) -> tuple[int, ...]:
    degrees = [0] * int(q.n)
    for left, right in q.q2:
        degrees[int(left)] += 1
        degrees[int(right)] += 1
    return tuple(degrees)


def _normalize_bp_message(message: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(message)))
    if not np.isfinite(scale) or scale == 0.0:
        return np.asarray([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)
    return np.asarray(message / scale, dtype=np.complex128)


def _complex_log(value: complex) -> complex | None:
    if value == 0j:
        return None
    return cmath.log(value)


def _sum_q3_free_bethe_bp_scaled(
    q,
    *,
    config: SolverConfig | None = None,
) -> ScaledComplex | None:
    """Approximate q3-free partition sum by loopy BP / Bethe TN contraction."""
    if q.q3:
        return None
    cfg = _get_solver_config() if config is None else config
    if q.n > int(cfg.approx_tensor_max_vars):
        return None

    omega = _omega_table(q.level)
    unary = np.empty((q.n, 2), dtype=np.complex128)
    unary[:, 0] = 1.0 + 0j
    for var in range(q.n):
        unary[var, 1] = omega[int(q.q1[var]) % q.mod_q1]

    q2_lift = q.mod_q1 // q.mod_q2 if q.mod_q2 else 0
    adjacency: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(q.n)]
    for (left, right), coeff in sorted(q.q2.items()):
        left = int(left)
        right = int(right)
        residue = (q2_lift * int(coeff)) % q.mod_q1
        phase = omega[residue]
        factor = np.asarray([[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, phase]], dtype=np.complex128)
        adjacency[left].append((right, factor))
        adjacency[right].append((left, factor.T.copy()))

    if any(len(neighbors) > int(cfg.approx_tensor_max_degree) for neighbors in adjacency):
        return None

    messages: dict[tuple[int, int], np.ndarray] = {}
    for var, neighbors in enumerate(adjacency):
        for neighbor, _factor in neighbors:
            messages[(var, neighbor)] = np.asarray([1.0 + 0j, 1.0 + 0j], dtype=np.complex128)

    damping = min(max(float(cfg.approx_tensor_bp_damping), 0.0), 1.0)
    max_iters = max(1, int(cfg.approx_tensor_bp_max_iters))
    tol = max(0.0, float(cfg.approx_tensor_bp_tol))

    for _iteration in range(max_iters):
        next_messages: dict[tuple[int, int], np.ndarray] = {}
        max_delta = 0.0
        for var, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            full_product = unary[var].copy()
            for neighbor, _factor in neighbors:
                full_product *= messages[(neighbor, var)]
            for neighbor, factor in neighbors:
                incoming = full_product / messages[(neighbor, var)]
                updated = incoming @ factor
                updated = _normalize_bp_message(updated)
                old = messages[(var, neighbor)]
                if damping:
                    updated = _normalize_bp_message((1.0 - damping) * updated + damping * old)
                next_messages[(var, neighbor)] = updated
                max_delta = max(max_delta, float(np.max(np.abs(updated - old))))
        messages = next_messages
        if max_delta <= tol:
            break

    log_z = cmath.log(cmath.exp(2j * cmath.pi * float(q.q0)))
    log_z_i: list[complex | None] = [None] * q.n
    for var, neighbors in enumerate(adjacency):
        local = unary[var].copy()
        for neighbor, _factor in neighbors:
            local *= messages[(neighbor, var)]
        local_sum = complex(local[0] + local[1])
        local_log = _complex_log(local_sum)
        if local_log is None:
            return _make_scaled_complex(0j)
        log_z_i[var] = local_log
        log_z -= (len(neighbors) - 1) * local_log

    for left, neighbors in enumerate(adjacency):
        left_base = unary[left].copy()
        for neighbor, _factor in neighbors:
            left_base *= messages[(neighbor, left)]
        for right, factor in neighbors:
            if left >= right:
                continue
            right_neighbors = adjacency[right]
            right_base = unary[right].copy()
            for neighbor, _factor in right_neighbors:
                right_base *= messages[(neighbor, right)]
            left_cavity = left_base / messages[(right, left)]
            right_cavity = right_base / messages[(left, right)]
            edge_sum = complex(left_cavity @ factor @ right_cavity)
            edge_log = _complex_log(edge_sum)
            if edge_log is None:
                return _make_scaled_complex(0j)
            log_z += edge_log

    return _scaled_from_complex_log(log_z)
