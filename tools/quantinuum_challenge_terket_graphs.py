"""Generate TerKet diagnostic graphs for the Quantinuum TN challenge bundle."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import cProfile
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction


_NON_GATE_METADATA_FIELDS = {"circuit_name", "family", "hardness", "qubits"}


def _int_field(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(value) if value not in ("", None) else 0


def _input_gate_count(row: dict[str, str]) -> int:
    return sum(_int_field(row, key) for key in row if key not in _NON_GATE_METADATA_FIELDS)


def _support_status(row: dict[str, str]) -> str:
    if _int_field(row, "Measure"):
        return "measure_unsupported"
    if _int_field(row, "CnX"):
        return "cnx_unsupported"
    if _int_field(row, "PauliExpBox"):
        return "angle_path_sum"
    if sum(_int_field(row, key) for key in ("Rz", "Rx", "ZZPhase", "XXPhase")):
        return "angle_path_sum"
    return "terket_exact"


def _pytket_phase_radians(param: object) -> float:
    if isinstance(param, str):
        if param.endswith("/pi"):
            return float(param[:-3])
        return float(Fraction(param)) * math.pi
    return float(param) * math.pi


def _angle_radians(command: dict[str, object]) -> float:
    return _pytket_phase_radians(command["op"].get("params", [0.0])[0])


def _append_rz(gates: list[tuple[object, ...]], qubit: int, angle: float) -> None:
    turns = math.remainder(float(angle) / math.pi, 2.0)
    if math.isclose(turns, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return
    if math.isclose(abs(turns), 1.0, rel_tol=0.0, abs_tol=1e-12):
        gates.append(("z", int(qubit)))
        return
    if math.isclose(turns, 0.5, rel_tol=0.0, abs_tol=1e-12):
        gates.append(("s", int(qubit)))
        return
    if math.isclose(turns, -0.5, rel_tol=0.0, abs_tol=1e-12):
        gates.append(("sdg", int(qubit)))
        return
    gates.append(("pauli_expbox", ("Z",), (int(qubit),), float(angle)))


def _append_rx(gates: list[tuple[object, ...]], qubit: int, angle: float) -> None:
    turns = math.remainder(float(angle) / math.pi, 2.0)
    if math.isclose(turns, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return
    if math.isclose(abs(turns), 1.0, rel_tol=0.0, abs_tol=1e-12):
        gates.append(("x", int(qubit)))
        return
    gates.append(("pauli_expbox", ("X",), (int(qubit),), float(angle)))


def _append_rzz(gates: list[tuple[object, ...]], left: int, right: int, angle: float) -> None:
    gates.append(("pauli_expbox", ("Z", "Z"), (int(left), int(right)), float(angle)))


def _append_rxx(gates: list[tuple[object, ...]], left: int, right: int, angle: float) -> None:
    gates.append(("pauli_expbox", ("X", "X"), (int(left), int(right)), float(angle)))


def _run_status(row: dict[str, str]) -> str:
    status = _support_status(row)
    if status == "angle_path_sum":
        return "terket_exact"
    return status


def _qidx(arg, mapping: dict[tuple[str, tuple[int, ...]], int]) -> int:
    return mapping[(arg[0], tuple(arg[1]))]


def _append_basic_gate(gates: list[tuple[object, ...]], gate_type: str, args: list[int]) -> None:
    if gate_type == "H":
        gates.append(("h", args[0]))
    elif gate_type == "CX":
        gates.append(("cnot", args[0], args[1]))
    elif gate_type == "CnX" and len(args) == 2:
        gates.append(("cnot", args[0], args[1]))
    elif gate_type == "T":
        gates.append(("pauli_expbox", ("Z",), (int(args[0]),), math.pi / 4.0))
    elif gate_type == "Tdg":
        gates.append(("pauli_expbox", ("Z",), (int(args[0]),), -math.pi / 4.0))
    elif gate_type == "S":
        gates.append(("s", args[0]))
    elif gate_type == "Sdg":
        gates.append(("sdg", args[0]))
    elif gate_type == "X":
        gates.append(("x", args[0]))
    elif gate_type == "Z":
        gates.append(("z", args[0]))
    else:
        raise ValueError(f"Unsupported gate {gate_type!r}.")


def _load_pytket_json(path: Path):
    from terket import make_circuit

    data = json.loads(path.read_text(encoding="utf-8"))
    qubits = [(qubit[0], tuple(qubit[1])) for qubit in data["qubits"]]
    mapping = {qubit: idx for idx, qubit in enumerate(qubits)}
    gates = []
    counts = Counter()

    for command in data["commands"]:
        gate_type = command["op"]["type"]
        counts[gate_type] += 1
        if gate_type == "Measure":
            continue
        if gate_type == "Conditional":
            inner_type = command["op"]["conditional"]["op"]["type"]
            qargs = [arg for arg in command.get("args", []) if (arg[0], tuple(arg[1])) in mapping]
            args = [_qidx(arg, mapping) for arg in qargs]
            _append_basic_gate(gates, inner_type, args)
            continue
        args = [_qidx(arg, mapping) for arg in command.get("args", [])]
        if gate_type in {"H", "CX", "CnX", "T", "Tdg", "S", "Sdg", "X", "Z"}:
            _append_basic_gate(gates, gate_type, args)
        elif gate_type == "Rz":
            _append_rz(gates, args[0], _angle_radians(command))
        elif gate_type == "Rx":
            _append_rx(gates, args[0], _angle_radians(command))
        elif gate_type == "ZZPhase":
            _append_rzz(gates, args[0], args[1], _angle_radians(command))
        elif gate_type == "XXPhase":
            _append_rxx(gates, args[0], args[1], _angle_radians(command))
        elif gate_type == "PauliExpBox":
            box = command["op"]["box"]
            paulis = tuple(str(pauli).upper() for pauli in box["paulis"])
            if len(paulis) != len(args):
                raise ValueError(f"PauliExpBox has {len(paulis)} Paulis for {len(args)} args.")
            gates.append(("pauli_expbox", paulis, tuple(args), _pytket_phase_radians(box["phase"])))
        else:
            raise ValueError(f"Unsupported gate {gate_type!r}.")

    return make_circuit(len(qubits), gates, name=path.stem), counts


def _blank_result(
    row: dict[str, str],
    *,
    status: str,
    profile_path: str = "",
) -> dict[str, object]:
    return {
        "circuit_name": row["circuit_name"],
        "family": row["family"],
        "hardness": row["hardness"],
        "status": status,
        "qubits": _int_field(row, "qubits"),
        "input_gates": _input_gate_count(row),
        "terket_gates": "",
        "import_s": "",
        "analyze_s": "",
        "initial_free": "",
        "remaining_free": "",
        "branches": "",
        "cost_model_r": "",
        "cubic_obstruction": "",
        "gauss_obstruction": "",
        "phase3_backend": "",
        "is_zero": "",
        "profile_path": profile_path,
    }


def _run_one(
    json_path: Path,
    *,
    snap_dyadic_level: int | None = None,
    snap_max_error: float | None = None,
    snap_max_total_error: float | None = None,
) -> dict[str, object]:
    from terket import analyze_amplitudes, snap_arbitrary_angles

    start = time.perf_counter()
    spec, counts = _load_pytket_json(json_path)
    if snap_dyadic_level is not None:
        spec = snap_arbitrary_angles(
            spec,
            max_level=snap_dyadic_level,
            max_error=snap_max_error,
            max_total_error=snap_max_total_error,
        )
    import_s = time.perf_counter() - start
    start = time.perf_counter()
    info = analyze_amplitudes(
        spec,
        [0] * spec.n_qubits,
        [[0] * spec.n_qubits],
        allow_tensor_contraction=True,
    )[0]
    analyze_s = time.perf_counter() - start
    return {
        "circuit_name": json_path.stem,
        "status": "ok",
        "qubits": spec.n_qubits,
        "input_gates": sum(counts.values()),
        "terket_gates": len(spec.gates),
        "import_s": import_s,
        "analyze_s": analyze_s,
        "initial_free": info["initial_free"],
        "remaining_free": info["remaining_free"],
        "branches": info["branches"],
        "cost_model_r": info["cost_model_r"],
        "cubic_obstruction": info["cubic_obstruction"],
        "gauss_obstruction": info["gauss_obstruction"],
        "phase3_backend": info["phase3_backend"] or "",
        "is_zero": info["is_zero"],
        "approximation_phase_count": spec.metadata.get("approximation_phase_count", 0),
        "approximation_max_angle_error": spec.metadata.get("approximation_max_angle_error", 0.0),
    }


def _child_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--snap-dyadic-level", type=int, default=None)
    parser.add_argument("--snap-max-error", type=float, default=None)
    parser.add_argument("--snap-max-total-error", type=float, default=None)
    args = parser.parse_args(argv)
    profile_path = os.environ.get("TERKET_PROFILE_PATH")
    if profile_path:
        profiler = cProfile.Profile()
        result = profiler.runcall(
            _run_one,
            args.json_path,
            snap_dyadic_level=args.snap_dyadic_level,
            snap_max_error=args.snap_max_error,
            snap_max_total_error=args.snap_max_total_error,
        )
        Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(profile_path)
    else:
        result = _run_one(
            args.json_path,
            snap_dyadic_level=args.snap_dyadic_level,
            snap_max_error=args.snap_max_error,
            snap_max_total_error=args.snap_max_total_error,
        )
    print(json.dumps(result), flush=True)
    return 0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _child_command(
    json_path: Path,
    *,
    snap_dyadic_level: int | None,
    snap_max_error: float | None,
    snap_max_total_error: float | None,
) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--child", str(json_path)]
    if snap_dyadic_level is not None:
        command.extend(("--snap-dyadic-level", str(snap_dyadic_level)))
    if snap_max_error is not None:
        command.extend(("--snap-max-error", str(snap_max_error)))
    if snap_max_total_error is not None:
        command.extend(("--snap-max-total-error", str(snap_max_total_error)))
    return command


def _svg_bar(path: Path, title: str, labels: list[str], series, *, ylabel: str = "") -> None:
    width, height = 1120, 540
    margin_l, margin_r, margin_t, margin_b = 175, 35, 55, 150
    plot_w, plot_h = width - margin_l - margin_r, height - margin_t - margin_b
    maxv = max([value for _name, values in series for value in values] or [1])
    maxv = max(1.0, float(maxv))
    colors = ["#0f766e", "#b45309", "#334155", "#9f1239"]
    group_w = plot_w / max(1, len(labels))
    bar_w = group_w / (len(series) + 0.6)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf7ef"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Georgia" font-size="22" fill="#1f2937">{html.escape(title)}</text>',
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1):
        y = margin_t + plot_h * (1 - frac)
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" stroke="#e5dccb"/>')
        parts.append(f'<text x="{margin_l - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Consolas" font-size="11" fill="#475569">{maxv * frac:.2g}</text>')
    for series_idx, (name, values) in enumerate(series):
        for label_idx, value in enumerate(values):
            x = margin_l + label_idx * group_w + series_idx * bar_w + bar_w * 0.2
            h = plot_h * (float(value) / maxv)
            y = margin_t + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w * 0.82:.1f}" height="{h:.1f}" fill="{colors[series_idx % len(colors)]}" rx="3"><title>{html.escape(labels[label_idx])}: {html.escape(name)}={value}</title></rect>')
    for label_idx, label in enumerate(labels):
        x = margin_l + label_idx * group_w + group_w / 2
        short = label.replace("_", " ")
        if len(short) > 24:
            short = short[:21] + "..."
        y = height - margin_b + 20
        parts.append(f'<text x="{x:.1f}" y="{y}" transform="rotate(55 {x:.1f} {y})" font-family="Georgia" font-size="12" fill="#334155">{html.escape(short)}</text>')
    legend_x, legend_y = width - margin_r - 225, 55
    for series_idx, (name, _values) in enumerate(series):
        parts.append(f'<rect x="{legend_x}" y="{legend_y + series_idx * 22}" width="14" height="14" fill="{colors[series_idx % len(colors)]}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y + 12 + series_idx * 22}" font-family="Georgia" font-size="13" fill="#334155">{html.escape(name)}</text>')
    if ylabel:
        parts.append(f'<text x="25" y="{height / 2}" transform="rotate(-90 25 {height / 2})" text-anchor="middle" font-family="Georgia" font-size="14" fill="#334155">{html.escape(ylabel)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-dir", type=Path, default=Path("tmp/tn_sim_challenge/challenge_files/attachments"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/tn_sim_challenge"))
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--snap-dyadic-level", type=int, default=None)
    parser.add_argument("--snap-max-error", type=float, default=None)
    parser.add_argument("--snap-max-total-error", type=float, default=None)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.challenge_dir / "metadata.csv"
    circuit_zip_path = args.challenge_dir / "circuit_suite.zip"
    metadata = list(csv.DictReader(metadata_path.open(newline="", encoding="utf-8")))
    family_status: dict[str, Counter[str]] = defaultdict(Counter)
    for row in metadata:
        family_status[row["family"]][_support_status(row)] += 1

    selected = [row for row in metadata if _run_status(row) == "terket_exact"]
    extracted_dir = args.out_dir / "exact_json"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []
    timed_out_families: set[str] = set()

    with zipfile.ZipFile(circuit_zip_path) as archive:
        for row in selected:
            name = row["circuit_name"]
            if row["family"] in timed_out_families:
                result = _blank_result(row, status="skipped_after_family_timeout")
                result_rows.append(result)
                print(f"{name}: {result['status']}", flush=True)
                continue
            source_dir = (
                "pytket_orig"
                if any(_int_field(row, key) for key in ("Rz", "Rx", "ZZPhase", "XXPhase", "PauliExpBox"))
                else "pytket_decomp"
            )
            member = f"circuit_suite/{source_dir}/{name}.json"
            json_path = extracted_dir / f"{name}.json"
            json_path.write_bytes(archive.read(member))
            env = os.environ.copy()
            profile_path = ""
            if args.profile_dir is not None:
                profile_path = str(args.profile_dir / f"{name}.pstats")
                env["TERKET_PROFILE_PATH"] = profile_path
            try:
                proc = subprocess.run(
                    _child_command(
                        json_path,
                        snap_dyadic_level=args.snap_dyadic_level,
                        snap_max_error=args.snap_max_error,
                        snap_max_total_error=args.snap_max_total_error,
                    ),
                    text=True,
                    capture_output=True,
                    timeout=args.timeout_s,
                    env=env,
                )
                if proc.returncode != 0:
                    raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
                result = json.loads(proc.stdout)
                result.update({"family": row["family"], "hardness": row["hardness"]})
                result["profile_path"] = profile_path
            except subprocess.TimeoutExpired:
                result = _blank_result(
                    row,
                    status=f"timeout>{args.timeout_s:g}s",
                    profile_path=profile_path,
                )
                timed_out_families.add(row["family"])
            except RuntimeError as exc:
                result = _blank_result(
                    row,
                    status=f"error:{str(exc).splitlines()[-1][:160]}",
                    profile_path=profile_path,
                )
            result_rows.append(result)
            print(f"{name}: {result['status']}", flush=True)

    fields = [
        "circuit_name", "family", "hardness", "status", "qubits", "input_gates", "terket_gates",
        "import_s", "analyze_s", "initial_free", "remaining_free", "branches", "cost_model_r",
        "cubic_obstruction", "gauss_obstruction", "phase3_backend", "is_zero", "profile_path",
        "approximation_phase_count", "approximation_max_angle_error",
    ]
    normalized_rows = [{field: row.get(field, "") for field in fields} for row in result_rows]
    _write_csv(args.out_dir / "terket_qec_exact_results.csv", normalized_rows)

    support_rows = []
    for family in sorted(family_status):
        counts = family_status[family]
        support_rows.append({
            "family": family,
            "terket_exact": counts.get("terket_exact", 0),
            "angle_path_sum": counts.get("angle_path_sum", 0),
            "pauli_expbox_unsupported": counts.get("pauli_expbox_unsupported", 0),
            "measure_unsupported": counts.get("measure_unsupported", 0),
            "cnx_unsupported": counts.get("cnx_unsupported", 0),
            "total": sum(counts.values()),
        })
    _write_csv(args.out_dir / "terket_challenge_support_summary.csv", support_rows)

    ok_rows = [row for row in normalized_rows if row["status"] == "ok"]
    labels = [str(row["circuit_name"]) for row in ok_rows]
    if ok_rows:
        _svg_bar(
            args.out_dir / "terket_qec_exact_runtime.svg",
            "TerKet on Quantinuum Challenge exact/path-sum subset",
            labels,
            [("analyze seconds", [float(row["analyze_s"]) for row in ok_rows])],
            ylabel="seconds",
        )
        _svg_bar(
            args.out_dir / "terket_qec_exact_obstructions.svg",
            "TerKet exact/path-sum subset solver diagnostics",
            labels,
            [
                ("initial_free", [int(row["initial_free"]) for row in ok_rows]),
                ("gauss_obstruction", [int(row["gauss_obstruction"]) for row in ok_rows]),
                ("cost_model_r", [int(row["cost_model_r"]) for row in ok_rows]),
            ],
            ylabel="count",
        )

    families = [row["family"] for row in support_rows]
    _svg_bar(
        args.out_dir / "terket_challenge_support_by_family.svg",
        "Quantinuum challenge support profile for TerKet exact path",
        families,
        [
            ("terket_exact", [int(row["terket_exact"]) for row in support_rows]),
            ("angle_path_sum", [int(row["angle_path_sum"]) for row in support_rows]),
            ("pauli_expbox_unsupported", [int(row["pauli_expbox_unsupported"]) for row in support_rows]),
            ("measure_unsupported", [int(row["measure_unsupported"]) for row in support_rows]),
            ("cnx_unsupported", [int(row["cnx_unsupported"]) for row in support_rows]),
        ],
        ylabel="circuit count",
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        raise SystemExit(_child_main(sys.argv[2:]))
    raise SystemExit(_main(sys.argv[1:]))
