"""Build comparison/profile artifacts for TerKet Quantinuum challenge runs."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path
import pstats
from statistics import median


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _public_metrics(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for path in root.glob("*/*/METRICS.csv"):
        submission = "/".join(path.parts[-3:-1])
        for row in _read_csv(path):
            runtime = row.get("total_runtime", "")
            if not runtime:
                continue
            try:
                value = float(runtime)
            except ValueError:
                continue
            circuit = row["circuit_name"]
            metrics.setdefault(circuit, {})[submission] = {
                "total_runtime": value,
                "mirror_fidelity": float(row["mirror_fidelity"]) if row.get("mirror_fidelity") else math.nan,
                "fidelity_estimate": float(row["fidelity_estimate"]) if row.get("fidelity_estimate") else math.nan,
            }
    return metrics


def _metadata(path: Path) -> dict[str, dict[str, str]]:
    return {row["circuit_name"]: row for row in _read_csv(path)}


def _safe_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _log10(value: float) -> float:
    return math.log10(max(value, 1e-12))


def _svg_log_scatter(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 980, 680
    ml, mr, mt, mb = 90, 35, 55, 95
    plot_w, plot_h = width - ml - mr, height - mt - mb
    xs = [_log10(float(row["public_best_total_runtime_s"])) for row in rows]
    ys = [_log10(float(row["terket_analyze_s"])) for row in rows]
    lo = math.floor(min(xs + ys))
    hi = math.ceil(max(xs + ys))
    span = max(1, hi - lo)
    colors = {
        "condensed_matter": "#0f766e",
        "mvsp": "#b45309",
        "qec_non_ft": "#334155",
    }

    def px(v: float) -> float:
        return ml + (v - lo) / span * plot_w

    def py(v: float) -> float:
        return mt + (hi - v) / span * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf7ef"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Georgia" font-size="22" fill="#1f2937">TerKet vs public best runtime</text>',
    ]
    for tick in range(lo, hi + 1):
        x = px(tick)
        y = py(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{height-mb}" stroke="#eadfce"/>')
        parts.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{width-mr}" y2="{y:.1f}" stroke="#eadfce"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-mb+24}" text-anchor="middle" font-family="Consolas" font-size="12" fill="#475569">1e{tick}</text>')
        parts.append(f'<text x="{ml-12}" y="{y+4:.1f}" text-anchor="end" font-family="Consolas" font-size="12" fill="#475569">1e{tick}</text>')
    parts.append(f'<line x1="{px(lo):.1f}" y1="{py(lo):.1f}" x2="{px(hi):.1f}" y2="{py(hi):.1f}" stroke="#9f1239" stroke-width="2" stroke-dasharray="6 5"/>')
    for row, x, y in zip(rows, xs, ys):
        family = str(row["family"])
        color = colors.get(family, "#64748b")
        title = (
            f'{row["circuit_name"]}: TerKet={float(row["terket_analyze_s"]):.4g}s, '
            f'public_best={float(row["public_best_total_runtime_s"]):.4g}s, '
            f'ratio={float(row["terket_over_public_best"]):.4g}'
        )
        parts.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="5" fill="{color}" opacity="0.85"><title>{html.escape(title)}</title></circle>')
    parts.append(f'<text x="{width/2}" y="{height-35}" text-anchor="middle" font-family="Georgia" font-size="15" fill="#334155">public best total runtime (s, log10)</text>')
    parts.append(f'<text x="25" y="{height/2}" transform="rotate(-90 25 {height/2})" text-anchor="middle" font-family="Georgia" font-size="15" fill="#334155">TerKet analyze runtime (s, log10)</text>')
    legend_x, legend_y = width - 210, 72
    for idx, (family, color) in enumerate(colors.items()):
        parts.append(f'<circle cx="{legend_x}" cy="{legend_y+idx*23}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{legend_x+16}" y="{legend_y+idx*23+5}" font-family="Georgia" font-size="13" fill="#334155">{html.escape(family)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _svg_ratio_by_family(path: Path, rows: list[dict[str, object]]) -> None:
    families = sorted({str(row["family"]) for row in rows})
    stats = []
    for family in families:
        ratios = sorted(float(row["terket_over_public_best"]) for row in rows if row["family"] == family)
        faster = sum(1 for ratio in ratios if ratio < 1.0)
        stats.append((family, min(ratios), median(ratios), max(ratios), faster, len(ratios)))
    width, height = 860, 430
    ml, mr, mt, mb = 95, 35, 55, 85
    plot_w, plot_h = width - ml - mr, height - mt - mb
    max_log = max(1.0, max(abs(_log10(v)) for _f, mn, md, mx, _fa, _n in stats for v in (mn, md, mx)))
    zero_y = mt + plot_h / 2

    def y_for(ratio: float) -> float:
        return zero_y - (_log10(ratio) / max_log) * (plot_h / 2)

    group_w = plot_w / max(1, len(stats))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbf7ef"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Georgia" font-size="22" fill="#1f2937">TerKet / public best runtime ratio</text>',
        f'<line x1="{ml}" y1="{zero_y:.1f}" x2="{width-mr}" y2="{zero_y:.1f}" stroke="#9f1239" stroke-width="2" stroke-dasharray="5 5"/>',
    ]
    for i, (family, mn, md, mx, faster, count) in enumerate(stats):
        x = ml + i * group_w + group_w / 2
        parts.append(f'<line x1="{x:.1f}" y1="{y_for(mn):.1f}" x2="{x:.1f}" y2="{y_for(mx):.1f}" stroke="#334155" stroke-width="3"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y_for(md):.1f}" r="8" fill="#0f766e"><title>{html.escape(family)} median={md:.4g}, min={mn:.4g}, max={mx:.4g}</title></circle>')
        parts.append(f'<text x="{x:.1f}" y="{height-mb+25}" text-anchor="middle" font-family="Georgia" font-size="13" fill="#334155">{html.escape(family)}</text>')
        parts.append(f'<text x="{x:.1f}" y="{height-mb+43}" text-anchor="middle" font-family="Consolas" font-size="11" fill="#475569">{faster}/{count} faster</text>')
    parts.append(f'<text x="{ml-12}" y="{zero_y+4:.1f}" text-anchor="end" font-family="Consolas" font-size="12" fill="#475569">1x</text>')
    parts.append(f'<text x="25" y="{height/2}" transform="rotate(-90 25 {height/2})" text-anchor="middle" font-family="Georgia" font-size="15" fill="#334155">log ratio; lower is faster</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _profile_summary(profile_dir: Path, result_rows: list[dict[str, str]], limit: int = 8) -> list[dict[str, object]]:
    rows = []
    by_name = {row["circuit_name"]: row for row in result_rows}
    for profile in sorted(profile_dir.glob("*.pstats")):
        circuit = profile.stem
        stats = pstats.Stats(str(profile))
        entries = []
        for (filename, line, func), data in stats.stats.items():
            cc, nc, tt, ct, _callers = data
            entries.append((ct, tt, nc, f"{Path(filename).name}:{line}:{func}"))
        entries.sort(reverse=True)
        for rank, (ct, tt, nc, func) in enumerate(entries[:limit], start=1):
            rows.append({
                "circuit_name": circuit,
                "family": by_name.get(circuit, {}).get("family", ""),
                "status": by_name.get(circuit, {}).get("status", ""),
                "rank": rank,
                "cumtime_s": ct,
                "tottime_s": tt,
                "calls": nc,
                "function": func,
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/tn_sim_challenge"))
    parser.add_argument("--results-root", type=Path, default=Path("tmp/public_tn_sim_challenge_results"))
    parser.add_argument("--metadata", type=Path, default=Path("tmp/tn_sim_challenge/challenge_files/attachments/metadata.csv"))
    parser.add_argument("--profile-dir", type=Path, default=Path("results/tn_sim_challenge/profiles_full_20260518"))
    args = parser.parse_args(argv)

    terket_rows = _read_csv(args.out_dir / "terket_qec_exact_results.csv")
    public = _public_metrics(args.results_root)
    meta = _metadata(args.metadata)
    compare_rows: list[dict[str, object]] = []
    for row in terket_rows:
        runtime = _safe_float(row.get("analyze_s"))
        if row.get("status") != "ok" or runtime is None:
            continue
        circuit = row["circuit_name"]
        public_rows = public.get(circuit, {})
        if not public_rows:
            continue
        ranked = sorted(public_rows.items(), key=lambda item: item[1]["total_runtime"])
        best_submission, best = ranked[0]
        compare_rows.append({
            "circuit_name": circuit,
            "family": row.get("family") or meta.get(circuit, {}).get("family", ""),
            "hardness": row.get("hardness") or meta.get(circuit, {}).get("hardness", ""),
            "terket_analyze_s": runtime,
            "terket_backend": row.get("phase3_backend", ""),
            "public_best_submission": best_submission,
            "public_best_total_runtime_s": best["total_runtime"],
            "public_best_mirror_fidelity": best["mirror_fidelity"],
            "terket_over_public_best": runtime / best["total_runtime"],
            "public_all_top4": "; ".join(f"{name}:{data['total_runtime']:.6g}" for name, data in ranked[:4]),
        })
    compare_rows.sort(key=lambda row: (str(row["family"]), float(row["terket_over_public_best"])))
    _write_csv(args.out_dir / "terket_vs_public_runtime.csv", compare_rows)
    _svg_log_scatter(args.out_dir / "terket_vs_public_runtime_scatter.svg", compare_rows)
    _svg_ratio_by_family(args.out_dir / "terket_vs_public_runtime_ratio_by_family.svg", compare_rows)

    if args.profile_dir.exists():
        profile_rows = _profile_summary(args.profile_dir, terket_rows)
        _write_csv(args.out_dir / "terket_profile_top_functions.csv", profile_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
