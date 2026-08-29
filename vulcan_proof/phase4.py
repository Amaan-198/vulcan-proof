"""Phase-4 κ, OAT, LHS, robustness, chart, and verdict pipeline."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable
from typing import Any

from .errors import InvariantError
from .params import P, Params
from .sweep.charts import generate_all_charts
from .sweep.common import require_min_seeds, write_json
from .sweep.kappa import run_kappa_sweep
from .sweep.lhs import run_lhs_sweep
from .sweep.oat import run_oat_sweep
from .sweep.robustness import run_robustness_sweeps


def _average_diagnostic(summary: dict[str, Any], arm: str, path: tuple[str, ...]) -> float:
    """Average one nested diagnostic over the point's seeds."""
    values: list[float] = []
    for diagnostics in summary["diagnostics"].values():
        value: Any = diagnostics[arm]
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(sum(values) / len(values))


def _chart_data(kappa_result: dict[str, Any], oat_result: dict[str, Any], lhs_result: dict[str, Any], robustness_result: dict[str, Any], params: Params) -> dict[str, Any]:
    """Assemble the compact data contract consumed by all chart factories."""
    kappa_points = kappa_result["points"]
    central = min(
        kappa_points,
        key=lambda point: abs(float(point["summary"]["kappa"]) - float(params["sim.kappa.canonical"])),
    )
    central_summary = central["summary"]
    defense_rows = []
    for arm in ("arm1", "arm5"):
        defense_rows.append({
            "arm": arm,
            "headline": _average_diagnostic(central_summary, arm, ("defense_only_win_rate", "correct_fulfillment")),
            "carrier_fault": _average_diagnostic(central_summary, arm, ("defense_only_win_rate", "carrier_fault")),
        })
    split_rows = []
    coverage_rows = []
    for arm in ("arm4", "arm5"):
        split_rows.append({
            "arm": arm,
            "prevention": _average_diagnostic(central_summary, arm, ("prevention_rate",)),
            "defence": _average_diagnostic(central_summary, arm, ("defence_rate",)),
        })
        coverage_rows.append({
            "arm": arm,
            "coverage_any": _average_diagnostic(central_summary, arm, ("coverage_any",)),
            "otp_requested_pct": _average_diagnostic(central_summary, arm, ("friction", "otp_requested_pct")),
        })
    boundary_rows = [
        {"x": row["value"], "boundaries": row["boundaries"]}
        for row in oat_result["rows"]
    ]
    return {
        "kappa_table": kappa_result["kappa_star"]["table"],
        "oat_rows": oat_result["rows"],
        "lhs_scatter": lhs_result["scatter"],
        "boundary_rows": boundary_rows,
        "defense_rows": defense_rows,
        "split_rows": split_rows,
        "coverage_rows": coverage_rows,
        "lorenz_rows": [
            {
                "kappa": point["summary"]["kappa"],
                "deciles": point["summary"]["lorenz"]["deciles"],
            }
            for point in kappa_points
        ],
        "robustness_rows": robustness_result["rows"],
    }


def _write_report(path: pathlib.Path, params: Params, kappa_data: dict[str, Any], oat_data: dict[str, Any], lhs_data: dict[str, Any], robustness_data: dict[str, Any]) -> None:
    """Write the judge-facing Phase-4 report with the verdict first."""
    star = kappa_data["kappa_star"]
    table = star["table"]
    if star["reason"] == "found":
        row = next(row for row in table if float(row["kappa"]) == float(star["kappa_star"]))
        zero = next(row for row in table if float(row["kappa"]) == 0.0)
        achievable = float(zero["arm5_minus_arm4_mean"]) + float(zero["arm4_minus_arm1_mean"])
        captured = 0.0 if achievable == 0.0 else 100.0 * float(zero["arm4_minus_arm1_mean"]) / achievable
        verdict = (
            f"κ* = {float(star['kappa_star']):g}. Above this signal strength the optimizer beats the tuned rule by "
            f"{float(row['arm5_minus_arm4_mean']):.3f} ₹/1,000 (95% CI [{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]). "
            f"At κ = 0 the tuned rule captures {captured:.1f}% of achievable value."
        )
    else:
        zero = next(row for row in table if float(row["kappa"]) == 0.0)
        verdict = (
            f"κ* not found on [0, 1]. The ML claim is dropped; the orchestration layer (Arm 4 − Arm 1 = "
            f"{float(zero['arm4_minus_arm1_mean']):.3f} ₹/1,000, CI [{float(zero['arm4_minus_arm1_ci_low']):.3f}, "
            f"{float(zero['arm4_minus_arm1_ci_high']):.3f}]) is the product."
        )
    lines = [verdict, "", "# Phase 4 report — κ sweep, sensitivity, and kill condition", ""]
    lines.extend([
        f"Verdict code: `{star['verdict']}`; κ* reason: `{star['reason']}`.",
        "",
        "## κ sweep",
        "",
        "| κ | Arm 5 − Arm 4 mean | 95% CI | Arm 4 − Arm 1 | Lorenz lift |",
        "|---:|---:|---|---:|---:|",
    ])
    for row in table:
        lines.append(
            f"| {float(row['kappa']):g} | {float(row['arm5_minus_arm4_mean']):.3f} | "
            f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}] | "
            f"{float(row['arm4_minus_arm1_mean']):.3f} | {float(row['lorenz_lift']):.3f} |"
        )
    lines.extend(["", "The κ = 0 guard passed.", "", "## OAT tornado ranking", "", "| Rank | Parameter | Low Δ | High Δ |", "|---:|---|---:|---:|"])
    oat_rows = sorted(oat_data["rows"], key=lambda row: (int(row["rank"]), str(row["parameter"]), str(row["level"])))
    grouped: dict[str, dict[str, Any]] = {}
    for row in oat_rows:
        parameter = str(row["parameter"])
        if parameter not in grouped:
            grouped[parameter] = {"rank": row["rank"]}
        grouped[parameter][str(row["level"])] = row["arm5_minus_arm4"]["mean"]
    for parameter, row in grouped.items():
        lines.append(f"| {int(row['rank'])} | {parameter} | {float(row['lo']):.3f} | {float(row['hi']):.3f} |")
    if oat_data.get("disabled"):
        lines.extend(["", f"Disabled by configuration: {oat_data['reason']}."])
    lines.extend(["", "## LHS", ""])
    if lhs_data.get("disabled"):
        lines.extend([
            f"Disabled by configuration: {lhs_data['reason']}.",
            "Fraction with CI_low > 0: not applicable (disabled).",
            "Fraction with CI_high < 0: not applicable (disabled).",
        ])
    else:
        lines.extend([
            f"Fraction with CI_low > 0: {float(lhs_data['fractions']['ci_low_positive']):.3f}.",
            f"Fraction with CI_high < 0: {float(lhs_data['fractions']['ci_high_negative']):.3f}.",
        ])
    lines.extend([
        "",
        "## Robustness",
        "",
        "| World | Arm 5 − Arm 4 | Carrier-fault Arm 1 | Carrier-fault Arm 5 | Stage-A ECE |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in robustness_data["rows"]:
        ece = row.get("stage_a_ece_test", "—")
        ece_text = "—" if isinstance(ece, str) else f"{float(ece):.6f}"
        lines.append(
            f"| {row['name']} | {float(row['arm5_minus_arm4']['mean']):.3f} | "
            f"{float(row['carrier_fault_win_rate_arm1']):.6f} | {float(row['carrier_fault_win_rate_arm5']):.6f} | {ece_text} |"
        )
    lines.extend(["", "Artifacts: `outputs/phase4/kappa_star.json`, `oat.json`, `lhs.json`, `robustness.json`, and nine PNG charts.", "", "All ₹ figures are simulator results.", "", str(params["report.simulator_footer"]), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase4(
    params: Params = P,
    output_root: pathlib.Path | None = None,
    seeds: Iterable[int] | None = None,
    kappas: Iterable[float] | None = None,
    n_orders: int | None = None,
    parallel: bool = True,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run all Phase-4 sweeps, write charts/report, and enforce no param edits."""
    before = params.path.read_bytes()
    chosen_seeds = require_min_seeds(
        range(1, int(params["run.n_seeds_sweep"]) + 1) if seeds is None else seeds,
        params,
    )
    root = pathlib.Path(output_root).resolve() if output_root is not None else params.path.resolve().parents[1] / "outputs" / "phase4"
    root.mkdir(parents=True, exist_ok=True)
    kappa_data = run_kappa_sweep(params, root, chosen_seeds, kappas, n_orders, parallel, allow_dirty)
    oat_data = run_oat_sweep(params, root, chosen_seeds, n_orders, parallel, allow_dirty)
    lhs_data = run_lhs_sweep(params, root, chosen_seeds, n_orders, parallel, allow_dirty)
    robustness_data = run_robustness_sweeps(params, root, chosen_seeds, n_orders, parallel, allow_dirty)
    chart_data = _chart_data(kappa_data, oat_data, lhs_data, robustness_data, params)
    chart_paths = generate_all_charts(chart_data, root, params)
    write_json(root / "progress.json", {
        "kappa_points": len(kappa_data["points"]),
        "oat_points": len(oat_data["points"]),
        "lhs_points": len(lhs_data["points"]),
        "robustness_points": len(robustness_data["points"]),
        "completed_points": [str(point["point_dir"]) for point in (
            kappa_data["points"] + oat_data["points"] + lhs_data["points"] + robustness_data["points"]
        )],
        "charts": [str(path) for path in chart_paths],
    })
    _write_report(root.parent / "phase4_REPORT.md", params, kappa_data, oat_data, lhs_data, robustness_data)
    if params.path.read_bytes() != before:
        raise InvariantError("params.yaml changed during Phase 4")
    return {
        "kappa": kappa_data,
        "oat": oat_data,
        "lhs": lhs_data,
        "robustness": robustness_data,
        "charts": chart_paths,
        "report": root.parent / "phase4_REPORT.md",
    }


run = run_phase4
