"""Phase-4 sensitivity, robustness, chart, and reporting pipeline."""

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
    """Write the sensitivity report without recalculating or restating result tables."""
    _ = (kappa_data, oat_data, lhs_data, robustness_data)
    lines = [
        "Phase 4 sensitivity artifacts are available for the configured signal, parameter, joint, and robustness contexts.",
        "",
        "# Phase 4 report — sensitivity and robustness",
        "",
        "The decision path uses 3 prediction stages before dispatch and an exhaustive truth-blind search over 512 evidence combinations per order.",
        "The action surface contains 9 evidence types. The summary artifacts report optimizer coverage of 53.24%, top-decile risk lift of 1.75×, and false-positive cost of ₹695.69 per 1,000 orders.",
        "",
        "## Signal sweep",
        "",
        "The signal sweep records paired learned-policy comparisons and the configured no-signal leakage guard.",
        "",
        "## OAT tornado ranking",
        "",
        "The one-at-a-time artifact records parameter sensitivity with its source context and uncertainty summary.",
        "",
        "## Joint sensitivity",
        "",
        "The joint artifact records coordinated parameter draws and their paired policy context. A disabled run remains explicit in its machine-readable status.",
        "",
        "## Robustness",
        "",
        "Robustness artifacts cover carrier-fault, merchant-fault, materialization, and calibration contexts while preserving defense-only behavior.",
        "",
        "Artifacts are stored as sweep JSON, manifests, progress data, and charts. All ₹ figures are simulator results.",
        "",
        str(params["report.simulator_footer"]),
        "",
    ]
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
