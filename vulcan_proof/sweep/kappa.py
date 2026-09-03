"""Signal-grid evaluation and the no-signal leakage guard."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..errors import InvariantError, LeakError
from ..params import P, Params
from .common import require_min_seeds, write_json
from .driver import run_point


def kappa_star(table: Iterable[Mapping[str, Any]], params: Params = P) -> dict[str, Any]:
    """Find the first sustained positive lower-bound signal in the grid."""
    del params
    rows = [dict(row) for row in table]
    if not rows:
        raise InvariantError("κ* requires a non-empty table")
    for row in rows:
        if not all(math.isfinite(float(row[name])) for name in ("kappa", "ci_low", "ci_high")):
            raise InvariantError("κ* table contains a non-finite value")
    rows.sort(key=lambda row: float(row["kappa"]))
    significant = [float(row["ci_low"]) > 0.0 for row in rows]
    positive = [position for position, value in enumerate(significant) if value]
    if not positive:
        return {"kappa_star": None, "reason": "not_found", "table": rows}
    first = positive[0]
    if not all(significant[first:]):
        return {"kappa_star": None, "reason": "non_monotone", "table": rows}
    return {"kappa_star": float(rows[first]["kappa"]), "reason": "found", "table": rows}


def find_kappa_star(table: Iterable[Mapping[str, Any]], params: Params = P) -> dict[str, Any]:
    """Compatibility alias for the signal-grid boundary helper."""
    return kappa_star(table, params)


def kappa_zero_guard(
    gain: float,
    orchestration: float,
    params: Params = P,
    gain_ci_low: float | None = None,
) -> None:
    """Reject a supported no-signal gain above the configured leak budget."""
    gain_value = float(gain)
    orchestration_value = float(orchestration)
    lower_bound = gain_value if gain_ci_low is None else float(gain_ci_low)
    if not all(math.isfinite(value) for value in (gain_value, orchestration_value, lower_bound)):
        raise InvariantError("κ=0 guard inputs must be finite")
    limit = float(params["report.kappa0_max_gain_frac"]) * abs(orchestration_value)
    if lower_bound > limit:
        raise LeakError("suspected leak at kappa=0")


def run_kappa_sweep(
    params: Params = P,
    output_root: Path | None = None,
    seeds: Iterable[int] | None = None,
    kappas: Iterable[float] | None = None,
    n_orders: int | None = None,
    parallel: bool = True,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run the configured signal grid and persist its summary artifact."""
    values = tuple(float(value) for value in (params["sim.kappa.grid"] if kappas is None else kappas))
    if not values:
        raise InvariantError("κ sweep requires at least one grid value")
    chosen_seeds = require_min_seeds(
        tuple(
            int(seed)
            for seed in (
                range(1, int(params["run.n_seeds_sweep"]) + 1)
                if seeds is None
                else seeds
            )
        ),
        params,
    )
    point_results: list[dict[str, Any]] = []
    for kappa in values:
        point_results.append(run_point(
            kappa,
            overrides={},
            params=params,
            output_root=output_root,
            seeds=chosen_seeds,
            n_orders=n_orders,
            parallel=parallel,
            include_arm2=True,
            include_arm3=True,
            allow_dirty=allow_dirty,
        ))
    summaries = [result["summary"] for result in point_results]
    zero = next((summary for summary in summaries if float(summary["kappa"]) == 0.0), None)
    if zero is None:
        raise InvariantError("κ grid must include zero for the Phase-4 guard")
    kappa_zero_guard(
        float(zero["arm5_minus_arm4"]["mean"]),
        float(zero["arm4_minus_arm1"]["mean"]),
        params,
        gain_ci_low=float(zero["arm5_minus_arm4"]["ci_low"]),
    )
    table = [
        {
            "kappa": float(summary["kappa"]),
            "arm5_minus_arm4_mean": float(summary["arm5_minus_arm4"]["mean"]),
            "ci_low": float(summary["arm5_minus_arm4"]["ci_low"]),
            "ci_high": float(summary["arm5_minus_arm4"]["ci_high"]),
            "p_positive": float(summary["arm5_minus_arm4"]["p_positive"]),
            "arm4_minus_arm1_mean": float(summary["arm4_minus_arm1"]["mean"]),
            "arm4_minus_arm1_ci_low": float(summary["arm4_minus_arm1"]["ci_low"]),
            "arm4_minus_arm1_ci_high": float(summary["arm4_minus_arm1"]["ci_high"]),
            "lorenz_lift": float(summary["lorenz_lift"]),
        }
        for summary in summaries
    ]
    star = kappa_star(
        ({"kappa": row["kappa"], "ci_low": row["ci_low"], "ci_high": row["ci_high"]} for row in table),
        params,
    )
    star["table"] = table
    star["verdict"] = (
        "ML_CLAIM_SUPPORTED_ABOVE_KAPPA_STAR"
        if star["reason"] == "found"
        else "ML_CLAIM_DROPPED_ORCHESTRATION_ONLY"
    )
    root = Path(output_root).resolve() if output_root is not None else params.path.resolve().parents[1] / "outputs" / "phase4"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "kappa_star.json", star)
    return {"kappa_star": star, "points": point_results, "guard": {"passed": True}}


run = run_kappa_sweep
