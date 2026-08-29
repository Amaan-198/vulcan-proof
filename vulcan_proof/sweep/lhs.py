"""Joint Latin-hypercube sensitivity sweep."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import InvariantError
from ..params import P, Params
from .common import json_value, lhs_design, point_parameter_paths, require_min_seeds, write_json
from .driver import run_point


def run_lhs_sweep(
    params: Params = P,
    output_root: Path | None = None,
    seeds: Iterable[int] | None = None,
    n_orders: int | None = None,
    parallel: bool = True,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run the configured joint LHS points at canonical κ."""
    chosen_seeds = require_min_seeds(
        range(1, int(params["run.n_seeds_sweep"]) + 1) if seeds is None else seeds,
        params,
    )
    paths = point_parameter_paths(int(params["sweep.lhs_max_rank"]), params)
    if not paths:
        root = Path(output_root).resolve() if output_root is not None else params.path.resolve().parents[1] / "outputs" / "phase4"
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "disabled": True,
            "reason": "no parameters qualify for sweep.lhs_max_rank",
            "kappa": float(params["sim.kappa.canonical"]),
            "paths": [],
            "design": [],
            "rows": [],
            "fractions": {},
            "scatter": [],
        }
        write_json(root / "lhs.json", payload)
        return {**payload, "points": []}
    design = lhs_design(paths, params)
    kappa = float(params["sim.kappa.canonical"])
    rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for values in np.asarray(design, dtype="float64"):
        overrides = {path: float(value) for path, value in zip(paths, values, strict=True)}
        result = run_point(
            kappa,
            overrides=overrides,
            params=params,
            output_root=output_root,
            seeds=chosen_seeds,
            n_orders=n_orders,
            parallel=parallel,
            allow_dirty=allow_dirty,
        )
        points.append(result)
        summary = result["summary"]
        rows.append({
            "point": len(rows),
            "overrides": json_value(overrides),
            "arm5_minus_arm4": summary["arm5_minus_arm4"],
            "arm4_minus_arm1": summary["arm4_minus_arm1"],
        })
    if not rows:
        raise InvariantError("LHS produced no points")
    positive = sum(float(row["arm5_minus_arm4"]["ci_low"]) > 0.0 for row in rows)
    negative = sum(float(row["arm5_minus_arm4"]["ci_high"]) < 0.0 for row in rows)
    fractions = {
        "ci_low_positive": float(positive / len(rows)),
        "ci_high_negative": float(negative / len(rows)),
    }
    scatter = []
    for row in rows:
        overrides = row["overrides"]
        scatter.append({
            "mean_difference": float(row["arm5_minus_arm4"]["mean"]),
            "uplift_multiplier": overrides["uplift_true.sweep_multiplier"],
            "hourly_rate": overrides["econ.hourly_rate"],
        })
    root = Path(output_root).resolve() if output_root is not None else params.path.resolve().parents[1] / "outputs" / "phase4"
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "kappa": kappa,
        "paths": list(paths),
        "design": design,
        "rows": rows,
        "fractions": fractions,
        "scatter": scatter,
    }
    write_json(root / "lhs.json", payload)
    return {"kappa": kappa, "paths": paths, "design": design, "rows": rows, "fractions": fractions, "scatter": scatter, "points": points}


run = run_lhs_sweep
