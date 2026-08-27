"""One-at-a-time sensitivity sweep."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .. import ev_reference
from ..errors import InvariantError
from ..params import P, Params
from .common import json_value, oat_levels, point_parameter_paths, require_min_seeds, write_json
from .driver import run_point


def _boundary_table() -> dict[str, float]:
    """Return the frozen judge-facing recommendation boundaries."""
    return {
        "otp_electronics": float(ev_reference.threshold("otp", "Electronics")),
        "packing_apparel": float(ev_reference.threshold("packing", "Apparel")),
    }


def run_oat_sweep(
    params: Params = P,
    output_root: Path | None = None,
    seeds: Iterable[int] | None = None,
    n_orders: int | None = None,
    parallel: bool = True,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run all rank-qualified OAT points at canonical κ."""
    chosen_seeds = require_min_seeds(
        range(1, int(params["run.n_seeds_sweep"]) + 1) if seeds is None else seeds,
        params,
    )
    paths = point_parameter_paths(int(params["sweep.oat_max_rank"]), params)
    if not paths:
        raise InvariantError("OAT sweep has no rank-qualified parameters")
    kappa = float(params["sim.kappa.canonical"])
    rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    labels = ("lo", "central", "hi")
    for path in paths:
        levels = oat_levels(path, params)
        if len(levels) != len(labels):
            raise InvariantError(f"OAT parameter does not provide three levels: {path}")
        for label, level in zip(labels, levels, strict=True):
            result = run_point(
                kappa,
                overrides={path: level},
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
                "parameter": path,
                "rank": int(params.meta(path)["rank"]),
                "level": label,
                "value": json_value(level),
                "arm4_minus_arm1": summary["arm4_minus_arm1"],
                "arm5_minus_arm4": summary["arm5_minus_arm4"],
                "boundaries": _boundary_table(),
            })
    root = Path(output_root).resolve() if output_root is not None else params.path.resolve().parents[1] / "outputs" / "phase4"
    root.mkdir(parents=True, exist_ok=True)
    payload = {"kappa": kappa, "rows": rows}
    write_json(root / "oat.json", payload)
    return {"kappa": kappa, "rows": rows, "points": points}


run = run_oat_sweep

