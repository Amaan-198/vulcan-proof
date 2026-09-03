"""Robustness contexts for the sensitivity package."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..params import P, Params
from .common import write_json, require_min_seeds
from .driver import run_point


def robustness_points(params: Params = P) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return the named robustness overrides from the phase specification."""
    return (
        ("observational_history", {"archetypes.random_stratum_frac": params.meta("archetypes.random_stratum_frac")["sweep"][0]}),
        ("carrier_fault_exposure", {"uplift_true.on_misdelivered_otp": params.meta("uplift_true.on_misdelivered_otp")["sweep"][-1]}),
        (
            "deterrence_silence",
            {
                "sim.deterrence": params.meta("sim.deterrence")["sweep"][-1],
                "sim.silence_yield": params.meta("sim.silence_yield")["sweep"][-1],
            },
        ),
        ("world_d_shift", {"sim.shift.enabled_default": True}),
    )


def _carrier_fault_rate(summary: dict[str, Any], arm: str) -> float:
    """Average defense-only carrier-fault outcome over a point's seeds."""
    values = []
    for diagnostics in summary["diagnostics"].values():
        values.append(float(diagnostics[arm]["defense_only_win_rate"]["carrier_fault"]))
    return float(sum(values) / len(values))


def run_robustness_sweeps(
    params: Params = P,
    output_root: Path | None = None,
    seeds: Iterable[int] | None = None,
    n_orders: int | None = None,
    parallel: bool = True,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run and persist the configured robustness points."""
    chosen_seeds = require_min_seeds(
        range(1, int(params["run.n_seeds_sweep"]) + 1) if seeds is None else seeds,
        params,
    )
    kappa = float(params["sim.kappa.canonical"])
    rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for name, overrides in robustness_points(params):
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
        row: dict[str, Any] = {
            "name": name,
            "overrides": overrides,
            "arm5_minus_arm4": summary["arm5_minus_arm4"],
            "arm4_minus_arm1": summary["arm4_minus_arm1"],
            "carrier_fault_win_rate_arm1": _carrier_fault_rate(summary, "arm1"),
            "carrier_fault_win_rate_arm5": _carrier_fault_rate(summary, "arm5"),
        }
        if name == "world_d_shift":
            row["stage_a_ece_test"] = summary["stage_a_ece"]
        rows.append(row)
    root = Path(output_root).resolve() if output_root is not None else params.path.resolve().parents[1] / "outputs" / "phase4"
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "robustness.json", {"kappa": kappa, "rows": rows})
    return {"kappa": kappa, "rows": rows, "points": points}


run = run_robustness_sweeps
