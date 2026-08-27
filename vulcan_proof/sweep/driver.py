"""Resumable execution and aggregation for Phase-4 parameter points."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd

from ..errors import InvariantError
from ..manifest import load_manifest
from ..params import P, Params, load
from ..report.paired import paired_report
from .common import json_value, point_id, require_min_seeds, write_json, write_params_snapshot
from .runner import run_seed_worker
from ..sim.calibrate import calibrate_funnel


def _default_root(params: Params) -> pathlib.Path:
    """Return the repository Phase-4 output root."""
    return params.path.resolve().parents[1] / "outputs" / "phase4"


def _point_dir(kappa: float, overrides: Mapping[str, Any], root: pathlib.Path) -> pathlib.Path:
    """Return the stable directory for one parameter point."""
    return root / point_id(kappa, overrides)


def _snapshot_point(params: Params, overrides: Mapping[str, Any], point_dir: pathlib.Path, n_orders: int) -> tuple[Params, pathlib.Path, pathlib.Path]:
    """Create a point parameter snapshot and its isolated calibration file."""
    point_params = apply_overrides(params, overrides)
    snapshot = point_dir / "params.yaml"
    theta_path = point_dir / "theta.json"
    write_params_snapshot(point_params, snapshot)
    calibrate_funnel(
        seed=int(point_params["run.master_seed"]),
        n_orders=int(n_orders),
        params_path=snapshot,
        output_path=theta_path,
    )
    return load(snapshot), snapshot, theta_path


def point_complete(point_dir: pathlib.Path, kappa: float, seeds: Iterable[int], required_arms: Iterable[str] = ("arm1", "arm4", "arm5"), n_orders: int | None = None) -> bool:
    """Return whether a point has a completed summary and all seed manifests."""
    summary_path = point_dir / "point_summary.json"
    if not summary_path.is_file():
        return False
    expected_seeds = tuple(int(seed) for seed in seeds)
    expected_arms = tuple(str(arm) for arm in required_arms)
    for seed in expected_seeds:
        manifest_path = point_dir / f"seed_{seed}" / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = load_manifest(manifest_path)
            if manifest.get("phase") != "phase4" or manifest.get("kappa") != float(kappa) or manifest.get("seed") != seed:
                return False
            if n_orders is not None and manifest.get("n_orders") != int(n_orders):
                return False
            if not all((point_dir / f"seed_{seed}" / f"outcome_{arm}.parquet").is_file() for arm in expected_arms):
                return False
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
    return True


def _read_seed_frame(point_dir: pathlib.Path, seed: int, arm: str) -> pd.DataFrame:
    """Read one completed outcome artifact."""
    path = point_dir / f"seed_{seed}" / f"outcome_{arm}.parquet"
    if not path.is_file():
        raise InvariantError(f"missing Phase-4 artifact: {path}")
    return pd.read_parquet(path)


def summarise_point(point_dir: pathlib.Path, kappa: float, seeds: Iterable[int], params: Params = P) -> dict[str, Any]:
    """Aggregate paired reports and per-seed diagnostics for one point."""
    chosen = require_min_seeds(seeds, params)
    arm1 = {seed: _read_seed_frame(point_dir, seed, "arm1") for seed in chosen}
    arm4 = {seed: _read_seed_frame(point_dir, seed, "arm4") for seed in chosen}
    arm5 = {seed: _read_seed_frame(point_dir, seed, "arm5") for seed in chosen}
    summary: dict[str, Any] = {
        "kappa": float(kappa),
        "seeds": list(chosen),
        "arm4_minus_arm1": paired_report(arm1, arm4, params),
        "arm5_minus_arm4": paired_report(arm4, arm5, params),
        "arm5_minus_arm1": paired_report(arm1, arm5, params),
    }
    summary["n_orders"] = int(len(arm1[chosen[0]]))
    def mean_side(report: Mapping[str, Any], side: str) -> float:
        values = [float(row[f"{side}_net_per_1000"]) for row in report["per_seed"]]
        return float(sum(values) / len(values))

    summary["net"] = {
        "arm1": mean_side(summary["arm4_minus_arm1"], "left"),
        "arm4": mean_side(summary["arm4_minus_arm1"], "right"),
        "arm5": mean_side(summary["arm5_minus_arm4"], "right"),
    }
    if all((point_dir / f"seed_{seed}" / "outcome_arm3.parquet").is_file() for seed in chosen):
        arm3 = {seed: _read_seed_frame(point_dir, seed, "arm3") for seed in chosen}
        summary["arm3_minus_arm1"] = paired_report(arm1, arm3, params)
        summary["net"]["arm3"] = mean_side(summary["arm3_minus_arm1"], "right")
    if all((point_dir / f"seed_{seed}" / "outcome_arm2.parquet").is_file() for seed in chosen):
        arm2 = {seed: _read_seed_frame(point_dir, seed, "arm2") for seed in chosen}
        summary["arm2_minus_arm1"] = paired_report(arm1, arm2, params)
        summary["net"]["arm2"] = mean_side(summary["arm2_minus_arm1"], "right")
    diagnostics: dict[str, Any] = {}
    lorenz: list[float] = []
    model_hashes: list[str] = []
    stage_a_ece: list[float] = []
    lorenz_table: list[dict[str, Any]] = []
    for seed in chosen:
        metrics_path = point_dir / f"seed_{seed}" / "seed_metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        diagnostics[str(seed)] = payload["diagnostics"]
        lorenz.append(float(payload["lorenz"]["top_decile_lift"]))
        model_hashes.append(str(payload["model_hashes"]["defensibility"]))
        stage_a_ece.append(float(payload["model_metrics"]["stage_a"]["ece"]))
        if not lorenz_table:
            lorenz_table = payload["lorenz"]["deciles"]
    summary["diagnostics"] = diagnostics
    summary["lorenz_lift"] = float(sum(lorenz) / len(lorenz))
    summary["model_hashes"] = model_hashes
    summary["stage_a_ece"] = float(sum(stage_a_ece) / len(stage_a_ece))
    summary["lorenz"] = {"deciles": lorenz_table, "top_decile_lift": summary["lorenz_lift"]}
    return summary


def run_point(
    kappa: float,
    overrides: Mapping[str, Any] | None = None,
    params: Params = P,
    output_root: pathlib.Path | None = None,
    seeds: Iterable[int] | None = None,
    n_orders: int | None = None,
    parallel: bool = False,
    include_arm2: bool = False,
    include_arm3: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run or resume one κ/parameter point."""
    chosen_seeds = require_min_seeds(
        range(1, int(params["run.n_seeds_sweep"]) + 1) if seeds is None else seeds,
        params,
    )
    chosen_n = int(params["run.n_orders_sweep"]) if n_orders is None else int(n_orders)
    if chosen_n <= 0:
        raise InvariantError("Phase-4 n_orders must be positive")
    supplied = {} if overrides is None else dict(overrides)
    root = pathlib.Path(output_root).resolve() if output_root is not None else _default_root(params)
    directory = _point_dir(float(kappa), supplied, root)
    directory.mkdir(parents=True, exist_ok=True)
    required_arms = ("arm1", "arm4", "arm5") + (("arm2",) if include_arm2 else ()) + (("arm3",) if include_arm3 else ())
    if point_complete(directory, float(kappa), chosen_seeds, required_arms, chosen_n):
        summary = json.loads((directory / "point_summary.json").read_text(encoding="utf-8"))
        return {"point_dir": directory, "summary": summary, "resumed": True}

    point_params, snapshot, theta_path = _snapshot_point(params, supplied, directory, chosen_n)
    tasks: list[tuple[float, int, dict[str, Any], str | pathlib.Path]] = []
    for seed in chosen_seeds:
        metadata = dict(supplied)
        metadata.update({
            "__output_dir__": str(directory / f"seed_{seed}"),
            "__theta_path__": str(theta_path),
            "__n_orders__": chosen_n,
            "__allow_dirty__": allow_dirty,
            "__include_arm2__": include_arm2,
            "__include_arm3__": include_arm3,
            "__manifest_params_path__": str(params.path),
        })
        tasks.append((float(kappa), seed, metadata, str(snapshot)))
    worker_results: list[dict[str, Any]] = []
    if parallel:
        with ProcessPoolExecutor(max_workers=int(params["run.parallel_workers"])) as executor:
            for result in executor.map(run_seed_worker, tasks):
                worker_results.append(result)
    else:
        for task in tasks:
            worker_results.append(run_seed_worker(task))
    summary = summarise_point(directory, float(kappa), chosen_seeds, point_params)
    summary["overrides"] = json_value(supplied)
    summary["point_id"] = directory.name
    summary["params_sha256"] = point_params.sha256
    summary["base_params_sha256"] = params.sha256
    summary["theta_path"] = str(theta_path)
    write_json(directory / "point_summary.json", summary)
    return {
        "point_dir": directory,
        "point_id": directory.name,
        "summary": summary,
        "runs": worker_results,
        "params": point_params,
        "snapshot": snapshot,
        "theta_path": theta_path,
        "resumed": False,
    }


def load_point_summary(point_dir: pathlib.Path) -> dict[str, Any]:
    """Load one previously completed point summary."""
    return json.loads((point_dir / "point_summary.json").read_text(encoding="utf-8"))
