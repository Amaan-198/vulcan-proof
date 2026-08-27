"""One-world Phase-4 execution, kept outside the earlier-phase modules."""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..arms.arm1 import plan as arm1_plan
from ..arms.arm2 import plan as arm2_plan
from ..arms.arm3 import plan as arm3_plan
from ..arms.arm5 import plan as arm5_plan
from ..errors import InvariantError
from ..manifest import finish_run, start_run, write_artifact
from ..models import fit_models
from ..models.labels import eligible
from ..params import P, Params, load
from ..report.paired import arm_diagnostics
from ..sim.arm0_history import plan as arm0_plan
from ..sim.generator import generate_world
from ..sim.history import add_history_features
from ..sim.resolve import resolve, score_masks
from ..arms.arm4 import Arm4Policy, tune as tune_arm4
from .common import apply_overrides, model_artifact_hash, write_json


class _Arm4Scorer:
    """Score Arm-4 candidate plans using one shared latent world."""

    def __init__(self, observed: pd.DataFrame, hidden: pd.DataFrame, seed: int, params: Params) -> None:
        self.observed_by_id = observed.set_index("order_id", drop=False)
        self.hidden_by_id = hidden.set_index("order_id", drop=False)
        self.seed = int(seed)
        self.params = params

    def _selected(self, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Align latent rows to a candidate plan."""
        ids = candidate["order_id"].astype(str).tolist()
        return (
            self.observed_by_id.loc[ids].reset_index(drop=True),
            self.hidden_by_id.loc[ids].reset_index(drop=True),
        )

    def __call__(self, candidate: pd.DataFrame) -> float:
        """Score one candidate plan."""
        observed, hidden = self._selected(candidate)
        return float(score_masks([candidate], hidden, observed, self.seed, self.params)[0])

    def score_batch(self, candidates: Sequence[pd.DataFrame]) -> tuple[float, ...]:
        """Score a candidate batch with shared random draws."""
        if not candidates:
            raise InvariantError("Arm 4 requested an empty score batch")
        observed, hidden = self._selected(candidates[0])
        values = score_masks(candidates, hidden, observed, self.seed, self.params)
        return tuple(float(value) for value in values)


def _scored_lorenz(observed: pd.DataFrame, outcome: pd.DataFrame, scores: np.ndarray, params: Params) -> dict[str, Any]:
    """Compute Stage-A top-decile lift on eligible test rows."""
    del observed
    mask = eligible(outcome, "test", purpose="evaluate").to_numpy()
    if not bool(mask.any()):
        mask = np.ones(len(outcome), dtype=bool)
    labels = outcome["dispute_opened"].to_numpy(dtype="int8").astype("float64")
    values = np.asarray(scores, dtype="float64")
    order = np.argsort(-values, kind="mergesort")
    selected = order[mask[order]]
    total = float(labels[selected].sum())
    deciles = int(params["report.lorenz_deciles"])
    rows: list[dict[str, Any]] = []
    for position in range(deciles):
        start = (position * len(selected)) // deciles
        end = ((position + 1) * len(selected)) // deciles
        share = float(labels[selected[start:end]].sum() / total) if total > 0.0 and end > start else 0.0
        rows.append({"decile": position + 1, "share_of_disputes": share})
    top = rows[0]["share_of_disputes"] if rows and total > 0.0 else 0.0
    return {"deciles": rows, "top_decile_lift": float(top * deciles)}


def run_seed(
    kappa: float,
    seed: int,
    params: Params = P,
    n_orders: int | None = None,
    theta_path: pathlib.Path | None = None,
    output_dir: pathlib.Path | None = None,
    allow_dirty: bool = False,
    include_arm2: bool = False,
    include_arm3: bool = False,
    manifest_params: Params | None = None,
) -> dict[str, Any]:
    """Generate, refit, plan, and resolve one independent sweep seed."""
    chosen_n = int(params["run.n_orders_sweep"]) if n_orders is None else int(n_orders)
    if chosen_n <= 0:
        raise InvariantError("Phase-4 n_orders must be positive")
    if theta_path is None:
        root = params.path.resolve().parents[1]
        theta_path = root / "outputs" / "theta.json"
    observed, hidden = generate_world(
        kappa=float(kappa),
        seed=int(seed),
        n_orders=chosen_n,
        shift_enabled=bool(params["sim.shift.enabled_default"]),
        params_path=params.path,
        theta_path=theta_path,
    )
    historical_plan = arm0_plan(hidden, observed, params)
    historical = resolve(historical_plan, hidden, observed, seed=int(seed), params=params, arm_id="arm0")
    completed = add_history_features(observed, historical, params)
    policy: Arm4Policy = tune_arm4(completed, _Arm4Scorer(completed, hidden, int(seed), params), params)

    plans = {
        "arm1": arm1_plan(completed, params),
        "arm4": policy.plan(completed),
    }
    if include_arm2:
        plans["arm2"] = arm2_plan(completed, params)
    if include_arm3:
        plans["arm3"] = arm3_plan(completed, params)
    models = fit_models(completed, historical, params, int(seed))
    plans["arm5"] = arm5_plan(completed, models, models.support_mask, params)
    outcomes = {
        arm: resolve(plan, hidden, completed, seed=int(seed), params=params, arm_id=arm)
        for arm, plan in plans.items()
    }
    result: dict[str, Any] = {
        "kappa": float(kappa),
        "seed": int(seed),
        "n_orders": chosen_n,
        "observed": completed,
        "hidden": hidden,
        "historical": historical,
        "plans": plans,
        "outcomes": outcomes,
        "policy": policy,
        "models": models,
        "model_hashes": {"defensibility": model_artifact_hash(models)},
        "model_metrics": {"stage_a": models.metrics["stage_a"]},
        "diagnostics": {arm: arm_diagnostics(frame, params) for arm, frame in outcomes.items()},
        "lorenz": _scored_lorenz(completed, historical, models.stage_a.predict(completed), params),
    }
    if output_dir is not None:
        result["manifest"] = _write_seed_artifacts(result, pathlib.Path(output_dir), params if manifest_params is None else manifest_params, allow_dirty)
    return result


def _write_seed_artifacts(result: dict[str, Any], output_dir: pathlib.Path, params: Params, allow_dirty: bool) -> dict[str, Any]:
    """Persist one seed's observed outcomes and write its manifest last."""
    context = start_run("phase4", params, allow_dirty=allow_dirty, run_dir=output_dir)
    try:
        write_artifact(context, result["observed"], "observed_orders")
        for arm in sorted(result["outcomes"]):
            write_artifact(context, result["outcomes"][arm], f"outcome_{arm}")
        policy = result["policy"]
        write_json(context.run_dir / "arm4_policy.json", policy.to_dict())
        write_json(context.run_dir / "seed_metrics.json", {
            "kappa": result["kappa"],
            "seed": result["seed"],
            "model_hashes": result["model_hashes"],
            "model_metrics": result["model_metrics"],
            "diagnostics": result["diagnostics"],
            "lorenz": result["lorenz"],
        })
        context.manifest.update({
            "kappa": result["kappa"],
            "seed": result["seed"],
            "n_orders": result["n_orders"],
            "arms": sorted(result["outcomes"]),
            "model_hashes": result["model_hashes"],
            "history_features": [
                "merchant_dispute_rate_hist",
                "merchant_contest_rate_hist",
                "merchant_compliance_hist",
            ],
        })
        finish_run(context)
    except Exception:
        finish_run(context)
        raise
    return context.manifest


def run_seed_worker(task: tuple[float, int, dict[str, Any], str | pathlib.Path]) -> dict[str, Any]:
    """Spawn-safe worker entry point; workers reload parameters from disk."""
    kappa, seed, raw_overrides, params_path = task
    overrides = dict(raw_overrides)
    output_dir = pathlib.Path(str(overrides["__output_dir__"]))
    theta_path = pathlib.Path(str(overrides["__theta_path__"]))
    n_orders = int(overrides["__n_orders__"])
    allow_dirty = bool(overrides["__allow_dirty__"])
    manifest_params = load(overrides["__manifest_params_path__"])
    params = apply_overrides(load(params_path), overrides)
    result = run_seed(
        float(kappa),
        int(seed),
        params=params,
        n_orders=n_orders,
        theta_path=theta_path,
        output_dir=output_dir,
        allow_dirty=allow_dirty,
        include_arm2=bool(overrides.get("__include_arm2__", False)),
        include_arm3=bool(overrides.get("__include_arm3__", False)),
        manifest_params=manifest_params,
    )
    return {
        "kappa": result["kappa"],
        "seed": result["seed"],
        "n_orders": result["n_orders"],
        "run_dir": str(output_dir),
        "manifest": result["manifest"],
        "model_hashes": result["model_hashes"],
        "model_metrics": result["model_metrics"],
        "diagnostics": result["diagnostics"],
        "lorenz": result["lorenz"],
    }
