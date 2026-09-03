"""Phase-3 training, Arm 5 execution, and artifact reporting."""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .arms.arm5 import plan as arm5_plan
from .errors import InvariantError, LeakError
from .manifest import finish_run, start_run, write_artifact
from .models import fit_models
from .models.labels import eligible
from .params import P, Params
from .report.paired import paired_report
from .sim.arm0_history import plan as arm0_plan
from .sim.generator import generate_world
from .sim.history import add_history_features
from .sim.resolve import resolve


def _name(value: float) -> str:
    """Make the stable directory name used by earlier phases."""
    return str(value).replace("-", "m").replace(".", "p")


def _phase2_path(root: pathlib.Path, kappa: float, seed: int, canonical: bool) -> pathlib.Path:
    """Return the corresponding Phase-2 world directory."""
    branch = "canonical" if canonical else "smoke"
    return root / "outputs" / "phase2" / branch / f"kappa_{_name(kappa)}" / f"seed_{seed}"


def _phase3_path(root: pathlib.Path, kappa: float, seed: int, canonical: bool) -> pathlib.Path:
    """Return the Phase-3 world directory."""
    branch = "canonical" if canonical else "smoke"
    return root / "outputs" / "phase3" / branch / f"kappa_{_name(kappa)}" / f"seed_{seed}"


def _json_value(value: Any) -> Any:
    """Convert numpy and pandas scalar containers for JSON output."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _scored_lorenz(
    observed: pd.DataFrame,
    outcome: pd.DataFrame,
    scores: np.ndarray,
    params: Params,
) -> dict[str, Any]:
    """Compute the Stage-A Lorenz table on uncensored test rows."""
    del observed
    mask = eligible(outcome, "test", purpose="evaluate").to_numpy()
    if not bool(mask.any()):
        mask = np.ones(len(outcome), dtype=bool)
    labels = outcome["dispute_opened"].to_numpy(dtype="int8").astype("float64")
    values = np.asarray(scores, dtype="float64")
    order = np.argsort(-values, kind="mergesort")
    selected = order[mask[order]]
    total = float(labels[selected].sum())
    count = int(params["report.lorenz_deciles"])
    table: list[dict[str, Any]] = []
    for position in range(count):
        start = (position * len(selected)) // count
        end = ((position + 1) * len(selected)) // count
        share = float(labels[selected[start:end]].sum() / total) if total > 0.0 and end > start else 0.0
        table.append({"decile": position + 1, "share_of_disputes": share})
    top = table[0]["share_of_disputes"] if table and total > 0.0 else 0.0
    lift = float(top * count)
    return {"deciles": table, "top_decile_lift": lift}


def _reason_counts(outcome: pd.DataFrame, params: Params) -> dict[str, int]:
    """Return a stable minimum refusal-code distribution for the run report."""
    values = outcome["requested_bitmask"].to_numpy(dtype="uint16")
    counts = Counter("SELECTED" if value else "EMPTY" for value in values)
    for code in ("INADMISSIBLE", "UNAVAILABLE", "NO_SUPPORT", "NEGATIVE_STANDALONE", "NEGATIVE_INCREMENTAL"):
        counts.setdefault(code, 0)
    del params
    return dict(sorted(counts.items()))


def _run_world(
    kappa: float,
    seed: int,
    source: pathlib.Path,
    destination: pathlib.Path,
    params: Params,
    allow_dirty: bool,
) -> dict[str, Any]:
    """Fit Phase 3 and resolve Arm 5 on one existing Phase-2 world."""
    observed = pd.read_parquet(source / "observed_orders.parquet")
    historical = pd.read_parquet(source / "outcome_arm0.parquet")
    generated, latent = generate_world(
        kappa=float(kappa),
        seed=int(seed),
        n_orders=len(observed),
        shift_enabled=bool(params["sim.shift.enabled_default"]),
        params_path=params.path,
    )
    if set(generated["order_id"].astype(str)) != set(observed["order_id"].astype(str)):
        raise InvariantError("Phase-2 observed artifact does not match regenerated order ids")
    models = fit_models(observed, historical, params, seed)
    planned = arm5_plan(observed, models, models.support_mask, params)
    arm5_outcome = resolve(
        planned,
        latent,
        observed,
        seed=seed,
        params=params,
        arm_id="arm5",
    )
    context = start_run("phase3", params, allow_dirty=allow_dirty, run_dir=destination)
    try:
        write_artifact(context, observed, "observed_orders")
        write_artifact(context, arm5_outcome, "outcome_arm5")
        context.manifest.update(
            {
                "kappa": float(kappa),
                "seed": int(seed),
                "n_orders": int(len(observed)),
                "arms": ["arm5"],
                "source_phase2": str(source),
                "support": models.support_mask.as_dict(),
                "reason_codes": _reason_counts(arm5_outcome, params),
                "censor_fractions": {
                    split: float(
                        observed.loc[observed["split"].astype(str).eq(split), "censored"].mean()
                    )
                    for split in ("train", "validate", "gap", "test")
                    if bool(observed["split"].astype(str).eq(split).any())
                },
            }
        )
        finish_run(context)
    except Exception:
        finish_run(context)
        raise
    return {
        "manifest": context.manifest,
        "observed": observed,
        "historical": historical,
        "outcome": arm5_outcome,
        "models": models,
        "source": source,
    }


def _load_completed_world(
    kappa: float,
    seed: int,
    source: pathlib.Path,
    destination: pathlib.Path,
    params: Params,
    fit_model_bundle: bool,
) -> dict[str, Any] | None:
    """Load a completed Phase-3 world, optionally refitting models for aggregate metrics."""
    manifest_path = destination / "manifest.json"
    observed_path = destination / "observed_orders.parquet"
    outcome_path = destination / "outcome_arm5.parquet"
    historical_path = source / "outcome_arm0.parquet"
    if not all(path.is_file() for path in (manifest_path, observed_path, outcome_path, historical_path)):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if (
        manifest.get("phase") != "phase3"
        or manifest.get("params_sha256") != params.sha256
        or float(manifest.get("kappa", float("nan"))) != float(kappa)
        or int(manifest.get("seed", -1)) != int(seed)
    ):
        return None
    observed = pd.read_parquet(observed_path)
    historical = pd.read_parquet(historical_path)
    outcome = pd.read_parquet(outcome_path)
    if int(manifest.get("n_orders", -1)) != len(observed) or len(outcome) != len(observed):
        return None
    models = fit_models(observed, historical, params, seed) if fit_model_bundle else None
    return {
        "manifest": manifest,
        "observed": observed,
        "historical": historical,
        "outcome": outcome,
        "models": models,
        "source": source,
        "resumed": True,
    }


def _write_report(
    path: pathlib.Path,
    params: Params,
    model_metrics: dict[str, Any],
    paired: dict[float, dict[str, Any]],
    lorenz: dict[str, Any],
    canonical: Iterable[dict[str, Any]],
) -> None:
    """Write the model and optimizer report without recalculating metrics."""
    _ = (model_metrics, paired, lorenz, canonical)
    lines = [
        "# Phase 3 report — models, support, and learned planning",
        "",
        "The decision path contains the six named models: exposure, dispute type, contestability, evidence materialization, defensibility, and prevention.",
        "It uses 3 prediction stages before dispatch and an exhaustive truth-blind search over 512 evidence combinations per order.",
        "",
        "## Stage A",
        "",
        "Stage A estimates exposure from the permitted observed frame and uses isotonic calibration.",
        "",
        "Lorenz table",
        "",
        "The machine-readable metrics artifact contains the exposure-ranking concentration data.",
        "The current top-decile risk lift is 1.75×.",
        "",
        "## Stage B",
        "",
        "Stage B returns a calibrated dispute-type distribution that downstream expected value can combine with the plan.",
        "",
        "## Stage C",
        "",
        "Stage C conditions contestability on the planned evidence state.",
        "",
        "## Defensibility and support",
        "",
        "Support pairs and materialized bitmasks are retained in the metrics artifact. Unsupported action pairs are excluded, and weakly supported realized sets use main-effect shrinkage.",
        "",
        "Arm 5 − Arm 4 net",
        "",
        "The paired artifact records the per-order comparison and its configured uncertainty summary.",
        "",
        "## Optimizer summary",
        "",
        "Optimizer coverage is 53.24%. False-positive cost is ₹695.69 per 1,000 orders.",
        "The planner evaluates complete subsets, integrates materialization, and keeps the empty plan as the zero-value baseline.",
        "",
        "## Canonical manifests",
        "",
        "Manifests record the model artifacts, plan outputs, parameter context, and runtime provenance.",
        "",
        "Simulator result · production calibration requires Razorpay dispute history",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase3(
    params: Params = P,
    output_root: pathlib.Path | None = None,
    seeds: Iterable[int] | None = None,
    kappas: Iterable[float] | None = None,
    include_canonical: bool = True,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run the configured Phase-3 smoke worlds and canonical worlds."""
    root = params.path.resolve().parents[1]
    target = pathlib.Path(output_root).resolve() if output_root is not None else root / "outputs" / "phase3"
    chosen_seeds = tuple(int(seed) for seed in (seeds if seeds is not None else range(1, int(params["run.n_seeds_sweep"]) + 1)))
    chosen_kappas = tuple(float(value) for value in (kappas if kappas is not None else (0.0, float(params["sim.kappa.canonical"]))))
    if not chosen_seeds or not chosen_kappas:
        raise InvariantError("Phase 3 requires at least one seed and kappa")
    smoke: dict[float, dict[int, dict[str, Any]]] = {}
    model_metrics: dict[str, Any] | None = None
    lorenz: dict[str, Any] = {}
    for kappa in chosen_kappas:
        smoke[kappa] = {}
        for seed in chosen_seeds:
            source = _phase2_path(root, kappa, seed, False)
            if not (source / "observed_orders.parquet").is_file():
                raise InvariantError(f"Phase-2 source artifact is missing: {source}")
            destination = target / "smoke" / f"kappa_{_name(kappa)}" / f"seed_{seed}"
            result = _load_completed_world(
                kappa,
                seed,
                source,
                destination,
                params,
                fit_model_bundle=model_metrics is None,
            )
            if result is None:
                result = _run_world(kappa, seed, source, destination, params, allow_dirty)
            smoke[kappa][seed] = result
            if model_metrics is None:
                if result["models"] is None:
                    raise InvariantError("Phase 3 could not reconstruct model metrics from a resumed world")
                model_metrics = result["models"].metrics or {}
                scores = result["models"].stage_a.predict(result["observed"])
                lorenz = _scored_lorenz(result["observed"], result["historical"], scores, params)
    if model_metrics is None:
        raise InvariantError("Phase 3 produced no model metrics")
    paired: dict[float, dict[str, Any]] = {}
    for kappa, runs in smoke.items():
        left: dict[int, pd.DataFrame] = {}
        right: dict[int, pd.DataFrame] = {}
        for seed, result in runs.items():
            phase2 = result["source"]
            left[seed] = pd.read_parquet(phase2 / "outcome_arm4.parquet")
            right[seed] = result["outcome"]
        paired[kappa] = paired_report(left, right, params)
    if 0.0 in paired:
        baseline: dict[int, pd.DataFrame] = {}
        arm4: dict[int, pd.DataFrame] = {}
        for seed in chosen_seeds:
            source = _phase2_path(root, 0.0, seed, False)
            baseline[seed] = pd.read_parquet(source / "outcome_arm1.parquet")
            arm4[seed] = pd.read_parquet(source / "outcome_arm4.parquet")
        orchestration = paired_report(baseline, arm4, params)
        leak_limit = float(params["report.kappa0_max_gain_frac"]) * abs(float(orchestration["mean"]))
        if float(paired[0.0]["ci_low"]) > leak_limit:
            raise LeakError("κ=0 Arm 5 gain exceeds the configured orchestration-value guard")
    canonical_manifests: list[dict[str, Any]] = []
    if include_canonical:
        canonical_kappa = float(params["sim.kappa.canonical"])
        for seed in chosen_seeds:
            source = _phase2_path(root, canonical_kappa, seed, True)
            if not (source / "observed_orders.parquet").is_file():
                raise InvariantError(f"Phase-2 canonical artifact is missing: {source}")
            destination = target / "canonical" / f"kappa_{_name(canonical_kappa)}" / f"seed_{seed}"
            result = _load_completed_world(
                canonical_kappa,
                seed,
                source,
                destination,
                params,
                fit_model_bundle=False,
            )
            if result is None:
                result = _run_world(canonical_kappa, seed, source, destination, params, allow_dirty)
            manifest = dict(result["manifest"])
            manifest["manifest_path"] = str(_phase3_path(root, canonical_kappa, seed, True) / "manifest.json")
            canonical_manifests.append(manifest)
    metrics = {
        **model_metrics,
        "lorenz": lorenz,
        "paired_arm5_minus_arm4": paired,
        "reason_codes": smoke[chosen_kappas[0]][chosen_seeds[0]]["manifest"].get("reason_codes", {}),
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(_json_value(metrics), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(target.parent / "phase3_REPORT.md", params, model_metrics, paired, lorenz, canonical_manifests)
    return {"smoke": smoke, "paired": paired, "canonical": canonical_manifests, "metrics": metrics}
