"""End-to-end Phase-0 Olist run."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import pandas as pd

from ..manifest import finish_run, start_run, write_artifact
from ..params import P, Params
from .charts import write_charts
from .evaluate import evaluate_predictions
from .features import build_features
from .label import build_labels, label_statistics
from .load import download_olist, load_olist
from .split import assign_splits, split_statistics
from .train import calibration_summary, fit_detector


def _read_metric(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _temporal_drift_diagnostics(
    split_labels: pd.DataFrame,
    calibration_checks: dict[str, dict[str, float]],
    params: Params,
) -> dict[str, Any]:
    """Attribute out-of-time prevalence movement without altering predictions."""
    selected = split_labels.loc[split_labels["split"].isin({"validate", "test"})].copy()
    selected["month"] = selected["purchase_ts"].dt.to_period("M").astype("string")
    positive_reasons = sorted(
        reason for reason in selected["label_reason"].astype(str).unique() if reason != "none"
    )
    monthly: list[dict[str, Any]] = []
    for split_name in ("validate", "test"):
        split_frame = selected.loc[selected["split"].eq(split_name)]
        for month, group in split_frame.groupby("month", sort=True):
            reason_counts = group["label_reason"].astype(str).value_counts()
            row: dict[str, Any] = {
                "split": split_name,
                "month": str(month),
                "n_orders": int(len(group)),
                "label_rate": float(group["label"].mean()),
            }
            for reason in positive_reasons:
                row[f"{reason}_rate"] = float(reason_counts.get(reason, 0) / len(group))
            monthly.append(row)

    reason_rates: dict[str, dict[str, float]] = {}
    for split_name in ("validate", "test"):
        group = selected.loc[selected["split"].eq(split_name)]
        counts = group["label_reason"].astype(str).value_counts()
        reason_rates[split_name] = {
            reason: float(counts[reason] / len(group)) for reason in positive_reasons
        }
    drops = {
        reason: reason_rates["validate"][reason] - reason_rates["test"][reason]
        for reason in positive_reasons
    }
    largest_drop_reason = max(drops, key=drops.__getitem__)
    test_check = calibration_checks["test"]
    relative_error = abs(test_check["mean_vs_rate_ratio"] - 1.0)
    return {
        "calibration_protocol": "isotonic fitted on validation labels only; test labels used for evaluation only",
        "test_within_mean_tolerance": relative_error < float(params["models.calib.mean_tolerance"]),
        "test_relative_mean_error": relative_error,
        "validation_to_test_prevalence_ratio": (
            calibration_checks["validate"]["empirical_rate"] / test_check["empirical_rate"]
        ),
        "label_reason_rates": reason_rates,
        "label_reason_rate_drops": drops,
        "largest_rate_drop_reason": largest_drop_reason,
        "monthly_label_rates": monthly,
    }


def _write_report(
    path: pathlib.Path,
    metrics: dict[str, Any],
    manifest_path: pathlib.Path,
) -> None:
    _ = metrics
    _ = _read_metric(manifest_path)
    lines = [
        "# Phase 0 report — Olist detection anchor",
        "",
        "## Detection scope",
        "",
        "This report describes the public-data detection anchor and its leakage-safe feature path.",
        "Olist has no chargeback or evidence data; this measures detection only.",
        "",
        "## Calibration and temporal behavior",
        "",
        "The machine-readable metrics artifact retains probability, reliability, split, and temporal diagnostics.",
        "Isotonic calibration is learned on validation data; the test period is used for evaluation and drift description.",
        "",
        "## Artifacts",
        "",
        "The run manifest, feature artifact, labels, reliability data, and charts are recorded together.",
        "The public-data anchor is not a source of dispute-economics or evidence-yield claims.",
        "",
        "Chart footer: Olist public dataset; no chargeback or evidence data; detection only.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase0(
    params: Params = P,
    output_dir: pathlib.Path | None = None,
    download: bool = False,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Run the real-data detection anchor and write its report and artifacts."""
    root = params.path.resolve().parents[1]
    target = output_dir if output_dir is not None else root / "outputs" / "phase0"
    target = pathlib.Path(target).resolve()
    if download:
        data_dir = download_olist(params)
    else:
        data_dir = None
    context = start_run("phase0", params, allow_dirty=allow_dirty, run_dir=target)
    try:
        tables = load_olist(data_dir, params)
        orders = tables["olist_orders_dataset.csv"]
        reviews = tables["olist_order_reviews_dataset.csv"]
        excluded_statuses = orders["order_status"].isin({"canceled", "unavailable", "created"})
        labels = build_labels(orders, reviews, params)
        assigned = assign_splits(labels, params)
        mature = assigned[assigned["split"].isin({"train", "validate", "test"})].copy()
        mature_labels = labels.loc[labels["order_id"].isin(mature["order_id"])].reset_index(drop=True)
        mature["order_id"] = mature["order_id"].astype("string")
        features = build_features(tables, mature_labels, params)
        write_artifact(context, mature_labels, "olist_labels")
        write_artifact(context, features, "olist_features")
        detector = fit_detector(features, mature_labels, mature.reset_index(drop=True), params)
        test_mask = mature["split"].eq("test").to_numpy()
        test_scores = detector.predict(features.loc[test_mask].reset_index(drop=True))
        test_y = mature_labels.loc[test_mask, "label"].to_numpy(dtype="int8")
        test_metrics = evaluate_predictions(test_y, test_scores, params)
        validation = calibration_summary(detector, features, mature_labels, mature, "validate")
        test_calibration = calibration_summary(detector, features, mature_labels, mature, "test")
        calibration_checks = {"validate": validation, "test": test_calibration}
        label_summary = label_statistics(labels, int(excluded_statuses.sum()))
        split_counts = split_statistics(assigned)
        metrics: dict[str, Any] = {
            **test_metrics,
            **label_summary,
            "split_counts": split_counts,
            # Keep this report field on the same eligible-labelled boundary as the split summary.
            "immature_drop_count": split_counts["immature"],
            "calibration_checks": calibration_checks,
            "temporal_drift": _temporal_drift_diagnostics(mature, calibration_checks, params),
            "olist_footer": params["report.olist_footer"],
            "params_sha256": params.sha256,
        }
        metrics_path = target / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        metrics_digest = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
        context.manifest["metrics_sha256"] = metrics_digest
        context.manifest["metrics_path"] = str(metrics_path)
        write_charts(target, test_y, test_scores, test_metrics["reliability"], params)
        finish_run(context)
        _write_report(root / "outputs" / "phase0_REPORT.md" if output_dir is None else target.parent / "phase0_REPORT.md", metrics, context.manifest_path)
        return metrics
    except Exception:
        finish_run(context)
        raise
