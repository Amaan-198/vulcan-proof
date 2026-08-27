"""End-to-end Phase-0 Olist run."""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

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


def _write_report(
    path: pathlib.Path,
    metrics: dict[str, Any],
    manifest_path: pathlib.Path,
) -> None:
    operating_rows = metrics["operating_points"]
    lines = [
        "# Phase 0 report — Olist detection anchor",
        "",
        "| Metric | Test value |",
        "|---|---:|",
        f"| PR-AUC | {metrics['pr_auc']:.6f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.6f} |",
        f"| Brier | {metrics['brier']:.6f} |",
        f"| ECE | {metrics['ece']:.6f} |",
        "",
        f"Label rate: {metrics['label_rate']:.6f}",
        "",
        "Reason mix:",
    ]
    for reason, count in metrics["reason_mix"].items():
        lines.append(f"- {reason}: {count}")
    lines.extend(
        [
            "",
            f"Immature-drop count: {metrics['immature_drop_count']}",
            "",
            "Operating points:",
            "",
            "| Target recall | Precision | Flagged fraction | FP / 1,000 | FP cost / 1,000 INR |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in operating_rows:
        lines.append(
            f"| {row['target_recall']:.2f} | {row['precision']:.6f} | "
            f"{row['flagged_fraction']:.6f} | {row['false_positives_per_1000']:.3f} | "
            f"{row['fp_cost_inr_per_1000']:.3f} |"
        )
    test_calibration = metrics["calibration_checks"]["test"]
    calibration_tolerance = float(P["models.calib.mean_tolerance"])
    test_relative_error = abs(test_calibration["mean_vs_rate_ratio"] - 1.0)
    if test_relative_error > calibration_tolerance:
        calibration_note = (
            "Calibration note: validation-only isotonic calibration matches the validation rate, "
            f"but the test mean/rate ratio is {test_calibration['mean_vs_rate_ratio']:.6f}. "
            "This is recorded as temporal prevalence drift between the validation and test windows; "
            "test labels were not used for calibration."
        )
    else:
        calibration_note = (
            "Calibration note: validation-only isotonic calibration matches both validation and "
            "test rates within the configured tolerance."
        )
    lines.extend(
        [
            "",
            f"Split counts: {metrics['split_counts']}",
            f"Calibration checks: {metrics['calibration_checks']}",
            calibration_note,
            "",
            "Olist has no chargeback or evidence data; this measures detection only.",
            "",
            f"Manifest: `{manifest_path}`",
            "",
            f"Chart footer: {metrics['olist_footer']}",
            "",
        ]
    )
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
        label_summary = label_statistics(labels, int(excluded_statuses.sum()))
        split_counts = split_statistics(assigned)
        metrics: dict[str, Any] = {
            **test_metrics,
            **label_summary,
            "split_counts": split_counts,
            # Count only eligible labelled orders, using the exact same boundary
            # as assign_splits.  Counting raw orders included excluded statuses
            # and made this value disagree with split_counts["immature"].
            "immature_drop_count": split_counts.get("immature", 0),
            "calibration_checks": {"validate": validation, "test": test_calibration},
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
