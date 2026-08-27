"""Phase-0 fixtures test labels, temporal maturity, schemas, and metrics."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from vulcan_proof.olist.evaluate import evaluate_predictions
from vulcan_proof.olist.features import _history_features
from vulcan_proof.olist.label import build_labels
from vulcan_proof.olist.split import assign_splits
from vulcan_proof.params import P
from vulcan_proof.schemas import OLIST_ORDER_FEATURES, check


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d", "e", "f"],
            "order_status": ["delivered", "delivered", "shipped", "delivered", "delivered", "canceled"],
            "order_purchase_timestamp": ["2018-01-01"] * 6,
            "order_delivered_customer_date": ["2018-01-12", "2018-01-10", None, "2018-01-10", "2018-01-10", None],
            "order_estimated_delivery_date": ["2018-01-10", "2018-01-10", None, "2018-01-12", "2018-01-12", None],
        }
    )


def _reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["a", "b", "c", "d", "e", "f"],
            "review_score": [1, 1, np.nan, 1, 5, 1],
            "review_comment_message": ["", "produto errado", "", "", "", ""],
            "review_creation_date": ["2018-01-20"] * 6,
        }
    )


def test_label_rule_known_rows() -> None:
    labels = build_labels(_orders(), _reviews(), P)
    assert labels["order_id"].astype(str).tolist() == ["a", "b", "c", "d", "e"]
    assert labels["label"].tolist() == [1, 1, 1, 0, 0]
    assert labels["label_reason"].astype(str).tolist() == [
        "late_low_score",
        "comment_match",
        "not_delivered",
        "none",
        "none",
    ]


def test_no_forbidden_features() -> None:
    categorical = {"product_category_en", "payment_type", "customer_state", "seller_state"}
    frame = pd.DataFrame({column: pd.Series([1.0], dtype="float32") for column in OLIST_ORDER_FEATURES})
    for column in categorical:
        frame[column] = pd.Series(["x"], dtype="category")
    for column in ("n_items", "n_sellers", "product_photos_qty", "payment_installments"):
        frame[column] = pd.Series([1], dtype="int16")
    for column in ("product_description_length", "seller_prior_orders", "customer_prior_orders"):
        frame[column] = pd.Series([1], dtype="int32")
    for column in ("same_state", "purchase_month", "purchase_dow", "purchase_hour"):
        frame[column] = pd.Series([1], dtype="int8")
    check(frame, "OLIST_ORDER_FEATURES")
    assert not set(frame.columns).intersection(P["olist.features.forbidden"])


def test_split_monotone() -> None:
    dates = pd.to_datetime(["2018-01-01", "2018-04-01", "2018-06-30"])
    labels = pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "purchase_ts": dates,
            "label": pd.Series([1, 1, 1], dtype="int8"),
            "label_reason": pd.Series(["comment_match", "none", "not_delivered"], dtype="string"),
        }
    )
    assigned = assign_splits(labels, P)
    train = assigned.loc[assigned["split"] == "train", "purchase_ts"]
    validate = assigned.loc[assigned["split"] == "validate", "purchase_ts"]
    test = assigned.loc[assigned["split"] == "test", "purchase_ts"]
    assert train.max() < validate.min() < test.min()
    assert assigned["order_id"].is_unique


def test_prior_stats_are_strictly_past() -> None:
    dates = pd.to_datetime(["2018-01-01", "2018-01-11", "2018-04-20"])
    labels = pd.DataFrame(
        {
            "order_id": pd.Series(["a", "b", "c"], dtype="string"),
            "purchase_ts": dates,
            "label": pd.Series([1, 0, 0], dtype="int8"),
            "label_reason": pd.Series(
                ["comment_match", "none", "none"], dtype="string"
            ),
        }
    )
    items = pd.DataFrame(
        {
            "order_id": ["a", "b", "c"],
            "seller_id": ["seller", "seller", "seller"],
            "price": [1.0, 1.0, 1.0],
            "order_item_id": [1, 1, 1],
        }
    )
    orders = pd.DataFrame(
        {"order_id": ["a", "b", "c"], "customer_id": ["x", "y", "z"]}
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["x", "y", "z"],
            "customer_unique_id": ["ux", "uy", "uz"],
        }
    )

    history = _history_features(labels, items, orders, customers, P)

    # Ten days is inside the configured 75-day maturity window, so b cannot
    # observe a's outcome.  By c's purchase, both earlier outcomes are mature.
    assert int(history.loc["b", "seller_prior_orders"]) == 0
    assert int(history.loc["c", "seller_prior_orders"]) == 2
    train_rate = labels.loc[labels["purchase_ts"] < pd.Timestamp("2018-03-01"), "label"].mean()
    expected = (1 + float(P["olist.prior_shrinkage_n"]) * train_rate) / (
        2 + float(P["olist.prior_shrinkage_n"])
    )
    assert float(history.loc["c", "seller_prior_complaint_rate"]) == pytest.approx(expected)


def test_metrics_json_complete() -> None:
    path = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "phase0" / "metrics.json"
    if not path.exists():
        pytest.skip("run_phase0 is required before artifact checks")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    for key in ("pr_auc", "roc_auc", "brier", "ece", "operating_points", "reliability", "lorenz"):
        assert key in metrics
    assert len(metrics["operating_points"]) == len(P["olist.operating_recalls"])
    assert all(np.isfinite(float(metrics[key])) for key in ("pr_auc", "brier", "ece"))
    assert metrics["immature_drop_count"] == metrics["split_counts"]["immature"]


def test_validation_calibration_and_test_drift_diagnostic() -> None:
    path = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "phase0" / "metrics.json"
    if not path.exists():
        pytest.skip("run_phase0 is required before calibration checks")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    tolerance = float(P["models.calib.mean_tolerance"])
    validation = metrics["calibration_checks"]["validate"]
    validation_error = abs(validation["mean_prediction"] - validation["empirical_rate"]) / validation["empirical_rate"]
    assert validation_error < tolerance

    test = metrics["calibration_checks"]["test"]
    test_error = abs(test["mean_prediction"] - test["empirical_rate"]) / test["empirical_rate"]
    drift = metrics["temporal_drift"]
    assert drift["test_relative_mean_error"] == pytest.approx(test_error)
    assert drift["test_within_mean_tolerance"] is (test_error < tolerance)
    if test_error >= tolerance:
        assert drift["validation_to_test_prevalence_ratio"] > 1 + tolerance
        assert drift["largest_rate_drop_reason"] == "late_low_score"
        assert drift["label_reason_rate_drops"]["late_low_score"] > 0


def test_temporal_calibration_drift_is_attributed() -> None:
    path = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "phase0" / "metrics.json"
    if not path.exists():
        pytest.skip("run_phase0 is required before calibration checks")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    drift = metrics["temporal_drift"]
    assert drift["calibration_protocol"] == (
        "isotonic fitted on validation labels only; test labels used for evaluation only"
    )
    assert np.isfinite(metrics["calibration_checks"]["test"]["raw_mean_prediction"])
    assert len(drift["monthly_label_rates"]) == 6


def test_footer_present() -> None:
    source = (pathlib.Path(__file__).resolve().parents[1] / "vulcan_proof" / "olist" / "charts.py").read_text(encoding="utf-8")
    assert "report.olist_footer" in source
