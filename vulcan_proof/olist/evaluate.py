"""Test-window metrics for the Olist detector."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

from ..errors import InvariantError
from ..params import P, Params


def _reliability(y_true: np.ndarray, scores: np.ndarray, bins: int) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bins - 1:
            selected = (scores >= lower) & (scores <= upper)
        else:
            selected = (scores >= lower) & (scores < upper)
        count = int(selected.sum())
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "mean_prediction": float(scores[selected].mean()) if count else 0.0,
                "empirical_rate": float(y_true[selected].mean()) if count else 0.0,
                "count": count,
            }
        )
    return rows


def _ece(reliability: list[dict[str, Any]], total: int) -> float:
    if total == 0:
        raise InvariantError("cannot calculate ECE for an empty test set")
    return float(
        sum(
            abs(row["mean_prediction"] - row["empirical_rate"]) * row["count"] / total
            for row in reliability
        )
    )


def _operating_point(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
    seconds: float,
    hourly_rate: float,
) -> dict[str, float]:
    positive_count = int(y_true.sum())
    if positive_count == 0:
        raise InvariantError("operating recall requires at least one positive test label")
    order = np.argsort(-scores, kind="mergesort")
    positive_order = order[y_true[order] == 1]
    required = max(1, int(np.ceil(target_recall * positive_count)))
    threshold = float(scores[positive_order[required - 1]])
    flagged = scores >= threshold
    true_positive = int((flagged & (y_true == 1)).sum())
    false_positive = int((flagged & (y_true == 0)).sum())
    flagged_count = int(flagged.sum())
    precision = true_positive / flagged_count if flagged_count else 0.0
    recall = true_positive / positive_count
    fp_per_thousand = false_positive / len(y_true) * 1000
    fp_cost = fp_per_thousand * seconds * hourly_rate / 3600
    return {
        "target_recall": float(target_recall),
        "achieved_recall": float(recall),
        "threshold": threshold,
        "precision": float(precision),
        "flagged_fraction": float(flagged_count / len(y_true)),
        "false_positives_per_1000": float(fp_per_thousand),
        "fp_cost_inr_per_1000": float(fp_cost),
    }


def _lorenz(y_true: np.ndarray, scores: np.ndarray, deciles: int) -> tuple[list[dict[str, float]], float]:
    positive_count = int(y_true.sum())
    if positive_count == 0:
        raise InvariantError("Lorenz table requires at least one positive label")
    order = np.argsort(-scores, kind="mergesort")
    rows: list[dict[str, float]] = []
    parts = np.array_split(order, deciles)
    for index, part in enumerate(parts):
        share = float(y_true[part].sum() / positive_count)
        rows.append(
            {
                "decile": float(index + 1),
                "share_of_positives": share,
                "count": float(len(part)),
            }
        )
    top_share = rows[0]["share_of_positives"]
    lift = top_share / (1 / deciles)
    return rows, float(lift)


def evaluate_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    params: Params = P,
) -> dict[str, Any]:
    """Calculate all required metrics from test labels and calibrated scores."""
    y = np.asarray(y_true, dtype="int8")
    prediction = np.asarray(scores, dtype="float64")
    if y.ndim != 1 or prediction.ndim != 1 or len(y) != len(prediction) or len(y) == 0:
        raise InvariantError("test labels and scores must be non-empty aligned vectors")
    if not set(np.unique(y)).issubset({0, 1}):
        raise InvariantError("labels must be binary")
    if not np.isfinite(prediction).all() or ((prediction < 0) | (prediction > 1)).any():
        raise InvariantError("scores must be finite probabilities")
    reliability = _reliability(y, prediction, int(params["models.calib.ece_bins"]))
    operating = [
        _operating_point(
            y,
            prediction,
            float(recall),
            float(params["olist.fp_cost_proxy_seconds"]),
            float(params["econ.hourly_rate"]),
        )
        for recall in params["olist.operating_recalls"]
    ]
    lorenz, lift = _lorenz(y, prediction, int(params["report.lorenz_deciles"]))
    empirical = float(y.mean())
    calibrated_mean = float(prediction.mean())
    return {
        "pr_auc": float(average_precision_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, prediction)),
        "brier": float(brier_score_loss(y, prediction)),
        "ece": _ece(reliability, len(y)),
        "calibrated_mean": calibrated_mean,
        "empirical_rate": empirical,
        "calibrated_mean_vs_empirical_rate_ratio": calibrated_mean / empirical if empirical else float("inf"),
        "reliability": reliability,
        "operating_points": operating,
        "lorenz": lorenz,
        "top_decile_lift": lift,
    }
