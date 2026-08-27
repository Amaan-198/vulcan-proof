"""Olist PR and calibration charts with the required public-data footer."""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from ..params import P, Params


def write_charts(
    output_dir: pathlib.Path,
    y_true: np.ndarray,
    scores: np.ndarray,
    reliability: list[dict[str, Any]],
    params: Params = P,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write deterministic PR and reliability PNGs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    footer = str(params["report.olist_footer"])
    precision, recall, _ = precision_recall_curve(y_true, scores)
    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(recall, precision)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Olist fulfillment-complaint precision-recall")
    figure.text(0.5, 0, footer, ha="center")
    figure.tight_layout()
    pr_path = output_dir / "pr_curve.png"
    figure.savefig(pr_path, dpi=100)
    plt.close(figure)

    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1)
    populated = [row for row in reliability if row["count"]]
    axis.plot(
        [row["mean_prediction"] for row in populated],
        [row["empirical_rate"] for row in populated],
        marker="o",
    )
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlabel("Mean prediction")
    axis.set_ylabel("Empirical rate")
    axis.set_title("Olist reliability diagram")
    figure.text(0.5, 0, footer, ha="center")
    figure.tight_layout()
    reliability_path = output_dir / "reliability.png"
    figure.savefig(reliability_path, dpi=100)
    plt.close(figure)
    return pr_path, reliability_path
