"""Temporal splits and explicit censoring flags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ._helpers import category_series


SPLIT_NAMES = ["train", "validate", "gap", "test", "immature"]


def assign_splits(
    order_day: np.ndarray,
    open_day: np.ndarray,
    resolution_day: np.ndarray,
    params: Params = P,
    potential: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assign month-based splits and mark rows whose outcomes are censored."""
    if not (len(order_day) == len(open_day) == len(resolution_day)):
        raise InvariantError("latency and order arrays are not aligned")
    days_per_month = int(params["sim.timeline.days_per_month"])
    month = order_day // days_per_month + 1
    split_values = np.full(len(order_day), "immature", dtype=object)
    split_ranges = {
        "train": params["sim.timeline.train_months"],
        "validate": params["sim.timeline.validate_months"],
        "gap": params["sim.timeline.gap_months"],
        "test": params["sim.timeline.test_months"],
    }
    for name, month_range in split_ranges.items():
        rows = (month >= int(month_range[0])) & (month <= int(month_range[1]))
        split_values[rows] = name
    boundary = np.full(
        len(order_day),
        int(params["sim.timeline.outcome_observed_through_month"]) * days_per_month,
        dtype="int32",
    )
    decision_boundary = int(params["sim.timeline.decision_month_end"]) * days_per_month
    boundary[np.isin(split_values, ["train", "validate", "gap"])] = decision_boundary
    full_window = (
        int(params["sim.latency.expected_delivery_days"])
        + int(params["sim.latency.dispute_max_days"])
        + int(params["sim.latency.response_days"])
        + int(params["sim.latency.resolution_lognormal_p95_days"])
    )
    has_open = open_day >= 0
    no_open_could_still_dispute = ~has_open
    if potential is not None:
        if len(potential) != len(order_day):
            raise InvariantError("potential and order arrays are not aligned")
        # A generated row with no potential has no downstream event to
        # censor.  This keeps the synthetic negative label observable while
        # retaining the conservative no-open branch for direct callers.
        no_open_could_still_dispute &= potential.astype(bool)
    censored = (has_open & (resolution_day > boundary)) | (
        no_open_could_still_dispute & (order_day + full_window > boundary)
    )
    return pd.DataFrame(
        {
            "split": category_series(split_values, SPLIT_NAMES),
            "censored": censored.astype("int8"),
        }
    )
