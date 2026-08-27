"""Maturity-safe observed outcome labels."""

from __future__ import annotations

import pandas as pd

from ..errors import InvariantError
from ..schemas import check


def eligible(outcome_df: pd.DataFrame, split: str, *, purpose: str = "fit") -> pd.Series:
    """Return uncensored rows eligible for a train/validation operation.

    Evaluation callers may request a gap or test mask explicitly, but fitting
    callers are rejected for those splits so the restriction cannot be
    bypassed by a model module.
    """
    check(outcome_df, "OUTCOME")
    split_name = str(split)
    if purpose == "fit" and split_name not in {"train", "validate"}:
        raise InvariantError(f"labels from {split_name!r} are not eligible for fitting")
    values = outcome_df["split"].astype(str).eq(split_name)
    values &= outcome_df["censored"].eq(0)
    return values.astype(bool)


label_mask = eligible
