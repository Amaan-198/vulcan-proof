"""Temporal Olist split and maturity handling."""

from __future__ import annotations

import pathlib
from typing import Any

import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ..schemas import check


def assign_splits(labels: pd.DataFrame, params: Params = P) -> pd.DataFrame:
    """Assign each eligible label to train, validate, test, or immature."""
    check(labels, "OLIST_LABELS")
    result = labels.copy()
    purchase = result["purchase_ts"]
    train_end = pd.Timestamp(params["olist.split.train_end"]) + pd.Timedelta(days=1)
    validate_end = pd.Timestamp(params["olist.split.validate_end"]) + pd.Timedelta(days=1)
    dataset_end = pd.Timestamp(params["olist.label.dataset_end"])
    maturity = int(params["olist.split.maturity_days"])
    mature_end = dataset_end - pd.to_timedelta(maturity, unit="D") + pd.Timedelta(days=1)
    split = pd.Series("immature", index=result.index, dtype="string")
    split.loc[purchase < train_end] = "train"
    split.loc[(purchase >= train_end) & (purchase < validate_end)] = "validate"
    split.loc[(purchase >= validate_end) & (purchase < mature_end)] = "test"
    result["split"] = split
    for name in ("train", "validate", "test"):
        part = result.loc[result["split"] == name]
        if part.empty:
            raise InvariantError(f"Olist {name} split is empty")
        if int(part["label"].sum()) == 0:
            raise InvariantError(f"Olist {name} split has no positive label")
    return result


def split_statistics(split_labels: pd.DataFrame) -> dict[str, int]:
    """Count rows in each assigned temporal split."""
    return {
        str(name): int(count)
        for name, count in split_labels["split"].value_counts().sort_index().items()
    }
