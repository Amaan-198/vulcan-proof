"""Small dtype and categorical helpers shared by simulator modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def string_series(values: Iterable[object]) -> pd.Series:
    """Create a compact pandas string series backed by Arrow when available."""
    return pd.Series(list(values), dtype="string[pyarrow]")


def category_series(values: Sequence[object], categories: Sequence[object]) -> pd.Series:
    """Create an ordered categorical series with a stable category dictionary."""
    return pd.Series(pd.Categorical(values, categories=list(categories), ordered=False))


def category_lookup(values: Sequence[object], names: Sequence[str], mapping: dict[str, float]) -> np.ndarray:
    """Map categorical values to a numeric array without an implicit default."""
    result = np.empty(len(values), dtype="float64")
    for name in names:
        if name not in mapping:
            raise KeyError(name)
        result[np.asarray(values, dtype=object) == name] = float(mapping[name])
    if not np.isfinite(result).all():
        raise ValueError("category lookup produced a non-finite value")
    return result
