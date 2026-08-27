"""Arm 3: the deliberately naive category-by-value rules from parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from .base import bit_for, make_plan


def plan(observed: pd.DataFrame, params: Params = P) -> pd.DataFrame:
    """Apply the configured literal threshold rules and availability gates."""
    from ..schemas import check

    check(observed, "ORDER_OBSERVED")
    category = observed["category"].astype(str).to_numpy()
    value = observed["order_value"].to_numpy(dtype="float64")
    result = np.zeros(len(observed), dtype="uint16")
    for category_name in params["categories.order"]:
        rules = params[f"arms.arm3_rules"][category_name]
        rows = category == category_name
        for evidence_name, threshold in rules.items():
            bit = bit_for(evidence_name, params)
            if not np.isfinite(float(threshold)):
                raise InvariantError("Arm 3 threshold is not finite")
            result[rows & (value >= float(threshold))] |= bit
    return make_plan(observed, result, "arm3", params)


arm3 = plan

