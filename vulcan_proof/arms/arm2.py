"""Arm 2: request every evidence type allowed for the order."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..params import P, Params
from .base import all_requested_mask, make_plan


def plan(observed: pd.DataFrame, params: Params = P) -> pd.DataFrame:
    """Return the availability-gated full evidence plan."""
    requested = np.full(
        len(observed),
        all_requested_mask(params),
        dtype="uint16",
    )
    return make_plan(observed, requested, "arm2", params)


arm2 = plan

