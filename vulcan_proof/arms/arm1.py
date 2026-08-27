"""Arm 1: no evidence is requested."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..params import P, Params
from .base import make_plan


def plan(observed: pd.DataFrame, params: Params = P) -> pd.DataFrame:
    """Return the empty plan for every observed order."""
    return make_plan(
        observed,
        np.zeros(len(observed), dtype="uint16"),
        "arm1",
        params,
    )


arm1 = plan

