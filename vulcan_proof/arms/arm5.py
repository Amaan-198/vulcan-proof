"""Arm 5: truth-blind per-order exhaustive evidence planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..errors import InvariantError
from ..opt.optimizer import PlanResult, best_plan, plan_frame
from ..params import P, Params
from ..schemas import check
from .base import make_plan


@dataclass
class Arm5Policy:
    """Fitted model bundle and support mask used by Arm 5."""

    models: Any
    masks: Any = None
    params: Params = P

    def plan(
        self,
        observed_orders: pd.DataFrame,
        *,
        details: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, list[PlanResult]]:
        """Plan in bounded chunks while retaining row-local model calls."""
        check(observed_orders, "ORDER_OBSERVED")
        chunk_size = int(self.params["run.ev_chunk_orders"])
        if chunk_size < 1:
            raise InvariantError("EV chunk size must be positive")
        masks: list[int] = []
        explanations: list[PlanResult] = []
        for start in range(0, len(observed_orders), chunk_size):
            chunk = observed_orders.iloc[start : start + chunk_size]
            if details:
                for _, row in chunk.iterrows():
                    result = best_plan(row, self.models, self.masks, self.params)
                    masks.append(result.requested_bitmask)
                    explanations.append(result)
            else:
                masks.extend(plan_frame(chunk, self.models, self.masks, self.params).astype("uint16").tolist())
        planned = make_plan(
            observed_orders,
            pd.Series(masks, dtype="uint16").to_numpy(),
            "arm5",
            self.params,
        )
        return (planned, explanations) if details else planned


def plan(
    observed_orders: pd.DataFrame,
    models: Any,
    masks: Any = None,
    params: Params = P,
    *,
    details: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[PlanResult]]:
    """Apply Arm 5 to an observed frame."""
    return Arm5Policy(models, masks, params).plan(observed_orders, details=details)


arm5 = plan
