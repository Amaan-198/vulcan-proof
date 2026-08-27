"""Historical policy plan used to create the observable merchant history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ..schemas import check
from ..arms.base import make_plan


def plan(
    historical_orders: pd.DataFrame,
    observed_orders: pd.DataFrame | None = None,
    params: Params = P,
) -> pd.DataFrame:
    """Return arm 0's historical request set from the latent policy column."""
    check(historical_orders, "ORDER_HIDDEN")
    masks = historical_orders["hidden_requested_bitmask"].to_numpy(dtype="uint16")
    if observed_orders is None:
        result = pd.DataFrame(
            {
                "order_id": historical_orders["order_id"].astype("string").reset_index(drop=True),
                "requested_bitmask": masks,
            }
        )
        from ..schemas import cast

        return cast(result, "PLAN")
    check(observed_orders, "ORDER_OBSERVED")
    if not historical_orders["order_id"].astype(str).equals(
        observed_orders["order_id"].astype(str)
    ):
        if set(historical_orders["order_id"].astype(str)) != set(observed_orders["order_id"].astype(str)):
            raise InvariantError("historical and observed order ids do not match")
        lookup = pd.Series(masks, index=historical_orders["order_id"].astype(str))
        masks = observed_orders["order_id"].astype(str).map(lookup).to_numpy(dtype="uint16")
    return make_plan(observed_orders, masks, "arm0", params)


arm0 = plan

