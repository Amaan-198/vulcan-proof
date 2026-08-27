"""Customer-side observable features for simulated orders."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..params import P, Params
from ..seeds import SeedTree
from ._helpers import string_series


def build_customers(
    order_ids: Sequence[object],
    params: Params = P,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate per-order customer features with a small repeat-customer pool."""
    if seed is None:
        seed = int(params["run.master_seed"])
    order_count = len(order_ids)
    rng = SeedTree(int(seed)).child("customers")
    pool_size = max(100, order_count // 100)
    pool_indices = rng.integers(0, pool_size, size=order_count)
    new_customer = (
        rng.random(order_count) < float(params["sim.customer_features.p_new_customer"])
    ).astype("int8")
    address_mismatch = (
        rng.random(order_count)
        < float(params["sim.customer_features.p_address_mismatch"])
    ).astype("int8")
    prior_disputes = rng.poisson(
        float(params["sim.customer_features.prior_disputes_poisson_mean"]),
        size=order_count,
    ).astype("int16")
    account_age_days = rng.lognormal(
        mean=np.log(float(params["sim.customer_features.account_age_lognormal_median_days"])),
        sigma=1.0,
        size=order_count,
    ).astype("float32")
    prior_disputes[new_customer == 1] = 0
    account_age_days[new_customer == 1] = 0.0
    return pd.DataFrame(
        {
            "order_id": string_series(order_ids),
            "customer_id": string_series(
                f"customer_{int(index)}" for index in pool_indices
            ),
            "new_customer": new_customer,
            "address_mismatch": address_mismatch,
            "prior_disputes": prior_disputes,
            "account_age_days": account_age_days,
        }
    )
