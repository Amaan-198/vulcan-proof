"""Order and payment feature generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..params import P, Params
from ..seeds import SeedTree
from ._helpers import category_series, string_series


def build_orders(
    merchants: pd.DataFrame,
    params: Params = P,
    seed: int | None = None,
    n_orders: int | None = None,
    shift_enabled: bool = False,
) -> pd.DataFrame:
    """Generate payment-time order features and merchant references."""
    if seed is None:
        seed = int(params["run.master_seed"])
    if n_orders is None:
        n_orders = int(params["run.n_orders_smoke"])
    if n_orders < 1:
        raise ValueError("n_orders must be positive")
    tree = SeedTree(int(seed))
    day_rng = tree.child("orders", "day")
    merchant_rng = tree.child("orders", "merchant")
    shift_rng = tree.child("orders", "shift_merchant")
    value_rng = tree.child("orders", "value")
    payment_rng = tree.child("orders", "payment")
    days_per_month = int(params["sim.timeline.days_per_month"])
    month_range = params["sim.timeline.order_months"]
    first_day = (int(month_range[0]) - 1) * days_per_month
    end_day = int(month_range[1]) * days_per_month
    order_day = day_rng.integers(first_day, end_day, size=n_orders, dtype="int32")
    month = (order_day // days_per_month + 1).astype("int8")
    test_months = params["sim.timeline.test_months"]
    is_test = (month >= int(test_months[0])) & (month <= int(test_months[1]))

    merchant_count = len(merchants)
    base_weights = merchants["size_weight"].to_numpy(dtype="float64")
    merchant_indices = merchant_rng.choice(
        merchant_count, size=n_orders, p=base_weights
    ).astype("int32")
    if shift_enabled and bool(is_test.any()):
        categories = list(params["categories.order"])
        shifted_shares = params["sim.shift.category_share_shift"]
        shifted_weights = np.zeros(merchant_count, dtype="float64")
        merchant_categories = merchants["category"].astype(str).to_numpy()
        for category in categories:
            category_rows = merchant_categories == category
            category_weight = base_weights[category_rows]
            shifted_weights[category_rows] = (
                float(shifted_shares[category])
                * category_weight
                / category_weight.sum()
            )
        shifted_indices = shift_rng.choice(
            merchant_count, size=int(is_test.sum()), p=shifted_weights
        ).astype("int32")
        merchant_indices[is_test] = shifted_indices

    merchant_categories = merchants["category"].astype(str).to_numpy()
    order_categories = merchant_categories[merchant_indices]
    category_vmin = {
        category: float(params[f"categories.{category}"]["vmin"])
        for category in params["categories.order"]
    }
    category_vmax = {
        category: float(params[f"categories.{category}"]["vmax"])
        for category in params["categories.order"]
    }
    value_min = np.asarray([category_vmin[name] for name in order_categories])
    value_max = np.asarray([category_vmax[name] for name in order_categories])
    log_min = np.log(value_min)
    log_max = np.log(value_max)
    order_value = np.exp(log_min + value_rng.random(n_orders) * (log_max - log_min))

    method_values = np.asarray(["card", "upi", "netbanking", "wallet"], dtype=object)
    payment_method = payment_rng.choice(method_values, size=n_orders)
    network = np.where(
        payment_rng.random(n_orders) < float(params["sim.network_visa_share"]),
        "Visa",
        "Mastercard",
    )
    issuer_count = int(params["sim.n_issuer_families"])
    issuer_indices = payment_rng.integers(0, issuer_count, size=n_orders)
    issuer_family = np.char.add("issuer_", issuer_indices.astype(str))
    hours_in_day = int(pd.Timedelta(days=1).total_seconds() // 3600)
    hour_of_day = payment_rng.integers(0, hours_in_day, size=n_orders).astype("int8")
    cart_items = (1 + payment_rng.poisson(1, size=n_orders)).astype("int16")
    order_ids = string_series(
        f"order_{int(seed)}_{index:07d}" for index in range(n_orders)
    )
    return pd.DataFrame(
        {
            "order_id": order_ids,
            "merchant_index": merchant_indices,
            "category": category_series(order_categories, params["categories.order"]),
            "order_value": order_value.astype("float32"),
            "payment_method": category_series(
                payment_method, method_values.tolist()
            ),
            "network": category_series(network, ["Visa", "Mastercard"]),
            "issuer_family": category_series(
                issuer_family, [f"issuer_{index}" for index in range(issuer_count)]
            ),
            "cart_items": cart_items,
            "hour_of_day": hour_of_day,
            "month": month,
            "order_day": order_day,
            "decision_date": order_day.copy(),
        }
    )
