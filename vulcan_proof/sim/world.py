"""Construction of the pre-dispute latent world shared by calibration and runs."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib

import numpy as np
import pandas as pd

from ..params import P, Params
from ._helpers import category_series, string_series
from .customers import build_customers
from .merchants import build_merchants
from .orders import build_orders
from .policy import build_historical_policy
from .risk import build_risk
from .truth import draw_truth, truth_probabilities


@dataclass
class LatentWorld:
    """All generator state needed before potential disputes and latency are drawn."""

    observed: pd.DataFrame
    merchants: pd.DataFrame
    truth: pd.Series
    z_type: np.ndarray
    risk_mult: np.ndarray
    z_risk: np.ndarray
    requested_bitmask: np.ndarray
    random_stratum: np.ndarray
    merchant_index: np.ndarray


def _merchant_order_count(orders: pd.DataFrame) -> np.ndarray:
    """Count each merchant's earlier generated orders without hidden inputs."""
    work = pd.DataFrame(
        {
            "merchant_index": orders["merchant_index"].to_numpy(dtype="int32"),
            "order_day": orders["order_day"].to_numpy(dtype="int32"),
            "position": np.arange(len(orders), dtype="int32"),
        }
    ).sort_values(["merchant_index", "order_day", "position"], kind="mergesort")
    work["count"] = work.groupby("merchant_index", sort=False).cumcount()
    result = np.empty(len(orders), dtype="int32")
    result[work["position"].to_numpy()] = work["count"].to_numpy(dtype="int32")
    return result


def build_latent_world(
    kappa: float,
    seed: int,
    n_orders: int,
    shift_enabled: bool,
    params: Params = P,
) -> LatentWorld:
    """Build orders, observed features, truth, risk, and historical policy."""
    merchants = build_merchants(params, seed=seed)
    orders = build_orders(
        merchants,
        params=params,
        seed=seed,
        n_orders=n_orders,
        shift_enabled=shift_enabled,
    )
    customers = build_customers(orders["order_id"].tolist(), params=params, seed=seed)
    risk = build_risk(
        orders,
        customers,
        params=params,
        kappa=kappa,
        seed=seed,
        shift_enabled=shift_enabled,
    )
    probabilities = truth_probabilities(orders, merchants, params=params)
    truth = draw_truth(probabilities, params=params, seed=seed)
    requested_bitmask, random_stratum = build_historical_policy(
        orders, merchants, params=params, seed=seed
    )
    merchant_index = orders["merchant_index"].to_numpy(dtype="int32")
    merchant_id = merchants.loc[merchant_index, "merchant_id"].reset_index(drop=True)
    tier = merchants.loc[merchant_index, "eligible_tier"].reset_index(drop=True)
    verified = merchants.loc[
        merchant_index, "verified_contact_available"
    ].to_numpy(dtype="int8")
    ack_optin = merchants.loc[merchant_index, "ack_optin"].to_numpy(dtype="int8")
    category_values = orders["category"].astype(str).to_numpy()
    history_nan = np.full(n_orders, np.nan, dtype="float32")
    observed = pd.DataFrame(
        {
            "order_id": string_series(orders["order_id"].tolist()),
            "merchant_id": string_series(merchant_id.astype(str).tolist()),
            "customer_id": string_series(customers["customer_id"].tolist()),
            "category": category_series(category_values, params["categories.order"]),
            "order_value": orders["order_value"].to_numpy(dtype="float32"),
            "payment_method": orders["payment_method"].astype("category"),
            "network": orders["network"].astype("category"),
            "issuer_family": orders["issuer_family"].astype("category"),
            "new_customer": customers["new_customer"].to_numpy(dtype="int8"),
            "address_mismatch": customers["address_mismatch"].to_numpy(dtype="int8"),
            "prior_disputes": customers["prior_disputes"].to_numpy(dtype="int16"),
            "account_age_days": customers["account_age_days"].to_numpy(dtype="float32"),
            "cart_items": orders["cart_items"].to_numpy(dtype="int16"),
            "hour_of_day": orders["hour_of_day"].to_numpy(dtype="int8"),
            "month": orders["month"].to_numpy(dtype="int8"),
            "merchant_order_count": _merchant_order_count(orders),
            "merchant_dispute_rate_hist": history_nan.copy(),
            "merchant_contest_rate_hist": history_nan.copy(),
            "merchant_compliance_hist": history_nan.copy(),
            "verified_contact_available": verified,
            "ack_optin": ack_optin,
            "eligible_tier": tier.astype("category"),
            "decision_date": orders["decision_date"].to_numpy(dtype="int32"),
        }
    )
    return LatentWorld(
        observed=observed,
        merchants=merchants,
        truth=truth,
        z_type=risk["hidden_z_type"].to_numpy(dtype="float32"),
        risk_mult=risk["hidden_risk_mult"].to_numpy(dtype="float32"),
        z_risk=risk["hidden_z_risk"].to_numpy(dtype="float32"),
        requested_bitmask=requested_bitmask,
        random_stratum=random_stratum,
        merchant_index=merchant_index,
    )
