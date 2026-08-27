"""Customer-level risk and dispute-type signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ..seeds import SeedTree


def build_risk(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    params: Params = P,
    kappa: float = 0.0,
    seed: int | None = None,
    shift_enabled: bool = False,
) -> pd.DataFrame:
    """Generate standardised hidden risk variables for each order."""
    if seed is None:
        seed = int(params["run.master_seed"])
    if not 0.0 <= float(kappa) <= 1.0:
        raise InvariantError("kappa must be within the configured [0, 1] dial")
    if len(orders) != len(customers):
        raise InvariantError("orders and customers are not aligned")
    rng = SeedTree(int(seed)).child("risk")
    noise = rng.normal(size=len(orders))
    type_noise = rng.normal(size=len(orders))
    new_customer = customers["new_customer"].to_numpy(dtype="float64")
    address_mismatch = customers["address_mismatch"].to_numpy(dtype="float64")
    prior_disputes = customers["prior_disputes"].to_numpy(dtype="float64")
    account_age = customers["account_age_days"].to_numpy(dtype="float64")
    base_multiplier = np.ones(len(orders), dtype="float64")
    if shift_enabled:
        test_months = params["sim.timeline.test_months"]
        month = orders["month"].to_numpy(dtype="int8")
        test_rows = (month >= int(test_months[0])) & (month <= int(test_months[1]))
        base_multiplier[test_rows] = float(params["sim.shift.risk_coef_multiplier"])
    raw = base_multiplier * (
        float(params["sim.kappa.coef_new_customer"]) * new_customer
        + float(params["sim.kappa.coef_address_mismatch"]) * address_mismatch
        + float(params["sim.kappa.coef_prior_disputes"]) * np.log1p(prior_disputes)
        - float(params["sim.kappa.coef_account_age"]) * np.log1p(
            account_age / float(params["sim.timeline.days_per_month"])
        )
        + float(params["sim.kappa.coef_noise"]) * noise
    )
    raw_std = float(raw.std())
    if not np.isfinite(raw_std) or raw_std <= 0.0:
        raise InvariantError("risk signal has no finite variation")
    z_risk = (raw - raw.mean()) / raw_std
    type_raw = address_mismatch + 0.5 * new_customer + type_noise
    type_std = float(type_raw.std())
    if not np.isfinite(type_std) or type_std <= 0.0:
        raise InvariantError("dispute-type signal has no finite variation")
    z_type = (type_raw - type_raw.mean()) / type_std
    kappa_value = float(kappa)
    risk_mult = np.exp(kappa_value * z_risk - kappa_value**2 / 2.0)
    return pd.DataFrame(
        {
            "hidden_z_risk": z_risk.astype("float32"),
            "hidden_z_type": z_type.astype("float32"),
            "hidden_risk_mult": risk_mult.astype("float32"),
        }
    )
