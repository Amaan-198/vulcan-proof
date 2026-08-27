"""End-to-end Phase-1 hidden-truth world generator."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params, load
from ..schemas import ORDER_HIDDEN, check
from .ack import response_uniforms
from .disputes import draw_disputes
from .latency import draw_latency
from .splits import assign_splits
from .world import build_latent_world


def _calibration_payload(
    params: Params,
    theta_path: str | pathlib.Path | None,
) -> dict[str, object]:
    """Read the required derived calibration file."""
    if theta_path is None:
        root = params.path.resolve().parents[1]
        path = root / "outputs" / "theta.json"
    else:
        path = pathlib.Path(theta_path).resolve()
    if not path.exists():
        raise InvariantError(f"derived calibration file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("theta", "gamma"):
        if key not in payload:
            raise InvariantError(f"derived calibration is missing {key}: {path}")
    return payload


def _nullable_int(values: np.ndarray) -> pd.Series:
    """Represent -1 sentinels as pandas nullable Int32 values."""
    result = pd.array(values, dtype="Int32")
    result[values < 0] = pd.NA
    return pd.Series(result)


def generate_world(
    kappa: float,
    seed: int,
    n_orders: int,
    shift_enabled: bool,
    params_path: str | pathlib.Path | None = None,
    theta_path: str | pathlib.Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate observed and hidden Phase-1 order frames."""
    params = P if params_path is None else load(params_path)
    calibration = _calibration_payload(params, theta_path)
    world = build_latent_world(
        kappa=float(kappa),
        seed=int(seed),
        n_orders=int(n_orders),
        shift_enabled=bool(shift_enabled),
        params=params,
    )
    theta = float(calibration["theta"])
    gamma_payload = calibration["gamma"]
    if not isinstance(gamma_payload, dict):
        raise InvariantError("derived gamma must be a mapping")
    gamma = {
        name: float(gamma_payload[name]) for name in params["categories.order"]
    }
    potential, dispute_type, _, _ = draw_disputes(
        world.observed,
        world.truth,
        world.risk_mult,
        world.z_type,
        world.requested_bitmask,
        theta,
        gamma,
        kappa=float(kappa),
        params=params,
        seed=int(seed),
    )
    order_day = world.observed["decision_date"].to_numpy(dtype="int32")
    open_day, resolution_day = draw_latency(
        order_day, potential, params=params, seed=int(seed)
    )
    split_data = assign_splits(
        order_day, open_day, resolution_day, params=params, potential=potential
    )
    observed = world.observed.copy()
    observed["split"] = split_data["split"]
    observed["censored"] = split_data["censored"].to_numpy(dtype="int8")
    observed["order_day"] = order_day
    observed = observed.loc[
        :, list(params["features.permitted"]) + ["split", "censored", "order_day"]
    ]
    check(observed, "ORDER_OBSERVED")

    merchant_index = world.merchant_index
    merchants = world.merchants
    hidden = pd.DataFrame(
        {
            "order_id": observed["order_id"].copy(),
            "hidden_truth": world.truth,
            "hidden_z_risk": world.z_risk,
            "hidden_z_type": world.z_type,
            "hidden_risk_mult": world.risk_mult,
            "hidden_quality": merchants.loc[merchant_index, "hidden_quality"].to_numpy(dtype="float32"),
            "hidden_carrier_reliability": merchants.loc[
                merchant_index, "carrier_reliability"
            ].to_numpy(dtype="float32"),
            "hidden_archetype": merchants.loc[
                merchant_index, "hidden_archetype"
            ].astype("category").reset_index(drop=True),
            "hidden_compliance": merchants.loc[
                merchant_index, "hidden_compliance"
            ].to_numpy(dtype="float32"),
            "hidden_contest_base": merchants.loc[
                merchant_index, "hidden_contest_base"
            ].to_numpy(dtype="float32"),
            "hidden_requested_bitmask": world.requested_bitmask.astype("uint16"),
            "hidden_random_stratum": world.random_stratum.astype("int8"),
            "hidden_dispute_potential": potential.astype("int8"),
            "hidden_dispute_type": dispute_type,
            "hidden_dispute_open_day": _nullable_int(open_day),
            "hidden_resolution_day": _nullable_int(resolution_day),
            "hidden_u_response": response_uniforms(
                len(observed), params=params, seed=int(seed)
            ),
        }
    )
    hidden = hidden.loc[:, list(ORDER_HIDDEN)]
    check(hidden, "ORDER_HIDDEN")
    return observed, hidden
