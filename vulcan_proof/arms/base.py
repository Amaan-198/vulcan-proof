"""Observed-frame validation and availability handling for arms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ..schemas import PLAN, check


def evidence_names(params: Params = P) -> tuple[str, ...]:
    """Return evidence names in their configured bit order."""
    return tuple(params["evidence.order"])


def bit_for(name: str, params: Params = P) -> np.uint16:
    """Return the configured bit for one evidence type."""
    names = evidence_names(params)
    if name not in names:
        raise InvariantError(f"unknown evidence type: {name!r}")
    return np.uint16(1 << names.index(name))


def available_mask(observed: pd.DataFrame, params: Params = P) -> np.ndarray:
    """Return the per-order mask permitted by tier and acknowledgement opt-in."""
    tiers = observed["eligible_tier"].astype(str).to_numpy()
    opt_in = observed["ack_optin"].to_numpy(dtype="int8")
    mask = np.zeros(len(observed), dtype="uint16")
    post_only = tiers == "POST_DELIVERY_ONLY"
    none = tiers == "NONE"
    for name in evidence_names(params):
        bit = bit_for(name, params)
        permitted = ~none
        if name not in {"ack", "vack"}:
            permitted &= ~post_only
        else:
            permitted &= opt_in.astype(bool)
        mask[permitted] |= bit
    return mask


def gate_masks(
    observed: pd.DataFrame,
    requested_bitmask: np.ndarray,
    params: Params = P,
) -> np.ndarray:
    """Apply the availability mask to candidate plan bitmasks."""
    if len(requested_bitmask) != len(observed):
        raise InvariantError("plan and observed frame are not aligned")
    values = np.asarray(requested_bitmask, dtype="uint16")
    known = np.uint16((1 << len(evidence_names(params))) - 1)
    if bool((values & np.uint16(~known)).any()):
        raise InvariantError("plan contains an unknown evidence bit")
    return np.bitwise_and(values, available_mask(observed, params))


def make_plan(
    observed: pd.DataFrame,
    requested_bitmask: np.ndarray,
    arm_id: str,
    params: Params = P,
) -> pd.DataFrame:
    """Validate an observed frame and return a strict PLAN frame."""
    from ..schemas import cast

    check(observed, "ORDER_OBSERVED")
    values = gate_masks(observed, requested_bitmask, params)
    result = pd.DataFrame(
        {
            "order_id": observed["order_id"].astype("string").reset_index(drop=True),
            "requested_bitmask": values.astype("uint16"),
        }
    )
    del arm_id
    return cast(result, "PLAN")


def all_requested_mask(params: Params = P) -> np.uint16:
    """Return the mask containing every configured evidence type."""
    return np.uint16((1 << len(evidence_names(params))) - 1)

