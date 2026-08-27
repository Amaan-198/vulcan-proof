"""Historical merchant evidence-request policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..params import P, Params
from ..seeds import SeedTree


def build_historical_policy(
    orders: pd.DataFrame,
    merchants: pd.DataFrame,
    params: Params = P,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return historical evidence request bitmasks and random-stratum flags."""
    if seed is None:
        seed = int(params["run.master_seed"])
    count = len(orders)
    evidence = list(params["evidence.order"])
    rng = SeedTree(int(seed)).child("policy")
    merchant_index = orders["merchant_index"].to_numpy(dtype="int32")
    archetype = merchants.loc[merchant_index, "hidden_archetype"].astype(str).to_numpy()
    tier = merchants.loc[merchant_index, "eligible_tier"].astype(str).to_numpy()
    ack_optin = merchants.loc[merchant_index, "ack_optin"].to_numpy(dtype="int8")
    order_value = orders["order_value"].to_numpy(dtype="float64")
    random_stratum = (
        rng.random(count) < float(params["archetypes.random_stratum_frac"])
    ).astype("int8")
    bitmask = np.zeros(count, dtype="uint16")
    archetypes = list(params["archetypes.order"])
    for archetype_name in archetypes:
        archetype_rows = archetype == archetype_name
        policy = params[f"archetypes.{archetype_name}"]["policy"]
        for evidence_index, evidence_name in enumerate(evidence):
            bit = np.uint16(1 << evidence_index)
            if policy == "random":
                requested = rng.random(count) < 0.5
            elif evidence_name in policy:
                requested = order_value >= float(policy[evidence_name])
            else:
                requested = np.zeros(count, dtype=bool)
            bitmask[archetype_rows & requested] |= bit

    random_rows = random_stratum == 1
    for evidence_index, evidence_name in enumerate(evidence):
        bit = np.uint16(1 << evidence_index)
        random_requested = rng.random(count) < 0.5
        bitmask[random_rows & random_requested] |= bit
        bitmask[random_rows & ~random_requested] &= np.uint16(~bit)

    post_only = tier == "POST_DELIVERY_ONLY"
    none = tier == "NONE"
    for evidence_index, evidence_name in enumerate(evidence):
        bit = np.uint16(1 << evidence_index)
        if evidence_name in {"ack", "vack"}:
            bitmask[ack_optin == 0] &= np.uint16(~bit)
        allowed_post = evidence_name in {"ack", "vack"}
        if not allowed_post:
            bitmask[post_only] &= np.uint16(~bit)
        bitmask[none] &= np.uint16(~bit)
    return bitmask, random_stratum
