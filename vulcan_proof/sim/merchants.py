"""Merchant population generation for the hidden-truth simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ..seeds import SeedTree
from ._helpers import category_series, string_series


def build_merchants(params: Params = P, seed: int | None = None) -> pd.DataFrame:
    """Generate the merchant population and its hidden merchant attributes."""
    if seed is None:
        seed = int(params["run.master_seed"])
    rng = SeedTree(int(seed)).child("merchants")
    count = int(params["merchants.n_merchants"])
    categories = list(params["categories.order"])
    archetypes = list(params["archetypes.order"])
    tiers = ["FULL", "POST_DELIVERY_ONLY", "NONE"]

    size_raw = rng.lognormal(
        mean=0.0,
        sigma=float(params["merchants.size_lognormal_sigma"]),
        size=count,
    )
    size_weight = size_raw / size_raw.sum()
    quality_raw = rng.lognormal(
        mean=0.0,
        sigma=float(params["merchants.quality_lognormal_sigma"]),
        size=count,
    )
    hidden_quality = quality_raw / quality_raw.mean()

    ranked_archetypes = sorted(
        archetypes,
        key=lambda name: int(params[f"archetypes.{name}"]["quality_rank"]),
    )
    sorted_merchants = np.argsort(-hidden_quality, kind="mergesort")
    archetype_values = np.empty(count, dtype=object)
    cursor = 0
    for position, name in enumerate(ranked_archetypes):
        if position == len(ranked_archetypes) - 1:
            block_size = count - cursor
        else:
            block_size = int(
                np.rint(float(params[f"archetypes.{name}"]["share"]) * count)
            )
        archetype_values[sorted_merchants[cursor : cursor + block_size]] = name
        cursor += block_size
    if cursor != count:
        raise InvariantError("merchant archetype blocks do not cover the population")

    category_shares = np.asarray(
        [float(params[f"categories.{name}"]["share"]) for name in categories],
        dtype="float64",
    )
    merchant_categories = rng.choice(categories, size=count, p=category_shares)
    tier_draw = rng.random(count)
    full_share = float(params["merchants.tier_full"])
    post_share = float(params["merchants.tier_post_delivery_only"])
    tier_values = np.select(
        [tier_draw < full_share, tier_draw < full_share + post_share],
        [tiers[0], tiers[1]],
        default=tiers[2],
    )
    ack_optin = (
        rng.random(count) < float(params["archetypes.ack_optin_rate"])
    ).astype("int8")

    carrier_count = int(params["sim.n_carriers"])
    carrier_ids = rng.integers(0, carrier_count, size=count, dtype="int16")
    carrier_raw = rng.lognormal(
        mean=0.0,
        sigma=float(params["sim.carrier_reliability_sigma"]),
        size=carrier_count,
    )
    # Truth rates divide by reliability.  Normalising the reciprocal keeps
    # the population mean of the carrier multiplier at one, as intended by
    # the mean-normalised truth-base rates.
    carrier_reliability = carrier_raw * np.mean(1.0 / carrier_raw)
    merchant_carrier_reliability = carrier_reliability[carrier_ids]

    archetype_compliance = {
        name: float(params[f"archetypes.{name}"]["compliance"])
        for name in archetypes
    }
    archetype_contest = {
        name: float(params[f"archetypes.{name}"]["contest"]) for name in archetypes
    }
    archetype_array = np.asarray(archetype_values, dtype=object)
    frame = pd.DataFrame(
        {
            "merchant_id": string_series(
                f"merchant_{index:06d}" for index in range(count)
            ),
            "size_weight": size_weight.astype("float64"),
            "category": category_series(merchant_categories, categories),
            "eligible_tier": category_series(tier_values, tiers),
            "verified_contact_available": (tier_values != tiers[2]).astype("int8"),
            "ack_optin": ack_optin,
            "carrier_id": carrier_ids,
            "carrier_reliability": merchant_carrier_reliability.astype("float32"),
            "hidden_quality": hidden_quality.astype("float32"),
            "hidden_archetype": category_series(archetype_array, archetypes),
            "hidden_compliance": np.asarray(
                [archetype_compliance[str(name)] for name in archetype_array],
                dtype="float32",
            ),
            "hidden_contest_base": np.asarray(
                [archetype_contest[str(name)] for name in archetype_array],
                dtype="float32",
            ),
        }
    )
    return frame
