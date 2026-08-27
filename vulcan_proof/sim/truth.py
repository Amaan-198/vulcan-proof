"""Fulfillment truth generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ..seeds import SeedTree
from ._helpers import category_series


TRUTH_STATES = [
    "delivered_correct",
    "misdelivered",
    "never_handed_off",
    "merchant_fault",
    "transit_damage",
]


def truth_probabilities(
    orders: pd.DataFrame,
    merchants: pd.DataFrame,
    params: Params = P,
) -> dict[str, np.ndarray]:
    """Return row-level probabilities for each hidden fulfillment state."""
    categories = list(params["categories.order"])
    merchant_index = orders["merchant_index"].to_numpy(dtype="int32")
    order_categories = orders["category"].astype(str).to_numpy()
    quality = merchants.loc[merchant_index, "hidden_quality"].to_numpy(dtype="float64")
    carrier = merchants.loc[
        merchant_index, "carrier_reliability"
    ].to_numpy(dtype="float64")
    misdelivered = np.full(
        len(orders), float(params["sim.truth_base_rate.misdelivered"]), dtype="float64"
    )
    never_handed_off = np.full(
        len(orders), float(params["sim.truth_base_rate.never_handed_off"]), dtype="float64"
    )
    merchant_fault = np.full(
        len(orders), float(params["sim.truth_base_rate.merchant_fault"]), dtype="float64"
    )
    damage_multiplier = np.empty(len(orders), dtype="float64")
    for category in categories:
        rows = order_categories == category
        category_params = params[f"categories.{category}"]
        damage_multiplier[rows] = np.exp(
            float(params["sim.fragility_effect"])
            * (
                float(category_params["fragility"])
                - float(params["sim.fragility_ref"])
            )
        )
    transit_damage = (
        float(params["sim.truth_base_rate.transit_damage"])
        * damage_multiplier
        / carrier
    )
    misdelivered = misdelivered / carrier
    never_handed_off = never_handed_off / carrier
    merchant_fault = merchant_fault * quality
    total_failure = misdelivered + never_handed_off + merchant_fault + transit_damage
    delivered_correct = 1.0 - total_failure
    probabilities = {
        "delivered_correct": delivered_correct,
        "misdelivered": misdelivered,
        "never_handed_off": never_handed_off,
        "merchant_fault": merchant_fault,
        "transit_damage": transit_damage,
    }
    matrix = np.column_stack([probabilities[name] for name in TRUTH_STATES])
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise InvariantError("truth probabilities are not finite and non-negative")
    if (delivered_correct <= 0.0).any():
        raise InvariantError("truth probabilities leave no correct-fulfillment mass")
    return probabilities


def draw_truth(
    probabilities: dict[str, np.ndarray],
    params: Params = P,
    seed: int | None = None,
) -> pd.Series:
    """Draw one fulfillment state per order from row-level probabilities."""
    del params
    if seed is None:
        raise ValueError("seed is required for truth generation")
    count = len(probabilities[TRUTH_STATES[0]])
    rng = SeedTree(int(seed)).child("truth")
    uniform = rng.random(count)
    cumulative = np.zeros(count, dtype="float64")
    values = np.empty(count, dtype=object)
    values[:] = None
    for state in TRUTH_STATES[1:]:
        cumulative += probabilities[state]
        rows = (values == None) & (uniform < cumulative)  # noqa: E711
        values[rows] = state
    values[values == None] = TRUTH_STATES[0]  # noqa: E711
    return category_series(values, TRUTH_STATES)
