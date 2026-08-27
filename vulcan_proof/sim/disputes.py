"""Potential-dispute funnel and dispute-type generation."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ..seeds import SeedTree
from ._helpers import category_series


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def potential_probabilities(
    orders: pd.DataFrame,
    truth: pd.Series,
    risk_mult: np.ndarray,
    requested_bitmask: np.ndarray,
    theta: float,
    gamma: Mapping[str, float],
    params: Params = P,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute false/genuine potential probabilities and cap count."""
    categories = list(params["categories.order"])
    category_values = orders["category"].astype(str).to_numpy()
    order_value = orders["order_value"].to_numpy(dtype="float64")
    value_term = np.power(
        order_value / float(params["sim.false_claim_value_ref"]),
        float(params["sim.false_claim_value_elasticity"]),
    )
    gamma_values = np.empty(len(orders), dtype="float64")
    for category in categories:
        if category not in gamma:
            raise KeyError(category)
        gamma_values[category_values == category] = float(gamma[category])
    correct = truth.astype(str).to_numpy() == "delivered_correct"
    genuine_probability = gamma_values * risk_mult.astype("float64") * value_term
    false_probability = genuine_probability * float(theta)
    handoff_bits = np.zeros(len(orders), dtype="uint16")
    for evidence_name in ("geotag", "otp", "signature"):
        evidence_index = list(params["evidence.order"]).index(evidence_name)
        handoff_bits |= np.uint16(1 << evidence_index)
    deterrence = float(params["sim.deterrence"])
    if deterrence != 0.0:
        has_handoff = (requested_bitmask & handoff_bits) != 0
        false_probability = false_probability * (
            1.0 - deterrence * has_handoff.astype("float64")
        )
    probability = np.where(correct, false_probability, genuine_probability)
    capped = probability >= 0.5
    probability = np.minimum(probability, 0.5)
    if not np.isfinite(probability).all() or (probability < 0.0).any():
        raise InvariantError("potential-dispute probabilities are invalid")
    return probability, false_probability, int(capped.sum())


def draw_disputes(
    orders: pd.DataFrame,
    truth: pd.Series,
    risk_mult: np.ndarray,
    z_type: np.ndarray,
    requested_bitmask: np.ndarray,
    theta: float,
    gamma: Mapping[str, float],
    kappa: float,
    params: Params = P,
    seed: int | None = None,
) -> tuple[np.ndarray, pd.Series, np.ndarray, int]:
    """Draw potential disputes and their types from the calibrated funnel."""
    if seed is None:
        seed = int(params["run.master_seed"])
    probability, _, cap_count = potential_probabilities(
        orders,
        truth,
        risk_mult,
        requested_bitmask,
        theta,
        gamma,
        params,
    )
    tree = SeedTree(int(seed))
    potential = tree.child("disputes", "dispute_potential").random(len(orders)) < probability
    type_uniform = tree.child("disputes", "type").random(len(orders))
    categories = list(params["categories.order"])
    category_values = orders["category"].astype(str).to_numpy()
    truth_values = truth.astype(str).to_numpy()
    type_values = np.empty(len(orders), dtype=object)
    type_values[:] = None
    for category in categories:
        category_rows = category_values == category
        mix = params[f"categories.{category}"]["mix"]
        nad_eb = float(mix["NAD"]) + float(mix["EB"])
        genuine_nr = truth_values == "misdelivered"
        genuine_nr |= truth_values == "never_handed_off"
        genuine_rows = category_rows & ~genuine_nr & (truth_values != "delivered_correct")
        correct_rows = category_rows & (truth_values == "delivered_correct")
        nr_probability = _sigmoid(
            _logit(float(mix["NR"]))
            + float(params["sim.kappa.type_shift"])
            * float(kappa)
            * z_type.astype("float64")
        )
        type_values[category_rows & genuine_nr] = "NR"
        type_values[genuine_rows & (type_uniform < float(mix["NAD"]) / nad_eb)] = "NAD"
        type_values[genuine_rows & (type_uniform >= float(mix["NAD"]) / nad_eb)] = "EB"
        type_values[correct_rows & (type_uniform < nr_probability)] = "NR"
        type_values[
            correct_rows
            & (type_uniform >= nr_probability)
            & (
                type_uniform
                < nr_probability
                + (1.0 - nr_probability) * float(mix["NAD"]) / nad_eb
            )
        ] = "NAD"
        type_values[
            correct_rows
            & (type_uniform >= nr_probability)
            & (
                type_uniform
                >= nr_probability
                + (1.0 - nr_probability) * float(mix["NAD"]) / nad_eb
            )
        ] = "EB"
    type_values[~potential] = None
    return (
        potential.astype("int8"),
        category_series(type_values, ["NR", "NAD", "EB"]),
        probability,
        cap_count,
    )
