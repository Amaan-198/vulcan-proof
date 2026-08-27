"""Outcome-derived merchant history features for the completed observed frame."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ..schemas import check


HISTORY_COLUMNS = (
    "merchant_dispute_rate_hist",
    "merchant_contest_rate_hist",
    "merchant_compliance_hist",
)


def _population_rates(outcome_arm0: pd.DataFrame, params: Params) -> tuple[float, float, float]:
    """Compute shrinkage priors from non-censored training outcomes."""
    training = outcome_arm0.loc[
        outcome_arm0["split"].astype(str).eq("train")
        & outcome_arm0["censored"].eq(0)
    ]
    if training.empty:
        training = outcome_arm0.loc[outcome_arm0["censored"].eq(0)]
    if training.empty:
        raise InvariantError("history requires at least one non-censored outcome")
    opened = training["dispute_opened"].to_numpy(dtype="float64")
    contested = training["contested"].to_numpy(dtype="float64")
    complied = training["complied"].to_numpy(dtype="float64")
    requested = (
        training["requested_bitmask"].to_numpy(dtype="uint16") != 0
    ).astype("float64")
    dispute_rate = float(opened.mean())
    opened_count = float(opened.sum())
    requested_count = float(requested.sum())
    contest_rate = float(contested.sum() / opened_count) if opened_count else 0.0
    compliance_rate = float(complied.sum() / requested_count) if requested_count else 0.0
    del params
    return dispute_rate, contest_rate, compliance_rate


def add_history_features(
    observed_orders: pd.DataFrame,
    outcome_arm0: pd.DataFrame,
    params: Params = P,
) -> pd.DataFrame:
    """Fill merchant history columns using only prior arm 0 outcome rows.

    A prior row is eligible only when it is at least the configured lookback
    behind the current order and was not censored.  The deliberate Phase 2
    simplification is that history follows the resolved historical policy for
    every arm; deployed-arm outcomes are not used to update it.
    """
    check(observed_orders, "ORDER_OBSERVED")
    check(outcome_arm0, "OUTCOME")
    if not observed_orders["order_id"].is_unique or not outcome_arm0["order_id"].is_unique:
        raise InvariantError("history inputs require unique order ids")
    if set(observed_orders["order_id"].astype(str)) != set(outcome_arm0["order_id"].astype(str)):
        raise InvariantError("history inputs contain different order ids")

    rates = _population_rates(outcome_arm0, params)
    prior = float(params["features.hist_shrinkage_n"])
    lookback = int(params["features.merchant_hist_lookback_months"]) * int(
        params["sim.timeline.days_per_month"]
    )

    order_part = observed_orders[["order_id", "merchant_id", "order_day"]].copy()
    order_part["order_id"] = order_part["order_id"].astype("string")
    outcome_part = outcome_arm0[
        [
            "order_id",
            "requested_bitmask",
            "complied",
            "dispute_opened",
            "contested",
            "censored",
        ]
    ].copy()
    outcome_part["order_id"] = outcome_part["order_id"].astype("string")
    work = order_part.merge(outcome_part, on="order_id", how="left", validate="one_to_one")
    if work[["requested_bitmask", "complied", "dispute_opened", "contested", "censored"]].isna().any().any():
        raise InvariantError("history join introduced missing outcome rows")
    valid = work["censored"].to_numpy(dtype="int8") == 0
    work["valid_count"] = valid.astype("float64")
    work["dispute_count"] = (
        valid & work["dispute_opened"].to_numpy(dtype="int8").astype(bool)
    ).astype("float64")
    work["contest_count"] = (
        valid & work["contested"].to_numpy(dtype="int8").astype(bool)
    ).astype("float64")
    work["requested_count"] = (
        valid
        & (work["requested_bitmask"].to_numpy(dtype="uint16") != 0)
    ).astype("float64")
    work["compliance_count"] = (
        work["requested_count"].to_numpy(dtype="float64")
        * work["complied"].to_numpy(dtype="int8")
    )
    work = work.sort_values(
        ["merchant_id", "order_day", "order_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    group = work.groupby("merchant_id", sort=False, observed=True)
    for column in ("valid_count", "dispute_count", "contest_count", "requested_count", "compliance_count"):
        work[f"cum_{column}"] = group[column].cumsum()

    right = work[
        [
            "merchant_id",
            "order_day",
            "cum_valid_count",
            "cum_dispute_count",
            "cum_contest_count",
            "cum_requested_count",
            "cum_compliance_count",
        ]
    ].sort_values(["order_day", "merchant_id"], kind="mergesort")
    left = work[["order_id", "merchant_id", "order_day"]].copy()
    left["cutoff_day"] = left["order_day"].astype("int32") - lookback
    left = left.sort_values(["cutoff_day", "merchant_id"], kind="mergesort")
    joined = pd.merge_asof(
        left,
        right,
        left_on="cutoff_day",
        right_on="order_day",
        by="merchant_id",
        direction="backward",
        allow_exact_matches=True,
    )
    for column in (
        "cum_valid_count",
        "cum_dispute_count",
        "cum_contest_count",
        "cum_requested_count",
        "cum_compliance_count",
    ):
        joined[column] = joined[column].fillna(0.0)
    n = joined["cum_valid_count"].to_numpy(dtype="float64")
    disputes = joined["cum_dispute_count"].to_numpy(dtype="float64")
    contests = joined["cum_contest_count"].to_numpy(dtype="float64")
    requested = joined["cum_requested_count"].to_numpy(dtype="float64")
    compliance = joined["cum_compliance_count"].to_numpy(dtype="float64")
    joined[HISTORY_COLUMNS[0]] = (disputes + prior * rates[0]) / (n + prior)
    joined[HISTORY_COLUMNS[1]] = (contests + prior * rates[1]) / (
        joined["cum_dispute_count"].to_numpy(dtype="float64") + prior
    )
    joined[HISTORY_COLUMNS[2]] = (compliance + prior * rates[2]) / (requested + prior)

    values = joined.set_index("order_id")[list(HISTORY_COLUMNS)]
    result = observed_orders.copy()
    result = result.set_index("order_id", drop=False)
    for column in HISTORY_COLUMNS:
        result[column] = values[column].reindex(result.index).to_numpy(dtype="float32")
    result = result.reset_index(drop=True)
    check(result, "ORDER_OBSERVED")
    if result[list(HISTORY_COLUMNS)].isna().any().any():
        raise InvariantError("history features contain nulls after construction")
    return result


build_history_features = add_history_features
fill_history = add_history_features

