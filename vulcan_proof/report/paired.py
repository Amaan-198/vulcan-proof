"""Paired per-seed reporting for simulator arms."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ..errors import InvariantError
from ..params import P, Params
from ..schemas import check


def _test_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return headline rows, with a small-data fallback for unit callers."""
    test = frame["split"].astype(str).eq("test").to_numpy()
    if not bool(test.any()):
        test = np.ones(len(frame), dtype=bool)
    return test


def _per_order_values(frame: pd.DataFrame, params: Params) -> tuple[float, float, int, int]:
    """Return uncensored headline and zero-valued-censoring rates."""
    test = _test_mask(frame)
    censored = frame["censored"].to_numpy(dtype="int8").astype(bool)
    headline = test & ~censored
    count = int(headline.sum())
    total = int(test.sum())
    if count == 0 or total == 0:
        raise InvariantError("paired report has no usable test rows")
    scale = float(params["report.per_orders"])
    headline_value = scale * float(frame.loc[headline, "net"].sum()) / count
    censored_zero_value = scale * float(frame.loc[test & ~censored, "net"].sum()) / total
    return headline_value, censored_zero_value, count, total


def _interval(values: np.ndarray, params: Params) -> tuple[float, float, float]:
    """Return the mean and its configured uncertainty bounds."""
    mean = float(values.mean())
    degrees = len(values) - 1
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    tail = (1.0 - float(params["report.ci_level"])) / 2.0
    critical = float(student_t.ppf(1.0 - tail, degrees))
    return mean, mean - critical * standard_error, mean + critical * standard_error


def paired_report(
    left: Mapping[int, pd.DataFrame],
    right: Mapping[int, pd.DataFrame],
    params: Params = P,
) -> dict[str, object]:
    """Compare two arms on identical seeds and order sets."""
    required_seeds = int(params["report.min_seeds"])
    if len(left) != len(right) or set(left) != set(right):
        raise InvariantError("paired arms do not have the same seed set")
    if len(left) < required_seeds:
        raise InvariantError("paired reporting requires the configured minimum seed count")
    differences: list[float] = []
    censored_differences: list[float] = []
    per_seed: list[dict[str, object]] = []
    for seed in sorted(left):
        left_frame = left[seed]
        right_frame = right[seed]
        check(left_frame, "OUTCOME")
        check(right_frame, "OUTCOME")
        left_ids = set(left_frame["order_id"].astype(str))
        right_ids = set(right_frame["order_id"].astype(str))
        if left_ids != right_ids:
            raise InvariantError(f"paired order ids differ for seed {seed}")
        right_by_id = right_frame.set_index("order_id").reindex(left_frame["order_id"])
        if right_by_id.index.isna().any():
            raise InvariantError("paired alignment introduced missing rows")
        left_censored = left_frame.set_index("order_id").loc[right_by_id.index, "censored"]
        if not left_censored.equals(right_by_id["censored"]):
            raise InvariantError(f"censoring differs between paired arms for seed {seed}")
        left_value, left_zero, eligible, total = _per_order_values(left_frame, params)
        right_value, right_zero, _, _ = _per_order_values(right_frame, params)
        difference = right_value - left_value
        zero_difference = right_zero - left_zero
        differences.append(difference)
        censored_differences.append(zero_difference)
        per_seed.append(
            {
                "seed": int(seed),
                "left_net_per_1000": left_value,
                "right_net_per_1000": right_value,
                "difference_net_per_1000": difference,
                "left_censored_zero_net_per_1000": left_zero,
                "right_censored_zero_net_per_1000": right_zero,
                "difference_censored_zero_net_per_1000": zero_difference,
                "n_eligible": eligible,
                "n_test": total,
            }
        )
        del right_by_id, left_censored
    values = np.asarray(differences, dtype="float64")
    zero_values = np.asarray(censored_differences, dtype="float64")
    mean, low, high = _interval(values, params)
    zero_mean, zero_low, zero_high = _interval(zero_values, params)
    return {
        "n_seeds": len(values),
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "p_positive": float((values > 0.0).mean()),
        "censored_zero_mean": zero_mean,
        "censored_zero_ci_low": zero_low,
        "censored_zero_ci_high": zero_high,
        "per_seed": per_seed,
    }


def arm_diagnostics(outcome: pd.DataFrame, params: Params = P) -> dict[str, object]:
    """Return coverage, friction, materialisation, and defence diagnostics."""
    check(outcome, "OUTCOME")
    mask = _test_mask(outcome)
    uncensored = ~outcome["censored"].to_numpy(dtype="int8").astype(bool)
    rows = mask & uncensored
    if not bool(rows.any()):
        raise InvariantError("arm diagnostics have no usable test rows")
    evidence = tuple(params["evidence.order"])
    requested = outcome["requested_bitmask"].to_numpy(dtype="uint16")
    materialised = outcome["materialised_bitmask"].to_numpy(dtype="uint16")
    coverage = {
        name: float((((requested & np.uint16(1 << evidence.index(name))) != 0) & rows).sum() / rows.sum())
        for name in evidence
    }
    materialisation = {}
    for name in evidence:
        bit = np.uint16(1 << evidence.index(name))
        requested_rows = rows & ((requested & bit) != 0)
        materialisation[name] = (
            float(((materialised & bit) != 0)[requested_rows].mean())
            if bool(requested_rows.any())
            else 0.0
        )
    ack = outcome["ack_sent"].to_numpy(dtype="int8").astype(bool)
    prevention = outcome["prevented"].to_numpy(dtype="int8").astype(bool)
    opened = outcome["dispute_opened"].to_numpy(dtype="int8").astype(bool)
    contested = outcome["contested"].to_numpy(dtype="int8").astype(bool)
    friction = {
        "otp_requested_pct": float(
            (rows & ((requested & np.uint16(1 << evidence.index("otp"))) != 0)).sum()
            / rows.sum()
        ),
        "ack_sent_pct": float((rows & ack).sum() / rows.sum()),
        "mean_ack_taps": float((rows & ack).sum() / rows.sum()),
    }
    class_rates: dict[str, float] = {}
    claim_classes = ("correct_fulfillment", "merchant_fault", "carrier_fault")
    for claim_class in claim_classes:
        class_rows = rows & contested & outcome["claim_class"].astype(str).eq(claim_class).to_numpy()
        class_rates[claim_class] = float(outcome.loc[class_rows, "won"].mean()) if bool(class_rows.any()) else 0.0
    mode_rates = {
        mode: float((rows & (outcome["prevention_mode"].astype(str).to_numpy() == mode)).sum() / rows.sum())
        for mode in ("explanation", "refund", "replacement")
    }
    return {
        "coverage": coverage,
        "coverage_any": float((rows & (requested != 0)).sum() / rows.sum()),
        "friction": friction,
        "materialisation_rate": materialisation,
        "prevention_rate": float((rows & prevention).sum() / rows.sum()),
        "defence_rate": float((rows & opened & ~prevention).sum() / rows.sum()),
        "prevention_mode_share": mode_rates,
        "defense_only_win_rate": class_rates,
        "contested_count": int((rows & contested).sum()),
    }


paired_difference = paired_report
