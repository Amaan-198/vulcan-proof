"""Truth-conditional outcome resolution with common random numbers.

All stochastic decisions are addressed by ``(seed, purpose, order_id)`` or
the Phase-1 response uniform.  There is deliberately no arm label in a draw
key: identical plans therefore produce identical outcomes across arms, which
keeps paired differences low variance.

Reports without dispute potential have no modelled cost.  They are ordinary
returns-channel activity shared by every arm and are counted as false friction
by the reporting layer.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .. import economics
from ..errors import InvariantError
from ..params import P, Params
from ..schemas import OUTCOME, PLAN, check, cast
from .prevention import draw_prevention_modes, prevention_cost_for_mode


def _keyed_uniforms(
    order_ids: pd.Series,
    seed: int,
    purpose: str,
) -> np.ndarray:
    """Return stable, order-addressed uniforms for one resolver purpose."""
    maximum = float(np.iinfo(np.uint64).max)
    values = np.empty(len(order_ids), dtype="float64")
    for position, order_id in enumerate(order_ids.astype(str)):
        label = f"{int(seed)}|resolve|{purpose}|{order_id}"
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        raw = int.from_bytes(digest[:np.dtype(np.uint64).itemsize], "little")
        values[position] = (float(raw) + 0.5) / (maximum + 1.0)
    return values


def _logit(value: np.ndarray) -> np.ndarray:
    """Map valid probabilities to log odds."""
    if not np.isfinite(value).all() or (value <= 0.0).any() or (value >= 1.0).any():
        raise InvariantError("contest probabilities must lie strictly inside [0, 1]")
    return np.log(value / (1.0 - value))


def _dispute_types(params: Params) -> tuple[str, ...]:
    """Return dispute types from the configured category mix catalogue."""
    first_category = params["categories.order"][0]
    return tuple(params[f"categories.{first_category}.mix"])


def _truth_modifier(truth: np.ndarray, params: Params) -> np.ndarray:
    """Return the hidden truth modifier applied to evidence uplifts."""
    result = np.ones(len(truth), dtype="float64")
    result[truth == "merchant_fault"] = float(params["uplift_true.on_merchant_fault"])
    result[truth == "transit_damage"] = float(params["uplift_true.on_transit_damage"])
    result[truth == "never_handed_off"] = float(params["uplift_true.on_never_handed_off"])
    return result


def _set_uplift(
    dispute_type: str,
    materialised: int,
    truth: str,
    params: Params,
) -> float:
    """Compute the overlapping uplift for one contested row."""
    uplift_values: list[float] = []
    modifier = 1.0
    if truth == "merchant_fault":
        modifier = float(params["uplift_true.on_merchant_fault"])
    elif truth == "transit_damage":
        modifier = float(params["uplift_true.on_transit_damage"])
    elif truth == "never_handed_off":
        modifier = float(params["uplift_true.on_never_handed_off"])
    for evidence_name, uplift in params[f"uplift_true.{dispute_type}"].items():
        evidence_index = list(params["evidence.order"]).index(evidence_name)
        bit = 1 << evidence_index
        if materialised & bit:
            item = float(uplift) * modifier
            if truth == "misdelivered" and evidence_name in {"otp", "signature"}:
                item *= float(params["uplift_true.on_misdelivered_otp"])
            uplift_values.append(item)
    if not uplift_values:
        return 0.0
    ordered = sorted(uplift_values, reverse=True)
    overlap = float(params[f"overlap.{dispute_type}"])
    total = ordered[0] + (1.0 - overlap) * sum(ordered[1:])
    return total * float(params["uplift_true.sweep_multiplier"])


def _category_cogs(observed: pd.DataFrame, params: Params) -> np.ndarray:
    """Map observed categories to their configured COGS fractions."""
    categories = observed["category"].astype(str).to_numpy()
    result = np.empty(len(observed), dtype="float64")
    for category_name in params["categories.order"]:
        result[categories == category_name] = float(
            params[f"categories.{category_name}"]["cogs"]
        ) * float(params["categories.cogs_sweep_multiplier"])
    if not np.isfinite(result).all():
        raise InvariantError("category COGS mapping is incomplete")
    return result


def _response_values(
    observed: pd.DataFrame,
    hidden: pd.DataFrame,
    requested: np.ndarray,
    ack_sent: np.ndarray,
    vack_sent: np.ndarray,
    params: Params,
) -> np.ndarray:
    """Map the stored response uniform through the chosen acknowledgement."""
    truth = hidden["hidden_truth"].astype(str).to_numpy()
    uniform = hidden["hidden_u_response"].to_numpy(dtype="float64")
    response = np.empty(len(observed), dtype=object)
    response[:] = "none"
    response_rows = ack_sent.astype(bool)
    for truth_name, response_key in (
        ("delivered_correct", "delivered_correct"),
        ("merchant_fault", "merchant_fault_or_damage"),
        ("transit_damage", "merchant_fault_or_damage"),
        ("misdelivered", "not_received"),
        ("never_handed_off", "not_received"),
    ):
        rows = response_rows & (truth == truth_name)
        if not bool(rows.any()):
            continue
        probabilities = np.asarray(
            params[f"sim.customer_response.{response_key}"], dtype="float64"
        )
        confirm = probabilities[0]
        report = probabilities[1]
        if bool(vack_sent[rows].any()):
            vack_rows = rows & vack_sent.astype(bool)
            report_probability = report * (
                1.0 - float(params["sim.verified_ack_response_penalty"])
            )
            confirm_rows = vack_rows & (uniform < confirm)
            report_rows = vack_rows & (uniform >= confirm) & (
                uniform < confirm + report_probability
            )
            response[confirm_rows] = "confirm"
            response[report_rows] = "report"
            response[vack_rows & ~(confirm_rows | report_rows)] = "silent"
        ack_rows = rows & ~vack_sent.astype(bool)
        confirm_rows = ack_rows & (uniform < confirm)
        report_rows = ack_rows & (uniform >= confirm) & (uniform < confirm + report)
        response[confirm_rows] = "confirm"
        response[report_rows] = "report"
        response[ack_rows & ~(confirm_rows | report_rows)] = "silent"
    return response


def _value_for_rows(
    order_value: np.ndarray,
    prevented: np.ndarray,
    prevention_values: np.ndarray,
    opened: np.ndarray,
    contested: np.ndarray,
    won: np.ndarray,
    params: Params,
) -> np.ndarray:
    """Apply the central money table to each row."""
    result = np.zeros(len(order_value), dtype="float64")
    for position in range(len(order_value)):
        if prevented[position]:
            result[position] = economics.money(
                "prevented",
                float(order_value[position]),
                float(prevention_values[position]),
                params=params,
            )
        elif opened[position] and contested[position] and won[position]:
            result[position] = economics.money(
                "opened_contested_won", float(order_value[position]), params=params
            )
        elif opened[position] and contested[position]:
            result[position] = economics.money(
                "opened_contested_lost", float(order_value[position]), params=params
            )
        elif opened[position]:
            result[position] = economics.money(
                "opened_not_contested", float(order_value[position]), params=params
            )
    return result


def score_masks(
    plans: Sequence[pd.DataFrame],
    hidden_orders: pd.DataFrame,
    observed_orders: pd.DataFrame,
    seed: int | None = None,
    params: Params = P,
) -> np.ndarray:
    """Score several same-row plans while sharing all arm-invariant draws.

    This is the batch path used while tuning Arm 4.  It is algebraically the
    same resolver as :func:`resolve`, but avoids rebuilding a DataFrame and
    re-drawing common random numbers for every subset.
    """
    if not plans:
        raise InvariantError("score_masks requires at least one plan")
    check(hidden_orders, "ORDER_HIDDEN")
    check(observed_orders, "ORDER_OBSERVED")
    check(plans[0], "PLAN")
    if seed is None:
        seed = int(params["run.master_seed"])
    order_ids = observed_orders["order_id"]
    hidden = hidden_orders.copy()
    hidden["order_id"] = hidden["order_id"].astype("string")
    hidden = hidden.set_index("order_id").reindex(order_ids.astype("string")).reset_index()
    if hidden["order_id"].isna().any():
        raise InvariantError("hidden order alignment introduced missing values")
    evidence = tuple(params["evidence.order"])
    known_mask = np.uint16((1 << len(evidence)) - 1)
    requested_values: list[np.ndarray] = []
    first_ids = plans[0]["order_id"].astype(str)
    for candidate in plans:
        candidate_ids = candidate["order_id"].astype(str)
        if not candidate_ids.equals(first_ids) or set(candidate_ids) != set(order_ids.astype(str)):
            raise InvariantError("batch plans and observed order ids do not match")
        values = candidate["requested_bitmask"].to_numpy(dtype="uint16")
        if bool((values & np.uint16(~known_mask)).any()):
            raise InvariantError("plan contains an unknown evidence bit")
        requested_values.append(values)

    tiers = observed_orders["eligible_tier"].astype(str).to_numpy()
    opt_in = observed_orders["ack_optin"].to_numpy(dtype="int8").astype(bool)
    available = np.zeros(len(observed_orders), dtype="uint16")
    for evidence_name in evidence:
        bit = np.uint16(1 << evidence.index(evidence_name))
        allowed = tiers != "NONE"
        if evidence_name not in {"ack", "vack"}:
            allowed &= tiers != "POST_DELIVERY_ONLY"
        else:
            allowed &= opt_in
        available[allowed] |= bit
    compliance_u = _keyed_uniforms(order_ids, int(seed), "compliance")
    compliance = compliance_u < hidden["hidden_compliance"].to_numpy(dtype="float64")
    truth = hidden["hidden_truth"].astype(str).to_numpy()
    presence_u = {
        name: _keyed_uniforms(order_ids, int(seed), f"presence_{name}")
        for name in ("otp", "signature")
    }
    contest_u = _keyed_uniforms(order_ids, int(seed), "contest")
    win_u = _keyed_uniforms(order_ids, int(seed), "win")
    contest_base = hidden["hidden_contest_base"].to_numpy(dtype="float64")
    contest_logit = _logit(contest_base)
    potential = hidden["hidden_dispute_potential"].to_numpy(dtype="int8").astype(bool)
    order_value = observed_orders["order_value"].to_numpy(dtype="float64")
    cogs = _category_cogs(observed_orders, params)
    all_modes = draw_prevention_modes(
        np.ones(len(observed_orders), dtype=bool),
        len(observed_orders),
        params=params,
        seed=int(seed),
    )
    dispute_types = _dispute_types(params)
    mask_values = np.vstack(requested_values).astype("uint16")
    requested = np.bitwise_and(mask_values, available[None, :])
    row_count = len(observed_orders)
    mask_count = len(plans)
    materialised = np.zeros((mask_count, row_count), dtype="uint16")
    cash_cost = np.zeros((mask_count, row_count), dtype="float64")
    time_cost = np.zeros((mask_count, row_count), dtype="float64")
    for evidence_name in evidence:
        bit = np.uint16(1 << evidence.index(evidence_name))
        requested_rows = (requested & bit) != 0
        if bool(params[f"evidence.{evidence_name}"]["system_sent"]):
            continue
        captured = requested_rows & compliance[None, :]
        if evidence_name in presence_u:
            captured &= presence_u[evidence_name][None, :] < float(
                params[f"evidence.{evidence_name}"]["presence_factor"]
            )
        if evidence_name in {"geotag", "otp", "signature"}:
            captured &= truth[None, :] != "never_handed_off"
        materialised |= np.where(captured, bit, 0).astype("uint16")
        cash_cost += captured.astype("float64") * float(
            params[f"evidence.{evidence_name}"]["cash"]
        )
        time_cost += requested_rows.astype("float64") * float(
            params[f"evidence.{evidence_name}"]["seconds"]
        ) * float(params["econ.hourly_rate"]) / 3600.0

    ack_bit = np.uint16(1 << evidence.index("ack"))
    vack_bit = np.uint16(1 << evidence.index("vack"))
    ack_requested = (requested & ack_bit) != 0
    vack_requested = (requested & vack_bit) != 0
    ack_sent = ack_requested | vack_requested
    truth_response = np.empty(row_count, dtype="object")
    truth_response[:] = "delivered_correct"
    truth_response[truth == "merchant_fault"] = "merchant_fault_or_damage"
    truth_response[truth == "transit_damage"] = "merchant_fault_or_damage"
    truth_response[truth == "misdelivered"] = "not_received"
    truth_response[truth == "never_handed_off"] = "not_received"
    confirm_probability = np.empty(row_count, dtype="float64")
    report_probability = np.empty(row_count, dtype="float64")
    for response_key in (
        "delivered_correct",
        "merchant_fault_or_damage",
        "not_received",
    ):
        rows = truth_response == response_key
        probabilities = np.asarray(
            params[f"sim.customer_response.{response_key}"], dtype="float64"
        )
        confirm_probability[rows] = probabilities[0]
        report_probability[rows] = probabilities[1]
    response_uniform = hidden["hidden_u_response"].to_numpy(dtype="float64")
    vack_report = report_probability * (
        1.0 - float(params["sim.verified_ack_response_penalty"])
    )
    report_cutoff = np.where(vack_requested, confirm_probability + vack_report, confirm_probability + report_probability)
    confirms = ack_sent & (response_uniform[None, :] < confirm_probability[None, :])
    reports = ack_sent & (response_uniform[None, :] >= confirm_probability[None, :]) & (
        response_uniform[None, :] < report_cutoff
    )
    silent_code = len(("none", "confirm", "report"))
    response_code = np.zeros((mask_count, row_count), dtype="int8")
    response_code[confirms] = 1
    response_code[reports] = 1 + 1
    response_code[ack_sent & ~(confirms | reports)] = silent_code
    active_ack = np.where(vack_requested, vack_bit, ack_bit).astype("uint16")
    ack_materialised = ack_sent & (response_code != silent_code)
    materialised |= np.where(ack_materialised, active_ack, 0).astype("uint16")
    for evidence_name in ("ack", "vack"):
        bit = np.uint16(1 << evidence.index(evidence_name))
        sent = (active_ack == bit) & ack_sent
        cash_cost += sent.astype("float64") * float(
            params[f"evidence.{evidence_name}"]["cash"]
        )
        time_cost += ((requested & bit) != 0).astype("float64") * float(
            params[f"evidence.{evidence_name}"]["seconds"]
        ) * float(params["econ.hourly_rate"]) / 3600.0

    preventable = potential[None, :] & ack_sent & reports
    prevented = preventable.copy()
    mode_cost = np.asarray(
        [
            prevention_cost_for_mode(
                str(all_modes[position]),
                float(order_value[position]),
                float(cogs[position]),
                params,
            )
            for position in range(row_count)
        ],
        dtype="float64",
    )
    prevention_values = prevented.astype("float64") * mode_cost[None, :]
    opened = potential[None, :] & ~prevented
    dispute_type = hidden["hidden_dispute_type"].astype("object").to_numpy()
    materialised_count = np.zeros((mask_count, row_count), dtype="int16")
    for dispute_name in dispute_types:
        rows = opened & (dispute_type[None, :] == dispute_name)
        for evidence_name in evidence:
            if dispute_name in params[f"evidence.{evidence_name}"]["admissible"]:
                bit = np.uint16(1 << evidence.index(evidence_name))
                materialised_count += (rows & ((materialised & bit) != 0)).astype("int16")
    contest_probability = np.zeros((mask_count, row_count), dtype="float64")
    contest_probability[opened] = 1.0 / (
        1.0
        + np.exp(
            -(
                np.broadcast_to(contest_logit, (mask_count, row_count))[opened]
                + float(params["archetypes.contest_evidence_slope"])
                * materialised_count[opened].astype("float64")
            )
        )
    )
    contested = opened & (contest_u[None, :] < contest_probability)
    win_probability = np.zeros((mask_count, row_count), dtype="float64")
    for dispute_name in dispute_types:
        rows = contested & (dispute_type[None, :] == dispute_name)
        base = float(params[f"base_win.{dispute_name}"])
        for coordinates in np.argwhere(rows):
            mask_position = int(coordinates[0])
            row_position = int(coordinates[1])
            uplift = _set_uplift(
                dispute_name,
                int(materialised[mask_position, row_position]),
                str(truth[row_position]),
                params,
            )
            if response_code[mask_position, row_position] == silent_code:
                uplift += float(params["sim.silence_yield"])
            win_probability[mask_position, row_position] = float(
                np.clip(base + uplift, 0.0, 1.0)
            )
    won = contested & (win_u[None, :] < win_probability)
    value = np.zeros((mask_count, row_count), dtype="float64")
    value[prevented] = economics.money_array(
        "prevented", np.broadcast_to(order_value, (mask_count, row_count))[prevented], prevention_values[prevented], params
    )
    not_contested = opened & ~contested
    value[not_contested] = economics.money_array(
        "opened_not_contested", np.broadcast_to(order_value, (mask_count, row_count))[not_contested], params=params
    )
    contested_won = contested & won
    value[contested_won] = economics.money_array(
        "opened_contested_won", np.broadcast_to(order_value, (mask_count, row_count))[contested_won], params=params
    )
    contested_lost = contested & ~won
    value[contested_lost] = economics.money_array(
        "opened_contested_lost", np.broadcast_to(order_value, (mask_count, row_count))[contested_lost], params=params
    )
    return (value - cash_cost - time_cost).astype("float32").sum(axis=1).astype("float64")


def resolve(
    plan: pd.DataFrame,
    hidden_orders: pd.DataFrame,
    observed_orders: pd.DataFrame,
    seed: int | None = None,
    params: Params = P,
    arm_id: str = "arm",
) -> pd.DataFrame:
    """Resolve a PLAN against the matching observed and hidden order frames."""
    check(plan, "PLAN")
    check(hidden_orders, "ORDER_HIDDEN")
    check(observed_orders, "ORDER_OBSERVED")
    if seed is None:
        seed = int(params["run.master_seed"])
    if not plan["order_id"].is_unique or not hidden_orders["order_id"].is_unique:
        raise InvariantError("resolver inputs require unique order ids")
    if not observed_orders["order_id"].is_unique:
        raise InvariantError("observed orders require unique order ids")
    observed_ids = observed_orders["order_id"].astype(str)
    if set(plan["order_id"].astype(str)) != set(observed_ids):
        raise InvariantError("plan and observed order ids do not match")
    if set(hidden_orders["order_id"].astype(str)) != set(observed_ids):
        raise InvariantError("hidden and observed order ids do not match")

    plan_by_id = plan.copy()
    plan_by_id["order_id"] = plan_by_id["order_id"].astype("string")
    requested = (
        plan_by_id.set_index("order_id")["requested_bitmask"]
        .reindex(observed_orders["order_id"].astype("string"))
        .to_numpy(dtype="uint16")
    )
    hidden = hidden_orders.copy()
    hidden["order_id"] = hidden["order_id"].astype("string")
    hidden = hidden.set_index("order_id").reindex(observed_orders["order_id"].astype("string")).reset_index()
    if hidden["order_id"].isna().any():
        raise InvariantError("hidden order alignment introduced missing values")

    evidence = tuple(params["evidence.order"])
    known_mask = np.uint16((1 << len(evidence)) - 1)
    if bool((requested & np.uint16(~known_mask)).any()):
        raise InvariantError("plan contains an unknown evidence bit")
    tiers = observed_orders["eligible_tier"].astype(str).to_numpy()
    opt_in = observed_orders["ack_optin"].to_numpy(dtype="int8").astype(bool)
    available = np.zeros(len(observed_orders), dtype="uint16")
    for evidence_name in evidence:
        index = evidence.index(evidence_name)
        bit = np.uint16(1 << index)
        allowed = tiers != "NONE"
        if evidence_name not in {"ack", "vack"}:
            allowed &= tiers != "POST_DELIVERY_ONLY"
        else:
            allowed &= opt_in
        available[allowed] |= bit
    requested = np.bitwise_and(requested, available)

    order_ids = observed_orders["order_id"]
    compliance_u = _keyed_uniforms(order_ids, int(seed), "compliance")
    compliance_probability = hidden["hidden_compliance"].to_numpy(dtype="float64")
    complied = compliance_u < compliance_probability
    truth = hidden["hidden_truth"].astype(str).to_numpy()
    materialised = np.zeros(len(observed_orders), dtype="uint16")
    cash_cost = np.zeros(len(observed_orders), dtype="float64")
    time_cost = np.zeros(len(observed_orders), dtype="float64")
    for evidence_name in evidence:
        index = evidence.index(evidence_name)
        bit = np.uint16(1 << index)
        requested_rows = (requested & bit) != 0
        system_sent = bool(params[f"evidence.{evidence_name}"]["system_sent"])
        if system_sent:
            continue
        captured = requested_rows & complied
        presence = float(params[f"evidence.{evidence_name}"]["presence_factor"])
        if presence < 1.0:
            captured &= _keyed_uniforms(order_ids, int(seed), f"presence_{evidence_name}") < presence
        if evidence_name in {"geotag", "otp", "signature"}:
            captured &= truth != "never_handed_off"
        materialised[captured] |= bit
        cash_cost += captured.astype("float64") * float(
            params[f"evidence.{evidence_name}"]["cash"]
        )
        time_cost += requested_rows.astype("float64") * float(
            params[f"evidence.{evidence_name}"]["seconds"]
        ) * float(params["econ.hourly_rate"]) / 3600.0

    ack_bit = np.uint16(1 << evidence.index("ack"))
    vack_bit = np.uint16(1 << evidence.index("vack"))
    ack_requested = (requested & ack_bit) != 0
    vack_requested = (requested & vack_bit) != 0
    vack_sent = vack_requested
    ack_sent = ack_requested | vack_sent
    response = _response_values(
        observed_orders,
        hidden,
        requested,
        ack_sent.astype("int8"),
        vack_sent.astype("int8"),
        params,
    )
    active_ack = np.where(vack_sent, vack_bit, ack_bit).astype("uint16")
    ack_materialised = ack_sent & (response != "silent")
    materialised[ack_materialised] |= active_ack[ack_materialised]
    for evidence_name in ("ack", "vack"):
        bit = np.uint16(1 << evidence.index(evidence_name))
        sent = (active_ack == bit) & ack_sent
        cash_cost += sent.astype("float64") * float(
            params[f"evidence.{evidence_name}"]["cash"]
        )
        time_cost += ((requested & bit) != 0).astype("float64") * float(
            params[f"evidence.{evidence_name}"]["seconds"]
        ) * float(params["econ.hourly_rate"]) / 3600.0

    potential = hidden["hidden_dispute_potential"].to_numpy(dtype="int8").astype(bool)
    preventable = potential & ack_sent & (response == "report")
    modes = draw_prevention_modes(
        preventable,
        len(observed_orders),
        params=params,
        seed=int(seed),
    )
    cogs = _category_cogs(observed_orders, params)
    prevention_values = np.zeros(len(observed_orders), dtype="float64")
    prevented = preventable.copy()
    for position in np.flatnonzero(prevented):
        mode = str(modes[position])
        prevention_values[position] = prevention_cost_for_mode(
            mode,
            float(observed_orders["order_value"].iloc[position]),
            float(cogs[position]),
            params,
        )

    opened = potential & ~prevented
    dispute_type = hidden["hidden_dispute_type"].astype("object").to_numpy()
    materialised_count = np.zeros(len(observed_orders), dtype="int16")
    for dispute_name in _dispute_types(params):
        rows = opened & (dispute_type == dispute_name)
        for evidence_name in evidence:
            if dispute_name in params[f"evidence.{evidence_name}"]["admissible"]:
                bit = np.uint16(1 << evidence.index(evidence_name))
                materialised_count += (rows & ((materialised & bit) != 0)).astype("int16")
    contest_probability = np.zeros(len(observed_orders), dtype="float64")
    opened_rows = np.flatnonzero(opened)
    if len(opened_rows):
        contest_base = hidden["hidden_contest_base"].to_numpy(dtype="float64")
        contest_probability[opened] = 1.0 / (
            1.0
            + np.exp(
                -(
                    _logit(contest_base[opened])
                    + float(params["archetypes.contest_evidence_slope"])
                    * materialised_count[opened].astype("float64")
                )
            )
        )
    contest_u = _keyed_uniforms(order_ids, int(seed), "contest")
    contested = opened & (contest_u < contest_probability)
    win_probability = np.zeros(len(observed_orders), dtype="float64")
    base_win = np.zeros(len(observed_orders), dtype="float64")
    for dispute_name in _dispute_types(params):
        rows = contested & (dispute_type == dispute_name)
        base = float(params[f"base_win.{dispute_name}"])
        base_win[rows] = base
        for position in np.flatnonzero(rows):
            uplift = _set_uplift(
                dispute_name,
                int(materialised[position]),
                str(truth[position]),
                params,
            )
            if response[position] == "silent":
                uplift += float(params["sim.silence_yield"])
            win_probability[position] = float(np.clip(base + uplift, 0.0, 1.0))
    win_u = _keyed_uniforms(order_ids, int(seed), "win")
    won = contested & (win_u < win_probability)
    value = _value_for_rows(
        observed_orders["order_value"].to_numpy(dtype="float64"),
        prevented,
        prevention_values,
        opened,
        contested,
        won,
        params,
    )
    net = value - cash_cost - time_cost
    wrong_recipient = (
        (truth == "misdelivered")
        & (((materialised & np.uint16(1 << evidence.index("otp"))) != 0)
           | ((materialised & np.uint16(1 << evidence.index("signature"))) != 0))
    )
    claim_class = np.full(len(observed_orders), "none", dtype=object)
    claim_class[potential & (truth == "delivered_correct")] = "correct_fulfillment"
    claim_class[potential & (truth == "merchant_fault")] = "merchant_fault"
    claim_class[potential & np.isin(truth, ["misdelivered", "never_handed_off", "transit_damage"])] = "carrier_fault"
    outcome = pd.DataFrame(
        {
            "order_id": observed_orders["order_id"].astype("string").reset_index(drop=True),
            "arm_id": pd.Series([str(arm_id)] * len(observed_orders), dtype="string"),
            "requested_bitmask": requested.astype("uint16"),
            "complied": complied.astype("int8"),
            "materialised_bitmask": materialised.astype("uint16"),
            "wrong_recipient": wrong_recipient.astype("int8"),
            "cash_cost": cash_cost.astype("float32"),
            "time_cost": time_cost.astype("float32"),
            "ack_sent": ack_sent.astype("int8"),
            "response": pd.Series(pd.Categorical(response, categories=["none", "confirm", "report", "silent"])),
            "prevented": prevented.astype("int8"),
            "prevention_mode": pd.Series(pd.Categorical(modes, categories=["explanation", "refund", "replacement"])),
            "dispute_opened": opened.astype("int8"),
            "dispute_type": pd.Series(pd.Categorical(dispute_type, categories=list(_dispute_types(params)))),
            "contested": contested.astype("int8"),
            "won": won.astype("int8"),
            "value": value.astype("float32"),
            "net": net.astype("float32"),
            "censored": observed_orders["censored"].to_numpy(dtype="int8"),
            "split": observed_orders["split"].astype("category").reset_index(drop=True),
            "claim_class": pd.Series(pd.Categorical(
                claim_class,
                categories=["correct_fulfillment", "merchant_fault", "carrier_fault", "none"],
            )),
        }
    )
    return cast(outcome, "OUTCOME")


def resolve_outcomes(
    observed_orders: pd.DataFrame,
    hidden_orders: pd.DataFrame,
    plan: pd.DataFrame,
    seed: int | None = None,
    params: Params = P,
    arm_id: str = "arm",
) -> pd.DataFrame:
    """Compatibility wrapper with observed-first argument order."""
    return resolve(plan, hidden_orders, observed_orders, seed, params, arm_id)


resolve_plan = resolve
