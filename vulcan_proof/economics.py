"""Pure money and prevention calculations shared by simulator components."""

from __future__ import annotations

import math

import numpy as np

from .errors import InvariantError
from .params import P, Params


def prevention_cost(
    mode: str,
    order_value: float,
    cogs: float,
    params: Params = P,
) -> float:
    """Return the cost of resolving a prevented dispute in ``mode``."""
    modes = tuple(
        name.removeprefix("share_")
        for name in params["econ.prevention"]
        if name.startswith("share_")
    )
    if mode not in modes:
        raise InvariantError(f"unknown prevention mode: {mode!r}")
    value = float(order_value)
    cost_of_goods = float(cogs)
    if not math.isfinite(value) or not math.isfinite(cost_of_goods):
        raise InvariantError("prevention inputs must be finite")
    if mode == "explanation":
        return float(params["econ.prevention.support_cost"])
    if mode == "refund":
        return value - float(params["econ.prevention.salvage_sigma"]) * cost_of_goods * value
    return (
        cost_of_goods * value
        + float(params["econ.prevention.reship_cost"])
        + float(params["econ.prevention.support_cost"])
    )


def _money_state(
    state: str,
    order_value: float,
    prevention_value: float | None,
    prevention_mode: str | None,
    cogs: float | None,
    params: Params,
) -> float:
    """Apply the resolver's mutually exclusive money table branches."""
    value = float(order_value)
    if not math.isfinite(value):
        raise InvariantError("order value must be finite")
    if state == "none":
        return 0.0
    if state == "prevented":
        if prevention_value is None:
            if prevention_mode is None or cogs is None:
                raise InvariantError("prevented money requires a prevention cost")
            prevention_value = prevention_cost(prevention_mode, value, cogs, params)
        return -float(prevention_value)
    fee = float(params["econ.dispute_fee"])
    if state == "opened_not_contested":
        return -value - fee
    if state == "opened_contested_won":
        return -fee
    if state == "opened_contested_lost":
        return -value - fee - float(params["econ.ratio_damage"])
    raise InvariantError(f"unknown money state: {state!r}")


def money(
    state_or_order_value: str | float | None = None,
    order_value: float | None = None,
    prevention_value: float | None = None,
    *,
    state: str | None = None,
    potential: bool = False,
    prevented: bool = False,
    dispute_opened: bool = False,
    contested: bool = False,
    won: bool = False,
    prevention_mode: str | None = None,
    cogs: float | None = None,
    params: Params = P,
) -> float:
    """Return value from the six-branch outcome money table.

    The compact form is ``money("opened_contested_lost", value)``.  The
    keyword form accepts resolver flags and is useful when building a frame.
    ``potential`` is accepted explicitly so a potential row that was not
    opened can be represented as ``none`` without inventing a loss.
    """
    if isinstance(state_or_order_value, str):
        if state is not None and state != state_or_order_value:
            raise InvariantError("money state was supplied twice")
        state = state_or_order_value
        if order_value is None:
            raise InvariantError("money state form requires order_value")
        value = float(order_value)
    else:
        if state_or_order_value is not None:
            if order_value is not None:
                raise InvariantError("money order value was supplied twice")
            value = float(state_or_order_value)
        elif order_value is not None:
            value = float(order_value)
        else:
            raise InvariantError("money requires order_value")

    if state is None:
        if prevented:
            state = "prevented"
        elif dispute_opened and contested and won:
            state = "opened_contested_won"
        elif dispute_opened and contested:
            state = "opened_contested_lost"
        elif dispute_opened:
            state = "opened_not_contested"
        elif not potential:
            state = "none"
        else:
            state = "none"
    return _money_state(
        state,
        value,
        prevention_value,
        prevention_mode,
        cogs,
        params,
    )


def money_array(
    state: str,
    order_value: np.ndarray,
    prevention_value: np.ndarray | None = None,
    params: Params = P,
) -> np.ndarray:
    """Apply one money-table branch to a numeric array."""
    values = np.asarray(order_value, dtype="float64")
    if not np.isfinite(values).all():
        raise InvariantError("order values must be finite")
    if state == "none":
        return np.zeros(len(values), dtype="float64")
    if state == "prevented":
        if prevention_value is None:
            raise InvariantError("prevented money requires prevention values")
        costs = np.asarray(prevention_value, dtype="float64")
        if len(costs) != len(values) or not np.isfinite(costs).all():
            raise InvariantError("prevention values are not aligned and finite")
        return -costs
    fee = float(params["econ.dispute_fee"])
    if state == "opened_not_contested":
        return -values - fee
    if state == "opened_contested_won":
        return np.full(len(values), -fee, dtype="float64")
    if state == "opened_contested_lost":
        return -values - fee - float(params["econ.ratio_damage"])
    raise InvariantError(f"unknown money state: {state!r}")


def prevention_gain(
    order_value: float,
    cogs: float,
    mode: str,
    contest_probability: float,
    win_probability: float,
    params: Params = P,
) -> float:
    """Return prevention value minus the expected value of an open dispute."""
    contest = float(contest_probability)
    win = float(win_probability)
    if not (math.isfinite(contest) and math.isfinite(win)):
        raise InvariantError("dispute probabilities must be finite")
    if not 0.0 <= contest <= 1.0 or not 0.0 <= win <= 1.0:
        raise InvariantError("dispute probabilities must be within [0, 1]")
    value = float(order_value)
    dispute_value = (
        (1.0 - contest) * _money_state("opened_not_contested", value, None, None, None, params)
        + contest
        * (
            win * _money_state("opened_contested_won", value, None, None, None, params)
            + (1.0 - win)
            * _money_state("opened_contested_lost", value, None, None, None, params)
        )
    )
    return -prevention_cost(mode, value, cogs, params) - dispute_value
