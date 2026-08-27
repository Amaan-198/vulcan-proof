"""Deterministic six-target calibration for the simulated dispute funnel."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

from ..errors import InvariantError
from ..params import P, Params, load
from .world import LatentWorld, build_latent_world


def _category_factors(
    world: LatentWorld,
    theta: float,
    params: Params,
) -> dict[str, np.ndarray]:
    """Return uncapped potential-probability factors by category."""
    categories = list(params["categories.order"])
    category_values = world.observed["category"].astype(str).to_numpy()
    order_value = world.observed["order_value"].to_numpy(dtype="float64")
    value_term = np.power(
        order_value / float(params["sim.false_claim_value_ref"]),
        float(params["sim.false_claim_value_elasticity"]),
    )
    correct = world.truth.astype(str).to_numpy() == "delivered_correct"
    genuine_factor = world.risk_mult.astype("float64") * value_term
    false_factor = genuine_factor * float(theta)
    handoff_bits = np.zeros(len(world.observed), dtype="uint16")
    for evidence_name in ("geotag", "otp", "signature"):
        index = list(params["evidence.order"]).index(evidence_name)
        handoff_bits |= np.uint16(1 << index)
    deterrence = float(params["sim.deterrence"])
    if deterrence != 0.0:
        handoff = (world.requested_bitmask & handoff_bits) != 0
        false_factor = false_factor * (1.0 - deterrence * handoff)
    factor = np.where(correct, false_factor, genuine_factor)
    return {category: factor[category_values == category] for category in categories}


def _solve_gamma(
    factor: np.ndarray,
    target: float,
    params: Params,
) -> tuple[float, float]:
    """Biselect one category's gamma and return gamma plus achieved rate."""
    search = params["sim.gamma_search"]
    lower = float(search[0])
    upper = float(search[1])

    def rate(value: float) -> float:
        return float(np.minimum(0.5, value * factor).mean())

    lower_rate = rate(lower)
    upper_rate = rate(upper)
    if lower_rate > target or upper_rate < target:
        raise InvariantError("gamma search interval does not bracket its target")
    value = lower
    for _ in range(100):
        value = (lower + upper) / 2.0
        achieved = rate(value)
        relative_error = abs(achieved - target) / target
        if relative_error <= float(params["sim.calibration_rel_tol"]):
            return value, achieved
        if achieved < target:
            lower = value
        else:
            upper = value
    return value, rate(value)


def _solve_for_theta(
    world: LatentWorld,
    params: Params,
) -> tuple[float, dict[str, float], dict[str, float], float, float]:
    """Solve theta and gamma, returning all calibration diagnostics."""
    categories = list(params["categories.order"])
    target_multiplier = float(params["categories.rate_sweep_multiplier"])
    targets = {
        category: float(params[f"categories.{category}"]["target_rate"])
        * target_multiplier
        for category in categories
    }

    def at(theta: float) -> tuple[dict[str, float], dict[str, float], float, float]:
        factors = _category_factors(world, theta, params)
        gamma: dict[str, float] = {}
        achieved: dict[str, float] = {}
        for category in categories:
            gamma[category], achieved[category] = _solve_gamma(
                factors[category], targets[category], params
            )
        row_probability = np.empty(len(world.observed), dtype="float64")
        category_values = world.observed["category"].astype(str).to_numpy()
        for category in categories:
            rows = category_values == category
            row_probability[rows] = np.minimum(
                0.5, gamma[category] * factors[category]
            )
        correct = world.truth.astype(str).to_numpy() == "delivered_correct"
        genuine_share = float(
            row_probability[~correct].sum() / row_probability.sum()
        )
        population_rate = float(row_probability.mean())
        return gamma, achieved, genuine_share, population_rate

    search = params["sim.theta_search"]
    lower = float(search[0])
    upper = float(search[1])
    _, _, lower_share, _ = at(lower)
    _, _, upper_share, _ = at(upper)
    if not upper_share < lower_share:
        raise InvariantError("genuine share is not decreasing in theta")
    target_share = float(params["sim.genuine_share_target"])
    if not lower_share >= target_share >= upper_share:
        raise InvariantError("theta search interval does not bracket genuine share")
    theta = lower
    solved: tuple[dict[str, float], dict[str, float], float, float] | None = None
    for _ in range(100):
        theta = (lower + upper) / 2.0
        solved = at(theta)
        share = solved[2]
        relative_error = abs(share - target_share) / target_share
        if relative_error <= float(params["sim.calibration_rel_tol"]):
            break
        if share > target_share:
            lower = theta
        else:
            upper = theta
    if solved is None:
        raise InvariantError("theta calibration did not execute")
    gamma, achieved, genuine_share, population_rate = solved
    return theta, gamma, achieved, genuine_share, population_rate


def calibrate_funnel(
    seed: int,
    n_orders: int,
    params_path: str | pathlib.Path | None = None,
    output_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Calibrate and persist theta/gamma using expected probabilities only."""
    params = P if params_path is None else load(params_path)
    if output_path is None:
        root = params.path.resolve().parents[1]
        output_path = root / "outputs" / "theta.json"
    output = pathlib.Path(output_path).resolve()
    canonical_kappa = float(params["sim.kappa.canonical"])
    world = build_latent_world(
        kappa=canonical_kappa,
        seed=int(seed),
        n_orders=int(n_orders),
        shift_enabled=False,
        params=params,
    )
    theta, gamma, achieved, genuine_share, population_rate = _solve_for_theta(
        world, params
    )
    payload: dict[str, Any] = {
        "theta": float(theta),
        "gamma": {name: float(gamma[name]) for name in params["categories.order"]},
        "seed": int(seed),
        "kappa": canonical_kappa,
        "n_orders": int(n_orders),
        "achieved_category_rates": {
            name: float(achieved[name]) for name in params["categories.order"]
        },
        "achieved_genuine_share": float(genuine_share),
        "implied_population_rate": float(population_rate),
        "implied_phi": float(1.0 - genuine_share),
    }
    phi_tolerance = float(params["models.calib.mean_tolerance"]) - float(
        params["sim.max_censor_frac"]
    )
    if abs(payload["implied_phi"] - float(params["reference.phi"])) >= phi_tolerance:
        raise InvariantError("calibrated phi disagrees with the reference world")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload
