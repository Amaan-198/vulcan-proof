"""Dispute opening and resolution timing."""

from __future__ import annotations

import numpy as np

from ..params import P, Params
from ..seeds import SeedTree


def draw_latency(
    order_day: np.ndarray,
    potential: np.ndarray,
    params: Params = P,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw nullable-as-sentinel dispute open and resolution days."""
    if seed is None:
        seed = int(params["run.master_seed"])
    count = len(order_day)
    rng = SeedTree(int(seed)).child("latency")
    expected_delivery = float(params["sim.latency.expected_delivery_days"])
    fast_max = float(params["sim.latency.dispute_fast_max_days"])
    dispute_max = float(params["sim.latency.dispute_max_days"])
    fast = rng.random(count) < float(params["sim.latency.dispute_fast_share"])
    fast_offset = rng.uniform(0.0, fast_max, size=count)
    slow_offset = rng.uniform(fast_max, dispute_max, size=count)
    offset = np.floor(expected_delivery + np.where(fast, fast_offset, slow_offset))
    open_day = np.full(count, -1, dtype="int32")
    open_day[potential.astype(bool)] = (
        order_day[potential.astype(bool)] + offset[potential.astype(bool)]
    ).astype("int32")

    median = float(params["sim.latency.resolution_lognormal_median_days"])
    p95 = float(params["sim.latency.resolution_lognormal_p95_days"])
    spread = np.log(p95 / median)
    resolution_extra = np.ceil(rng.lognormal(np.log(median), spread, size=count))
    resolution_day = np.full(count, -1, dtype="int32")
    resolution_day[potential.astype(bool)] = (
        open_day[potential.astype(bool)]
        + int(params["sim.latency.response_days"])
        + resolution_extra[potential.astype(bool)].astype("int32")
    ).astype("int32")
    return open_day, resolution_day
