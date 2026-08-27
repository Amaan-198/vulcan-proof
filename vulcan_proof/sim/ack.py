"""Arm-invariant acknowledgement response draws."""

from __future__ import annotations

import numpy as np

from ..params import P, Params
from ..seeds import SeedTree


def response_uniforms(
    count: int,
    params: Params = P,
    seed: int | None = None,
) -> np.ndarray:
    """Draw one response uniform per order; the resolver maps it by plan."""
    del params
    if seed is None:
        raise ValueError("seed is required for response generation")
    return SeedTree(int(seed)).child("ack", "response").random(count).astype("float32")
