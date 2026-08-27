"""Prevention-mode draws for acknowledgements that report a problem."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..errors import InvariantError
from ..params import P, Params
from ..seeds import SeedTree
from ..economics import prevention_cost


def draw_prevention_modes(
    preventable: np.ndarray,
    count: int,
    params: Params = P,
    seed: int | None = None,
) -> np.ndarray:
    """Draw a shared prevention mode for each row, or ``None`` when inactive."""
    if seed is None:
        raise ValueError("seed is required for prevention draws")
    if len(preventable) != count:
        raise InvariantError("prevention mask and count are not aligned")
    mode_names: Sequence[str] = tuple(
        name.removeprefix("share_")
        for name in params["econ.prevention"]
        if name.startswith("share_")
    )
    if not mode_names:
        raise InvariantError("prevention mode catalogue is empty")
    uniform = SeedTree(int(seed)).child("resolve", "prevention").random(count)
    result = np.empty(count, dtype=object)
    result[:] = None
    cumulative = np.zeros(count, dtype="float64")
    active = preventable.astype(bool)
    for mode in mode_names:
        cumulative += float(params[f"econ.prevention.share_{mode}"])
        rows = active & (result == None) & (uniform < cumulative)  # noqa: E711
        result[rows] = mode
    if bool((active & (result == None)).any()):  # noqa: E711
        raise InvariantError("prevention shares did not cover an active row")
    return result


def prevention_cost_for_mode(
    mode: str,
    order_value: float,
    cogs: float,
    params: Params = P,
) -> float:
    """Delegate prevention economics to the central pure implementation."""
    return prevention_cost(mode, order_value, cogs, params)

