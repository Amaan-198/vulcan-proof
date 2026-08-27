"""Arm 4: validation-tuned category/value/history cells without model inputs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from .base import bit_for, evidence_names, make_plan


Cell = tuple[str, int, int, str]


def _value_edges(params: Params) -> dict[str, tuple[float, ...]]:
    """Return log-spaced value boundaries for each configured category."""
    band_count = int(params["arms.arm4_value_bands"])
    fractions = np.linspace(0.0, 1.0, band_count + 1)
    result: dict[str, tuple[float, ...]] = {}
    for category_name in params["categories.order"]:
        category = params[f"categories.{category_name}"]
        low = math.log(float(category["vmin"]))
        high = math.log(float(category["vmax"]))
        result[category_name] = tuple(
            float(value) for value in np.exp(low + fractions * (high - low))
        )
    return result


def _contest_edges(observed: pd.DataFrame, params: Params) -> tuple[float, ...]:
    """Return contest-history tercile boundaries from validation observations."""
    bin_count = int(params["arms.arm4_contest_bins"])
    fractions = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    values = observed["merchant_contest_rate_hist"].to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise InvariantError("Arm 4 requires completed merchant history features")
    return tuple(float(value) for value in np.quantile(values, fractions))


def _cells(
    observed: pd.DataFrame,
    value_edges: Mapping[str, tuple[float, ...]],
    contest_edges: tuple[float, ...],
    params: Params,
) -> np.ndarray:
    """Encode each row as a deterministic Arm 4 policy cell."""
    categories = observed["category"].astype(str).to_numpy()
    values = observed["order_value"].to_numpy(dtype="float64")
    history = observed["merchant_contest_rate_hist"].to_numpy(dtype="float64")
    tiers = observed["eligible_tier"].astype(str).to_numpy()
    result = np.empty(len(observed), dtype=object)
    for category_name in params["categories.order"]:
        rows = categories == category_name
        edges = np.asarray(value_edges[category_name], dtype="float64")
        band = np.searchsorted(edges[1:-1], values[rows], side="right")
        contest_bin = np.searchsorted(
            np.asarray(contest_edges, dtype="float64"), history[rows], side="right"
        )
        positions = np.flatnonzero(rows)
        for position, value_band, history_bin in zip(
            positions, band, contest_bin, strict=True
        ):
            result[position] = (
                category_name,
                int(value_band),
                int(history_bin),
                str(tiers[position]),
            )
    return result


def _allowed_names(tier: str, params: Params) -> tuple[str, ...]:
    """Return evidence names allowed by a cell's merchant tier."""
    if tier == "NONE":
        return ()
    if tier == "POST_DELIVERY_ONLY":
        return tuple(name for name in evidence_names(params) if name in {"ack", "vack"})
    return evidence_names(params)


def _candidate_masks(tier: str, params: Params) -> tuple[int, ...]:
    """Enumerate every subset for a tier, including the empty subset."""
    import itertools

    names = _allowed_names(tier, params)
    masks: list[int] = []
    for size in range(len(names) + 1):
        for chosen in itertools.combinations(names, size):
            mask = 0
            for name in chosen:
                mask |= int(bit_for(name, params))
            masks.append(mask)
    return tuple(masks)


@dataclass(frozen=True)
class Arm4Policy:
    """A serialisable validation-tuned Arm 4 cell policy."""

    value_edges: dict[str, tuple[float, ...]]
    contest_edges: tuple[float, ...]
    cell_masks: dict[Cell, int]
    params: Params = P

    def plan(self, observed: pd.DataFrame) -> pd.DataFrame:
        """Plan on an observed frame using the stored validation boundaries."""
        from ..schemas import check

        check(observed, "ORDER_OBSERVED")
        cells = _cells(
            observed,
            self.value_edges,
            self.contest_edges,
            self.params,
        )
        requested = np.zeros(len(observed), dtype="uint16")
        for position, cell in enumerate(cells):
            requested[position] = np.uint16(self.cell_masks.get(cell, 0))
        return make_plan(observed, requested, "arm4", self.params)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of the policy."""
        cells = []
        for cell, mask in sorted(self.cell_masks.items()):
            cells.append(
                {
                    "category": cell[0],
                    "value_band": cell[1],
                    "contest_bin": cell[2],
                    "eligible_tier": cell[3],
                    "requested_bitmask": mask,
                    "evidence": [
                        name
                        for name in evidence_names(self.params)
                        if mask & int(bit_for(name, self.params))
                    ],
                }
            )
        return {
            "value_edges": self.value_edges,
            "contest_edges": self.contest_edges,
            "cells": cells,
        }


def tune(
    observed: pd.DataFrame,
    score_plan: Callable[[pd.DataFrame], float],
    params: Params = P,
) -> Arm4Policy:
    """Tune cell subsets from validation rows using a caller-owned score callback.

    The callback receives only a PLAN.  This keeps truth-bearing frames out of
    the arm package while allowing the simulator runner to score each subset.
    """
    from ..schemas import check

    check(observed, "ORDER_OBSERVED")
    validation = observed.loc[
        observed["split"].astype(str).eq("validate")
    ].reset_index(drop=True)
    if validation.empty:
        raise InvariantError("Arm 4 cannot tune without validation rows")
    value_edges = _value_edges(params)
    contest_edges = _contest_edges(validation, params)
    cells = _cells(validation, value_edges, contest_edges, params)
    unique_cells = sorted(set(cells.tolist()))
    cell_masks: dict[Cell, int] = {}
    for cell in unique_cells:
        rows = np.flatnonzero(np.asarray([item == cell for item in cells], dtype=bool))
        subset = validation.iloc[rows].reset_index(drop=True)
        best_mask = 0
        best_score = -math.inf
        best_size = len(evidence_names(params)) + 1
        candidate_masks = _candidate_masks(cell[3], params)
        candidate_plans = [
            pd.DataFrame(
                {
                    "order_id": subset["order_id"].astype("string").reset_index(drop=True),
                    "requested_bitmask": np.full(len(subset), mask, dtype="uint16"),
                }
            )
            for mask in candidate_masks
        ]
        batch_score = getattr(score_plan, "score_batch", None)
        if batch_score is not None:
            scores = tuple(float(value) for value in batch_score(candidate_plans))
        else:
            scores = tuple(float(score_plan(candidate)) for candidate in candidate_plans)
        if len(scores) != len(candidate_masks):
            raise InvariantError("Arm 4 score callback returned the wrong number of scores")
        for mask, candidate, score in zip(candidate_masks, candidate_plans, scores, strict=True):
            size = int((mask & int((1 << len(evidence_names(params))) - 1)).bit_count())
            if score > best_score or (score == best_score and size < best_size):
                best_score = score
                best_mask = mask
                best_size = size
        cell_masks[cell] = best_mask
    return Arm4Policy(value_edges, contest_edges, cell_masks, params)


def plan(observed: pd.DataFrame, policy: Arm4Policy) -> pd.DataFrame:
    """Apply a previously tuned Arm 4 policy."""
    if not isinstance(policy, Arm4Policy):
        raise InvariantError("Arm 4 requires an Arm4Policy from tune")
    return policy.plan(observed)


arm4 = plan
tune_arm4 = tune
