"""Truth-blind exhaustive subset optimizer."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..errors import InvariantError, SchemaError
from ..params import P, Params
from ..schemas import cast, check


def evidence_names(params: Params = P) -> tuple[str, ...]:
    """Return evidence names in their configured bit order."""
    return tuple(params["evidence.order"])


def bit_for(name: str, params: Params = P) -> np.uint16:
    """Return the configured bit for one evidence name."""
    names = evidence_names(params)
    if name not in names:
        raise InvariantError(f"unknown evidence type: {name!r}")
    return np.uint16(1 << names.index(name))


def available_mask(observed: pd.DataFrame, params: Params = P) -> np.ndarray:
    """Return the per-row availability mask without importing arm code."""
    tiers = observed["eligible_tier"].astype(str).to_numpy()
    opt_in = observed["ack_optin"].to_numpy(dtype="int8").astype(bool)
    result = np.zeros(len(observed), dtype="uint16")
    for name in evidence_names(params):
        allowed = tiers != "NONE"
        if name not in {"ack", "vack"}:
            allowed &= tiers != "POST_DELIVERY_ONLY"
        else:
            allowed &= opt_in
        result[allowed] |= bit_for(name, params)
    return result


@dataclass(frozen=True)
class PlanResult:
    """Selected subset plus explanation values for one order."""

    requested_bitmask: int
    ev: float
    standalone_ev: dict[str, float]
    incremental_ev: dict[str, float]
    reasons: dict[str, str]


def _frame(row: pd.Series | pd.DataFrame) -> pd.DataFrame:
    """Normalise a scalar optimizer input and apply the schema gate."""
    if isinstance(row, pd.DataFrame):
        result = row.reset_index(drop=True)
        expected = set(P["features.permitted"]) | {"split", "censored", "order_day"}
        if set(result.columns) != expected:
            check(result, "ORDER_OBSERVED")
        try:
            check(result, "ORDER_OBSERVED")
        except SchemaError:
            result = cast(result, "ORDER_OBSERVED")
        return result
    if not isinstance(row, pd.Series):
        raise SchemaError("optimizer expects an observed Series or DataFrame")
    values = row.to_frame().T
    result = cast(values, "ORDER_OBSERVED")
    check(result, "ORDER_OBSERVED")
    return result


def _single(row: pd.DataFrame) -> pd.Series:
    """Return the one observed row required by scalar model protocols."""
    if len(row) != 1:
        raise InvariantError("scalar optimizer expects exactly one observed row")
    return row.iloc[0]


def _value(result: Any) -> float:
    """Convert a model scalar or one-element array to a finite probability."""
    values = np.asarray(result, dtype="float64").reshape(-1)
    if len(values) != 1 or not np.isfinite(values[0]):
        raise InvariantError("model returned an invalid scalar")
    value = float(values[0])
    if value < 0.0 or value > 1.0:
        raise InvariantError("model probability is outside [0, 1]")
    return value


def _call_contest(models: Any, row: pd.Series, held: int) -> float:
    """Call either the documented two-argument or typed contest protocol."""
    try:
        return _value(models.pC(row, held))
    except TypeError as first:
        try:
            return _value(models.pC(row, "NR", held))
        except TypeError:
            raise first


def _call_win(models: Any, row: pd.Series, dispute_type: str, held: int) -> float:
    """Call the win model with the documented scalar protocol."""
    return _value(models.pW(row, dispute_type, held))


def _type_rates(models: Any, row: pd.Series) -> dict[str, float]:
    """Read and validate the conditional type distribution."""
    rates = models.pB(row)
    if isinstance(rates, Mapping):
        result = {str(name): float(value) for name, value in rates.items()}
    else:
        values = np.asarray(rates, dtype="float64").reshape(-1)
        names = tuple(("NR", "NAD", "EB"))
        if len(values) != len(names):
            raise InvariantError("type model returned an invalid vector")
        result = {name: float(values[position]) for position, name in enumerate(names)}
    total = 0.0
    for value in result.values():
        if not np.isfinite(value) or value < 0.0:
            raise InvariantError("type model returned invalid probabilities")
        total += value
    if total <= 0.0:
        raise InvariantError("type model returned no probability mass")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        result = {name: value / total for name, value in result.items()}
    return result


def _support_allowed(masks: Any, dispute_type: str, evidence: str) -> bool:
    """Resolve support information from the model bundle or a plain mapping."""
    if masks is None:
        return True
    method = getattr(masks, "pair_allowed", None)
    if method is not None:
        return bool(method(dispute_type, evidence))
    if isinstance(masks, Mapping):
        key = (dispute_type, evidence)
        if key in masks:
            return bool(masks[key])
        nested = masks.get(dispute_type)
        if isinstance(nested, Mapping) and evidence in nested:
            return bool(nested[evidence])
    return True


def _support_source(models: Any, masks: Any) -> Any:
    """Use explicit masks first, then the bundle's support field."""
    if masks is not None:
        return masks
    return getattr(models, "support_mask", None)


def _ack_kind(chosen: Sequence[str]) -> str | None:
    """Return the one acknowledgement kind that controls prevention."""
    if "vack" in chosen:
        return "vack"
    if "ack" in chosen:
        return "ack"
    return None


def _cost(row: pd.Series, chosen: Sequence[str], models: Any, params: Params) -> float:
    """Compute request-time and expected cash costs."""
    result = 0.0
    for name in chosen:
        item = params[f"evidence.{name}"]
        probability = _value(models.pM(row, name))
        if not bool(item["system_sent"]):
            result += float(item["cash"]) * probability
        else:
            result += float(item["cash"])
        result += float(item["seconds"]) * float(params["econ.hourly_rate"]) / 3600.0
    return result


def _subset_ev(
    row: pd.Series,
    chosen: Sequence[str],
    models: Any,
    masks: Any,
    params: Params,
) -> float:
    """Evaluate one evidence subset by enumerating materialisation patterns."""
    if not chosen:
        return 0.0
    p_a = _value(models.pA(row))
    types = _type_rates(models, row)
    probabilities = [_value(models.pM(row, name)) for name in chosen]
    names = evidence_names(params)
    base_values = {
        dispute_type: _call_win(models, row, dispute_type, 0)
        for dispute_type, rate in types.items()
        if rate > 0.0
    }
    value = float(row["order_value"])
    defence = 0.0
    prevention_gain = 0.0
    for bits in itertools.product((0, 1), repeat=len(chosen)):
        probability = 1.0
        held = 0
        for name, bit, material_probability in zip(chosen, bits, probabilities, strict=True):
            if bit:
                probability *= material_probability
                held |= int(bit_for(name, params))
            else:
                probability *= 1.0 - material_probability
        if probability <= 0.0:
            continue
        contest = _call_contest(models, row, held)
        for dispute_type, type_rate in types.items():
            if type_rate <= 0.0:
                continue
            win = _call_win(models, row, dispute_type, held)
            uplift = contest * (win - base_values[dispute_type]) * value
            defence += probability * type_rate * uplift
            gain_method = getattr(models, "prevention_gain", None)
            if gain_method is not None:
                prevention_gain += probability * type_rate * float(gain_method(row, contest, win))
    ack = _ack_kind(chosen)
    if ack is not None and bool(params["opt.include_prevention"]):
        prevention_probability = _value(models.pPrev(row, ack)) if hasattr(models, "pPrev") else 0.0
        defence += prevention_probability * prevention_gain
    return float(p_a * defence - _cost(row, chosen, models, params))


def admissible_names(
    row: pd.Series | pd.DataFrame,
    models: Any,
    masks: Any = None,
    params: Params = P,
) -> tuple[str, ...]:
    """Return available, type-admissible, support-qualified evidence names."""
    observed = _frame(row)
    scalar = _single(observed)
    type_rates = _type_rates(models, scalar)
    availability = int(available_mask(observed, params)[0])
    support = _support_source(models, masks)
    result: list[str] = []
    for name in evidence_names(params):
        if (availability & int(bit_for(name, params))) == 0:
            continue
        admissible = False
        for dispute_type, rate in type_rates.items():
            if rate <= 0.0:
                continue
            item = params[f"evidence.{name}"]
            if dispute_type in item["admissible"] and _support_allowed(support, dispute_type, name):
                admissible = True
                break
        if admissible:
            result.append(name)
    return tuple(result)


def evaluate(
    row: pd.Series | pd.DataFrame,
    models: Any,
    masks: Any = None,
    params: Params = P,
) -> PlanResult:
    """Evaluate all admissible subsets and return an explained best plan."""
    observed = _frame(row)
    scalar = _single(observed)
    support = _support_source(models, masks)
    all_names = evidence_names(params)
    allowed = admissible_names(observed, models, support, params)
    type_rates = _type_rates(models, scalar)
    availability = int(available_mask(observed, params)[0])
    values: dict[int, float] = {0: 0.0}
    chosen_names = tuple(allowed)
    for size in range(len(chosen_names) + 1):
        for subset in itertools.combinations(chosen_names, size):
            bitmask = 0
            for name in subset:
                bitmask |= int(bit_for(name, params))
            values[bitmask] = _subset_ev(scalar, subset, models, support, params)
    best_mask = 0
    best_value = 0.0
    for bitmask, current in values.items():
        if current > best_value + 1e-9:
            best_mask = bitmask
            best_value = current
    standalone: dict[str, float] = {}
    incremental: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for name in all_names:
        bit = int(bit_for(name, params))
        item = params[f"evidence.{name}"]
        if name not in chosen_names:
            if (availability & bit) == 0:
                reasons[name] = "UNAVAILABLE"
            else:
                has_type = any(
                    rate > 0.0 and str(dispute_type) in item["admissible"]
                    for dispute_type, rate in type_rates.items()
                )
                reasons[name] = "INADMISSIBLE" if not has_type else "NO_SUPPORT"
            continue
        single_mask = bit
        standalone[name] = values[single_mask]
        with_name = best_mask | bit
        without_name = best_mask & ~bit
        if best_mask & bit:
            incremental[name] = values[best_mask] - values.get(without_name, 0.0)
        else:
            incremental[name] = values.get(with_name, values[best_mask]) - values[best_mask]
        if standalone[name] <= 0.0:
            reasons[name] = "NEGATIVE_STANDALONE"
        elif not (best_mask & bit) and incremental[name] < 0.0:
            reasons[name] = "NEGATIVE_INCREMENTAL"
    return PlanResult(best_mask, best_value, standalone, incremental, reasons)


def best_plan(
    row: pd.Series | pd.DataFrame,
    models: Any,
    masks: Any = None,
    params: Params = P,
) -> PlanResult:
    """Return the best explained plan for one observed order."""
    return evaluate(row, models, masks, params)


def plan(
    row: pd.Series | pd.DataFrame,
    models: Any,
    masks: Any = None,
    params: Params = P,
) -> int:
    """Return only the selected evidence bitmask."""
    return best_plan(row, models, masks, params).requested_bitmask


def ev_set(
    row: pd.Series | pd.DataFrame,
    requested_bitmask: int,
    models: Any,
    params: Params = P,
) -> float:
    """Return EV for one requested bitmask, including the empty-plan zero."""
    observed = _frame(row)
    scalar = _single(observed)
    chosen = tuple(
        name for name in evidence_names(params)
        if int(requested_bitmask) & int(bit_for(name, params))
    )
    return _subset_ev(scalar, chosen, models, _support_source(models, None), params)


class Optimizer:
    """Reusable wrapper around the exhaustive subset evaluator."""

    def __init__(self, models: Any, masks: Any = None, params: Params = P) -> None:
        self.models = models
        self.masks = masks
        self.params = params

    def plan(self, row: pd.Series | pd.DataFrame) -> int:
        """Return the best requested bitmask."""
        return plan(row, self.models, self.masks, self.params)

    def evaluate(self, row: pd.Series | pd.DataFrame) -> PlanResult:
        """Return the best plan and explanation values."""
        return best_plan(row, self.models, self.masks, self.params)

    def ev(self, row: pd.Series | pd.DataFrame, requested_bitmask: int) -> float:
        """Return EV for one requested bitmask."""
        return ev_set(row, requested_bitmask, self.models, self.params)


def _batch_vector(value: Any, count: int, name: str) -> np.ndarray:
    """Normalise a batch model result to one finite vector."""
    result = np.asarray(value, dtype="float64").reshape(-1)
    if len(result) != count or not np.isfinite(result).all():
        raise InvariantError(f"{name} returned an invalid batch vector")
    if bool((result < 0.0).any()) or bool((result > 1.0).any()):
        raise InvariantError(f"{name} returned probabilities outside [0, 1]")
    return result


def _batch_amount(value: Any, count: int, name: str) -> np.ndarray:
    """Normalise a finite non-probability batch vector."""
    result = np.asarray(value, dtype="float64").reshape(-1)
    if len(result) != count or not np.isfinite(result).all():
        raise InvariantError(f"{name} returned an invalid batch vector")
    return result


def _batch_types(value: Any, count: int) -> np.ndarray:
    """Normalise a batch type model result to rows in NR/NAD/EB order."""
    if isinstance(value, Mapping):
        result = np.column_stack([_batch_vector(value[name], count, "pB") for name in ("NR", "NAD", "EB")])
    else:
        result = np.asarray(value, dtype="float64")
        if result.ndim != 2 or result.shape != (count, len(("NR", "NAD", "EB"))):
            raise InvariantError("pB returned an invalid batch matrix")
    if not np.isfinite(result).all() or bool((result < 0.0).any()):
        raise InvariantError("pB returned invalid batch probabilities")
    totals = result.sum(axis=1)
    if bool((totals <= 0.0).any()):
        raise InvariantError("pB returned a row without probability mass")
    return result / totals[:, None]


def _batch_contest(models: Any, frame: pd.DataFrame, held: int) -> np.ndarray:
    """Evaluate contest probability for one held pattern across a chunk."""
    try:
        return _batch_vector(models.pC(frame, held), len(frame), "pC")
    except TypeError as first:
        try:
            return _batch_vector(models.pC(frame, "NR", held), len(frame), "pC")
        except TypeError:
            raise first


def _batch_win(models: Any, frame: pd.DataFrame, dispute_type: str, held: int) -> np.ndarray:
    """Evaluate win probability for one type and held pattern."""
    return _batch_vector(models.pW(frame, dispute_type, held), len(frame), "pW")


def _batch_subset_ev(
    frame: pd.DataFrame,
    positions: np.ndarray,
    chosen: Sequence[str],
    models: Any,
    params: Params,
    p_a: np.ndarray,
    p_b: np.ndarray,
    material: Mapping[str, np.ndarray],
    contest_cache: dict[int, np.ndarray],
    win_cache: dict[tuple[str, int], np.ndarray],
    gain_cache: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    """Evaluate one subset for all rows in one chunk."""
    count = len(frame)
    names = evidence_names(params)
    values = frame["order_value"].to_numpy(dtype="float64")
    result = np.zeros(count, dtype="float64")
    expected_gain = np.zeros(count, dtype="float64")
    type_names = ("NR", "NAD", "EB")
    base_cache: dict[str, np.ndarray] = {}
    for dispute_type in type_names:
        key = (dispute_type, 0)
        if key not in win_cache:
            win_cache[key] = _batch_win(models, frame, dispute_type, 0)
        base_cache[dispute_type] = win_cache[key]
    for bits in itertools.product((0, 1), repeat=len(chosen)):
        probability = np.ones(count, dtype="float64")
        held = 0
        for name, bit in zip(chosen, bits, strict=True):
            if bit:
                probability *= material[name]
                held |= int(bit_for(name, params))
            else:
                probability *= 1.0 - material[name]
        if not bool((probability > 0.0).any()):
            continue
        if held not in contest_cache:
            contest_cache[held] = _batch_contest(models, frame, held)
        contest = contest_cache[held]
        for type_position, dispute_type in enumerate(type_names):
            type_rate = p_b[:, type_position]
            if not bool((type_rate > 0.0).any()):
                continue
            key = (dispute_type, held)
            if key not in win_cache:
                win_cache[key] = _batch_win(models, frame, dispute_type, held)
            win = win_cache[key]
            result += probability * type_rate * contest * (win - base_cache[dispute_type]) * values
            gain_method = getattr(models, "prevention_gain", None)
            if gain_method is not None:
                if key not in gain_cache:
                    gain_cache[key] = _batch_amount(
                        gain_method(frame, contest, win), count, "prevention_gain"
                    )
                expected_gain += probability * type_rate * gain_cache[key]
    ack = _ack_kind(chosen)
    if ack is not None and bool(params["opt.include_prevention"]):
        prevention_method = getattr(models, "pPrev", None)
        gain_method = getattr(models, "prevention_gain", None)
        if prevention_method is not None and gain_method is not None:
            prevention = _batch_vector(prevention_method(frame, ack), count, "pPrev")
            result += prevention * expected_gain
    cost = np.zeros(count, dtype="float64")
    for name in chosen:
        item = params[f"evidence.{name}"]
        probability = material[name]
        if bool(item["system_sent"]):
            cost += float(item["cash"])
        else:
            cost += float(item["cash"]) * probability
        cost += float(item["seconds"]) * float(params["econ.hourly_rate"]) / 3600.0
    result = p_a * result - cost
    result[~np.isin(np.arange(count), positions)] = 0.0
    return result


def _weighted_subsets(
    values: np.ndarray,
    chosen: Sequence[str],
    material: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Evaluate every materialisation subset with a weighted zeta transform."""
    result = np.asarray(values, dtype="float64").copy()
    for position, name in enumerate(chosen):
        bit = 1 << position
        probability = material[name]
        absent = 1.0 - probability
        for mask in range(len(result)):
            if mask & bit:
                result[mask] = absent * result[mask ^ bit] + probability * result[mask]
    return result


def _batch_group_ev(
    frame: pd.DataFrame,
    positions: np.ndarray,
    chosen: Sequence[str],
    models: Any,
    params: Params,
    p_a: np.ndarray,
    p_b: np.ndarray,
    material: Mapping[str, np.ndarray],
    contest_cache: dict[int, np.ndarray],
    win_cache: dict[tuple[str, int], np.ndarray],
    gain_cache: dict[tuple[str, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return every subset EV and its corresponding bitmask for one group."""
    count = len(frame)
    type_names = ("NR", "NAD", "EB")
    subset_count = 1 << len(chosen)
    held_masks = np.zeros(subset_count, dtype="uint16")
    for mask in range(subset_count):
        for position, name in enumerate(chosen):
            if mask & (1 << position):
                held_masks[mask] |= bit_for(name, params)
    base: dict[str, np.ndarray] = {}
    for dispute_type in type_names:
        key = (dispute_type, 0)
        if key not in win_cache:
            win_cache[key] = _batch_win(models, frame, dispute_type, 0)
        base[dispute_type] = win_cache[key]
    benefit = np.zeros((subset_count, count), dtype="float64")
    gain = np.zeros((subset_count, count), dtype="float64")
    gain_method = getattr(models, "prevention_gain", None)
    for local_mask, held_value in enumerate(held_masks):
        held = int(held_value)
        if held not in contest_cache:
            contest_cache[held] = _batch_contest(models, frame, held)
        contest = contest_cache[held]
        for type_position, dispute_type in enumerate(type_names):
            key = (dispute_type, held)
            if key not in win_cache:
                win_cache[key] = _batch_win(models, frame, dispute_type, held)
            value = contest * (win_cache[key] - base[dispute_type]) * frame["order_value"].to_numpy(dtype="float64")
            benefit[local_mask] += p_b[:, type_position] * value
            if gain_method is not None:
                if key not in gain_cache:
                    gain_cache[key] = _batch_amount(
                        gain_method(frame, contest, win_cache[key]), count, "prevention_gain"
                    )
                gain[local_mask] += p_b[:, type_position] * gain_cache[key]
    transformed_benefit = _weighted_subsets(benefit, chosen, material)
    transformed_gain = _weighted_subsets(gain, chosen, material)
    ev = p_a[None, :] * transformed_benefit
    ack = _ack_kind(chosen)
    if ack is not None and bool(params["opt.include_prevention"]):
        prevention_method = getattr(models, "pPrev", None)
        if prevention_method is not None and gain_method is not None:
            prevention = _batch_vector(prevention_method(frame, ack), count, "pPrev")
            ev += prevention[None, :] * transformed_gain
    cost = np.zeros((subset_count, count), dtype="float64")
    for local_mask in range(subset_count):
        for position, name in enumerate(chosen):
            if local_mask & (1 << position):
                item = params[f"evidence.{name}"]
                if bool(item["system_sent"]):
                    cost[local_mask] += float(item["cash"])
                else:
                    cost[local_mask] += float(item["cash"]) * material[name]
                cost[local_mask] += float(item["seconds"]) * float(params["econ.hourly_rate"]) / 3600.0
    ev -= cost
    del positions
    return ev, held_masks


def plan_frame(
    observed: pd.DataFrame,
    models: Any,
    masks: Any = None,
    params: Params = P,
) -> np.ndarray:
    """Plan a frame with chunk-local vectorised model calls.

    A model that only implements the scalar protocol is handled by the
    exhaustive scalar function, which keeps the optimizer useful for tests
    and third-party model adapters.
    """
    check(observed, "ORDER_OBSERVED")
    if len(observed) == 0:
        return np.asarray([], dtype="uint16")
    try:
        p_a = _batch_vector(models.pA(observed), len(observed), "pA")
        p_b = _batch_types(models.pB(observed), len(observed))
        material = {name: _batch_vector(models.pM(observed, name), len(observed), "pM") for name in evidence_names(params)}
    except (AttributeError, KeyError, TypeError, IndexError, ValueError):
        return np.asarray([plan(row, models, masks, params) for _, row in observed.iterrows()], dtype="uint16")
    batch_models = models.prepare_batch(observed) if hasattr(models, "prepare_batch") else models
    support = _support_source(models, masks)
    availability = available_mask(observed, params).astype("uint16")
    type_names = ("NR", "NAD", "EB")
    groups: dict[tuple[str, ...], list[int]] = {}
    for position, row in enumerate(observed.itertuples(index=False)):
        del row
        selected: list[str] = []
        for name in evidence_names(params):
            bit = int(bit_for(name, params))
            if availability[position] & bit == 0:
                continue
            admissible = False
            for type_position, dispute_type in enumerate(type_names):
                if p_b[position, type_position] <= 0.0:
                    continue
                item = params[f"evidence.{name}"]
                if dispute_type in item["admissible"] and _support_allowed(support, dispute_type, name):
                    admissible = True
                    break
            if admissible:
                selected.append(name)
        key = tuple(selected)
        groups.setdefault(key, []).append(position)
    contest_cache: dict[int, np.ndarray] = {}
    win_cache: dict[tuple[str, int], np.ndarray] = {}
    gain_cache: dict[tuple[str, int], np.ndarray] = {}
    result = np.zeros(len(observed), dtype="uint16")
    all_rows = np.arange(len(observed))
    for chosen, positions_list in groups.items():
        positions = np.asarray(positions_list, dtype="int64")
        if not chosen:
            continue
        values, bitmasks = _batch_group_ev(
            observed,
            positions,
            chosen,
            batch_models,
            params,
            p_a,
            p_b,
            material,
            contest_cache,
            win_cache,
            gain_cache,
        )
        local_values = values[:, positions]
        local_best = local_values.argmax(axis=0)
        result[positions] = bitmasks[local_best]
    return result
