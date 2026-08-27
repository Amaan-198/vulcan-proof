"""Defensibility model and observed support masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ._common import (
    FeatureEncoder,
    align_outcome,
    bit_columns,
    calibration_guard,
    calibrator,
    feature_frame,
    fit_binary,
    make_binary_model,
    make_metrics,
    numeric_prediction,
    observed_features,
    validate_observed,
)
from .labels import eligible


DISPUTE_TYPES = ("NR", "NAD", "EB")


@dataclass(frozen=True)
class SupportMasks:
    """Pair and bitmask support learned from eligible contested rows."""

    pair_counts: dict[tuple[str, str], int]
    bit_counts: dict[tuple[str, int], int]
    minimum: int

    def pair_allowed(self, dispute_type: str, evidence: str) -> bool:
        """Return whether a type/evidence pair has enough support."""
        return self.pair_counts.get((str(dispute_type), str(evidence)), 0) >= self.minimum

    def bit_confidence(self, dispute_type: str, bitmask: int) -> float:
        """Return support shrinkage confidence for one type and held mask."""
        count = self.bit_counts.get((str(dispute_type), int(bitmask)), 0)
        return float(min(1.0, count / self.minimum)) if self.minimum else 1.0

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-ready support tables."""
        return {
            "minimum": self.minimum,
            "pair_counts": {f"{d}|{e}": count for (d, e), count in self.pair_counts.items()},
            "pair_allowed": {f"{d}|{e}": self.pair_allowed(d, e) for d, e in self.pair_counts},
            "bit_counts": {f"{d}|{mask}": count for (d, mask), count in self.bit_counts.items()},
            "pairs_masked": int(sum(not allowed for allowed in [self.pair_allowed(d, e) for d, e in self.pair_counts])),
            "bitmasks_supported": int(sum(count >= self.minimum for count in self.bit_counts.values())),
        }


def build_support_masks(
    outcome: pd.DataFrame,
    params: Params = P,
    observed: pd.DataFrame | None = None,
) -> SupportMasks:
    """Count pair and realised-bitmask support on eligible train disputes."""
    from ..schemas import check

    check(outcome, "OUTCOME")
    rows = eligible(outcome, "train", purpose="evaluate").to_numpy()
    rows &= outcome["dispute_opened"].to_numpy(dtype="int8").astype(bool)
    rows &= outcome["contested"].to_numpy(dtype="int8").astype(bool)
    names = tuple(params["evidence.order"])
    pair_counts: dict[tuple[str, str], int] = {}
    bit_counts: dict[tuple[str, int], int] = {}
    dispute = outcome["dispute_type"].astype(str).to_numpy()
    masks = outcome["materialised_bitmask"].to_numpy(dtype="uint16")
    for d in DISPUTE_TYPES:
        type_rows = rows & (dispute == d)
        for position, name in enumerate(names):
            bit = np.uint16(1 << position)
            pair_counts[(d, name)] = int((type_rows & ((masks & bit) != 0)).sum())
        for mask_value, count in zip(*np.unique(masks[type_rows], return_counts=True), strict=True):
            bit_counts[(d, int(mask_value))] = int(count)
    del observed
    return SupportMasks(pair_counts, bit_counts, int(params["models.support_min"]))


class DefensibilityModel:
    """Calibrated win model with support-weighted main-effect shrinkage."""

    def __init__(self, params: Params = P) -> None:
        self.params = params
        self.encoder: FeatureEncoder | None = None
        self.model: object | None = None
        self.fitted_calibrator: object | None = None
        self.support_masks = SupportMasks({}, {}, int(params["models.support_min"]))
        self.metrics: dict[str, Any] = {}
        self.feature_columns: tuple[str, ...] = ()
        self.validation_rate = 0.0
        self.main_effects: dict[tuple[str, int], float] = {}

    def fit(
        self,
        observed: pd.DataFrame,
        outcome: pd.DataFrame,
        params: Params | None = None,
        seed: int = 0,
    ) -> "DefensibilityModel":
        """Fit on eligible contested rows and calibrate on validation rows."""
        if params is not None:
            self.params = params
        validate_observed(observed)
        aligned = align_outcome(observed, outcome)
        self.support_masks = build_support_masks(aligned, self.params, observed)
        train = eligible(aligned, "train", purpose="fit").to_numpy()
        valid = eligible(aligned, "validate", purpose="fit").to_numpy()
        opened = aligned["dispute_opened"].to_numpy(dtype="int8").astype(bool)
        contested = aligned["contested"].to_numpy(dtype="int8").astype(bool)
        train &= opened & contested
        valid &= opened & contested
        if not bool(train.any()):
            raise InvariantError("defensibility model has no eligible contested rows")
        names = tuple(self.params["evidence.order"])
        columns = observed_features(self.params)
        self.feature_columns = columns + ("dispute_type",) + tuple(f"materialised_{name}" for name in names)
        base = feature_frame(observed, columns)
        train_frame = self._augmented(base.loc[train], aligned.loc[train], names)
        valid_frame = self._augmented(base.loc[valid], aligned.loc[valid], names)
        self.encoder = FeatureEncoder(self.feature_columns).fit(train_frame)
        x_train = self.encoder.transform(train_frame)
        x_valid = self.encoder.transform(valid_frame)
        y_train = aligned.loc[train, "won"].to_numpy(dtype="int8")
        y_valid = aligned.loc[valid, "won"].to_numpy(dtype="int8")
        self.model, _ = fit_binary(
            make_binary_model("stage_w", self.params, seed),
            x_train,
            y_train,
            x_valid,
            y_valid,
            float(y_train.mean()),
            self.params,
        )
        raw_valid = self._raw(x_valid)
        self.validation_rate = float(y_valid.mean()) if len(y_valid) else float(y_train.mean())
        self.fitted_calibrator = calibrator(raw_valid, y_valid, self.validation_rate)
        calibrated = self._calibrate(raw_valid)
        calibration_guard(calibrated, y_valid, self.params)
        self._learn_main_effects(train_frame, y_train, names)
        self.metrics = make_metrics(calibrated, y_valid, self.params)
        self.metrics["validation_rate"] = self.validation_rate
        self.metrics["feature_names"] = list(self.feature_columns)
        self.metrics["support"] = self.support_masks.as_dict()
        return self

    @staticmethod
    def _augmented(base: pd.DataFrame, outcome: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
        disputes = outcome["dispute_type"].astype(str).reset_index(drop=True).rename("dispute_type")
        bits = bit_columns(outcome["materialised_bitmask"].to_numpy(dtype="uint16"), names)
        return pd.concat([base.reset_index(drop=True), disputes, bits], axis=1)

    def _learn_main_effects(self, frame: pd.DataFrame, labels: np.ndarray, names: Sequence[str]) -> None:
        disputes = frame["dispute_type"].astype(str).to_numpy()
        masks = np.zeros(len(frame), dtype="uint16")
        for position, name in enumerate(names):
            masks |= np.where(frame[f"materialised_{name}"].to_numpy(dtype="int8") != 0, np.uint16(1 << position), np.uint16(0))
        for d in DISPUTE_TYPES:
            base = labels[(disputes == d) & (masks == 0)]
            base_rate = float(base.mean()) if len(base) else float(labels.mean())
            self.main_effects[(d, 0)] = base_rate
            for position, name in enumerate(names):
                bit = int(1 << position)
                rows = (disputes == d) & (masks == bit)
                rate = float(labels[rows].mean()) if bool(rows.any()) else base_rate
                self.main_effects[(d, bit)] = rate - base_rate

    def _raw(self, values: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise InvariantError("defensibility model has not been fitted")
        return numeric_prediction(self.model, values)

    def _calibrate(self, values: np.ndarray) -> np.ndarray:
        if self.fitted_calibrator is None:
            raise InvariantError("defensibility calibrator is missing")
        return np.clip(np.asarray(self.fitted_calibrator.predict(values), dtype="float64"), 0.0, 1.0)

    def predict(
        self,
        observed: pd.DataFrame,
        dispute_type: str,
        held_mask: int | Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return support-shrunk calibrated win probabilities."""
        validate_observed(observed)
        if self.encoder is None:
            raise InvariantError("defensibility model has not been fitted")
        if dispute_type not in DISPUTE_TYPES:
            raise InvariantError(f"unknown dispute type: {dispute_type!r}")
        if held_mask is None:
            masks = np.zeros(len(observed), dtype="uint16")
        elif np.isscalar(held_mask):
            masks = np.full(len(observed), int(held_mask), dtype="uint16")
        else:
            masks = np.asarray(held_mask, dtype="uint16")
            if len(masks) != len(observed):
                raise InvariantError("held masks are not aligned")
        disputes = pd.Series(pd.Categorical([dispute_type] * len(observed), categories=list(DISPUTE_TYPES)), name="dispute_type")
        frame = pd.concat([
            feature_frame(observed, observed_features(self.params)).reset_index(drop=True),
            disputes,
            bit_columns(masks, tuple(self.params["evidence.order"])),
        ], axis=1)
        full = self._calibrate(self._raw(self.encoder.transform(frame)))
        base_frame = frame.copy()
        names = tuple(self.params["evidence.order"])
        for name in names:
            base_frame[f"materialised_{name}"] = 0
        base = self._calibrate(self._raw(self.encoder.transform(base_frame)))
        main = np.full(len(observed), self.main_effects.get((dispute_type, 0), 0.0), dtype="float64")
        for position, name in enumerate(names):
            bit = int(1 << position)
            effect = self.main_effects.get((dispute_type, bit), 0.0)
            main += np.where((masks & np.uint16(bit)) != 0, effect, 0.0)
        confidence = np.asarray([self.support_masks.bit_confidence(dispute_type, int(mask)) for mask in masks], dtype="float64")
        result = confidence * full + (1.0 - confidence) * np.clip(base + main - self.main_effects.get((dispute_type, 0), 0.0), 0.0, 1.0)
        return np.clip(result, 0.0, 1.0)

    predict_proba = predict


def fit_defensibility(
    observed: pd.DataFrame,
    outcome: pd.DataFrame,
    params: Params = P,
    seed: int = 0,
) -> DefensibilityModel:
    """Fit and return the defensibility model."""
    return DefensibilityModel(params).fit(observed, outcome, seed=seed)


fit = fit_defensibility
Defensibility = DefensibilityModel
SupportMask = SupportMasks
