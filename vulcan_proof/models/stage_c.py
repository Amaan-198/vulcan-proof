"""Stage C evidence-dependent contest model."""

from __future__ import annotations

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


class StageCModel:
    """Estimate contest probability as a function of the held evidence mask."""

    def __init__(self, params: Params = P) -> None:
        self.params = params
        self.encoder: FeatureEncoder | None = None
        self.model: object | None = None
        self.fitted_calibrator: object | None = None
        self.metrics: dict[str, Any] = {}
        self.validation_rate = 0.0
        self.feature_columns: tuple[str, ...] = ()
        self.plan_effects: dict[str, float] = {}

    def fit(
        self,
        observed: pd.DataFrame,
        outcome: pd.DataFrame,
        params: Params | None = None,
        seed: int = 0,
    ) -> "StageCModel":
        """Fit on uncensored opened rows using historical materialisation bits."""
        if params is not None:
            self.params = params
        validate_observed(observed)
        aligned = align_outcome(observed, outcome)
        train = eligible(aligned, "train", purpose="fit").to_numpy()
        valid = eligible(aligned, "validate", purpose="fit").to_numpy()
        opened = aligned["dispute_opened"].to_numpy(dtype="int8").astype(bool)
        train &= opened
        valid &= opened
        y_all = aligned["contested"].to_numpy(dtype="int8")
        names = tuple(self.params["evidence.order"])
        columns = observed_features(self.params)
        self.feature_columns = columns + tuple(f"materialised_{name}" for name in names)
        base = feature_frame(observed, columns)
        train_bits = bit_columns(aligned.loc[train, "materialised_bitmask"].to_numpy(dtype="uint16"), names)
        valid_bits = bit_columns(aligned.loc[valid, "materialised_bitmask"].to_numpy(dtype="uint16"), names)
        train_frame = pd.concat([base.loc[train].reset_index(drop=True), train_bits], axis=1)
        valid_frame = pd.concat([base.loc[valid].reset_index(drop=True), valid_bits], axis=1)
        self.encoder = FeatureEncoder(self.feature_columns).fit(train_frame)
        x_train = self.encoder.transform(train_frame)
        x_valid = self.encoder.transform(valid_frame)
        y_train = y_all[train]
        y_valid = y_all[valid]
        self.model, _ = fit_binary(
            make_binary_model("stage_c", self.params, seed),
            x_train,
            y_train,
            x_valid,
            y_valid,
            float(y_train.mean()) if len(y_train) else 0.0,
            self.params,
        )
        self._learn_plan_effects(train_frame, y_train, names)
        valid_masks = aligned.loc[valid, "materialised_bitmask"].to_numpy(dtype="uint16")
        raw_valid = self._raw_matrix(x_valid) + self._plan_adjustment(valid_masks, names)
        raw_valid = np.clip(raw_valid, 0.0, 1.0)
        self.validation_rate = float(y_valid.mean()) if len(y_valid) else float(y_train.mean())
        self.fitted_calibrator = calibrator(raw_valid, y_valid, self.validation_rate)
        calibrated_valid = self._calibrate(raw_valid)
        calibration_guard(calibrated_valid, y_valid, self.params)
        self.metrics = make_metrics(calibrated_valid, y_valid, self.params)
        self.metrics["validation_rate"] = self.validation_rate
        self.metrics["feature_names"] = list(self.feature_columns)
        return self

    def _learn_plan_effects(self, frame: pd.DataFrame, labels: np.ndarray, names: Sequence[str]) -> None:
        """Learn empirical plan effects as a fallback for shallow trees."""
        for name in names:
            held = frame[f"materialised_{name}"].to_numpy(dtype="int8").astype(bool)
            left = float(labels[held].mean()) if bool(held.any()) else float(labels.mean())
            right = float(labels[~held].mean()) if bool((~held).any()) else float(labels.mean())
            self.plan_effects[name] = left - right

    def _plan_adjustment(self, masks: np.ndarray, names: Sequence[str]) -> np.ndarray:
        """Return the row-local learned adjustment for a held-mask vector."""
        result = np.zeros(len(masks), dtype="float64")
        for position, name in enumerate(names):
            result += np.where((masks & np.uint16(1 << position)) != 0, self.plan_effects.get(name, 0.0), 0.0)
        return result

    def _frame_with_mask(self, observed: pd.DataFrame, values: int | Sequence[int] | np.ndarray | None) -> pd.DataFrame:
        validate_observed(observed)
        if values is None:
            masks = np.zeros(len(observed), dtype="uint16")
        elif np.isscalar(values):
            masks = np.full(len(observed), int(values), dtype="uint16")
        else:
            masks = np.asarray(values, dtype="uint16")
            if len(masks) != len(observed):
                raise InvariantError("planned mask is not aligned to observed rows")
        bits = bit_columns(masks, tuple(self.params["evidence.order"]))
        return pd.concat([feature_frame(observed, observed_features(self.params)).reset_index(drop=True), bits], axis=1)

    def _raw_matrix(self, values: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise InvariantError("Stage C has not been fitted")
        return numeric_prediction(self.model, values)

    def _calibrate(self, values: np.ndarray) -> np.ndarray:
        if self.fitted_calibrator is None:
            raise InvariantError("Stage C calibrator is missing")
        return np.clip(np.asarray(self.fitted_calibrator.predict(values), dtype="float64"), 0.0, 1.0)

    def predict_raw(
        self,
        observed: pd.DataFrame,
        planned_bitmask: int | Sequence[int] | np.ndarray | None = None,
        *,
        plan: int | Sequence[int] | np.ndarray | None = None,
        held_mask: int | Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return uncalibrated probabilities for a planned held mask."""
        if plan is not None:
            planned_bitmask = plan
        if held_mask is not None:
            planned_bitmask = held_mask
        if self.encoder is None:
            raise InvariantError("Stage C has not been fitted")
        matrix = self._frame_with_mask(observed, planned_bitmask)
        masks = self._mask_values(observed, planned_bitmask)
        return np.clip(
            self._raw_matrix(self.encoder.transform(matrix))
            + self._plan_adjustment(masks, tuple(self.params["evidence.order"])),
            0.0,
            1.0,
        )

    def predict(
        self,
        observed: pd.DataFrame,
        planned_bitmask: int | Sequence[int] | np.ndarray | None = None,
        *,
        plan: int | Sequence[int] | np.ndarray | None = None,
        held_mask: int | Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Return calibrated probabilities for a planned held mask."""
        if plan is not None:
            planned_bitmask = plan
        if held_mask is not None:
            planned_bitmask = held_mask
        return self._calibrate(self.predict_raw(observed, planned_bitmask))

    @staticmethod
    def _mask_values(observed: pd.DataFrame, values: int | Sequence[int] | np.ndarray | None) -> np.ndarray:
        """Normalise planned mask values for empirical plan adjustments."""
        if values is None:
            return np.zeros(len(observed), dtype="uint16")
        if np.isscalar(values):
            return np.full(len(observed), int(values), dtype="uint16")
        result = np.asarray(values, dtype="uint16")
        if len(result) != len(observed):
            raise InvariantError("planned mask is not aligned to observed rows")
        return result

    predict_proba = predict


def fit_stage_c(
    observed: pd.DataFrame,
    outcome: pd.DataFrame,
    params: Params = P,
    seed: int = 0,
) -> StageCModel:
    """Fit and return Stage C."""
    return StageCModel(params).fit(observed, outcome, seed=seed)


fit = fit_stage_c
StageC = StageCModel
