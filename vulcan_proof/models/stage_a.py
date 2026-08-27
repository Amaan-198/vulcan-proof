"""Stage A exposure model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ._common import (
    FeatureEncoder,
    align_outcome,
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


class StageAModel:
    """Calibrated estimate of exposure before the deployed evidence plan."""

    target_name = "exposure"

    def __init__(self, params: Params = P) -> None:
        self.params = params
        self.encoder: FeatureEncoder | None = None
        self.model: object | None = None
        self.fitted_calibrator: object | None = None
        self.metrics: dict[str, Any] = {}
        self.validation_rate = 0.0

    def fit(
        self,
        observed: pd.DataFrame,
        outcome: pd.DataFrame,
        params: Params | None = None,
        seed: int = 0,
    ) -> "StageAModel":
        """Fit on uncensored train rows and calibrate on uncensored validation rows."""
        if params is not None:
            self.params = params
        validate_observed(observed)
        aligned = align_outcome(observed, outcome)
        train = eligible(aligned, "train", purpose="fit").to_numpy()
        valid = eligible(aligned, "validate", purpose="fit").to_numpy()
        target = (aligned["dispute_opened"].to_numpy(dtype="int8") != 0) | (
            aligned["prevented"].to_numpy(dtype="int8") != 0
        )
        columns = observed_features(self.params)
        features = feature_frame(observed, columns)
        self.encoder = FeatureEncoder(columns).fit(features.loc[train])
        x_train = self.encoder.transform(features.loc[train])
        x_valid = self.encoder.transform(features.loc[valid])
        y_train = target[train].astype("int8")
        y_valid = target[valid].astype("int8")
        if len(y_train) == 0:
            raise InvariantError("Stage A has no eligible training rows")
        self.model, _ = fit_binary(
            make_binary_model("stage_a", self.params, seed),
            x_train,
            y_train,
            x_valid,
            y_valid,
            float(y_train.mean()),
            self.params,
        )
        raw_valid = self._raw_matrix(x_valid)
        self.validation_rate = float(y_valid.mean()) if len(y_valid) else float(y_train.mean())
        self.fitted_calibrator = calibrator(raw_valid, y_valid, self.validation_rate)
        calibrated_valid = self._calibrate(raw_valid)
        calibration_guard(calibrated_valid, y_valid, self.params)
        self.metrics = make_metrics(calibrated_valid, y_valid, self.params)
        self.metrics["validation_rate"] = self.validation_rate
        self.metrics["feature_names"] = list(columns)
        return self

    def _raw_matrix(self, values: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise InvariantError("Stage A has not been fitted")
        return numeric_prediction(self.model, values)

    def _calibrate(self, values: np.ndarray) -> np.ndarray:
        if self.fitted_calibrator is None:
            raise InvariantError("Stage A calibrator is missing")
        return np.clip(np.asarray(self.fitted_calibrator.predict(values), dtype="float64"), 0.0, 1.0)

    def predict_raw(self, observed: pd.DataFrame) -> np.ndarray:
        """Return uncalibrated exposure probabilities."""
        validate_observed(observed)
        if self.encoder is None:
            raise InvariantError("Stage A has not been fitted")
        return self._raw_matrix(self.encoder.transform(feature_frame(observed, self.encoder.columns)))

    def predict(self, observed: pd.DataFrame) -> np.ndarray:
        """Return calibrated exposure probabilities."""
        validate_observed(observed)
        return self._calibrate(self.predict_raw(observed))

    predict_proba = predict


def fit_stage_a(
    observed: pd.DataFrame,
    outcome: pd.DataFrame,
    params: Params = P,
    seed: int = 0,
) -> StageAModel:
    """Fit and return Stage A."""
    return StageAModel(params).fit(observed, outcome, seed=seed)


fit = fit_stage_a
StageA = StageAModel
