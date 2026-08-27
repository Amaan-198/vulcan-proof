"""Stage B conditional dispute-type model."""

from __future__ import annotations

from typing import Any
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ._common import (
    ConstantMulticlass,
    FeatureEncoder,
    align_outcome,
    calibration_guard,
    calibrator,
    feature_frame,
    make_multiclass_model,
    observed_features,
    validate_observed,
)
from ..seeds import SeedTree
from .labels import eligible


DISPUTE_TYPES = ("NR", "NAD", "EB")


class StageBModel:
    """Estimate dispute type conditional on exposure opening."""

    def __init__(self, params: Params = P) -> None:
        self.params = params
        self.encoder: FeatureEncoder | None = None
        self.model: object | None = None
        self.calibrators: list[object] = []
        self.metrics: dict[str, Any] = {}
        self.validation_rates = np.zeros(len(DISPUTE_TYPES), dtype="float64")

    def fit(
        self,
        observed: pd.DataFrame,
        outcome: pd.DataFrame,
        params: Params | None = None,
        seed: int = 0,
    ) -> "StageBModel":
        """Fit only on uncensored rows where a dispute opened."""
        if params is not None:
            self.params = params
        validate_observed(observed)
        aligned = align_outcome(observed, outcome)
        train = eligible(aligned, "train", purpose="fit").to_numpy()
        valid = eligible(aligned, "validate", purpose="fit").to_numpy()
        opened = aligned["dispute_opened"].to_numpy(dtype="int8").astype(bool)
        train &= opened
        valid &= opened
        labels = aligned["dispute_type"].astype(str).to_numpy()
        y_train = np.asarray([DISPUTE_TYPES.index(value) for value in labels[train]], dtype="int32")
        y_valid = np.asarray([DISPUTE_TYPES.index(value) for value in labels[valid]], dtype="int32")
        if len(y_train) == 0:
            raise InvariantError("Stage B has no eligible opened training rows")
        columns = observed_features(self.params)
        features = feature_frame(observed, columns)
        self.encoder = FeatureEncoder(columns).fit(features.loc[train])
        x_train = self.encoder.transform(features.loc[train])
        x_valid = self.encoder.transform(features.loc[valid])
        frequencies = np.bincount(y_train, minlength=len(DISPUTE_TYPES)).astype("float64")
        if float(frequencies.sum()) > 0.0:
            frequencies /= frequencies.sum()
        else:
            frequencies[:] = 1.0 / len(DISPUTE_TYPES)
        if len(np.unique(y_train)) < len(DISPUTE_TYPES):
            self.model = ConstantMulticlass(frequencies)
        else:
            config = dict(self.params["models.lgbm.stage_b"])
            config["num_threads"] = int(self.params["run.lgbm_threads"])
            config["random_state"] = int(
                SeedTree(int(seed)).child("stage_b").integers(0, int(np.iinfo("uint32").max))
            )
            self.model = lgb.LGBMClassifier(**config)
            if len(y_valid) > 0 and len(np.unique(y_valid)) > 1:
                self.model.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_valid, y_valid)],
                    callbacks=[lgb.early_stopping(int(self.params["models.early_stopping_rounds"]), verbose=False)],
                )
            else:
                self.model.fit(x_train, y_train)
        raw = self._raw(x_valid)
        self.validation_rates = np.asarray(
            [(y_valid == position).mean() if len(y_valid) else frequencies[position] for position in range(len(DISPUTE_TYPES))],
            dtype="float64",
        )
        self.calibrators = [
            calibrator(raw[:, position], (y_valid == position).astype("int8"), self.validation_rates[position])
            for position in range(len(DISPUTE_TYPES))
        ]
        calibrated = self._calibrated(raw)
        self._assert_rows(calibrated)
        for position in range(len(DISPUTE_TYPES)):
            calibration_guard(calibrated[:, position], (y_valid == position).astype("int8"), self.params)
        self.metrics = self._metrics(calibrated, y_valid)
        self.metrics["feature_names"] = list(columns)
        return self

    def _raw(self, values: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise InvariantError("Stage B has not been fitted")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = np.asarray(self.model.predict_proba(values), dtype="float64")
        if result.ndim != 2 or result.shape[1] != len(DISPUTE_TYPES):
            raise InvariantError("Stage B returned an invalid probability matrix")
        return result

    def _calibrated(self, values: np.ndarray) -> np.ndarray:
        if len(self.calibrators) != len(DISPUTE_TYPES):
            raise InvariantError("Stage B calibrators are missing")
        result = np.column_stack([
            np.clip(calibrator_item.predict(values[:, position]), 0.0, 1.0)
            for position, calibrator_item in enumerate(self.calibrators)
        ])
        totals = result.sum(axis=1)
        bad = totals <= 0.0
        result[~bad] /= totals[~bad, None]
        if bool(bad.any()):
            result[bad] = 1.0 / len(DISPUTE_TYPES)
        self._assert_rows(result)
        return result

    @staticmethod
    def _assert_rows(values: np.ndarray) -> None:
        if not np.all(np.abs(values.sum(axis=1) - 1.0) < 1e-9):
            raise InvariantError("Stage B probabilities do not sum to one")

    def _metrics(self, values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

        predicted = values.argmax(axis=1) if len(values) else np.asarray([], dtype="int32")
        return {
            "per_class": {
                name: {
                    "precision": float(precision_score(labels, predicted, labels=[position], average="macro", zero_division=0)) if len(labels) else 0.0,
                    "recall": float(recall_score(labels, predicted, labels=[position], average="macro", zero_division=0)) if len(labels) else 0.0,
                    "f1": float(f1_score(labels, predicted, labels=[position], average="macro", zero_division=0)) if len(labels) else 0.0,
                    "calibrated_mean": float(values[:, position].mean()) if len(values) else 0.0,
                    "rate": float((labels == position).mean()) if len(labels) else 0.0,
                }
                for position, name in enumerate(DISPUTE_TYPES)
            },
            "confusion": confusion_matrix(labels, predicted, labels=list(range(len(DISPUTE_TYPES)))).tolist() if len(labels) else [],
        }

    def predict_proba(self, observed: pd.DataFrame) -> np.ndarray:
        """Return calibrated, row-normalised type probabilities."""
        validate_observed(observed)
        if self.encoder is None:
            raise InvariantError("Stage B has not been fitted")
        return self._calibrated(self._raw(self.encoder.transform(feature_frame(observed, self.encoder.columns))))

    def predict(self, observed: pd.DataFrame) -> np.ndarray:
        """Alias for :meth:`predict_proba`."""
        return self.predict_proba(observed)


def fit_stage_b(
    observed: pd.DataFrame,
    outcome: pd.DataFrame,
    params: Params = P,
    seed: int = 0,
) -> StageBModel:
    """Fit and return Stage B."""
    return StageBModel(params).fit(observed, outcome, seed=seed)


fit = fit_stage_b
StageB = StageBModel
