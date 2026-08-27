"""LightGBM Stage-A-shaped detector and validation isotonic calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ..errors import InvariantError
from ..params import P, Params
from ..schemas import check
from ..seeds import SeedTree


def _encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode declared categorical columns without changing the feature set."""
    check(frame, "OLIST_ORDER_FEATURES")
    encoded = frame.copy()
    for column in encoded.columns:
        if isinstance(encoded[column].dtype, pd.CategoricalDtype):
            encoded[column] = encoded[column].cat.codes.astype("float32")
    return encoded.astype("float32")


@dataclass
class OlistDetector:
    """Fitted raw and calibrated detector."""

    estimator: lgb.LGBMClassifier
    calibrator: IsotonicRegression
    feature_columns: tuple[str, ...]

    def _matrix(self, features: pd.DataFrame) -> pd.DataFrame:
        check(features, "OLIST_ORDER_FEATURES")
        if tuple(features.columns) != self.feature_columns:
            raise InvariantError("prediction feature columns differ from the training schema")
        return _encode_features(features)

    def predict_raw(self, features: pd.DataFrame) -> np.ndarray:
        """Predict uncalibrated complaint probabilities."""
        return np.asarray(self.estimator.predict_proba(self._matrix(features))[:, 1], dtype="float64")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict validation-calibrated complaint probabilities."""
        raw = self.predict_raw(features)
        calibrated = np.asarray(self.calibrator.predict(raw), dtype="float64")
        if not np.isfinite(calibrated).all():
            raise InvariantError("calibrated Olist probabilities contain non-finite values")
        return calibrated


def _label_vector(labels: pd.DataFrame, feature_count: int) -> np.ndarray:
    if len(labels) != feature_count:
        raise InvariantError("labels and features have different row counts")
    return labels["label"].to_numpy(dtype="int8")


def fit_detector(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split_labels: pd.DataFrame,
    params: Params = P,
) -> OlistDetector:
    """Fit LightGBM on train and calibrate its probabilities on validation."""
    check(features, "OLIST_ORDER_FEATURES")
    check(labels, "OLIST_LABELS")
    if len(features) != len(labels) or not split_labels.index.equals(labels.index):
        raise InvariantError("features, labels, and split labels are not aligned")
    train_mask = split_labels["split"].eq("train").to_numpy()
    validate_mask = split_labels["split"].eq("validate").to_numpy()
    if not train_mask.any() or not validate_mask.any():
        raise InvariantError("Olist training and validation splits must be non-empty")
    y = _label_vector(labels, len(features))
    encoded = _encode_features(features)
    settings: dict[str, Any] = dict(params["models.lgbm.stage_a"])
    if float(settings["scale_pos_weight"]) != 1.0:
        raise InvariantError("Olist Stage A must use scale_pos_weight=1")
    seed_tree = SeedTree(int(params["run.master_seed"]))
    seed_generator = seed_tree.child("olist", "lgbm")
    seed = int(seed_generator.integers(0, np.iinfo(np.uint32).max))
    settings["random_state"] = seed
    settings["seed"] = seed
    settings["num_threads"] = int(params["run.lgbm_threads"])
    settings["verbosity"] = int(settings.pop("verbose"))
    estimator = lgb.LGBMClassifier(**settings)
    callback = lgb.early_stopping(
        int(params["models.early_stopping_rounds"]), verbose=False
    )
    estimator.fit(
        encoded.loc[train_mask],
        y[train_mask],
        eval_set=[(encoded.loc[validate_mask], y[validate_mask])],
        eval_metric="binary_logloss",
        callbacks=[callback],
    )
    raw_validation = np.asarray(
        estimator.predict_proba(encoded.loc[validate_mask])[:, 1], dtype="float64"
    )
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_validation, y[validate_mask])
    return OlistDetector(estimator, calibrator, tuple(features.columns))


def calibration_summary(
    detector: OlistDetector,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split_labels: pd.DataFrame,
    split_name: str,
) -> dict[str, float]:
    """Summarise raw and calibrated means versus empirical rate for one split."""
    mask = split_labels["split"].eq(split_name).to_numpy()
    raw_scores = detector.predict_raw(features.loc[mask])
    scores = np.asarray(detector.calibrator.predict(raw_scores), dtype="float64")
    y = labels.loc[mask, "label"].to_numpy(dtype="int8")
    rate = float(y.mean())
    mean = float(scores.mean())
    ratio = mean / rate if rate else float("inf")
    return {
        "raw_mean_prediction": float(raw_scores.mean()),
        "mean_prediction": mean,
        "empirical_rate": rate,
        "mean_vs_rate_ratio": ratio,
    }
