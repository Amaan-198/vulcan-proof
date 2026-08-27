"""Shared feature, label, and calibration utilities for Phase 3."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from ..errors import InvariantError, SchemaError
from ..params import P, Params
from ..schemas import check
from ..seeds import SeedTree


class ConstantCalibrator:
    """Small calibrator used when a validation fold has one observed class."""

    def __init__(self, value: float) -> None:
        self.value = float(np.clip(value, 0.0, 1.0))

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Return the constant calibrated probability for every input."""
        return np.full(len(values), self.value, dtype="float64")


class ProbabilityCalibrator:
    """Isotonic calibrator with a safe constant fallback."""

    def __init__(self, fitted: IsotonicRegression | ConstantCalibrator) -> None:
        self.fitted = fitted

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Calibrate a one-dimensional probability vector."""
        values = np.asarray(values, dtype="float64")
        if isinstance(self.fitted, ConstantCalibrator):
            return np.full(len(values), self.fitted.value, dtype="float64")
        return np.asarray(self.fitted.predict(values), dtype="float64")


class FeatureEncoder:
    """Deterministic numeric encoding for mixed pandas frames."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self.kinds: dict[str, str] = {}
        self.levels: dict[str, tuple[str, ...]] = {}

    def fit(self, frame: pd.DataFrame) -> "FeatureEncoder":
        """Learn categorical levels and validate the feature frame."""
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise SchemaError(f"model features are missing columns: {sorted(missing)}")
        for column in self.columns:
            series = frame[column]
            if isinstance(series.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(series):
                levels = tuple(sorted(series.astype(str).unique().tolist()))
                self.kinds[column] = "category"
                self.levels[column] = levels
            else:
                self.kinds[column] = "numeric"
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Encode a frame using levels learned by :meth:`fit`."""
        missing = set(self.columns) - set(frame.columns)
        if missing:
            raise SchemaError(f"model features are missing columns: {sorted(missing)}")
        result: list[np.ndarray] = []
        for column in self.columns:
            series = frame[column]
            if self.kinds.get(column) == "category":
                lookup = {name: position for position, name in enumerate(self.levels[column])}
                values = series.astype(str).map(lookup).fillna(-1).to_numpy(dtype="float64")
            else:
                values = pd.to_numeric(series, errors="raise").to_numpy(dtype="float64")
            result.append(values)
        if not result:
            return np.empty((len(frame), 0), dtype="float64")
        return np.column_stack(result).astype("float32", copy=False)


def observed_features(params: Params = P) -> tuple[str, ...]:
    """Return permitted model features after removing identifiers and time."""
    ids = {"order_id", "merchant_id", "customer_id", "decision_date"}
    return tuple(column for column in params["features.permitted"] if column not in ids)


def validate_observed(frame: pd.DataFrame) -> None:
    """Apply the exact observed schema and reject missing model inputs."""
    check(frame, "ORDER_OBSERVED")
    permitted = tuple(P["features.permitted"])
    if frame[list(permitted)].isna().any().any():
        raise SchemaError("Phase-3 model inputs may not contain NaN permitted features")
    for column in permitted:
        if not isinstance(frame[column].dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(frame[column]):
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
            if not np.isfinite(values).all():
                raise SchemaError(f"Phase-3 feature {column!r} is not finite")


def align_outcome(observed: pd.DataFrame, outcome: pd.DataFrame) -> pd.DataFrame:
    """Align an OUTCOME frame to observed-order order without changing labels."""
    check(outcome, "OUTCOME")
    left = observed["order_id"].astype(str)
    right = outcome["order_id"].astype(str)
    if not right.is_unique or not left.is_unique:
        raise InvariantError("model inputs require unique order ids")
    if set(left) != set(right):
        raise InvariantError("observed and outcome order ids differ")
    aligned = outcome.copy()
    aligned["_order_key"] = right.to_numpy()
    aligned = aligned.set_index("_order_key").loc[left.to_numpy()].reset_index(drop=True)
    return aligned


def feature_frame(observed: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Return a copy of the permitted, non-identifier feature columns."""
    validate_observed(observed)
    chosen = tuple(columns) if columns is not None else observed_features()
    result = observed.loc[:, list(chosen)].copy()
    if result.isna().any().any():
        raise SchemaError("model feature frame contains NaN")
    return result


def bit_columns(
    values: Iterable[int],
    names: Sequence[str],
    prefix: str = "materialised_",
) -> pd.DataFrame:
    """Expand bitmasks into deterministic binary columns."""
    masks = np.asarray(list(values), dtype="uint16")
    result: dict[str, np.ndarray] = {}
    for position, name in enumerate(names):
        bit = np.uint16(1 << position)
        result[f"{prefix}{name}"] = ((masks & bit) != 0).astype("int8")
    return pd.DataFrame(result)


def make_binary_model(label: str, params: Params, seed: int) -> lgb.LGBMClassifier:
    """Create a deterministic LightGBM binary model from the parameter tree."""
    config = dict(params[f"models.lgbm.{label}"])
    config["num_threads"] = int(params["run.lgbm_threads"])
    config["random_state"] = int(
        SeedTree(int(seed)).child(label).integers(0, int(np.iinfo("uint32").max))
    )
    return lgb.LGBMClassifier(**config)


def make_multiclass_model(label: str, params: Params, seed: int) -> lgb.LGBMClassifier:
    """Create a deterministic LightGBM multiclass model."""
    config = dict(params[f"models.lgbm.{label}"])
    config["num_threads"] = int(params["run.lgbm_threads"])
    config["random_state"] = int(
        SeedTree(int(seed)).child(label).integers(0, int(np.iinfo("uint32").max))
    )
    return lgb.LGBMClassifier(**config)


def fit_binary(
    model: lgb.LGBMClassifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    fallback: float,
    params: Params,
) -> tuple[object, float]:
    """Fit LightGBM or return a constant predictor for a degenerate fold."""
    if len(y_train) == 0:
        raise InvariantError("binary model has no training rows")
    if len(np.unique(y_train)) < 2:
        return ConstantBinary(float(y_train.mean())), float(y_train.mean())
    if len(y_valid) > 0 and len(np.unique(y_valid)) > 1:
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            callbacks=[lgb.early_stopping(int(params["models.early_stopping_rounds"]), verbose=False)],
        )
    else:
        model.fit(x_train, y_train)
    del fallback
    return model, float(y_train.mean())


class ConstantBinary:
    """Binary predictor for tiny or single-class training partitions."""

    def __init__(self, value: float) -> None:
        self.value = float(np.clip(value, 0.0, 1.0))

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        """Return two-column probabilities."""
        count = len(values)
        positive = np.full(count, self.value, dtype="float64")
        return np.column_stack((1.0 - positive, positive))


class ConstantMulticlass:
    """Multiclass predictor for a partition lacking enough classes."""

    def __init__(self, values: Sequence[float]) -> None:
        probabilities = np.asarray(values, dtype="float64")
        total = float(probabilities.sum())
        if total <= 0.0:
            probabilities = np.full(len(probabilities), 1.0 / len(probabilities), dtype="float64")
        else:
            probabilities = probabilities / total
        self.values = probabilities

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        """Return the same calibrated-base distribution for every row."""
        return np.tile(self.values, (len(values), 1))


def calibrator(values: np.ndarray, labels: np.ndarray, fallback: float) -> ProbabilityCalibrator:
    """Fit isotonic calibration, falling back safely for an empty fold."""
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        value = float(labels.mean()) if len(labels) else float(fallback)
        return ProbabilityCalibrator(ConstantCalibrator(value))
    fitted = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    fitted.fit(np.asarray(values, dtype="float64"), np.asarray(labels, dtype="float64"))
    return ProbabilityCalibrator(fitted)


def calibration_guard(
    predictions: np.ndarray,
    labels: np.ndarray,
    params: Params,
) -> None:
    """Enforce the validation base-rate guard from the engineering rules."""
    if len(labels) == 0:
        return
    mean = float(np.asarray(predictions, dtype="float64").mean())
    rate = float(np.asarray(labels, dtype="float64").mean())
    error = abs(mean - rate) if rate <= 0.0 else abs(mean - rate) / rate
    tolerance = float(params["models.calib.mean_tolerance"])
    if error >= tolerance:
        raise InvariantError(f"calibrated validation mean is outside tolerance: {error}")


def ece(predictions: np.ndarray, labels: np.ndarray, params: Params = P) -> float:
    """Compute expected calibration error over configured probability bins."""
    if len(labels) == 0:
        return 0.0
    bins = int(params["models.calib.ece_bins"])
    edges = np.linspace(0.0, 1.0, bins + 1)
    values = np.asarray(predictions, dtype="float64")
    actual = np.asarray(labels, dtype="float64")
    total = float(len(actual))
    result = 0.0
    for position in range(bins):
        if position == bins - 1:
            rows = (values >= edges[position]) & (values <= edges[position + 1])
        else:
            rows = (values >= edges[position]) & (values < edges[position + 1])
        if bool(rows.any()):
            result += float(rows.sum()) / total * abs(float(values[rows].mean()) - float(actual[rows].mean()))
    return float(result)


def reliability_table(predictions: np.ndarray, labels: np.ndarray, params: Params = P) -> list[dict[str, float]]:
    """Return a compact reliability table."""
    bins = int(params["models.calib.ece_bins"])
    edges = np.linspace(0.0, 1.0, bins + 1)
    values = np.asarray(predictions, dtype="float64")
    actual = np.asarray(labels, dtype="float64")
    result: list[dict[str, float]] = []
    for position in range(bins):
        if position == bins - 1:
            rows = (values >= edges[position]) & (values <= edges[position + 1])
        else:
            rows = (values >= edges[position]) & (values < edges[position + 1])
        if bool(rows.any()):
            result.append({
                "bin": float(position),
                "count": float(rows.sum()),
                "mean_prediction": float(values[rows].mean()),
                "event_rate": float(actual[rows].mean()),
            })
    return result


def numeric_prediction(model: object, values: np.ndarray) -> np.ndarray:
    """Get positive-class predictions from either LightGBM or a fallback."""
    if len(values) == 0:
        return np.asarray([], dtype="float64")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        probabilities = np.asarray(model.predict_proba(values), dtype="float64")
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise InvariantError("binary model returned an invalid probability matrix")
    return np.clip(probabilities[:, 1], 0.0, 1.0)


def make_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    params: Params = P,
) -> dict[str, Any]:
    """Return calibration and classification diagnostics without hidden inputs."""
    from sklearn.metrics import average_precision_score, brier_score_loss

    if len(labels) == 0:
        return {"pr_auc": 0.0, "brier": 0.0, "ece": 0.0, "reliability": []}
    auc = float(average_precision_score(labels, predictions)) if len(np.unique(labels)) > 1 else 0.0
    return {
        "pr_auc": auc,
        "brier": float(brier_score_loss(labels, predictions)),
        "ece": ece(predictions, labels, params),
        "reliability": reliability_table(predictions, labels, params),
    }
