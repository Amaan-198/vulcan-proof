"""Evidence materialisation probability models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ._common import (
    FeatureEncoder,
    align_outcome,
    calibrator,
    feature_frame,
    fit_binary,
    make_binary_model,
    numeric_prediction,
    observed_features,
    validate_observed,
)
from .labels import eligible


class MaterialisationModel:
    """Estimate materialisation conditional on a requested evidence type."""

    def __init__(self, params: Params = P) -> None:
        self.params = params
        self.models: dict[str, object] = {}
        self.encoders: dict[str, FeatureEncoder] = {}
        self.calibrators: dict[str, object] = {}
        self.rates_by_type: dict[str, float] = {}
        self.support_by_type: dict[str, int] = {}
        self.fallback_by_type: dict[str, bool] = {}
        self.metrics: dict[str, Any] = {}

    def fit(
        self,
        observed: pd.DataFrame,
        outcome: pd.DataFrame,
        params: Params | None = None,
        seed: int = 0,
    ) -> "MaterialisationModel":
        """Fit one requested-to-materialised model per evidence type."""
        if params is not None:
            self.params = params
        validate_observed(observed)
        aligned = align_outcome(observed, outcome)
        train_rows = eligible(aligned, "train", purpose="fit").to_numpy()
        valid_rows = eligible(aligned, "validate", purpose="fit").to_numpy()
        names = tuple(self.params["evidence.order"])
        columns = observed_features(self.params)
        features = feature_frame(observed, columns)
        requested = aligned["requested_bitmask"].to_numpy(dtype="uint16")
        materialised = aligned["materialised_bitmask"].to_numpy(dtype="uint16")
        metrics: dict[str, Any] = {}
        for position, name in enumerate(names):
            bit = np.uint16(1 << position)
            train = train_rows & ((requested & bit) != 0)
            valid = valid_rows & ((requested & bit) != 0)
            support = int(train.sum())
            self.support_by_type[name] = support
            if support == 0:
                self.rates_by_type[name] = 0.0
                self.fallback_by_type[name] = True
                metrics[name] = {"support": support, "fallback": True, "rate": 0.0}
                continue
            y_train = ((materialised[train] & bit) != 0).astype("int8")
            y_valid = ((materialised[valid] & bit) != 0).astype("int8")
            rate = float(y_train.mean())
            self.rates_by_type[name] = rate
            use_fallback = support < int(self.params["models.materialisation_min_support"])
            self.fallback_by_type[name] = use_fallback
            if use_fallback:
                metrics[name] = {"support": support, "fallback": True, "rate": rate}
                continue
            encoder = FeatureEncoder(columns).fit(features.loc[train])
            self.encoders[name] = encoder
            model, _ = fit_binary(
                make_binary_model("stage_c", self.params, seed),
                encoder.transform(features.loc[train]),
                y_train,
                encoder.transform(features.loc[valid]),
                y_valid,
                rate,
                self.params,
            )
            self.models[name] = model
            raw_valid = numeric_prediction(model, encoder.transform(features.loc[valid]))
            self.calibrators[name] = calibrator(raw_valid, y_valid, rate)
            metrics[name] = {
                "support": support,
                "fallback": False,
                "rate": rate,
                "validation_rate": float(y_valid.mean()) if len(y_valid) else rate,
            }
        self.metrics = metrics
        return self

    def predict(self, observed: pd.DataFrame, evidence: str) -> np.ndarray:
        """Return materialisation probabilities for one evidence type."""
        validate_observed(observed)
        if evidence not in self.rates_by_type:
            raise InvariantError(f"unknown evidence type: {evidence!r}")
        if self.fallback_by_type.get(evidence, True):
            return np.full(len(observed), self.rates_by_type[evidence], dtype="float64")
        encoder = self.encoders[evidence]
        model = self.models[evidence]
        raw = numeric_prediction(model, encoder.transform(feature_frame(observed, encoder.columns)))
        return np.clip(
            np.asarray(self.calibrators[evidence].predict(raw), dtype="float64"),
            0.0,
            1.0,
        )

    predict_proba = predict

    def rates(self) -> dict[str, float]:
        """Return observed materialisation rates by evidence type."""
        return dict(self.rates_by_type)


def fit_materialisation(
    observed: pd.DataFrame,
    outcome: pd.DataFrame,
    params: Params = P,
    seed: int = 0,
) -> MaterialisationModel:
    """Fit and return the materialisation model."""
    return MaterialisationModel(params).fit(observed, outcome, seed=seed)


fit = fit_materialisation
Materialisation = MaterialisationModel
