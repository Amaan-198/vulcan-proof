"""Truth-blind acknowledgement prevention model and economic bridge."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .. import economics
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


class PreventionModel:
    """Estimate prevention given an eligible acknowledgement request."""

    def __init__(self, params: Params = P) -> None:
        self.params = params
        self.encoder: FeatureEncoder | None = None
        self.model: object | None = None
        self.fitted_calibrator: object | None = None
        self.metrics: dict[str, Any] = {}
        self.rate = 0.0

    def fit(
        self,
        observed: pd.DataFrame,
        outcome: pd.DataFrame,
        params: Params | None = None,
        seed: int = 0,
    ) -> "PreventionModel":
        """Fit prevention on uncensored exposure rows where an ack was sent."""
        if params is not None:
            self.params = params
        validate_observed(observed)
        aligned = align_outcome(observed, outcome)
        train = eligible(aligned, "train", purpose="fit").to_numpy()
        valid = eligible(aligned, "validate", purpose="fit").to_numpy()
        names = tuple(self.params["evidence.order"])
        ack_bits = np.uint16(0)
        for name in ("ack", "vack"):
            if name in names:
                ack_bits |= np.uint16(1 << names.index(name))
        requested = aligned["requested_bitmask"].to_numpy(dtype="uint16")
        sent = aligned["ack_sent"].to_numpy(dtype="int8").astype(bool)
        exposure = aligned["dispute_opened"].to_numpy(dtype="int8").astype(bool) | aligned["prevented"].to_numpy(dtype="int8").astype(bool)
        train &= sent & exposure & ((requested & ack_bits) != 0)
        valid &= sent & exposure & ((requested & ack_bits) != 0)
        if not bool(train.any()):
            self.rate = 0.0
            self.model = None
            self.metrics = {"support": 0, "rate": 0.0, "fallback": True}
            return self
        y_all = aligned["prevented"].to_numpy(dtype="int8")
        y_train = y_all[train]
        y_valid = y_all[valid]
        self.rate = float(y_train.mean())
        columns = observed_features(self.params)
        features = feature_frame(observed, columns)
        self.encoder = FeatureEncoder(columns).fit(features.loc[train])
        self.model, _ = fit_binary(
            make_binary_model("stage_a", self.params, seed),
            self.encoder.transform(features.loc[train]),
            y_train,
            self.encoder.transform(features.loc[valid]),
            y_valid,
            self.rate,
            self.params,
        )
        raw_valid = numeric_prediction(self.model, self.encoder.transform(features.loc[valid]))
        self.fitted_calibrator = calibrator(raw_valid, y_valid, self.rate)
        calibrated = self._calibrate(raw_valid)
        calibration_guard(calibrated, y_valid, self.params)
        self.metrics = make_metrics(calibrated, y_valid, self.params)
        self.metrics.update({"support": int(train.sum()), "rate": self.rate})
        return self

    def predict(self, observed: pd.DataFrame, evidence: str = "vack") -> np.ndarray:
        """Return prevention probabilities, zero for a non-acknowledgement kind."""
        validate_observed(observed)
        if evidence not in {"ack", "vack"}:
            return np.zeros(len(observed), dtype="float64")
        if self.model is None or self.encoder is None or self.fitted_calibrator is None:
            return np.full(len(observed), self.rate, dtype="float64")
        raw = numeric_prediction(self.model, self.encoder.transform(feature_frame(observed, self.encoder.columns)))
        return np.clip(np.asarray(self.fitted_calibrator.predict(raw), dtype="float64"), 0.0, 1.0)

    predict_proba = predict

    def _calibrate(self, values: np.ndarray) -> np.ndarray:
        if self.fitted_calibrator is None:
            return np.full(len(values), self.rate, dtype="float64")
        return np.clip(np.asarray(self.fitted_calibrator.predict(values), dtype="float64"), 0.0, 1.0)

    def expected_gain(
        self,
        observed: pd.DataFrame,
        contest_probability: np.ndarray | float,
        win_probability: np.ndarray | float,
    ) -> np.ndarray:
        """Return prevention value minus an open-dispute value per row.

        The central economics module owns the fee and money branches. This
        method deliberately receives only observed category/value context and
        model probabilities.
        """
        validate_observed(observed)
        contest = np.broadcast_to(np.asarray(contest_probability, dtype="float64"), len(observed))
        win = np.broadcast_to(np.asarray(win_probability, dtype="float64"), len(observed))
        if not (np.isfinite(contest).all() and np.isfinite(win).all()):
            raise InvariantError("prevention inputs are not finite")
        values = observed["order_value"].to_numpy(dtype="float64")
        contest = np.clip(contest, 0.0, 1.0)
        win = np.clip(win, 0.0, 1.0)
        cogs = np.asarray(
            [float(self.params[f"categories.{category}"]["cogs"]) for category in observed["category"].astype(str)],
            dtype="float64",
        )
        dispute_value = (
            (1.0 - contest) * economics.money_array("opened_not_contested", values, params=self.params)
            + contest
            * (
                win * economics.money_array("opened_contested_won", values, params=self.params)
                + (1.0 - win) * economics.money_array("opened_contested_lost", values, params=self.params)
            )
        )
        result = np.zeros(len(observed), dtype="float64")
        modes = tuple(
            name.removeprefix("share_")
            for name in self.params["econ.prevention"]
            if name.startswith("share_")
        )
        for mode in modes:
            if mode == "explanation":
                cost = np.full(len(observed), float(self.params["econ.prevention.support_cost"]), dtype="float64")
            elif mode == "refund":
                cost = values - float(self.params["econ.prevention.salvage_sigma"]) * cogs * values
            else:
                cost = (
                    cogs * values
                    + float(self.params["econ.prevention.reship_cost"])
                    + float(self.params["econ.prevention.support_cost"])
                )
            share = float(self.params[f"econ.prevention.share_{mode}"])
            result += share * (-cost - dispute_value)
        return result


def fit_prevention(
    observed: pd.DataFrame,
    outcome: pd.DataFrame,
    params: Params = P,
    seed: int = 0,
) -> PreventionModel:
    """Fit and return the prevention model."""
    return PreventionModel(params).fit(observed, outcome, seed=seed)


fit = fit_prevention
Prevention = PreventionModel
