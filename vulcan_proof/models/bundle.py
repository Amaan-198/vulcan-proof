"""Phase-3 fitted model bundle implementing the optimizer protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..params import P, Params
from ..schemas import cast
from ._common import bit_columns, feature_frame, numeric_prediction, observed_features, validate_observed
from .defensibility import DefensibilityModel, SupportMasks, build_support_masks
from .materialisation import MaterialisationModel
from .prevention import PreventionModel
from .stage_a import StageAModel
from .stage_b import DISPUTE_TYPES, StageBModel
from .stage_c import StageCModel


@dataclass
class ModelBundle:
    """Fitted stages and their truth-blind planning interface."""

    stage_a: StageAModel
    stage_b: StageBModel
    stage_c: StageCModel
    materialisation: MaterialisationModel
    defensibility: DefensibilityModel
    prevention: PreventionModel
    support_mask: SupportMasks
    params: Params = P
    metrics: dict[str, Any] | None = None

    @classmethod
    def fit(
        cls,
        observed: pd.DataFrame,
        outcome: pd.DataFrame,
        params: Params = P,
        seed: int = 0,
    ) -> "ModelBundle":
        """Fit all stages from one completed observed/outcome pair."""
        stage_a = StageAModel(params).fit(observed, outcome, seed=seed)
        stage_b = StageBModel(params).fit(observed, outcome, seed=seed)
        stage_c = StageCModel(params).fit(observed, outcome, seed=seed)
        materialisation = MaterialisationModel(params).fit(observed, outcome, seed=seed)
        defensibility = DefensibilityModel(params).fit(observed, outcome, seed=seed)
        prevention = PreventionModel(params).fit(observed, outcome, seed=seed)
        return cls(
            stage_a,
            stage_b,
            stage_c,
            materialisation,
            defensibility,
            prevention,
            defensibility.support_masks,
            params,
            {
                "stage_a": stage_a.metrics,
                "stage_b": stage_b.metrics,
                "stage_c": stage_c.metrics,
                "materialisation": materialisation.metrics,
                "defensibility": defensibility.metrics,
                "prevention": prevention.metrics,
            },
        )

    def _frame(self, row: pd.Series | pd.DataFrame) -> pd.DataFrame:
        """Normalise one row for stage calls."""
        if isinstance(row, pd.DataFrame):
            result = row.reset_index(drop=True)
        else:
            result = row.to_frame().T.reset_index(drop=True)
        expected = set(self.params["features.permitted"]) | {"split", "censored", "order_day"}
        if set(result.columns) != expected:
            from ..schemas import check

            check(result, "ORDER_OBSERVED")
        return cast(result, "ORDER_OBSERVED")

    def prepare_batch(self, observed: pd.DataFrame) -> "BatchModelBundle":
        """Prepare reusable encoded matrices for vectorised optimizer calls."""
        validate_observed(observed)
        from ..schemas import check

        check(observed, "ORDER_OBSERVED")
        return BatchModelBundle(self, observed)

    def pA(self, row: pd.Series | pd.DataFrame) -> float | np.ndarray:
        """Return exposure probability."""
        result = self.stage_a.predict(self._frame(row))
        return float(result[0]) if isinstance(row, pd.Series) else result

    def pB(self, row: pd.Series | pd.DataFrame) -> dict[str, float] | np.ndarray:
        """Return type probabilities."""
        result = self.stage_b.predict(self._frame(row))
        if isinstance(row, pd.Series):
            return {name: float(result[0, position]) for position, name in enumerate(DISPUTE_TYPES)}
        return result

    def pC(self, row: pd.Series | pd.DataFrame, held_mask: int | np.ndarray) -> float | np.ndarray:
        """Return evidence-dependent contest probability."""
        result = self.stage_c.predict(self._frame(row), held_mask)
        return float(result[0]) if isinstance(row, pd.Series) else result

    def pM(self, row: pd.Series | pd.DataFrame, evidence: str) -> float | np.ndarray:
        """Return probability that requested evidence materialises."""
        result = self.materialisation.predict(self._frame(row), evidence)
        return float(result[0]) if isinstance(row, pd.Series) else result

    def pW(
        self,
        row: pd.Series | pd.DataFrame,
        dispute_type: str,
        held_mask: int | np.ndarray,
    ) -> float | np.ndarray:
        """Return support-shrunk win probability."""
        result = self.defensibility.predict(self._frame(row), dispute_type, held_mask)
        return float(result[0]) if isinstance(row, pd.Series) else result

    def pPrev(self, row: pd.Series | pd.DataFrame, evidence: str) -> float | np.ndarray:
        """Return predicted acknowledgement prevention probability."""
        result = self.prevention.predict(self._frame(row), evidence)
        return float(result[0]) if isinstance(row, pd.Series) else result

    def prevention_gain(
        self,
        row: pd.Series | pd.DataFrame,
        contest_probability: float | np.ndarray,
        win_probability: float | np.ndarray,
    ) -> float | np.ndarray:
        """Return the economics bridge used by the optimizer."""
        result = self.prevention.expected_gain(self._frame(row), contest_probability, win_probability)
        return float(result[0]) if isinstance(row, pd.Series) else result


def fit_models(
    observed: pd.DataFrame,
    outcome: pd.DataFrame,
    params: Params = P,
    seed: int = 0,
) -> ModelBundle:
    """Fit and return a complete model bundle."""
    return ModelBundle.fit(observed, outcome, params, seed)


class BatchModelBundle:
    """Fast batch view of a fitted bundle for one validated observed chunk."""

    def __init__(self, owner: ModelBundle, observed: pd.DataFrame) -> None:
        self.owner = owner
        self.observed = observed
        self.params = owner.params
        self.support_mask = owner.support_mask
        names = tuple(self.params["evidence.order"])
        base = feature_frame(observed, observed_features(self.params)).reset_index(drop=True)
        zero_bits = bit_columns(np.zeros(len(observed), dtype="uint16"), names)

        if owner.stage_c.encoder is None or owner.stage_c.model is None:
            raise InvariantError("Stage C has not been fitted")
        self.stage_c = owner.stage_c
        self.stage_c_base = owner.stage_c.encoder.transform(pd.concat([base, zero_bits], axis=1))
        self.stage_c_bit_start = len(owner.stage_c.feature_columns) - len(names)

        if owner.defensibility.encoder is None or owner.defensibility.model is None:
            raise InvariantError("defensibility model has not been fitted")
        self.defensibility = owner.defensibility
        self.stage_w_base: dict[str, np.ndarray] = {}
        disputes = {}
        for dispute_type in DISPUTE_TYPES:
            category = pd.Series(
                pd.Categorical([dispute_type] * len(observed), categories=list(DISPUTE_TYPES)),
                name="dispute_type",
            )
            frame = pd.concat([base, category, zero_bits], axis=1)
            self.stage_w_base[dispute_type] = owner.defensibility.encoder.transform(frame)
        self.stage_w_bit_start = len(owner.defensibility.feature_columns) - len(names)
        del disputes

        prevention = owner.prevention
        self.prevention = prevention
        self.prevention_features = (
            prevention.encoder.transform(base)
            if prevention.encoder is not None and prevention.model is not None
            else None
        )

    def _with_bits(self, base: np.ndarray, held_mask: int, start: int) -> np.ndarray:
        """Copy an encoded base matrix and set one held-evidence pattern."""
        result = base.copy()
        for position, name in enumerate(self.params["evidence.order"]):
            result[:, start + position] = 1.0 if int(held_mask) & int(1 << position) else 0.0
        return result

    def pC(self, frame: pd.DataFrame, held_mask: int) -> np.ndarray:
        """Return Stage-C predictions for one held pattern."""
        if frame is not self.observed:
            raise InvariantError("batch model frame does not match prepared chunk")
        matrix = self._with_bits(self.stage_c_base, held_mask, self.stage_c_bit_start)
        raw = numeric_prediction(self.stage_c.model, matrix)
        names = tuple(self.params["evidence.order"])
        adjustment = self.stage_c._plan_adjustment(
            np.full(len(frame), int(held_mask), dtype="uint16"), names
        )
        return self.stage_c._calibrate(np.clip(raw + adjustment, 0.0, 1.0))

    def pW(self, frame: pd.DataFrame, dispute_type: str, held_mask: int) -> np.ndarray:
        """Return support-shrunk defensibility predictions for one pattern."""
        if frame is not self.observed:
            raise InvariantError("batch model frame does not match prepared chunk")
        if dispute_type not in DISPUTE_TYPES:
            raise InvariantError(f"unknown dispute type: {dispute_type!r}")
        matrix = self._with_bits(
            self.stage_w_base[dispute_type], held_mask, self.stage_w_bit_start
        )
        full = self.defensibility._calibrate(
            numeric_prediction(self.defensibility.model, matrix)
        )
        base = self._base_win[dispute_type]
        names = tuple(self.params["evidence.order"])
        main = self.defensibility.main_effects.get((dispute_type, 0), 0.0)
        for position, name in enumerate(names):
            if int(held_mask) & int(1 << position):
                main += self.defensibility.main_effects.get((dispute_type, int(1 << position)), 0.0)
        confidence = self.support_mask.bit_confidence(dispute_type, int(held_mask))
        result = confidence * full + (1.0 - confidence) * np.clip(base + main - self.defensibility.main_effects.get((dispute_type, 0), 0.0), 0.0, 1.0)
        return np.clip(result, 0.0, 1.0)

    @property
    def _base_win(self) -> dict[str, np.ndarray]:
        """Return cached calibrated no-evidence predictions by dispute type."""
        if not hasattr(self, "_base_win_cache"):
            cache: dict[str, np.ndarray] = {}
            for dispute_type, matrix in self.stage_w_base.items():
                cache[dispute_type] = self.defensibility._calibrate(
                    numeric_prediction(self.defensibility.model, matrix)
                )
            self._base_win_cache = cache
        return self._base_win_cache

    def pPrev(self, frame: pd.DataFrame, evidence: str) -> np.ndarray:
        """Return acknowledgement prevention predictions."""
        if frame is not self.observed:
            raise InvariantError("batch model frame does not match prepared chunk")
        if evidence not in {"ack", "vack"}:
            return np.zeros(len(frame), dtype="float64")
        if self.prevention_features is None:
            return np.full(len(frame), self.prevention.rate, dtype="float64")
        raw = numeric_prediction(self.prevention.model, self.prevention_features)
        return np.clip(self.prevention._calibrate(raw), 0.0, 1.0)

    def prevention_gain(
        self,
        frame: pd.DataFrame,
        contest_probability: np.ndarray,
        win_probability: np.ndarray,
    ) -> np.ndarray:
        """Return vectorised prevention economics for the prepared chunk."""
        if frame is not self.observed:
            raise InvariantError("batch model frame does not match prepared chunk")
        return self.prevention.expected_gain(frame, contest_probability, win_probability)
