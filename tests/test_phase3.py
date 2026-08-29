"""Phase-3 model, support-mask, and optimizer checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vulcan_proof import ev_reference as reference
from vulcan_proof.errors import SchemaError
from vulcan_proof.models import ModelBundle, fit_models
from vulcan_proof.models.labels import eligible
from vulcan_proof.models.stage_b import StageBModel
from vulcan_proof.opt.optimizer import best_plan, ev_set
from vulcan_proof.params import P


ROOT = Path(__file__).resolve().parents[1]


def test_stage_b_multiclass_scaling_preserves_validation_margins() -> None:
    """Row normalisation must not undo Stage B's one-vs-rest calibration margins."""
    values = np.asarray(
        [
            [0.80, 0.15, 0.05],
            [0.55, 0.35, 0.10],
            [0.20, 0.65, 0.15],
            [0.10, 0.55, 0.35],
        ],
        dtype="float64",
    )
    targets = np.asarray([0.25, 0.50, 0.25], dtype="float64")
    scales = StageBModel._fit_calibration_scales(values, targets)
    adjusted = StageBModel._normalise_rows(values * scales)
    assert np.allclose(adjusted.sum(axis=1), 1.0)
    assert np.allclose(adjusted.mean(axis=0), targets, atol=1e-10)


@pytest.fixture(scope="module")
def phase2_world() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_path = ROOT / "outputs" / "phase2" / "smoke" / "kappa_0p6" / "seed_1" / "observed_orders.parquet"
    outcome_path = ROOT / "outputs" / "phase2" / "smoke" / "kappa_0p6" / "seed_1" / "outcome_arm0.parquet"
    if not observed_path.is_file() or not outcome_path.is_file():
        pytest.skip("Phase-2 smoke artifact is required for Phase-3 model tests")
    return pd.read_parquet(observed_path), pd.read_parquet(outcome_path)


class PerfectModels:
    """Reference-equivalent protocol stub for exhaustive optimizer tests."""

    support_mask: dict[tuple[str, str], bool] = {}

    def __init__(self, risk: float = 1.0) -> None:
        self.risk = risk

    def pA(self, row: pd.Series) -> float:
        category = str(row["category"])
        return float(P[f"categories.{category}"]["target_rate"]) * self.risk

    def pB(self, row: pd.Series) -> dict[str, float]:
        return dict(P[f"categories.{str(row['category'])}"]["mix"])

    def pC(self, row: pd.Series, held_mask: int) -> float:
        del row, held_mask
        return reference.PC_POP

    def pM(self, row: pd.Series, evidence: str) -> float:
        del row
        item = reference.EV_TYPES[evidence]
        return item["mat"] if evidence in reference.SYSTEM_SENT else reference.COMPLIANCE_POP * item["mat"]

    def pW(self, row: pd.Series, dispute_type: str, held_mask: int) -> float:
        del row
        present = [
            name
            for position, name in enumerate(reference.EV_TYPES)
            if held_mask & (1 << position)
        ]
        return reference.BASE_WIN[dispute_type] + reference.PHI * reference.set_uplift(dispute_type, present)


def _row(frame: pd.DataFrame, category: str, value: float) -> pd.DataFrame:
    result = frame.iloc[:1].copy()
    result["category"] = category
    result["order_value"] = value
    result["eligible_tier"] = "FULL"
    result["ack_optin"] = 1
    return result


def test_known_answer_perfect_models(phase2_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, _ = phase2_world
    names = tuple(P["evidence.order"])
    for category, value, risk in (
        ("Electronics", 45000.0, 1.0),
        ("Electronics", 45000.0, 2.0),
        ("Electronics", 45000.0, 4.0),
        ("Jewellery", 200000.0, 1.0),
        ("Apparel", 3500.0, 1.0),
    ):
        models = PerfectModels(risk)
        row = _row(observed, category, value)
        result = best_plan(row, models)
        expected = reference.best_subset(category, value, risk=risk)
        chosen = tuple(name for name in names if result.requested_bitmask & (1 << names.index(name)))
        assert chosen == expected[1]
        assert result.ev == pytest.approx(expected[0], abs=1e-6)
    assert "otp" in best_plan(_row(observed, "Electronics", 45000.0), PerfectModels(2.0)).reasons


def test_empty_plan_ev_zero(phase2_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, _ = phase2_world
    assert ev_set(_row(observed, "Electronics", 45000.0), 0, PerfectModels()) == 0.0


def test_stage_a_target_is_exposure(phase2_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, outcome = phase2_world
    model = fit_models(observed, outcome, P, 1)
    assert model.stage_a.target_name == "exposure"


def test_calibrated_means_and_stage_b_rows(phase2_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, outcome = phase2_world
    bundle = fit_models(observed, outcome, P, 1)
    validation = outcome["split"].astype(str).eq("validate") & outcome["censored"].eq(0)
    stage_a_rate = float(((outcome["dispute_opened"] != 0) | (outcome["prevented"] != 0))[validation].mean())
    assert float(bundle.stage_a.predict(observed[validation.to_numpy()]).mean()) == pytest.approx(stage_a_rate, rel=0.05)
    probabilities = bundle.stage_b.predict(observed)
    assert np.all(np.abs(probabilities.sum(axis=1) - 1.0) < 1e-9)


def test_stage_c_uses_plan(phase2_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, outcome = phase2_world
    bundle = fit_models(observed, outcome, P, 1)
    empty = bundle.stage_c.predict(observed, 0)
    full = bundle.stage_c.predict(observed, (1 << len(P["evidence.order"])) - 1)
    assert np.all(empty != full)


def test_no_nan_features(phase2_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, outcome = phase2_world
    bad = observed.copy()
    bad.loc[bad.index[0], "merchant_dispute_rate_hist"] = np.nan
    with pytest.raises(SchemaError):
        fit_models(bad, outcome, P, 1)


def test_no_training_on_gap_or_test(phase2_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, outcome = phase2_world
    with pytest.raises(Exception):
        eligible(outcome, "test")
    with pytest.raises(Exception):
        eligible(outcome, "gap")
