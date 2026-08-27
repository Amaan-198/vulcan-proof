"""Reproducibility checks for Phase 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vulcan_proof.models.stage_a import StageAModel
from vulcan_proof.params import P


def test_lgbm_deterministic() -> None:
    root = Path(__file__).resolve().parents[1] / "outputs" / "phase2" / "smoke" / "kappa_0p6" / "seed_1"
    observed_path = root / "observed_orders.parquet"
    outcome_path = root / "outcome_arm0.parquet"
    if not observed_path.is_file() or not outcome_path.is_file():
        return
    observed = pd.read_parquet(observed_path)
    outcome = pd.read_parquet(outcome_path)
    left = StageAModel(P).fit(observed, outcome, seed=1).predict(observed)
    right = StageAModel(P).fit(observed, outcome, seed=1).predict(observed)
    assert np.array_equal(left, right)
