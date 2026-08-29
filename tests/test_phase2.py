"""Phase-2 resolver, history, and arm guard tests."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from vulcan_proof.arms.arm1 import plan as arm1
from vulcan_proof.arms.arm2 import plan as arm2
from vulcan_proof.arms.arm3 import plan as arm3
from vulcan_proof.economics import money, prevention_cost
from vulcan_proof.errors import InvariantError, SchemaError
from vulcan_proof.params import P
from vulcan_proof.report.paired import paired_report
from vulcan_proof.schemas import check
from vulcan_proof.sim.arm0_history import plan as arm0
from vulcan_proof.sim.generator import generate_world
from vulcan_proof.sim.history import add_history_features
from vulcan_proof.sim.phase2 import _score_callback
from vulcan_proof.sim.resolve import resolve
from vulcan_proof.arms.arm4 import tune
from vulcan_proof.arms.base import make_plan


@pytest.fixture(scope="module")
def world() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed, hidden = generate_world(0.6, 1, 20000, False, P.path)
    historical = resolve(arm0(hidden, observed, P), hidden, observed, seed=1, params=P, arm_id="arm0")
    completed = add_history_features(observed, historical, P)
    return completed, hidden, historical


def test_arms_only_see_observed(world: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden, _ = world
    bad = observed.copy()
    bad["hidden_x"] = 1
    for policy in (arm1, arm2, arm3):
        with pytest.raises(SchemaError):
            policy(bad, P)
    assert hidden["hidden_truth"].notna().all()


def test_common_random_numbers(world: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden, _ = world
    policy = arm2(observed, P)
    left = resolve(policy, hidden, observed, seed=1, params=P, arm_id="left")
    right = resolve(policy, hidden, observed, seed=1, params=P, arm_id="right")
    for column in ("complied", "response", "materialised_bitmask", "contested", "won"):
        assert left[column].astype(str).equals(right[column].astype(str))


def test_never_handed_off_no_handoff(world: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden, _ = world
    forced = hidden.copy()
    forced["hidden_truth"] = pd.Categorical(
        ["never_handed_off"] * len(forced),
        categories=list(P["sim.truth_base_rate"] | {"delivered_correct": 0}),
    )
    forced["hidden_dispute_potential"] = np.ones(len(forced), dtype="int8")
    forced["hidden_dispute_type"] = pd.Categorical(["NR"] * len(forced), categories=["NR", "NAD", "EB"])
    outcome = resolve(arm2(observed, P), forced, observed, seed=1, params=P)
    handoff = np.uint16(
        (1 << P["evidence.order"].index("geotag"))
        | (1 << P["evidence.order"].index("otp"))
        | (1 << P["evidence.order"].index("signature"))
    )
    assert not bool((outcome["materialised_bitmask"].to_numpy(dtype="uint16") & handoff).any())


def test_cash_on_compliance(world: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden, _ = world
    forced = hidden.copy()
    forced["hidden_compliance"] = np.zeros(len(forced), dtype="float32")
    packing_bit = np.uint16(1 << P["evidence.order"].index("packing"))
    packing_plan = make_plan(observed, np.full(len(observed), packing_bit, dtype="uint16"), "test", P)
    outcome = resolve(packing_plan, forced, observed, seed=1, params=P)
    assert float(outcome.loc[outcome["requested_bitmask"].ne(0), "cash_cost"].max()) == 0.0
    assert float(outcome["time_cost"].max()) > 0.0


def test_money_table() -> None:
    assert money("none", 45000) == 0.0
    assert money("prevented", 45000, 38500) == -38500.0
    assert money("opened_not_contested", 45000) == -45500.0
    assert money("opened_contested_won", 45000) == -500.0
    assert money("opened_contested_lost", 45000) == -45500.0
    assert prevention_cost("replacement", 45000, 0.85) == 38500.0


def test_win_lose_delta_is_order_value() -> None:
    won = money("opened_contested_won", 45000)
    lost = money("opened_contested_lost", 45000)
    assert won - lost == 45000.0 + float(P["econ.ratio_damage"])


def test_history_no_nan_after_phase2(world: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    observed, _, _ = world
    assert not observed[list(
        ("merchant_dispute_rate_hist", "merchant_contest_rate_hist", "merchant_compliance_hist")
    )].isna().any().any()
    check(observed, "ORDER_OBSERVED")


def test_history_maturity() -> None:
    observed, hidden = generate_world(0.6, 1, 2000, False, P.path)
    historical = resolve(arm0(hidden, observed, P), hidden, observed, seed=1, params=P, arm_id="arm0")
    completed = add_history_features(observed, historical, P)
    assert float(completed["merchant_dispute_rate_hist"].min()) >= 0.0


def test_paired_min_seeds(world: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden, _ = world
    outcome = resolve(arm1(observed, P), hidden, observed, seed=1, params=P, arm_id="arm1")
    minimum = int(P["report.min_seeds"])
    with pytest.raises(InvariantError):
        paired_report(
            {seed: outcome for seed in range(1, minimum)},
            {seed: outcome for seed in range(1, minimum)},
            P,
        )


def test_arm4_tuning_uses_validation_only(world: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden, _ = world
    policy = tune(observed, _score_callback(observed, hidden, 1, P), P)
    planned = policy.plan(observed)
    check(planned, "PLAN")
    assert len(policy.cell_masks) > 0


def test_defense_only_split(world: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden, _ = world
    outcome = resolve(arm2(observed, P), hidden, observed, seed=1, params=P, arm_id="arm2")
    contested = outcome["contested"].astype(bool)
    correct = contested & outcome["claim_class"].astype(str).eq("correct_fulfillment")
    merchant = contested & outcome["claim_class"].astype(str).eq("merchant_fault")
    if bool(correct.any()) and bool(merchant.any()):
        assert float(outcome.loc[merchant, "won"].mean()) <= float(outcome.loc[correct, "won"].mean())


def test_fee_scope() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    matches = {
        str(path): path.read_text(encoding="utf-8").count("econ.dispute_fee")
        for path in (root / "vulcan_proof").rglob("*.py")
        if "econ.dispute_fee" in path.read_text(encoding="utf-8")
    }
    assert list(matches) == [str(root / "vulcan_proof" / "economics.py")]
    assert "economics.money" in (root / "vulcan_proof" / "sim" / "resolve.py").read_text(encoding="utf-8")
