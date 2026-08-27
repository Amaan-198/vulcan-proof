"""Phase-1 simulator tests and guards."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from vulcan_proof.errors import LeakError
from vulcan_proof.manifest import start_run, write_artifact
from vulcan_proof.params import P
from vulcan_proof.schemas import ORDER_HIDDEN, ORDER_OBSERVED, check
from vulcan_proof.sim.calibrate import calibrate_funnel
from vulcan_proof.sim.merchants import build_merchants
from vulcan_proof.sim.generator import generate_world


ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def calibration_file() -> pathlib.Path:
    path = ROOT / "outputs" / "theta.json"
    if not path.exists():
        calibrate_funnel(
            seed=int(P["run.master_seed"]),
            n_orders=int(P["run.n_orders_sweep"]),
            params_path=P.path,
            output_path=path,
        )
    return path


@pytest.fixture(scope="module")
def smoke_world(calibration_file: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    del calibration_file
    return generate_world(
        kappa=0.6,
        seed=1,
        n_orders=int(P["run.n_orders_smoke"]),
        shift_enabled=False,
        params_path=P.path,
    )


@pytest.fixture(scope="module")
def large_world(calibration_file: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    del calibration_file
    return generate_world(
        kappa=0.6,
        seed=1,
        n_orders=200000,
        shift_enabled=False,
        params_path=P.path,
    )


def test_smoke_generates(smoke_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden = smoke_world
    check(observed, "ORDER_OBSERVED")
    check(hidden, "ORDER_HIDDEN")
    assert len(observed) == int(P["run.n_orders_smoke"])
    assert observed["order_id"].is_unique
    assert hidden["order_id"].is_unique


def test_no_hidden_in_observed(smoke_world: tuple[pd.DataFrame, pd.DataFrame], tmp_path: pathlib.Path) -> None:
    observed, _ = smoke_world
    assert not any(str(column).startswith("hidden_") for column in observed.columns)
    context = start_run("phase1_test", P, allow_dirty=True, run_dir=tmp_path / "run")
    with pytest.raises(LeakError):
        write_artifact(context, pd.DataFrame({"hidden_x": [1]}), "observed_bad")


def test_forbidden_absent(smoke_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, _ = smoke_world
    assert not set(observed.columns).intersection(P["features.forbidden"])


def test_reproducible(calibration_file: pathlib.Path, tmp_path: pathlib.Path) -> None:
    del calibration_file
    left = generate_world(kappa=0.6, seed=1, n_orders=20000, shift_enabled=False, params_path=P.path)
    right = generate_world(kappa=0.6, seed=1, n_orders=20000, shift_enabled=False, params_path=P.path)
    for index, (first, second) in enumerate(zip(left, right, strict=True)):
        first_path = tmp_path / f"first_{index}.parquet"
        second_path = tmp_path / f"second_{index}.parquet"
        first.to_parquet(first_path, index=False)
        second.to_parquet(second_path, index=False)
        assert first_path.read_bytes() == second_path.read_bytes()


def test_risk_mult_mean_one(calibration_file: pathlib.Path) -> None:
    del calibration_file
    for kappa in (0.0, 0.6, 1.0):
        _, hidden = generate_world(kappa=kappa, seed=1, n_orders=20000, shift_enabled=False, params_path=P.path)
        assert abs(float(hidden["hidden_risk_mult"].mean()) - 1.0) < 0.02


def test_kappa_zero_no_signal(calibration_file: pathlib.Path) -> None:
    del calibration_file
    _, hidden = generate_world(kappa=0.0, seed=1, n_orders=200000, shift_enabled=False, params_path=P.path)
    rho = float(spearmanr(hidden["hidden_z_risk"], hidden["hidden_dispute_potential"]).statistic)
    assert abs(rho) < 0.01


def test_truth_shares(large_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    _, hidden = large_world
    rates = hidden["hidden_truth"].astype(str).value_counts(normalize=True)
    for name in P["sim.truth_base_rate"]:
        assert float(rates[name]) == pytest.approx(
            float(P[f"sim.truth_base_rate.{name}"]), rel=0.15
        )


def test_category_rates_within_tolerance(large_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden = large_world
    joined = observed[["category"]].copy()
    joined["potential"] = hidden["hidden_dispute_potential"].to_numpy()
    target_multiplier = float(P["categories.rate_sweep_multiplier"])
    for category in P["categories.order"]:
        rate = float(joined.loc[joined["category"].astype(str).eq(category), "potential"].mean())
        target = float(P[f"categories.{category}"]["target_rate"]) * target_multiplier
        assert rate == pytest.approx(target, rel=float(P["categories.category_rate_tolerance"]))


def test_genuine_share_and_phi(large_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden = large_world
    potential = hidden["hidden_dispute_potential"].astype(bool).to_numpy()
    truth = hidden["hidden_truth"].astype(str).to_numpy()
    genuine = potential & (truth != "delivered_correct")
    share = float(genuine.sum() / potential.sum())
    assert share == pytest.approx(float(P["sim.genuine_share_target"]), abs=0.03)
    assert 1.0 - share == pytest.approx(float(P["reference.phi"]), abs=0.02)
    assert len(observed) == len(hidden)


def test_calibration_deterministic(tmp_path: pathlib.Path) -> None:
    first = calibrate_funnel(
        seed=int(P["run.master_seed"]),
        n_orders=20000,
        params_path=P.path,
        output_path=tmp_path / "first.json",
    )
    second = calibrate_funnel(
        seed=int(P["run.master_seed"]),
        n_orders=20000,
        params_path=P.path,
        output_path=tmp_path / "second.json",
    )
    assert first == second
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()


def test_hist_columns_nan(smoke_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, _ = smoke_world
    for column in ("merchant_dispute_rate_hist", "merchant_contest_rate_hist", "merchant_compliance_hist"):
        assert observed[column].isna().all()


def test_every_evidence_has_support(large_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden = large_world
    frame = pd.DataFrame(
        {
            "archetype": hidden["hidden_archetype"].astype(str),
            "bitmask": hidden["hidden_requested_bitmask"].to_numpy(),
        }
    )
    evidence = list(P["evidence.order"])
    for item in evidence:
        bit = 1 << evidence.index(item)
        shares = frame.groupby("archetype")["bitmask"].apply(lambda values: float(((values.to_numpy() & bit) != 0).mean()))
        assert bool((shares >= 0.01).any())
    otp_bit = 1 << evidence.index("otp")
    signature_bit = 1 << evidence.index("signature")
    assert int(
        (
            ((frame["bitmask"] & otp_bit) != 0)
            & ((frame["bitmask"] & signature_bit) != 0)
        ).sum()
    ) >= 200
    assert len(observed) == len(frame)


def test_archetype_blocks() -> None:
    merchants = build_merchants(P, seed=1)
    shares = merchants["hidden_archetype"].astype(str).value_counts(normalize=True)
    for name in P["archetypes.order"]:
        assert float(shares[name]) == pytest.approx(float(P[f"archetypes.{name}"]["share"]), abs=1 / len(merchants))
    means = merchants.groupby("hidden_archetype", observed=True)["hidden_quality"].mean()
    ordered = sorted(P["archetypes.order"], key=lambda name: int(P[f"archetypes.{name}"]["quality_rank"]))
    assert all(means[left] > means[right] for left, right in zip(ordered, ordered[1:], strict=False))


def test_headline_misdelivered_zero() -> None:
    assert float(P["uplift_true.on_misdelivered_otp"]) == 0.0


def test_censoring_excluded_not_negative(smoke_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden = smoke_world
    assert not set(observed.columns).intersection({"label", "downstream_label"})
    for split in ("train", "validate"):
        group = observed.loc[observed["split"].astype(str).eq(split)]
        assert float(group["censored"].mean()) <= float(P["sim.max_censor_frac"])
    assert len(hidden) == len(observed)


def test_latency_bounds(calibration_file: pathlib.Path) -> None:
    del calibration_file
    observed, hidden = generate_world(kappa=0.6, seed=1, n_orders=200000, shift_enabled=False, params_path=P.path)
    potential = hidden["hidden_dispute_potential"].astype(bool).to_numpy()
    open_day = hidden.loc[potential, "hidden_dispute_open_day"].astype("int32").to_numpy()
    order_day = observed.loc[potential, "order_day"].to_numpy()
    resolution = hidden.loc[potential, "hidden_resolution_day"].astype("int32").to_numpy()
    offset = open_day - order_day
    assert float(offset.min()) >= float(P["sim.latency.expected_delivery_days"])
    assert float(offset.max()) <= float(P["sim.latency.expected_delivery_days"] + P["sim.latency.dispute_max_days"])
    assert float((resolution - open_day).min()) >= float(P["sim.latency.response_days"])


def test_declared_dtypes(smoke_world: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    observed, hidden = smoke_world
    assert observed.memory_usage(deep=True).sum() / len(observed) <= 120
    assert hidden.memory_usage(deep=True).sum() / len(hidden) <= 80
    assert all(column in ORDER_OBSERVED for column in observed.columns)
    assert all(column in ORDER_HIDDEN for column in hidden.columns)
