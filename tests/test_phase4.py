"""Phase-4 sweep, guard, reproducibility, and chart checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vulcan_proof.errors import InvariantError, LeakError
from vulcan_proof.params import P
from vulcan_proof.sweep.charts import chart_functions, figure_footer
from vulcan_proof.sweep.common import (
    apply_overrides,
    lhs_design,
    require_min_seeds,
)
from vulcan_proof.sweep.kappa import kappa_star, kappa_zero_guard
from vulcan_proof.sweep.lhs import run_lhs_sweep
from vulcan_proof.sweep.oat import run_oat_sweep


def test_kappa_star_monotone_logic() -> None:
    """The first significant grid point must remain significant thereafter."""
    found = kappa_star(
        [
            {"kappa": 0.0, "ci_low": -1.0, "ci_high": 1.0},
            {"kappa": 0.6, "ci_low": 0.1, "ci_high": 1.0},
            {"kappa": 1.0, "ci_low": 0.2, "ci_high": 1.1},
        ]
    )
    assert found["kappa_star"] == pytest.approx(0.6)
    assert found["reason"] == "found"

    not_found = kappa_star(
        [
            {"kappa": 0.0, "ci_low": -1.0, "ci_high": 1.0},
            {"kappa": 0.6, "ci_low": -0.1, "ci_high": 1.0},
        ]
    )
    assert not_found["kappa_star"] is None
    assert not_found["reason"] == "not_found"

    non_monotone = kappa_star(
        [
            {"kappa": 0.0, "ci_low": -1.0, "ci_high": 1.0},
            {"kappa": 0.6, "ci_low": 0.1, "ci_high": 1.0},
            {"kappa": 1.0, "ci_low": -0.1, "ci_high": 1.0},
        ]
    )
    assert non_monotone["kappa_star"] is None
    assert non_monotone["reason"] == "non_monotone"


def test_kappa_zero_guard_fires() -> None:
    """The κ=0 guard rejects a gain larger than the configured leak budget."""
    with pytest.raises(LeakError):
        kappa_zero_guard(1.0, 1.0, P)
    kappa_zero_guard(0.0, 1.0, P)
    kappa_zero_guard(1.0, 1.0, P, gain_ci_low=-1.0)
    with pytest.raises(LeakError):
        kappa_zero_guard(1.0, 1.0, P, gain_ci_low=0.5)


def test_min_seeds_enforced() -> None:
    """Paired sweep summaries enforce the configured seed minimum."""
    minimum = int(P["report.min_seeds"])
    with pytest.raises(InvariantError):
        require_min_seeds(range(1, minimum), P)
    require_min_seeds(range(1, minimum + 1), P)


def test_lhs_reproducible() -> None:
    """The same configured seed produces the same LHS design matrix."""
    paths = ("uplift_true.sweep_multiplier", "econ.hourly_rate")
    left = lhs_design(paths, P, points=5)
    right = lhs_design(paths, P, points=5)
    assert np.array_equal(left, right)


def test_rank_zero_disables_oat_and_lhs(tmp_path: Path) -> None:
    """A zero max rank writes explicit empty artifacts without running points."""
    params = apply_overrides(P, {"sweep.oat_max_rank": 0, "sweep.lhs_max_rank": 0})
    seeds = range(1, int(params["report.min_seeds"]) + 1)
    oat = run_oat_sweep(params, tmp_path, seeds=seeds)
    lhs = run_lhs_sweep(params, tmp_path, seeds=seeds)
    assert oat["disabled"] is True and oat["rows"] == [] and oat["points"] == []
    assert lhs["disabled"] is True and lhs["rows"] == [] and lhs["points"] == []
    assert json.loads((tmp_path / "oat.json").read_text(encoding="utf-8"))["disabled"] is True
    assert json.loads((tmp_path / "lhs.json").read_text(encoding="utf-8"))["disabled"] is True


def test_point_theta_isolated(tmp_path: Path) -> None:
    """Point overrides never mutate the repository parameter file or its copy."""
    before = hashlib.sha256(P.path.read_bytes()).hexdigest()
    central = apply_overrides(P, {"uplift_true.sweep_multiplier": 1.2})
    assert central["uplift_true.sweep_multiplier"] == pytest.approx(1.2)
    assert hashlib.sha256(P.path.read_bytes()).hexdigest() == before
    theta = tmp_path / "theta.json"
    theta.write_text(json.dumps({"theta": 1, "gamma": {"Electronics": 1}}), encoding="utf-8")
    assert theta.read_bytes() != P.path.read_bytes()


def test_footer_on_every_chart() -> None:
    """Every chart factory adds the configured simulator footer to its figure."""
    for function in chart_functions():
        figure = function({}, P)
        assert figure_footer(figure) == str(P["report.simulator_footer"])
        plt.close(figure)
