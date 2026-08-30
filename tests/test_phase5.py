"""Phase-5 artefact and API checks."""

from __future__ import annotations

import json
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ready() -> bool:
    return (ROOT / "outputs" / "phase3" / "canonical").is_dir()


@pytest.fixture(scope="module")
def phase5_service():
    if not _ready():
        pytest.skip("Phase-3 canonical artefacts are required")
    from vulcan_proof.api.service import Phase5Service

    return Phase5Service(root=ROOT)


def test_phase4_deferred_status(phase5_service) -> None:
    payload = phase5_service.kappa_report()
    if not (ROOT / "outputs" / "phase4" / "kappa_star.json").is_file():
        assert payload["production_sweep"] == "deferred"
        assert payload["production_results_available"] is False


def test_demo_script_contains_eight_beats(phase5_service) -> None:
    payload = phase5_service.demo_script()
    assert [beat["beat"] for beat in payload["beats"]] == list(range(1, 9))


def test_api_plan_matches_arm5_artifact(phase5_service) -> None:
    observed_path = phase5_service.phase3_observed_path
    outcome_path = phase5_service.phase3_outcome_path
    import pandas as pd

    observed = pd.read_parquet(observed_path, columns=["order_id", "split"])
    outcome = pd.read_parquet(outcome_path, columns=["order_id", "requested_bitmask"])
    order_id = str(observed.loc[observed["split"].astype(str).eq("test"), "order_id"].iloc[0])
    expected = int(outcome.loc[outcome["order_id"].astype(str).eq(order_id), "requested_bitmask"].iloc[0])
    assert phase5_service.plan(order_id)["plan"]["requested_bitmask"] == expected


def test_order_picker_does_not_hide_non_empty_plans(phase5_service) -> None:
    """The first picker page must expose stored evidence-bearing orders."""
    visible = phase5_service.orders(category="Electronics", limit=36)["orders"]
    assert any(phase5_service._stored_mask(item["order_id"]) != 0 for item in visible)


def test_order_picker_can_focus_on_plan_examples(phase5_service) -> None:
    """The demo-only view returns only stored evidence-bearing orders."""
    payload = phase5_service.orders(category="Electronics", limit=36, plans_only=True)
    assert payload["plans_only"] is True
    assert payload["orders"]
    assert all(item["has_plan"] for item in payload["orders"])


def test_order_picker_can_focus_on_package_ready_examples(phase5_service) -> None:
    """Package-ready filtering returns only opened disputes with captured evidence."""
    payload = phase5_service.orders(
        category="Electronics",
        limit=36,
        plans_only=True,
        package_ready_only=True,
    )
    assert payload["package_ready_only"] is True
    assert payload["orders"]
    assert all(item["has_plan"] and item["package_available"] for item in payload["orders"])


def test_order_picker_prioritizes_package_ready_examples(phase5_service) -> None:
    """The compact demo page keeps package-ready rows ahead of plan-only rows."""
    for category in ("Electronics", "Jewellery", "Apparel", "Home", "FMCG"):
        rows = phase5_service.orders(category=category, limit=5, plans_only=True)["orders"]
        ready_positions = [index for index, row in enumerate(rows) if row["package_available"]]
        assert ready_positions == list(range(len(ready_positions)))


def test_plan_reports_when_package_is_unavailable(phase5_service) -> None:
    """A selected plan can exist without an opened dispute package."""
    outcome = phase5_service.arm5_outcome
    test_ids = set(phase5_service._test_rows()["order_id"].astype(str))
    plan_only = outcome.loc[
        outcome["order_id"].astype(str).isin(test_ids)
        &
        outcome["requested_bitmask"].ne(0)
        & (~outcome["dispute_opened"].astype(bool) | outcome["materialised_bitmask"].eq(0))
    ].iloc[0]
    payload = phase5_service.plan(str(plan_only["order_id"]))
    assert payload["plan"]["requested_bitmask"] != 0
    assert payload["package_available"] is False


def test_llm_explanation_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VP_EXPLAIN_LLM", raising=False)
    from vulcan_proof.api.main import explain
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as caught:
        explain({"plan": {"evidence": []}})
    assert caught.value.status_code == 404


def test_phase5_report_and_script_are_present() -> None:
    script_path = ROOT / "outputs" / "phase5" / "demo_script.json"
    report_path = ROOT / "outputs" / "phase5_REPORT.md"
    if not script_path.is_file() or not report_path.is_file():
        pytest.skip("run_phase5 has not generated the handoff artefacts")
    payload = json.loads(script_path.read_text(encoding="utf-8"))
    assert len(payload["beats"]) == 8
    assert "Demo mode" in report_path.read_text(encoding="utf-8")
