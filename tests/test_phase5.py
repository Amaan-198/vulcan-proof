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
    observed_path = next((ROOT / "outputs" / "phase3" / "canonical").glob("kappa_*/seed_*/observed_orders.parquet"))
    outcome_path = observed_path.parent / "outcome_arm5.parquet"
    import pandas as pd

    observed = pd.read_parquet(observed_path, columns=["order_id", "split"])
    outcome = pd.read_parquet(outcome_path, columns=["order_id", "requested_bitmask"])
    order_id = str(observed.loc[observed["split"].astype(str).eq("test"), "order_id"].iloc[0])
    expected = int(outcome.loc[outcome["order_id"].astype(str).eq(order_id), "requested_bitmask"].iloc[0])
    assert phase5_service.plan(order_id)["plan"]["requested_bitmask"] == expected


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
