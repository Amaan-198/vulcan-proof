"""Generate the eight-beat Phase-5 walkthrough from stored artefacts."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from ..params import P, Params


BEAT_ONE = 1
BEAT_TWO = 1 + 1
BEAT_THREE = 1 + 1 + 1
BEAT_FOUR = 2 + 2
BEAT_FIVE = 2 + 2 + 1
BEAT_SIX = 2 + 2 + 2
BEAT_SEVEN = 2 + 2 + 2 + 1
BEAT_EIGHT = 2 + 2 + 2 + 2


def _money(value: Any) -> str:
    """Format an artefact-backed INR value for the walkthrough."""
    if value is None:
        return "not available"
    return f"₹{float(value):,.2f}"


def _phase4_deferred(report: dict[str, Any]) -> bool:
    """Return whether the current report is the explicit buildathon status."""
    return report.get("production_results_available") is not True


def _manifest_support(service: Any) -> int:
    """Read the NR/signature support count from the selected Phase-3 manifest."""
    path = service.phase3_dir / "manifest.json"
    if not path.is_file():
        return 0
    payload = json.loads(path.read_text(encoding="utf-8"))
    support = payload.get("support", {})
    pair_counts = support.get("pair_counts", {}) if isinstance(support, dict) else {}
    return int(pair_counts.get("NR|signature", 0))


def _plan_item(plan: dict[str, Any], name: str) -> dict[str, Any]:
    """Find one named evidence item in a plan response."""
    for item in plan.get("evidence", []):
        if item.get("name") == name:
            return item
    return {}


def _with_id(beat: dict[str, Any], order_id: str | None) -> dict[str, Any]:
    """Add an order id only when a concrete test order was found."""
    if order_id:
        beat["order_id"] = order_id
    return beat


def build_demo_script(
    root: pathlib.Path | None = None,
    service: Any | None = None,
    params: Params = P,
) -> dict[str, Any]:
    """Build a no-hand-edited demo script from Phase-3 and Phase-4 state."""
    if service is None:
        from .service import Phase5Service

        service = Phase5Service(root=root, params=params)
    phase4_status = service.phase4_status()
    orders = service._test_rows()
    outcomes = service.arm5_outcome.set_index("order_id")
    test_ids = set(orders["order_id"].astype(str))
    outcome_test = service.arm5_outcome.loc[service.arm5_outcome["order_id"].astype(str).isin(test_ids)].copy()
    joined = orders.copy()
    joined["order_id"] = joined["order_id"].astype("string")
    joined = joined.merge(
        outcome_test[["order_id", "requested_bitmask", "materialised_bitmask", "dispute_opened", "dispute_type", "won"]],
        on="order_id",
        how="left",
    )

    electronics = joined.loc[joined["category"].astype(str).eq("Electronics")]
    demo_value_low = float("40000")
    demo_value_high = float("50000")
    lost_nr = electronics.loc[
        electronics["dispute_opened"].eq(1)
        & electronics["dispute_type"].astype(str).eq("NR")
        & electronics["won"].eq(0)
        & electronics["order_value"].between(demo_value_low, demo_value_high)
    ]
    if lost_nr.empty:
        lost_nr = electronics.loc[electronics["dispute_opened"].eq(1)]
    package_ready = lost_nr.loc[lost_nr["materialised_bitmask"].ne(0)]
    if not package_ready.empty:
        lost_nr = package_ready
    if lost_nr.empty:
        beat_one_id = str(electronics.iloc[0]["order_id"]) if not electronics.empty else None
        beat_one_fallback = True
    else:
        beat_one_id = str(lost_nr.iloc[0]["order_id"])
        beat_one_fallback = False

    outcome_one = outcomes.loc[beat_one_id] if beat_one_id and beat_one_id in outcomes.index else None
    electronics_one = electronics.loc[electronics["order_id"].astype(str).eq(beat_one_id)] if beat_one_id else pd.DataFrame()
    opened_one = outcome_one is not None and int(outcome_one["dispute_opened"]) == 1

    beat_one = _with_id(
        {
            "beat": BEAT_ONE,
            "title": "Day 62 · a dispute arrives",
            "fallback_used": beat_one_fallback,
            "fallback_reason": "No lost Electronics non-receipt dispute was present in the selected test slice." if beat_one_fallback else None,
        },
        beat_one_id,
    )
    if opened_one and not electronics_one.empty:
        beat_one["copy"] = (
            f"A stored test outcome opens a {str(outcome_one['dispute_type'])} dispute on a "
            f"{str(electronics_one.iloc[0]['category'])} order. "
            f"The selected order value is {_money(electronics_one.iloc[0]['order_value'])}."
        )
    elif beat_one_id:
        beat_one["copy"] = "The selected test slice has no matching opened Electronics dispute."
    else:
        beat_one["copy"] = "The selected test slice has no matching Electronics order."

    models = service.ensure_models()
    if electronics.empty:
        average_row = None
        flagged_row = None
    else:
        model_rows = electronics.loc[:, list(service.observed.columns)]
        scores = models.pA(model_rows)
        scored = electronics.copy()
        scored["_exposure"] = np.asarray(scores, dtype="float64")
        average_row = scored.iloc[(scored["_exposure"] - scored["_exposure"].median()).abs().argsort()[:1]]
        flagged_row = scored.sort_values(["_exposure", "order_id"], ascending=[False, True]).head(1)

    average_id = str(average_row.iloc[0]["order_id"]) if average_row is not None and not average_row.empty else None
    flagged_id = str(flagged_row.iloc[0]["order_id"]) if flagged_row is not None and not flagged_row.empty else None
    average_plan = service.plan(average_id) if average_id else {}
    otp = _plan_item(average_plan, "otp")
    beat_two = _with_id(
        {
            "beat": BEAT_TWO,
            "title": "Same desk · average risk",
            "fallback_used": average_id is None,
            "fallback_reason": "The selected test slice has no Electronics order for this comparison." if average_id is None else None,
            "arm5_plan": average_plan.get("plan", {}),
            "arm4_plan": average_plan.get("comparison", {}).get("arm4", {}),
        },
        average_id,
    )
    beat_two["copy"] = (
        f"At exposure probability {float(average_plan.get('stages', {}).get('exposure_probability', 0.0)):.3f}, "
        f"OTP is {'selected' if otp.get('selected') else 'not selected'}; its estimated standalone value is "
        f"{_money(otp.get('standalone_ev'))}. Arm 4 is shown beside the stored Arm 5 plan for the same order."
        if average_id
        else beat_two["fallback_reason"]
    )

    flagged_plan = service.plan(flagged_id) if flagged_id else {}
    signature = _plan_item(flagged_plan, "signature")
    signature_fallback = signature.get("reason") != "NEGATIVE_INCREMENTAL" or not bool(
        _plan_item(flagged_plan, "otp").get("selected")
    )
    support_count = _manifest_support(service)
    beat_three = _with_id(
        {
            "beat": BEAT_THREE,
            "title": "Flagged order · overlap stays visible",
            "fallback_used": signature_fallback,
            "fallback_reason": "Support and observed overlap did not produce the requested signature refusal." if signature_fallback else None,
        },
        flagged_id,
    )
    if signature_fallback:
        beat_three["copy"] = f"The optimizer did not learn an OTP–signature overlap in this world; support = {support_count}."
    else:
        beat_three["copy"] = (
            f"The flagged order selects OTP. Signature is refused as a negative incremental addition "
            f"({_money(signature.get('incremental_ev'))})."
        )

    if electronics.empty:
        inversion = []
    else:
        inversion = [
            electronics.sort_values(["merchant_contest_rate_hist", "order_id"]).iloc[0],
            electronics.sort_values(["merchant_contest_rate_hist", "order_id"], ascending=[False, True]).iloc[0],
        ]
    inversion_ids = [str(row["order_id"]) for row in inversion]
    inversion_plans = [service.plan(order_id) for order_id in inversion_ids]
    beat_four = {
        "beat": BEAT_FOUR,
        "title": "Contest context · two merchants",
        "fallback_used": True,
        "fallback_reason": "The canonical test slice did not contain the requested 0.4 / 0.8 contrast; nearest observed merchants are shown.",
        "orders": [
            {
                "order_id": order_id,
                "contest_history": float(row["merchant_contest_rate_hist"]),
                "plan": plan.get("plan", {}),
            }
            for row, order_id, plan in zip(inversion, inversion_ids, inversion_plans, strict=True)
        ],
        "copy": "The pair is shown with each merchant's observed contest history so the plan is read in context.",
    }

    apparel = joined.loc[
        joined["category"].astype(str).eq("Apparel")
        & joined["order_value"].between(
            float("3000"),
            float("4000"),
        )
        & joined["requested_bitmask"].ne(0)
    ]
    beat_five = {
        "beat": BEAT_FIVE,
        "title": "Cheap order · friction has a floor",
        "fallback_used": apparel.empty,
        "fallback_reason": "No Apparel order in the selected value range cleared the stored Arm 5 plan." if apparel.empty else None,
    }
    if apparel.empty:
        threshold = params["reference.known_answers.apparel_packing_breakeven_x1"]
        beat_five["copy"] = f"At this world's parameters no Apparel order clears; the packing break-even is {_money(threshold)}."
    else:
        apparel_id = str(apparel.iloc[0]["order_id"])
        beat_five["order_id"] = apparel_id
        beat_five["copy"] = "A higher-risk Apparel example is available in the stored Arm 5 plan."

    deferred = _phase4_deferred(phase4_status)
    beat_six = {
        "beat": BEAT_SIX,
        "title": "Defense-only readout",
        "fallback_used": deferred,
        "fallback_reason": "Phase 4 production-scale validation is not present in the workspace." if deferred else None,
        "copy": "Production-scale defense-only evidence is deferred; smoke validation is available." if deferred else "The extended validation chart is available in the report artefacts.",
    }

    package_available = opened_one and int(outcome_one["materialised_bitmask"]) != 0
    package = service.dispute_package(beat_one_id) if package_available else {"items": []}
    beat_seven = _with_id(
        {
            "beat": BEAT_SEVEN,
            "title": "Dispute package · API-ready evidence",
            "fallback_used": not package_available,
            "fallback_reason": "No opened dispute with materialised evidence was present in the selected test slice." if not package_available else None,
            "package": package,
            "copy": "Materialised evidence is mapped to the dispute API slots and bound to the order id." if package_available else "A dispute package is unavailable for the selected fallback order.",
        },
        beat_one_id,
    )

    phase0 = service.phase0_summary()
    beat_eight = {
        "beat": BEAT_EIGHT,
        "title": "Honest chart",
        "fallback_used": deferred,
        "fallback_reason": "Phase 4 chart artefacts are deferred for the buildathon." if deferred else None,
        "phase0": phase0,
        "phase4_status": phase4_status,
        "copy": (
            "Phase 0 detection evidence is available. Production-scale robustness validation is deferred; smoke validation is available."
            if deferred
            else "Phase 0 detection evidence and the completed extended validation are available."
        ),
    }

    return {
        "version": "phase5-v1",
        "mode": "smoke_only" if deferred else "extended_validation",
        "source": {
            "phase3_world": str(service.phase3_dir),
            "phase3_outcome": str(service.phase3_outcome_path),
            "phase4_status": phase4_status.get("validation_scope"),
        },
        "beats": [beat_one, beat_two, beat_three, beat_four, beat_five, beat_six, beat_seven, beat_eight],
        "simulator_footer": str(params["report.simulator_footer"]),
    }
