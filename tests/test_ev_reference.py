"""Known-answer tests for the frozen EV oracle (vulcan_proof/ev_reference.py).

These run in EVERY phase. They prove two things:
  1. The oracle still reproduces the pre-build arithmetic (someone did not edit it).
  2. params/params.yaml and the oracle agree on every shared constant (trap B19).
Failure of (1): restore ev_reference.py from git. Failure of (2): fix params.yaml — NEVER the oracle.
"""
import math, pathlib, sys
import pytest, yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vulcan_proof import ev_reference as R  # noqa: E402

PARAMS = yaml.safe_load((ROOT / "params" / "params.yaml").read_text())
KA = {k: v["value"] for k, v in PARAMS["reference"]["known_answers"].items()}


def _inc(base, e, cat, V, **kw):
    a, _, _ = R.ev_set(base, cat, V, **kw)
    b, _, _ = R.ev_set(tuple(base) + (e,), cat, V, **kw)
    return b - a


def test_electronics_45k_x1_otp_standalone():
    ev, _, _ = R.standalone("otp", "Electronics", 45000)
    assert round(ev, 2) == KA["electronics_45k_x1_otp_standalone_ev"]


def test_electronics_45k_x1_otp_breakeven():
    assert round(R.threshold("otp", "Electronics")) == KA["electronics_45k_x1_otp_breakeven"]


def test_apparel_3500_packing():
    ev, _, _ = R.standalone("packing", "Apparel", 3500)
    assert round(ev, 2) == KA["apparel_3500_x1_packing_standalone_ev"]
    assert round(R.threshold("packing", "Apparel")) == KA["apparel_packing_breakeven_x1"]


def test_electronics_45k_x2_otp_refused_incrementally():
    bev, bs = R.best_subset("Electronics", 45000, risk=2)
    assert "otp" not in bs and "geotag" in bs and "ack" in bs
    base = tuple(x for x in bs if x != "otp")
    assert round(_inc(base, "otp", "Electronics", 45000, risk=2), 2) == KA["electronics_45k_x2_otp_incremental"]


def test_electronics_45k_x4_signature_refused():
    bev, bs = R.best_subset("Electronics", 45000, risk=4)
    assert "otp" in bs and "signature" not in bs
    base = tuple(x for x in bs if x != "signature")
    assert round(_inc(base, "signature", "Electronics", 45000, risk=4), 2) == KA["electronics_45k_x4_signature_incremental"]


def test_jewellery_200k_x1_signature_refused():
    bev, bs = R.best_subset("Jewellery", 200000, risk=1)
    assert "otp" in bs and "signature" not in bs
    base = tuple(x for x in bs if x != "signature")
    assert round(_inc(base, "signature", "Jewellery", 200000, risk=1), 2) == KA["jewellery_200k_x1_signature_incremental"]


def test_apparel_3500_x1_empty_plan():
    bev, bs = R.best_subset("Apparel", 3500, risk=1)
    assert bs == () and bev == 0.0


def test_empty_plan_ev_exactly_zero():
    ev, b, c = R.ev_set((), "Electronics", 45000)
    assert ev == 0.0 and b == 0.0 and c == 0.0


def test_cash_on_compliance():
    # cost term for a cash item must be cash × materialisation (compliance × presence), not raw cash
    _, _, cost = R.ev_set(("otp",), "Electronics", 45000)
    assert math.isclose(cost, 25 * R.COMPLIANCE_POP * 0.90)


def test_reference_matches_params():
    P = PARAMS
    assert R.HOURLY == P["econ"]["hourly_rate"]["value"]
    assert R.PHI == P["reference"]["phi"]["value"]
    assert math.isclose(R.PC_POP, P["reference"]["pc_population"]["value"], abs_tol=1e-6)
    assert math.isclose(R.COMPLIANCE_POP, P["reference"]["compliance_population"]["value"], abs_tol=1e-6)
    for e in P["evidence"]["order"]["value"]:
        row = P["evidence"][e]
        assert R.EV_TYPES[e]["cash"] == row["cash"], e
        assert R.EV_TYPES[e]["sec"] == row["seconds"], e
        assert math.isclose(R.EV_TYPES[e]["mat"], row["presence_factor"]), e
        assert R.EV_TYPES[e]["adm"] == set(row["admissible"]), e
        assert (e in R.SYSTEM_SENT) == row["system_sent"], e
    for d in ("NR", "NAD", "EB"):
        assert R.UPLIFT[d] == P["uplift_true"][d], d
        assert R.OVERLAP[d] == P["overlap"][d]["value"], d
        assert R.BASE_WIN[d] == P["base_win"][d]["value"], d
    for c in P["categories"]["order"]["value"]:
        row = P["categories"][c]
        assert R.CAT[c]["pA"] == row["target_rate"], c
        assert R.CAT[c]["mix"] == row["mix"], c
        assert R.CAT[c]["vmin"] == row["vmin"] and R.CAT[c]["vmax"] == row["vmax"], c
    # archetype-derived population rates
    A = P["archetypes"]
    pc = sum(A[a]["share"] * A[a]["contest"] for a in A["order"]["value"])
    comp = sum(A[a]["share"] * A[a]["compliance"] for a in A["order"]["value"])
    assert math.isclose(pc, R.PC_POP, abs_tol=1e-9)
    assert math.isclose(comp, R.COMPLIANCE_POP, abs_tol=1e-9)
