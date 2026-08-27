# 03 — Phases, order, and halt protocol

Six phases. Each produces something mechanically checkable before the next begins. An agent
implements exactly one phase per instruction and halts.

| Phase | Produces | Depends on | Done when (`python scripts\task.py check-phase N`) |
|---|---|---|---|
| **0** | Infrastructure (params loader, schemas, errors, manifest, artifact writer) **+** Olist real-data anchor | nothing | `tests/test_params.py`, `tests/test_ev_reference.py`, `tests/test_phase0.py` pass; `outputs/phase0/metrics.json` exists with PR-AUC, Brier, ECE, FP-cost table; `outputs/phase0_REPORT.md` written |
| **1** | Hidden-truth simulator: orders, merchants, truth, evidence policies, latencies, censoring, θ calibration | 0 | `tests/test_phase1.py` pass; smoke world (20k) and canonical world (3M, 1 seed) generate; calibration targets within tolerance; censor fraction ≤ max |
| **2** | Outcome resolver, prevention, **arm0 (historical policy) + merchant history features**, Arms 1–4; paired ₹ reporting | 1 | `tests/test_phase2.py` pass; Arms 1–4 net ₹ with CI on 5 seeds of the smoke world at κ ∈ {0, 0.6}; Arm 2 ≤ Arm 4; history features written |
| **3** | Stages A/B/C, materialisation model, defensibility model, support mask, calibration, subset optimizer = Arm 5 | 2 | `tests/test_phase3.py`, `tests/test_firewall.py`, `tests/test_repro.py` pass; known-answer optimizer cases match `ev_reference`; calibrated means within tolerance; κ = 0 gain ≤ 1% of orchestration value |
| **4** | κ-sweep with κ\*; OAT and LHS sweeps; kill-condition verdict; all charts | 3 | `tests/test_phase4.py` pass; `outputs/phase4/kappa_star.json` exists with a value or `null` and the verdict; every chart carries the footer |
| **5** | FastAPI service + React surface; demo script instantiated from Phase 4 numbers | 4 | `tests/test_phase5.py` pass; demo script JSON built from `outputs/phase4/`; smoke run of the five-minute flow |

## Rules that apply to every phase

1. Read `docs/00_context.md`, `docs/02_engineering_rules.md`, `params/params.yaml`, and your
   phase document. Do not read other phase documents.
2. Do not modify `params/params.yaml` values. If a value is missing, stop and report; do not invent.
   Adding a *new* key is permitted only if the phase document lists it as "to be added in this phase".
3. Do not modify `vulcan_proof/ev_reference.py`. Ever.
4. Do not modify code owned by an earlier phase except to fix a bug that a test in your phase
   exposes, and then say so in the report.
5. Write tests first for every trap listed in your phase document, then the code.
6. Run `python scripts\task.py check-phase <N>` inside the venv; write `outputs/phase<N>_REPORT.md`; halt.
7. Every `scripts/run_phase*.py` starts with `require_venv()` and wraps its body in
   `if __name__ == "__main__":` (Windows spawn).

## `scripts/check_phase.py`

Takes the phase number, runs the phase's mechanical done-criteria (file existence, manifest
validity, threshold checks listed in the phase doc), prints a pass/fail table, and exits non-zero
on any failure. Phase 0 implements this script with a registry so later phases only add a
function `check_phase_N()` to it.

## Smoke-world contract

Every phase from 1 onward must run end-to-end on `run.n_orders_smoke` (20,000 orders) in under
two minutes on a laptop. Tests use the smoke world. The canonical world (3M) is run once per phase
by the agent and its manifest committed; it is not run in tests.

## Directory contract

```
vulcan_proof/
  params.py        Phase 0   loader; P["a.b.c"] access; lint
  errors.py        Phase 0   InvariantError, SchemaError, LeakError
  schemas.py       Phase 0   named column schemas + check()
  manifest.py      Phase 0   run_id, manifest.json, write_artifact()
  seeds.py         Phase 0   SeedSequence tree
  envcheck.py      Phase 0   require_venv()
  olist/           Phase 0   load, label, features, train, evaluate
  sim/             Phase 1   merchants, customers, orders, risk, truth, policy, disputes, ack, latency, splits, calibrate
  economics.py     Phase 2   pure economics (money table, prevention cost/gain; the ONLY reference to econ.dispute_fee)
  sim/resolve.py   Phase 2   outcome resolver (truth-conditional; the ONLY consumer of hidden_ columns)
  sim/prevention.py Phase 2  prevention mode economics (no fee reference)
  sim/history.py   Phase 2   merchant history features from OUTCOME only (inside the firewall walk)
  sim/arm0_history.py Phase 2 historical policy → PLAN (reads hidden_requested_bitmask; hence under sim/)
  arms/            Phase 2   arm1..arm4; Phase 3 adds arm5
  models/prevention.py Phase 3 Stage P (prevention gain; second permitted fee reference)
  models/          Phase 3   stage_a, stage_b, stage_c, materialisation, defensibility, calibrate
  opt/             Phase 3   subset optimizer (truth-blind; firewall-checked)
  sweep/           Phase 4   kappa, oat, lhs, kappa_star, charts
  api/             Phase 5   FastAPI
  ui/              Phase 5   React
scripts/
  check_phase.py   Phase 0
  run_phase0.py … run_phase5.py
tests/
  test_params.py, test_ev_reference.py       Phase 0 (run in every phase)
  test_phase0.py … test_phase5.py
  test_firewall.py, test_repro.py, test_claims.py   Phase 3 onward (firewall/repro), claims from Phase 0
```
