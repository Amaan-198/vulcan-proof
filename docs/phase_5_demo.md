# Phase 5 — Product surface and demo, scripted from available Phase 3/4 evidence

**Read first:** `docs/01_claims.md` (every line), `docs/00_context.md` §4, §8;
Phase 3 artefacts and, when available, the extended Phase 4 artefacts. **Produces:** `vulcan_proof/api/`
(FastAPI), `vulcan_proof/ui/` (React, single-page), `outputs/phase5/demo_script.json`.
**Must not touch:** anything that changes a number. The demo *reads* Phase 3/4 artefacts and the
Phase 4 validation status; it never recomputes. **Halt** after `python scripts\task.py check-phase 5`.

## Principle

The demo was originally written before the parameters existed and two of its beats were
arithmetically false. Every number on every screen is read from `outputs/phase3/` or from genuine
Phase 4 artefacts when they are available; the script is generated, not typed. If a beat's
condition does not hold in the data, the generator emits the **fallback** text for that beat. No
beat may be hand-edited.

## Phase 4 input modes

### Buildathon mode (default)

The Phase 4 production sweep is officially deferred. Missing `outputs/phase4/kappa_star.json`,
`outputs/phase4_REPORT.md`, and production charts are therefore valid inputs to Phase 5. Their
absence must not cause the API, UI, demo generator, or Phase 5 checker to fail.

`GET /report/kappa` must return an explicit non-numeric status payload in this mode, for example:

```json
{
  "validation_scope": "buildathon_smoke",
  "smoke_validation": "completed",
  "production_sweep": "deferred",
  "production_results_available": false
}
```

This status is not a measured κ result. The UI and demo must say that production-scale robustness
validation is deferred. They must not convert the missing result into `κ* = null`, a zero, or a
production claim. Smoke values may be shown only when labelled as smoke/simulator validation.

### Extended-validation mode (optional later)

If the genuine Phase 4 production artifacts exist, Phase 5 may read them without recomputing or
editing them. The response must identify the extended scope, and the Phase 4 extended-validation
criteria must have passed before those values are used as final or publication-grade evidence. A
measured `κ* = null` from a completed sweep remains distinct from the buildathon deferred status.

## API (`vulcan_proof/api/main.py`)
- `GET /order/{order_id}/plan` — runs the trained Arm 5 optimizer on one observed order from the
  canonical test split; returns the plan, per-type standalone and incremental EV, refusal reason
  codes, Stage A/B/C outputs, and the tier. Also returns Arm 4's plan for the same order.
- `GET /order/{order_id}/dispute-package` — for a test-split order with `dispute_opened`, the
  materialised evidence mapped to `evidence.*.api_slot`.
- `GET /report/kappa` — the validated Phase 4 κ report when available, otherwise the explicit
  buildathon deferred-status payload; missing production files must not produce an unhandled error.
- `GET /report/arm4-policy` — the tuned rule table.
- `GET /demo/script` — `demo_script.json`.
- Optional `POST /explain` calls an external LLM to render one sentence from the plan JSON. It is
  **off by default**, behind `VP_EXPLAIN_LLM=1`, has zero decision authority, and the README says so.

## Demo script generator (`vulcan_proof/api/demo_script.py`)
Selects concrete orders from the canonical test split that satisfy each beat's condition, and
writes `demo_script.json` with the numbers filled in. Conditions and fallbacks:

| Beat | Selection condition | Fallback if not found |
|---|---|---|
| 1 Day 62 | any Electronics test order, value 40–50k, `dispute_opened`, type NR, Arm 1 lost | use any lost NR Electronics dispute |
| 2 Same order, average risk | that order's Stage A ≈ category rate (0.8–1.2×): Arm 5 plan; show OTP refusal with EV and the **Arm 4 plan for the same cell alongside** — at average risk they should match; say so | show whatever Arm 5 chose and the OTP EV |
| 3 Same order, flagged | an Electronics test order, 40–50k, Stage A ≥ 3.5× category rate, Arm 5 plan includes otp and refuses signature with `NEGATIVE_INCREMENTAL` | if no signature refusal exists (support < 50 or ρ too low): "The optimizer did not learn an OTP–signature overlap in this world; support = <n>." |
| 4 Contest inversion | two Electronics test orders 40–50k, Stage A 1.8–2.2×, merchants with contest hist ≥ 0.8 and ≤ 0.4; Arm 5 includes otp for the first, no handoff evidence for the second | show the two plans whatever they are, with the merchants' contest rates |
| 5 Cheap order flagged | an Apparel test order 3–4k with Stage A ≥ 3× and a non-empty pre-dispatch plan; state the average-risk plan is empty | "At this world's parameters no apparel order clears; the packing break-even is ₹<threshold>." |
| 6 Defense-only | Phase 4 chart 6 when extended artefacts exist | if production validation is deferred: "Production-scale defense-only evidence is deferred; smoke validation is available." |
| 7 Dispute package | beat 1's order under Arm 5, materialised evidence, API slots | — |
| 8 Honest chart | Phase 0 PR/reliability + Phase 4 chart 1 and verdict when extended artefacts exist | show Phase 0 evidence plus the explicit Phase 4 deferred-validation status |

Every beat's copy is templated from `docs/01_claims.md` "Say" phrases; the generator runs the
NEVER-list check on its own output.

## UI (`vulcan_proof/ui/`)
Single React page (Vite; `ui/package.json` pins exact versions and `package-lock.json` is committed;
Node 22 LTS on Windows; build with `npm ci && npm run build` from PowerShell). The Python
`scripts/run_phase5.py` serves the built `dist/` via FastAPI static files; no separate dev server in the demo.
Screens: Order → Plan (with per-type EV bars and refusal reasons, Arm 4 plan side-by-side) →
Dispute package → Report (κ chart and verdict when available, otherwise deferred-validation
status). Footer string on every ₹ view.

## Tests (`tests/test_phase5.py`)
- `test_api_plan_matches_arm5_artifact` — `/order/{id}/plan` bitmask equals the stored Arm 5 plan for that order (no recomputation drift).
- `test_demo_script_numbers_match_artifacts` — every number in `demo_script.json` is present in
  Phase 3/4 metrics or plan artefacts (string match to 2 dp); deferred-status text contains no
  fabricated production number.
- `test_demo_script_never_list` — NEVER-list grep on `demo_script.json`.
- `test_fallbacks_render` — with a synthetic Phase 4 output where κ\* is null and no signature refusal exists, the generator emits both fallback texts.
- `test_phase4_deferred_status` — with no production Phase 4 directory, `/report/kappa` returns
  the deferred-status payload and the two chart-dependent beats use their documented fallbacks.
- `test_phase4_null_is_not_deferred` — a genuine completed-sweep `κ* = null` response is labelled
  as an extended-validation result, not as an unrun/deferred sweep.
- `test_llm_off_by_default` — `/explain` returns 404 unless env var set.
- `test_footer_in_ui_bundle` — built bundle contains the simulator footer string.

## Done-criteria (`check_phase_5`)
1. All tests pass.
2. `demo_script.json` exists, all eight beats present (with fallbacks where applicable), NEVER-list clean.
3. `uvicorn vulcan_proof.api.main:app` serves `/demo/script` and `/report/kappa`; `/report/kappa`
   returns either validated extended Phase 4 data or the explicit deferred-status payload.
4. `phase5_REPORT.md` lists which beats used fallbacks, why, and whether the demo used smoke-only
   or extended Phase 4 evidence.
5. Buildathon completion does not require `outputs/phase4/` production artifacts. If those artifacts
   are present, they must be complete and pass the Phase 4 extended-validation checks before any
   production-scale claim is surfaced.

**Then halt.** The project is complete.
