# Phase 5 — Product surface and demo, scripted from Phase 4 output

**Read first:** `docs/01_claims.md` (every line), `docs/00_context.md` §4, §8;
`outputs/phase4/kappa_star.json` and `outputs/phase4_REPORT.md`. **Produces:** `vulcan_proof/api/`
(FastAPI), `vulcan_proof/ui/` (React, single-page), `outputs/phase5/demo_script.json`.
**Must not touch:** anything that changes a number. The demo *reads* Phase 3/4 artefacts; it never
recomputes. **Halt** after `python scripts\task.py check-phase 5`.

## Principle

The demo was originally written before the parameters existed and two of its beats were
arithmetically false. Every number on every screen is now read from `outputs/phase3/` and
`outputs/phase4/`; the script is generated, not typed. If a beat's condition does not hold in the
data, the generator emits the **fallback** text for that beat. No beat may be hand-edited.

## API (`vulcan_proof/api/main.py`)
- `GET /order/{order_id}/plan` — runs the trained Arm 5 optimizer on one observed order from the
  canonical test split; returns the plan, per-type standalone and incremental EV, refusal reason
  codes, Stage A/B/C outputs, and the tier. Also returns Arm 4's plan for the same order.
- `GET /order/{order_id}/dispute-package` — for a test-split order with `dispute_opened`, the
  materialised evidence mapped to `evidence.*.api_slot`.
- `GET /report/kappa` — `kappa_star.json`.
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
| 6 Defense-only | Phase 4 chart 6 | — |
| 7 Dispute package | beat 1's order under Arm 5, materialised evidence, API slots | — |
| 8 Honest chart | Phase 0 PR/reliability + Phase 4 chart 1 + verdict sentence | — |

Every beat's copy is templated from `docs/01_claims.md` "Say" phrases; the generator runs the
NEVER-list check on its own output.

## UI (`vulcan_proof/ui/`)
Single React page (Vite; `ui/package.json` pins exact versions and `package-lock.json` is committed;
Node 22 LTS on Windows; build with `npm ci && npm run build` from PowerShell). The Python
`scripts/run_phase5.py` serves the built `dist/` via FastAPI static files; no separate dev server in the demo.
Screens: Order → Plan (with per-type EV bars and refusal reasons, Arm 4 plan side-by-side) →
Dispute package → Report (κ chart, verdict). Footer string on every ₹ view.

## Tests (`tests/test_phase5.py`)
- `test_api_plan_matches_arm5_artifact` — `/order/{id}/plan` bitmask equals the stored Arm 5 plan for that order (no recomputation drift).
- `test_demo_script_numbers_match_artifacts` — every number in `demo_script.json` is present in Phase 3/4 metrics or plan artefacts (string match to 2 dp).
- `test_demo_script_never_list` — NEVER-list grep on `demo_script.json`.
- `test_fallbacks_render` — with a synthetic Phase 4 output where κ\* is null and no signature refusal exists, the generator emits both fallback texts.
- `test_llm_off_by_default` — `/explain` returns 404 unless env var set.
- `test_footer_in_ui_bundle` — built bundle contains the simulator footer string.

## Done-criteria (`check_phase_5`)
1. All tests pass.
2. `demo_script.json` exists, all eight beats present (with fallbacks where applicable), NEVER-list clean.
3. `uvicorn vulcan_proof.api.main:app` serves `/demo/script` and `/report/kappa`.
4. `phase5_REPORT.md` lists which beats used fallbacks and why.

**Then halt.** The project is complete.
