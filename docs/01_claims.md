# 01 — Claims discipline

Applies to everything a judge reads: `README.md`, `outputs/**/*REPORT.md`,
`outputs/phase5/demo_script.json`, `vulcan_proof/api/**`, `vulcan_proof/ui/src/**`, and
`vulcan_proof/sweep/charts.py` (chart titles/annotations). `tests/test_claims.py` greps those
paths for the NEVER list (case-insensitive, whole phrase) and fails on any hit. The design docs
under `docs/` are **excluded** — they must be able to name the forbidden phrases in order to forbid them.

## Say

- Razorpay has an unusually strong native position: payment intelligence, order metadata, dispute
  outcomes and Dispute Responder in one ecosystem.
- The orchestration layer is the product. Under the simulator's parameters it is worth ₹10k–50k
  per 1,000 orders in electronics and jewellery with zero ML.
- Per-order optimisation adds 5–12% on top of tuned rules when individual-risk signal exists;
  κ\* is where that goes to zero. Whether Razorpay's data sits above κ\* cannot be answered from outside.
- Under the simulator's parameters, at κ = 0 a tuned category × value × contest-band rule captures 99.5%+ of achievable value. That tie was pre-registered.
- Paid handoff evidence is an electronics and jewellery capability.
- Provenance proves when and for which order an artifact was captured.
- Silence is a risk feature, not dispute evidence.
- The admissibility matrix is simplified, versioned, and derived from published network guidance.
- Yield estimates are observational, doubly confounded, and not causal. Production needs a pilot.
- Per-evidence uplifts are assumed values, stated, and swept first.
- Dispute fees are not recovered on a win (Razorpay published guidance). They enter prevention only.
- Mastercard non-receipt is coded under 4853.
- Detection generalises on real orders (Olist); everything with a ₹ sign is simulated.
- The share of disputes arising despite correct fulfillment is an output of the simulator, reported per run.

## Never say

| Never | Why |
|---|---|
| "only Razorpay could build this" | market-uniqueness overclaim |
| "unforgeable" | provenance proves timing and binding, not honesty |
| "structurally incapable of helping bad merchants" | it can, on carrier-fault cases; reported |
| "nobody works at day three" | unverifiable |
| "most disputes are false" | share is a simulator output |
| "confirmation defeats the chargeback" | ack is one input among several |
| "no threshold rule could do this" | a threshold rule captures 99.5% at κ = 0 |
| "beats rules" / "outperforms static policy" (unqualified) | true only above κ\*; always qualify |
| "the fee is recovered on a win" | it is not |
| "MC 4855" | retired into 4853 |
| "Vulcan uses…" / any claim about Vulcan internals | not published |
| "contest more" / "raise your contest rate" | offense-capable under Track 02 |
| "reimbursement", "settlement hold", "escrow", "insurance" | product promises the system does not make |
| "causal", "proven yield" (about uplifts) | observational |
| "real-world savings" without "simulated" in the same sentence | labelling rule |

## Labelling rule

Every chart with a ₹ axis carries the footer: `Simulator result · production calibration requires
Razorpay dispute history`. Every Olist chart carries: `Olist public dataset · Brazil 2016–18 ·
no chargeback or evidence data · detection only`. `tests/test_claims.py` checks the footer strings
are present in the chart-generation code.
