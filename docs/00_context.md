# 00 — Context: why every constraint exists

Read this before anything else. You have not seen the design history. This file carries forward
the reasons behind the rules, because an implementer who knows why a rule exists will not work
around it when it becomes inconvenient. Nine review rounds and one full arithmetic pass went into
these constraints. Each one closed a way to produce a plausible-looking number that is wrong.

## 1. What the system does

A merchant sells a ₹45,000 phone. Payment clears, it is delivered. Day 62: "I never received it."
The merchant's dispute tool finds a courier record reading `DELIVERED` and nothing else. The
merchant loses the money and the goods. The evidence that would have won — an OTP handoff, a
packing photo, a customer acknowledgement — had to be created on day three.

Vulcan Proof runs at payment time. It estimates dispute exposure, decides which admissible evidence
is worth its cost, and orchestrates capture. Three dispute types are in scope: non-receipt (Visa
13.1 / Mastercard 4853), not-as-described (Visa 13.3 / MC 4853), empty box. Nine evidence types
(`params.yaml: evidence`). Prepaid physical goods only.

## 2. The simulator and the firewall

There is no Razorpay data. Everything is measured on a simulator with **hidden truth**: each order
has a fulfillment truth (delivered correct / misdelivered / never handed off / merchant fault /
transit damage), true evidence uplifts, and true merchant behaviour. The optimizer trains on
*observations* from that world and is scored against the *hidden* truth.

**The optimizer must never see truth, true uplifts, or any simulator parameter.** If it does, it
wins every experiment and nothing in the output will look wrong. This is the single most
important rule in the repo. It is enforced by:

- package separation (`vulcan_proof/sim/` may import truth; `vulcan_proof/opt/` may not);
- a static check that fails the build if any symbol containing `truth`, `uplift_true`, `hidden`,
  or `sim.` is referenced under `vulcan_proof/opt/`, `vulcan_proof/models/`, `vulcan_proof/arms/`;
- a runtime assertion that the order frame passed to any model or optimizer contains only columns
  in `params.yaml: features.permitted`.

The original design wrote one EV equation containing `P(truth | order)`. That equation is the
simulator's, not the optimizer's. The optimizer's equation (§3) marginalises over truth because
its training data does.

## 3. The two EV equations

**Simulator (generative, truth-conditional)** — used only to resolve outcomes:

```
value_true(E | x) = Σ_d P(d | x, truth) · P(contest | d, merchant, E) · Σ_truth P(truth | x)
                    · P(materialise | truth, E) · ΔP(win | contested, d, E, truth) · order_value
                    − cash_cost(E realised) − time_cost(E requested)
```

**Optimizer (estimable, truth-blind)** — the only equation `vulcan_proof/opt/` computes:

```
EV̂(E | x) = Σ_d P̂_A(dispute | x) · P̂_B(d | dispute, x) · P̂_C(contest | d, merchant, E)
            · P̂_M(materialise | E, merchant, ctx)
            · [ P̂_W(win | contested, d, E, ctx) − P̂_W(win | contested, d, ∅, ctx) ] · order_value
            − cash_cost(E) · P̂_M(materialise | E, merchant, ctx) − time_cost(E)
```

Every `P̂` is a fitted model. Cash cost is paid on **compliance** (the merchant actually books the
OTP), so it is expected, not certain. Merchant time is paid on **request** (reading the
recommendation). `vulcan_proof/ev_reference.py` is the frozen known-answer implementation of the
optimizer's equation using the parameter values as if they were perfectly estimated; every phase
tests against it.

## 4. What the arithmetic found (and the design decisions it forced)

The optimizer's equation was evaluated by hand on 20 orders before any code. Results that bind:

| Finding | Consequence in this repo |
|---|---|
| A tuned category × value rule captures 99.5–99.8% of achievable ₹ at zero individual-risk signal | The headline claim is the orchestration layer, not the ML. κ\* (Phase 4) measures where ML starts paying. |
| Oracle per-order optimisation adds 7–12% at realistic signal (σ≈0.7); a real model recovers a fraction | The ML claim is "5–12% on top", never "beats rules". |
| OTP, signature, geotag never clear cost in apparel, home, FMCG at ≤4× risk | Paid handoff evidence is scoped to electronics and jewellery. The optimizer may still evaluate it elsewhere; it will refuse. |
| Packing photo break-even in apparel is ₹10.4k at ₹300/hr, above the whole category range | **The "₹3,500 kurta gets a packing photo" demo is dead at average risk.** The supported inversion is the *contest-rate* case (§5). |
| ₹45k phone: OTP is refused at 1× (₹−6.86) and still at 2× (incremental −₹1.27 over geotag+ack) | Optimizer must evaluate subsets, not items. A per-item threshold optimizer is wrong by construction. |
| Signature is refused when OTP is held, at ₹45k/4× and ₹200k jewellery/1× | The "stacking refusal" demo exists only if the (OTP, signature) bitmask has training support ≥ 50 — see support mask. |
| Per-evidence uplifts are **not in the design**; values were assumed | They are marked `ASSUMED` in `params.yaml`, sensitivity rank 1, and are swept first in Phase 4. |
| Break-even has elasticity exactly 1 in every parameter | Sweep priority = width of plausible range, not elasticity. Ranks are in `params.yaml`. |
| Compliance cancels out of cash-item thresholds | Do not "correct" OTP thresholds for compliance. It does not cancel for time items. |

## 5. Why each economic rule is what it is

- **Dispute fee is not in the win/lose delta.** Razorpay's published guidance: chargeback fees are
  charged per dispute regardless of outcome and are not refunded on a win. In the resolver's money
  table the fee is charged **whenever a dispute is opened**, win or lose, so `value(won) − value(lost)
  = order_value` exactly and the fee cancels out of every evidence decision. Prevention avoids the
  dispute and therefore avoids the fee — that is the only place the fee changes a decision. The fee
  is read in exactly one place, `vulcan_proof/economics.py` (the money table and prevention gain),
  called by the resolver and by Stage P; it is never read under `opt/`.
- **COGS and handling are excluded from the defence delta** for the same reason: goods are gone
  whether the merchant wins or loses.
- **Contest rate is a merchant feature and depends on the plan.** Merchants contest more when they
  hold evidence. Stage C is `P(contest | d, merchant, E)`. This is a real feedback loop, and it
  doubly confounds the observational yield estimate (merchant quality × evidence-dependent
  contesting). Named, not hidden.
- **Acknowledgement uplift is 0.4× OTP on non-receipt** (ASSUMED, anchored to Visa's 2021 rule that
  issuers must address evidence the cardholder or an authorised person received goods at the agreed
  location — carrier handoff is that class of evidence; a customer tap is not). If the ratio were
  near 1 the optimizer would never buy OTP and would be right. Swept.

## 6. Why the simulator is shaped the way it is

- **Dispute rate is derived, not set — and the funnel has exactly as many unknowns as targets.**
  Two channels produce dispute potential: genuine failures (truth ≠ correct) and false claims
  (truth = correct). Both scale with a per-category escalation weight γ_cat and with order value;
  the false channel additionally carries θ. Six unknowns (five γ, one θ) are solved against six
  targets (five category rates, one genuine-share = 0.35). The population dispute rate and the
  share of disputes against correct fulfillment (≈0.65, the oracle's φ) are then *outputs*. The
  first draft set truth rates, category rates, and the false-claim share as three independent
  inputs and was inconsistent at the top of its own sweep; the second draft nearly repeated the
  error with a fixed genuine-dispute probability that alone exceeded the FMCG target.
- **Latency is modelled and censoring is explicit.** Visa 13.1 allows 120 days from expected
  delivery; response 30 days; resolution follows. Train labels are only used when resolution is
  observed before the simulated decision date. Censored disputes are **excluded** from labels,
  never labelled negative, and the exclusion rate is reported.
- **Every archetype has compliance < 1 and every evidence type appears in some archetype's policy**,
  plus a 5% random-assignment stratum. Without this, evidence presence is collinear with merchant
  identity and six of nine yields are unidentifiable.
- **Merchant history features are computed from resolved outcomes, never from hidden state.** An
  early draft seeded `merchant_contest_rate_hist` from the merchant's hidden contest propensity
  "as a placeholder". That is a leak with a friendly name. History features are built in Phase 2
  from the resolved historical policy (`arm0`) using only OUTCOME columns and dates, with the same
  maturity rule as labels. `merchant_integrity_hist` was dropped from the MVP because no observable
  process produced it.
- **Support mask.** Any (dispute type, evidence type) pair with < 50 contested disputes in training
  is removed from the optimizer's action space. The optimizer cannot buy what it could not learn.
- **Deterrence = 0 and silence yield = 0 in the headline.** Non-zero deterrence makes Stage A circular.
- **Wrong-recipient OTP uplift = 0 in the headline.** A partial uplift there is a win against a
  customer who truly never received the goods; that is reported as a carrier-fault sweep, not
  built into the headline.

## 7. Why the evaluation is shaped the way it is

- **Seeds and paired differences.** The test window holds ~850 contested disputes; a single-seed
  Arm 5 − Arm 4 difference is inside its own standard error. Every reported difference is a paired
  (same orders) mean over ≥5 seeds with a 95% CI.
- **κ dial instead of "worlds".** "Arm 5 beats Arm 4 in the world built for Arm 5" is a tautology.
  κ ∈ [0,1] scales how much customer-level features move risk and type. κ\* is reported. Arm 4
  conditions on category × value band × contest-history tercile × tier so that at κ = 0 the only
  thing Arm 5 could exploit is noise; the κ = 0 guard therefore asserts Arm 5 − Arm 4 is within 1%
  of the orchestration value, not that it is exactly zero (contest-history is a legitimate observed
  feature and a per-order optimizer can use it slightly better than a tercile).
- **Kill condition.** If no κ ≤ 1 yields a paired Arm 5 − Arm 4 CI excluding zero, the ML claim
  is dropped. This is a permitted result. Phase 4 must be able to produce it.
- **Calibration over ranking.** The optimizer multiplies three probabilities. Miscalibration
  compounds. Isotonic calibration on validation for every stage, and the calibrated mean must equal
  the empirical base rate (intercept trap, `02_engineering_rules.md`).

## 8. Defense-only

Track 02: "anything offense-capable is disqualified." Evidence effects are conditioned on truth in
the simulator: packing photo of the wrong item has negative uplift; weight of an empty box convicts
the merchant. Win rate is reported three ways — against correct fulfillment (↑), merchant fault
(flat or ↓), carrier fault (reported honestly). The product never tells a merchant to contest more;
it may say "you hold admissible evidence on this dispute" and nothing else.

## 9. Platform

Windows 11 native, CPython 3.13, no WSL, no GPU. Every path is `pathlib`; every subprocess is an
argument list; every parallel section is `ProcessPoolExecutor` under an `if __name__ == "__main__"`
guard (Windows has no fork). See `02_engineering_rules.md` A12–A14.

## 10. Things that are simulated and must stay labelled

All ₹. All dispute rates and mixes (anchored to US/global chargeback statistics; no Indian
prepaid-goods figures exist publicly). All uplifts (assumed). Olist provides the only real-data
number: detection PR-AUC and calibration on a temporal holdout.
