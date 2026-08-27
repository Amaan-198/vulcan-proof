# Phase 1 — Hidden-truth simulator

**Read first:** `docs/00_context.md` §2, §6; `docs/02_engineering_rules.md` B1, B3, B9, B11;
`params/params.yaml` blocks `sim`, `categories`, `archetypes`, `merchants`, `evidence`, `uplift_true`,
`features`. **Produces:** `vulcan_proof/sim/` — a generator that turns `(kappa, seed, n_orders,
shift_enabled)` into an order frame with hidden truth, historical evidence policies, dispute
timing, and censoring flags. **Must not touch:** nothing under `arms/`, `models/`, `opt/` may be
created. No outcome resolution (that is Phase 2) — this phase produces truth and *potential*
disputes, not wins/losses. **Halt** after `python scripts\task.py check-phase 1`.

## What the simulator must produce

Two parquet frames per run, written with `write_artifact`:

- `observed_orders` — schema `ORDER_OBSERVED` = exactly `features.permitted` plus `split`
  (train/validate/gap/test/immature) plus `censored` plus `order_day`. **No `hidden_` column.**
  The four `merchant_*_hist` columns are written as **NaN (nullable float32)** in this phase —
  Phase 2 computes them from resolved history. Nothing in Phase 1 may fill them from hidden state.
- `hidden_orders` — schema `ORDER_HIDDEN` = `order_id` + every hidden column:
  `hidden_truth` (categorical: delivered_correct | misdelivered | never_handed_off | merchant_fault
  | transit_damage), `hidden_z_risk`, `hidden_z_type`, `hidden_risk_mult`, `hidden_quality`,
  `hidden_carrier_reliability`, `hidden_archetype`, `hidden_compliance`, `hidden_contest_base`,
  `hidden_requested_bitmask` (evidence the merchant's historical policy requests, tier- and opt-in-gated; compliance is NOT drawn here),
  `hidden_random_stratum` (bool), `hidden_dispute_potential` (bool: would dispute if not prevented),
  `hidden_dispute_type` (NR/NAD/EB or null), `hidden_dispute_open_day` (nullable int),
  `hidden_resolution_day` (nullable int), `hidden_u_response` (float32 uniform draw — the
  customer's response *if* an acknowledgement is sent; Phase 2 maps it to confirm/report/silent
  per the plan, so the draw is identical across arms).

Phase 2 joins them on `order_id`. Nothing else ever does.

## Generation order (deterministic; each step gets its own `SeedTree` child labelled by step name and seed)

All generation for the canonical world runs single-process. Sweeps (Phase 4) parallelise over
seeds with `ProcessPoolExecutor`; therefore `generate_world(kappa, seed, n_orders, shift, params_path)`
must be a module-level function taking only picklable arguments.

### 1. Merchants (`sim/merchants.py`)
`n = merchants.n_merchants`. Size weight ~ lognormal(σ = `size_lognormal_sigma`), normalised to
sum 1. `hidden_quality` ~ lognormal(σ = `quality_lognormal_sigma`) mean-normalised; it **multiplies the
merchant-fault rate, so higher = worse**. Archetype: sort merchants by `hidden_quality` descending
(worst first) and assign archetypes in contiguous blocks in `quality_rank` order with block sizes
`round(share × n_merchants)` (last block absorbs rounding) — Erratic gets the worst 10%, Minimal
the next 30%, …, Diligent the best 15%. Shares are of merchants and are exact by construction;
the order-weighted shares are reported. This is the deliberate quality↔archetype confounding. `hidden_compliance`, `hidden_contest_base`
from archetype. Category: each merchant sells one category, drawn by `categories.*.share`.
Tier: `eligible_tier` ∈ {FULL, POST_DELIVERY_ONLY, NONE} by `merchants.tier_*`.
`verified_contact_available` = tier ≠ NONE. Ack opt-in ~ Bernoulli(`archetypes.ack_optin_rate`), written to the observed frame as `ack_optin`.
Carrier: each merchant uses one of `sim.n_carriers`; carrier reliability ~ lognormal(σ) mean-normalised.

### 2. Customers (`sim/customers.py`)
Per order (not persistent — customer_id is a hash of order_id and a small pool so that
`prior_disputes`/`account_age` are features, not history): `new_customer` ~ Bern(p),
`address_mismatch` ~ Bern(p), `prior_disputes` ~ Poisson(mean), `account_age_days` ~ lognormal.
If `new_customer`: `prior_disputes = 0`, `account_age_days = 0`.

### 3. Orders (`sim/orders.py`)
`n_orders`. Merchant drawn by size weight. `order_day` ~ uniform over `timeline.order_months`
days. `order_value` ~ log-uniform over the merchant's category range. `payment_method`,
`network` (Visa share), `issuer_family`, `cart_items` ~ 1 + Poisson(1), `hour_of_day`, `month`.
`decision_date = order_day`.

### 4. Individual risk (`sim/risk.py`)
```
z_raw  = c_new·new + c_addr·mismatch + c_prior·log1p(prior) − c_age·log1p(age/30) + c_noise·ε
z_risk = (z_raw − mean)/std          (standardised over the generated population)
risk_mult = exp(κ·z_risk − κ²/2)     (mean ≈ 1 by construction; assert |mean−1| < 0.02)
z_type = standardise(mismatch + 0.5·new + ε₂)
```
`hidden_risk_mult` multiplies both dispute channels in step 7 equally. `z_type` shifts the NR share of the category mix by
`logit(NR) += κ·type_shift·z_type`, then renormalises NAD/EB proportionally.

### 5. Truth (`sim/truth.py`)
Per order, categorical draw with probabilities:
```
p_misdel  = base.misdelivered / carrier_reliability
p_never   = base.never_handed_off / carrier_reliability
p_mfault  = base.merchant_fault × hidden_quality
p_damage  = base.transit_damage × exp(fragility_effect·(fragility − fragility_ref)) / carrier_reliability
p_correct = 1 − sum
```
Assert `p_correct > 0.9` for every row. Assert population shares of each failure state within
±15% of the base rates (mean-normalisation must hold).

### 6. Historical evidence policy (`sim/policy.py`)
`hidden_requested_bitmask`: for the merchant's archetype policy, evidence type `e` is *requested*
if `order_value ≥ policy[e]`. `Erratic`: each type requested independently with p = 0.5.
`hidden_random_stratum` ~ Bern(`random_stratum_frac`): if true, the requested set is a uniform
random subset (each type independently p = 0.5) regardless of archetype. Tier gating: NONE →
nothing requested; POST_DELIVERY_ONLY → only ack/vack. Acks requested only if `ack_optin`.
**Compliance is not drawn in Phase 1.** The Phase 2 resolver draws one compliance Bernoulli per
order (keyed on order, not arm) and applies it to whichever plan is being resolved — arm0's
historical plan included. Drawing it here too would create two inconsistent draws.

### 7. Dispute potential (`sim/disputes.py`) — the derived funnel (trap B9)
```
v_term = (order_value / false_claim_value_ref) ^ false_claim_value_elasticity
if truth == delivered_correct:
    p = min(0.5, gamma[category] · θ · risk_mult · v_term)          # false-claim channel
else:
    p = min(0.5, gamma[category] · risk_mult · v_term)              # genuine-failure channel
hidden_dispute_potential ~ Bern(p)
```
Six unknowns (`gamma` per category, `θ`) are solved in step 12 against six targets. Assert
`p < 0.5` never binds on more than 0.01% of rows (report the count). Note: `risk_mult` scales
both channels — it is "this customer's propensity to escalate", not "this customer's honesty".
Type: if genuine — determined by truth (misdelivered/never → NR; merchant_fault → NAD or EB with
the category's NAD:EB ratio; transit_damage → NAD or EB likewise). If false — drawn from the
κ-shifted category mix. Deterrence = `sim.deterrence` (0 in headline): if > 0, multiply `p_false`
by `1 − deterrence·[any handoff evidence in hidden_requested_bitmask]` — this is the only place it
may appear. Note this uses the *historical* request set, so the deterrence robustness run is an
approximation (it does not respond to the deployed arm's plan); state that in the Phase 4 report.

Prevention is **not** resolved here (Phase 2 resolves it against acks and truth). Phase 1 records
`hidden_dispute_potential` = would dispute absent prevention.

### 8. Customer response draw (`sim/ack.py`)
Phase 1 does **not** decide whether an ack is sent — that is the plan's decision and differs by
arm. It stores one uniform `hidden_u_response ~ U(0,1)` per order. Phase 2's resolver maps it:
given the truth class's `[confirm, report, silent]` triple (with `report` reduced by
`verified_ack_response_penalty` when only vack is sent, mass moved to `silent`), response =
`confirm` if u < p_confirm, `report` if u < p_confirm + p_report, else `silent`; `none` if no ack sent.
Storing the uniform rather than the outcome is what makes the response arm-invariant (B13).

### 9. Latency (`sim/latency.py`)
For rows with `hidden_dispute_potential`:
```
open_offset  = expected_delivery_days + (U(0, fast_max) w.p. fast_share else U(fast_max, dispute_max))
open_day     = order_day + open_offset
resolution   = open_day + response_days + LogNormal(median, p95)
```
`hidden_dispute_open_day`, `hidden_resolution_day` (int days). Null otherwise.

### 10. Splits and censoring (`sim/splits.py`) — trap B3
`split` by `order_day` month per `sim.timeline` (`days_per_month`). Observation boundary: for train/validate =
`decision_month_end × days_per_month`; for test = `outcome_observed_through_month × days_per_month`.
```
censored = (open_day is not null and resolution_day > boundary)
        or (open_day is null and order_day + expected_delivery + dispute_max + response + p95_resolution > boundary)
```
The second clause marks "no dispute yet, but could still" rows as censored — a negative label is
only trustworthy once the full window has elapsed. Assert train and validate censor fraction ≤
`sim.max_censor_frac`; write test's fraction to manifest. **Censored rows keep their truth; they
are simply excluded from labels downstream.**

### 11. Merchant history features — NOT in this phase
`merchant_order_count` is computable here (count of the merchant's earlier orders) and is written.
`merchant_dispute_rate_hist`, `merchant_contest_rate_hist`, `merchant_compliance_hist` require
resolved outcomes and are written as NaN; Phase 2 `sim/history.py` fills them from OUTCOME rows
only. **Do not fill them from `hidden_contest_base` or `hidden_quality`** — that is a leak with a
placeholder's name (trap B19a).

### 12. Funnel calibration (`sim/calibrate.py`) — six unknowns, six targets
Run once on the canonical seed, κ = `sim.kappa.canonical`, `n_orders_sweep` orders, **using
expected probabilities, not Bernoulli draws** (so the solve is deterministic and exact):
1. For a trial θ, and for each category, solve `gamma[cat]` by bisection over `gamma_search` so that
   `mean_over_category_orders(p)` = `categories.<cat>.target_rate` within `calibration_rel_tol`.
2. Compute the population genuine share `Σ p_genuine / Σ p`. Bisect θ over `theta_search` until it
   equals `genuine_share_target` within `calibration_rel_tol`. (Higher θ → more false claims → lower
   genuine share; monotone, so bisection is valid — assert monotonicity on the first two brackets.)
3. Write `outputs/theta.json`: `{theta, gamma: {cat: value}, seed, kappa, n_orders,
   achieved_category_rates, achieved_genuine_share, implied_population_rate, implied_phi}`.
   `implied_phi = 1 − achieved_genuine_share` — this is the realised counterpart of the oracle's
   `reference.phi`; assert |implied_phi − reference.phi| < 0.02.
All other runs **read** this file (`P.derived("theta")`, `P.derived("gamma")`); if absent, raise.
θ and γ are fixed across κ and seeds so the dial changes dispersion, not level; assert the realised
population potential rate stays within ±10% of `implied_population_rate` at every κ and report it.

### 13. World D shift
If `shift_enabled`: for test-split orders only, category shares from `sim.shift.category_share_shift`
and risk coefficients multiplied by `risk_coef_multiplier`. Record in manifest.

## Tests (`tests/test_phase1.py`) — every one required
- `test_smoke_generates` — 20k orders, κ = 0.6, seed 1, < 60 s, both frames written, schemas pass.
- `test_no_hidden_in_observed` — `observed_orders` has no column starting `hidden_`; `write_artifact` raises if attempted.
- `test_forbidden_absent` — no `features.forbidden` name in `observed_orders`.
- `test_reproducible` — same (κ, seed, n) twice → byte-identical parquet.
- `test_risk_mult_mean_one` — |mean(risk_mult) − 1| < 0.02 at κ ∈ {0, 0.6, 1.0}.
- `test_kappa_zero_no_signal` — at κ = 0, Spearman(z_risk, dispute_potential) ≈ 0 (|ρ| < 0.01 on 200k orders).
- `test_truth_shares` — each failure state within ±15% of base rate on 200k orders.
- `test_category_rates_within_tolerance` — after calibration, per-category realised potential rate (Bernoulli, 200k orders) within `category_rate_tolerance` of target. *Failure means:* the funnel is inconsistent; do not widen the tolerance, report.
- `test_genuine_share_and_phi` — realised genuine share within 0.03 of `genuine_share_target`; `implied_phi` within 0.02 of `reference.phi`.
- `test_calibration_deterministic` — two calibration runs produce identical `theta.json`.
- `test_hist_columns_nan` — the three outcome-dependent `*_hist` columns are all-NaN in Phase 1 output (so Phase 2 cannot silently inherit a leak).
- `test_every_evidence_has_support` — on 200k orders, every evidence type is *requested* on ≥ 1% of that archetype's orders for at least one archetype; and (otp, signature) are co-requested on ≥ 200 orders. *Failure means:* the identifiability redesign is broken.
- `test_archetype_blocks` — archetype merchant shares exact to rounding; mean `hidden_quality` strictly decreasing from Erratic to Diligent.
- `test_headline_misdelivered_zero` — `uplift_true.on_misdelivered_otp` == 0 in the loaded params.
- `test_censoring_excluded_not_negative` — no row has `censored == True` and a downstream label; and censor fraction train/validate ≤ max.
- `test_latency_bounds` — open_offset ∈ [7, 127]; resolution − open ≥ 30.
- `test_deterrence_zero_headline` — with deterrence 0, p_false independent of requested bitmask (compare two policies on identical orders; identical dispute_potential draws given identical seeds).
- `test_declared_dtypes` — observed frame memory ≤ 120 bytes/row; hidden ≤ 80 bytes/row (rule A15).

## Done-criteria (`check_phase_1`)
1. All Phase 1 tests pass plus Phase 0 tests.
2. `outputs/theta.json` exists; every `achieved_category_rate` within `calibration_rel_tol`; `achieved_genuine_share` within tol; `implied_phi` within 0.02 of `reference.phi`.
3. Canonical world generated once (3M orders, κ = 0.6, seed = master); manifest committed with `wall_seconds ≤ run.max_canonical_wall_seconds` and `peak_rss_mb ≤ run.max_peak_rss_gb × 1024`; observed frame has zero `hidden_` columns (check re-reads the parquet schema).
4. Per-category realised potential-dispute rate table in `phase1_REPORT.md`, all within tolerance.
5. Censor fractions per split in report; train/validate ≤ 3%.
6. Evidence-type request table (requested count per type per archetype) in report; (otp, signature) co-request count reported.

**Then halt.**
