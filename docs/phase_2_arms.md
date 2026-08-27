# Phase 2 — Outcome resolver, prevention, historical policy (arm0), merchant history features, Arms 1–4, paired reporting

**Read first:** `docs/00_context.md` §3 (simulator equation), §5, §6; `docs/02_engineering_rules.md`
B6, B8, B11, B12, B13, B14, B15, B19a. **Produces:** `vulcan_proof/economics.py` (pure functions:
`prevention_cost(mode, order_value, cogs)`, `money(...)` table — shared by the resolver and, in
Phase 3, by Stage P; imports only `params`), `vulcan_proof/sim/resolve.py`,
`sim/prevention.py` (mode draw; calls `economics.prevention_cost`), `sim/history.py`, `sim/arm0_history.py`, `vulcan_proof/arms/{base,arm1,arm2,arm3,arm4}.py`,
`vulcan_proof/report/paired.py`, `scripts/run_phase2.py`. **Must not touch:** `models/`, `opt/`
(do not create). Arm 5 does not exist yet. **Halt** after `python scripts\task.py check-phase 2`.

## Why this phase exists before any ML

The orchestration layer is the product. Arms 1–4 are the no-ML policies; Arm 4 is the competitor
Arm 5 must beat above κ\*. This phase also produces (a) the training data for Phase 3 — the
resolved outcomes of each merchant's *historical* policy (arm0) — and (b) the merchant history
features, which can only be computed from resolved outcomes. It produces the first headline figure:
**net ₹ of the orchestration layer with zero ML** (Arm 4 − Arm 1).

## Concepts

An **arm** is `plan(observed_orders: ORDER_OBSERVED) -> PLAN` with `PLAN` = `order_id,
requested_bitmask (uint16 over evidence.order)`. Arms see **only** `ORDER_OBSERVED` (schema check
at entry). The **resolver** takes `PLAN + hidden_orders + observed_orders` and returns `OUTCOME`.
The resolver is the only code that reads `hidden_` columns, and it lives under `sim/`.

## Order of operations in `scripts/run_phase2.py` (under `if __name__ == "__main__":`)

1. Load world (Phase 1 parquet).
2. **arm0** (`sim/arm0_history.py` — it lives under `sim/`, not `arms/`, because it reads
   `hidden_requested_bitmask`; it is the history generator, not a policy): `requested_bitmask =
   hidden_requested_bitmask` (the archetype policy, tier- and opt-in-gated in Phase 1). Resolve →
   `OUTCOME_arm0`. Nothing under `arms/` may read a hidden column; the firewall walk covers `arms/`.
3. **History features** (`sim/history.py`): from `OUTCOME_arm0` + `order_day` only, overwrite the
   NaN `merchant_*_hist` columns in `observed_orders` and write `observed_orders` again (same
   `write_artifact` leak check). Then re-run `schemas.check` — no NaN remains in permitted features.
4. Arms 1–4 plan on the completed observed frame; resolve each.
5. Paired report.

## `sim/resolve.py` — truth-conditional outcome

Per order, given `requested_bitmask`:

1. **Compliance.** `complied ~ Bern(hidden_compliance)` from `SeedTree.child("resolve", seed, "compliance", order_id)`
   — **no arm id in the key** (B13), so a merchant is equally compliant under every arm. Same for
   customer presence, ack response, contest draw, and win draw: keyed on `(seed, purpose, order_id)`
   and, where the outcome depends on the plan, on the *plan-relevant quantity* (e.g. the win draw
   uses a uniform `u_win` keyed on order_id and compares it to `p_win(plan)` — identical `u` across
   arms, so arms differ only through `p`). Document this "common random numbers" design in the
   module docstring; it is what makes paired differences low-variance.
2. **Materialisation.** For each requested and complied type: presence draw ~ Bern(presence_factor)
   for otp/signature (uniform keyed on `(seed, "presence", order_id)`); ack/vack are sent iff
   requested ∧ `ack_optin` ∧ tier allows, and materialise iff the response ≠ silent. Response is
   `hidden_u_response` mapped through the truth-class triple (Phase 1 §8) — identical `u` across arms. Truth gating: `never_handed_off` → geotag/otp/signature
   **cannot** materialise; `misdelivered` → otp/signature materialise, flagged `wrong_recipient`.
3. **Costs.** `cash_cost = Σ cash_e · [requested_e ∧ complied]` for merchant items; acks cost cash
   when sent (system-sent, if opt-in and tier allows). `time_cost = Σ seconds_e · [requested_e] × hourly/3600`.
4. **Prevention** (`sim/prevention.py`). If an ack was sent, response == `report`, and
   `hidden_dispute_potential`: the dispute is prevented. Draw mode by `econ.prevention.share_*`.
   ```
   explanation:  prevention_cost = support_cost
   refund:       prevention_cost = order_value − salvage_sigma · cogs · order_value
   replacement:  prevention_cost = cogs · order_value + reship_cost + support_cost
   ```
   `prevention.py` does **not** reference the dispute fee. Prevention's advantage over a dispute
   comes from the fee being charged in the dispute branch below.
   **Reports without dispute potential** (customer reports a problem but would not have disputed)
   cost nothing in this model — they are assumed to flow through the merchant's ordinary returns
   channel identically under every arm — and are counted as `false_friction` in the report. State
   this assumption in the resolver docstring.
5. **Dispute.** If potential and not prevented: `dispute_opened = True`, `dispute_type` from hidden.
   `p_contest = sigmoid(logit(hidden_contest_base) + contest_evidence_slope · n_materialised_admissible)` (B12).
6. **Win.** If contested:
   ```
   base = base_win[type]
   S = materialised types admissible for type, with truth modifiers (uplift_true.on_*):
       merchant_fault → pre-dispatch uplifts × on_merchant_fault (−1)
       transit_damage → × on_transit_damage (+1)
       misdelivered   → otp/signature uplift × on_misdelivered_otp (0 in headline)
   uplift = (max(u) + (1 − overlap[type]) · Σ rest) × uplift_true.sweep_multiplier, clipped so p_win ∈ [0, 1]
   won = (u_win < base + uplift)
   ```
   Silence yield (0 in headline) adds to uplift if response == silent.
7. **Money** — the table, implemented as `economics.money(...)`; `econ.dispute_fee` is read inside `vulcan_proof/economics.py` only.
   ```
   none (no potential, or potential resolved without dispute):  value = 0
   prevented:                    value = −prevention_cost
   opened, not contested:        value = −order_value − fee
   opened, contested, won:       value = −fee
   opened, contested, lost:      value = −order_value − fee − ratio_damage
   net = value − cash_cost − time_cost
   ```
   `value(won) − value(lost) = order_value + ratio_damage` — the fee cancels (B8).

`OUTCOME` schema: `order_id, arm_id, requested_bitmask, complied, materialised_bitmask,
wrong_recipient, cash_cost, time_cost, ack_sent, response, prevented, prevention_mode,
dispute_opened, dispute_type, contested, won, value, net, censored, split, claim_class`
with `claim_class` ∈ {correct_fulfillment, merchant_fault, carrier_fault, none} (carrier_fault ←
misdelivered / never_handed_off / transit_damage).

## `sim/history.py` — inside the firewall (B19a)

Inputs: `OUTCOME_arm0` (only columns `order_id, complied, requested_bitmask, dispute_opened,
contested, censored`) and `observed_orders[order_id, merchant_id, order_day]`. For each order, over
the merchant's prior orders with `order_day_prior ≤ order_day − lookback_days` **and not censored
at `order_day`** (re-use the Phase 1 censoring rule with boundary = this order's `order_day`):
- `merchant_dispute_rate_hist` = (disputes + prior·pop_rate) / (n + prior), `prior = hist_shrinkage_n`,
  `pop_rate` = train-split population rate computed from *non-censored* arm0 outcomes.
- `merchant_contest_rate_hist` = (contested + prior·pop) / (opened + prior).
- `merchant_compliance_hist` = (complied + prior·pop) / (requested_any + prior).
Vectorise with a sorted merge-asof per merchant; no Python loops over 3M rows.
Known simplification, stated in the docstring and the report: history features for test orders
are derived from arm0 (historical) outcomes of earlier orders, not from the deployed arm's
outcomes. This is the same for every arm, so paired comparisons are unaffected.

## Arms

All arms respect the **availability mask**: tier NONE → empty; POST_DELIVERY_ONLY → acks only;
acks only if `ack_optin`. Arms read `eligible_tier`, `ack_optin`, `verified_contact_available`.

- **Arm 1** — empty plan.
- **Arm 2** — every admissible type the tier allows. Sanity check; must be ≤ Arm 4 in net ₹.
- **Arm 3** — `arms.arm3_rules` literal thresholds.
- **Arm 4** — cells = category × log value band (`arm4_value_bands`) × contest-history tercile
  (terciles of `merchant_contest_rate_hist` computed on the **validation** split) × tier. For each
  cell, choose the subset of the 9 types maximising realised net ₹ on the **validation split**
  (resolve each of the ≤ 512 subsets on validation, sum `net`, argmax; ties → fewer types).
  Never reads test (B15). Written to `outputs/<run>/arm4_policy.json`; this is the artefact
  Phase 5 shows as "the rule Arm 5 must beat".

## Paired reporting (`vulcan_proof/report/paired.py`)

For two arms over the **same** seeds: per seed, `net_per_1000 = per_orders × Σ net / n_orders` on
the test split, **censored rows excluded from Σ net and from n_orders** (headline) and also with
censored rows valued 0 (reported as a second column). Paired difference per seed; mean; 95% CI
from Student-t with `n_seeds − 1` dof; `P(diff > 0)`. `InvariantError` if `n_seeds < report.min_seeds`.
Per arm: coverage, friction (% OTP requested, % taps, mean taps), materialisation rate per type,
prevention vs defence split, win-rate by `claim_class` among contested.

## Tests (`tests/test_phase2.py`)
- `test_arms_only_see_observed` — extra `hidden_x` column → `SchemaError` for every arm.
- `test_arms_share_orders` — 5 seeds × Arms 0–4 on smoke world: identical order_id sets and `hidden_truth`.
- `test_common_random_numbers` — same order, same seed, two arms with the same requested set → identical `complied, response, materialised, contested, won`.
- `test_never_handed_off_no_handoff` — force truth = never_handed_off, full plans: otp/geotag/signature never materialise.
- `test_merchant_fault_negative` — truth = merchant_fault, plan = packing, contested: mean p_win < base_win[NAD].
- `test_misdelivered_zero_headline` — truth = misdelivered, plan = otp: p_win == base_win[NR] exactly.
- `test_cash_on_compliance` — non-complied order: cash_cost 0, time_cost > 0.
- `test_fee_scope` — token `econ.dispute_fee` appears in exactly one file, `vulcan_proof/economics.py`; `sim/resolve.py` calls `economics.money`; nothing else references the fee.
- `test_win_lose_delta_is_order_value` — 1,000 resolved contested rows: `value(won) − value(lost) == order_value + ratio_damage` per row (evaluate both branches with the same inputs).
- `test_money_table` — six table-driven cases, exact; includes replacement on ₹45,000 @ cogs 0.85: `value = −38,500`; contested-lost: `−45,500`; contested-won: `−500`.
- `test_prevention_no_fee_reference` — `sim/prevention.py` source contains no `dispute_fee`.
- `test_history_from_outcomes_only` — (B19a) two worlds identical in observed columns, hidden contest bases permuted, resolved with the same seeds: history features equal wherever arm0 outcomes are equal.
- `test_history_maturity` — a merchant's dispute opened after `order_day` does not enter that order's hist.
- `test_history_no_nan_after_phase2` — permitted features have no NaN.
- `test_arm2_dominated` — smoke world, 5 seeds, κ = 0.6: paired (Arm 4 − Arm 2) > 0. *Failure means:* costs or uplifts wired wrong.
- `test_arm4_no_test_access` — monkeypatch test-split loader to raise; tuning completes.
- `test_arm4_paid_handoff_scope` — Arm 4 policy requests otp/signature/geotag only in Electronics/Jewellery cells (expected from arithmetic; if it fails, report — do not adjust).
- `test_paired_min_seeds` — 4 seeds → raise.
- `test_defense_only_split` — merchant_fault win rate ≤ correct_fulfillment win rate.

## Done-criteria (`check_phase_2`)
1. All Phase 0–2 tests pass.
2. `outputs/phase2/` has OUTCOME parquet for Arms 0–4 × 5 seeds × κ ∈ {0, 0.6} on the smoke world, and one 1M run (κ = 0.6, 5 seeds) for Arms 0, 1, 4; manifests with wall/RSS within limits.
3. `phase2_REPORT.md`: Arm 4 − Arm 1 net ₹/1,000 with CI (**the orchestration-layer headline**), Arm 4 − Arm 2 > 0, coverage, friction, prevention/defence split, defense-only win rates by claim class, the Arm 4 policy table, and the realised `implied_phi` from arm0 outcomes.
4. Arm 4 policy shows paid handoff items only in Electronics/Jewellery cells, or the report explains why not.

**Then halt.**
