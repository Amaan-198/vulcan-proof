# Phase 3 — Stages A/B/C, materialisation, defensibility, support mask, optimizer (Arm 5)

**Read first:** `docs/00_context.md` §2, §3, §4 (every finding); `docs/02_engineering_rules.md`
B1, B2, B4, B5, B7, B10, B12, B16, B17, B19. **Produces:** `vulcan_proof/models/`,
`vulcan_proof/opt/`, `vulcan_proof/arms/arm5.py`, `tests/test_firewall.py`, `tests/test_repro.py`.
**Must not touch:** `sim/`, `arms/arm1..4`, `report/` except to add Arm 5 as a consumer. **Halt**
after `python scripts\task.py check-phase 3`.

## The rule that overrides every other consideration

`vulcan_proof/models/` and `vulcan_proof/opt/` are **truth-blind**. They import nothing from
`vulcan_proof/sim/`. They receive `ORDER_OBSERVED` frames and *observed* outcome labels from
`OUTCOME` (columns `dispute_opened, dispute_type, contested, won, materialised_bitmask, complied,
censored, split`) — never `hidden_*`, never truth. `tests/test_firewall.py` enforces this by AST
walk. If you find yourself needing truth to make something work, stop and report; the design says
you must not have it.

## Labels — maturity rule (trap B3)
A label from OUTCOME may be used for training/validation **only if `censored == False`**.
Build `labels.py::eligible(outcome_df, split) -> mask` and call it everywhere. Training on the
`gap` or `test` split is forbidden; `labels.eligible` raises if asked for them in `fit`.

Training data is `OUTCOME_arm0` from Phase 2 (the resolved historical policy) joined to the
completed `observed_orders` (history features filled). Do not re-resolve anything here.

## Stage A — `models/stage_a.py`
Binary target **`exposure = dispute_opened ∨ prevented`** (`opt.stage_a_target`) on train
(eligible rows). Why not `dispute_opened`: whether a dispute *opens* depends on whether the
historical policy sent an ack that prevented it; a model of `dispute_opened` would learn the
historical ack policy and be wrong the moment the deployed policy changes ack coverage. Exposure
is observable (both branches are recorded) and, with deterrence = 0, invariant to the evidence
policy. `tests::test_stage_a_target_is_exposure` asserts the label column. Features = `features.permitted` minus ids and
`decision_date`. LightGBM `models.lgbm.stage_a`; `num_threads = run.lgbm_threads`; seed from
`SeedTree("stage_a", seed)`. **No undersampling. `scale_pos_weight` must equal 1** (assert on the
params dict). Early stopping on validation logloss. Isotonic calibration on validation. Assert
calibrated mean within `models.calib.mean_tolerance` of validation rate (trap B4) and ECE ≤
`models.calib.max_ece`. Output: PR-AUC, Brier, ECE, reliability table, Lorenz table (share of
disputes per score decile; top-decile lift). Lorenz is **reported before Arm 5 runs** and must be
in `phase3_REPORT.md`; if top-decile lift < `report.lorenz_top_decile_min_lift` at the canonical κ,
write the sentence "Stage A signal is below the level at which per-order optimisation was expected
to pay; see Phase 4 κ\*."

## Stage B — `models/stage_b.py`
3-class `dispute_type` on eligible train rows with `dispute_opened` (prevented rows have no type).
This is the right conditional: the defence term applies to the non-prevented mass, whose type
distribution is `P(type | opened)`. Noted in the report. Per-class isotonic on
validation, then **renormalise rows to sum 1; assert |Σ − 1| < 1e-9** (trap B16). Report per-class
precision/recall/F1, confusion, per-class calibrated mean vs rate.

## Stage C — `models/stage_c.py`
Binary `contested` on eligible train rows with `dispute_opened`. Features = permitted features
**+ `materialised_bitmask` as 9 binary columns** (trap B12: contest depends on evidence held).
For training, the historical materialised bitmask. At planning time the optimizer does not know
what will materialise; it evaluates `pC` at the *expected* held set by enumerating materialisation
patterns exactly as it does for `P̂_W` (the same 2^|E| loop — compute `pC` and `P̂_W` per pattern
and take the expectation of their product with the uplift, not the product of expectations). Isotonic on validation; assert
calibrated mean. `tests::test_stage_c_uses_plan`: predictions for ∅ vs all-nine differ on every row.

## Stage P — prevention model (`models/prevention.py`) — `opt.include_prevention`
The frozen oracle values evidence for *defence* only. In the simulator, acknowledgements also
*prevent* disputes, and Arm 4 — tuned on realised net ₹ — captures that value automatically. An
Arm 5 that ignored prevention would under-send acks and lose to Arm 4 for a reason that has
nothing to do with signal. So the optimizer carries a second, truth-blind term:
```
P̂_prev(ack kind ∈ E | merchant, ctx) = P(prevented | exposure, that ack sent)   — LightGBM on history rows where an ack was sent; 0 if no ack in E
D(E)   = defence value of a dispute given E, per unit exposure, summed over type and materialisation pattern h:
         = Σ_d pB(d) · Σ_h P(h | E, pM) · { pC(d,h) · [ pW(d,h)·(−fee) + (1 − pW(d,h))·(−V − fee) ] + (1 − pC(d,h))·(−V − fee) }
         = −V − fee + Σ_d pB(d) · Σ_h P(h | E, pM) · pC(d,h) · pW(d,h) · V
Pv(ctx) = −Σ_mode share_mode · prevention_cost_mode(ctx)                 (econ params; cogs is a category param — observed)
EV̂(E)  = pA · [ P̂_prev · Pv + (1 − P̂_prev) · D(E) ]  −  pA · D(∅)  −  cost(E)
```
This is the full truth-blind expectation: prevention replaces the *whole* dispute outcome
(including the fee and the wins the merchant would have had anyway), and the defence term applies
only to the non-prevented mass. With `include_prevention = false`, `P̂_prev ≡ 0` and the expression
reduces algebraically to the oracle's `pA·pC·[pW(E) − pW(∅)]·V − cost(E)` — `test_reduces_to_oracle`
asserts this identity numerically on the 20-order grid. Mode shares and cost formulas come from
`vulcan_proof/economics.py` (Phase 2), the same pure functions the resolver uses — `models/` may import
`vulcan_proof.economics`; the firewall allows it because it contains no hidden state.
Only one ack kind counts (vack supersedes ack if both present). The fee enters through `economics.prevention_gain(...)` / `economics.money(...)`; Stage P never reads
`econ.dispute_fee` directly (`test_fee_scope`).
Known-answer tests run with `include_prevention = false` so they match the oracle; a separate
test `test_prevention_term_positive` asserts that with it true, the ack's EV on Electronics ₹45k
is strictly larger than the oracle's.

## Materialisation model — `models/materialisation.py`
For each evidence type `e`: `P̂_M(e | merchant features, tier)` = observed rate of
`materialised_e` among rows where `requested_e`, as a LightGBM binary or, if support <
`models.materialisation_min_support` for a type, the merchant-archetype-blind population rate
with a flag. Acks: response rate model among sent. Report per-type rates.

## Defensibility model — `models/defensibility.py`
Target `won` on eligible train rows with `contested`. Features: `dispute_type`, the 9
`materialised_e` binaries (bitmask), `network`, `issuer_family`, `category`, `order_value` (log),
merchant features. LightGBM `stage_w`. Isotonic on validation (contested rows).

**Support (trap B5).** From the eligible training rows (contested, not censored):
- `support_pair[(d, e)]` = count of contested disputes of type `d` with `materialised_e`.
- `support_mask[(d, e)]` = `support_pair ≥ models.support_min`.
- `support_bitmask[(d, mask)]` = count for each realised bitmask.

**Prediction with shrinkage.** For a query `(d, mask, ctx)`:
```
p_full  = model(d, mask, ctx)
p_main  = base(d, ctx) + Σ_e∈mask main_effect[d, e]      # main_effect from single-evidence rows vs none, same ctx bucket
conf    = min(1, support_bitmask[(d, mask)] / support_min)
P̂_W     = conf · p_full + (1 − conf) · p_main
```
`base(d, ctx)` = model prediction at mask = ∅. Report MAE of the learned single-type uplift `P̂_W(d, {e}) − P̂_W(d, ∅)` (averaged over test-split
contested rows of type `d`) against `implied_phi × uplift_true[d][e] × sweep_multiplier`
(the truth-marginal expectation a perfect truth-blind model would learn) — **in the test harness
only** (`tests/` may read `uplift_true` and `theta.json`; library code may not).

## Optimizer — `opt/optimizer.py` (trap B7, B10)
```
def plan(order_row, models, masks) -> bitmask:
    adm   = admissible types for any d with P̂_B(d) > 0, intersected with availability (tier, opt-in)
    adm  &= {e : support_mask[(d, e)] for some admissible d}       # support mask
    best = (0.0, 0)                                                  # EV of empty plan is exactly 0
    for E in all subsets of adm (≤ 512):
        ev = Σ_d pA·pB(d)·pC(d, E)·[ E_mat[ P̂_W(d, mat(E)) ] − P̂_W(d, ∅) ]·V
             − Σ_e cash_e·P̂_M(e) − Σ_e sec_e·hourly/3600
        # E_mat: expectation over independent materialisation of each e ∈ E using P̂_M — enumerate 2^|E|
        # (cap |E| ≤ 9 so 2^9 patterns; memoise P̂_W calls per (d, mask))
        if ev > best[0] + 1e-9: best = (ev, E)
    return best[1]
```
`pC(d, E)` is Stage C evaluated with `planned_bitmask = E`. No thresholds anywhere. No per-item
shortcuts. `ev_reference.best_subset` is the known-answer twin of this function when every `P̂`
equals its population parameter. `tests/test_phase3.py` builds a `PerfectModels` stub with exactly
these behaviours, and the optimizer must accept any object with this interface:
```
pA(row)                      -> categories.<cat>.target_rate × risk      (risk supplied by the test)
pB(row)                      -> categories.<cat>.mix                     (dict NR/NAD/EB)
pC(row, held_mask)           -> reference.pc_population                  (constant; slope ignored)
pM(row, e)                   -> compliance_population × presence_factor  (system-sent: presence only)
pW(row, d, held_mask)        -> base_win[d] + reference.phi × ev_reference.set_uplift(d, held)
support_mask                 -> all True
```
With this stub the optimizer's plan and its per-type standalone/incremental EVs must equal
`ev_reference.best_subset` / `ev_set` to 1e-6 on the full 20-order grid in
`docs/appendix_arithmetic.md` §3, not only the known-answer rows. Costs: merchant items
`cash × pM`; system-sent acks `cash × 1` (the reference charges ack cash unconditionally when
requested; match it).

Arm 5 = `arms/arm5.py`: `plan(observed_orders)` applies the optimizer in chunks of
`run.ev_chunk_orders` rows. **No cross-row caching of `P̂_W` or `P̂_C`** — both depend on
order-level context (category, value, network, issuer), so any cache keyed on merchant is
incomplete by construction. Instead, vectorise: for each dispute type `d` and each *held pattern*
`h` that any subset can produce (≤ 512), call `P̂_W(d, h, chunk_rows)` and `P̂_C(d, h, chunk_rows)`
once on the whole chunk (≤ 3 × 512 predict calls per chunk), then assemble `U[row, subset] =
Σ_d pB·Σ_h P(h | subset, pM)·pC(d,h)·[pW(d,h) − pW(d,∅)]` from those arrays. The EV matrix is
`float64`, chunk × subsets, freed per chunk. The plan is the argmax per row; EV of ∅ is 0.
`test_no_cross_row_cache`: perturb one row's `network`; only that row's EV changes.

Plan explanation (for Phase 5): the optimizer also returns, per order, the standalone EV and the
incremental EV of every admissible type, and the reason code for refusals:
`INADMISSIBLE | UNAVAILABLE | NO_SUPPORT | NEGATIVE_STANDALONE | NEGATIVE_INCREMENTAL`.

## Tests
`tests/test_firewall.py`
- `test_no_truth_symbols_in_opt_models_arms` — AST walk; any Name/Attribute/Import containing
  `truth`, `hidden`, `uplift_true`, `sim`, `generator` under `opt/`, `models/`, `arms/` fails, listing file:line.
- `test_schema_gate_on_fit_predict` — every model `fit` and `predict` raises `SchemaError` on a frame with `hidden_z_risk`.
- `test_permitted_forbidden_disjoint`.
- `test_fee_scope` (re-run from Phase 2).
`tests/test_repro.py`
- `test_lgbm_deterministic` — fit Stage A twice, same seed → identical predictions (B17).
- `test_arm5_reproducible` — same seed → identical plans.
`tests/test_phase3.py`
- `test_calibrated_mean` — Stage A, each Stage B class, Stage C, defensibility: |mean − rate|/rate < tolerance on validation (B4).
- `test_scale_pos_weight_is_one` — params dict assert.
- `test_stage_b_rows_sum_one`.
- `test_stage_c_uses_plan`.
- `test_support_mask_applied` — smoke world with `archetypes` patched so `signature` is never requested → optimizer never selects signature; reason code `NO_SUPPORT`.
- `test_known_answer_perfect_models` — with `PerfectModels` and `opt.include_prevention = false`: for every (category, value, risk) in the 20-order grid × {1, 2, 4}, plan == `ev_reference.best_subset` and all standalone/incremental EVs match `ev_reference` to 1e-6; plus the named cases: Electronics ₹45k 2× excludes otp with `NEGATIVE_INCREMENTAL`; ₹45k 4× includes otp, excludes signature; Jewellery ₹200k 1× refuses signature; Apparel ₹3,500 1× → empty plan. *Failure means:* the optimizer's arithmetic differs from the reference — fix the optimizer, never the reference.
- `test_no_cross_row_cache` — see Arm 5 above.
- `test_stage_a_target_is_exposure`.
- `test_prevention_term_positive` — see Stage P.
- `test_reduces_to_oracle` — see Stage P.
- `test_expectation_of_product` — a case where pC varies with held set: optimizer result equals brute-force E[pC·uplift] over patterns, not E[pC]·E[uplift].
- `test_no_nan_features` — a NaN in any permitted feature at fit or predict → `SchemaError`.
- `test_subset_not_item` — the same ₹45k 2× case; a per-item variant (provided in the test as a foil) picks otp; the real optimizer does not.
- `test_empty_plan_ev_zero` — EV of ∅ is exactly 0.0.
- `test_uplift_mae_reported` — test harness computes MAE of learned vs true uplift on smoke world; asserts it is written to metrics; no threshold (report only).
- `test_no_training_on_gap_or_test` — `labels.eligible(..., "test")` in fit context raises.

## Done-criteria (`check_phase_3`)
1. All tests Phase 0–3 + firewall + repro pass.
2. `outputs/phase3/metrics.json`: Stage A PR-AUC/Brier/ECE, Lorenz top-decile lift; Stage B per-class; Stage C calibration; defensibility MAE vs true uplift; support tables (pairs masked, bitmasks with support ≥ 50 — count reported); per-type materialisation rates.
3. Arm 5 run on smoke world, 5 seeds, κ ∈ {0, 0.6}; paired Arm 5 − Arm 4 with CI reported. **At κ = 0, `mean(Arm5 − Arm4) ≤ report.kappa0_max_gain_frac × mean(Arm4 − Arm1)`** (B19b). If violated: stop, do not proceed, report which firewall test should have caught it.
4. Arm 5 on the 1M canonical world, κ = 0.6, 5 seeds; paired vs Arm 4 reported.
5. Refusal reason-code distribution reported.

**Then halt.**
