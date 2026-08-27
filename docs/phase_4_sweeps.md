# Phase 4 — κ-sweep, κ\*, OAT and LHS sweeps, kill condition, charts

**Read first:** `docs/00_context.md` §4, §7; `docs/02_engineering_rules.md` B14; `params.yaml`
`sweep`, `report`, and every `rank`. **Produces:** `vulcan_proof/sweep/`; `outputs/phase4/` is
produced only when the optional extended-validation sweep is launched.
**Must not touch:** any model, arm, or simulator code. If a sweep exposes a bug, stop and report;
do not patch and continue. **Halt** after `python scripts\task.py check-phase 4`.

## What this phase decides

Whether the ML claim survives. The pre-build arithmetic says: at κ = 0 a tuned rule captures
99.5%+; at σ≈0.7 the oracle gap is 7–12%. This phase measures **κ\*** — the smallest κ on the grid
at which the paired Arm 5 − Arm 4 95% CI lies entirely above zero — and applies the kill
condition. Both outcomes are legitimate. The reporting code must be able to write `null`.

## Buildathon completion policy

The full approximately 270-run, 1M-order-per-run sweep is intentionally **skipped for the
buildathon**. It is optional extended validation, not a completion blocker. Do not launch it as
part of the buildathon milestone and do not create fake production tables, charts, or report values.

The buildathon milestone is complete when the sweep engine and all κ/OAT/LHS/robustness logic,
charts, tests, smoke/end-to-end runs, parameter-integrity checks, and `check_phase_4` are present
and passing. The explicit deferral must be recorded in the review documentation. No
`outputs/phase4/` production artifact is required for this track.

When publication-grade or final robustness evidence is needed, the existing driver can be run later
with the configured production parameters. That later run is expected to produce the full artifacts
and report described under **Extended-validation criteria** below.

## Optional extended-validation runs

When launched, production-scale sweep runs use `run.n_orders_sweep` (1M) and `run.n_seeds_sweep`
(5) seeds, parallelised over
seeds with `ProcessPoolExecutor(max_workers=run.parallel_workers)` from `scripts/run_phase4.py`
under `if __name__ == "__main__":`; workers take `(kappa, seed, param_overrides: dict, params_path)`
and reload params. Peak RSS per worker ≈ 2 GB at 1M orders; 4 workers ≈ 8 GB — within the 16 GB guard.
Overrides are applied to an in-memory copy of params before generation:
`evidence.customer_presence_sweep` → `presence_factor` of `otp` and `signature`;
`uplift_true.sweep_multiplier` and `ack_otp_ratio` → the uplift block (ack = ratio × otp);
`categories.rate_sweep_multiplier` → every `target_rate`.

**Everything is re-fit per point.** For each (parameter point, κ): (1) if any *funnel* parameter
is overridden (`categories.rate_sweep_multiplier`, `sim.genuine_share_target`,
`sim.false_claim_value_elasticity`, `sim.truth_base_rate.*`, `sim.kappa.type_shift`), re-run the
Phase 1 calibration for that point and write `theta.json` **inside the point's run directory** —
never overwrite `outputs/theta.json`; (2) generate orders once per (point, κ, seed) and share
across arms (B13); (3) resolve arm0, build history features; (4) re-tune Arm 4 on that world's
validation split; (5) re-train Stages A/B/C/P/W and the materialisation model on that world's
train split; (6) run Arms 1, 4, 5 (Arms 2, 3 only on the κ-grid at central parameters). A model
trained on one world and applied to another is a leak of a different kind — the sweep is over
worlds, and each world must be learned from its own history.

**Budget and resumability.** The complete extended validation is ≈ 270 runs at 1M orders (κ-grid
30, OAT 120, LHS 100, robustness 20). At ≈ 4–6 min per run with 4 workers this is 5–8 hours.
This budget is intentionally deferred for the buildathon. Each run writes its manifest last; the
sweep driver skips any point whose manifest already exists and validates, so a later run can be
interrupted and resumed. `outputs/phase4/progress.json` records completed points.

### 4.1 κ-sweep (`sweep/kappa.py`)
For κ in `sim.kappa.grid`, central parameters, 5 seeds: paired Arm 5 − Arm 4, Arm 4 − Arm 1,
Arm 5 − Arm 1. Also the Stage A Lorenz top-decile lift at each κ (from Phase 3 code, retrained per
κ — Stage A must be retrained per world because the signal changes).
**κ\*** = min κ with CI_low > 0 **and** CI_low > 0 for every larger κ on the grid (monotone
requirement; if non-monotone, κ\* = null with reason `non_monotone`).
Write `outputs/phase4/kappa_star.json`:
```
{ "kappa_star": <float|null>, "reason": "found|not_found|non_monotone",
  "table": [ {kappa, arm5_minus_arm4_mean, ci_low, ci_high, p_positive, arm4_minus_arm1_mean, lorenz_lift} ... ],
  "verdict": "ML_CLAIM_SUPPORTED_ABOVE_KAPPA_STAR" | "ML_CLAIM_DROPPED_ORCHESTRATION_ONLY" }
```
Assert at κ = 0 (B19b): `mean(Arm5 − Arm4) ≤ report.kappa0_max_gain_frac × mean(Arm4 − Arm1)`.
If violated → `LeakError("suspected leak at kappa=0")`; this is a Phase 3 bug and Phase 4 must not continue.

### 4.2 OAT sweep (`sweep/oat.py`)
Parameters with `rank ≤ sweep.oat_max_rank`, three levels each (lo, central, hi) from `sweep`,
at κ = `sim.kappa.canonical`. For uplift blocks use `uplift_true.sweep_multiplier` and
`uplift_true.ack_otp_ratio`; for latency/timeline nothing is swept. Each level: 5 seeds, Arms 1/4/5.
Output table: parameter, level, Arm 4 − Arm 1 (orchestration value), Arm 5 − Arm 4 with CI,
and the two **boundaries** from `ev_reference.threshold` (OTP break-even in Electronics; packing
break-even in Apparel) so the judge sees how the recommendation boundary moves.

### 4.3 LHS joint sweep (`sweep/lhs.py`)
`sweep.lhs_points` points over parameters with `rank ≤ sweep.lhs_max_rank`, `scipy.stats.qmc.LatinHypercube`
seeded from `SeedTree("lhs", master + lhs_seed_offset)`, uniform within each `sweep` range,
κ = canonical, 5 seeds each. Report: fraction of points where Arm 5 − Arm 4 CI_low > 0; fraction
where CI_high < 0; scatter of mean difference vs `uplift_true.sweep_multiplier` and vs `econ.hourly_rate`.

### 4.4 Robustness runs (each one point, 5 seeds, κ canonical)
- `archetypes.random_stratum_frac = 0` (pure observational history).
- `uplift_true.on_misdelivered_otp = 0.5` (carrier-fault exposure) — report carrier-fault win rate.
- `sim.deterrence = 0.2`, `sim.silence_yield = 0.10` (labelled robustness-only).
- World D: `shift_enabled = True` — report Arm 5 calibration drift (Stage A ECE on test) and paired difference.

### 4.5 Charts (`sweep/charts.py`) — every ₹ chart carries `report.simulator_footer`
1. Net ₹/1,000 vs κ: Arms 1, 3, 4, 5 with CI bands; κ\* marked (or "κ\* not found" annotation).
2. Arm 5 − Arm 4 vs κ with CI; zero line.
3. OAT tornado: Arm 5 − Arm 4 at lo/hi for each swept parameter, ordered by rank.
4. LHS scatter: difference vs uplift multiplier; vs hourly rate.
5. Recommendation-boundary chart: OTP break-even (Electronics) and packing break-even (Apparel) vs hourly rate and vs uplift multiplier, from `ev_reference.threshold`.
6. Defense-only: win rate by claim class for Arm 1 vs Arm 5 (headline and carrier-fault sweep).
7. Prevention vs defence split for Arm 4 and Arm 5.
8. Coverage and friction in physical units for Arms 4 and 5.
9. Lorenz curves at κ ∈ {0, 0.6, 1.0}.

## Tests (`tests/test_phase4.py`)
- `test_kappa_star_monotone_logic` — synthetic tables: found / not_found / non_monotone → correct output.
- `test_kappa_zero_guard_fires` — on smoke world: the κ = 0 guard raises `LeakError` when fed a leaked Arm 5 (the test builds one by replacing Stage A's output with `0.99` on rows where `hidden_dispute_potential` is true and `0.001` elsewhere — tests may read hidden columns; note `hidden_risk_mult` is identically 1 at κ = 0 and would not leak) and passes with the real Arm 5. *This test proves the guard works.*
- `test_models_refit_per_point` — two sweep points with different `uplift_true.sweep_multiplier` produce different defensibility-model artefact hashes.
- `test_point_theta_isolated` — a point that overrides a funnel parameter writes its own `theta.json` and `outputs/theta.json` is byte-unchanged.
- `test_min_seeds_enforced` — 4 seeds → raise.
- `test_orders_shared_across_arms_in_sweep` — same order_ids and truth for Arms 1/4/5 at a sweep point.
- `test_lhs_reproducible` — same seed → same design matrix.
- `test_footer_on_every_chart` — every PNG-producing function's figure text includes the footer string.
- `test_verdict_written` — `kappa_star.json` has `verdict` in the allowed set and `kappa_star` is float or null.
- `test_no_param_edits` — `params.yaml` sha256 at Phase 4 start equals sha256 at end.

## Done-criteria (`check_phase_4`)

### Buildathon implementation milestone (default)

1. All Phase 4 tests and the repository's relevant firewall/reproducibility checks pass.
2. The implemented sweep engine, κ/OAT/LHS/robustness logic, chart functions, smoke/end-to-end
   path, and parameter-integrity check are present.
3. `check_phase_4` passes while explicitly reporting that the production sweep is optional and
   deferred. Missing production artifacts are acceptable only when no partial production run is
   being presented and the deferral is documented.
4. The review documentation states that no full production sweep was launched and no production
   results were generated. Smoke results may be reported, but must be labelled as smoke validation.

### Extended-validation criteria (optional, later)

When the full production sweep is run, all of the following become required:

1. `outputs/phase4/kappa_star.json` exists, valid, with the κ = 0 guard satisfied and its value reported.
2. OAT table for all rank ≤ 12 parameters; LHS table for 20 points; four robustness runs.
3. All nine charts exist, each with footer.
4. `phase4_REPORT.md` opens with the verdict sentence — one of:
   - "κ\* = <value>. Above this signal strength the optimizer beats the tuned rule by <mean> ₹/1,000 (95% CI …). At κ = 0 the tuned rule captures <x>% of achievable value."
   - "κ\* not found on [0, 1]. The ML claim is dropped; the orchestration layer (Arm 4 − Arm 1 = <value> ₹/1,000, CI …) is the product."
   and includes the OAT tornado ranking, the LHS fractions, the robustness results, and the sentence "All ₹ figures are simulator results."

Only after these extended-validation criteria pass may the production outputs be used as final or
publication-grade robustness evidence. **Then halt.**
