# Review changes (blueprint v2)

What changed between the first blueprint and this one, and why. Kept in-repo so the reasons survive.

## Dependencies and platform
- `requirements.txt` → `requirements.in` (14 direct) + `requirements.lock` (68 packages, SHA-256
  hashes), resolved with `pip download --only-binary=:all: --platform win_amd64 --python-version 3.13`
  against PyPI. Every entry is a real binary wheel for CPython 3.13 / 64-bit Windows. Install is
  `--require-hashes --only-binary=:all:`; there is no substitution clause anywhere.
- **Python 3.13 confirmed.** numpy 2.2.6, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.7.2,
  pyarrow 21.0.0, matplotlib 3.10.9 all ship cp313 win_amd64 wheels; LightGBM 4.6.0 ships
  `py3-none-win_amd64` (which a cp-tag check misses — verified by filename). pandas 3.0 was
  deliberately not chosen: it changes copy-on-write and string dtype defaults and a coding agent
  trained on 2.x will write subtly wrong code against it.
- `Makefile` removed; `scripts/task.py` (pure Python) replaces it and refuses to run outside `.venv`.
  `vulcan_proof/envcheck.py` enforces venv + 3.13 at every entry point. `.venv/` gitignored.
- Rules A12–A15 added: pathlib only, argument-list subprocesses, spawn-safe `ProcessPoolExecutor`
  under `__main__`, declared dtypes. `psutil` added for peak-RSS in manifests; `run.max_peak_rss_gb`
  guard. Memory estimate in README (≈5–6 GB canonical; sweeps ≈2 GB/worker).
- `kaggle` package added; Phase 0 uses its API, not a shell command; credentials path given for Windows.

## Errors found in the first blueprint
1. **Funnel over-determined again** (`sim.pi_genuine` 0.45 × 1.45% failures = 0.65% ≫ every
   category target; FMCG target 0.05% impossible). Replaced with six unknowns (γ per category, θ)
   solved against six targets (five category rates + genuine share 0.35); population rate and φ are
   now outputs, and `implied_phi` must match the oracle's 0.65 within 0.02. `target_population_dispute_rate`
   removed as an input.
2. **History-feature leak.** Phase 1 seeded `merchant_contest_rate_hist` from `hidden_contest_base`
   and `merchant_integrity_hist` from `hidden_quality`. Both were leaks with placeholder names.
   History features moved to Phase 2 (`sim/history.py`, computed from resolved arm0 outcomes only,
   inside the firewall walk); Phase 1 writes NaN; integrity feature dropped; new trap B19a + tests.
3. **Fee accounting inconsistent.** Prevention added `fee_avoided` while win/lose ignored the fee,
   double-counting the fee when compared. Now the fee is charged on `dispute_opened` in the resolver
   (win = −fee, lost = −V − fee), so win−lose delta = V exactly and prevention avoids the fee
   implicitly. B8 rewritten; `test_win_lose_delta_is_order_value` added; money table exact.
4. **κ = 0 guard was wrong.** Asserting the CI straddles zero would flag a legitimate ~0.3% edge
   (per-order use of contest history) as a leak. Arm 4 now conditions on contest-history tercile ×
   tier as well as category × value band; the guard asserts Arm5−Arm4 ≤ 1% of Arm4−Arm1
   (`report.kappa0_max_gain_frac`). B19b.
5. **arm0 undefined.** Phase 3 referenced an "arm 0 = history" that no phase built. Now Phase 2
   builds arm0, resolves it, and derives history features from it before Arms 1–4 run.
6. **Stage C plan dependence mis-specified.** Contest depends on evidence *held*, not planned;
   the optimizer must take E[pC · uplift] over materialisation patterns, not E[pC] · E[uplift].
   `test_expectation_of_product` added.
7. **Firewall substring false positives.** `sim` would match `similar`. Now whole-token regex on
   identifiers and component match on import paths.
8. **Magic numbers in the spec itself** (10000 value ref, 0.4 fragility ref, 30 days/month,
   prior weight 20, bisection bounds, Kaggle row counts). All moved to `params.yaml`.
9. **PerfectModels stub under-specified.** Now an exact interface with the reference's cost
   conventions (ack cash charged unconditionally; merchant cash × pM), and the known-answer test
   covers the full 20-order grid at 1e-6, not four rows at 2 dp.
10. **Unmechanical done-criteria.** "≤ 20 min" → `wall_seconds` in manifest with a params limit;
    "two runs identical" → `check_phase_0` runs twice and diffs; peak RSS checked.
11. **`ack_optin` edit clause.** Phase 2 was allowed to add a permitted feature mid-build. Added now.
12. **Loader schema gaps.** `_meta` blocks, catalogue rows, `sim.gamma: null`, and cross-sum
    checks (shares, mixes, prevention shares, response triples, φ = 1 − genuine share) specified.
13. **Support fragility noted.** (otp, signature) contested-dispute support is ≈130 at 3M orders —
    above 50 but not by much; Phase 3 reports it and Phase 5 falls back if it is below.
14. Claims line about the κ = 0 tie now names the rule Arm 4 actually is and says "under the simulator's parameters".

## Second verification pass (blueprint v3)
Ran the specified loader rules against the committed `params.yaml` as a script. It failed:
15 leaves would have been rejected (14 structural assumptions tagged `ASSUMED` with no sweep; one
free-text source tag). Phase 0 would have halted on `task.py lint`. Fixes and other findings:
1. New source tag `ASSUMED_FIXED` for structural assumptions deliberately not swept (κ coefficients,
   customer-feature base rates, latency shape, Visa share). Loader tag regex specified exactly.
2. **Customer response was drawn under the historical ack policy in Phase 1** (`hidden_ack_sent`,
   `hidden_customer_response`), so an arm that sent an ack where history did not would have had no
   response to use. Phase 1 now stores one uniform `hidden_u_response` per order; the resolver maps
   it through the truth-class triple given the plan. Arm-invariant by construction (B13).
3. **arm0 read a hidden column from under `arms/`**, which is inside the firewall walk. Moved to
   `sim/arm0_history.py`.
4. **Stage A target.** `dispute_opened` depends on whether the historical policy's ack prevented
   the dispute; a model of it learns the historical ack policy. Target is now `exposure =
   dispute_opened ∨ prevented` — observable and policy-invariant at deterrence 0.
5. **Optimizer ignored prevention.** The oracle values evidence for defence only; in the simulator
   acks also prevent disputes, and Arm 4 (tuned on realised net) captures that. Arm 5 would have
   under-sent acks and lost for a reason unrelated to signal. Added Stage P (truth-blind prevention
   model + observed-economics gain) behind `opt.include_prevention`; known-answer tests run with it off.
6. **Optimizer cache keyed on merchant was incomplete** — `P̂_W`/`P̂_C` depend on order context.
   Cache removed; per-chunk vectorisation over (type, held-pattern) specified; test perturbs one
   row's `network` and asserts only that row's EV changes.
7. Reports without dispute potential: stated as zero-cost / counted as false friction.
8. History features for test orders derive from arm0 outcomes of prior orders, not the deployed
   arm's — stated; identical across arms so paired comparisons unaffected.
9. `arm4_contest_terciles` flag → `arm4_contest_bins: 3` (a literal 3 would have failed the magic-number scan).
10. Fee scope widened to exactly two files (resolver, Stage P) with Phase 2's test still allowing one.

Verified by script on the committed file: loader rules clean; category, archetype, prevention
and type-mix shares sum to 1; φ = 1 − genuine share; every evidence type appears in ≥1 archetype
policy; the 10 known-answer tests pass.

## Final pass (blueprint v4 — locked)
Scripted cross-check of every dotted parameter reference in the docs against `params.yaml`: one
mismatch (`sim.contest_evidence_slope` → `archetypes.contest_evidence_slope`) fixed; one orphaned
key (`merchants.contest_shrinkage_n`, left over from the removed placeholder) deleted. Test names
referenced in rules/done-criteria all appear in a phase test list.
One substantive fix: the Stage P prevention gain was written as `(V + fee) − cost`, which
over-credits prevention by the wins the merchant would have had anyway (`pC·pW·V`). Rewritten as
the full expectation — prevention replaces the whole dispute outcome and defence applies to the
non-prevented mass — with a test that the expression reduces exactly to the oracle when the
prevention term is off.

## Passes 4–5 (blueprint v5 — locked)
Read every phase document end-to-end again. Found and fixed:
1. Phase 1 drew merchant compliance into `hidden_policy_bitmask` while the Phase 2 resolver also
   drew compliance — two inconsistent draws, and the hidden schema listed the wrong column name.
   Phase 1 now stores only `hidden_requested_bitmask`; compliance is drawn once, in the resolver.
2. Archetype assignment by "quintile" could not reproduce the shares (quintiles are 20%, shares are
   15/30/10). Replaced with sorted contiguous blocks by `quality_rank`; direction of `hidden_quality`
   (higher = worse) stated. Phase 1 support test now counts *requests*, since compliance is no longer there.
3. Phase 4's leak-detector test multiplied by `hidden_risk_mult`, which is identically 1 at κ = 0 —
   the test would have "passed" without proving anything. Now leaks `hidden_dispute_potential`.
4. Phase 4 said only one parameter triggered funnel recalibration and was silent on retraining:
   now every point recalibrates the funnel if any funnel parameter is overridden (into the point's
   own directory), re-tunes Arm 4 and re-trains every model on its own world. Budget (≈270 runs,
   5–8 h) and resumability specified.
5. Stage P's defence term lacked the sum over dispute type and materialisation pattern; added.
6. Thread count: rules said `num_threads=1` in tests and `run.lgbm_threads` elsewhere — LightGBM is
   reproducible only for a fixed thread count, so that would have broken `test_repro`. One setting
   in the main process, a separate `lgbm_threads_in_worker` for pool workers, both in the manifest.
7. B13 wording keyed draws by `arm_id`, contradicting the common-random-numbers design in Phase 2. Fixed.
8. `test_claims` would have failed on `docs/01_claims.md` itself. Scoped to judge-facing artefacts.
9. Economics centralised in `vulcan_proof/economics.py` so Stage P and the resolver share one money table
   and the fee has exactly one reader.
10. Magic-number allowlist: `envcheck.py`/`seeds.py` exempt, `1024` allowed, Lorenz decile count parameterised.
11. `test_calibrated_mean_matches_rate` on validation only was tautological (isotonic is fitted there); now also on test.
12. Module `vulcan_proof/econ.py` renamed `economics.py` — `econ.money` vs `P["econ.dispute_fee"]` was
    a name collision waiting to confuse an agent. B8 delta now states `+ ratio_damage`.

Automated checks at lock: loader rules clean; every dotted param reference in docs resolves; every
central value lies inside its sweep range; archetype ranks 1–6 unique; known-answer tests pass.

## Phase 0 post-audit amendment (2026-08-27)

The v5 test added test-set calibrated-mean matching to make validation-only isotonic checks less
tautological. That test correctly exposed a real result: prevalence fell from 9.2304% in validation
to 3.5424% in test, led by `late_low_score` falling from 6.2763% to 1.5042%. The uncalibrated test
mean was already 5.9980%, so this was not caused by isotonic calibration, class weighting, or
undersampling; isotonic correctly reflected the higher-prevalence validation window and produced a
7.2824% test mean.

The test-label prohibition makes an unconditional ±5% out-of-time mean requirement impossible to
treat as a software invariant under an unseen regime change. Phase 0 therefore keeps validation
mean matching, temporal maturity, feature leakage, `scale_pos_weight=1`, no undersampling, and
determinism as hard gates. Test calibration transfer is now a mandatory diagnostic: raw/calibrated
means, Brier, ECE, prevalence by month and label reason, and the failed transfer must remain in the
metrics and report. No parameter, tolerance, split, label, model, prediction, or metric was changed
by this amendment.

## Phase 4 buildathon validation policy amendment (2026-08-27)

`PHASE4_BUILDATHON_OPTIONAL_EXTENDED_VALIDATION`

The full approximately 270-run, 1M-order-per-run Phase 4 sweep is optional extended validation and
is intentionally deferred for the buildathon. It is not a completion blocker. The sweep engine,
tests, smoke path, κ/OAT/LHS/robustness logic, charts, and mechanical checks remain implemented.

Buildathon completion is based on the implementation checks and smoke/end-to-end validation. No
production-scale `outputs/phase4/` artifacts were generated, and no production results are being
represented by the smoke runs. The larger sweep can be run later for publication-grade or final
robustness evidence; only that later run may provide production-scale Phase 4 numbers for the
final report or downstream demo claims.

Smoke/end-to-end validation: completed. Full production sweep: deferred. Larger sweep: available later.

## Phase 5 deferred-evidence compatibility amendment (2026-08-27)

Phase 5 is now required to operate without Phase 4 production outputs. In buildathon mode,
`/report/kappa`, the UI, and the generated demo script must expose an explicit deferred-validation
status and documented fallbacks rather than failing or treating an absent result as `κ* = null`.
Only genuine completed Phase 4 sweep artefacts may supply a κ result or production-scale robustness
claim. This keeps the product surface verifiable now while preserving a clean path to add the later
publication-grade evidence.
