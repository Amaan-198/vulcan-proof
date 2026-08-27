# 02 — Engineering rules, traps, and the guard for each

These are not style preferences. A silent error in this project does not crash; it produces a
plausible number that is wrong and gets presented to Razorpay engineers. Every rule below closes a
specific way that happened, or nearly happened, during design.

## A. Non-negotiable coding rules

1. **No magic numbers.** Every constant lives in `params/params.yaml` and is read through
   `vulcan_proof.params.P`. `tests/test_params.py` AST-scans `vulcan_proof/` (excluding
   `ev_reference.py` and `ui/`) for numeric literals other than `0`, `1`, `-1`, `2`, `0.5`, `100`,
   `1000`, `1024`, `3600`, `1e-9`, `1e-12` and fails on any, reporting file:line. Literals inside a
   subscript (`x[3]`) or a `shape=`/`reshape` call are exempt; `envcheck.py` and `seeds.py` are
   exempt files (interpreter version and hash width are not tunables). Tolerances belong in params too.
2. **No defaults that mask a missing value.** `P["a.b.c"]` raises `KeyError` on a missing key.
   Never `dict.get(key, default)` on parameters. Never `argparse` defaults for anything numeric.
3. **Assert, never warn.** Invariant violations raise. Use `assert` only in tests; in library code
   use `raise InvariantError(...)` (defined in `vulcan_proof/errors.py`) so `python -O` cannot
   strip it.
4. **Fail loudly on anything unexpected.** Unknown column → raise. Unknown category → raise.
   NaN in a probability → raise. Empty split → raise. Extra keys in a parameter block → raise.
5. **Deterministic seeding everywhere.** One `SeedSequence(master_seed)`; each module receives a
   child via `spawn`. Never `np.random.seed`, `random.seed`, or the global RNG. LightGBM gets
   `seed=`, `deterministic=True`, `force_row_wise=True`, and an explicit `num_threads` from params.
   LightGBM is reproducible only for a fixed thread count, so the thread count is part of the
   reproducibility identity and is recorded in every manifest: `run.lgbm_threads` in the main
   process (tests included), `run.lgbm_threads_in_worker` inside pool workers.
6. **Pure functions over state.** Generators and models take explicit inputs and return frames.
   No module-level mutable state except the loaded `P`.
7. **Frames, not dicts.** All data moves as pandas DataFrames with a declared schema
   (`vulcan_proof/schemas.py`). Every function validates its input schema on entry
   (`schemas.check(df, SCHEMA_NAME)`) — column names, dtypes, no extras.
8. **Parquet only** for artefacts. Every artefact is written with `write_artifact(df, run_dir, name)`
   which also appends its SHA-256 to `manifest.json`.
9. **No `try/except` that swallows.** Catch only to add context and re-raise.
10. **Type hints on every public function.** `mypy --strict` is not required, but signatures must exist.
11. **Tests are part of the phase.** A phase whose tests do not pass is not done.
12. **Windows-native.** All paths via `pathlib.Path`; never string concatenation or `/` literals.
    All subprocesses as argument lists (`subprocess.run([...])`), never `shell=True`. File I/O with
    explicit `encoding="utf-8"`. No symlinks. No `os.fork`, no `signal.SIGALRM`.
13. **Spawn-safe parallelism.** Parallel sections use `concurrent.futures.ProcessPoolExecutor`
    (`max_workers = run.parallel_workers`) and are invoked only from a function called under
    `if __name__ == "__main__":` in `scripts/run_phase*.py`. Worker functions are module-level and
    receive only picklable arguments (seed ints, κ floats, paths) — never a `Params` object, never
    a model; workers reload params from the path. LightGBM inside workers uses `num_threads = run.lgbm_threads_in_worker`
    and the pool size is `run.parallel_workers`; outside workers `num_threads = run.lgbm_threads`.
14. **Venv only.** Every entry point calls `vulcan_proof.envcheck.require_venv()` first, which
    exits unless `sys.prefix` is under `<repo>/.venv` and `sys.version_info[:2] == (3, 13)`.
15. **Declared dtypes.** Every schema column has an explicit dtype; frames are cast on creation
    (`int32` ids, `float32` values, `category` for enums, `int8` flags). Memory is a correctness
    concern here because a silent float64 default doubles the EV matrix.

## B. Traps and guards

Each trap names a plausible implementation choice that would quietly corrupt results, and the
mechanical guard that catches it.

### B1. Truth leaking into the optimizer
**Trap:** joining the truth column for "debugging", passing the full order frame to a model, or
computing `P(truth | order)` in the optimizer because the original design's equation had it.
**Guard:** (a) `tests/test_firewall.py` walks the AST of every file under `vulcan_proof/opt/`,
`vulcan_proof/models/`, `vulcan_proof/arms/`, and `vulcan_proof/sim/history.py`, and fails if any
`Import`/`ImportFrom` module path has a component equal to `sim`, `resolve`, `truth`, or
`generator` (allowed imports under the walk: `vulcan_proof.params`, `.schemas`, `.errors`, `.seeds`, `.manifest`, `.econ`, `.ev_reference` is NOT allowed under `opt/`/`models/`), or if any `Name` or
`Attribute` identifier matches the regex `^(hidden|truth|uplift_true|gamma|theta)(_|$)` — whole-token
match, so `similar`/`simple` are not false positives and `hidden_z_risk` is caught; (b) `schemas.check(df, "ORDER_OBSERVED")`
is called at the entry of every model `fit`/`predict` and every optimizer call, and the schema
lists exactly `params.yaml: features.permitted` — an extra column raises; (c) truth columns are
prefixed `hidden_` and `write_artifact` refuses to write any frame with a `hidden_` column into
`outputs/<run>/observed/`.

### B2. Forbidden post-payment features
**Trap:** delivery timestamp, ack outcome, compliance on this order, evidence captured, or dispute
outcome used as Stage A/B/C features because they are "available in the frame".
**Guard:** `features.permitted` is an explicit allowlist. `features.forbidden` is also listed and
`tests/test_firewall.py` asserts the two are disjoint and that no forbidden name appears in any
`ORDER_OBSERVED` frame.

### B3. Label maturity and censoring
**Trap:** labelling a dispute as "no dispute" because its resolution had not occurred by the split
boundary; or silently dropping late disputes so training skews to fast, cheap ones.
**Guard:** every order row carries `decision_date`, `dispute_open_date` (nullable),
`resolution_date` (nullable). A label may be used only if `resolution_date <= observation_boundary`
for the split, **or** `dispute_open_date is null and order_day + expected_delivery_days + dispute_max_days +
response_days + resolution_p95_days <= observation_boundary` (the full possible window). Rows failing both are `censored = True` and are **excluded** from labels.
`tests/test_phase1.py::test_censoring_excluded_not_negative` asserts no censored row has label 0.
The censoring fraction per split is written to `manifest.json` and must be ≤ `sim.max_censor_frac`.

### B4. The intercept-correction trap
**Trap:** undersampling negatives or using `scale_pos_weight` for a 0.3% positive rate, then
feeding the inflated probabilities into EV arithmetic. AUC stays perfect; every ₹ is wrong by the
sampling ratio.
**Guard:** undersampling is **forbidden** (3M × 0.3% is fine for LightGBM). `scale_pos_weight`
must be 1. After isotonic calibration on validation, `tests/test_phase3.py::test_calibrated_mean`
asserts `abs(mean(p_cal) − empirical_rate) / empirical_rate < params.calib.mean_tolerance` on the
validation split, for Stage A, each Stage B class, and Stage C.

### B5. Support mask
**Trap:** the optimizer buys an evidence type whose yield was estimated from 12 disputes, or
learns an interaction from a bitmask with no support.
**Guard:** after Phase 3 training, compute contested-dispute support for every (type, evidence)
pair and every bitmask. Pairs with support < `models.support_min` are masked out of the action
space; bitmasks with support < `models.support_min` shrink to main effects with
`confidence = support / support_min`. `tests/test_phase3.py::test_support_mask_applied` builds a
tiny training set with one evidence type absent and asserts the optimizer never selects it.

### B6. Cash cost timing
**Trap:** charging OTP's ₹25 on request. Merchants who ignore recommendations would then be
charged for nothing, and low-compliance merchants look worse than they are.
**Guard:** in the simulator, cash is debited only when `complied = True`. In the optimizer,
cash is multiplied by `P̂_M`. Merchant seconds are charged on request in both.
`tests/test_ev_reference.py::test_cash_on_compliance` pins the reference's cost convention;
`tests/test_phase3.py::test_known_answer_perfect_models` holds the optimizer to it; and
`tests/test_phase2.py::test_cash_on_compliance` holds the resolver to it.

### B7. Subset vs item optimisation
**Trap:** recommending every evidence type whose standalone EV > 0. At ₹45k/2× this buys OTP
(standalone +₹4.85) which is wrong by ₹1.27 because geotag + ack already cover the uplift.
**Guard:** the optimizer enumerates all admissible subsets (≤ 512) and picks the argmax of
`EV̂(E)`. `tests/test_phase3.py::test_subset_not_item` uses the known-answer case
(Electronics, ₹45,000, risk 2×) and asserts OTP is not in the chosen set while geotag and ack are.

### B8. Dispute fee in the wrong branch
**Trap:** `loss_avoided = order_value + fee`; or charging the fee only on loss.
**Guard:** the resolver charges `econ.dispute_fee` on `dispute_opened`, before win/lose is known
(`value(won) = −fee`, `value(lost) = −order_value − fee − ratio_damage`), so the win–lose delta
is exactly `order_value + ratio_damage` (ratio_damage = 0 in the headline). `tests/test_phase2.py::test_win_lose_delta_is_order_value` asserts it on 1,000
resolved rows. `tests/test_firewall.py::test_fee_scope` asserts the token `econ.dispute_fee`
occurs in exactly one file, `vulcan_proof/economics.py` (function `money(...)` and `prevention_gain(...)`),
and that only `sim/resolve.py` and `models/prevention.py` call those two functions.

### B9. Over-determined dispute generation
**Trap:** setting truth base rates, category dispute rates, and false-claim share as three inputs.
**Guard:** six unknowns (γ per category, θ) solved against six targets (five category rates,
one genuine-share) on expected probabilities; population rate and φ are outputs.
`tests/test_phase1.py::test_category_rates_within_tolerance` and `test_genuine_share_and_phi`
assert the realised values; `implied_phi` must land within 0.02 of `reference.phi` or the oracle
and the simulator disagree about the world.

### B10. Compliance cancelling
**Trap:** dividing an OTP threshold by compliance "to correct for it".
**Guard:** thresholds are never computed in the optimizer; it computes EV. `ev_reference.threshold`
is the only threshold function and is used in tests only.

### B11. Wrong-recipient OTP
**Trap:** giving OTP a partial positive uplift on misdelivered orders in the headline run.
**Guard:** `sim.uplift_misdelivered_otp = 0` in the headline; sweep value is a separate key.
`tests/test_phase1.py::test_headline_misdelivered_zero`.

### B12. Contest independent of evidence
**Trap:** modelling Stage C as `P(contest | merchant)` only.
**Guard:** Stage C takes the planned evidence bitmask as a feature; the simulator's contest
propensity increases with `held_evidence_count` by `archetypes.contest_evidence_slope`.
`tests/test_phase3.py::test_stage_c_uses_plan` asserts predictions differ for ∅ vs full plan.

### B13. Seed reuse across arms
**Trap:** re-drawing orders per arm, so Arm 5 − Arm 4 includes order-sampling noise.
**Guard:** orders and truth are generated once per (world, seed); arms consume the same parquet.
All resolver draws (compliance, presence, response, contest, win) use `spawn`ed children keyed by
`(seed, purpose, order_id)` — **never by arm** — so identical decisions produce identical outcomes
across arms and differences are attributable to the plan alone (common random numbers).
`tests/test_phase2.py::test_arms_share_orders` asserts identical `order_id` sets and truth columns.

### B14. Single-seed conclusions
**Trap:** reporting Arm 5 − Arm 4 from one seed.
**Guard:** `report.min_seeds = 5`; any aggregate reporting function raises if given fewer.
Every reported difference carries mean, 95% CI (t-distribution on paired seed differences), n_seeds.

### B15. Time-leakage in tuning Arm 4
**Trap:** grid-searching Arm 4's per-band subsets on the test split.
**Guard:** Arm 4 tuning reads only the validation split; `tests/test_phase2.py::test_arm4_no_test_access`
monkeypatches the test-split loader to raise during tuning.

### B16. Stage B renormalisation
**Trap:** per-class isotonic calibration leaving class probabilities that do not sum to 1.
**Guard:** renormalise after per-class calibration; assert row sums within 1e-9.

### B17. LightGBM non-determinism
**Trap:** multithreaded histogram building giving run-to-run drift that masks real bugs.
**Guard:** `deterministic=True, force_row_wise=True, num_threads=P["run.lgbm_threads"]`, and
`tests/test_repro.py` trains twice from the same seed and asserts identical predictions.

### B18. Olist label leakage
**Trap:** using review score, delivery date, or order status as a *feature* in Phase 0.
**Guard:** Phase 0 feature allowlist is `olist.features.permitted`; label-derived columns are in
`olist.features.forbidden`; same disjointness test as B2.

### B19a. History features seeded from hidden state
**Trap:** filling `merchant_contest_rate_hist` (or any `*_hist`) from `hidden_contest_base` or
`hidden_quality` "until real history exists".
**Guard:** `sim/history.py` is inside the firewall walk (B1) and may import only `schemas`,
`params`, `errors`; it consumes an OUTCOME frame and dates. `tests/test_phase2.py::test_history_from_outcomes_only`
generates two worlds identical in observed columns but with hidden contest bases permuted, resolves
arm0 with the **same** seeds, and asserts the history features differ only where realised outcomes differ.
Phase 1 writes `*_hist` as NaN (nullable); Phase 3 raises on NaN in any permitted feature.

### B19b. κ = 0 tie enforced as exact zero
**Trap:** asserting Arm 5 − Arm 4 CI straddles zero at κ = 0 — a legitimate ~0.3% edge from
per-order use of contest history would then be reported as a leak, or the implementer would
"fix" Arm 5 to make it tie.
**Guard:** at κ = 0 assert `mean(Arm5 − Arm4) ≤ report.kappa0_max_gain_frac × mean(Arm4 − Arm1)`
(1%). Above that, raise `LeakError`. Below, report the value.

### B19c. Per-item shortcut inside the subset loop
**Trap:** pruning subsets by dropping any type whose standalone EV < 0 "to save time".
**Guard:** forbidden — a type with negative standalone EV can be positive inside a set only when
ρ < 0 (never), but a type with *positive* standalone EV can be negative inside a set (OTP at
₹45k/2×). Enumerate all subsets of the support-masked admissible set; the reference does the same.
`test_known_answer_perfect_models` catches deviation.

### B19. Parameter drift between reference and implementation
**Trap:** `ev_reference.py` and `params.yaml` disagreeing after an edit.
**Guard:** `tests/test_ev_reference.py::test_reference_matches_params` loads both and asserts every
shared constant is equal. `ev_reference.py` is frozen; if `params.yaml` changes, the test tells you.

## C. Halt protocol for coding agents

When the phase's `python scripts\task.py check-phase <n>` passes: write `outputs/phase<n>_REPORT.md` containing the
done-criteria table with pass/fail, any unexpected observation, and the
manifest path. Then **stop**. Do not read the next phase document. Do not refactor other phases'
code. If a done-criterion cannot be met, stop, report the failure and the exact assertion, and
do not "fix" it by relaxing a tolerance in `params.yaml`.
