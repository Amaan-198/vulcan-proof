# Phase 0 — Infrastructure + Olist real-data anchor

**Read first:** `docs/00_context.md`, `docs/02_engineering_rules.md`, `params/params.yaml`.
**Produces:** the shared infrastructure every later phase imports, and the only real-data number in
the project. **Must not touch:** anything under `vulcan_proof/sim/`, `arms/`, `models/`, `opt/`
(they do not exist yet and must not be created here). **Halt** after `python scripts\task.py check-phase 0` passes
and `outputs/phase0_REPORT.md` is written.

## Why this phase exists

Track 02 asks for measured precision/recall on a held-out set. Every ₹ in this project is
simulated. Olist (public, ~100k Brazilian e-commerce orders, 2016–18) has purchase-time features,
actual vs estimated delivery, and customer reviews — enough to define a *fulfillment complaint*
label and measure whether a Stage-A-shaped detector generalises on real orders. It has no
chargebacks and no evidence. The claim is "detection generalises"; nothing more.

## Part A — Infrastructure

### A1. `vulcan_proof/params.py`
- `load(path) -> Params`. Parses YAML. Walks every leaf. A leaf is a dict with keys exactly
  `{value, unit, source, sweep, rank}` **or** a catalogue row (evidence/category/archetype rows,
  which have their own fixed key sets — define them as frozen sets and check exactly).
- Leaf kinds, checked exactly: (i) **parameter leaf** = dict with keys `{value, unit, source, sweep, rank}`;
  (ii) **catalogue row** under `evidence.<name>` = keys `{cash, seconds, presence_factor, system_sent, window, admissible, api_slot}`;
  under `categories.<name>` = `{share, target_rate, mix, vmin, vmax, cogs, fragility}`; under
  `archetypes.<name>` = `{share, compliance, contest, quality_rank, policy}`; under `uplift_true.<NR|NAD|EB>`
  = a flat `{evidence: float}` dict; (iii) **`_meta`** = keys `{source, sweep}`. Anything else raises.
- Source tags: the `source` string must start with one of `SPEC`, `CITED`, `ASSUMED`, `ASSUMED_FIXED`,
  `DERIVED` (regex `^(SPEC|CITED|ASSUMED_FIXED|ASSUMED|DERIVED)(\b|:| —)`; test `ASSUMED_FIXED` before `ASSUMED`).
- Rejects: missing field, extra field, tag `ASSUMED` (not `ASSUMED_FIXED`) with `sweep: null`, `rank`
  non-null when `sweep` is null, duplicate keys (strict YAML loader), any leaf `value: null` except
  `sim.theta` and `sim.gamma` (DERIVED; read from `outputs/theta.json` via `P.derived("theta")`).
  `tests/test_params.py::test_lint_passes` must pass on the committed file — if it does not, the
  file is wrong, not the rule; report rather than relax.
- Cross-checks: `categories.*.share` sum to 1 ± 1e-9; `archetypes.*.share` likewise; each
  `categories.*.mix` sums to 1; `econ.prevention.share_*` sum to 1; every `customer_response` triple sums to 1;
  `merchants.tier_full + tier_post_delivery_only ≤ 1`; every `evidence.*.admissible` ⊆ {NR, NAD, EB};
  `features.permitted ∩ features.forbidden = ∅`; `olist.features` likewise;
  `reference.phi == 1 − sim.genuine_share_target` (1e-9).
- `P["econ.hourly_rate"]` returns the `value`. `P.meta("econ.hourly_rate")` returns the leaf dict.
  Missing key → `KeyError` with the full path. No `get`.
- `P.sha256` = hash of the file bytes.
- `python -m vulcan_proof.params --lint params/params.yaml` runs the checks and exits non-zero on failure.
- `P.derived_pc_population()` and `P.derived_compliance_population()` compute the archetype-weighted
  contest rate and compliance and assert they equal `reference.pc_population` and
  `reference.compliance_population` to 1e-6.

### A1b. `vulcan_proof/envcheck.py`
`require_venv()`: exits with a message unless `sys.version_info[:2] == (3, 13)` and
`Path(sys.prefix).resolve()` is `<repo>/.venv` or below. Called first in every `scripts/*.py` and
in `vulcan_proof/params.py`'s `__main__`.

### A2. `vulcan_proof/errors.py`
`class VulcanError(Exception)`, `InvariantError`, `SchemaError`, `LeakError` (subclasses). Library
code raises these; never `assert`.

### A3. `vulcan_proof/schemas.py`
Named schemas as ordered `dict[str, dtype]`. At minimum in this phase: `OLIST_ORDER_FEATURES`
(from `olist.features.permitted`), `OLIST_LABELS`. Later phases add `ORDER_OBSERVED`
(= `features.permitted`), `ORDER_HIDDEN`, `OUTCOME`, `PLAN`.
`check(df, name, allow_extra=False)`: exact column set (extras raise `SchemaError`), dtype match,
no NaN in non-nullable columns (nullable columns are declared with `"nullable"` suffix in the
schema value, e.g. `"float64:nullable"`).

### A4. `vulcan_proof/manifest.py`
`start_run(phase: str, params: Params) -> RunContext` creates `outputs/<phase>_<utc-ts>_<seed>/`
(all `pathlib`), writes `manifest.json` with: git commit (`subprocess.run(["git","rev-parse","HEAD"])`,
raise if `git status --porcelain` is non-empty unless `--allow-dirty`), params sha256, master seed,
phase, timestamp, python version, installed versions via `importlib.metadata` (not `pip freeze`),
hostname, and — on `finish_run(ctx)` — `wall_seconds` and `peak_rss_mb` (`psutil.Process().memory_info().peak_wset`
on Windows; `resource` elsewhere). `check_phase` fails if `peak_rss_mb > run.max_peak_rss_gb × 1024`.
`write_artifact(ctx, df, name)` writes `<name>.parquet`, appends `{name, sha256, n_rows}` to the
manifest. Refuses (raises `LeakError`) any frame containing a column starting with `hidden_` when
`name` starts with `observed_`.

### A5. `vulcan_proof/seeds.py`
`SeedTree(master_seed)` with `.child(*labels) -> np.random.Generator`. Labels are strings or ints
(ints are stringified); the child seed is `SeedSequence(master_seed, spawn_key=(hash_of_labels,))`. Deterministic: same labels →
same generator state. Document that `hash_of_labels` is `int.from_bytes(sha256("|".join(labels)))`
truncated to 8 bytes — never Python's `hash()`.

### A6. `scripts/check_phase.py`
Registry `CHECKS: dict[int, Callable[[], list[tuple[str, bool, str]]]]`. Prints a table, exits 1
on any `False`. Implements `check_phase_0`.

### A7. Tests that ship with Part A
- `tests/test_params.py`
  - `test_lint_passes` — the committed file lints.
  - `test_missing_key_raises` — `P["does.not.exist"]` → `KeyError`.
  - `test_assumed_requires_sweep` — a temp YAML with `source: ASSUMED, sweep: null` fails lint.
  - `test_no_magic_numbers` — AST-walk `vulcan_proof/` (excluding `ev_reference.py`) for numeric
    constants not in the allowlist in `02_engineering_rules.md` A1; fail with file:line on any.
  - `test_derived_population_rates` — archetype-weighted contest = 0.6125 and compliance = 0.825.
- `tests/test_ev_reference.py` (provided in this repo; do not edit) — known-answer values in
  `reference.known_answers` reproduce from `ev_reference.py` to 2 dp; `test_reference_matches_params`
  asserts every shared constant (evidence cash/seconds/presence, uplifts, overlap, base win, category
  rates/mixes, hourly rate) is equal between `ev_reference.py` and `params.yaml`.
  *What failure means:* someone edited one without the other. Fix `params.yaml` or stop; never edit the reference.
- `tests/test_seeds.py` — same labels → identical first 10 draws; different labels → different.
- `tests/test_manifest.py` — `write_artifact` with a `hidden_x` column and `observed_` name raises `LeakError`.

## Part B — Olist anchor

### B1. Data
`python scripts\run_phase0.py --download` calls the `kaggle` package API (`kaggle.api.dataset_download_files`,
`unzip=True`) into `Path(P["olist.data_dir"])`. Credentials: `%USERPROFILE%\.kaggle\kaggle.json` on
Windows (`~/.kaggle/kaggle.json` elsewhere). If absent, print the exact manual-download instruction
(Kaggle dataset page → Download → unzip into `data\olist`) and exit 2. Assert all
`olist.required_files` exist and the row counts in `olist.expected_rows` hold within
`olist.row_count_tolerance` (raise otherwise — a truncated download silently changes every metric).

### B2. Label (`vulcan_proof/olist/label.py`)
Implement exactly the rule in `params.yaml: olist.label` comments. One row per order. Orders with
`order_status ∈ {canceled, unavailable, created}` are excluded and counted. Reviews: take the
*earliest* review per order (there are duplicates). Output schema `OLIST_LABELS`:
`order_id, purchase_ts, label (int8), label_reason (str: late_low_score | comment_match |
not_delivered | none)`. Report the label rate and the reason mix in `metrics.json`.

### B3. Features (`vulcan_proof/olist/features.py`)
Exactly `olist.features.permitted`. Definitions:
- `price_total`, `freight_total`: sums over items. `n_items`, `n_sellers`: counts.
- `product_category_en`: translation table; multi-item orders take the category of the highest-price item.
- `product_weight_g`, `product_volume_cm3` (l×w×h), `product_photos_qty`, `product_description_length`: of the highest-price item.
- `payment_type`, `payment_installments`: of the largest payment row.
- `customer_state`, `seller_state` (of highest-price item's seller), `same_state`.
- `purchase_month`, `purchase_dow`, `purchase_hour`.
- `seller_prior_orders`, `seller_prior_complaint_rate`: computed **strictly from orders with
  `purchase_ts` earlier than this order's `purchase_ts` minus `olist.split.maturity_days`** (so the
  label of a prior order would have been observable). Shrink the rate toward the global train rate
  with prior weight `olist.prior_shrinkage_n`. `customer_prior_orders`: by `customer_unique_id`, strictly earlier.
- Every forbidden column (`olist.features.forbidden`) must be absent; `schemas.check` enforces.

### B4. Split
By `purchase_ts`: train ≤ `train_end`; validate in (`train_end`, `validate_end`]; test in
(`validate_end`, `dataset_end − maturity_days`]. Orders after that are dropped as immature and
counted. Assert each split non-empty and label present in each.

### B5. Model (`vulcan_proof/olist/train.py`)
LightGBM with `models.lgbm.stage_a` params, `early_stopping_rounds` on validation log-loss,
`seed` from `SeedTree("olist","lgbm")`, `num_threads = run.lgbm_threads`. **No undersampling,
`scale_pos_weight = 1`** (trap B4). Isotonic calibration fitted on validation predictions.

### B6. Metrics (`vulcan_proof/olist/evaluate.py`) → `outputs/phase0/metrics.json`
On **test** only: PR-AUC, ROC-AUC, Brier, ECE (`models.calib.ece_bins`), reliability table
(bin edges, mean pred, empirical rate, count), calibrated-mean vs empirical-rate ratio.
Operating points at recall ∈ `olist.operating_recalls`: precision, flagged fraction, false
positives per 1,000 orders, and FP cost proxy = FP × `olist.fp_cost_proxy_seconds` ×
`econ.hourly_rate`/3600 ₹ per 1,000 orders. Also the Lorenz table: share of positives in each
score decile, and top-decile lift. Charts: PR curve and reliability diagram as PNG with footer
`report.olist_footer`.

### B7. Tests that ship with Part B (`tests/test_phase0.py`)
- `test_label_rule_known_rows` — hand-built 6-row fixture covering each reason and the exclusion; exact labels.
- `test_no_forbidden_features` — feature frame has no column from `olist.features.forbidden`.
- `test_prior_stats_are_strictly_past` — construct two orders from one seller 10 days apart; the
  later must not see the earlier's label (maturity gap). *Failure means:* label leakage through seller history.
- `test_split_monotone` — max train ts < min validate ts < min test ts; no order in two splits.
- `test_validation_calibration_and_test_drift_diagnostic` — on validation,
  |mean − rate|/rate < `models.calib.mean_tolerance` remains a hard gate. On test, the same value is
  an out-of-time transfer diagnostic: it must be reported, and an out-of-tolerance result must include
  raw prediction mean, monthly prevalence, per-reason prevalence, and the largest reason-rate drop.
  Test labels are evaluation-only and must never alter the model or calibrator. *Failure to produce or
  attribute the diagnostic is a gate failure; the diagnosed out-of-time mismatch itself is not.*
- `test_metrics_json_complete` — every key listed in B6 present, finite.
- `test_footer_present` — chart code contains `report.olist_footer` string.
- `tests/test_claims.py` — grep for the NEVER list over the paths listed in `docs/01_claims.md` (README, reports, demo script, api, ui, charts); `docs/` excluded.

## Done-criteria (`check_phase_0`)
1. `python scripts\task.py lint` exits 0.
2. `pytest tests/test_params.py tests/test_ev_reference.py tests/test_seeds.py tests/test_manifest.py tests/test_phase0.py tests/test_claims.py` all pass.
3. `outputs/phase0/metrics.json` exists; PR-AUC, Brier, ECE finite; three operating points present.
4. `outputs/phase0/manifest.json` valid; git tree was clean at run time.
5. `check_phase_0` runs the Olist pipeline twice from the same seed and asserts `metrics.json`
   byte-equal after deleting the keys `timestamp`, `wall_seconds`, `peak_rss_mb` (Olist is small; this is cheap).
6. `outputs/phase0_REPORT.md` contains the metrics table, the label rate and reason mix, the
   immature-drop count, and the sentence: "Olist has no chargeback or evidence data; this measures detection only."
7. Validation calibrated mean is within `models.calib.mean_tolerance`. Test PR-AUC, ROC-AUC, Brier,
   ECE, lift, operating points, calibrated-mean ratio, and temporal prevalence diagnostics are all
   reported. An out-of-tolerance test calibrated mean is explicitly labelled as calibration-transfer
   failure under temporal drift; it is not silently converted into a passing calibration claim.

### Post-audit specification amendment (2026-08-27)

The original test required the validation-fitted isotonic mean to match the unseen test prevalence
within 5%. The frozen run showed validation prevalence 9.2304% versus test prevalence 3.5424%; the
raw test mean was already 5.9980%, and validation isotonic calibration raised it to 7.2824%.
The largest shift was `late_low_score` (6.2763% validation versus 1.5042% test). With no test-label
adaptation allowed, unconditional mean matching is not an integrity invariant: it is an empirical
calibration-transfer outcome. This amendment does not change the data, label, split, permitted
features, model parameters, seed, tolerance, or predictions. It keeps validation calibration and
all temporal/leakage checks hard, and requires the failed test transfer to remain visible.

**Then halt.**
