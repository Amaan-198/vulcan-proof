# Engineering rules, traps, and guards

These rules describe the invariants that keep Vulcan Proof representative of a pre-dispatch
decision service. Runtime constants belong in `params/params.yaml`; the reference arithmetic and
the tests are the authority for behavior.

## Non-negotiable rules

### Information boundary

Model fitting and prediction accept only the permitted observed schema. Hidden fulfillment truth,
latent risk, true evidence effects, and simulator-only fields remain in the simulation and resolver
packages. Static import checks and runtime schema gates enforce the separation.

### Feature timing

Features used for a pre-dispatch plan must exist before dispatch. Post-payment outcomes, later
materialization, final dispute resolution, and hidden state cannot enter a training or prediction
frame. Merchant history is calculated from resolved observed outcomes using the same maturity rule as
the label.

### Centralized parameters

Every tunable constant, category setting, evidence cost, availability rule, calibration setting,
split boundary, and sweep choice is read through the parameter loader. Production code must not
introduce a local copy of a value that belongs in the parameter file.

### Deterministic execution

Use the seed tree for every random stream. Do not use a process-global random generator. Parallel
work must be safe on Windows and must preserve the association between an artifact and its manifest.
The manifest records the parameter digest, run context, artifact paths, and runtime information.

### Data types and schemas

Keep identifiers stable, nullable fields explicit, and numeric columns in the declared schema. Model
fit and predict must reject unknown, hidden, or missing features with `SchemaError`. Data files are
written through the artifact helpers so that the schema and provenance are retained.

### Costs and outcomes

Use the shared economics functions for money tables, prevention gain, evidence costs, contest
outcomes, and resolver outcomes. Cash cost is conditional on compliance; time cost is conditional
on the request. The dispute fee belongs in dispute-opening and prevention branches, not in a win/lose
incremental defense delta.

### Complete plans

The optimizer evaluates evidence subsets as plans. It must account for plan-dependent
contestability, expected materialization, defensibility, prevention, availability, support, cash,
and time. The search covers 512 evidence combinations per order and retains the empty plan as the
zero-value option.

### Calibration and reporting

Use isotonic calibration on validation data for every probability stage, then check the calibrated
mean against the empirical validation rate. Test data is for evaluation, not for calibration or
tuning. Machine-readable metrics remain complete for tests; human-facing reports explain their
meaning without turning simulator behavior into a production guarantee.

## Common traps and their guards

### Truth leaking into decision packages

An import, feature, helper, or report path that carries hidden state into models, arms, or the
optimizer invalidates the experiment. The firewall test walks the syntax tree, and the fit/predict
schema gate rejects hidden columns at runtime.

### Post-payment leakage

A feature such as final delivery outcome, materialized evidence, or resolved dispute state may be
useful for analysis but cannot be used for a payment-time plan. Keep such fields in outcome frames
and build the training view explicitly.

### Immature and censored labels

Label maturity follows the simulated dispute timeline. A censored row is not a negative example.
Training filters must be shared with the label builder, and the report should describe how censoring
was handled.

### Intercept and calibration drift

When a calibrated probability is used in expected value, a shifted mean compounds through the
downstream stages. Calibration is validation-only, and the validation mean check is a required
invariant. Temporal drift in the test window is reported diagnostically rather than corrected by
peeking at the test labels.

### Weak support

The optimizer must not select a dispute/evidence pair that has insufficient observed training
support. Support masks are built from eligible, contested training rows. Low-support realized
bitmasks use the documented main-effect shrinkage path, and the reason code identifies the support
failure.

### Item decisions instead of subset decisions

An evidence type can be attractive on its own and harmful when held with another type. Evaluate the
plan, materialization patterns, and incremental value together. There are no independent item
threshold shortcuts inside the subset loop.

### Incorrect cost timing

Do not charge a merchant cash cost when a requested item fails to materialize, and do not hide a
request-time task inside a compliance probability. System-sent acknowledgements follow their own
documented cost path. The reference implementation and fee-scope test protect these branches.

### Wrong-recipient effects

An evidence effect must be conditioned on dispute and fulfillment context. A handoff record that
helps a correctly delivered order cannot be treated as a universal defense against merchant fault
or carrier fault. Keep carrier-fault behavior visible in robustness reporting.

### Contestability independent of the plan

Contest behavior can change when a merchant holds evidence. Stage C therefore receives the planned
evidence state, and materialization is integrated with contestability when expected value is
assembled. A population contest rate is not a substitute for this plan-dependent prediction.

### Seed reuse and unpaired comparisons

Use independent named streams for world components and paired order contexts for policy comparisons.
Single-run differences are not a substitute for the paired reporting path. Reproducibility tests
must be able to repeat the same plan from the same seed context.

### Time leakage in the tuned policy

The tuned comparison rule is fitted only on the permitted historical window. Do not tune it on the
test outcome or use a later artifact to change an earlier policy table.

### Stage distribution handling

Stage B outputs a calibrated distribution over dispute type, and downstream expected value sums over
that distribution. Renormalization must preserve the declared schema and should never smuggle hidden
truth into the class probabilities.

### Non-deterministic tree training

Keep the LightGBM configuration, seed handling, feature order, and thread behavior aligned with the
reproducibility contract. Two equivalent runs should produce equivalent model artifacts and plans.

### Public-data label leakage

The Olist anchor may use delivery and review data only when those fields are available at the
declared decision boundary. Its labels and features stay separate from the simulator's dispute and
evidence world because Olist has no chargeback or evidence data.

### History seeded from hidden state

Merchant history is an observed feature derived from resolved outcomes. A hidden merchant propensity
may exist in the simulator, but it is not a valid feature seed for the learner.

### Exact-zero signal assumptions

The no-signal setting is a guard against accidental access to individual truth, not a demand that
every legitimate observed history feature become exactly neutral. Compare the optimizer with the
tuned context rule and inspect the paired result.

### Per-item shortcuts inside the loop

The complete subset search must remain the single decision path. A standalone evidence score is
useful for explanation, but it cannot replace plan-level expected value.

### Reference drift

`vulcan_proof/ev_reference.py` is a frozen known-answer oracle. If an implementation disagrees with
it, investigate the implementation or the parameter plumbing; do not alter the oracle to make a
test pass.

## Halt protocol

Each phase has a checker, report, and done-criteria document. An implementation should run the
phase-specific tests, inspect the generated manifest and report, and stop at the phase boundary.
Do not continue to a later phase while an invariant, firewall, schema, or artifact check is failing.
The task runner provides the cross-platform commands used by contributors and judges.
