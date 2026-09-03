# Tests

The test suite protects the data boundary, expected-value arithmetic, reproducibility contract,
phase artifacts, and product surface. Run it from the repository environment with the task runner
or the project interpreter.

## Shared coverage

The parameter and reference tests verify that the centralized configuration loads and that the
frozen EV oracle remains stable. Schema tests cover observed, hidden, outcome, and artifact frames.
Manifest and seed tests check provenance and deterministic child streams. Claims tests keep
judge-facing reports, API copy, charts, and the UI aligned with the product description.

## Phase coverage

The Olist tests cover label timing, feature leakage, temporal splits, calibration, and the public
data boundary. Simulator tests cover world construction, hidden/observed separation, derived funnel
behavior, latency, censoring, and evidence policy. Resolver tests cover outcome branches,
prevention, fee scope, materialization, merchant history, tuned policy behavior, and paired
reporting.

Model and optimizer tests cover the six named models, isotonic calibration, support masks,
plan-dependent contestability, materialization expectation, prevention reduction, exhaustive
truth-blind planning, reference agreement, missing-feature rejection, and reproducible plans. Sweep
tests cover parameter immutability, status handling, sensitivity artifacts, robustness, charts, and
the no-signal guard. Product tests cover API routes, artifact-backed plans, demo selection and
fallbacks, optional explanation behavior, and the UI bundle.

## Test information boundary

Tests may inspect hidden columns and true evidence effects when checking the resolver or constructing
a deliberately leaked foil. Library code under `vulcan_proof/opt`, `models`, and `arms` may not use
those fields. A failing invariant is fixed in the implementation or its data boundary; it is not
fixed by weakening the parameter contract.

## Completion

Every phase has a corresponding report and checker. Run the relevant phase tests after generating
artifacts, inspect the manifest and report together, and stop at the phase boundary when the
invariants pass.
