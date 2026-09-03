# Phases, order, and halt protocol

Vulcan Proof is built as a sequence of boundaries. Each phase produces artifacts that the next
phase may consume, and each phase stops after its tests and checker have established the contract.

## Phase map

### Olist detection anchor

The public-data path establishes the feature, label, split, calibration, manifest, and chart
infrastructure. Olist has no chargeback or evidence data, so this phase measures detection behavior
and guards against feature leakage rather than dispute economics.

### Hidden-truth simulator

The simulator creates merchants, customers, orders, risk, fulfillment truth, historical evidence
policy, dispute potential, acknowledgement response, latency, splits, and censoring. It writes an
observed frame for learning and a hidden frame for resolution. Calibration is derived from the
configured funnel rather than assembled from unrelated rates.

### Outcome resolver and tuned policy

The resolver applies truth-conditional outcomes, prevention, historical policy, merchant history,
and the tuned evidence policy. Shared economics functions keep dispute fees, goods value, cash,
time, contestability, and prevention branches consistent. Paired artifacts establish the comparison
surface for later model evaluation.

### Models and optimizer

The learned path contains the six named models: exposure, dispute type, contestability, evidence
materialization, defensibility, and prevention. The decision flow uses 3 prediction stages before
dispatch. The optimizer applies admissibility and support masks, integrates materialization, and
performs an exhaustive truth-blind search over 512 evidence combinations per order.

### Sensitivity and robustness

The sweep package explores signal strength, individual assumptions, joint parameter variation, and
robustness contexts. It writes machine-readable sweep artifacts and charts with source labels so
that a judge can see which part of the result is public-data detection, simulator behavior, or a
production prerequisite.

### Product surface

The API reads completed artifacts and serves an order plan, the evidence explanation, the dispute
package, the tuned policy, and the generated demo script. The React/Vite interface follows the same
order-to-plan-to-package flow. It does not recompute or rewrite evaluation artifacts.

## Rules for every phase

- Read the context, claims, engineering rules, and parameter contract before implementing work.
- Keep data, model, optimizer, report, and UI responsibilities separate.
- Use the configured seed tree and write a manifest for each run.
- Keep hidden truth out of observed training and prediction frames.
- Preserve the declared schema and use the shared artifact helpers.
- Label public-data, simulator, and production contexts in the surrounding prose.
- Run the phase tests and checker before moving on.
- Stop when the phase contract passes; do not silently repair an artifact by hand.

## Checkers and artifacts

`scripts/check_phase.py` verifies the presence and shape of phase artifacts, manifests, required
report sections, schema invariants, and buildathon status. It is a structural checker; the phase
tests cover the deeper arithmetic, firewall, reproducibility, and API behavior.

Reports are human-readable companions to JSON metrics, Parquet tables, charts, demo scripts, and
manifests. Generated report text is owned by the corresponding pipeline so that a rerun cannot
silently retain stale prose.

## Smoke-world contract

The smoke world exercises the complete path quickly: observed and hidden generation, resolution,
history features, tuned policy, model fit, optimizer planning, paired reporting, and product
artifacts. It is useful for development and integration checks. A smoke result is labelled in its
own context and is not presented as a production calibration.

## Extended validation

The extended sensitivity package can be run when the required compute and artifacts are available.
Its status is explicit in the report and demo data. A deferred extended run is different from a
completed run whose measured signal is inconclusive; both states retain their own provenance.

## Directory contract

Source code lives under `vulcan_proof/`, configuration under `params/`, task entry points under
`scripts/`, tests under `tests/`, reports and run artifacts under `outputs/`, and the judge-facing
explanations under `docs/`. A phase may add artifacts inside its output area, but it must not edit
the parameter file or rewrite another phase's source data as a side effect.

## Completion behavior

The final product surface reads the model and sweep artifacts that exist locally. It reports the
scope of those artifacts, uses documented fallbacks when an extended package is unavailable, and
keeps optional explanation text outside the decision authority. Once the phase checker and tests
pass, stop and hand the artifacts to the reviewer.
