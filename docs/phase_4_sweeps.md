# Sensitivity, signal, and robustness sweeps

This phase studies how the learned planner behaves as simulator assumptions and observed signal
change. It is a sensitivity package around the product workflow, not a second decision engine. Each
run consumes the same phase artifacts, writes a manifest, and keeps its parameter context visible.

## What this phase decides

The sweep package identifies whether the learned planner adds value beyond the tuned context policy,
which assumptions drive that comparison, and whether defense-only behavior remains coherent across
alternative fulfillment contexts. The report describes the source and scope of each result.

The decision path remains unchanged: 3 prediction stages before dispatch, 9 evidence types, and an
exhaustive truth-blind search over 512 evidence combinations per order. The current submission
summary includes optimizer coverage of 53.24%, top-decile risk lift of 1.75×, and false-positive
cost of ₹695.69 per 1,000 orders.

## Completion modes

The local buildathon path supports smoke and end-to-end checks with the artifacts available in the
repository. The extended sensitivity package can be run later when the required compute and source
artifacts are available. The demo and report state which context is present and do not turn a missing
extended artifact into a measured result.

## Signal sweep

The signal sweep varies the customer-level contribution to exposure and dispute type while keeping
the observed context and policy comparison explicit. It uses paired orders and records the learned
planner's difference from the tuned policy for each sweep point. The no-signal point is a leakage
guard: legitimate category, value, tier, and history context remains available, while hidden
individual truth remains unavailable.

## One-at-a-time sensitivity

The one-at-a-time sweep changes one configured assumption at a time. Candidate assumptions include
evidence effects, materialization, contestability, prevention, timing, costs, and category context.
The output ranks how the paired planner result changes over the configured range. Parameter names,
levels, seed context, and artifacts are preserved for review.

## Joint sensitivity

The joint sweep varies a configured collection of assumptions together. It records the parameter
draw, paired result, confidence information, and whether the result crossed the configured decision
boundary. Disabled or unavailable runs remain explicit in their JSON status and report copy.

## Robustness contexts

Robustness runs examine carrier-fault, merchant-fault, materialization, and calibration contexts.
The report keeps carrier-fault behavior visible and checks that a defense-only planner does not gain
its apparent value by encouraging contests that the product is not intended to promote.

## Charts and artifacts

Charts are generated from JSON artifacts and carry the appropriate simulator or public-data footer.
The chart builder does not calculate a new business claim; it renders the stored sweep result with
its parameter context. The report, progress file, sweep JSON, charts, and manifests share the same
run identity.

## Tests and completion

Tests cover sweep point construction, deterministic seed use, parameter immutability, deferred
artifact handling, robustness rows, chart creation, report structure, and the no-signal guard.
Completion requires the sweep package, policy status, reports, and charts to agree. If extended
validation is unavailable, the local product remains usable and the status is stated in the demo.
