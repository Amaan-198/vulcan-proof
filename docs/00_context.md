# Context: why the constraints exist

This document explains the operating boundary, economic objective, and evaluation design behind
Vulcan Proof. The implementation is easier to review when the reason for each guard is visible next
to the behavior it protects.

## What the system does

A merchant may know that an order was delivered, but a later dispute team can rely only on records
that were created while the order was being fulfilled. Vulcan Proof therefore runs before dispatch:
it estimates exposure, identifies plausible dispute types, chooses admissible evidence, and
coordinates its capture while the evidence window is still open.

The product covers prepaid physical goods and defense-only handling. Its evidence surface has 9
evidence types: weight, serial, sealed, packing, geotag, OTP, signature, acknowledgement, and
verified acknowledgement. Non-receipt, not-as-described, and empty-box disputes are represented as
distinct downstream classes.

## The simulator and the information firewall

The repository contains no Razorpay dispute history. The simulator supplies a controlled world with
observed order features and a separate hidden fulfillment state. The hidden state is used by the
resolver to create outcomes, evidence materialization, and prevention behavior. Models and the
optimizer receive the observed frame only.

This boundary is enforced in several places:

- simulation and resolution packages may read hidden state;
- model and optimizer packages may import only permitted observed features;
- static checks reject hidden-state symbols in decision packages; and
- runtime schema checks reject hidden columns, unknown fields, and missing values.

This makes the evaluation answer the production question: what can a decision service learn before
dispatch, when later fulfillment truth is not available?

## The two expected-value views

The resolver can condition on fulfillment truth because it is responsible for producing the
experiment's outcome. The optimizer uses a truth-blind expectation instead. Its objective has the
following shape:

```text
expected value(plan | observed order)
  = exposure
  × dispute-type mixture
  × contestability for the planned evidence
  × expected materialization
  × defensibility uplift
  × order value
  − expected cash cost
  − requested-time cost
  + prevention value when an acknowledgement prevents a dispute
```

The materialization expectation is taken over the evidence that may actually become available. The
optimizer evaluates the complete subset, not independent item thresholds, because contestability,
materialization, and defensibility can all depend on the plan as a whole. It evaluates 512 evidence
combinations per order and keeps the empty plan as the zero-value baseline.

Cash costs follow compliance: a merchant pays when a requested item is actually booked. Time costs
follow the request: the recommendation creates an operational task even when the evidence does not
materialize. Prevention is accounted for separately because preventing a dispute changes the whole
outcome, including dispute handling cost.

## Economic design decisions

The evidence decision is defense-only. A plan may improve the record for an order that is correctly
fulfilled, but it never instructs a merchant to contest a claim that lacks admissible support. The
resolver keeps goods, handling, dispute, contest, and prevention branches explicit so that a cost
does not accidentally appear in the wrong comparison.

Contestability is plan-dependent. A merchant may contest more often when a useful evidence package
exists, so the contestability model receives the planned evidence state. Evidence effects are also
conditioned on the relevant dispute type and fulfillment context; a record that helps one class can
be neutral or harmful for another.

The prevention model is separate from defense. Acknowledgement evidence can reduce the chance that a
dispute opens, while other evidence is mainly useful after a dispute exists. Combining those paths
would make a plan appear valuable for the wrong reason, so the resolver and the optimizer keep them
as separate terms.

## Simulator construction

World generation proceeds through merchants, customers, orders, individual risk, fulfillment truth,
historical evidence policy, derived dispute potential, acknowledgement response, latency, splits,
and censoring. Each component has its own named seed-tree stream. The observed frame is written
without latent fields; the hidden frame retains the information needed by the resolver.

Dispute potential is derived from genuine fulfillment failures and false claims rather than being
set as unrelated independent rates. Category context, order value, and customer-level risk can all
affect the funnel. Calibration targets and simulator assumptions live in `params/params.yaml`, while
the calibration artifact records the derived solution for a run.

Latency and label maturity are explicit. A training label exists only when the outcome is mature by
the simulated decision boundary. Censored disputes remain identifiable as censored history; they
are not silently converted into negative labels.

Merchant history is built from resolved observed outcomes and dates. It is never seeded from hidden
merchant propensity. This keeps the history feature useful while preserving the same information
available to a production learner.

## Evaluation design

The tuned policy provides an observable comparison point, and the learned optimizer is evaluated on
the same orders. Paired comparisons reduce noise from order mix and make each policy face the same
merchant, category, value, and dispute context. Sensitivity studies vary assumptions separately from
the primary product description.

The submission summary reports optimizer coverage of 53.24%, top-decile risk lift of 1.75×, and a
false-positive cost of ₹695.69 per 1,000 orders. The simulator's evidence surface is 9 evidence
types with 3 prediction stages before dispatch. The public-data detection anchor and the
simulator-based end-to-end evaluation answer different questions; production calibration requires
Razorpay dispute history.

## Defense-only behavior

The service may recommend evidence that strengthens a legitimate response, but it does not optimize
for aggressive contesting. Win behavior is inspected by fulfillment context so that correct,
merchant-fault, and carrier-fault cases remain distinguishable. The product language is therefore
about evidence held and package readiness, never about manufacturing a defense.

## Platform and reproducibility

The project runs with a CPU-only runtime with no GPU and no paid APIs. Paths use `pathlib`, subprocess
calls use argument lists, and process-based parallel work is guarded for Windows. The learned
stages use gradient-boosted trees and isotonic calibration. The optimizer performs an exhaustive
truth-blind search over the admissible evidence plans.

Randomness comes from a seed tree rather than a process-global generator. Manifests record the
parameter digest, phase context, artifact paths, package environment, and runtime observations.
Parquet files hold tabular artifacts, JSON files hold machine-readable summaries, and Markdown
reports explain how those artifacts should be read.

## Production boundary

Olist supplies public order, seller, product, delivery, and review records for detection work. It
does not contain chargeback or evidence records, so it cannot provide Razorpay dispute calibration.
The simulator supplies the controlled end-to-end environment used by the later stages. A deployment
must recalibrate on governed Razorpay dispute history and verify the evidence API mapping before
using the service for live decisions.
