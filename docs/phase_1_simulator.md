# Hidden-truth simulator

The simulator creates the controlled end-to-end world used to study dispute exposure, evidence
behavior, prevention, and plan value. It writes an observed frame for the learner and a hidden frame
for the resolver. The optimizer never receives the hidden frame.

## What the simulator produces

Each generated order has merchant and customer context, order and fulfillment attributes, individual
risk, latent fulfillment truth, historical evidence policy, dispute potential, acknowledgement
response, latency, split assignment, and censoring state. The observed artifact contains only fields
available to a pre-dispatch learner; the hidden artifact retains the latent fields needed to resolve
the experiment.

The evidence surface contains 9 evidence types: weight, serial, sealed, packing, geotag, OTP,
signature, acknowledgement, and verified acknowledgement. Every type has an admissibility rule,
capture window, materialization behavior, and API slot defined in the parameter contract.

## Generation order

World generation uses named child streams from the seed tree. The stages are:

- merchants, with observable history and latent process context;
- customers and order context;
- individual risk and dispute-type tendencies;
- fulfillment truth, including correct delivery, carrier issues, merchant fault, and damage;
- historical evidence requests and materialization behavior;
- derived dispute potential from genuine failures and false claims;
- acknowledgement response and prevention behavior;
- latency, maturity, split assignment, and censoring; and
- funnel calibration and optional world-shift context.

The order matters. Dispute potential is derived after the underlying context exists, and label
maturity is applied after latency exists. The simulator does not set unrelated rates independently
when a shared funnel relationship is required.

## Observed and hidden frames

The observed frame includes the order context, merchant history inputs, permitted pre-dispatch
features, evidence requests, and outcomes that have become observable by the relevant boundary. The
hidden frame includes fulfillment truth, true evidence effects, latent risk, and other state used by
the resolver. A runtime schema gate checks the boundary before model fit or prediction.

The optimizer can see neither true evidence effects nor hidden fulfillment truth. The resolver and
evaluation harness may use them to produce the realized outcome and to measure how well the observed
learner approximates the controlled world.

## Derived funnel and calibration

The dispute funnel combines genuine fulfillment failures with false claims. Category context,
order value, and customer-level risk influence that funnel. Calibration solves the configured
derived targets and writes its result to the calibration artifact consumed by world generation.

The report describes category behavior, censoring, historical requests, and the evidence-policy
diagnostic in qualitative context. Machine-readable summaries retain the complete values for tests
and for the resolver.

## Latency, maturity, and history

Disputes can mature after the decision date. Rows whose outcome is not mature by the training
boundary remain censored and are excluded from downstream labels. They are not silently assigned a
negative outcome. Merchant history is added later from resolved observed outcomes; it is not seeded
from latent merchant state here.

## Tests and completion

Tests cover deterministic streams, observed/hidden separation, funnel calibration, censoring,
evidence policy support, latency ordering, and world-shift behavior. Completion requires both
artifacts, a valid manifest, a readable report, and a parameter digest that matches the run.
