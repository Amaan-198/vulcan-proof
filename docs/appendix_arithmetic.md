# Vulcan Proof — expected-value derivation

This appendix explains how the evidence objective is assembled and how the decision service moves
from order context to a complete plan. It is a reasoning companion to
`vulcan_proof/ev_reference.py`, which remains the frozen known-answer implementation used by the
tests.

## What is evaluated

For an observed order and candidate evidence plan, the optimizer estimates:

```text
exposure
× dispute-type probability
× contestability given the plan
× evidence materialization expectation
× defensibility gain over the empty plan
× order value
− expected cash costs
− requested-time costs
+ prevention value for evidence that can stop a dispute opening
```

The resolver evaluates the same economic branches with access to fulfillment truth. The optimizer
is truth-blind and uses fitted observed-data models. The two views share the economics functions,
but they do not share information that would be unavailable at payment time.

## Evidence plans

The action surface contains 9 evidence types: weight, serial, sealed, packing, geotag, OTP,
signature, acknowledgement, and verified acknowledgement. The optimizer evaluates 512 evidence
combinations per order. It calculates materialization patterns within a plan, applies support masks,
and retains the empty plan as the zero-value baseline.

The plan value is not the sum of independent item values. Contestability can change with planned
evidence, materialization can be correlated through merchant behavior, and defensibility depends on
the realized set. The reference arithmetic therefore compares complete subsets and computes
incremental value after the rest of the plan is already held.

## Cost timing

Cash items are charged when the merchant complies and the item materializes. Request-time items are
charged when the recommendation creates the task. Acknowledgement prevention is applied to the
dispute-opening branch, while defense evidence changes the outcome after a dispute exists. This
separation prevents a dispute fee or handling cost from being charged in the wrong comparison.

## Worked reasoning anchors

The design uses two qualitative anchors. For a higher-value electronics order, a handoff record may
be worthwhile only when the exposure and contest context support it; adding another handoff record
can reduce incremental value when it overlaps with the first. For a lower-value apparel order,
pre-dispatch capture is selected only when the predicted exposure and expected defense value cover
the operational cost. These anchors explain why the optimizer must search subsets instead of applying
one threshold to each item.

## Sensitivity

The simulator's evidence effects are assumptions, so sensitivity work varies them before drawing a
deployment conclusion. The sweep package also varies exposure, contestability, materialization,
prevention, timing, and category context. Each run records parameters, seed context, artifacts, and
source labels in its manifest.

## Evaluation summary

The submission summary reports optimizer coverage of 53.24%, top-decile risk lift of 1.75×, and
false-positive cost of ₹695.69 per 1,000 orders. The learned path uses 3 prediction stages before
dispatch, gradient-boosted trees, isotonic calibration, and an exhaustive truth-blind search. The
runtime is CPU-only with no GPU and no paid APIs.

## Implementation notes

`ev_reference.best_subset` is the arithmetic twin used for known-answer tests. Model-backed
optimization must agree with that interface when supplied with the reference model stub. Any
disagreement is investigated in the implementation, feature plumbing, cost timing, or materialization
expectation; the reference is not adjusted to hide the discrepancy.
