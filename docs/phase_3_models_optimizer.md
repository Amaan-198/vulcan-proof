# Models, optimizer, support, and learned planning

This phase fits the observed-data decision path and applies it to the resolved simulator worlds.
The central rule is that every model and every optimizer call must be truth-blind: only the
permitted observed schema may cross into the learning packages.

## The model chain

The six named models are exposure, dispute type, contestability, evidence materialization,
defensibility, and prevention. The decision path uses 3 prediction stages before dispatch:

- Stage A, exposure, estimates whether a potential dispute enters the funnel;
- Stage B, dispute type, produces a calibrated distribution over non-receipt, not-as-described, and
  empty-box classes; and
- Stage C, contestability, conditions the contest probability on the planned evidence.

Evidence materialization, defensibility, and prevention supply the plan-level terms used by the
optimizer. Their outputs remain tied to observed context, policy, and evidence state.

## Labels and maturity

Training labels are built from outcomes that are mature by the simulated decision boundary. Censored
rows remain outside the label set. The label builder and the resolver share the maturity rule so that
the model does not learn from a future resolution.

## Stage A — exposure

Stage A uses permitted order, merchant, category, value, network, issuer, delivery-timing, and
history features available before dispatch. Its target is exposure, not a post-dispute outcome. The
stage is fit on observed training rows and calibrated on validation data with isotonic calibration.

## Stage B — dispute type

Stage B is conditioned on the exposure frame and predicts a distribution over the supported dispute
classes. The downstream expected value sums over that distribution instead of selecting a single
class. Class probabilities are calibrated independently and checked for a valid total distribution.

## Stage C — contestability

Stage C receives the planned evidence state because the likelihood of contesting can change when a
merchant holds useful evidence. It is trained from observed historical policy and mature outcomes.
At planning time, the model is evaluated for each planned state and combined with materialization
expectation before defensibility is applied.

## Evidence materialization

The materialization model estimates whether a requested evidence type becomes available for a
merchant and order context. It accounts for observed request behavior, merchant compliance, and
acknowledgement response. A type with weak training support uses the documented population fallback
and carries a support indicator.

The planner evaluates 9 evidence types and 512 evidence combinations per order. Materialization
patterns are integrated inside each complete candidate plan so that the value of a planned set is
not confused with the value of an idealized set that always materializes.

## Defensibility and support

The defensibility model predicts the win probability for a dispute type and realized materialized
evidence set. Its observed features include dispute context, materialized evidence indicators,
network, issuer family, category, order value, and merchant features. Isotonic calibration is fit on
validation contested rows.

Support masks are computed from eligible contested training rows. Unsupported dispute/evidence pairs
are removed from the action space. Realized bitmasks with weak support use the main-effect shrinkage
path, which combines a full model prediction with a context-matched single-evidence effect.

## Prevention

The prevention model estimates the chance that an acknowledgement prevents a dispute before it
opens. The optimizer combines that prevention value with the non-prevented defense value. This keeps
the prevention path distinct from a post-dispute win-probability uplift and makes the fee scope
consistent with the resolver.

## Optimizer

For each order, the optimizer:

- builds the admissible evidence set from dispute-type support, capture availability, merchant opt-in,
  and plan constraints;
- enumerates 512 complete evidence combinations;
- evaluates Stage C for the planned set;
- integrates evidence materialization patterns;
- applies calibrated defensibility and prevention terms;
- subtracts cash and request-time costs using the shared economics functions; and
- returns the highest-value plan, with the empty plan as the zero-value choice.

The search is exhaustive and truth-blind. It never reads hidden truth, true evidence effects, or
simulator-only columns. Standalone and incremental values are computed for explanation, but the
selection itself is always plan-level.

## Artifacts and report

The phase writes model metrics, calibration summaries, support tables, per-type materialization
rates, refusal reason codes, paired outcomes, manifests, plans, and the human-readable report. The
submission summary reports optimizer coverage of 53.24% and top-decile risk lift of 1.75×. The
false-positive cost reference is ₹695.69 per 1,000 orders.

## Tests

The tests cover the firewall, schema rejection, calibration means, class-probability handling,
plan-dependent contestability, support masks, prevention reduction to the reference objective,
expectation of the product across materialization, no cross-row cache, no missing features,
reproducible plans, and reference agreement with a perfect-model stub. They also verify that training
does not consume gap or test outcomes.

## Completion

Completion requires model artifacts for the smoke and configured canonical contexts, valid manifests,
paired results, support and refusal reporting, and a report that identifies the simulator scope.
Once the phase checker and tests pass, the product phase may read these artifacts without recomputing
their values.
