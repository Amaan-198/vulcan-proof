# Outcome resolver, prevention, history, and tuned policy

This phase turns the simulator's observed and hidden frames into resolved outcome artifacts. It
exists before machine learning so that the economics, evidence behavior, maturity boundary, and
merchant history are defined independently of the optimizer.

## Why this phase exists before learning

The resolver is the source of realized outcomes. It can read hidden fulfillment truth because that
state is needed to decide whether a dispute is correct, merchant-fault, carrier-fault, or otherwise
resolved. The learner later sees only the observed outcome frame. Keeping these responsibilities
separate prevents the model from inheriting the resolver's privileged information.

## Concepts

The phase distinguishes:

- exposure to a potential dispute;
- the dispute type and the claim class;
- whether a merchant contests;
- which requested evidence materializes;
- whether the realized evidence supports a defense;
- whether an acknowledgement prevents the dispute; and
- the money and time consequences of each branch.

The evidence surface contains 9 evidence types. Their admissibility and capture windows are shared
with the model and API layers through the parameter contract.

## Order of operations

The phase runner loads a completed simulator world, resolves the historical policy, derives merchant
history from mature observed outcomes, applies the comparison policies, and writes one outcome frame
per policy. It then creates paired summaries using the same orders and manifest context.

The historical policy is resolved before the tuned comparison is fitted. The comparison rule can
use category, value band, contest history, tier, and other permitted context, but it cannot use future
test outcomes. The optimizer's later comparison therefore has a stable policy baseline.

## Truth-conditional resolution

The resolver uses hidden fulfillment truth to determine the outcome branch. Evidence materialization
depends on requested evidence, merchant behavior, customer response, and capture windows. Contest
behavior can depend on the evidence held. Defensibility is evaluated against the realized dispute
type and evidence set, while prevention is evaluated before a dispute opens.

The money table keeps order value, dispute fees, cash evidence costs, request-time costs, prevention
costs, and outcome deltas explicit. The false-positive cost reference is ₹695.69 per 1,000 orders.
The same pure economics helpers are used by the resolver and the optimizer-facing reference path.

## Merchant history

History features are built from resolved outcomes whose labels are mature by the decision boundary.
The history builder uses observed policy and outcome columns plus dates. It never reads hidden
merchant propensity and never lets a future resolution leak into an earlier plan.

## Policy arms and artifacts

The baseline represents the simulator's ordinary evidence behavior. Intervention policies isolate
prevention, free evidence, paid handoff evidence, and the tuned category/value/history rule. Their
outcomes are written to `outcome_arm*.parquet`, with the tuned policy table and history artifacts in
the corresponding run directory.

The tuned policy is intentionally interpretable. It conditions on permitted order and merchant
context, applies admissibility and cost rules, and provides the comparison that the learned planner
must match or improve under the evaluated context.

## Paired reporting

`vulcan_proof/report/paired.py` aligns policy outcomes by order identifier, computes the per-order
difference, aggregates by seed, and returns the paired summary used by later reports. The report
layer describes the comparison and its context; it does not change the outcome data.

## Tests and completion

Tests cover resolver branch accounting, prevention, fee scope, evidence materialization, history
timing, policy admissibility, paired alignment, and artifact schemas. Completion requires the
observed and outcome artifacts, the tuned policy table, valid manifests, and a report containing the
resolver comparison sections and simulator context.
