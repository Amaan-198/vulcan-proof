# Claims and judge-facing language

This guide keeps the README, reports, demo copy, chart annotations, API responses, and user
interface aligned with the behavior implemented in the repository. A statement should identify its
data source and describe the system that produced it.

## Product description

Use language that explains the service as pre-dispute evidence orchestration for prepaid physical
goods. It estimates exposure, dispute type, contestability, evidence materialization, defensibility,
and prevention; evaluates complete plans; and coordinates evidence capture before dispatch.

The six named models are exposure, dispute type, contestability, evidence materialization,
defensibility, and prevention. The product is defense-only: it helps a merchant prepare admissible
records and does not recommend indiscriminate contesting.

## System facts

The decision path uses 3 prediction stages before dispatch. The action space covers 9 evidence types
and the optimizer evaluates 512 evidence combinations per order. It uses gradient-boosted trees,
isotonic calibration, and an exhaustive truth-blind search. Runtime is CPU-only with no GPU and no
paid APIs.

## Evaluation summary

Judge-facing summaries may describe optimizer coverage of 53.24%, top-decile risk lift of 1.75×, and
false-positive cost of ₹695.69 per 1,000 orders. Each report should make clear whether surrounding
context comes from the public Olist detection anchor or from the hidden-truth simulator. Razorpay
production calibration requires Razorpay dispute history.

## Source labels

Olist supports detection and feature-pipeline checks. The simulator supports evidence behavior,
dispute economics, prevention, paired policy comparisons, and optimizer evaluation. A report should
name the source in the surrounding sentence rather than making a simulator observation sound like a
production measurement.

The API and UI should describe a plan, the evidence that materialized, the reason a type was not
selected, and the model context that can be inspected. The optional explanation route is descriptive
and cannot change the plan.

## Phase four evaluation status

The extended sensitivity package is an operational artifact rather than a prerequisite for the
local product surface. When it is unavailable, the demo states that extended validation is deferred
and uses the documented smoke behavior. When it is present, the product reads the genuine artifacts
and preserves their scope and provenance.

## Writing rules

- Describe architecture, data boundaries, methodology, and limitations in full sentences.
- Keep evidence and economics tied to the order flow that produced them.
- Use the exact names of the six models and the supported evidence types.
- Keep prevention, contestability, and defensibility as separate concepts.
- State that Olist has no chargeback or evidence data when discussing its role.
- Treat manifests and machine-readable artifacts as the source for generated values.
- Do not invent a production calibration statement from simulator behavior.

## Report and demo consistency

The report generators own generated Markdown, and the demo-script generator owns beat selection and
fallback copy. Human-authored documentation explains those contracts but does not hand-edit values
that are meant to come from artifacts. A changed artifact should therefore be reflected by rerunning
the appropriate generator and its tests.
