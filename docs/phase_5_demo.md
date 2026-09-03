# Product surface and scripted demo

The product surface turns completed model and evaluation artifacts into a judge-friendly workflow.
It serves an order plan, explains the evidence decision, and presents the materialized dispute
package. The API and UI read artifacts; they do not recompute or revise evaluation results.

## Product principle

Every order starts with observed pre-dispatch context. The service estimates exposure, dispute type,
contestability, evidence materialization, defensibility, and prevention; chooses a complete plan;
and records the reasons for selected and refused evidence. The evidence surface contains 9 types,
the planner evaluates 512 combinations per order, and the decision path uses 3 prediction stages
before dispatch.

The generated demo script selects concrete artifact-backed orders. When a selection condition is not
available in the local artifacts, the generator emits a clear product explanation of the available
context. Copy is generated from the artifacts and the claims guide rather than typed into the UI.

## Artifact modes

In the local smoke mode, the API and demo use the completed model and smoke artifacts. An extended
sensitivity package may be read when it exists and is complete. The script carries the mode and
scope so that the interface never presents a deferred extended run as a measured production result.

## API

`vulcan_proof/api/main.py` exposes:

- `GET /order/{order_id}/plan`, which returns the learned plan, per-type standalone and incremental
  value, availability, support, refusal reasons, and the Stage A/B/C outputs;
- `GET /order/{order_id}/dispute-package`, which maps materialized evidence to contest-API slots for
  an order with an opened dispute;
- `GET /report/arm4-policy`, which returns the tuned comparison policy; and
- `GET /demo/script`, which returns the generated walkthrough.

The optional explanation route can render a sentence from the existing plan when explicitly enabled.
It is disabled by default and has no authority over the plan.

## Demo script generator

`vulcan_proof/api/demo_script.py` chooses artifact-backed orders for an ordinary plan, a flagged plan,
a contestability contrast, a defense-only context, a dispute package, and an honest evaluation
status. Each beat records its selection context, plan, evidence details, and fallback when the
required artifact condition is unavailable.

The generator validates that displayed values come from the Phase 3 or extended sensitivity
artifacts. It also checks the judge-facing language before writing `outputs/phase5/demo_script.json`.

## UI

The React/Vite interface follows the order-to-plan-to-package flow:

- the order view shows the selected order context and exposure information;
- the plan view shows requested evidence, per-type value, availability, support, refusal reasons,
  contestability context, and the tuned comparison plan; and
- the dispute-package view shows the evidence that materialized and its API slots.

There is no separate validation dashboard in the UI. Evaluation scope remains in the generated
script and the underlying reports. The interface is laptop-first and uses the FastAPI service for
all artifact-backed data.

## Runtime

The Python service serves the built UI through FastAPI. The local development task starts the UI and
API together. The runtime is CPU-only with no GPU and no paid APIs; model work uses gradient-boosted
trees and isotonic calibration, and plan selection uses an exhaustive truth-blind search.

## Tests and completion

Tests cover route registration, plan/artifact agreement, generated script provenance, fallback
rendering, artifact-mode status, optional explanation behavior, and the UI footer. Completion
requires the API package, built UI, demo script, phase report, and artifact-backed plans to agree.
