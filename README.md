# Vulcan Proof — Pre-dispute evidence orchestration for Razorpay

Vulcan Proof is a decision service for prepaid physical-goods orders. It estimates the kinds of
dispute exposure an order presents, chooses evidence that is admissible and economically useful,
and coordinates collection before the relevant fulfillment window closes. The result is a
defense-only evidence plan that can be inspected by a merchant, is designed to plug into a tool
like Dispute Responder, and can be traced back to the order context and model outputs that
produced it.

The system is designed around a simple operational fact: a dispute team can use only evidence that
already exists. A delivery record, packing record, acknowledgement, or handoff artifact has to be
created while the order is still actionable. Vulcan Proof therefore makes the evidence decision at
payment time and keeps the later dispute package deterministic.

## What the system does

For each order, the service:

- reads permitted pre-dispatch and order-context features;
- estimates exposure, dispute type, contestability, evidence materialization, defensibility, and
  prevention signals;
- filters evidence by dispute admissibility, merchant availability, opt-in, and learned support;
- evaluates the complete evidence plan rather than making independent per-item decisions;
- returns the best plan, its expected value, standalone and incremental evidence values, and clear
  refusal reasons; and
- assembles materialized evidence into API slots when a dispute package is requested.

These six models do not operate independently; one optimizer combines their outputs into a single
evidence plan for each order.

The working evidence surface has 9 evidence types. The optimizer evaluates 512 evidence
combinations per order, and the decision path uses 3 prediction stages before dispatch. The
evidence types are weight, serial, sealed, packing, geotag, OTP, signature, acknowledgement, and
verified acknowledgement.

The current submission summary is:

- optimizer coverage: 53.24%;
- top-decile risk lift: 1.75×; and
- false-positive cost: ₹695.69 per 1,000 orders.

These figures describe the evaluation artifacts; the public Olist data supplies a detection anchor,
while the end-to-end economics and evidence behavior come from the hidden-truth simulator.
Production calibration requires Razorpay dispute history.

## Per-order decision flow

### Intake and exposure

The API accepts an observed order frame that has passed the schema gate. Features include order
context, merchant history derived from resolved outcomes, category and value bands, network and
issuer context, fulfillment signals available before dispatch, and the permitted evidence state.
Post-payment outcomes and simulator-only truth columns are rejected. Stage A estimates exposure,
the probability that the order will enter the dispute funnel.

### Dispute type

Stage B converts exposure into a calibrated distribution over the supported dispute classes:
non-receipt, not-as-described, and empty-box disputes. The distribution is used downstream instead
of selecting one class early, so a plan can be valuable across several plausible dispute paths.

### Contestability and evidence materialization

Stage C estimates the probability that the merchant will contest a dispute given the planned
evidence. This dependency matters because evidence can change behavior as well as the eventual
defense record. The materialization model estimates which requested evidence will actually become
available, including merchant compliance and acknowledgement response behavior.

The defensibility model estimates the win probability for a dispute type and materialized evidence
set. Low-support combinations use a conservative main-effect fallback, and unsupported
dispute/evidence pairs are removed from the action space. The prevention model separately estimates
the value of an acknowledgement that prevents a dispute before it opens.

### Exhaustive plan selection

For each admissible plan, the optimizer combines the Stage A exposure, Stage B dispute mix, Stage C
contestability, materialization expectation, defensibility uplift, prevention value, cash cost, and
merchant time. It enumerates materialization patterns inside each candidate plan and selects the
highest-value complete subset. The empty plan has zero expected value, so evidence is requested
only when the full plan remains positive after its costs and dependencies.

This search is exhaustive and truth-blind. It learns from observed orders and resolved outcomes;
the simulator's hidden fulfillment truth and assumed evidence uplifts are available only to the
resolver and evaluation harness. This boundary prevents the optimizer from choosing a plan using
information that will not exist in production.

### Fulfillment and dispute package

The selected plan is written with per-type expected value, incremental value, availability, support,
and refusal reason codes. Downstream capture systems can request the plan's evidence while the
order is actionable. If a dispute opens, the package endpoint maps the evidence that materialized
to the corresponding Razorpay contest-API slots. The plan endpoint also exposes the three-stage
predictions and the tuned comparison plan used by the evaluation arms.

## Architecture

The repository separates public-data detection, simulation, outcome resolution, model training,
optimization, reporting, and product serving:

| Area | Responsibility |
| --- | --- |
| `vulcan_proof/olist/` | Load the public Olist tables, build leakage-safe features and labels, and train the detection anchor. |
| `vulcan_proof/sim/` | Generate observed and hidden order worlds, including fulfillment truth, latency, evidence behavior, and censoring. |
| `vulcan_proof/economics.py` | Keep the money table, prevention gain, evidence costs, and resolver economics in pure shared functions. |
| `vulcan_proof/models/` | Fit the exposure, dispute type, contestability, materialization, defensibility, and prevention models from observed training data. |
| `vulcan_proof/opt/` | Apply schema gates, support masks, materialization expectation, and exhaustive truth-blind plan selection. |
| `vulcan_proof/arms/` | Run the tuned evidence policy and the learned optimizer for paired evaluation. |
| `vulcan_proof/sweep/` | Run signal, one-at-a-time, joint, and robustness sensitivity studies and build charts. |
| `vulcan_proof/api/` | Serve order plans, dispute packages, policy details, and the generated demo script through FastAPI. |
| `vulcan_proof/ui/` | Provide the React/Vite order, plan, and dispute-package views. |
| `outputs/` | Store manifests, model metrics, plans, reports, charts, and demo artifacts produced by runs. |

The project is organized as a sequence of phases. The Olist anchor establishes the real-data
detection boundary. The simulator creates a controlled observed/hidden world. The resolver builds
the outcome and history features used by the tuned policy. Model training and optimization then
operate only on observed columns, and the final product surface reads the resulting artifacts.

## Data boundary and evaluation design

Olist is a public Brazilian commerce dataset with order, seller, product, delivery, and review
records. It has no chargeback or evidence data, so it is used for detection behavior and leakage
checks, not for dispute economics or evidence-yield claims. Razorpay-specific production
calibration is a deployment requirement because the repository does not contain Razorpay dispute
history.

The simulator maintains two related views. The observed frame contains the features and outcomes a
production learner may use. The hidden frame contains fulfillment truth and the latent effects
needed to resolve the experiment. The resolver may read both views; model fitting and optimization
receive only the permitted observed schema. Historical labels obey the same maturity and censoring
boundary that would apply to a live dispute workflow.

Evaluation uses paired orders across the tuned policy and optimizer so that the comparison reflects
the same order context. Sensitivity runs vary simulator assumptions and record their scope in
manifests and reports. Reports distinguish the public-data detection anchor, simulator behavior,
and production prerequisites in plain language.

## Technology and runtime

The core stack is Python with pandas and NumPy for tabular data, LightGBM gradient-boosted trees
for the learned stages, and isotonic calibration for probability outputs. FastAPI exposes the
service, while React and Vite provide the local demonstration interface. Parquet artifacts keep
large intermediate frames separate from the source code, and JSON manifests make each run
reproducible and inspectable.

The runtime is CPU-only with no GPU and no paid APIs. The optional explanation route is disabled by
default and can render a sentence from an existing plan; it has no authority over the decision.
All decision-making remains in the local model and optimizer path.

## Install and run

From the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes --only-binary=:all: -r requirements.lock
.\.venv\Scripts\python.exe scripts\task.py verify-env
```

The environment check confirms that the interpreter is inside the project environment, locked
packages import correctly, the gradient-boosted-tree dependency can train, and the frozen EV
reference tests pass. The lock file is the installation source; use the task runner when it needs
to be regenerated.

To build and run the local product surface:

```powershell
npm ci --prefix vulcan_proof/ui
npm run dev:all
```

The browser opens the order view first. From there a judge can inspect a plan, compare evidence
types and refusal reasons, and open the dispute package for an order with materialized evidence.
The API also exposes the generated demo script and the tuned policy table. Phase reports remain
file-based artifacts rather than a separate validation screen in the UI.

Useful task-runner commands are `verify-env`, `test`, `lock`, and `check-phase`. Each phase runner
writes its own manifest and report under `outputs/`; the report is a human-readable companion to
the machine-readable metrics and parquet artifacts.

## Repository guide

- `docs/00_context.md` explains the information boundary, economics, and design rationale.
- `docs/01_claims.md` defines the language used by judge-facing surfaces.
- `docs/02_engineering_rules.md` records the invariants and the guard attached to each trap.
- `docs/03_phases.md` describes phase order, smoke contracts, and halt behavior.
- `docs/phase_0_olist.md` through `docs/phase_5_demo.md` describe implementation responsibilities.
- `docs/appendix_arithmetic.md` records the EV derivation and how the optimizer's objective is
  assembled.
- `params/params.yaml` is the source of runtime constants and sweep configuration.
- `scripts/` contains cross-platform runners and environment checks.
- `tests/` contains invariant, schema, firewall, reproducibility, and phase tests.

## Reproducibility and safety

Every random stream is derived from the run seed through the seed tree; modules do not use a
process-global random state. Manifests record the parameter digest, phase, seed context, artifact
paths, runtime details, and installed package information. The schema gate rejects hidden columns,
unknown features, and missing values before model fit or prediction.

The truth firewall is enforced both statically and at runtime. Simulation and resolution may use
hidden state to create outcomes, but model and optimizer packages cannot import it. This keeps the
decision service representative of the information available before dispatch.

## Scope and limitations

The supported workflow is prepaid physical goods and defense-only dispute handling. The simulator
is an evaluation instrument, not a substitute for Razorpay production calibration. Evidence
availability depends on merchant process and capture integrations, and the optional explanation
surface is descriptive rather than a source of decisions. A deployment should recalibrate the
models on governed Razorpay dispute history, verify evidence API mappings, and retain the same
pre-dispatch information boundary.
