# Review changes and design decisions

This note records the design decisions that shape the current repository. It is intended to help
a reviewer understand why the implementation has separate simulation, resolution, learning, and
serving boundaries.

## Platform and dependency choices

The project targets native Windows execution. Python dependencies are pinned in the lock file and
are installed from binary distributions. The core learner uses gradient-boosted trees, probabilities
are post-processed with isotonic calibration, and the runtime is CPU-only with no GPU and no paid
APIs. The task runner keeps environment checks, tests, locking, and phase checks reproducible.

## Information-boundary decisions

The simulator writes an observed frame and a hidden frame. The resolver can use hidden fulfillment
truth to create outcomes, while model and optimizer packages can use only permitted observed
features. Static import checks and runtime schema checks protect this boundary.

Merchant history is derived from resolved observed outcomes. It is never initialized from a hidden
merchant propensity. Labels follow the maturity boundary, and censored disputes remain separate
from negative outcomes.

## Economic decisions

The evidence choice is a plan-level expected-value problem. Cash cost follows compliance, request
time follows the recommendation, and prevention is evaluated as a separate path from post-dispute
defense. Contestability depends on the planned evidence because the existence of a package can
change whether a merchant contests.

The action space contains 9 evidence types. The optimizer evaluates 512 evidence combinations per
order and uses 3 prediction stages before dispatch. Its search is exhaustive and truth-blind, so it
cannot use fulfillment truth or latent evidence effects that are unavailable at payment time.

## Evaluation decisions

The public Olist dataset is a detection anchor and has no chargeback or evidence data. The hidden-
truth simulator supplies the controlled end-to-end evaluation context. Paired order contexts are
used when comparing the tuned policy and optimizer. Reports identify the source and scope of each
result so that the production boundary remains clear.

The current summary is optimizer coverage of 53.24%, top-decile risk lift of 1.75×, and
false-positive cost of ₹695.69 per 1,000 orders. Production calibration requires Razorpay dispute
history.

## Product-surface decisions

The API reads completed model and evaluation artifacts instead of recomputing them for the demo.
The UI follows the order, plan, and dispute-package flow. Optional explanation text is descriptive
and has no decision authority. When extended sensitivity artifacts are unavailable, the demo carries
an explicit deferred status and uses the documented fallback copy.

## Current review state

The repository's Markdown reports, design notes, and judge-facing copy describe the same architecture
and the same operational boundary. Generated reports remain owned by their pipeline functions, while
the documentation explains how to interpret the corresponding machine-readable artifacts.
