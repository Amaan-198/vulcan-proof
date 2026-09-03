# Olist detection anchor

This phase establishes the real-data boundary and the shared infrastructure used by the later
simulator and optimizer phases. It is deliberately narrow: Olist has no chargeback or evidence
data, so it supports detection, feature timing, calibration, manifests, and leakage checks rather
than dispute-economics conclusions.

## Why this phase exists

The public dataset provides realistic order, seller, product, delivery, and review relationships.
Those relationships are useful for checking whether a pre-dispatch feature pipeline can identify
delivery-related exposure without reading later outcomes. The phase produces a temporal holdout,
calibrated detector artifacts, reliability data, and a report that states the limits of the source.

## Infrastructure

The shared infrastructure includes:

- `params.py`, which loads the centralized parameter contract and exposes its digest;
- `envcheck.py`, which requires the project environment and checks the locked runtime;
- `errors.py`, which defines schema, invariant, and leakage failures;
- `schemas.py`, which validates observed, hidden, outcome, and artifact frames;
- `manifest.py`, which records run identity, parameter provenance, artifact paths, and runtime data;
- `seeds.py`, which creates named child streams for deterministic work;
- `scripts/check_phase.py`, which checks phase artifacts without changing them; and
- the infrastructure tests for parameters, schemas, manifests, seeds, and reproducibility.

Every artifact is written through the same helpers used by later phases. This keeps the data
contract visible and gives a reviewer a direct path from a report statement to the source artifact.

## Olist data path

The loader reads the committed or downloaded Olist tables and normalizes identifiers and dates.
The label builder defines the detection target from fields that are valid at the declared decision
boundary. Feature construction joins order, seller, product, delivery, and review context without
allowing a later outcome to become a feature.

The split builder creates a time-ordered train, validation, and test arrangement. Maturity and
excluded-status handling are applied before fitting. The temporal test frame is held for final
diagnostics, while isotonic calibration is learned on validation data.

## Detector and report artifacts

The detector is a gradient-boosted-tree model over the permitted Olist feature frame. Evaluation
writes probability metrics, reliability data, label summaries, split summaries, and temporal drift
diagnostics to `outputs/phase0/metrics.json`. Charts are generated from those artifacts, and the
human-readable report preserves the detection boundary in its footer.

The report includes the required statement: “Olist has no chargeback or evidence data; this measures
detection only.” It describes calibration transfer and temporal behavior without presenting Olist as
a source of dispute or evidence claims.

## Tests and completion

Tests cover label timing, feature leakage, split identity, schema validity, manifest contents, seed
reproducibility, and the public-data claim boundary. The phase is complete when the detector,
artifacts, report, charts, and infrastructure checks agree and the parameter file remains unchanged.

The later simulator phases may consume the shared infrastructure, but they must not reinterpret the
Olist detector as production Razorpay calibration.
