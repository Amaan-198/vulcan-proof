# Vulcan Proof — Pre-Dispute Evidence Orchestration for Razorpay

**Razorpay AI Buildathon · Track 02: AI Risk Manager · solo build · Windows 11 native**

> Dispute Responder can only fight with evidence that exists. Vulcan Proof makes sure the right
> evidence exists before it becomes impossible to collect.

## The honest claim

Vulcan Proof predicts which fulfillment disputes a prepaid physical-goods order is exposed to,
decides which admissible evidence is economically worth collecting, and orchestrates its capture
before the physical window closes.

Pre-build arithmetic (`docs/00_context.md` §4) established, before any code was written:

1. **The orchestration layer is the product.** Recommending free pre-dispatch evidence and a ₹0.30
   acknowledgement is worth ₹10k–50k per 1,000 orders in electronics and jewellery under the
   simulator's parameters, with zero ML.
2. **The ML is a 5–12% improvement on that**, and only when individual-risk signal exists. At zero
   signal (κ = 0) a tuned category × value × contest-band rule captures 99.5%+ of achievable value.
3. **κ\*** — the minimum signal strength at which per-order optimisation beats the tuned rule with a
   confidence interval excluding zero — is the number this repo exists to measure. If κ\* does not
   exist in [0, 1], the ML claim is dropped and the orchestration layer ships alone.
4. **Paid handoff evidence (OTP, signature, geotag) is an electronics-and-jewellery capability.**

## What is proven vs simulated

| Component | Status | Where |
|---|---|---|
| Detection generalises to real orders | **Measured** on the public Olist dataset, temporal holdout | Phase 0 |
| Dispute economics, evidence yields, optimizer value, κ\* | **Simulated** in a hidden-truth world the optimizer cannot see | Phases 1–4 |
| Production calibration | **Requires Razorpay dispute history** — not available | — |

Every ₹ figure is a simulator result and is labelled as such.

## Environment — Python 3.13, Windows 11, no WSL, no GPU

The stack runs natively on 64-bit Windows under CPython **3.13**. Every dependency in
`requirements.lock` was resolved by `pip download --only-binary=:all: --platform win_amd64
--python-version 3.13` against PyPI on 2026-08-26 and is pinned by SHA-256. LightGBM's wheel is
`py3-none-win_amd64`; nothing builds from source; nothing touches CUDA. The RTX 5070 Ti is unused
by design — the workload is LightGBM on tabular data and stays on CPU.

**Install (paste into PowerShell from the repo root):**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip==25.2"
python -m pip install --require-hashes --only-binary=:all: -r requirements.lock
python scripts\task.py verify-env
```

`verify-env` asserts: interpreter is 3.13.x, `sys.prefix` is inside `.venv`, every locked package
imports at its locked version, LightGBM trains a 100-row model, and `pytest tests\test_ev_reference.py`
passes. If any step fails, the install is wrong; nothing in this repo runs against the system Python
and every script refuses to start outside a venv (`vulcan_proof/envcheck.py`).

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
`requirements.in` lists the 14 direct dependencies; `requirements.lock` is the only file pip reads.
Do not edit the lock by hand; regenerate it with `python scripts\task.py lock` (documented there) and
commit the diff.

**Memory.** The 3M-order canonical run peaks at roughly 5–6 GB RSS (observed frame ≈ 250 MB with
declared dtypes, hidden ≈ 180 MB, one OUTCOME frame ≈ 150 MB per arm, LightGBM binned dataset
≈ 600 MB, optimizer EV matrix chunked at 100k orders × 512 subsets × 8 B ≈ 400 MB per chunk). Every
manifest records `peak_rss_mb` via `psutil`; `check-phase` fails if it exceeds `run.max_peak_rss_gb`
(16 GB) so a memory regression is caught, not paged.

## Repository layout

```
README.md                        this file
docs/00_context.md               WHY each constraint exists — read first, always
docs/01_claims.md                say / never-say, machine-checked
docs/02_engineering_rules.md     non-negotiable coding rules, traps, and the guard for each
docs/03_phases.md                phase order, halt protocol, done-criteria index
docs/phase_0_olist.md            infrastructure + real-data detection anchor
docs/phase_1_simulator.md        hidden-truth world generator
docs/phase_2_arms.md             resolver, prevention, history features, Arms 0–4
docs/phase_3_models_optimizer.md Stages A/B/C, defensibility, optimizer (Arm 5)
docs/phase_4_sweeps.md           κ-sweep, κ*, OAT, LHS, kill condition
docs/phase_5_demo.md             product surface, scripted from Phase 4 output
docs/appendix_arithmetic.md      pre-build arithmetic (provenance of known answers)
params/params.yaml               EVERY constant: value, unit, source, sweep range, sensitivity rank
requirements.in / requirements.lock
scripts/task.py                  cross-platform task runner (replaces make)
vulcan_proof/ev_reference.py     frozen known-answer oracle
tests/test_ev_reference.py       known-answer tests (pass now)
```

## Phase 5 demo

Install the UI dependencies once from the repository root, then use one command to run both the
Vite frontend and FastAPI backend:

```powershell
npm ci --prefix vulcan_proof/ui
npm run dev:all
```

Open `http://localhost:5173`. `dev:all` generates the Phase 5 demo artefacts when they are missing,
starts Vite on port 5173, and starts FastAPI on port 8765 with the Vite API proxy. The demo is
laptop-first. The generated walkthrough records whether the Phase 4 production-scale validation is
available; the current buildathon artefacts are smoke-only. The optional `/explain` route is disabled
unless `VP_EXPLAIN_LLM=1`; it cannot change a plan.

## How to work in this repo

Point a coding agent at the repo and say: **"Read `docs/00_context.md`, `docs/02_engineering_rules.md`,
`params/params.yaml`, and `docs/phase_N_*.md`. Implement Phase N. Stop when its done-criteria pass."**
The agent completes the phase, runs `python scripts\task.py check-phase N`, writes
`outputs\phaseN_REPORT.md`, and **halts**.

## Reproducibility contract

- One master seed (`run.master_seed = 20260826`). Every random stream is spawned from it via
  `numpy.random.SeedSequence`; no module calls `np.random.seed` or the global RNG.
- Every run writes `outputs/<run_id>/manifest.json`: git commit, `params.yaml` SHA-256, master seed,
  phase, timestamp, wall seconds, peak RSS, installed package versions. Two runs with equal
  manifests (minus timestamp/wall/rss) produce byte-identical parquet outputs.
- No parameter exists outside `params/params.yaml`.

No LLM in any decision path. An optional explanation sentence in Phase 5 is off by default.
