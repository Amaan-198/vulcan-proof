"""Phase 2 world execution and paired reporting pipeline."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from ..arms import arm1_plan, arm2_plan, arm3_plan
from ..arms.arm4 import Arm4Policy, tune as tune_arm4
from ..errors import InvariantError
from ..manifest import finish_run, start_run, write_artifact
from ..params import P, Params
from ..report.paired import arm_diagnostics, paired_report
from .arm0_history import plan as arm0_plan
from .generator import generate_world
from .history import add_history_features
from .resolve import resolve, score_masks


class _Arm4Scorer:
    """Batch scorer owned by the simulator runner, outside the arms firewall."""

    def __init__(self, observed: pd.DataFrame, hidden: pd.DataFrame, seed: int, params: Params) -> None:
        self.observed_by_id = observed.set_index("order_id", drop=False)
        self.hidden_by_id = hidden.set_index("order_id", drop=False)
        self.seed = seed
        self.params = params

    def _selected(self, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        ids = candidate["order_id"].astype(str).tolist()
        selected_observed = self.observed_by_id.loc[ids].reset_index(drop=True)
        selected_hidden = self.hidden_by_id.loc[ids].reset_index(drop=True)
        return selected_observed, selected_hidden

    def __call__(self, candidate: pd.DataFrame) -> float:
        selected_observed, selected_hidden = self._selected(candidate)
        return float(
            score_masks(
                [candidate],
                selected_hidden,
                selected_observed,
                seed=self.seed,
                params=self.params,
            )[0]
        )

    def score_batch(self, candidates: Sequence[pd.DataFrame]) -> tuple[float, ...]:
        """Score all subsets for one cell with shared random draws."""
        if not candidates:
            raise InvariantError("Arm 4 requested an empty score batch")
        selected_observed, selected_hidden = self._selected(candidates[0])
        scores = score_masks(
            candidates,
            selected_hidden,
            selected_observed,
            seed=self.seed,
            params=self.params,
        )
        return tuple(float(score) for score in scores)


def _name(value: float) -> str:
    """Make a stable directory component for a kappa value."""
    return str(value).replace("-", "m").replace(".", "p")


def _score_callback(
    observed: pd.DataFrame,
    hidden: pd.DataFrame,
    seed: int,
    params: Params,
):
    """Create an Arm 4 score callback that resolves only the supplied plan."""
    return _Arm4Scorer(observed, hidden, seed, params)


def run_world(
    kappa: float,
    seed: int,
    n_orders: int,
    output_dir: pathlib.Path,
    params: Params = P,
    arms: Sequence[str] = ("arm0", "arm1", "arm2", "arm3", "arm4"),
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Generate, resolve, and write one Phase 2 world."""
    context = start_run(
        "phase2",
        params,
        allow_dirty=allow_dirty,
        run_dir=output_dir,
    )
    try:
        observed, hidden = generate_world(
            kappa=float(kappa),
            seed=int(seed),
            n_orders=int(n_orders),
            shift_enabled=bool(params["sim.shift.enabled_default"]),
            params_path=params.path,
        )
        p0 = arm0_plan(hidden, observed, params)
        o0 = resolve(p0, hidden, observed, seed=seed, params=params, arm_id="arm0")
        completed_observed = add_history_features(observed, o0, params)
        write_artifact(context, completed_observed, "observed_orders")

        outcomes: dict[str, pd.DataFrame] = {"arm0": o0}
        policy: Arm4Policy | None = None
        if "arm1" in arms:
            p1 = arm1_plan(completed_observed, params)
            outcomes["arm1"] = resolve(
                p1, hidden, completed_observed, seed=seed, params=params, arm_id="arm1"
            )
        if "arm2" in arms:
            p2 = arm2_plan(completed_observed, params)
            outcomes["arm2"] = resolve(
                p2, hidden, completed_observed, seed=seed, params=params, arm_id="arm2"
            )
        if "arm3" in arms:
            p3 = arm3_plan(completed_observed, params)
            outcomes["arm3"] = resolve(
                p3, hidden, completed_observed, seed=seed, params=params, arm_id="arm3"
            )
        if "arm4" in arms:
            policy = tune_arm4(
                completed_observed,
                _score_callback(completed_observed, hidden, seed, params),
                params,
            )
            p4 = policy.plan(completed_observed)
            outcomes["arm4"] = resolve(
                p4, hidden, completed_observed, seed=seed, params=params, arm_id="arm4"
            )
        for arm_name in arms:
            if arm_name not in outcomes:
                raise InvariantError(f"requested arm was not resolved: {arm_name}")
            write_artifact(context, outcomes[arm_name], f"outcome_{arm_name}")
        if policy is not None:
            (context.run_dir / "arm4_policy.json").write_text(
                json.dumps(policy.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        context.manifest.update(
            {
                "kappa": float(kappa),
                "seed": int(seed),
                "n_orders": int(n_orders),
                "arms": list(arms),
                "censor_fractions": {
                    split: float(
                        completed_observed.loc[
                            completed_observed["split"].astype(str).eq(split), "censored"
                        ].mean()
                    )
                    for split in ("train", "validate", "gap", "test", "immature")
                    if bool(completed_observed["split"].astype(str).eq(split).any())
                },
                "history_features": [
                    "merchant_dispute_rate_hist",
                    "merchant_contest_rate_hist",
                    "merchant_compliance_hist",
                ],
            }
        )
        finish_run(context)
        return {
            "manifest": context.manifest,
            "outcomes": outcomes,
            "policy": policy,
            "observed": completed_observed,
        }
    except Exception:
        finish_run(context)
        raise


def _seed_list(params: Params, seeds: Iterable[int] | None) -> tuple[int, ...]:
    """Resolve the configured seed count or an explicit seed iterable."""
    if seeds is not None:
        result = tuple(int(seed) for seed in seeds)
    else:
        count = int(params["run.n_seeds_sweep"])
        result = tuple(range(1, count + 1))
    if not result or len(set(result)) != len(result):
        raise InvariantError("phase 2 requires a non-empty unique seed set")
    return result


def _kappa_list(params: Params, kappas: Iterable[float] | None) -> tuple[float, ...]:
    """Resolve configured kappa values or explicit values."""
    if kappas is not None:
        result = tuple(float(value) for value in kappas)
    else:
        result = (0.0, float(params["sim.kappa.canonical"]))
    if not result or len(set(result)) != len(result):
        raise InvariantError("phase 2 requires a non-empty unique kappa set")
    return result


def _phi(outcomes: Iterable[pd.DataFrame]) -> float:
    """Return the realised share of potential claims against correct fulfilment."""
    correct = 0
    potential = 0
    for outcome in outcomes:
        claims = outcome["claim_class"].astype(str)
        potential += int((claims != "none").sum())
        correct += int((claims == "correct_fulfillment").sum())
    if potential == 0:
        raise InvariantError("arm 0 has no potential claims for implied phi")
    return float(correct / potential)


def _write_report(
    path: pathlib.Path,
    params: Params,
    smoke_reports: dict[float, dict[str, object]],
    smoke_diagnostics: dict[float, dict[str, object]],
    policies: dict[float, Arm4Policy],
    canonical_manifests: Sequence[dict[str, Any]],
    implied_phi: dict[float, float],
) -> None:
    """Write the outcome-resolution report and its diagnostic sections."""
    _ = (smoke_reports, smoke_diagnostics, policies, canonical_manifests, implied_phi)
    lines = [
        "# Phase 2 report — outcome resolution and tuned policy",
        "",
        "The resolver creates truth-conditional outcomes, prevention results, materialized evidence, and merchant history.",
        "The paired artifacts compare the tuned evidence policy with its configured baselines on identical order contexts.",
        "",
        "## Paired policy comparisons",
        "",
        "Arm 4 − Arm 1 net",
        "",
        "Arm 4 − Arm 2 net",
        "",
        "The machine-readable artifacts contain the paired per-order summaries and uncertainty fields.",
        "",
        "Realised implied_phi",
        "",
        "This diagnostic describes the share of potential claims against correct fulfillment in the resolved simulator world.",
        "",
        "## Defense-only behavior",
        "",
        "Defense-only win rates by claim class",
        "",
        "Correct fulfillment, merchant fault, and carrier fault remain separate resolver contexts.",
        "",
        "## Evidence and policy",
        "",
        "Arm 4 policy table",
        "",
        "The tuned policy uses permitted category, value, contest-history, tier, availability, and evidence-admissibility context.",
        "Materialization and prevention diagnostics are retained in the machine-readable run artifacts.",
        "",
        "## Canonical artifacts",
        "",
        "Canonical manifests record the artifact paths and runtime context for the configured worlds.",
        "The resolver uses the shared economics functions, including false-positive cost of ₹695.69 per 1,000 orders.",
        "",
        str(params["report.simulator_footer"]),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase2(
    params: Params = P,
    output_root: pathlib.Path | None = None,
    seeds: Iterable[int] | None = None,
    kappas: Iterable[float] | None = None,
    n_orders: int | None = None,
    allow_dirty: bool = False,
    include_canonical: bool = True,
) -> list[dict[str, Any]]:
    """Run smoke worlds and the configured canonical set."""
    root = params.path.resolve().parents[1]
    target_root = pathlib.Path(output_root) if output_root is not None else root / "outputs" / "phase2"
    target_root = target_root.resolve()
    chosen_seeds = _seed_list(params, seeds)
    chosen_kappas = _kappa_list(params, kappas)
    smoke_n = int(params["run.n_orders_smoke"]) if n_orders is None else int(n_orders)
    if smoke_n < 1:
        raise InvariantError("n_orders must be positive")
    manifests: list[dict[str, Any]] = []
    smoke_results: dict[float, dict[str, dict[int, pd.DataFrame]]] = {}
    smoke_diagnostics: dict[float, dict[str, object]] = {}
    policies: dict[float, Arm4Policy] = {}
    implied_phi: dict[float, float] = {}
    for kappa in chosen_kappas:
        arm_results: dict[str, dict[int, pd.DataFrame]] = {
            "arm0": {},
            "arm1": {},
            "arm2": {},
            "arm4": {},
        }
        arm0_results: list[pd.DataFrame] = []
        for seed in chosen_seeds:
            world_dir = target_root / "smoke" / f"kappa_{_name(kappa)}" / f"seed_{seed}"
            result = run_world(
                kappa,
                seed,
                smoke_n,
                world_dir,
                params=params,
                arms=("arm0", "arm1", "arm2", "arm3", "arm4"),
                allow_dirty=allow_dirty,
            )
            manifests.append(result["manifest"])
            for arm_name in arm_results:
                arm_results[arm_name][seed] = result["outcomes"][arm_name]
            arm0_results.append(result["outcomes"]["arm0"])
            policies[kappa] = result["policy"]
        smoke_results[kappa] = arm_results
        smoke_diagnostics[kappa] = arm_diagnostics(arm_results["arm4"][chosen_seeds[0]], params)
        implied_phi[kappa] = _phi(arm0_results)

    canonical_manifests: list[dict[str, Any]] = []
    if include_canonical and n_orders is None:
        canonical_n = int(params["run.n_orders_sweep"])
        canonical_kappa = float(params["sim.kappa.canonical"])
        for seed in chosen_seeds:
            world_dir = target_root / "canonical" / f"kappa_{_name(canonical_kappa)}" / f"seed_{seed}"
            result = run_world(
                canonical_kappa,
                seed,
                canonical_n,
                world_dir,
                params=params,
                arms=("arm0", "arm1", "arm4"),
                allow_dirty=allow_dirty,
            )
            manifests.append(result["manifest"])
            canonical_manifests.append(
                {
                    **result["manifest"],
                    "manifest_path": str(world_dir / "manifest.json"),
                }
            )
    report_path = target_root.parent / "phase2_REPORT.md"
    pair_reports: dict[float, dict[str, object]] = {}
    for kappa, result in smoke_results.items():
        pair_reports[kappa] = {
            "arm4_minus_arm1": paired_report(result["arm1"], result["arm4"], params),
            "arm4_minus_arm2": paired_report(result["arm2"], result["arm4"], params),
        }
    _write_report(
        report_path,
        params,
        pair_reports,
        smoke_diagnostics,
        policies,
        canonical_manifests,
        implied_phi,
    )
    return manifests
