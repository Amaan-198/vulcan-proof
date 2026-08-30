"""Read-only Phase-5 service over the validated phase artefacts."""

from __future__ import annotations

import json
import pathlib
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..errors import InvariantError
from ..models import fit_models
from ..opt.optimizer import available_mask, best_plan, bit_for, ev_set, evidence_names, plan_frame
from ..params import P, Params
from ..schemas import check


ROOT = pathlib.Path(__file__).resolve().parents[2]
DEMO_SEED = 2


def _world_name(value: float) -> str:
    """Return the stable directory token used by the phase runners."""
    return str(value).replace("-", "m").replace(".", "p")


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    """Read one JSON object with a useful error when an artefact is malformed."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise InvariantError(f"expected a JSON object: {path}")
    return payload


def _json_value(value: Any) -> Any:
    """Convert pandas and numpy values into JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


class Phase5Service:
    """Load Phase-3/4 artefacts and expose a small, deterministic demo API."""

    def __init__(self, root: pathlib.Path | None = None, params: Params = P) -> None:
        self.root = (pathlib.Path(root) if root is not None else ROOT).resolve()
        self.params = params
        self._models: Any | None = None
        self._model_lock = threading.RLock()
        self._load_world()

    def _load_world(self) -> None:
        """Select the canonical demo world, falling back only when it is absent."""
        kappa = float(self.params["sim.kappa.canonical"])
        branches = ("canonical", "smoke")
        candidates: list[pathlib.Path] = []
        for branch in branches:
            base = self.root / "outputs" / "phase3" / branch / f"kappa_{_world_name(kappa)}"
            preferred = base / f"seed_{DEMO_SEED}" / "observed_orders.parquet"
            if preferred.is_file():
                candidates.append(preferred)
        if not candidates:
            for branch in branches:
                base = self.root / "outputs" / "phase3" / branch / f"kappa_{_world_name(kappa)}"
                candidates.extend(sorted(base.glob("seed_*/observed_orders.parquet")))
        if not candidates:
            raise FileNotFoundError("Phase-3 observed artefact is missing")
        observed_path = candidates[0]
        self.phase3_dir = observed_path.parent
        self.branch = observed_path.parents[2].name
        self.seed = int(observed_path.parent.name.removeprefix("seed_"))
        self.phase3_observed_path = observed_path
        self.phase3_outcome_path = observed_path.parent / "outcome_arm5.parquet"
        if not self.phase3_outcome_path.is_file():
            raise FileNotFoundError(self.phase3_outcome_path)

        phase2_branch = self.branch
        phase2_dir = self.root / "outputs" / "phase2" / phase2_branch / f"kappa_{_world_name(kappa)}" / f"seed_{self.seed}"
        if not (phase2_dir / "observed_orders.parquet").is_file():
            phase2_dir = self.root / "outputs" / "phase2" / "smoke" / f"kappa_{_world_name(kappa)}" / f"seed_{self.seed}"
        self.phase2_dir = phase2_dir
        self.phase2_observed_path = phase2_dir / "observed_orders.parquet"
        self.phase2_historical_path = phase2_dir / "outcome_arm0.parquet"
        self.phase2_arm4_path = phase2_dir / "outcome_arm4.parquet"
        self.policy_path = phase2_dir / "arm4_policy.json"

        self.observed = pd.read_parquet(self.phase3_observed_path)
        self.arm5_outcome = pd.read_parquet(self.phase3_outcome_path)
        check(self.observed, "ORDER_OBSERVED")
        check(self.arm5_outcome, "OUTCOME")
        self.arm5_by_id = self.arm5_outcome.set_index("order_id")
        if self.phase2_arm4_path.is_file():
            self.arm4_outcome = pd.read_parquet(self.phase2_arm4_path)
            check(self.arm4_outcome, "OUTCOME")
            self.arm4_by_id = self.arm4_outcome.set_index("order_id")
        else:
            self.arm4_outcome = None
            self.arm4_by_id = None
        self.metrics = _read_json(self.root / "outputs" / "phase3" / "metrics.json")

    def _test_rows(self) -> pd.DataFrame:
        """Return the canonical test split."""
        return self.observed.loc[self.observed["split"].astype(str).eq("test")].reset_index(drop=True)

    def _row(self, order_id: str) -> pd.Series:
        """Return one test row or raise a route-friendly lookup error."""
        rows = self.observed.loc[
            self.observed["order_id"].astype(str).eq(str(order_id))
            & self.observed["split"].astype(str).eq("test")
        ]
        if rows.empty:
            raise KeyError(order_id)
        return rows.iloc[0]

    def ensure_models(self) -> Any:
        """Fit the exact Phase-3 model bundle once, lazily."""
        if self._models is None:
            with self._model_lock:
                if self._models is None:
                    if not self.phase2_historical_path.is_file() or not self.phase2_observed_path.is_file():
                        raise FileNotFoundError("Phase-2 training artefacts are missing")
                    phase2_observed = pd.read_parquet(self.phase2_observed_path)
                    historical = pd.read_parquet(self.phase2_historical_path)
                    check(phase2_observed, "ORDER_OBSERVED")
                    check(historical, "OUTCOME")
                    self._models = fit_models(phase2_observed, historical, self.params, self.seed)
        return self._models

    def orders(
        self,
        category: str | None = None,
        query: str | None = None,
        limit: int | None = None,
        plans_only: bool = False,
        package_ready_only: bool = False,
    ) -> dict[str, Any]:
        """Return compact test-order rows for the order picker."""
        frame = self._test_rows()
        if category and category != "All":
            frame = frame.loc[frame["category"].astype(str).eq(category)]
        if query:
            needle = query.lower()
            frame = frame.loc[
                frame["order_id"].astype(str).str.lower().str.contains(needle, regex=False)
                | frame["merchant_id"].astype(str).str.lower().str.contains(needle, regex=False)
            ]
        if plans_only or package_ready_only:
            stored_masks = self.arm5_by_id["requested_bitmask"]
            has_plan = frame["order_id"].map(stored_masks).fillna(0).astype("int64").ne(0)
            frame = frame.loc[has_plan]
        if package_ready_only:
            opened = frame["order_id"].map(self.arm5_by_id["dispute_opened"]).fillna(False).astype(bool)
            materialised = frame["order_id"].map(self.arm5_by_id["materialised_bitmask"]).fillna(0).astype("int64").ne(0)
            frame = frame.loc[opened & materialised]
        total = len(frame)
        count = max(0, min(int(limit) if limit is not None else 100, 1000))
        frame = self._picker_sample(frame, count)
        result = []
        for _, row in frame.iterrows():
            order_id = str(row["order_id"])
            outcome = self.arm5_by_id.loc[order_id]
            has_plan = int(outcome["requested_bitmask"]) != 0
            package_available = has_plan and bool(outcome["dispute_opened"]) and int(outcome["materialised_bitmask"]) != 0
            result.append(
                {
                    "order_id": order_id,
                    "merchant_id": str(row["merchant_id"]),
                    "category": str(row["category"]),
                    "order_value": float(row["order_value"]),
                    "eligible_tier": str(row["eligible_tier"]),
                    "decision_date": int(row["decision_date"]),
                    "has_plan": has_plan,
                    "package_available": package_available,
                }
            )
        return {
            "orders": result,
            "total": int(total),
            "source": self.branch,
            "plans_only": bool(plans_only),
            "package_ready_only": bool(package_ready_only),
        }

    def _picker_sample(self, frame: pd.DataFrame, count: int) -> pd.DataFrame:
        """Return a useful, deterministic sample for the order picker.

        The test slice is value-sorted, so taking its first page hides nearly all
        stored plans: the cheapest orders quite reasonably have an empty plan.
        Keep a value-spread sample while reserving half the page for orders whose
        stored Arm 5 outcome requests evidence.
        """
        if count == 0 or frame.empty:
            return frame.iloc[:0]

        ordered = frame.sort_values(["category", "order_value", "order_id"], kind="mergesort").reset_index(drop=True)
        stored_masks = self.arm5_by_id["requested_bitmask"]
        opened = ordered["order_id"].map(self.arm5_by_id["dispute_opened"]).fillna(False).astype(bool)
        materialised = ordered["order_id"].map(self.arm5_by_id["materialised_bitmask"]).fillna(0).astype("int64").ne(0)
        has_plan = ordered["order_id"].map(stored_masks).fillna(0).astype("int64").ne(0)
        ordered = ordered.assign(
            _has_plan=has_plan,
            _package_ready=has_plan & opened & materialised,
        )
        if len(ordered) <= count:
            picked = ordered
        else:
            package_ready = ordered.loc[ordered["_package_ready"]]
            package_count = min(len(package_ready), count)
            package_positions = np.linspace(0, len(package_ready) - 1, num=package_count, dtype=int) if package_count else []
            picked_package = package_ready.iloc[np.unique(package_positions)] if package_count else package_ready.iloc[:0]

            remaining = ordered.drop(index=picked_package.index)
            with_plan = remaining.loc[remaining["_has_plan"]]
            plan_count = min(len(with_plan), max(0, count // 2 - len(picked_package)))
            plan_positions = np.linspace(0, len(with_plan) - 1, num=plan_count, dtype=int)
            picked_plan = with_plan.iloc[np.unique(plan_positions)]

            remaining = remaining.drop(index=picked_plan.index)
            remaining_count = count - len(picked_package) - len(picked_plan)
            value_positions = np.linspace(0, len(remaining) - 1, num=remaining_count, dtype=int)
            picked_value = remaining.iloc[np.unique(value_positions)] if remaining_count else remaining.iloc[:0]
            picked = pd.concat([picked_package, picked_plan, picked_value])

        return picked.sort_values(
            ["_package_ready", "_has_plan", "category", "order_value", "order_id"],
            ascending=[False, False, True, True, True],
            kind="mergesort",
        ).drop(columns=["_has_plan", "_package_ready"])

    def _stored_mask(self, order_id: str, arm: str = "arm5") -> int:
        """Read a stored plan mask from a phase outcome artefact."""
        lookup = self.arm5_by_id if arm == "arm5" else self.arm4_by_id
        if lookup is None or order_id not in lookup.index:
            return 0
        return int(lookup.loc[order_id, "requested_bitmask"])

    def plan(self, order_id: str) -> dict[str, Any]:
        """Return the stored plan plus model diagnostics for one test order."""
        row = self._row(order_id)
        stored_mask = self._stored_mask(order_id, "arm5")
        outcome = self.arm5_by_id.loc[order_id]
        package_available = stored_mask != 0 and bool(outcome["dispute_opened"]) and int(outcome["materialised_bitmask"]) != 0
        models = self.ensure_models()
        result = best_plan(row, models, models.support_mask, self.params)
        model_row = self.observed.loc[[row.name]]
        model_mask = int(plan_frame(model_row, models, models.support_mask, self.params)[0])
        selected_mask = stored_mask
        type_probabilities = models.pB(row)
        exposure = float(models.pA(row))
        held = selected_mask
        held_contest = float(models.pC(row, held))
        evidence_items: list[dict[str, Any]] = []
        availability = int(available_mask(row.to_frame().T, self.params)[0])
        support = models.support_mask
        for name in evidence_names(self.params):
            bit = int(bit_for(name, self.params))
            metadata = self.params[f"evidence.{name}"]
            selected = bool(selected_mask & bit)
            available = bool(availability & bit)
            admitted = any(
                float(probability) > 0.0
                and str(dispute_type) in metadata["admissible"]
                and support.pair_allowed(str(dispute_type), name)
                for dispute_type, probability in type_probabilities.items()
            )
            reason = "SELECTED" if selected else result.reasons.get(name, "NO_SUPPORT")
            evidence_items.append(
                {
                    "name": name,
                    "label": name.replace("_", " ").title(),
                    "window": str(metadata["window"]),
                    "api_slot": str(metadata["api_slot"]),
                    "cash_cost": float(metadata["cash"]),
                    "seconds": float(metadata["seconds"]),
                    "available": available,
                    "admissible": admitted,
                    "selected": selected,
                    "standalone_ev": _json_value(result.standalone_ev.get(name)),
                    "incremental_ev": _json_value(result.incremental_ev.get(name)),
                    "reason": reason,
                }
            )

        arm4_mask = self._stored_mask(order_id, "arm4")
        response = {
            "order": {
                "order_id": str(row["order_id"]),
                "merchant_id": str(row["merchant_id"]),
                "category": str(row["category"]),
                "order_value": float(row["order_value"]),
                "payment_method": str(row["payment_method"]),
                "network": str(row["network"]),
                "eligible_tier": str(row["eligible_tier"]),
                "decision_date": int(row["decision_date"]),
            },
            "plan": {
                "requested_bitmask": selected_mask,
                "evidence": [item["name"] for item in evidence_items if item["selected"]],
                "ev": float(ev_set(row, selected_mask, models, self.params)),
                "model_requested_bitmask": model_mask,
            },
            "package_available": package_available,
            "dispute_opened": bool(outcome["dispute_opened"]),
            "materialised_bitmask": int(outcome["materialised_bitmask"]),
            "evidence": evidence_items,
            "stages": {
                "exposure_probability": exposure,
                "dispute_type_probabilities": {str(key): float(value) for key, value in type_probabilities.items()},
                "contest_probability_with_plan": held_contest,
                "materialisation_probability": {
                    item["name"]: float(models.pM(row, item["name"])) for item in evidence_items
                },
                "win_probability_with_plan": {
                    dispute_type: float(models.pW(row, dispute_type, held))
                    for dispute_type in ("NR", "NAD", "EB")
                },
            },
            "comparison": {
                "arm4": {
                    "requested_bitmask": arm4_mask,
                    "evidence": [
                        name for name in evidence_names(self.params) if arm4_mask & int(bit_for(name, self.params))
                    ],
                },
                "stored_plan_source": str(self.phase3_outcome_path),
                "model_recomputed_mask_matches_stored": model_mask == stored_mask,
            },
            "simulator_footer": str(self.params["report.simulator_footer"]),
        }
        return _json_value(response)

    def dispute_package(self, order_id: str) -> dict[str, Any]:
        """Return materialised evidence for a test dispute."""
        row = self._row(order_id)
        outcome = self.arm5_by_id.loc[order_id]
        if not bool(outcome["dispute_opened"]):
            raise ValueError("order has no opened dispute in the stored Arm 5 outcome")
        materialised = int(outcome["materialised_bitmask"])
        requested = int(outcome["requested_bitmask"])
        items = []
        for name in evidence_names(self.params):
            bit = int(bit_for(name, self.params))
            if not materialised & bit:
                continue
            item = self.params[f"evidence.{name}"]
            items.append(
                {
                    "evidence": name,
                    "label": name.replace("_", " ").title(),
                    "api_slot": str(item["api_slot"]),
                    "window": str(item["window"]),
                    "requested": bool(requested & bit),
                    "captured": True,
                }
            )
        return _json_value(
            {
                "order_id": str(row["order_id"]),
                "category": str(row["category"]),
                "order_value": float(row["order_value"]),
                "dispute_type": str(outcome["dispute_type"]),
                "arm": "arm5",
                "items": items,
                "requested_bitmask": requested,
                "materialised_bitmask": materialised,
                "provenance": {
                    "bound_to_order": str(row["order_id"]),
                    "source": "stored Phase 3 simulator outcome",
                    "decision_day": int(row["decision_date"]),
                },
                "simulator_footer": str(self.params["report.simulator_footer"]),
            }
        )

    def phase4_status(self) -> dict[str, Any]:
        """Return extended Phase-4 data or the explicit buildathon status."""
        phase4 = self.root / "outputs" / "phase4"
        chart_names = (
            "01_kappa_net.png",
            "02_kappa_arm5_minus_arm4.png",
            "03_oat_tornado.png",
            "04_lhs_scatter.png",
            "05_recommendation_boundaries.png",
            "06_defense_only.png",
            "07_prevention_vs_defense.png",
            "08_coverage_friction.png",
            "09_lorenz.png",
        )
        required = [
            phase4 / "kappa_star.json",
            self.root / "outputs" / "phase4_REPORT.md",
            phase4 / "oat.json",
            phase4 / "lhs.json",
            phase4 / "robustness.json",
        ] + [phase4 / name for name in chart_names]
        if all(path.is_file() for path in required):
            payload = _read_json(phase4 / "kappa_star.json")
            payload.update(
                {
                    "validation_scope": "extended_validation",
                    "production_sweep": "completed",
                    "production_results_available": True,
                    "chart_urls": [f"/phase4/{name}" for name in chart_names],
                }
            )
            return _json_value(payload)
        phase3_pair = self.metrics.get("paired_arm5_minus_arm4", {})
        central_key = str(self.params["sim.kappa.canonical"])
        central = phase3_pair.get(central_key, {}) if isinstance(phase3_pair, Mapping) else {}
        return _json_value(
            {
                "validation_scope": "buildathon_smoke",
                "smoke_validation": "completed",
                "production_sweep": "deferred",
                "production_results_available": False,
                "message": "Production-scale robustness validation is deferred; smoke validation is available.",
                "smoke_simulator_result": {
                    "label": "Smoke simulator validation",
                    "kappa": float(self.params["sim.kappa.canonical"]),
                    "arm5_minus_arm4_net_per_1000": central.get("mean"),
                    "ci_low": central.get("ci_low"),
                    "ci_high": central.get("ci_high"),
                },
                "phase0": self.phase0_summary(),
                "simulator_footer": str(self.params["report.simulator_footer"]),
            }
        )

    def arm4_policy(self) -> dict[str, Any]:
        """Return the stored Arm-4 policy table."""
        if not self.policy_path.is_file():
            return {"available": False, "cells": []}
        payload = _read_json(self.policy_path)
        payload["available"] = True
        payload["source"] = str(self.policy_path)
        return _json_value(payload)

    def phase0_summary(self) -> dict[str, Any]:
        """Return the small Olist detection anchor used by the demo script."""
        metrics_path = self.root / "outputs" / "phase0" / "metrics.json"
        if not metrics_path.is_file():
            return {"available": False}
        metrics = _read_json(metrics_path)
        return {
            "available": True,
            "pr_auc": metrics.get("pr_auc"),
            "roc_auc": metrics.get("roc_auc"),
            "brier": metrics.get("brier"),
            "top_decile_lift": metrics.get("top_decile_lift"),
            "footer": str(self.params["report.olist_footer"]),
        }

    def demo_script(self) -> dict[str, Any]:
        """Build the demo narrative from the same stored artefacts."""
        from .demo_script import build_demo_script

        return build_demo_script(self.root, self)
