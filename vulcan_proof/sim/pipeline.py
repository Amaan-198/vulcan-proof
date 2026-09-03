"""Phase-1 calibration, canonical run, and reporting pipeline."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pandas as pd

from ..manifest import finish_run, start_run, write_artifact
from ..params import P, Params, load
from ..schemas import check
from .calibrate import calibrate_funnel
from .disputes import potential_probabilities
from .generator import generate_world


def _calibration_path(params: Params) -> pathlib.Path:
    """Return the repository-level derived calibration path."""
    return params.path.resolve().parents[1] / "outputs" / "theta.json"


def _summary(
    observed: pd.DataFrame,
    hidden: pd.DataFrame,
    params: Params,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Compute report-only diagnostics from the two Phase-1 frames."""
    probability, _, cap_count = potential_probabilities(
        observed,
        hidden["hidden_truth"],
        hidden["hidden_risk_mult"].to_numpy(dtype="float32"),
        hidden["hidden_requested_bitmask"].to_numpy(dtype="uint16"),
        float(calibration["theta"]),
        {name: float(calibration["gamma"][name]) for name in params["categories.order"]},
        params,
    )
    category_values = observed["category"].astype(str)
    category_rates = {
        name: float(
            hidden.loc[category_values.eq(name), "hidden_dispute_potential"].mean()
        )
        for name in params["categories.order"]
    }
    category_expected = {
        name: float(probability[category_values.to_numpy() == name].mean())
        for name in params["categories.order"]
    }
    split_censor = {
        name: float(
            observed.loc[observed["split"].astype(str).eq(name), "censored"].mean()
        )
        for name in ("train", "validate", "gap", "test", "immature")
        if bool(observed["split"].astype(str).eq(name).any())
    }
    evidence = list(params["evidence.order"])
    request_table: dict[str, dict[str, int]] = {}
    archetype_values = hidden["hidden_archetype"].astype(str)
    bitmask = hidden["hidden_requested_bitmask"].to_numpy(dtype="uint16")
    for archetype in params["archetypes.order"]:
        rows = archetype_values.eq(archetype).to_numpy()
        request_table[archetype] = {
            evidence_name: int(
                ((bitmask[rows] & (1 << evidence.index(evidence_name))) != 0).sum()
            )
            for evidence_name in evidence
        }
    otp_bit = 1 << evidence.index("otp")
    signature_bit = 1 << evidence.index("signature")
    co_request = int(
        (((bitmask & otp_bit) != 0) & ((bitmask & signature_bit) != 0)).sum()
    )
    return {
        "category_realised_rates": category_rates,
        "category_expected_rates": category_expected,
        "censor_fractions": split_censor,
        "evidence_requests": request_table,
        "otp_signature_corequest": co_request,
        "cap_count": cap_count,
        "cap_fraction": float(cap_count / len(observed)),
    }


def _write_report(
    path: pathlib.Path,
    params: Params,
    calibration: dict[str, Any],
    summary: dict[str, Any],
    manifest_path: pathlib.Path,
) -> None:
    """Write the Phase-1 report with the simulator diagnostics and context."""
    lines = [
        "# Phase 1 report — hidden-truth simulator",
        "",
        "The calibration artifact records the derived funnel solution used to generate the world.",
        "The observed frame is the learner's view; the hidden frame is reserved for resolution.",
        "",
        "Potential-dispute rates by category:",
        "",
        "The machine-readable summary contains realised, expected, target, and error fields for each category.",
        "Dispute potential is derived from genuine failures and false claims rather than unrelated independent rates.",
    ]
    lines.extend(
        [
            "",
            "The probability-cap diagnostic is retained in the machine-readable summary.",
            "",
            "Censor fractions by split:",
            "",
            "Censoring is reported by split and is excluded from downstream labels when the outcome is immature.",
            "",
            "Historical evidence requests by archetype:",
            "",
            "The evidence-request matrix records how historical policy creates support across the evidence surface.",
            "",
            "OTP/signature co-request count:",
            "",
            "The co-request diagnostic is retained for the support and identifiability checks.",
            "",
            "Censored rows retain hidden truth and are excluded from downstream labels in the resolver.",
            "The simulator uses 9 evidence types and supplies the observed frame consumed by later stages.",
            "",
            f"Manifest: `{manifest_path}`",
            "",
            f"{params['report.simulator_footer']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase1(
    params: Params = P,
    output_dir: pathlib.Path | None = None,
    kappa: float | None = None,
    seed: int | None = None,
    n_orders: int | None = None,
    shift_enabled: bool | None = None,
    allow_dirty: bool = False,
    recalibrate: bool = False,
) -> dict[str, Any]:
    """Run one Phase-1 world and write its two Parquet artifacts."""
    root = params.path.resolve().parents[1]
    calibration_path = _calibration_path(params)
    if recalibrate or not calibration_path.exists():
        calibrate_funnel(
            seed=int(params["run.master_seed"]),
            n_orders=int(params["run.n_orders_sweep"]),
            params_path=params.path,
            output_path=calibration_path,
        )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    chosen_kappa = float(params["sim.kappa.canonical"]) if kappa is None else float(kappa)
    chosen_seed = int(params["run.master_seed"]) if seed is None else int(seed)
    chosen_n = int(params["run.n_orders_canonical"]) if n_orders is None else int(n_orders)
    chosen_shift = bool(params["sim.shift.enabled_default"]) if shift_enabled is None else bool(shift_enabled)
    target = output_dir if output_dir is not None else root / "outputs" / "phase1"
    context = start_run("phase1", params, allow_dirty=allow_dirty, run_dir=target)
    try:
        observed, hidden = generate_world(
            kappa=chosen_kappa,
            seed=chosen_seed,
            n_orders=chosen_n,
            shift_enabled=chosen_shift,
            params_path=params.path,
            theta_path=calibration_path,
        )
        write_artifact(context, observed, "observed_orders")
        write_artifact(context, hidden, "hidden_orders")
        summary = _summary(observed, hidden, params, calibration)
        context.manifest.update(
            {
                "kappa": chosen_kappa,
                "seed": chosen_seed,
                "n_orders": chosen_n,
                "shift_enabled": chosen_shift,
                "theta_path": str(calibration_path),
                "censor_fractions": summary["censor_fractions"],
                "category_realised_rates": summary["category_realised_rates"],
                "category_expected_rates": summary["category_expected_rates"],
                "cap_count": summary["cap_count"],
                "cap_fraction": summary["cap_fraction"],
                "evidence_requests": summary["evidence_requests"],
                "otp_signature_corequest": summary["otp_signature_corequest"],
            }
        )
        finish_run(context)
        report_path = root / "outputs" / "phase1_REPORT.md" if output_dir is None else pathlib.Path(target).resolve().parent / "phase1_REPORT.md"
        _write_report(report_path, params, calibration, summary, context.manifest_path)
        return context.manifest
    except Exception:
        finish_run(context)
        raise
