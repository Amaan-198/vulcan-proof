"""Mechanical done-criteria checks for completed phases."""

from __future__ import annotations

import json
import math
import pathlib
import sys
import tempfile
from collections.abc import Callable

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulcan_proof.envcheck import require_venv
from vulcan_proof.olist.pipeline import run_phase0
from vulcan_proof.params import P
from vulcan_proof.schemas import check


Check = Callable[[], list[tuple[str, bool, str]]]


def _json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _phase0_files() -> list[tuple[str, bool, str]]:
    metrics_path = ROOT / "outputs" / "phase0" / "metrics.json"
    manifest_path = ROOT / "outputs" / "phase0" / "manifest.json"
    report_path = ROOT / "outputs" / "phase0_REPORT.md"
    results: list[tuple[str, bool, str]] = []
    for label, path in (("metrics exists", metrics_path), ("manifest exists", manifest_path), ("report exists", report_path)):
        results.append((label, path.is_file(), str(path)))
    if not metrics_path.is_file():
        return results
    metrics = _json(metrics_path)
    finite = all(math.isfinite(float(metrics[key])) for key in ("pr_auc", "brier", "ece"))
    results.append(("required test metrics finite", finite, "pr_auc, brier, ece"))
    calibration = metrics.get("calibration_checks", {})
    validation = calibration.get("validate", {})
    validation_rate = float(validation.get("empirical_rate", 0))
    validation_error = (
        abs(float(validation.get("mean_prediction", 0)) - validation_rate) / validation_rate
        if validation_rate
        else math.inf
    )
    results.append((
        "validation calibrated mean within tolerance",
        validation_error < float(P["models.calib.mean_tolerance"]),
        f"relative_error={validation_error:.6f}",
    ))
    drift = metrics.get("temporal_drift", {})
    monthly = drift.get("monthly_label_rates")
    drift_ok = (
        drift.get("calibration_protocol")
        == "isotonic fitted on validation labels only; test labels used for evaluation only"
        and isinstance(drift.get("test_within_mean_tolerance"), bool)
        and math.isfinite(float(drift.get("test_relative_mean_error", math.inf)))
        and isinstance(monthly, list)
        and bool(monthly)
        and bool(drift.get("largest_rate_drop_reason"))
    )
    results.append((
        "test calibration transfer reported diagnostically",
        drift_ok,
        str(drift.get("largest_rate_drop_reason")),
    ))
    operating = metrics.get("operating_points")
    expected_operating = len(P["olist.operating_recalls"])
    results.append(("three operating points present", isinstance(operating, list) and len(operating) == expected_operating, str(len(operating) if isinstance(operating, list) else None)))
    if manifest_path.is_file():
        manifest = _json(manifest_path)
        max_rss = float(P["run.max_peak_rss_gb"]) * 1024
        manifest_ok = (
            manifest.get("phase") == "phase0"
            and manifest.get("params_sha256") == P.sha256
            and manifest.get("git_clean_at_start") is True
            and math.isfinite(float(manifest["peak_rss_mb"]))
            and float(manifest["peak_rss_mb"]) <= max_rss
        )
        results.append(("manifest valid and within RSS limit", manifest_ok, str(manifest_path)))
    if report_path.is_file():
        required_sentence = "Olist has no chargeback or evidence data; this measures detection only."
        report = report_path.read_text(encoding="utf-8")
        results.append(("report contains required statement", required_sentence in report, str(report_path)))
    return results


def _phase0_reproducibility() -> tuple[str, bool, str]:
    with tempfile.TemporaryDirectory(prefix="vulcan_phase0_") as temporary:
        root = pathlib.Path(temporary)
        first = root / "first"
        second = root / "second"
        run_phase0(P, output_dir=first, allow_dirty=True)
        run_phase0(P, output_dir=second, allow_dirty=True)
        left = _json(first / "metrics.json")
        right = _json(second / "metrics.json")
        for payload in (left, right):
            for key in ("timestamp", "wall_seconds", "peak_rss_mb"):
                payload.pop(key, None)
        left_bytes = json.dumps(left, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        right_bytes = json.dumps(right, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        equal = left_bytes == right_bytes and left == right
        return ("same-seed metrics byte-identical", equal, "two independent temporary output directories")


def check_phase_0() -> list[tuple[str, bool, str]]:
    """Run the Phase-0 mechanical checks, including a two-run reproducibility check."""
    results = _phase0_files()
    try:
        results.append(_phase0_reproducibility())
    except Exception as exc:
        results.append(("same-seed metrics byte-identical", False, f"{type(exc).__name__}: {exc}"))
    return results


def _phase1_calibration() -> list[tuple[str, bool, str]]:
    """Validate the derived funnel calibration and its six-target diagnostics."""
    path = ROOT / "outputs" / "theta.json"
    results: list[tuple[str, bool, str]] = [("theta exists", path.is_file(), str(path))]
    if not path.is_file():
        return results
    payload = _json(path)
    categories = list(P["categories.order"])
    tolerance = float(P["sim.calibration_rel_tol"])
    rates = payload.get("achieved_category_rates")
    rates_ok = isinstance(rates, dict) and all(
        name in rates
        and abs(float(rates[name]) - float(P[f"categories.{name}"]["target_rate"]))
        / float(P[f"categories.{name}"]["target_rate"])
        <= tolerance
        for name in categories
    )
    results.append(("category calibration targets within tolerance", rates_ok, str(rates)))
    genuine = float(payload.get("achieved_genuine_share", math.nan))
    genuine_target = float(P["sim.genuine_share_target"])
    genuine_ok = math.isfinite(genuine) and abs(genuine - genuine_target) / genuine_target <= tolerance
    results.append(("genuine share within tolerance", genuine_ok, f"relative_error={abs(genuine - genuine_target) / genuine_target:.6f}"))
    phi = float(payload.get("implied_phi", math.nan))
    phi_tolerance = float(P["models.calib.mean_tolerance"]) - float(P["sim.max_censor_frac"])
    phi_ok = math.isfinite(phi) and abs(phi - float(P["reference.phi"])) < phi_tolerance
    results.append(("implied phi within tolerance", phi_ok, f"phi={phi:.6f}"))
    return results


def check_phase_1() -> list[tuple[str, bool, str]]:
    """Validate the canonical Phase-1 artifacts and report gates."""
    results = _phase1_calibration()
    phase_dir = ROOT / "outputs" / "phase1"
    manifest_path = phase_dir / "manifest.json"
    report_path = ROOT / "outputs" / "phase1_REPORT.md"
    observed_path = phase_dir / "observed_orders.parquet"
    hidden_path = phase_dir / "hidden_orders.parquet"
    for label, path in (
        ("phase1 manifest exists", manifest_path),
        ("phase1 report exists", report_path),
        ("observed artifact exists", observed_path),
        ("hidden artifact exists", hidden_path),
    ):
        results.append((label, path.is_file(), str(path)))
    if manifest_path.is_file():
        manifest = _json(manifest_path)
        expected_identity = (
            manifest.get("phase") == "phase1"
            and manifest.get("params_sha256") == P.sha256
            and manifest.get("n_orders") == int(P["run.n_orders_canonical"])
            and manifest.get("seed") == int(P["run.master_seed"])
            and manifest.get("kappa") == float(P["sim.kappa.canonical"])
            and manifest.get("shift_enabled") is False
        )
        results.append(("canonical manifest identity", expected_identity, str(manifest_path)))
        max_rss = float(P["run.max_peak_rss_gb"]) * 1024
        resource_ok = (
            math.isfinite(float(manifest.get("wall_seconds", math.inf)))
            and float(manifest["wall_seconds"]) <= float(P["run.max_canonical_wall_seconds"])
            and math.isfinite(float(manifest.get("peak_rss_mb", math.inf)))
            and float(manifest["peak_rss_mb"]) <= max_rss
        )
        results.append(("canonical wall and RSS limits", resource_ok, str(manifest_path)))
        results.append(("canonical run started clean", manifest.get("git_clean_at_start") is True, str(manifest_path)))
        censor = manifest.get("censor_fractions", {})
        censor_ok = all(
            name in censor and float(censor[name]) <= float(P["sim.max_censor_frac"])
            for name in ("train", "validate")
        )
        results.append(("train/validate censor fractions within limit", censor_ok, str(censor)))
        category_rates = manifest.get("category_realised_rates", {})
        category_ok = all(
            name in category_rates
            and abs(float(category_rates[name]) - float(P[f"categories.{name}"]["target_rate"]))
            / float(P[f"categories.{name}"]["target_rate"])
            <= float(P["categories.category_rate_tolerance"])
            for name in P["categories.order"]
        )
        results.append(("canonical category rates within tolerance", category_ok, str(category_rates)))
    if observed_path.is_file():
        try:
            observed = pd.read_parquet(observed_path)
            check(observed, "ORDER_OBSERVED")
            results.append(("observed schema has no hidden columns", not any(str(column).startswith("hidden_") for column in observed.columns), str(observed_path)))
        except Exception as exc:
            results.append(("observed schema has no hidden columns", False, f"{type(exc).__name__}: {exc}"))
    if hidden_path.is_file():
        try:
            check(pd.read_parquet(hidden_path), "ORDER_HIDDEN")
            results.append(("hidden schema valid", True, str(hidden_path)))
        except Exception as exc:
            results.append(("hidden schema valid", False, f"{type(exc).__name__}: {exc}"))
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        required = (
            "Potential-dispute rates by category:",
            "Censor fractions by split:",
            "Historical evidence requests by archetype:",
            "OTP/signature co-request count:",
            str(P["report.simulator_footer"]),
        )
        results.append(("phase1 report contains required tables", all(item in report for item in required), str(report_path)))
    return results


CHECKS: dict[int, Check] = {0: check_phase_0, 1: check_phase_1}


def main() -> None:
    """Run the selected phase's registry entry."""
    require_venv()
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/check_phase.py N")
    try:
        phase = int(sys.argv[1])
        checker = CHECKS[phase]
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"no mechanical checks registered for phase {sys.argv[1]!r}") from exc
    results = checker()
    for label, passed, detail in results:
        print(f"{'PASS' if passed else 'FAIL':4} | {label} | {detail}")
    if not all(passed for _, passed, _ in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
