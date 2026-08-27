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


def check_phase_2() -> list[tuple[str, bool, str]]:
    """Validate Phase-2 smoke/canonical artifacts and the paired report."""
    phase_dir = ROOT / "outputs" / "phase2"
    report_path = ROOT / "outputs" / "phase2_REPORT.md"
    results: list[tuple[str, bool, str]] = [
        ("phase2 directory exists", phase_dir.is_dir(), str(phase_dir)),
        ("phase2 report exists", report_path.is_file(), str(report_path)),
    ]
    seed_count = int(P["run.n_seeds_sweep"])
    kappas = (0.0, float(P["sim.kappa.canonical"]))
    max_rss = float(P["run.max_peak_rss_gb"]) * 1024
    max_censor = float(P["sim.max_censor_frac"])

    def manifest_ok(
        path: pathlib.Path,
        *,
        expected_orders: int,
        expected_seed: int,
        expected_kappa: float,
        canonical: bool,
    ) -> bool:
        """Check Phase-2 provenance and resource fields without trusting filenames."""
        try:
            manifest = _json(path)
            identity_ok = (
                manifest.get("phase") == "phase2"
                and manifest.get("params_sha256") == P.sha256
                and manifest.get("n_orders") == expected_orders
                and manifest.get("seed") == expected_seed
                and manifest.get("kappa") == expected_kappa
            )
            wall = float(manifest["wall_seconds"])
            rss = float(manifest["peak_rss_mb"])
            resource_ok = (
                math.isfinite(wall)
                and wall >= 0.0
                and math.isfinite(rss)
                and 0.0 <= rss <= max_rss
            )
            if canonical:
                resource_ok = resource_ok and wall <= float(P["run.max_canonical_wall_seconds"])
            censor = manifest["censor_fractions"]
            censor_ok = all(
                name in censor
                and math.isfinite(float(censor[name]))
                and 0.0 <= float(censor[name]) <= max_censor
                for name in ("train", "validate", "gap")
            )
            return identity_ok and resource_ok and censor_ok
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return False

    smoke_missing: list[str] = []
    smoke_manifest_failures: list[str] = []
    for kappa in kappas:
        for seed in range(1, seed_count + 1):
            directory = phase_dir / "smoke" / f"kappa_{str(kappa).replace('-', 'm').replace('.', 'p')}" / f"seed_{seed}"
            manifest_path = directory / "manifest.json"
            if not manifest_ok(
                manifest_path,
                expected_orders=int(P["run.n_orders_smoke"]),
                expected_seed=seed,
                expected_kappa=kappa,
                canonical=False,
            ):
                smoke_manifest_failures.append(str(manifest_path))
            for arm in ("arm0", "arm1", "arm2", "arm3", "arm4"):
                path = directory / f"outcome_{arm}.parquet"
                if not path.is_file():
                    smoke_missing.append(str(path))
            if not (directory / "observed_orders.parquet").is_file():
                smoke_missing.append(str(directory / "observed_orders.parquet"))
            if not (directory / "arm4_policy.json").is_file():
                smoke_missing.append(str(directory / "arm4_policy.json"))
    results.append(("smoke outcomes and policy artifacts exist", not smoke_missing, "; ".join(smoke_missing[:3])))
    results.append(("smoke manifests valid and within limits", not smoke_manifest_failures, "; ".join(smoke_manifest_failures[:3])))
    canonical_missing: list[str] = []
    canonical_manifest_failures: list[str] = []
    canonical_kappa = float(P["sim.kappa.canonical"])
    for seed in range(1, seed_count + 1):
        directory = phase_dir / "canonical" / f"kappa_{str(canonical_kappa).replace('-', 'm').replace('.', 'p')}" / f"seed_{seed}"
        manifest_path = directory / "manifest.json"
        if not manifest_ok(
            manifest_path,
            expected_orders=int(P["run.n_orders_sweep"]),
            expected_seed=seed,
            expected_kappa=canonical_kappa,
            canonical=True,
        ):
            canonical_manifest_failures.append(str(manifest_path))
        for arm in ("arm0", "arm1", "arm4"):
            path = directory / f"outcome_{arm}.parquet"
            if not path.is_file():
                canonical_missing.append(str(path))
    results.append(("canonical outcomes exist", not canonical_missing, "; ".join(canonical_missing[:3])))
    results.append(("canonical manifests valid and within limits", not canonical_manifest_failures, "; ".join(canonical_manifest_failures[:3])))
    sample = next(iter(sorted(phase_dir.glob("smoke/**/outcome_arm4.parquet"))), None) if phase_dir.is_dir() else None
    if sample is not None:
        try:
            frame = pd.read_parquet(sample)
            check(frame, "OUTCOME")
            results.append(("sample outcome schema valid", True, str(sample)))
        except Exception as exc:
            results.append(("sample outcome schema valid", False, f"{type(exc).__name__}: {exc}"))
    else:
        results.append(("sample outcome schema valid", False, "no outcome_arm4 parquet"))
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        required = (
            "Arm 4 − Arm 1 net",
            "Arm 4 − Arm 2 net",
            "Defense-only win rates by claim class",
            "Arm 4 policy table",
            "Realised implied_phi",
            str(P["report.simulator_footer"]),
        )
        results.append(("phase2 report contains required sections", all(item in report for item in required), str(report_path)))
    return results


def check_phase_3() -> list[tuple[str, bool, str]]:
    """Validate Phase-3 model, support, and Arm-5 artifacts."""
    phase_dir = ROOT / "outputs" / "phase3"
    metrics_path = phase_dir / "metrics.json"
    report_path = ROOT / "outputs" / "phase3_REPORT.md"
    results: list[tuple[str, bool, str]] = [
        ("phase3 directory exists", phase_dir.is_dir(), str(phase_dir)),
        ("phase3 metrics exists", metrics_path.is_file(), str(metrics_path)),
        ("phase3 report exists", report_path.is_file(), str(report_path)),
    ]
    if metrics_path.is_file():
        try:
            metrics = _json(metrics_path)
            required = ("stage_a", "stage_b", "stage_c", "defensibility", "materialisation", "lorenz", "paired_arm5_minus_arm4")
            results.append(("phase3 metrics contain model sections", all(name in metrics for name in required), str(metrics_path)))
            results.append(("stage A metrics contain calibration fields", all(name in metrics.get("stage_a", {}) for name in ("pr_auc", "brier", "ece")), str(metrics_path)))
            results.append(("support metrics are present", "support" in metrics.get("defensibility", {}), str(metrics_path)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            results.append(("phase3 metrics are valid JSON", False, f"{type(exc).__name__}: {exc}"))
    seeds = tuple(range(1, int(P["run.n_seeds_sweep"]) + 1))
    kappas = (0.0, float(P["sim.kappa.canonical"]))
    missing: list[str] = []
    invalid: list[str] = []
    max_rss = float(P["run.max_peak_rss_gb"]) * 1024
    for kappa in kappas:
        for seed in seeds:
            directory = phase_dir / "smoke" / f"kappa_{str(kappa).replace('-', 'm').replace('.', 'p')}" / f"seed_{seed}"
            manifest_path = directory / "manifest.json"
            for path in (manifest_path, directory / "observed_orders.parquet", directory / "outcome_arm5.parquet"):
                if not path.is_file():
                    missing.append(str(path))
            if manifest_path.is_file():
                try:
                    manifest = _json(manifest_path)
                    valid = (
                        manifest.get("phase") == "phase3"
                        and manifest.get("params_sha256") == P.sha256
                        and manifest.get("seed") == seed
                        and manifest.get("kappa") == kappa
                        and float(manifest.get("peak_rss_mb", math.inf)) <= max_rss
                    )
                    if not valid:
                        invalid.append(str(manifest_path))
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    invalid.append(str(manifest_path))
    results.append(("phase3 smoke artifacts exist", not missing, "; ".join(missing[:3])))
    results.append(("phase3 smoke manifests valid", not invalid, "; ".join(invalid[:3])))
    canonical_missing: list[str] = []
    canonical_kappa = float(P["sim.kappa.canonical"])
    for seed in seeds:
        directory = phase_dir / "canonical" / f"kappa_{str(canonical_kappa).replace('-', 'm').replace('.', 'p')}" / f"seed_{seed}"
        for path in (directory / "manifest.json", directory / "observed_orders.parquet", directory / "outcome_arm5.parquet"):
            if not path.is_file():
                canonical_missing.append(str(path))
    results.append(("phase3 canonical artifacts exist", not canonical_missing, "; ".join(canonical_missing[:3])))
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        required = ("Lorenz table", "Arm 5 − Arm 4 net", "Support pairs", str(P["report.simulator_footer"]))
        results.append(("phase3 report contains required sections", all(item in report for item in required), str(report_path)))
    return results


def check_phase_4() -> list[tuple[str, bool, str]]:
    """Validate Phase-4 implementation, with production artifacts optional for the buildathon."""
    sweep_dir = ROOT / "vulcan_proof" / "sweep"
    phase_tests = ROOT / "tests" / "test_phase4.py"
    phase_runner = ROOT / "scripts" / "run_phase4.py"
    changelog_path = ROOT / "docs" / "CHANGELOG_review.md"
    policy_markers = (
        "PHASE4_BUILDATHON_OPTIONAL_EXTENDED_VALIDATION",
        "Smoke/end-to-end validation: completed",
        "Full production sweep: deferred",
        "Larger sweep: available later",
    )
    try:
        changelog = changelog_path.read_text(encoding="utf-8")
        policy_ok = all(marker in changelog for marker in policy_markers)
    except OSError:
        policy_ok = False
    results: list[tuple[str, bool, str]] = [
        ("phase4 sweep package exists", sweep_dir.is_dir(), str(sweep_dir)),
        ("phase4 tests exist", phase_tests.is_file(), str(phase_tests)),
        ("phase4 runner exists", phase_runner.is_file(), str(phase_runner)),
        ("phase4 buildathon policy documented", policy_ok, str(changelog_path)),
    ]
    phase_dir = ROOT / "outputs" / "phase4"
    star_path = phase_dir / "kappa_star.json"
    report_path = ROOT / "outputs" / "phase4_REPORT.md"
    oat_path = phase_dir / "oat.json"
    lhs_path = phase_dir / "lhs.json"
    robustness_path = phase_dir / "robustness.json"
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
    production_paths = (star_path, oat_path, lhs_path, robustness_path, report_path) + tuple(
        phase_dir / name for name in chart_names
    )
    production_started = any(path.is_file() for path in production_paths)
    if not production_started:
        results.append(("optional production sweep", True, "not launched; smoke validation is sufficient for buildathon completion"))
        return results

    # A partial production directory is not silently accepted: once any production artifact exists,
    # the later extended-validation criteria apply to the whole artifact set.
    results.extend(
        [
            ("phase4 production directory exists", phase_dir.is_dir(), str(phase_dir)),
            ("kappa_star exists", star_path.is_file(), str(star_path)),
            ("phase4 report exists", report_path.is_file(), str(report_path)),
        ]
    )
    if star_path.is_file():
        try:
            payload = _json(star_path)
            allowed_verdicts = {"ML_CLAIM_SUPPORTED_ABOVE_KAPPA_STAR", "ML_CLAIM_DROPPED_ORCHESTRATION_ONLY"}
            table = payload["table"]
            star_ok = payload.get("verdict") in allowed_verdicts and (isinstance(payload.get("kappa_star"), (float, int)) or payload.get("kappa_star") is None)
            zero_rows = [row for row in table if float(row["kappa"]) == 0.0]
            guard_ok = bool(zero_rows) and float(zero_rows[0]["arm5_minus_arm4_mean"]) <= float(P["report.kappa0_max_gain_frac"]) * float(zero_rows[0]["arm4_minus_arm1_mean"])
            results.append(("kappa_star verdict and table valid", star_ok, str(star_path)))
            results.append(("kappa=0 guard reported satisfied", guard_ok, str(star_path)))
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            results.append(("kappa_star JSON valid", False, f"{type(exc).__name__}: {exc}"))
    expected_oat = len([path for path in _phase4_ranked_paths(int(P["sweep.oat_max_rank"]))]) * int(P["sweep.oat_levels"])
    if oat_path.is_file():
        try:
            rows = _json(oat_path).get("rows", [])
            results.append(("OAT table has all rank-qualified levels", len(rows) == expected_oat, f"{len(rows)}/{expected_oat}"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            results.append(("OAT table is valid JSON", False, str(oat_path)))
    else:
        results.append(("OAT table exists", False, str(oat_path)))
    if lhs_path.is_file():
        try:
            rows = _json(lhs_path).get("rows", [])
            results.append(("LHS table has configured points", len(rows) == int(P["sweep.lhs_points"]), f"{len(rows)}"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            results.append(("LHS table is valid JSON", False, str(lhs_path)))
    else:
        results.append(("LHS table exists", False, str(lhs_path)))
    if robustness_path.is_file():
        try:
            rows = _json(robustness_path).get("rows", [])
            results.append(("four robustness runs reported", len(rows) == 4, f"{len(rows)}"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            results.append(("robustness table is valid JSON", False, str(robustness_path)))
    else:
        results.append(("robustness table exists", False, str(robustness_path)))
    missing = [str(phase_dir / name) for name in chart_names if not (phase_dir / name).is_file()]
    results.append(("nine charts exist", not missing, "; ".join(missing[:3])))
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        first = report.splitlines()[0] if report.splitlines() else ""
        verdict_ok = first.startswith("κ* = ") or first.startswith("κ* not found on [0, 1].")
        required = ("OAT tornado ranking", "Fraction with CI_low > 0", "Robustness", "All ₹ figures are simulator results.", str(P["report.simulator_footer"]))
        results.append(("phase4 report starts with verdict and contains required sections", verdict_ok and all(item in report for item in required), str(report_path)))
    return results


def check_phase_5() -> list[tuple[str, bool, str]]:
    """Validate the Phase-5 API, demo script, report, and built UI."""
    api_dir = ROOT / "vulcan_proof" / "api"
    ui_dir = ROOT / "vulcan_proof" / "ui"
    output_dir = ROOT / "outputs" / "phase5"
    script_path = output_dir / "demo_script.json"
    report_path = ROOT / "outputs" / "phase5_REPORT.md"
    results: list[tuple[str, bool, str]] = [
        ("phase5 API package exists", (api_dir / "main.py").is_file(), str(api_dir)),
        ("phase5 UI source exists", (ui_dir / "src" / "main.jsx").is_file(), str(ui_dir)),
        ("phase5 UI bundle exists", (ui_dir / "dist" / "index.html").is_file(), str(ui_dir / "dist")),
        ("phase5 demo script exists", script_path.is_file(), str(script_path)),
        ("phase5 report exists", report_path.is_file(), str(report_path)),
    ]
    if script_path.is_file():
        try:
            payload = _json(script_path)
            beats = payload.get("beats", [])
            beat_numbers = [int(item["beat"]) for item in beats]
            results.append(("demo script contains eight ordered beats", beat_numbers == list(range(1, 9)), str(script_path)))
            results.append(("demo script exposes its evidence mode", payload.get("mode") in {"smoke_only", "extended_validation"}, str(script_path)))
            deferred_copy = "Production-scale defense-only evidence is deferred; smoke validation is available."
            deferred_present = any(item.get("copy") == deferred_copy for item in beats if isinstance(item, dict))
            if not (ROOT / "outputs" / "phase4").is_dir():
                results.append(("deferred validation fallback is present", deferred_present, str(script_path)))
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            results.append(("demo script is valid JSON", False, f"{type(exc).__name__}: {exc}"))
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        results.append(("phase5 report lists fallback status", "Fallback" in report and "Demo mode" in report, str(report_path)))
    return results


def _phase4_ranked_paths(max_rank: int) -> list[str]:
    """Return rank-qualified paths without importing sweep code in the checker."""
    paths: list[str] = []

    def visit(node: object, prefix: tuple[str, ...]) -> None:
        if isinstance(node, dict) and set(node) == {"value", "unit", "source", "sweep", "rank"}:
            if node["sweep"] is not None and node["rank"] is not None and int(node["rank"]) <= max_rank:
                paths.append(".".join(prefix))
            return
        if isinstance(node, dict):
            for key, child in node.items():
                visit(child, (*prefix, str(key)))

    visit(P.data, ())
    return paths


CHECKS: dict[int, Check] = {0: check_phase_0, 1: check_phase_1, 2: check_phase_2, 3: check_phase_3, 4: check_phase_4, 5: check_phase_5}


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
