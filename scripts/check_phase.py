"""Mechanical done-criteria checks for completed phases."""

from __future__ import annotations

import json
import math
import pathlib
import sys
import tempfile
from collections.abc import Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulcan_proof.envcheck import require_venv
from vulcan_proof.olist.pipeline import run_phase0
from vulcan_proof.params import P


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


CHECKS: dict[int, Check] = {0: check_phase_0}


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
