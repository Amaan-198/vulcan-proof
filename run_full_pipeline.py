"""Run Vulcan Proof phases 0 through 5 with logging and fail-fast handling."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulcan_proof.envcheck import require_venv
from vulcan_proof.params import P


PHASES = (
    (0, "Olist real-data anchor", ROOT / "scripts" / "run_phase0.py", ("--allow-dirty",)),
    (1, "Hidden-truth simulator", ROOT / "scripts" / "run_phase1.py", ("--allow-dirty",)),
    (2, "No-ML arms and outcomes", ROOT / "scripts" / "run_phase2.py", ("--allow-dirty",)),
    (3, "Models and optimizer", ROOT / "scripts" / "run_phase3.py", ("--allow-dirty",)),
    (4, "Kappa and robustness sweeps", ROOT / "scripts" / "run_phase4.py", ()),
    (5, "Demo artifacts", ROOT / "scripts" / "run_phase5.py", ()),
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _stamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _gpu_description() -> str:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return "not detected (nvidia-smi unavailable)"
    result = subprocess.run(
        [executable, "--query-gpu=name,memory.total", "--format=csv,noheader"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else "not detected"


def _print_startup() -> None:
    logical = os.cpu_count() or 1
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or logical
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_text = f"{physical} physical / {logical} logical cores; {memory_gb:.1f} GB RAM"
    except ImportError:
        cpu_text = f"{logical} logical cores"
    print("\nVulcan Proof full pipeline", flush=True)
    print("=" * 80)
    print(f"CPU: {cpu_text}")
    print(f"GPU: {_gpu_description()} (informational; the configured models are CPU LightGBM)")
    print("\nConfigured workload:")
    rows = (
        ("Smoke orders", P["run.n_orders_smoke"]),
        ("Canonical orders", P["run.n_orders_canonical"]),
        ("Sweep orders", P["run.n_orders_sweep"]),
        ("Canonical seeds", P["run.n_seeds_canonical"]),
        ("Sweep/report seeds", f"{P['run.n_seeds_sweep']} / {P['report.min_seeds']}"),
        ("Support minimum", P["models.support_min"]),
        ("Kappa grid", P["sim.kappa.grid"]),
        ("OAT max rank", f"{P['sweep.oat_max_rank']} (disabled)"),
        ("LHS max rank", f"{P['sweep.lhs_max_rank']} (disabled)"),
        ("LightGBM threads", P["run.lgbm_threads"]),
        ("Parallel workers", P["run.parallel_workers"]),
        ("Threads per worker", P["run.lgbm_threads_in_worker"]),
    )
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label:<{width}} : {value}")


def _clear_outputs() -> pathlib.Path:
    output_dir = ROOT / "outputs"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_log.txt"
    log_path.write_text(
        f"Vulcan Proof full pipeline\nStarted: {_stamp(_now())}\nParams SHA-256: {P.sha256}\n\n",
        encoding="utf-8",
    )
    return log_path


def _append_log(log_path: pathlib.Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _run_phase(number: int, name: str, runner: pathlib.Path, flags: tuple[str, ...]) -> tuple[bool, float, str]:
    started = _now()
    header = f"PHASE {number} STARTING: {name} | {_stamp(started)}"
    print(f"\n{'=' * 80}\n{header}\n{'=' * 80}", flush=True)
    command = [sys.executable, str(runner), *flags]
    output: list[str] = []
    began = time.perf_counter()
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.stdout is None:
            raise RuntimeError("phase runner did not provide a combined output stream")
        for line in process.stdout:
            output.append(line)
            print(line, end="", flush=True)
        return_code = process.wait()
    except Exception as exc:
        output.append(f"{type(exc).__name__}: {exc}\n")
        return_code = 1
    elapsed = time.perf_counter() - began
    passed = return_code == 0
    status = "PASS" if passed else "FAIL"
    print(f"\nPhase {number} {status} in {_duration(elapsed)}", flush=True)
    return passed, elapsed, "".join(output)


def _phase3_metrics() -> dict[str, Any]:
    path = ROOT / "outputs" / "phase3" / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_class = payload["stage_b"]["per_class"]
    reasons = payload.get("reason_codes", {})
    empty = int(reasons.get("EMPTY", 0))
    selected = int(reasons.get("SELECTED", 0))
    total = empty + selected
    coverage = float(selected / total) if total else 0.0
    comparisons = payload["paired_arm5_minus_arm4"]
    return {
        "stage_b_f1": {name: float(per_class[name]["f1"]) for name in ("NR", "NAD", "EB")},
        "empty_plans": empty,
        "selected_plans": selected,
        "coverage": coverage,
        "comparisons": comparisons,
        "source": path,
    }


def _print_phase3_metrics(metrics: dict[str, Any]) -> None:
    print("\nPhase 3 key metrics")
    print("-" * 80)
    f1 = metrics["stage_b_f1"]
    print(f"Stage B F1: NR={f1['NR']:.4f}, NAD={f1['NAD']:.4f}, EB={f1['EB']:.4f}")
    print(
        f"Optimizer plans: empty={metrics['empty_plans']:,}, "
        f"non-empty={metrics['selected_plans']:,}, evidence coverage={metrics['coverage']:.2%}"
    )
    for kappa, row in sorted(metrics["comparisons"].items(), key=lambda item: float(item[0])):
        print(
            f"Arm 5 - Arm 4 at kappa={float(kappa):g}: "
            f"mean={float(row['mean']):.3f} INR/1,000, "
            f"95% CI=[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}], "
            f"n={int(row['n_seeds'])}"
        )
    print(f"Source: {metrics['source']}")


def _print_summary(results: list[dict[str, Any]], metrics: dict[str, Any] | None) -> None:
    print(f"\n{'=' * 80}\nFINAL PIPELINE SUMMARY\n{'=' * 80}")
    print(f"{'Phase':<7} {'Status':<8} {'Duration':<10} Name")
    print("-" * 80)
    for row in results:
        print(f"{row['phase']:<7} {row['status']:<8} {_duration(row['duration']):<10} {row['name']}")
    if metrics is not None:
        _print_phase3_metrics(metrics)


def main() -> int:
    require_venv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-from",
        type=int,
        choices=range(0, 6),
        default=0,
        metavar="PHASE",
        help="resume from an existing phase number without clearing outputs (1-5)",
    )
    args = parser.parse_args()
    _print_startup()
    output_dir = ROOT / "outputs"
    if args.resume_from == 0:
        print("\nWARNING: This will permanently clear the entire outputs/ directory before Phase 0 starts.")
        confirmation = input("Type CLEAR AND RUN to continue: ").strip()
        if confirmation != "CLEAR AND RUN":
            print("Cancelled; outputs/ was not changed.")
            return 0
        log_path = _clear_outputs()
    else:
        if not output_dir.is_dir():
            print("Cannot resume: outputs/ does not exist.", file=sys.stderr)
            return 2
        log_path = output_dir / "run_log.txt"
        if not log_path.is_file():
            log_path.write_text(
                f"Vulcan Proof resumed pipeline\nParams SHA-256: {P.sha256}\n\n",
                encoding="utf-8",
            )
        _append_log(log_path, f"Resume requested from Phase {args.resume_from} | {_stamp(_now())}")
        print(
            f"\nResuming from Phase {args.resume_from}; existing outputs will be preserved.",
            flush=True,
        )
    results: list[dict[str, Any]] = []
    metrics: dict[str, Any] | None = None
    if args.resume_from > 3:
        try:
            metrics = _phase3_metrics()
        except Exception as exc:
            print(
                f"Cannot resume after Phase 3 because its metrics are unavailable: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 4
    selected_phases = tuple(phase for phase in PHASES if phase[0] >= args.resume_from)
    for number, name, runner, flags in selected_phases:
        started = _now()
        _append_log(log_path, f"Phase {number} START | {_stamp(started)} | {name}")
        passed, elapsed, output = _run_phase(number, name, runner, flags)
        ended = _now()
        status = "PASS" if passed else "FAIL"
        results.append({"phase": number, "name": name, "status": status, "duration": elapsed})
        _append_log(
            log_path,
            f"Phase {number} {status} | end={_stamp(ended)} | duration={_duration(elapsed)}",
        )
        if not passed:
            print(f"\nFULL ERROR OUTPUT FOR FAILED PHASE {number}\n{'-' * 80}\n{output}", file=sys.stderr)
            _append_log(log_path, f"Phase {number} error output:\n{output}")
            _print_summary(results, metrics)
            print(f"\nPipeline stopped at Phase {number}. Log: {log_path}", file=sys.stderr)
            return number + 1
        if number == 3:
            try:
                metrics = _phase3_metrics()
                _print_phase3_metrics(metrics)
            except Exception as exc:
                message = f"Failed to read Phase 3 metrics: {type(exc).__name__}: {exc}"
                print(message, file=sys.stderr)
                _append_log(log_path, message)
                results[-1]["status"] = "FAIL"
                _print_summary(results, None)
                return 4

    _print_summary(results, metrics)
    _append_log(log_path, f"Pipeline PASS | end={_stamp(_now())}")
    print(f"\nAll phases passed. Log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
