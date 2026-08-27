"""Run manifests and guarded Parquet artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import pathlib
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import psutil

from .errors import InvariantError, LeakError
from .params import Params


def _git_snapshot(root: pathlib.Path, allow_dirty: bool) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InvariantError(f"manifest requires a Git checkout at {root}") from exc
    clean = status == ""
    if not clean and not allow_dirty:
        raise InvariantError("refusing to start a run with a dirty Git tree")
    return commit, clean


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        versions[name] = distribution.version
    return dict(sorted(versions.items(), key=lambda item: item[0].lower()))


def _write_manifest(ctx: "RunContext") -> None:
    ctx.manifest_path.write_text(
        json.dumps(ctx.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@dataclass
class RunContext:
    """Mutable run state needed to update one manifest safely."""

    run_dir: pathlib.Path
    manifest_path: pathlib.Path
    manifest: dict[str, Any]
    started: float


def start_run(
    phase: str,
    params: Params,
    allow_dirty: bool = False,
    run_dir: pathlib.Path | None = None,
) -> RunContext:
    """Create a run directory and write its initial manifest."""
    root = params.path.resolve().parents[1]
    commit, clean = _git_snapshot(root, allow_dirty)
    now = datetime.now(timezone.utc)
    if run_dir is None:
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        run_dir = root / "outputs" / f"{phase}_{stamp}_{params['run.master_seed']}"
    run_dir = pathlib.Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git_commit": commit,
        "git_clean_at_start": clean,
        "dirty_allowed": allow_dirty,
        "params_sha256": params.sha256,
        "master_seed": params["run.master_seed"],
        "phase": phase,
        "timestamp": now.isoformat(),
        "python_version": sys.version,
        "installed_versions": _installed_versions(),
        "hostname": socket.gethostname(),
        "lgbm_threads": params["run.lgbm_threads"],
        "lgbm_threads_in_worker": params["run.lgbm_threads_in_worker"],
        "artifacts": [],
    }
    context = RunContext(run_dir, run_dir / "manifest.json", manifest, time.perf_counter())
    _write_manifest(context)
    return context


def _peak_rss_mb() -> float:
    memory = psutil.Process().memory_info()
    if sys.platform == "win32":
        return float(memory.peak_wset) / (1024 * 1024)
    return float(memory.rss) / (1024 * 1024)


def finish_run(context: RunContext) -> None:
    """Record elapsed time and peak resident memory in the manifest."""
    context.manifest["wall_seconds"] = time.perf_counter() - context.started
    context.manifest["peak_rss_mb"] = _peak_rss_mb()
    _write_manifest(context)


def write_artifact(context: RunContext, df: pd.DataFrame, name: str) -> pathlib.Path:
    """Write one Parquet artifact and append its digest to the manifest."""
    if any(str(column).startswith("hidden_") for column in df.columns) and name.startswith("observed_"):
        raise LeakError(f"hidden columns cannot be written as observed artifact: {name}")
    artifact_name = pathlib.Path(name)
    if artifact_name.name != name or name in {"", ".", ".."}:
        raise InvariantError(f"artifact name must be a simple file stem: {name!r}")
    path = context.run_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    context.manifest["artifacts"].append(
        {"name": name, "sha256": digest, "n_rows": len(df)}
    )
    _write_manifest(context)
    return path


def load_manifest(path: pathlib.Path) -> dict[str, Any]:
    """Read a manifest JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))
