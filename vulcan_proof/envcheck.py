"""Runtime guard for the pinned native Windows environment."""

from __future__ import annotations

import pathlib
import sys


def require_venv() -> None:
    """Exit unless the process is Python 3.13 inside this repository's venv."""
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    expected = (repo_root / ".venv").resolve()
    prefix = pathlib.Path(sys.prefix).resolve()
    if sys.version_info[:2] != (3, 13):
        raise SystemExit(f"REFUSING: Python 3.13 required, found {sys.version}")
    if expected not in (prefix, *prefix.parents):
        raise SystemExit(
            f"REFUSING: activate {expected} before running this project."
        )
