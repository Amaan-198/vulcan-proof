"""Artifact guards prevent hidden columns entering observed outputs."""

from __future__ import annotations

import pandas as pd
import pytest

from vulcan_proof.errors import LeakError
from vulcan_proof.manifest import start_run, write_artifact
from vulcan_proof.params import P


def test_observed_artifact_rejects_hidden_column(tmp_path) -> None:
    context = start_run("test", P, allow_dirty=True, run_dir=tmp_path / "run")
    frame = pd.DataFrame({"hidden_x": pd.Series([1], dtype="int8")})
    with pytest.raises(LeakError):
        write_artifact(context, frame, "observed_orders")
