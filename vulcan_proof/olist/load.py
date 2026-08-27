"""Olist download and raw-table loading with integrity checks."""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..errors import InvariantError
from ..params import P, Params


class DownloadUnavailable(InvariantError):
    """Raised when Kaggle credentials are unavailable for the requested download."""


def _required_files(params: Params) -> list[str]:
    return list(params["olist.required_files"])


def _check_files(data_dir: pathlib.Path, params: Params) -> None:
    missing = [name for name in _required_files(params) if not (data_dir / name).is_file()]
    if missing:
        raise InvariantError(f"Olist data is incomplete; missing files: {missing}")
    expected = params["olist.expected_rows"]
    tolerance = float(params["olist.row_count_tolerance"])
    for name, expected_rows in expected.items():
        rows = len(pd.read_csv(data_dir / name, usecols=[0], encoding="utf-8"))
        if abs(rows - int(expected_rows)) / int(expected_rows) > tolerance:
            raise InvariantError(
                f"row count for {name} is {rows}, expected {expected_rows} within {tolerance}"
            )


def download_olist(params: Params = P) -> pathlib.Path:
    """Download and validate the Kaggle Olist dataset using the Kaggle API."""
    data_dir = pathlib.Path(params["olist.data_dir"])
    if not data_dir.is_absolute():
        data_dir = params.path.resolve().parents[1] / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    credential = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    if not credential.is_file():
        page = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
        instruction = (
            f"Kaggle credentials are missing at {credential}. Open {page}, click Download, "
            "then unzip the archive into data\\olist and rerun without --download."
        )
        raise DownloadUnavailable(instruction)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise InvariantError("the kaggle package is required for --download") from exc
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        params["olist.kaggle_dataset"], path=str(data_dir), unzip=True
    )
    _check_files(data_dir, params)
    return data_dir


def load_olist(data_dir: pathlib.Path | str | None = None, params: Params = P) -> dict[str, pd.DataFrame]:
    """Read and validate all required Olist CSV tables."""
    if data_dir is None:
        configured = pathlib.Path(params["olist.data_dir"])
        data_dir = configured if configured.is_absolute() else params.path.resolve().parents[1] / configured
    directory = pathlib.Path(data_dir).resolve()
    _check_files(directory, params)
    tables: dict[str, pd.DataFrame] = {}
    for filename in _required_files(params):
        tables[filename] = pd.read_csv(directory / filename, encoding="utf-8")
    return tables


def table(tables: Mapping[str, pd.DataFrame], filename: str) -> pd.DataFrame:
    """Return a required raw table, raising a full-path error when absent."""
    if filename not in tables:
        raise KeyError(filename)
    return tables[filename]
