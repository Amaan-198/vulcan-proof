"""Shared Phase-4 parameter, execution, and serialisation helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import yaml
from scipy.stats import qmc

from ..errors import InvariantError
from ..params import P, Params
from ..seeds import SeedTree


_PARAM_KEYS = frozenset({"value", "unit", "source", "sweep", "rank"})
_POINT_KEYS = frozenset({"__output_dir__", "__theta_path__", "__n_orders__", "__allow_dirty__", "__include_arm2__", "__include_arm3__", "__manifest_params_path__"})


def _leaf(node: Any) -> bool:
    """Return whether a mapping is a validated parameter leaf."""
    return isinstance(node, Mapping) and frozenset(node) == _PARAM_KEYS


def _set_leaf(data: dict[str, Any], path: str, value: Any) -> None:
    """Set an existing parameter leaf and reject unknown paths."""
    node: Any = data
    components = path.split(".")
    for component in components[:-1]:
        if not isinstance(node, Mapping) or component not in node:
            raise KeyError(path)
        node = node[component]
    if not isinstance(node, Mapping) or components[-1] not in node:
        raise KeyError(path)
    target = node[components[-1]]
    if not _leaf(target):
        raise KeyError(path)
    target["value"] = copy.deepcopy(value)


def apply_overrides(params: Params = P, overrides: Mapping[str, Any] | None = None) -> Params:
    """Return an isolated in-memory parameter copy with existing leaves overridden.

    Phase-4 execution metadata uses double-underscore keys and is deliberately
    ignored here.  The two catalogue aliases are translated exactly as the
    phase specification describes them.
    """
    data = params.data
    supplied = {} if overrides is None else dict(overrides)
    for path, value in supplied.items():
        if path in _POINT_KEYS:
            continue
        if path == "evidence.customer_presence_sweep":
            _set_leaf(data, "evidence.customer_presence_sweep", value)
            _set_leaf(data, "evidence.otp.presence_factor", value)
            _set_leaf(data, "evidence.signature.presence_factor", value)
        elif path == "uplift_true.ack_otp_ratio":
            _set_leaf(data, path, value)
            otp = float(data["uplift_true"]["NR"]["otp"])
            data["uplift_true"]["NR"]["ack"] = otp * float(value)
        else:
            _set_leaf(data, path, value)
    if "sim.genuine_share_target" in supplied:
        _set_leaf(data, "reference.phi", 1.0 - float(supplied["sim.genuine_share_target"]))
    return Params(data, params.path)


def point_parameter_paths(max_rank: int, params: Params = P) -> tuple[str, ...]:
    """Return sweepable parameter leaves at or below ``max_rank``."""
    paths: list[str] = []

    def visit(node: Any, prefix: tuple[str, ...]) -> None:
        if _leaf(node):
            sweep = node["sweep"]
            rank = node["rank"]
            if sweep is not None and rank is not None and int(rank) <= int(max_rank):
                paths.append(".".join(prefix))
            return
        if isinstance(node, Mapping):
            for key, child in node.items():
                visit(child, (*prefix, str(key)))

    visit(params.data, ())
    return tuple(paths)


def oat_levels(path: str, params: Params = P) -> tuple[Any, ...]:
    """Return low, central, and high values for one OAT parameter."""
    sweep = params.meta(path)["sweep"]
    if sweep is None:
        raise InvariantError(f"parameter is not sweepable: {path}")
    values = list(sweep)
    if len(values) == 2:
        return (values[0], params[path], values[-1])
    return tuple(values)


def lhs_design(
    paths: Sequence[str],
    params: Params = P,
    points: int | None = None,
) -> np.ndarray:
    """Build a deterministic Latin-hypercube design in parameter units."""
    chosen_points = int(params["sweep.lhs_points"]) if points is None else int(points)
    if chosen_points <= 0 or not paths:
        raise InvariantError("LHS requires positive points and at least one parameter")
    lows: list[float] = []
    highs: list[float] = []
    for path in paths:
        sweep = params.meta(path)["sweep"]
        if sweep is None or len(sweep) != 2:
            raise InvariantError(f"LHS parameter must have a two-sided range: {path}")
        lows.append(float(sweep[0]))
        highs.append(float(sweep[1]))
    seed = int(params["run.master_seed"]) + int(params["sweep.lhs_seed_offset"])
    generator = SeedTree(seed).child("lhs")
    unit = qmc.LatinHypercube(d=len(paths), seed=generator).random(n=chosen_points)
    return qmc.scale(unit, np.asarray(lows, dtype="float64"), np.asarray(highs, dtype="float64"))


def require_min_seeds(seeds: Iterable[int], params: Params = P) -> tuple[int, ...]:
    """Validate the configured minimum and uniqueness of a seed set."""
    result = tuple(int(seed) for seed in seeds)
    minimum = int(params["report.min_seeds"])
    if len(result) < minimum:
        raise InvariantError("Phase-4 reporting requires the configured minimum seed count")
    if len(set(result)) != len(result):
        raise InvariantError("Phase-4 seed set must be unique")
    return result


def point_id(kappa: float, overrides: Mapping[str, Any]) -> str:
    """Create a stable filesystem-safe identifier for one parameter point."""
    payload = {str(key): json_value(value) for key, value in sorted(overrides.items()) if key not in _POINT_KEYS}
    encoded = json.dumps({"kappa": float(kappa), "overrides": payload}, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"kappa_{number_name(kappa)}_{digest[:len('0123456789ab')]}"


def number_name(value: float) -> str:
    """Create a stable path component for a numeric value."""
    return str(value).replace("-", "m").replace(".", "p")


def json_value(value: Any) -> Any:
    """Convert numpy and nested scalar values to strict-JSON values."""
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: pathlib.Path, payload: Any) -> None:
    """Write a strict, UTF-8 JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_params_snapshot(params: Params, path: pathlib.Path) -> None:
    """Write an isolated, validated parameter snapshot for generator APIs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(params.data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def model_artifact_hash(bundle: Any) -> str:
    """Hash the fitted defensibility artefact without relying on object addresses."""
    model = bundle.defensibility
    digest = hashlib.sha256()
    digest.update(json.dumps(json_value(model.metrics), sort_keys=True, allow_nan=False).encode("utf-8"))
    digest.update(json.dumps(json_value(model.support_masks.as_dict()), sort_keys=True, allow_nan=False).encode("utf-8"))
    digest.update(json.dumps(json_value(model.main_effects), sort_keys=True, allow_nan=False).encode("utf-8"))
    fitted = model.fitted_calibrator
    if fitted is not None and hasattr(fitted, "fitted"):
        fitted = fitted.fitted
    for name in ("X_thresholds_", "y_thresholds_"):
        if fitted is not None and hasattr(fitted, name):
            digest.update(np.asarray(getattr(fitted, name), dtype="float64").tobytes())
    fitted_model = model.model
    if fitted_model is not None and hasattr(fitted_model, "booster_"):
        digest.update(fitted_model.booster_.model_to_string().encode("utf-8"))
    elif fitted_model is not None and hasattr(fitted_model, "value"):
        digest.update(str(float(fitted_model.value)).encode("utf-8"))
    else:
        digest.update(type(fitted_model).__name__.encode("utf-8"))
    return digest.hexdigest()
