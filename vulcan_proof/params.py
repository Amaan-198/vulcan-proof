"""Strict loader and accessor for the single project parameter file."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import re
import sys
from collections.abc import Mapping
from typing import Any

import yaml

from .errors import InvariantError, SchemaError


PARAM_KEYS = frozenset({"value", "unit", "source", "sweep", "rank"})
META_KEYS = frozenset({"source", "sweep"})
EVIDENCE_KEYS = frozenset(
    {"cash", "seconds", "presence_factor", "system_sent", "window", "admissible", "api_slot"}
)
CATEGORY_KEYS = frozenset(
    {"share", "target_rate", "mix", "vmin", "vmax", "cogs", "fragility"}
)
ARCHETYPE_KEYS = frozenset(
    {"share", "compliance", "contest", "quality_rank", "policy"}
)
SOURCE_RE = re.compile(r"^(SPEC|CITED|ASSUMED_FIXED|ASSUMED|DERIVED)(?:\b|:| —)")
NULLABLE_DERIVED = frozenset({"sim.theta", "sim.gamma"})


class _StrictLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_mapping(loader: _StrictLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise SchemaError(f"duplicate parameter key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_param_leaf(value: Any) -> bool:
    return _is_mapping(value) and frozenset(value) == PARAM_KEYS


def _is_meta(value: Any) -> bool:
    return _is_mapping(value) and frozenset(value) == META_KEYS


def _source_tag(source: str) -> str:
    if source.startswith("ASSUMED_FIXED"):
        return "ASSUMED_FIXED"
    if source.startswith("ASSUMED"):
        return "ASSUMED"
    return source.split(":", 1)[0]


def _validate_source(path: str, source: Any) -> None:
    if not isinstance(source, str) or SOURCE_RE.match(source) is None:
        raise SchemaError(f"{path}.source has an invalid source tag: {source!r}")


def _validate_param_leaf(path: str, value: Mapping[str, Any]) -> None:
    if frozenset(value) != PARAM_KEYS:
        raise SchemaError(f"{path} must have exactly {sorted(PARAM_KEYS)}")
    _validate_source(path, value["source"])
    sweep = value["sweep"]
    rank = value["rank"]
    if sweep is None and rank is not None:
        raise SchemaError(f"{path}.rank must be null when sweep is null")
    if sweep is not None and rank is None:
        raise SchemaError(f"{path}.rank is required when sweep is not null")
    if _source_tag(value["source"]) == "ASSUMED" and sweep is None:
        raise SchemaError(f"{path} is ASSUMED and must declare a sweep")
    if value["value"] is None and path not in NULLABLE_DERIVED:
        raise SchemaError(f"{path}.value may not be null")


def _validate_meta(path: str, value: Mapping[str, Any]) -> None:
    if frozenset(value) != META_KEYS:
        raise SchemaError(f"{path} must have exactly {sorted(META_KEYS)}")
    _validate_source(path, value["source"])


def _validate_catalogue(path: str, value: Mapping[str, Any], expected: frozenset[str]) -> None:
    if frozenset(value) != expected:
        raise SchemaError(f"{path} must have exactly {sorted(expected)}")


def _validate_uplift_row(path: str, value: Mapping[str, Any]) -> None:
    if not value:
        raise SchemaError(f"{path} may not be empty")
    for evidence, uplift in value.items():
        if not isinstance(evidence, str) or not isinstance(uplift, (int, float)):
            raise SchemaError(f"{path} must map evidence names to numeric uplifts")
        if isinstance(uplift, bool) or not math.isfinite(float(uplift)):
            raise SchemaError(f"{path}.{evidence} must be finite")


def _catalogue_kind(root: Mapping[str, Any], path: tuple[str, ...]) -> frozenset[str] | None:
    if len(path) != 2:
        return None
    parent, name = path
    if parent == "evidence":
        names = root["evidence"]["order"]["value"]
        if name in names:
            return EVIDENCE_KEYS
    if parent == "categories":
        names = root["categories"]["order"]["value"]
        if name in names:
            return CATEGORY_KEYS
    if parent == "archetypes":
        names = root["archetypes"]["order"]["value"]
        if name in names:
            return ARCHETYPE_KEYS
    if parent == "uplift_true" and name in {"NR", "NAD", "EB"}:
        return frozenset()
    return None


def _walk(root: Mapping[str, Any], node: Any, path: tuple[str, ...] = ()) -> None:
    path_text = ".".join(path)
    if _is_param_leaf(node):
        _validate_param_leaf(path_text, node)
        return
    if _is_meta(node) and path and path[-1] == "_meta":
        _validate_meta(path_text, node)
        return
    if _is_mapping(node):
        catalogue = _catalogue_kind(root, path)
        if catalogue is not None:
            if catalogue == frozenset():
                _validate_uplift_row(path_text, node)
            else:
                _validate_catalogue(path_text, node, catalogue)
            return
        if not node:
            raise SchemaError(f"{path_text} is an unrecognised empty mapping")
        for key, child in node.items():
            if not isinstance(key, str):
                raise SchemaError(f"{path_text} contains a non-string key")
            _walk(root, child, (*path, key))
        return
    raise SchemaError(f"{path_text} is not a parameter leaf or recognised catalogue")


def _leaf_value(data: Mapping[str, Any], path: str) -> Any:
    node: Any = data
    for component in path.split("."):
        if not isinstance(node, Mapping) or component not in node:
            raise KeyError(path)
        node = node[component]
    if _is_param_leaf(node):
        return node["value"]
    return node


def _leaf_meta(data: Mapping[str, Any], path: str) -> dict[str, Any]:
    node: Any = data
    for component in path.split("."):
        if not isinstance(node, Mapping) or component not in node:
            raise KeyError(path)
        node = node[component]
    if not _is_param_leaf(node):
        raise KeyError(path)
    return copy.deepcopy(dict(node))


def _cross_checks(data: Mapping[str, Any]) -> None:
    def value(path: str) -> Any:
        return _leaf_value(data, path)

    categories = value("categories.order")
    if not math.isclose(
        sum(float(value(f"categories.{name}")["share"]) for name in categories),
        1.0,
        abs_tol=1e-9,
    ):
        raise InvariantError("categories shares do not sum to one")
    archetypes = value("archetypes.order")
    if not math.isclose(
        sum(float(value(f"archetypes.{name}")["share"]) for name in archetypes),
        1.0,
        abs_tol=1e-9,
    ):
        raise InvariantError("archetype shares do not sum to one")
    for name in categories:
        mix = value(f"categories.{name}")["mix"]
        if not math.isclose(sum(float(item) for item in mix.values()), 1.0, abs_tol=1e-9):
            raise InvariantError(f"category mix does not sum to one: {name}")
    prevention = value("econ.prevention")
    shares = [
        value(f"econ.prevention.{name}")
        for name in prevention
        if name.startswith("share_")
    ]
    if not math.isclose(sum(float(item) for item in shares), 1.0, abs_tol=1e-9):
        raise InvariantError("prevention shares do not sum to one")
    for name in value("sim.customer_response"):
        triple = value(f"sim.customer_response.{name}")
        if not math.isclose(sum(float(item) for item in triple), 1.0, abs_tol=1e-9):
            raise InvariantError(f"customer response does not sum to one: {name}")
    if float(value("merchants.tier_full")) + float(value("merchants.tier_post_delivery_only")) > 1.0 + 1e-9:
        raise InvariantError("merchant tier shares exceed one")
    allowed_disputes = {"NR", "NAD", "EB"}
    for name in value("evidence.order"):
        admissible = set(value(f"evidence.{name}")["admissible"])
        if not admissible.issubset(allowed_disputes):
            raise InvariantError(f"unknown admissibility class in evidence.{name}")
    for prefix in ("features", "olist.features"):
        permitted = set(value(f"{prefix}.permitted"))
        forbidden = set(value(f"{prefix}.forbidden"))
        if permitted.intersection(forbidden):
            raise InvariantError(f"permitted and forbidden features overlap in {prefix}")
    if not math.isclose(
        float(value("reference.phi")),
        1.0 - float(value("sim.genuine_share_target")),
        abs_tol=1e-9,
    ):
        raise InvariantError("reference.phi does not complement genuine_share_target")


class Params:
    """Validated parameter tree with dotted value access."""

    def __init__(self, data: dict[str, Any], path: pathlib.Path) -> None:
        self._data = data
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def __getitem__(self, path: str) -> Any:
        return _leaf_value(self._data, path)

    def meta(self, path: str) -> dict[str, Any]:
        """Return a copy of a parameter leaf, including provenance metadata."""
        return _leaf_meta(self._data, path)

    def derived(self, name: str) -> Any:
        """Read a derived calibration value from ``outputs/theta.json``."""
        root = self.path.resolve().parents[1]
        output = root / "outputs" / "theta.json"
        if not output.exists():
            raise InvariantError(f"derived calibration file is missing: {output}")
        payload = json.loads(output.read_text(encoding="utf-8"))
        if name not in payload:
            raise KeyError(name)
        return payload[name]

    def derived_pc_population(self) -> float:
        """Compute and cross-check the archetype-weighted contest rate."""
        value = sum(
            float(self[f"archetypes.{name}"]["share"])
            * float(self[f"archetypes.{name}"]["contest"])
            for name in self["archetypes.order"]
        )
        if not math.isclose(value, float(self["reference.pc_population"]), abs_tol=1e-9):
            raise InvariantError("derived population contest rate disagrees with reference")
        return value

    def derived_compliance_population(self) -> float:
        """Compute and cross-check the archetype-weighted compliance rate."""
        value = sum(
            float(self[f"archetypes.{name}"]["share"])
            * float(self[f"archetypes.{name}"]["compliance"])
            for name in self["archetypes.order"]
        )
        if not math.isclose(value, float(self["reference.compliance_population"]), abs_tol=1e-9):
            raise InvariantError("derived population compliance rate disagrees with reference")
        return value

    @property
    def data(self) -> dict[str, Any]:
        """Return a defensive copy of the validated tree."""
        return copy.deepcopy(self._data)


def load(path: str | pathlib.Path) -> Params:
    """Load, strictly validate, and return a parameter file."""
    parameter_path = pathlib.Path(path).resolve()
    if not parameter_path.exists():
        raise FileNotFoundError(parameter_path)
    data = yaml.load(parameter_path.read_text(encoding="utf-8"), Loader=_StrictLoader)
    if not isinstance(data, dict):
        raise SchemaError("parameter root must be a mapping")
    _walk(data, data)
    _cross_checks(data)
    return Params(data, parameter_path)


P = load(pathlib.Path(__file__).resolve().parents[1] / "params" / "params.yaml")


if __name__ == "__main__":
    from .envcheck import require_venv

    require_venv()
    if len(sys.argv) != 2 + 1 or sys.argv[1] != "--lint":
        raise SystemExit("usage: python -m vulcan_proof.params --lint params/params.yaml")
    loaded = load(sys.argv[2])
    loaded.derived_pc_population()
    loaded.derived_compliance_population()
    print(f"PARAMETERS OK: {loaded.sha256}")
