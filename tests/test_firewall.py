"""Mechanical truth-firewall checks for Phase 3."""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

from vulcan_proof.errors import SchemaError
from vulcan_proof.params import P


ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN_MODULES = {"sim", "resolve", "truth", "generator"}
FORBIDDEN_NAMES = re.compile(r"^(hidden|truth|uplift_true|gamma|theta)(_|$)")


def test_no_truth_symbols_in_opt_models_arms() -> None:
    violations: list[str] = []
    for directory in (ROOT / "vulcan_proof" / "opt", ROOT / "vulcan_proof" / "models", ROOT / "vulcan_proof" / "arms"):
        for path in directory.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                else:
                    modules = []
                for module in modules:
                    if any(part in FORBIDDEN_MODULES for part in module.split(".")):
                        violations.append(f"{path}:{node.lineno}: import {module}")
                if isinstance(node, ast.Name) and FORBIDDEN_NAMES.match(node.id):
                    violations.append(f"{path}:{node.lineno}: {node.id}")
                if isinstance(node, ast.Attribute) and FORBIDDEN_NAMES.match(node.attr):
                    violations.append(f"{path}:{node.lineno}: {node.attr}")
    assert not violations, "firewall violations:\n" + "\n".join(violations)


def test_permitted_forbidden_disjoint() -> None:
    assert not set(P["features.permitted"]).intersection(P["features.forbidden"])


def test_schema_gate_rejects_hidden_column() -> None:
    from vulcan_proof.schemas import ORDER_OBSERVED

    values = {column: [0] for column in ORDER_OBSERVED}
    values["hidden_z_risk"] = [0.0]
    with pytest.raises(SchemaError):
        from vulcan_proof.schemas import check

        check(__import__("pandas").DataFrame(values), "ORDER_OBSERVED")
