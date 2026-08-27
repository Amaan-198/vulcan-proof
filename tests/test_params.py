"""Phase-0 parameter-loader and coding-rule tests."""

from __future__ import annotations

import ast
import pathlib

import pytest

from vulcan_proof.errors import SchemaError
from vulcan_proof.params import P, load


ROOT = pathlib.Path(__file__).resolve().parents[1]
ALLOWED_NUMBERS = {0, 1, -1, 2, 0.5, 100, 1000, 1024, 3600, 1e-9, 1e-12}


def test_lint_passes() -> None:
    loaded = load(ROOT / "params" / "params.yaml")
    assert loaded.sha256 == P.sha256


def test_missing_key_raises() -> None:
    with pytest.raises(KeyError, match="does.not.exist"):
        _ = P["does.not.exist"]


def test_assumed_requires_sweep(tmp_path: pathlib.Path) -> None:
    text = (ROOT / "params" / "params.yaml").read_text(encoding="utf-8")
    text = text.replace(
        "source: ASSUMED, sweep: [0.5, 1.5]",
        "source: ASSUMED, sweep: null",
        1,
    )
    path = tmp_path / "params.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SchemaError):
        load(path)


def test_no_magic_numbers() -> None:
    violations: list[str] = []
    excluded = {"ev_reference.py", "envcheck.py", "seeds.py"}
    for path in (ROOT / "vulcan_proof").rglob("*.py"):
        if path.name in excluded or "ui" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for candidate in ast.walk(tree):
            for child in ast.iter_child_nodes(candidate):
                parents[child] = candidate
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, (int, float)):
                continue
            if isinstance(node.value, bool) or node.value in ALLOWED_NUMBERS:
                continue
            parent = parents.get(node)
            if isinstance(parent, ast.Subscript):
                continue
            if isinstance(parent, ast.Call) and any(keyword.arg in {"shape", "reshape"} for keyword in parent.keywords):
                continue
            violations.append(f"{path}:{node.lineno} -> {node.value}")
    assert not violations, "magic numbers found:\n" + "\n".join(violations)


def test_derived_population_rates() -> None:
    assert P.derived_pc_population() == pytest.approx(0.6125)
    assert P.derived_compliance_population() == pytest.approx(0.825)
