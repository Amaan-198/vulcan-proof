"""Judge-facing artifacts must not contain forbidden claims."""

from __future__ import annotations

import pathlib
import re


def test_judge_facing_claims_are_clean() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    paths = [root / "README.md"]
    paths.extend((root / "outputs").glob("**/*REPORT.md"))
    api = root / "vulcan_proof" / "api"
    ui = root / "vulcan_proof" / "ui" / "src"
    sweep = root / "vulcan_proof" / "sweep"
    if api.exists():
        paths.extend(api.glob("**/*"))
    if ui.exists():
        paths.extend(ui.glob("**/*"))
    if sweep.exists():
        paths.extend(sweep.glob("charts.py"))
    forbidden = [
        "only Razorpay could build this",
        "unforgeable",
        "structurally incapable of helping bad merchants",
        "nobody works at day three",
        "most disputes are false",
        "confirmation defeats the chargeback",
        "no threshold rule could do this",
        "beats rules",
        "outperforms static policy",
        "the fee is recovered on a win",
        "MC 4855",
        "Vulcan uses",
        "contest more",
        "raise your contest rate",
        "reimbursement",
        "settlement hold",
        "escrow",
        "insurance",
        "causal",
        "proven yield",
        "real-world savings",
    ]
    for path in paths:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for phrase in forbidden:
                assert re.search(re.escape(phrase), text, flags=re.IGNORECASE) is None, f"{phrase} in {path}"
