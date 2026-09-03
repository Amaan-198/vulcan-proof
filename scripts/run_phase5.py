"""Generate the Phase-5 demo script and report from existing artefacts."""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulcan_proof.envcheck import require_venv


def _write_report(root: pathlib.Path, script: dict[str, object]) -> pathlib.Path:
    """Write the artifact-backed Phase-5 handoff report."""
    lines = ["# Phase 5 report — product surface and demo", ""]
    lines.append(f"Demo mode: `{script.get('mode', 'unknown')}`")
    lines.append("")
    lines.extend(
        [
            "",
            "The product surface reads stored model and evaluation artifacts and preserves their source context.",
            "Fallback copy is recorded when an artifact-backed selection condition is unavailable.",
            "",
            str(script.get("simulator_footer", "")),
            "",
        ]
    )
    path = root / "outputs" / "phase5_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    """Generate the demo script and report."""
    require_venv()
    from vulcan_proof.api.service import Phase5Service

    root = ROOT
    service = Phase5Service(root=root)
    script = service.demo_script()
    output_dir = root / "outputs" / "phase5"
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / "demo_script.json"
    script_path.write_text(json.dumps(script, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path = _write_report(root, script)
    print(f"Phase 5 demo script: {script_path}")
    print(f"Phase 5 report: {report_path}")
    print(f"Mode: {script.get('mode')}")


if __name__ == "__main__":
    main()
