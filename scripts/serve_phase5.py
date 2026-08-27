"""Prepare the Phase-5 artefacts and serve the UI/API in one process."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import os

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulcan_proof.envcheck import require_venv


def _reexec_repo_venv() -> None:
    """Use the repository environment even when npm was launched from a plain shell."""
    if sys.prefix != sys.base_prefix:
        return
    candidates = (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    )
    interpreter = next((candidate for candidate in candidates if candidate.is_file()), None)
    if interpreter is not None:
        os.execv(str(interpreter), [str(interpreter), str(pathlib.Path(__file__)), *sys.argv[1:]])


def main() -> None:
    """Generate missing demo artefacts, then run Uvicorn."""
    _reexec_repo_venv()
    require_venv()
    demo_path = ROOT / "outputs" / "phase5" / "demo_script.json"
    if not demo_path.is_file():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "run_phase5.py")], cwd=ROOT, check=True)
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "vulcan_proof.api.main:app", "--host", "127.0.0.1", "--port", "8765"],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
