"""Run the Phase-0 Olist pipeline."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulcan_proof.envcheck import require_venv
from vulcan_proof.olist.load import DownloadUnavailable
from vulcan_proof.olist.pipeline import run_phase0
from vulcan_proof.params import P


def main() -> None:
    """Parse flags and execute Phase 0."""
    require_venv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    try:
        run_phase0(P, download=args.download, allow_dirty=args.allow_dirty)
    except DownloadUnavailable as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
