"""Run the Phase-1 hidden-truth simulator."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulcan_proof.envcheck import require_venv
from vulcan_proof.params import P
from vulcan_proof.sim.pipeline import run_phase1


def main() -> None:
    """Parse optional run overrides and execute Phase 1."""
    require_venv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n-orders", type=int)
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--recalibrate", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    run_phase1(
        P,
        kappa=args.kappa,
        seed=args.seed,
        n_orders=args.n_orders,
        shift_enabled=True if args.shift else None,
        recalibrate=args.recalibrate,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()
