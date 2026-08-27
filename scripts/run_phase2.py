"""Run Phase 2 outcome resolution and no-ML arms."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vulcan_proof.envcheck import require_venv
from vulcan_proof.params import P
from vulcan_proof.sim.phase2 import run_phase2


def main() -> None:
    """Parse optional run overrides and execute Phase 2."""
    require_venv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=pathlib.Path)
    parser.add_argument("--n-orders", type=int)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--kappa", action="append", type=float)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    run_phase2(
        P,
        output_root=args.output_root,
        seeds=args.seed,
        kappas=args.kappa,
        n_orders=args.n_orders,
        include_canonical=not args.smoke_only,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    main()

