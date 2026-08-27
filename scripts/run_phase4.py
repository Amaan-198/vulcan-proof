"""Run the complete Phase-4 sweep suite."""

from __future__ import annotations

from vulcan_proof.envcheck import require_venv


def main() -> None:
    """Run Phase 4 with configured 1M-order points and parallel seeds."""
    require_venv()
    from vulcan_proof.params import P
    from vulcan_proof.phase4 import run_phase4

    run_phase4(P, parallel=True, allow_dirty=True)


if __name__ == "__main__":
    main()

