# tests/

Run in every phase: `test_params.py` (Phase 0 writes it), `test_ev_reference.py` (provided; frozen).
Per phase: `test_phase<N>.py`, as specified in `docs/phase_<N>_*.md`. Cross-cutting from Phase 3:
`test_firewall.py`, `test_repro.py`. From Phase 0: `test_claims.py`, `test_seeds.py`, `test_manifest.py`.

Tests MAY read hidden columns and `uplift_true` (to compute MAE against truth, or to build a
deliberately-leaked foil). Library code under `vulcan_proof/opt`, `models`, `arms` may not.

Every test docstring states what failure means. A failing invariant test is never fixed by
relaxing a tolerance in `params.yaml`.
