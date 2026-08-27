"""Deterministic, label-addressed random-number streams.

The child key is ``int.from_bytes(sha256("|".join(labels)).digest()[:8],
"little")``.  It is deliberately not Python's process-randomised ``hash``.
Each child is a fresh ``numpy.random.Generator`` backed by a
``SeedSequence(master_seed, spawn_key=(hash_of_labels,))``.
"""

from __future__ import annotations

import hashlib

import numpy as np


class SeedTree:
    """Create reproducible child generators from a single master seed."""

    def __init__(self, master_seed: int) -> None:
        self.master_seed = master_seed

    def child(self, *labels: str | int) -> np.random.Generator:
        """Return a deterministic generator addressed by ``labels``."""
        normalised = [str(label) for label in labels]
        joined = "|".join(normalised).encode("utf-8")
        digest = hashlib.sha256(joined).digest()
        hash_of_labels = int.from_bytes(digest[:8], "little")
        sequence = np.random.SeedSequence(
            self.master_seed, spawn_key=(hash_of_labels,)
        )
        return np.random.default_rng(sequence)
