"""SeedTree tests prove label-addressed reproducibility and separation."""

from __future__ import annotations

import numpy as np

from vulcan_proof.params import P
from vulcan_proof.seeds import SeedTree


def test_same_labels_same_first_ten_draws() -> None:
    tree = SeedTree(P["run.master_seed"])
    left = tree.child("olist", "lgbm").random(10)
    right = tree.child("olist", "lgbm").random(10)
    assert np.array_equal(left, right)


def test_different_labels_differ() -> None:
    tree = SeedTree(P["run.master_seed"])
    left = tree.child("olist", "lgbm").random(10)
    right = tree.child("olist", "features").random(10)
    assert not np.array_equal(left, right)
