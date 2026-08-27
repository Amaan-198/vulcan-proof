"""Hidden-truth Phase-1 simulator package."""

from .calibrate import calibrate_funnel
from .generator import generate_world
from .resolve import resolve, resolve_outcomes
from .history import add_history_features

__all__ = [
    "add_history_features",
    "calibrate_funnel",
    "generate_world",
    "resolve",
    "resolve_outcomes",
]
