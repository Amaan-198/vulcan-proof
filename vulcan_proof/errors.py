"""Exception hierarchy for invariant failures in Vulcan Proof."""


class VulcanError(Exception):
    """Base class for errors raised by the project."""


class InvariantError(VulcanError):
    """Raised when an input or computed value violates a project invariant."""


class SchemaError(VulcanError):
    """Raised when a DataFrame or parameter block has the wrong schema."""


class LeakError(VulcanError):
    """Raised when hidden or forbidden information crosses an artifact boundary."""
