"""Safety-filter lifecycle errors, independent of optional solver imports."""


class FilterFailure(RuntimeError):
    """Stop the simulator before applying the rejected command."""
