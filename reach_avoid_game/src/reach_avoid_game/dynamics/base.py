"""Base marker class for OptimizedDP-style reach-avoid dynamics.

Solver-facing dynamics implement the OptimizedDP methods:
``opt_ctrl(t, state, spat_deriv)``, ``opt_dstb(t, state, spat_deriv)``,
and ``dynamics(t, state, uOpt, dOpt)``. Some classes also provide NumPy
helpers for tests and online control extraction.
"""


class DynamicsBase:
    """Marker base class for package-local type consistency."""
