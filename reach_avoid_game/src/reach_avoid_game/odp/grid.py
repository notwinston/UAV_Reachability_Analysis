"""OptimizedDP grid import used by the reach-avoid package."""

from __future__ import annotations

try:
    from odp.Grid import Grid
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise ImportError(
        "OptimizedDP is required for reach_avoid_game.odp.grid. "
        "Install/activate the SFU-MARS optimized_dp environment so "
        "`from odp.Grid import Grid` succeeds."
    ) from exc

__all__ = ["Grid"]
