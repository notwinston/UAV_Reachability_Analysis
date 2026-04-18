"""OptimizedDP solver imports used by the reach-avoid package.

This module intentionally re-exports the real SFU-MARS OptimizedDP
implementation. It exists so the rest of this package can keep importing
``reach_avoid_game.odp.solver`` while the backend is the installed ``odp``
package, not an hj_reachability compatibility wrapper.
"""

from __future__ import annotations

try:
    from odp.solver import HJSolver, TTRSolver, computeSpatDerivArray
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise ImportError(
        "OptimizedDP is required for reach_avoid_game.odp.solver. "
        "Install/activate the SFU-MARS optimized_dp environment so "
        "`from odp.solver import HJSolver` succeeds."
    ) from exc

__all__ = ["HJSolver", "TTRSolver", "computeSpatDerivArray"]
