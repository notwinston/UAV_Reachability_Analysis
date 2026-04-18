"""Small helpers for OptimizedDP dynamics classes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """Lightweight bounds container kept for tests and diagnostics."""

    lo: np.ndarray
    hi: np.ndarray


def make_box(lo, hi) -> Box:
    return Box(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))


def require_hcl():
    try:
        import heterocl as hcl
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise ImportError(
            "OptimizedDP dynamics require heterocl at solve time. "
            "Activate the optimized_dp conda environment before running HJSolver."
        ) from exc
    return hcl
