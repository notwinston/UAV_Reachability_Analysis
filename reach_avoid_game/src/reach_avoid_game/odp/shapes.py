"""Shape functions for defining initial value functions / target sets.

Compatible with OptimizedDP's Shapes API.
"""

from __future__ import annotations

import numpy as np

from reach_avoid_game.odp.grid import Grid


def CylinderShape(
    grid: Grid,
    ignore_dims: list[int],
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Create a cylinder shape (SDF).

    Negative inside, positive outside.

    Args:
        grid: Computational grid.
        ignore_dims: Dimensions to ignore (cylinder axis).
        center: Center coordinates (length = grid.dims).
        radius: Cylinder radius.

    Returns:
        SDF array (same shape as grid).
    """
    center = np.asarray(center, dtype=np.float64)
    sum_sq = np.zeros(tuple(grid.pts_each_dim))
    for d in range(grid.dims):
        if d not in ignore_dims:
            sum_sq = sum_sq + (grid.vs[d] - center[d]) ** 2
    return np.sqrt(sum_sq) - radius


def ShapeRectangle(
    grid: Grid,
    target_min: np.ndarray | list,
    target_max: np.ndarray | list,
) -> np.ndarray:
    """Create a rectangle/box shape (SDF).

    Negative inside, positive outside.

    Args:
        grid: Computational grid.
        target_min: Lower bounds for each dimension (use -np.inf to ignore).
        target_max: Upper bounds for each dimension (use np.inf to ignore).

    Returns:
        SDF array.
    """
    target_min = np.asarray(target_min, dtype=np.float64)
    target_max = np.asarray(target_max, dtype=np.float64)

    sdf = np.full(tuple(grid.pts_each_dim), -np.inf)
    for d in range(grid.dims):
        if target_min[d] != -np.inf:
            sdf = np.maximum(sdf, target_min[d] - grid.vs[d])
        if target_max[d] != np.inf:
            sdf = np.maximum(sdf, grid.vs[d] - target_max[d])
    return sdf
