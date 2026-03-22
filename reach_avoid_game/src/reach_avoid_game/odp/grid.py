"""Computational grid for HJ reachability — OptimizedDP-compatible API."""

from __future__ import annotations

import numpy as np


class Grid:
    """N-dimensional computational grid.

    Compatible with the SFU-MARS/optimized_dp Grid API.

    Args:
        minBounds: Lower bounds for each dimension.
        maxBounds: Upper bounds for each dimension.
        dims: Number of dimensions.
        pts_each_dim: Number of grid points per dimension.
        periodicDims: List of periodic dimension indices (0-indexed).
    """

    def __init__(
        self,
        minBounds: list | np.ndarray,
        maxBounds: list | np.ndarray,
        dims: int,
        pts_each_dim: list | np.ndarray,
        periodicDims: list | None = None,
    ):
        self.min = np.asarray(minBounds, dtype=np.float64)
        self.max = np.asarray(maxBounds, dtype=np.float64).copy()
        self.dims = dims
        self.pts_each_dim = np.asarray(pts_each_dim, dtype=np.int32)
        self.pDim = periodicDims or []

        assert len(self.min) == len(self.max) == len(self.pts_each_dim) == dims

        # Exclude upper bound for periodic dimensions
        for dim in self.pDim:
            self.max[dim] = self.min[dim] + (self.max[dim] - self.min[dim]) * (
                1 - 1.0 / self.pts_each_dim[dim]
            )

        self.dx = (self.max - self.min) / (self.pts_each_dim - 1.0)

        # Build per-dimension coordinate arrays
        self.vs = []
        self.grid_points = []
        for i in range(dims):
            pts = np.linspace(self.min[i], self.max[i], num=int(self.pts_each_dim[i]))
            self.grid_points.append(pts)
            # Reshape for broadcasting: shape (1,...,N_i,...,1)
            broadcast_shape = np.ones(dims, dtype=int)
            broadcast_shape[i] = self.pts_each_dim[i]
            self.vs.append(pts.reshape(tuple(broadcast_shape)))

    def get_indices(self, states: np.ndarray) -> tuple:
        """Get nearest grid indices for given states."""
        indices = np.round((states - self.min) / self.dx)
        indices = np.clip(indices, 0, self.pts_each_dim - 1)
        return tuple(indices.astype(int).T)

    def get_values(self, V: np.ndarray, states: np.ndarray) -> float | np.ndarray:
        """Get value function values at given states (nearest-neighbor)."""
        indices = self.get_indices(states)
        return V[indices]

    def __str__(self) -> str:
        return (
            f"Grid:\n"
            f"  min: {self.min}\n"
            f"  max: {self.max}\n"
            f"  pts_each_dim: {self.pts_each_dim}\n"
            f"  pDim: {self.pDim}\n"
            f"  dx: {self.dx}\n"
        )
