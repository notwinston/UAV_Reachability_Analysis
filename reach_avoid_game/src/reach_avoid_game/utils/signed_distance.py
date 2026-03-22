"""Signed distance function utilities for defining game regions."""

from __future__ import annotations

import numpy as np


def sdf_box(point: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray:
    """Signed distance to an axis-aligned box. Negative inside, positive outside."""
    center = (bounds_min + bounds_max) / 2.0
    half_size = (bounds_max - bounds_min) / 2.0
    q = np.abs(point - center) - half_size
    outside_dist = np.sqrt(np.sum(np.maximum(q, 0.0) ** 2, axis=-1))
    inside_dist = np.minimum(np.max(q, axis=-1), 0.0)
    return outside_dist + inside_dist


def sdf_cylinder_z(point_xy: np.ndarray, center_xy: np.ndarray, radius: float) -> np.ndarray:
    """Signed distance to a z-aligned cylinder in the xy-plane. Negative inside."""
    diff = point_xy - center_xy
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist - radius


def sdf_union(sdf1: np.ndarray, sdf2: np.ndarray) -> np.ndarray:
    """Union of two signed distance fields (min)."""
    return np.minimum(sdf1, sdf2)


def sdf_intersection(sdf1: np.ndarray, sdf2: np.ndarray) -> np.ndarray:
    """Intersection of two signed distance fields (max)."""
    return np.maximum(sdf1, sdf2)
