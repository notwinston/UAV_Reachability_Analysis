"""Signed distance function utilities for defining game regions."""

from __future__ import annotations

import jax.numpy as jnp


def sdf_box(point: jnp.ndarray, bounds_min: jnp.ndarray, bounds_max: jnp.ndarray) -> jnp.ndarray:
    """Signed distance to an axis-aligned box. Negative inside, positive outside."""
    center = (bounds_min + bounds_max) / 2.0
    half_size = (bounds_max - bounds_min) / 2.0
    q = jnp.abs(point - center) - half_size
    outside_dist = jnp.sqrt(jnp.sum(jnp.maximum(q, 0.0) ** 2, axis=-1))
    inside_dist = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return outside_dist + inside_dist


def sdf_cylinder_z(point_xy: jnp.ndarray, center_xy: jnp.ndarray, radius: float) -> jnp.ndarray:
    """Signed distance to a z-aligned cylinder in the xy-plane. Negative inside."""
    diff = point_xy - center_xy
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1))
    return dist - radius


def sdf_union(sdf1: jnp.ndarray, sdf2: jnp.ndarray) -> jnp.ndarray:
    """Union of two signed distance fields (min)."""
    return jnp.minimum(sdf1, sdf2)


def sdf_intersection(sdf1: jnp.ndarray, sdf2: jnp.ndarray) -> jnp.ndarray:
    """Intersection of two signed distance fields (max)."""
    return jnp.maximum(sdf1, sdf2)
