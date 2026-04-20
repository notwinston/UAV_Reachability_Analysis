"""Tests for value-function control extraction helpers."""

import numpy as np

from reach_avoid_game.solvers.control_extraction import (
    extract_optimal_control_vertical,
    extract_optimal_disturbance_vertical,
    is_deep_inside_invariant_set,
)
from reach_avoid_game.solvers.value_function_io import ValueFunctionData


def test_deep_inside_invariant_set_uses_distance_threshold():
    """Distance-like invariant values are deep inside below the shrunken threshold."""
    axis = np.linspace(0.0, 1.0, 5)
    values = np.broadcast_to(axis[:, None], (5, 3)).copy()
    vf_data = ValueFunctionData(
        values=values,
        grid_min=np.array([0.0, 0.0]),
        grid_max=np.array([1.0, 1.0]),
        grid_shape=values.shape,
    )

    assert is_deep_inside_invariant_set(
        vf_data, np.array([0.5, 0.5]), d_z=1.0, margin_fraction=0.3,
    )
    assert not is_deep_inside_invariant_set(
        vf_data, np.array([0.75, 0.5]), d_z=1.0, margin_fraction=0.3,
    )


def test_vertical_control_deadband_returns_zero():
    values = np.ones((3, 3, 3))
    vf_data = ValueFunctionData(
        values=values,
        grid_min=np.array([0.0, 0.0, 0.0]),
        grid_max=np.array([2.0, 2.0, 2.0]),
        grid_shape=values.shape,
    )

    assert extract_optimal_control_vertical(
        vf_data, np.array([1.0, 1.0, 1.0]), k_z=1.5, u_d_z=4.0,
    ) == 0.0


def test_vertical_disturbance_deadband_returns_zero():
    values = np.ones((3, 3))
    vf_data = ValueFunctionData(
        values=values,
        grid_min=np.array([0.0, 0.0]),
        grid_max=np.array([2.0, 2.0]),
        grid_shape=values.shape,
    )

    assert extract_optimal_disturbance_vertical(
        vf_data, np.array([1.0, 1.0]), u_a_z=2.0,
    ) == 0.0
