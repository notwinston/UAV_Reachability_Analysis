"""Tests for value function save/load round-trip."""

import tempfile
from pathlib import Path

import numpy as np

from reach_avoid_game.solvers.value_function_io import (
    ValueFunctionData,
    load_value_function,
    save_value_function,
)


def test_save_load_roundtrip():
    values = np.random.randn(10, 10, 10)
    data = ValueFunctionData(
        values=values,
        grid_min=np.array([-1.0, -1.0, -1.0]),
        grid_max=np.array([1.0, 1.0, 1.0]),
        grid_shape=(10, 10, 10),
        params={"time_horizon": 15.0, "accuracy": "medium"},
        description="test value function",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_vf.npz"
        save_value_function(path, data)
        loaded = load_value_function(path)

        np.testing.assert_array_almost_equal(loaded.values, values)
        np.testing.assert_array_almost_equal(loaded.grid_min, data.grid_min)
        np.testing.assert_array_almost_equal(loaded.grid_max, data.grid_max)
        assert loaded.grid_shape == (10, 10, 10)
        assert loaded.description == "test value function"


def test_save_load_2d():
    values = np.random.randn(20, 15)
    data = ValueFunctionData(
        values=values,
        grid_min=np.array([-2.0, -1.0]),
        grid_max=np.array([2.0, 1.0]),
        grid_shape=(20, 15),
        params={},
        description="2D test",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_2d.npz"
        save_value_function(path, data)
        loaded = load_value_function(path)

        np.testing.assert_array_almost_equal(loaded.values, values)
        assert loaded.grid_shape == (20, 15)
