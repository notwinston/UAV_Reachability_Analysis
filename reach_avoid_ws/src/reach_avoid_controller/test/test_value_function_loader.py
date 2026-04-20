"""Tests for ValueFunctionLoader -- standalone pytest, no rclpy required."""

import sys
import os
from pathlib import Path
import numpy as np
import pytest

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/workspace/reach_avoid_game/src")

from reach_avoid_controller.value_function_loader import ValueFunctionLoader, VF_NAMES

REPO_ROOT = Path(__file__).resolve().parents[4]
VF_DIR = str(REPO_ROOT / "data" / "value_functions")


@pytest.fixture(scope="module")
def loader():
    return ValueFunctionLoader(VF_DIR)


class TestLoading:
    def test_all_value_functions_loaded(self, loader):
        assert loader.all_loaded, f"Missing VFs: {set(VF_NAMES) - set(loader.loaded_names)}"

    def test_loaded_names(self, loader):
        for name in VF_NAMES:
            assert name in loader.loaded_names

    def test_params_available(self, loader):
        params = loader.get_params("phi_z")
        assert "d_z" in params
        assert "U_D_z" in params


class TestInterpolation:
    def test_phi_z_returns_finite(self, loader):
        state = np.array([10.0, 0.0, 10.0])
        val = loader.get_value("phi_z", state)
        assert np.isfinite(val)

    def test_V_z_inf_returns_finite(self, loader):
        state = np.array([0.0, 0.0])
        val = loader.get_value("V_z_inf", state)
        assert np.isfinite(val)

    def test_B_z_returns_finite(self, loader):
        state = np.array([0.0, 0.0])
        val = loader.get_value("B_z", state)
        assert np.isfinite(val)

    def test_phi_h_returns_finite(self, loader):
        # [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
        state = np.array([20.0, 12.0, 0.0, 0.0, 25.0, 12.0])
        val = loader.get_value("phi_h", state)
        assert np.isfinite(val)

    def test_V_h_T_returns_finite(self, loader):
        # [x_rel, y_rel, vx_D, vy_D]
        state = np.array([0.0, 0.0, 0.0, 0.0])
        val = loader.get_value("V_h_T", state)
        assert np.isfinite(val)

    def test_B_h_returns_finite(self, loader):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        val = loader.get_value("B_h", state)
        assert np.isfinite(val)

    def test_phi_A_reach_returns_finite(self, loader):
        state = np.array([20.0, 12.0])
        val = loader.get_value("phi_A_reach", state)
        assert np.isfinite(val)

    def test_B_z_inside_at_zero_rel(self, loader):
        """B_z should be ~1 (inside) at z_rel=0, v_D_z=0."""
        val = loader.get_value("B_z", np.array([0.0, 0.0]))
        assert val > 0.5

    def test_B_z_outside_at_large_rel(self, loader):
        """B_z should be 0 (outside) at large z_rel."""
        val = loader.get_value("B_z", np.array([5.0, 0.0]))
        assert val < 0.5

    def test_out_of_bounds_clamped(self, loader):
        """Out-of-bounds state should be clamped, not error."""
        state = np.array([100.0, 100.0, 100.0])
        val = loader.get_value("phi_z", state)
        assert np.isfinite(val)


class TestGradient:
    def test_phi_z_gradient_finite(self, loader):
        state = np.array([10.0, 0.0, 10.0])
        grad = loader.get_gradient("phi_z", state)
        assert grad.shape == (3,)
        assert np.all(np.isfinite(grad))

    def test_V_z_inf_gradient_finite(self, loader):
        state = np.array([0.0, 0.0])
        grad = loader.get_gradient("V_z_inf", state)
        assert grad.shape == (2,)
        assert np.all(np.isfinite(grad))

    def test_phi_h_gradient_finite(self, loader):
        state = np.array([20.0, 12.0, 0.0, 0.0, 25.0, 12.0])
        grad = loader.get_gradient("phi_h", state)
        assert grad.shape == (6,)
        assert np.all(np.isfinite(grad))

    def test_V_h_T_gradient_finite(self, loader):
        state = np.array([2.0, 2.0, 1.0, 1.0])
        grad = loader.get_gradient("V_h_T", state)
        assert grad.shape == (4,)
        assert np.all(np.isfinite(grad))

    def test_gradient_nonzero_at_nontrivial_point(self, loader):
        """At a non-trivial state, gradient should not be all zero."""
        state = np.array([5.0, 1.0, 8.0])
        grad = loader.get_gradient("phi_z", state)
        assert np.linalg.norm(grad) > 0


class TestErrorHandling:
    def test_missing_vf_raises(self, loader):
        with pytest.raises(KeyError):
            loader.get_value("nonexistent_vf", np.array([0.0]))

    def test_missing_dir_loads_nothing(self):
        loader2 = ValueFunctionLoader("/nonexistent/path/")
        assert not loader2.all_loaded
        assert len(loader2.loaded_names) == 0
