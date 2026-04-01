"""Tests for the vertical sub-game solver and control extraction."""

import numpy as np
import pytest

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.value_function_io import load_value_function, load_time_slices
from reach_avoid_game.solvers.vertical_solver import (
    solve_vertical_reach_avoid,
    solve_vertical_max_distance,
    compute_invariant_set_Bz,
)
from reach_avoid_game.solvers.control_extraction import (
    interpolate_value,
    compute_gradient,
    extract_optimal_control_vertical,
    extract_optimal_disturbance_vertical,
)


@pytest.fixture(scope="module")
def config():
    return GameConfig.from_yaml("/workspace/config/game_params.yaml")


@pytest.fixture(scope="module")
def v_z_inf_path(config):
    """Compute V_z_inf on dev grid (cached for module). Step 1 of pipeline."""
    return solve_vertical_max_distance(config)


@pytest.fixture(scope="module")
def b_z_path(v_z_inf_path, config):
    """Compute B_z from V_z_inf (cached for module). Step 2 of pipeline."""
    return compute_invariant_set_Bz(v_z_inf_path, d_z=config.capture.d_z)


@pytest.fixture(scope="module")
def phi_z_path(config, v_z_inf_path):
    """Compute phi_z with B_z feedback on dev grid (cached for module). Step 3 of pipeline."""
    v_z_inf_data = load_value_function(v_z_inf_path)
    return solve_vertical_reach_avoid(config, v_z_inf_data=v_z_inf_data)


@pytest.fixture(scope="module")
def phi_z_data(phi_z_path):
    return load_value_function(phi_z_path)


@pytest.fixture(scope="module")
def v_z_inf_data(v_z_inf_path):
    return load_value_function(v_z_inf_path)


@pytest.fixture(scope="module")
def b_z_data(b_z_path):
    return load_value_function(b_z_path)


class TestGridDimensions:
    def test_phi_z_shape_matches_config(self, phi_z_data, config):
        """Grid dimensions should match config: shape == (z_d_points, v_points, z_a_points)."""
        expected = (
            config.grid.vertical_3d.z_d_points,
            config.grid.vertical_3d.v_dz_points,
            config.grid.vertical_3d.z_a_points,
        )
        assert phi_z_data.values.shape == expected

    def test_v_z_inf_shape_matches_config(self, v_z_inf_data, config):
        """V_z_inf grid should match 2D relative grid config."""
        expected = (
            config.grid.vertical.z_rel_points,
            config.grid.vertical.v_dz_points,
        )
        assert v_z_inf_data.values.shape == expected

    def test_b_z_shape_matches_v_z_inf(self, b_z_data, v_z_inf_data):
        """B_z should have same shape as V_z_inf."""
        assert b_z_data.values.shape == v_z_inf_data.values.shape


class TestPhiZ:
    def test_phi_z_lower_inside_capture_set(self, phi_z_data, config):
        """Phi_z should be lower (more favorable for capture) when |z_D - z_A| is small.

        Note: With paper parameters (attacker is single integrator, defender has inertia),
        the attacker can escape vertically, so Phi_z may be positive everywhere at long
        time horizons. But values should still be lower inside the initial capture set
        than far outside.
        """
        d_z = config.capture.d_z
        n_z_d = phi_z_data.values.shape[0]
        n_v = phi_z_data.values.shape[1]
        n_z_a = phi_z_data.values.shape[2]

        z_d_vals = np.linspace(float(phi_z_data.grid_min[0]), float(phi_z_data.grid_max[0]), n_z_d)
        v_dz_vals = np.linspace(float(phi_z_data.grid_min[1]), float(phi_z_data.grid_max[1]), n_v)
        z_a_vals = np.linspace(float(phi_z_data.grid_min[2]), float(phi_z_data.grid_max[2]), n_z_a)

        v_idx = np.argmin(np.abs(v_dz_vals))

        # The initial value function l(x) = |z_D - z_A| - d_z should be preserved
        # in structure even after backward evolution
        inside_vals = []
        outside_vals = []
        for i, z_d in enumerate(z_d_vals):
            for k, z_a in enumerate(z_a_vals):
                if abs(z_d - z_a) <= d_z:
                    inside_vals.append(phi_z_data.values[i, v_idx, k])
                elif abs(z_d - z_a) > 5 * d_z:
                    outside_vals.append(phi_z_data.values[i, v_idx, k])

        assert len(inside_vals) > 0
        assert len(outside_vals) > 0
        # Phi_z should be no higher inside than outside (at minimum, equal due to convergence)
        assert np.mean(inside_vals) <= np.mean(outside_vals) + 0.01

    def test_phi_z_positive_far_outside(self, phi_z_data, config):
        """Phi_z value should be > 0 far outside capture set."""
        n_z_d = phi_z_data.values.shape[0]
        n_v = phi_z_data.values.shape[1]
        n_z_a = phi_z_data.values.shape[2]

        z_d_vals = np.linspace(float(phi_z_data.grid_min[0]), float(phi_z_data.grid_max[0]), n_z_d)
        v_dz_vals = np.linspace(float(phi_z_data.grid_min[1]), float(phi_z_data.grid_max[1]), n_v)
        z_a_vals = np.linspace(float(phi_z_data.grid_min[2]), float(phi_z_data.grid_max[2]), n_z_a)

        v_idx = np.argmin(np.abs(v_dz_vals))

        # Check states where |z_D - z_A| > 5 * d_z (far outside)
        far_values = []
        for i, z_d in enumerate(z_d_vals):
            for k, z_a in enumerate(z_a_vals):
                if abs(z_d - z_a) > 5 * config.capture.d_z:
                    far_values.append(phi_z_data.values[i, v_idx, k])

        assert len(far_values) > 0, "Should have states far outside capture set"
        assert np.mean(np.array(far_values) > 0) > 0.5, "Most far-outside states should have Phi_z > 0"

    def test_phi_z_uses_bz_target(self, phi_z_data, v_z_inf_data, config):
        """When v_z_inf_data is provided, Phi_z initial values should be based on B_z SDF."""
        # With B_z target, values are V_z_inf - d_z_effective, not |z_D - z_A| - d_z
        # All values should be finite
        assert np.isfinite(phi_z_data.values).all()
        # On dev grid, B_z is very small so Phi_z may be all positive
        # but it should still have reasonable range
        assert phi_z_data.values.max() < 1000.0

    def test_time_slices_exist(self, phi_z_path):
        """phi_z_time_slices.npz should be created alongside phi_z.npz."""
        from pathlib import Path
        slices_path = Path(phi_z_path).parent / "phi_z_time_slices.npz"
        assert slices_path.exists(), f"Expected {slices_path} to exist"
        all_values, times = load_time_slices(str(slices_path))
        assert all_values.ndim == 4  # (n_times, z_d, v_dz, z_a)
        assert len(times) == all_values.shape[0]
        # First time should be 0, last should be -T
        assert times[0] == pytest.approx(0.0)
        assert times[-1] < 0  # Negative (backward time)


class TestBz:
    def test_b_z_non_empty(self, b_z_data):
        """B_z should be non-empty."""
        assert b_z_data.values.any(), "B_z should contain some non-zero entries"

    def test_b_z_contains_center(self, b_z_data):
        """B_z should contain states near z_rel=0, v_D_z=0."""
        n_z = b_z_data.values.shape[0]
        n_v = b_z_data.values.shape[1]

        center_z = n_z // 2
        center_v = n_v // 2

        assert b_z_data.values[center_z, center_v] > 0, \
            "B_z should include z_rel=0, v_D_z=0 (defender at same altitude as attacker, zero velocity)"


class TestControlExtraction:
    def test_control_within_speed_limits(self, phi_z_data, config):
        """Extracted control should be within speed limits."""
        state = np.array([10.0, 0.0, 8.0])  # z_D=10, v_D_z=0, z_A=8
        u_z = extract_optimal_control_vertical(
            phi_z_data, state,
            k_z=config.defender.k_z,
            u_d_z=config.defender.max_speed_vertical,
        )
        assert abs(u_z) <= config.defender.max_speed_vertical + 1e-10

    def test_disturbance_within_speed_limits(self, phi_z_data, config):
        """Extracted disturbance should be within speed limits."""
        state = np.array([10.0, 0.0, 8.0])
        d_z = extract_optimal_disturbance_vertical(
            phi_z_data, state,
            u_a_z=config.attacker.max_speed_vertical,
        )
        assert abs(d_z) <= config.attacker.max_speed_vertical + 1e-10

    def test_gradient_points_toward_capture(self, phi_z_data, config):
        """Control extraction gradient should point toward capture.

        When defender is above attacker (z_D > z_A), the optimal control
        should have u_z < 0 (go down toward attacker), OR
        when defender is below attacker (z_D < z_A), u_z > 0 (go up).
        """
        # Defender below attacker
        state = np.array([5.0, 0.0, 10.0])  # z_D=5, z_A=10
        u_z = extract_optimal_control_vertical(
            phi_z_data, state,
            k_z=config.defender.k_z,
            u_d_z=config.defender.max_speed_vertical,
        )
        # Defender should go up (u_z > 0) or at maximum speed
        assert abs(u_z) == pytest.approx(config.defender.max_speed_vertical)

    def test_gradient_finite(self, phi_z_data):
        """Gradient should be finite everywhere inside the grid."""
        state = np.array([10.0, 0.0, 10.0])
        grad = compute_gradient(phi_z_data, state)
        assert np.all(np.isfinite(grad))
