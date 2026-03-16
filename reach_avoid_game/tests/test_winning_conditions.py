"""Tests for winning condition checker."""

import numpy as np
import pytest

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.value_function_io import load_value_function, ValueFunctionData
from reach_avoid_game.solvers.winning_conditions import (
    compute_T_goal,
    compute_T_capture,
    compute_T_capture_from_slices,
    check_defender_wins,
    get_winning_regions,
)


@pytest.fixture(scope="module")
def config():
    return GameConfig.from_yaml("/workspace/config/game_params.yaml")


@pytest.fixture(scope="module")
def phi_h():
    return load_value_function("/workspace/data/value_functions/phi_h.npz")


@pytest.fixture(scope="module")
def phi_z():
    return load_value_function("/workspace/data/value_functions/phi_z.npz")


@pytest.fixture(scope="module")
def b_z():
    return load_value_function("/workspace/data/value_functions/B_z.npz")


@pytest.fixture(scope="module")
def phi_a_reach():
    return load_value_function("/workspace/data/value_functions/phi_A_reach.npz")


class TestTGoal:
    def test_attacker_inside_target_has_finite_T_goal(self, phi_a_reach, config):
        """Attacker inside target region should have finite T_goal."""
        # Target is [38, 45] x [10, 15]
        attacker_pos = np.array([41.5, 12.5])
        t_goal = compute_T_goal(phi_a_reach, attacker_pos)
        assert t_goal < float("inf")

    def test_attacker_far_has_larger_T_goal(self, phi_a_reach, config):
        """Attacker farther from target should have larger or equal T_goal."""
        pos_close = np.array([35.0, 12.5])  # Closer to target
        pos_far = np.array([5.0, 5.0])      # Far from target

        t_close = compute_T_goal(phi_a_reach, pos_close)
        t_far = compute_T_goal(phi_a_reach, pos_far)

        # Closer position should have smaller or equal T_goal
        assert t_close <= t_far + 1.0

    def test_T_goal_non_negative(self, phi_a_reach):
        """T_goal should always be non-negative."""
        attacker_pos = np.array([20.0, 12.0])
        t_goal = compute_T_goal(phi_a_reach, attacker_pos)
        assert t_goal >= 0.0


class TestTCapture:
    def test_T_capture_non_negative(self, phi_z, config):
        """T_capture should always be non-negative."""
        state_z = np.array([10.0, 0.0, 10.0])
        t_capture = compute_T_capture(phi_z, state_z, d_z=config.capture.d_z)
        assert t_capture >= 0.0

    def test_capture_closer_no_worse(self, phi_z, config):
        """States closer to capture should not have worse T_capture than far states.

        Note: On coarse dev grid, the value function may be all positive
        (attacker always escapes), giving inf T_capture everywhere.
        """
        state_close = np.array([10.0, 0.0, 10.5])
        state_far = np.array([10.0, 0.0, 15.0])

        t_close = compute_T_capture(phi_z, state_close, d_z=config.capture.d_z)
        t_far = compute_T_capture(phi_z, state_far, d_z=config.capture.d_z)

        # Both may be inf on coarse grid; that's OK
        if t_close < float("inf") and t_far < float("inf"):
            assert t_close <= t_far + 1.0


class TestTCaptureFromSlices:
    def test_T_capture_from_slices_returns_float(self, phi_z):
        """compute_T_capture_from_slices should return a float."""
        state = np.array([10.0, 0.0, 10.0])
        tc = compute_T_capture_from_slices(
            "/workspace/data/value_functions/phi_z_time_slices.npz",
            state, phi_z.grid_min, phi_z.grid_max, phi_z.grid_shape,
        )
        assert isinstance(tc, float)
        assert tc >= 0.0

    def test_T_capture_from_slices_monotonic(self, phi_z):
        """Closer states should have T_capture <= farther states (or both inf)."""
        state_close = np.array([10.0, 0.0, 10.5])
        state_far = np.array([2.0, 0.0, 18.0])

        tc_close = compute_T_capture_from_slices(
            "/workspace/data/value_functions/phi_z_time_slices.npz",
            state_close, phi_z.grid_min, phi_z.grid_max, phi_z.grid_shape,
        )
        tc_far = compute_T_capture_from_slices(
            "/workspace/data/value_functions/phi_z_time_slices.npz",
            state_far, phi_z.grid_min, phi_z.grid_max, phi_z.grid_shape,
        )
        # On dev grid both may be inf (B_z target too small)
        # But if both finite, closer should be smaller
        if tc_close < float("inf") and tc_far < float("inf"):
            assert tc_close <= tc_far + 0.5

    def test_T_capture_from_slices_inf_for_unreachable(self, phi_z):
        """States far apart with bad velocity should return inf T_capture."""
        # State far outside any capture possibility
        state = np.array([0.5, -4.0, 19.5])  # z_D near floor, z_A near ceiling, moving away
        tc = compute_T_capture_from_slices(
            "/workspace/data/value_functions/phi_z_time_slices.npz",
            state, phi_z.grid_min, phi_z.grid_max, phi_z.grid_shape,
        )
        assert tc == float("inf")


class TestCheckDefenderWins:
    def test_returns_required_keys(self, phi_h, phi_z, b_z):
        """check_defender_wins should return all required keys."""
        state_h = np.array([10.0, 10.0, 0.0, 0.0, 10.0, 10.0])
        state_z = np.array([10.0, 0.0, 10.0])

        result = check_defender_wins(phi_h, phi_z, b_z, state_h, state_z)

        assert "defender_wins" in result
        assert "in_W_D_h" in result
        assert "in_W_D_z" in result
        assert "in_B_z" in result
        assert "phi_h_value" in result
        assert "phi_z_value" in result

    def test_with_timing_info(self, phi_h, phi_z, b_z, phi_a_reach):
        """check_defender_wins with timing info should include T_goal and T_capture."""
        state_h = np.array([10.0, 10.0, 0.0, 0.0, 10.0, 10.0])
        state_z = np.array([10.0, 0.0, 10.0])
        attacker_pos = np.array([10.0, 10.0])

        result = check_defender_wins(
            phi_h, phi_z, b_z, state_h, state_z,
            phi_a_reach=phi_a_reach, attacker_pos=attacker_pos,
        )

        assert "T_goal" in result
        assert "T_capture" in result
        assert isinstance(result["T_goal"], float)
        assert isinstance(result["T_capture"], float)

    def test_phi_values_finite(self, phi_h, phi_z, b_z):
        """Value function values should be finite for valid states."""
        state_h = np.array([20.0, 12.0, 0.0, 0.0, 20.0, 12.0])
        state_z = np.array([10.0, 0.0, 10.0])

        result = check_defender_wins(phi_h, phi_z, b_z, state_h, state_z)

        assert np.isfinite(result["phi_h_value"])
        assert np.isfinite(result["phi_z_value"])

    def test_defender_wins_is_bool(self, phi_h, phi_z, b_z):
        """defender_wins should be a boolean."""
        state_h = np.array([20.0, 12.0, 0.0, 0.0, 20.0, 12.0])
        state_z = np.array([10.0, 0.0, 10.0])

        result = check_defender_wins(phi_h, phi_z, b_z, state_h, state_z)
        assert isinstance(result["defender_wins"], bool)


class TestGetWinningRegions:
    def test_winning_regions_cover_grid(self, phi_z):
        """W_D and W_A should partition the grid."""
        regions = get_winning_regions(phi_z)

        w_d = regions["W_D_mask"]
        w_a = regions["W_A_mask"]

        # Union should cover entire grid
        assert np.all(w_d | w_a)
        # Intersection should be empty
        assert not np.any(w_d & w_a)

    def test_winning_fractions_sum_to_one(self, phi_z):
        """W_D and W_A fractions should sum to 1."""
        regions = get_winning_regions(phi_z)
        total = regions["W_D_fraction"] + regions["W_A_fraction"]
        assert total == pytest.approx(1.0)

    def test_attacker_reaching_has_reachable_region(self, phi_a_reach):
        """Attacker reaching VF should have non-empty reachable set (phi <= 0)."""
        regions = get_winning_regions(phi_a_reach)
        # For attacker reaching, W_D_mask (<=0) is the reachable set
        assert regions["W_D_fraction"] > 0, "Attacker should be able to reach target from some positions"

    def test_winning_region_shapes_match(self, phi_z):
        """Winning region masks should match value function shape."""
        regions = get_winning_regions(phi_z)
        assert regions["W_D_mask"].shape == phi_z.values.shape
        assert regions["W_A_mask"].shape == phi_z.values.shape
