"""Tests for dynamics models."""

import numpy as np
import jax.numpy as jnp
import pytest

from reach_avoid_game.config import GameConfig, DefenderConfig, AttackerConfig
from reach_avoid_game.dynamics import DefenderDynamics, AttackerDynamics, VerticalGameDynamics


@pytest.fixture
def config():
    return GameConfig.from_yaml("/workspace/config/game_params.yaml")


@pytest.fixture
def defender(config):
    return DefenderDynamics(config.defender)


@pytest.fixture
def attacker(config):
    return AttackerDynamics(config.attacker)


@pytest.fixture
def vertical_dynamics(config):
    return VerticalGameDynamics(config)


class TestDefenderDynamics:
    def test_state_derivatives(self, defender):
        """Verify dx/dt = [vx, vy, vz, k_x*(u_x-vx), k_y*(u_y-vy), k_z*(u_z-vz)]."""
        state = np.array([0.0, 0.0, 5.0, 1.0, 2.0, 0.5])
        control = np.array([3.0, 1.0, 2.0])

        deriv = defender.compute_derivatives(state, control)

        assert deriv[0] == pytest.approx(1.0)  # dx/dt = vx
        assert deriv[1] == pytest.approx(2.0)  # dy/dt = vy
        assert deriv[2] == pytest.approx(0.5)  # dz/dt = vz
        assert deriv[3] == pytest.approx(0.7 * (3.0 - 1.0))  # k_x*(u_x - vx)
        assert deriv[4] == pytest.approx(0.7 * (1.0 - 2.0))  # k_y*(u_y - vy)
        assert deriv[5] == pytest.approx(1.5 * (2.0 - 0.5))  # k_z*(u_z - vz)

    def test_zero_control_zero_velocity(self, defender):
        """With zero velocity and zero control, derivatives should be zero."""
        state = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0, 0.0])

        deriv = defender.compute_derivatives(state, control)
        np.testing.assert_array_almost_equal(deriv, np.zeros(6))

    def test_horizontal_speed_clamping(self, defender):
        """Input exceeding max horizontal speed should be clamped."""
        # max_speed_horizontal = 6.0
        control = np.array([10.0, 10.0, 0.0])
        clamped = defender.clamp_control(control)

        h_speed = np.sqrt(clamped[0] ** 2 + clamped[1] ** 2)
        assert h_speed == pytest.approx(6.0, abs=1e-10)

    def test_vertical_speed_clamping(self, defender):
        """Input exceeding max vertical speed should be clamped."""
        # max_speed_vertical = 4.0
        control = np.array([0.0, 0.0, 10.0])
        clamped = defender.clamp_control(control)
        assert clamped[2] == pytest.approx(4.0)

        control_neg = np.array([0.0, 0.0, -10.0])
        clamped_neg = defender.clamp_control(control_neg)
        assert clamped_neg[2] == pytest.approx(-4.0)

    def test_within_limits_not_clamped(self, defender):
        """Control within limits should not be modified."""
        control = np.array([2.0, 1.0, 1.5])
        clamped = defender.clamp_control(control)
        np.testing.assert_array_almost_equal(clamped, control)


class TestAttackerDynamics:
    def test_state_derivatives(self, attacker):
        """Verify dx/dt = [d_x, d_y, d_z]."""
        state = np.array([5.0, 10.0, 3.0])
        control = np.array([1.0, -2.0, 0.5])

        deriv = attacker.compute_derivatives(state, control)

        assert deriv[0] == pytest.approx(1.0)
        assert deriv[1] == pytest.approx(-2.0)
        assert deriv[2] == pytest.approx(0.5)

    def test_horizontal_speed_clamping(self, attacker):
        """Input exceeding max horizontal speed should be clamped."""
        # max_speed_horizontal = 3.0
        control = np.array([5.0, 5.0, 0.0])
        clamped = attacker.clamp_control(control)

        h_speed = np.sqrt(clamped[0] ** 2 + clamped[1] ** 2)
        assert h_speed == pytest.approx(3.0, abs=1e-10)

    def test_vertical_speed_clamping(self, attacker):
        """Input exceeding max vertical speed should be clamped."""
        # max_speed_vertical = 2.0
        control = np.array([0.0, 0.0, 5.0])
        clamped = attacker.clamp_control(control)
        assert clamped[2] == pytest.approx(2.0)


class TestVerticalGameDynamics:
    def test_state_derivatives_match_eq25(self, vertical_dynamics, config):
        """Verify 3D state derivatives match Eq. 25 from paper."""
        k_z = config.defender.k_z

        # State: [z_D, v_D_z, z_A]
        state = jnp.array([5.0, 1.0, 3.0])
        control = jnp.array([2.0])     # u_z
        disturbance = jnp.array([1.5])  # d_z

        deriv = vertical_dynamics(state, control, disturbance, 0.0)

        assert float(deriv[0]) == pytest.approx(1.0)              # z_D_dot = v_D_z
        assert float(deriv[1]) == pytest.approx(k_z * (2.0 - 1.0))  # v_D_z_dot = k_z*(u_z - v_D_z)
        assert float(deriv[2]) == pytest.approx(1.5)               # z_A_dot = d_z

    def test_open_loop_dynamics(self, vertical_dynamics, config):
        """Open-loop dynamics should show only drift."""
        k_z = config.defender.k_z
        state = jnp.array([5.0, 2.0, 3.0])

        f = vertical_dynamics.open_loop_dynamics(state, 0.0)

        assert float(f[0]) == pytest.approx(2.0)          # z_D_dot = v_D_z
        assert float(f[1]) == pytest.approx(-k_z * 2.0)   # -k_z * v_D_z
        assert float(f[2]) == pytest.approx(0.0)           # no disturbance

    def test_optimal_control_direction(self, vertical_dynamics):
        """Defender's optimal control moves toward attacker (for capture).

        When grad_value wrt v_D_z is positive (increasing value with increasing v_D_z),
        and defender maximizes value, the optimal u_z should be positive (max).
        """
        state = jnp.array([5.0, 0.0, 8.0])  # attacker above defender
        # Gradient pointing in direction where increasing v_D_z increases value
        grad_value = jnp.array([0.0, 1.0, 0.0])

        ctrl = vertical_dynamics.optimal_control(state, 0.0, grad_value)

        # Defender should choose max u_z to go up toward attacker
        assert float(ctrl[0]) == pytest.approx(4.0)  # U_D_z

    def test_optimal_disturbance_direction(self, vertical_dynamics):
        """Attacker's optimal disturbance moves away from defender.

        When grad_value wrt z_A is positive, and attacker minimizes value,
        the optimal d_z should be negative (to reduce value).
        """
        state = jnp.array([5.0, 0.0, 8.0])
        # Gradient: increasing z_A increases value
        grad_value = jnp.array([0.0, 0.0, 1.0])

        dstb = vertical_dynamics.optimal_disturbance(state, 0.0, grad_value)

        # Attacker minimizes → should go in negative z_A direction
        assert float(dstb[0]) == pytest.approx(-2.0)  # -U_A_z

    def test_control_space_bounds(self, vertical_dynamics, config):
        """Control space should match speed limits."""
        lo = vertical_dynamics.control_space.lo
        hi = vertical_dynamics.control_space.hi
        assert float(lo[0]) == pytest.approx(-config.defender.max_speed_vertical)
        assert float(hi[0]) == pytest.approx(config.defender.max_speed_vertical)

    def test_disturbance_space_bounds(self, vertical_dynamics, config):
        """Disturbance space should match speed limits."""
        lo = vertical_dynamics.disturbance_space.lo
        hi = vertical_dynamics.disturbance_space.hi
        assert float(lo[0]) == pytest.approx(-config.attacker.max_speed_vertical)
        assert float(hi[0]) == pytest.approx(config.attacker.max_speed_vertical)
