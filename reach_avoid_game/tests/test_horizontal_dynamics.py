"""Tests for horizontal sub-game dynamics."""

import jax.numpy as jnp
import pytest

from reach_avoid_game.config import GameConfig
from reach_avoid_game.dynamics.horizontal_game import HorizontalGameDynamics


@pytest.fixture
def config():
    return GameConfig.from_yaml("/workspace/config/game_params.yaml")


@pytest.fixture
def dynamics(config):
    return HorizontalGameDynamics(config)


class TestHorizontalDynamicsEquations:
    def test_state_derivatives_match_eq19(self, dynamics, config):
        """Verify 6D state derivatives match Eq. 19 from paper."""
        k_x = config.defender.k_x
        k_y = config.defender.k_y

        # State: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
        state = jnp.array([10.0, 5.0, 2.0, -1.0, 20.0, 15.0])
        control = jnp.array([3.0, 1.5])      # [u_x, u_y]
        disturbance = jnp.array([-1.0, 2.0])  # [d_x, d_y]

        deriv = dynamics(state, control, disturbance, 0.0)

        assert float(deriv[0]) == pytest.approx(2.0)                # x_D_dot = v_D_x
        assert float(deriv[1]) == pytest.approx(-1.0)               # y_D_dot = v_D_y
        assert float(deriv[2]) == pytest.approx(k_x * (3.0 - 2.0))  # v_D_x_dot
        assert float(deriv[3]) == pytest.approx(k_y * (1.5 - (-1.0)))  # v_D_y_dot
        assert float(deriv[4]) == pytest.approx(-1.0)               # x_A_dot = d_x
        assert float(deriv[5]) == pytest.approx(2.0)                # y_A_dot = d_y

    def test_open_loop_dynamics(self, dynamics, config):
        """Open-loop dynamics should show only drift from velocity."""
        k_x = config.defender.k_x
        k_y = config.defender.k_y
        state = jnp.array([10.0, 5.0, 3.0, -2.0, 20.0, 15.0])

        f = dynamics.open_loop_dynamics(state, 0.0)

        assert float(f[0]) == pytest.approx(3.0)           # x_D_dot = v_D_x
        assert float(f[1]) == pytest.approx(-2.0)          # y_D_dot = v_D_y
        assert float(f[2]) == pytest.approx(-k_x * 3.0)    # -k_x * v_D_x
        assert float(f[3]) == pytest.approx(-k_y * (-2.0))  # -k_y * v_D_y
        assert float(f[4]) == pytest.approx(0.0)
        assert float(f[5]) == pytest.approx(0.0)

    def test_zero_state_zero_derivatives(self, dynamics):
        """With zero velocity, zero control, zero disturbance, derivatives are zero."""
        state = jnp.array([5.0, 5.0, 0.0, 0.0, 10.0, 10.0])
        control = jnp.array([0.0, 0.0])
        disturbance = jnp.array([0.0, 0.0])

        deriv = dynamics(state, control, disturbance, 0.0)

        for i in range(6):
            assert float(deriv[i]) == pytest.approx(0.0, abs=1e-10)


class TestOptimalControlDirection:
    def test_defender_maximizes_value(self, dynamics, config):
        """Defender's optimal control should maximize the Hamiltonian.

        When dV/dv_D_x > 0, defender should choose u_x = U_D_h (max).
        """
        state = jnp.array([5.0, 5.0, 0.0, 0.0, 20.0, 15.0])
        # Gradient: positive in v_D_x direction
        grad_value = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        ctrl = dynamics.optimal_control(state, 0.0, grad_value)

        # Defender maximizes -> u_x should be positive (max speed)
        assert float(ctrl[0]) == pytest.approx(config.defender.max_speed_horizontal)

    def test_attacker_minimizes_value(self, dynamics, config):
        """Attacker's optimal disturbance should minimize the Hamiltonian.

        When dV/dx_A > 0, attacker should choose d_x = -U_A_h (min).
        """
        state = jnp.array([5.0, 5.0, 0.0, 0.0, 20.0, 15.0])
        # Gradient: positive in x_A direction
        grad_value = jnp.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

        dstb = dynamics.optimal_disturbance(state, 0.0, grad_value)

        # Attacker minimizes -> d_x should be negative
        assert float(dstb[0]) == pytest.approx(-config.attacker.max_speed_horizontal)


class TestSpeedConstraints:
    def test_control_space_bounds(self, dynamics, config):
        """Control space should match defender horizontal speed limit."""
        lo = dynamics.control_space.lo
        hi = dynamics.control_space.hi
        u_d_h = config.defender.max_speed_horizontal

        assert float(lo[0]) == pytest.approx(-u_d_h)
        assert float(lo[1]) == pytest.approx(-u_d_h)
        assert float(hi[0]) == pytest.approx(u_d_h)
        assert float(hi[1]) == pytest.approx(u_d_h)

    def test_disturbance_space_bounds(self, dynamics, config):
        """Disturbance space should match attacker horizontal speed limit."""
        lo = dynamics.disturbance_space.lo
        hi = dynamics.disturbance_space.hi
        u_a_h = config.attacker.max_speed_horizontal

        assert float(lo[0]) == pytest.approx(-u_a_h)
        assert float(lo[1]) == pytest.approx(-u_a_h)
        assert float(hi[0]) == pytest.approx(u_a_h)
        assert float(hi[1]) == pytest.approx(u_a_h)

    def test_control_jacobian_shape(self, dynamics):
        """Control Jacobian should be 6x2."""
        state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        jac = dynamics.control_jacobian(state, 0.0)
        assert jac.shape == (6, 2)

    def test_disturbance_jacobian_shape(self, dynamics):
        """Disturbance Jacobian should be 6x2."""
        state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        jac = dynamics.disturbance_jacobian(state, 0.0)
        assert jac.shape == (6, 2)
