"""Tests for safety monitor check_safety pure function.

Tests geofence, speed, altitude, inter-drone distance checks without
requiring ROS2 or rclpy.
"""

import sys
import os
import math

import pytest

# Add parent package to path for import
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), '..')
)
from reach_avoid_hw.safety_monitor_node import (
    SafetyConfig,
    SafetyViolation,
    check_safety,
)


@pytest.fixture
def config():
    """Default safety config for tests."""
    return SafetyConfig(
        room_x_min=0.0,
        room_x_max=10.0,
        room_y_min=0.0,
        room_y_max=10.0,
        room_z_min=0.0,
        room_z_max=5.0,
        geofence_margin=0.5,
        altitude_min=0.3,
        altitude_ceiling_margin=0.5,
        max_speed_defender=6.0,
        max_speed_attacker=3.0,
        speed_tolerance=1.1,
        min_inter_drone_distance=0.5,
        state_timeout=1.0,
    )


# ---- Geofence tests ----

class TestGeofence:
    def test_position_inside_bounds_is_safe(self, config):
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        geofence_violations = [v for v in violations if v.check_name == "geofence"]
        assert len(geofence_violations) == 0

    def test_defender_outside_x_min_triggers_violation(self, config):
        # x < room_x_min + margin = 0.0 + 0.5 = 0.5
        violations = check_safety(
            defender_pos=(0.3, 5.0, 2.0),
            attacker_pos=(5.0, 5.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        geofence_violations = [v for v in violations if v.check_name == "geofence"]
        assert len(geofence_violations) >= 1
        assert "defender" in geofence_violations[0].message

    def test_defender_outside_x_max_triggers_violation(self, config):
        # x > room_x_max - margin = 10.0 - 0.5 = 9.5
        violations = check_safety(
            defender_pos=(9.8, 5.0, 2.0),
            attacker_pos=(5.0, 5.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        geofence_violations = [v for v in violations if v.check_name == "geofence"]
        assert len(geofence_violations) >= 1

    def test_attacker_outside_y_min_triggers_violation(self, config):
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(5.0, 0.2, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        geofence_violations = [v for v in violations if v.check_name == "geofence"]
        assert len(geofence_violations) >= 1
        assert "attacker" in geofence_violations[0].message

    def test_attacker_outside_y_max_triggers_violation(self, config):
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(5.0, 9.8, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        geofence_violations = [v for v in violations if v.check_name == "geofence"]
        assert len(geofence_violations) >= 1

    def test_position_exactly_at_boundary_is_safe(self, config):
        """Position exactly at geofence boundary (margin edge) is safe."""
        # x = 0.5 is exactly at room_x_min + margin, should NOT trigger
        violations = check_safety(
            defender_pos=(0.5, 5.0, 2.0),
            attacker_pos=(5.0, 5.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        geofence_violations = [v for v in violations if v.check_name == "geofence"]
        assert len(geofence_violations) == 0

    def test_position_exactly_at_max_boundary_is_safe(self, config):
        """Position exactly at max geofence boundary is safe."""
        # x = 9.5 is exactly at room_x_max - margin, should NOT trigger
        violations = check_safety(
            defender_pos=(9.5, 5.0, 2.0),
            attacker_pos=(5.0, 5.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        geofence_violations = [v for v in violations if v.check_name == "geofence"]
        assert len(geofence_violations) == 0


# ---- Speed tests ----

class TestSpeed:
    def test_velocity_within_limits_is_safe(self, config):
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(1.0, 1.0, 1.0),
            attacker_vel=(0.5, 0.5, 0.5),
            config=config,
        )
        speed_violations = [v for v in violations if v.check_name == "speed"]
        assert len(speed_violations) == 0

    def test_defender_exceeding_speed_triggers_violation(self, config):
        # max_speed_defender=6.0, tolerance=1.1, limit=6.6
        # speed = sqrt(5^2 + 5^2 + 0) = 7.07 > 6.6
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(5.0, 5.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        speed_violations = [v for v in violations if v.check_name == "speed"]
        assert len(speed_violations) >= 1
        assert "defender" in speed_violations[0].message

    def test_attacker_exceeding_speed_triggers_violation(self, config):
        # max_speed_attacker=3.0, tolerance=1.1, limit=3.3
        # speed = sqrt(2.5^2 + 2.5^2) = 3.54 > 3.3
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(2.5, 2.5, 0.0),
            config=config,
        )
        speed_violations = [v for v in violations if v.check_name == "speed"]
        assert len(speed_violations) >= 1
        assert "attacker" in speed_violations[0].message

    def test_speed_exactly_at_limit_is_safe(self, config):
        # max_speed_defender=6.0, tolerance=1.1, limit=6.6
        # speed = 6.6 exactly
        vel_component = 6.6 / math.sqrt(3)
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(vel_component, vel_component, vel_component),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        speed_violations = [v for v in violations if v.check_name == "speed"]
        assert len(speed_violations) == 0

    def test_zero_velocity_is_safe(self, config):
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        speed_violations = [v for v in violations if v.check_name == "speed"]
        assert len(speed_violations) == 0


# ---- Altitude tests ----

class TestAltitude:
    def test_altitude_within_range_is_safe(self, config):
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        alt_violations = [v for v in violations if v.check_name == "altitude"]
        assert len(alt_violations) == 0

    def test_too_low_altitude_triggers_violation(self, config):
        # altitude_min = 0.3
        violations = check_safety(
            defender_pos=(5.0, 5.0, 0.1),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        alt_violations = [v for v in violations if v.check_name == "altitude"]
        assert len(alt_violations) >= 1
        assert "defender" in alt_violations[0].message

    def test_too_high_altitude_triggers_violation(self, config):
        # room_z_max=5.0, ceiling_margin=0.5, max altitude=4.5
        violations = check_safety(
            defender_pos=(5.0, 5.0, 4.8),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        alt_violations = [v for v in violations if v.check_name == "altitude"]
        assert len(alt_violations) >= 1
        assert "defender" in alt_violations[0].message

    def test_attacker_too_low_triggers_violation(self, config):
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(3.0, 3.0, 0.1),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        alt_violations = [v for v in violations if v.check_name == "altitude"]
        assert len(alt_violations) >= 1
        assert "attacker" in alt_violations[0].message

    def test_altitude_exactly_at_min_is_safe(self, config):
        """Altitude exactly at altitude_min should be safe."""
        violations = check_safety(
            defender_pos=(5.0, 5.0, 0.3),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        alt_violations = [v for v in violations if v.check_name == "altitude"]
        assert len(alt_violations) == 0

    def test_altitude_exactly_at_max_is_safe(self, config):
        """Altitude exactly at ceiling limit should be safe."""
        # max altitude = 5.0 - 0.5 = 4.5
        violations = check_safety(
            defender_pos=(5.0, 5.0, 4.5),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        alt_violations = [v for v in violations if v.check_name == "altitude"]
        assert len(alt_violations) == 0


# ---- Inter-drone distance tests ----

class TestInterDroneDistance:
    def test_drones_far_apart_is_safe(self, config):
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(3.0, 3.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        dist_violations = [v for v in violations if v.check_name == "inter_drone_distance"]
        assert len(dist_violations) == 0

    def test_drones_too_close_triggers_violation(self, config):
        # min_inter_drone_distance = 0.5
        # distance = sqrt(0.1^2 + 0.1^2 + 0.1^2) = 0.173 < 0.5
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(5.1, 5.1, 2.1),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        dist_violations = [v for v in violations if v.check_name == "inter_drone_distance"]
        assert len(dist_violations) >= 1

    def test_drones_exactly_at_min_distance_is_safe(self, config):
        """Drones exactly at min_inter_drone_distance should be safe."""
        # min_inter_drone_distance = 0.5, place them 0.5m apart on x-axis
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(5.5, 5.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        dist_violations = [v for v in violations if v.check_name == "inter_drone_distance"]
        assert len(dist_violations) == 0

    def test_drones_at_same_position_triggers_violation(self, config):
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=(5.0, 5.0, 2.0),
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        dist_violations = [v for v in violations if v.check_name == "inter_drone_distance"]
        assert len(dist_violations) >= 1


# ---- Multiple simultaneous violations ----

class TestMultipleViolations:
    def test_multiple_violations_detected(self, config):
        """Multiple violations at once: geofence + speed + altitude."""
        violations = check_safety(
            defender_pos=(0.1, 0.1, 0.1),  # geofence x, y + altitude too low
            attacker_pos=(5.0, 5.0, 2.0),
            defender_vel=(10.0, 10.0, 0.0),  # speed exceeds limit
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        check_names = {v.check_name for v in violations}
        assert "geofence" in check_names
        assert "speed" in check_names
        assert "altitude" in check_names
        assert len(violations) >= 3

    def test_both_drones_violating(self, config):
        """Both defender and attacker have violations."""
        violations = check_safety(
            defender_pos=(0.1, 5.0, 2.0),  # geofence
            attacker_pos=(9.9, 5.0, 2.0),  # geofence
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        messages = [v.message for v in violations]
        has_defender = any("defender" in m for m in messages)
        has_attacker = any("attacker" in m for m in messages)
        assert has_defender
        assert has_attacker


# ---- None position/velocity handling ----

class TestNoneHandling:
    def test_none_positions_no_crash(self, config):
        """None positions should not crash, just skip checks."""
        violations = check_safety(
            defender_pos=None,
            attacker_pos=None,
            defender_vel=None,
            attacker_vel=None,
            config=config,
        )
        assert violations == []

    def test_partial_none_defender_pos(self, config):
        """Only attacker has position - attacker checks still run."""
        violations = check_safety(
            defender_pos=None,
            attacker_pos=(0.1, 5.0, 2.0),  # geofence violation
            defender_vel=None,
            attacker_vel=(0.0, 0.0, 0.0),
            config=config,
        )
        geofence_violations = [v for v in violations if v.check_name == "geofence"]
        assert len(geofence_violations) >= 1

    def test_no_inter_drone_check_with_one_missing(self, config):
        """Inter-drone distance not checked if one position is None."""
        violations = check_safety(
            defender_pos=(5.0, 5.0, 2.0),
            attacker_pos=None,
            defender_vel=(0.0, 0.0, 0.0),
            attacker_vel=None,
            config=config,
        )
        dist_violations = [v for v in violations if v.check_name == "inter_drone_distance"]
        assert len(dist_violations) == 0


# ---- SafetyConfig defaults ----

class TestSafetyConfig:
    def test_default_config(self):
        """Default config should have sensible values."""
        config = SafetyConfig()
        assert config.room_x_max > config.room_x_min
        assert config.room_y_max > config.room_y_min
        assert config.room_z_max > config.room_z_min
        assert config.geofence_margin > 0
        assert config.altitude_min > 0
        assert config.min_inter_drone_distance > 0
        assert config.speed_tolerance >= 1.0
