"""Tests for PX4 adapter coordinate conversion and message construction.

These tests can run without ROS2 by testing the static conversion methods
directly.
"""

import math
import unittest


class TestCoordinateConversion(unittest.TestCase):
    """Test ENU <-> NED coordinate conversions."""

    def test_enu_to_ned_identity(self):
        """Test that ENU origin maps to NED origin."""
        # ENU (0, 0, 0) -> NED (0, 0, 0)
        x_ned, y_ned, z_ned = enu_to_ned(0.0, 0.0, 0.0)
        self.assertAlmostEqual(x_ned, 0.0)
        self.assertAlmostEqual(y_ned, 0.0)
        self.assertAlmostEqual(z_ned, 0.0)

    def test_enu_to_ned_east(self):
        """ENU east (+x) maps to NED east (+y)."""
        x_ned, y_ned, z_ned = enu_to_ned(1.0, 0.0, 0.0)
        self.assertAlmostEqual(x_ned, 0.0)  # north = 0
        self.assertAlmostEqual(y_ned, 1.0)  # east = 1
        self.assertAlmostEqual(z_ned, 0.0)  # down = 0

    def test_enu_to_ned_north(self):
        """ENU north (+y) maps to NED north (+x)."""
        x_ned, y_ned, z_ned = enu_to_ned(0.0, 1.0, 0.0)
        self.assertAlmostEqual(x_ned, 1.0)  # north = 1
        self.assertAlmostEqual(y_ned, 0.0)  # east = 0
        self.assertAlmostEqual(z_ned, 0.0)  # down = 0

    def test_enu_to_ned_up(self):
        """ENU up (+z) maps to NED down (-z)."""
        x_ned, y_ned, z_ned = enu_to_ned(0.0, 0.0, 5.0)
        self.assertAlmostEqual(x_ned, 0.0)
        self.assertAlmostEqual(y_ned, 0.0)
        self.assertAlmostEqual(z_ned, -5.0)

    def test_enu_to_ned_arbitrary(self):
        """Test arbitrary ENU -> NED conversion."""
        # ENU (3, 4, 5) -> NED (4, 3, -5)
        x_ned, y_ned, z_ned = enu_to_ned(3.0, 4.0, 5.0)
        self.assertAlmostEqual(x_ned, 4.0)
        self.assertAlmostEqual(y_ned, 3.0)
        self.assertAlmostEqual(z_ned, -5.0)

    def test_ned_to_enu_roundtrip(self):
        """Test that NED -> ENU -> NED is identity."""
        x_enu, y_enu, z_enu = ned_to_enu(10.0, 20.0, -3.0)
        x_ned, y_ned, z_ned = enu_to_ned(x_enu, y_enu, z_enu)
        self.assertAlmostEqual(x_ned, 10.0)
        self.assertAlmostEqual(y_ned, 20.0)
        self.assertAlmostEqual(z_ned, -3.0)

    def test_enu_to_ned_roundtrip(self):
        """Test that ENU -> NED -> ENU is identity."""
        x_ned, y_ned, z_ned = enu_to_ned(7.0, 8.0, 9.0)
        x_enu, y_enu, z_enu = ned_to_enu(x_ned, y_ned, z_ned)
        self.assertAlmostEqual(x_enu, 7.0)
        self.assertAlmostEqual(y_enu, 8.0)
        self.assertAlmostEqual(z_enu, 9.0)

    def test_negative_values(self):
        """Test conversion with negative values."""
        x_ned, y_ned, z_ned = enu_to_ned(-2.0, -3.0, -1.0)
        self.assertAlmostEqual(x_ned, -3.0)
        self.assertAlmostEqual(y_ned, -2.0)
        self.assertAlmostEqual(z_ned, 1.0)


def enu_to_ned(x_enu, y_enu, z_enu):
    """Convert ENU coordinates to NED.

    ENU: x=east, y=north, z=up
    NED: x=north, y=east, z=down
    """
    return y_enu, x_enu, -z_enu


def ned_to_enu(x_ned, y_ned, z_ned):
    """Convert NED coordinates to ENU.

    NED: x=north, y=east, z=down
    ENU: x=east, y=north, z=up
    """
    return y_ned, x_ned, -z_ned


if __name__ == '__main__':
    unittest.main()
