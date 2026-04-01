"""Unit tests for the SITL integration test parsers.

These run locally (no ROS2 needed) to verify parse_pose_xyz and
parse_twist_xyz handle real ros2 topic echo output correctly.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from test_sitl_headless import parse_pose_xyz, parse_twist_xyz


class TestParsePoseXYZ:
    def test_standard_output(self):
        text = """header:
  stamp:
    sec: 123
    nanosec: 456
  frame_id: world
pose:
  position:
    x: 1.5
    y: 6.5
    z: 0.487
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 1.0
---"""
        x, y, z = parse_pose_xyz(text)
        assert abs(x - 1.5) < 1e-6
        assert abs(y - 6.5) < 1e-6
        assert abs(z - 0.487) < 1e-6

    def test_negative_values(self):
        text = """pose:
  position:
    x: -3.14
    y: -0.001
    z: 2.718
  orientation:
    x: 0.0"""
        x, y, z = parse_pose_xyz(text)
        assert abs(x - (-3.14)) < 1e-6
        assert abs(y - (-0.001)) < 1e-6
        assert abs(z - 2.718) < 1e-6

    def test_no_position_returns_none(self):
        text = "some random garbage"
        x, y, z = parse_pose_xyz(text)
        assert x is None and y is None and z is None


class TestParseTwistXYZ:
    def test_twist_stamped(self):
        text = """header:
  stamp:
    sec: 10
    nanosec: 0
  frame_id: world
twist:
  linear:
    x: 0.5
    y: -1.2
    z: 0.03
  angular:
    x: 0.0
    y: 0.0
    z: 0.1"""
        x, y, z = parse_twist_xyz(text)
        assert abs(x - 0.5) < 1e-6
        assert abs(y - (-1.2)) < 1e-6
        assert abs(z - 0.03) < 1e-6

    def test_plain_twist(self):
        text = """linear:
  x: 3.0
  y: 4.0
  z: -0.5
angular:
  x: 0.0
  y: 0.0
  z: 0.0"""
        x, y, z = parse_twist_xyz(text)
        assert abs(x - 3.0) < 1e-6
        assert abs(y - 4.0) < 1e-6
        assert abs(z - (-0.5)) < 1e-6

    def test_no_linear_returns_none(self):
        text = "unrelated stuff"
        x, y, z = parse_twist_xyz(text)
        assert x is None and y is None and z is None


class TestParsePoseEdgeCases:
    def test_scientific_notation(self):
        text = """pose:
  position:
    x: 1.5e+01
    y: -2.3e-02
    z: 0.0e+00
  orientation:
    x: 0.0"""
        x, y, z = parse_pose_xyz(text)
        assert abs(x - 15.0) < 1e-6
        assert abs(y - (-0.023)) < 1e-6
        assert abs(z - 0.0) < 1e-6

    def test_integer_values(self):
        text = """pose:
  position:
    x: 5
    y: 10
    z: 2
  orientation:
    x: 0"""
        x, y, z = parse_pose_xyz(text)
        assert abs(x - 5.0) < 1e-6
        assert abs(y - 10.0) < 1e-6
        assert abs(z - 2.0) < 1e-6
