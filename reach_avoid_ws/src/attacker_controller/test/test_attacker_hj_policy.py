"""Tests for attacker online HJ target-reaching helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attacker_controller.attacker_node import extract_attacker_hj_reaching_command


class _FakeValueFunctionData:
    def __init__(self):
        self.values = np.zeros((5, 5), dtype=float)
        self.grid_min = np.array([0.0, 0.0], dtype=float)
        self.grid_max = np.array([4.0, 4.0], dtype=float)


class _FakeLoader:
    def __init__(self, gradient, value_fn):
        self._gradient = np.array(gradient, dtype=float)
        self._value_fn = value_fn
        self.vf_data = {"phi_A_reach": _FakeValueFunctionData()}

    def get_gradient(self, vf_name, state):
        return self._gradient

    def get_value(self, vf_name, state):
        return float(self._value_fn(np.asarray(state, dtype=float)))


def test_hj_reaching_command_uses_gradient_sign():
    loader = _FakeLoader([1.0, -1.0], lambda state: state[0] - state[1])

    cmd_x, cmd_y = extract_attacker_hj_reaching_command(
        loader, "phi_A_reach", 2.0, 2.0, 3.0,
    )

    assert cmd_x < 0.0
    assert cmd_y > 0.0
    assert math.hypot(cmd_x, cmd_y) == pytest.approx(3.0)


def test_hj_reaching_command_uses_local_value_descent_when_gradient_is_flat():
    loader = _FakeLoader(
        [0.0, 0.0],
        lambda state: -(state[0] + state[1]),
    )

    cmd_x, cmd_y = extract_attacker_hj_reaching_command(
        loader, "phi_A_reach", 2.0, 2.0, 3.0,
    )

    assert cmd_x > 0.0
    assert cmd_y > 0.0
    assert math.hypot(cmd_x, cmd_y) == pytest.approx(3.0)
