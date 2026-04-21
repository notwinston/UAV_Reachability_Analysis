"""Tests for ground-truth relay model-pose selection helpers."""

import unittest

from reach_avoid_sim.ground_truth_relay_node import _matches_exact_model_pose


class TestMatchesExactModelPose(unittest.TestCase):
    def test_matches_plain_model_name(self):
        self.assertTrue(_matches_exact_model_pose("x500_1", "x500_1"))

    def test_matches_scoped_world_model_name(self):
        self.assertTrue(
            _matches_exact_model_pose("reach_avoid_arena/x500_1", "x500_1")
        )

    def test_rejects_child_link_scope(self):
        self.assertFalse(
            _matches_exact_model_pose("x500_1::base_link", "x500_1")
        )

    def test_rejects_world_scoped_child_link(self):
        self.assertFalse(
            _matches_exact_model_pose(
                "reach_avoid_arena/x500_1::camera_link", "x500_1"
            )
        )


if __name__ == "__main__":
    unittest.main()
