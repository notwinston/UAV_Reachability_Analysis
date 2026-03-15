"""Abstract base class for reach-avoid game dynamics, built on hj_reachability."""

from __future__ import annotations

import hj_reachability as hj


# Re-export the base class from hj_reachability for convenience
DynamicsBase = hj.ControlAndDisturbanceAffineDynamics
