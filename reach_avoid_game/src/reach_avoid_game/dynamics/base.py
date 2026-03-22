"""Base interface for reach-avoid game dynamics.

All dynamics classes extend hj_reachability.ControlAndDisturbanceAffineDynamics
and additionally provide OptimizedDP-style pure-Python methods for online
control (opt_ctrl_numpy, opt_dstb_numpy, dynamics_numpy).
"""

import hj_reachability as hj

DynamicsBase = hj.ControlAndDisturbanceAffineDynamics
