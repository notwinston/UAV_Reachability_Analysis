# Reach-Avoid Differential Game

Implementation of Bui et al.'s "Reach-Avoid Differential Game with Reachability Analysis for UAVs — A Decomposition Approach." This project builds a full pipeline from offline Hamilton-Jacobi value function computation through Gazebo simulation to Crazyflie hardware deployment.

## Directory Structure

```
/workspace/
├── reach_avoid_game/          # Pure Python package for offline HJ computation
│   ├── src/reach_avoid_game/
│   │   ├── dynamics/          # Game dynamics models
│   │   ├── solvers/           # HJ PDE solvers and value function I/O
│   │   ├── visualization/     # Plotting tools
│   │   └── utils/             # Signed distance functions, helpers
│   ├── scripts/               # CLI entry points for offline computation
│   └── tests/                 # Unit tests
├── reach_avoid_ws/            # ROS2 workspace
│   └── src/
│       ├── reach_avoid_controller/  # Defender game controller node
│       ├── attacker_controller/     # Attacker controller (scripted/optimal/human)
│       ├── reach_avoid_sim/         # Gazebo world, PX4 adapter, ground truth
│       ├── reach_avoid_viz/         # RViz2 visualization
│       ├── reach_avoid_hw/          # Crazyswarm2 adapter, safety monitor
│       └── reach_avoid_bringup/     # Top-level launch files
├── config/
│   └── game_params.yaml       # Shared game parameters
└── data/
    └── value_functions/       # Computed value function .npz files
```

## Setup

### Python Package (Offline Computation)
Use the `odp` conda environment for reachability solves and tests. This
environment is built around SFU-MARS OptimizedDP and HeteroCL; the `.uav`
virtualenv is not the solver/test runtime unless it also has HeteroCL.

```bash
conda activate odp
/home/ros/anaconda3/envs/odp/bin/python -m pip install "PyYAML>=6,<7" "pytest>=8.3,<9" ruff
cd reach_avoid_game
/home/ros/anaconda3/envs/odp/bin/python -m pip install -e ".[dev]"
PYTHONPATH=src /home/ros/anaconda3/envs/odp/bin/python -m pytest -q tests
```

### ROS2 Workspace
```bash
cd reach_avoid_ws
colcon build --symlink-install
source install/setup.bash
```

## Technical Stack
- HJ Solver: SFU-MARS OptimizedDP (`odp`) with HeteroCL
- ROS2: Humble
- Gazebo: Harmonic
- Sim communication: px4_ros_com (XRCE-DDS)
- Hardware interface: Crazyswarm2
- Positioning: Vicon/OptiTrack
