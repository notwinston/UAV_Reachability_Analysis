# Reach-Avoid Differential Game

CMPT 419 (Spring 2026) final project — a re-implementation of Bui et al.,
*Reach-Avoid Differential Game with Reachability Analysis for UAVs — A
Decomposition Approach* (arXiv [2512.22793](https://arxiv.org/abs/2512.22793)).
The write-up is in `CMPT419__Final_Paper.pdf`; the paper being reimplemented
is in `Reachability.pdf`.

The pipeline goes from **offline** Hamilton–Jacobi value-function computation,
through a **Gazebo Harmonic** drone-on-drone simulation, to a **Crazyflie 2.1**
hardware port via Crazyswarm2. High-level architecture, module boundaries, and
data flow are documented in `docs/ARCHITECTURE.md`.

## Repository layout

```
Final_Project/
├── reach_avoid_game/              # Offline HJ solver (pure Python, odp + HeteroCL)
│   ├── src/reach_avoid_game/
│   │   ├── dynamics/              # Double / single integrator game dynamics
│   │   ├── solvers/               # HJ PDE solvers, grid utils, value-function I/O
│   │   ├── visualization/         # Value-function and trajectory plotting
│   │   └── utils/                 # Signed-distance helpers
│   ├── scripts/                   # CLI entry points (compute_vertical, compute_horizontal, …)
│   ├── tests/                     # pytest unit tests for solvers + dynamics
│   └── pyproject.toml
├── reach_avoid_ws/                # ROS2 Humble workspace
│   └── src/
│       ├── reach_avoid_controller/    # Defender node (Reach-Track control law)
│       ├── attacker_controller/       # Scripted / optimal / teleop attacker
│       ├── reach_avoid_sim/           # Gazebo world, PX4 SITL bridge, ground-truth relay
│       ├── reach_avoid_viz/           # RViz2 panels
│       ├── reach_avoid_hw/            # Crazyswarm2 adapter + safety monitor
│       └── reach_avoid_bringup/       # Top-level launch files
├── config/
│   ├── game_params.yaml               # Shared parameters — single source of truth
│   └── generated_calibrated_game_params.yaml   # Auto-generated (do not hand-edit)
├── data/value_functions/              # Computed .npz value functions (git-ignored)
├── optimized_dp/                      # Vendored SFU-MARS HJ PDE solver
├── docs/
│   ├── ARCHITECTURE.md                # Data flow, module responsibilities, ROS interfaces
│   ├── paper_code_comparison.pdf      # Paper equations ↔ code cross-reference
│   ├── project_documentation.pdf      # Long-form project doc
│   └── notes/Prof_Guidance.md         # Raw advisor notes (context only)
├── tests/                             # Cross-workspace integration + sim-config tests
├── paper_summary.md                   # Paper in brief (4 pages)
├── REACHABILITY_ANALYSIS_REPORT.md    # Technical report on the solver output
├── CMPT419__Final_Paper.pdf           # Final write-up
└── Reachability.pdf                   # Original paper
```

## Setup

Two environments are needed: the `odp` conda env for the offline solver, and a
ROS2 Humble workspace for the online controller. Gazebo Harmonic and PX4 SITL
are standard apt installs on Ubuntu 22.04.

### Offline solver — `reach_avoid_game`

Built around SFU-MARS [`optimized_dp`](optimized_dp/) and HeteroCL. The
`.venv/` at the repo root is a placeholder; use the `odp` conda env for solves
and tests.

```bash
conda activate odp
python -m pip install "PyYAML>=6,<7" "pytest>=8.3,<9" ruff
cd reach_avoid_game
python -m pip install -e ".[dev]"

# Unit tests
PYTHONPATH=src python -m pytest -q tests
```

### ROS2 workspace — `reach_avoid_ws`

Requires ROS2 Humble, Gazebo Harmonic, `px4_ros_com` (XRCE-DDS), and — for
hardware — Crazyswarm2.

```bash
cd reach_avoid_ws
colcon build --symlink-install
source install/setup.bash
```

### Docker (optional)

`Dockerfile.sim` and `docker-compose.sim.yml` build a self-contained image
with Gazebo Harmonic + ROS2 Humble + the `odp` Python env already wired up.

```bash
docker compose -f docker-compose.sim.yml up --build
```

## Quickstart

Compute the value functions (dev preset, ~minutes on a laptop), then launch
the Gazebo game:

```bash
# 1. Offline solves — outputs .npz files under data/value_functions/
conda activate odp
python -m reach_avoid_game.scripts.compute_vertical \
    --preset dev --config config/game_params.yaml \
    --output-dir data/value_functions/
python -m reach_avoid_game.scripts.compute_horizontal \
    --preset dev --config config/game_params.yaml \
    --output-dir data/value_functions/

# 2. Launch the full game in Gazebo
source reach_avoid_ws/install/setup.bash
ros2 launch reach_avoid_bringup full_game.launch.py
```

For a faster feedback loop without Gazebo, run the kinematic-only game:

```bash
ros2 launch reach_avoid_bringup kinematic_game.launch.py
```

## Grid presets

`config/game_params.yaml` selects resolution via `grid_preset`:

| Preset   | Horizontal 6D grid                      | Target hardware     | Runtime       |
| -------- | --------------------------------------- | ------------------- | ------------- |
| `dev`    | 9 × 7 × 5 × 5 × 9 × 7  = ~14k pts       | laptop              | minutes       |
| `medium` | 15 × 10 × 6 × 6 × 15 × 10 = ~810k pts   | workstation, ≥16 GB | ~hour         |
| `paper`  | matches paper figures                   | HPC / ≥64 GB        | multi-hour    |

The same preset is used for the vertical solves and the 4D relative helpers;
see the `grid_presets:` block in the YAML for exact point counts.

## Entry points

Defined in `reach_avoid_game/pyproject.toml` / scripts directory:

| Command                                              | What it does                                             |
| ---------------------------------------------------- | -------------------------------------------------------- |
| `compute_vertical.py`                                | Solves the 2D relative + 3D absolute vertical game.      |
| `compute_horizontal.py`                              | Solves the 4D relative + 6D absolute horizontal game.    |
| `numerical_sim.py`                                   | Pure-Python rollout using the loaded value functions.    |
| `visualize.py`                                       | Cross-section plots of value functions and trajectories. |
| `generate_paper_figures.py`                          | Reproduces the figures in the write-up.                  |
| `calibrate_game.py`                                  | Fits defender gains against logged traces.               |

Top-level ROS2 launch files live under
`reach_avoid_ws/src/reach_avoid_bringup/launch/` — see
`docs/ARCHITECTURE.md` §5 for what each one starts.

## Technical stack

- HJ solver: SFU-MARS [`optimized_dp`](optimized_dp/) with HeteroCL
- ROS2 Humble on Ubuntu 22.04
- Gazebo Harmonic (verify `gz sim --version`)
- Sim bridge: `px4_ros_com` over XRCE-DDS
- Hardware: Crazyflie 2.1 via Crazyswarm2, Vicon / OptiTrack positioning

## Further reading

- `docs/ARCHITECTURE.md` — module responsibilities, ROS topics, data flow.
- `paper_summary.md` — 4-page summary of the Bui et al. paper.
- `REACHABILITY_ANALYSIS_REPORT.md` — write-up on the produced value functions.
- `docs/paper_code_comparison.pdf` — line-by-line paper ↔ code mapping.
- `docs/notes/Prof_Guidance.md` — raw advisor notes (context, unstructured).
