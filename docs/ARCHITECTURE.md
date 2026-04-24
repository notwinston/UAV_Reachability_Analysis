# Architecture

This project is a re-implementation of Bui et al., *Reach-Avoid Differential
Game with Reachability Analysis for UAVs — A Decomposition Approach* (arXiv
2512.22793). It spans three layers: an **offline** Hamilton–Jacobi (HJ) solver,
an **online** ROS2 defender controller, and a **simulation / hardware** shim.

```
                      config/game_params.yaml                (single source of truth)
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ reach_avoid_game/   (offline, pure Python, conda env: odp + HeteroCL)    │
 │                                                                          │
 │  scripts/compute_vertical.py    scripts/compute_horizontal.py            │
 │          │                               │                               │
 │          ▼                               ▼                               │
 │  solvers/vertical_solver.py     solvers/horizontal_solver.py             │
 │          │                               │                               │
 │          └───────── solvers/value_function_io.py (.npz writer) ──────────│
 └──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (data/value_functions/*.npz)
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ reach_avoid_ws/   (online, ROS2 Humble)                                  │
 │                                                                          │
 │  reach_avoid_controller   defender_node                                  │
 │      ├── ValueFunctionLoader (RegularGridInterpolator)                   │
 │      ├── Reach-Track control law (paper Alg. 1 + Alg. 2)                 │
 │      └── Wall / obstacle safety layer                                    │
 │                                                                          │
 │  attacker_controller      scripted / optimal / human-teleop policies     │
 │                                                                          │
 │  reach_avoid_sim          Gazebo Harmonic world + PX4 adapter +          │
 │                           ground-truth relay                             │
 │                                                                          │
 │  reach_avoid_viz          RViz2 panels (value slices, trajectories)      │
 │  reach_avoid_hw           Crazyswarm2 adapter + safety monitor           │
 │  reach_avoid_bringup      Top-level launch files                         │
 └──────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                   Gazebo Harmonic  ──or──  Crazyflie 2.1 via Crazyswarm2
```

## 1. Offline HJ value-function computation

The paper's key idea (§IV) is to **decompose** the intractable 9D joint game
into a 6D horizontal game and a 3D vertical game, solve each with HJ
reachability, then recombine at runtime. The offline pipeline mirrors that
decomposition.

### Vertical pipeline — `scripts/compute_vertical.py`

Produces three `.npz` files in `data/value_functions/`:

| File                     | State space                  | Meaning                                                 |
| ------------------------ | ---------------------------- | ------------------------------------------------------- |
| `V_z_inf.npz`            | 2D relative: `(z_rel, v_dz)` | Infinite-horizon value for the relative vertical game.  |
| `B_z.npz`                | 2D relative                  | Backward-reachable set (capture tube) for vertical.     |
| `phi_z.npz`              | 3D absolute: `(z_D,v_dz,z_A)`| Absolute-frame value used by the runtime controller.    |
| `phi_z_time_slices.npz`  | 3D + time                    | Time-indexed slices for animation / diagnostics.        |

### Horizontal pipeline — `scripts/compute_horizontal.py`

Produces:

| File                | State space                                   | Meaning                                                       |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------- |
| `V_h_T.npz`         | 4D relative: `(Δx,Δy,Δvx,Δvy)`                | Terminal-time value of the relative horizontal game.          |
| `V_h_T_6d.npz`      | 6D absolute: `(x_D,y_D,vx_D,vy_D,x_A,y_A)`    | 6D extension including arena walls + obstacles in avoid set.  |
| `B_h.npz`           | 4D relative                                   | Horizontal capture tube.                                      |
| `phi_A_reach.npz`   | 2D: `(x_A,y_A)`                               | Attacker backward-reach to target region.                     |
| `phi_h.npz`         | 6D absolute                                   | Horizontal reach-avoid value consumed by `defender_node`.     |

Numerical backend is SFU-MARS `optimized_dp` (HeteroCL-compiled HJ PDE
solver). All three grid presets (`dev`, `medium`, `paper`) share the same code
paths — only resolution changes.

### Supporting modules

- `solvers/grid_utils.py` — builds `optimized_dp` `Grid` objects from YAML.
- `solvers/winning_conditions.py` — capture + target + avoid set constructors.
- `solvers/control_extraction.py` — gradient → optimal defender acceleration.
- `dynamics/` — `VerticalGameDynamics`, `HorizontalGameDynamics`, relative
  and absolute variants; each exposes the Hamiltonian pieces the solver needs.

## 2. Online defender controller (`reach_avoid_controller`)

`defender_node.py` is the long-running ROS2 node that closes the loop in
simulation or on hardware.

1. **Load** all six `.npz` value functions once at startup and wrap each in a
   `scipy.interpolate.RegularGridInterpolator`.
2. **Subscribe** to ground-truth poses (Gazebo) or Vicon estimates (hardware)
   for both drones.
3. **At each control tick** (default 50 Hz):
   - Evaluate `phi_z`, `phi_h` gradients at the current joint state.
   - Apply the paper's **Reach-Track** decomposition (Alg. 1): pick horizontal
     acceleration from `phi_h`, vertical acceleration from `phi_z`, combine.
   - Apply the wall / obstacle safety layer (`_apply_wall_avoidance`) as a
     backstop against interpolation error near arena edges.
4. **Publish** a velocity command on `/defender/cmd_vel` (Gazebo) or a
   Crazyswarm2 `NamedPose` target (hardware).

Key ROS interfaces:

| Direction | Topic / service                | Msg type                     |
| --------- | ------------------------------ | ---------------------------- |
| Sub       | `/defender/ground_truth/pose`  | `geometry_msgs/PoseStamped`  |
| Sub       | `/attacker/ground_truth/pose`  | `geometry_msgs/PoseStamped`  |
| Pub       | `/defender/cmd_vel`            | `geometry_msgs/Twist`        |
| Pub       | `/defender/value_marker`       | `visualization_msgs/Marker`  |

## 3. Attacker controller (`attacker_controller`)

Three policies selectable via launch argument `attacker_policy`:

- `scripted` — waypoint path used for deterministic regression tests.
- `optimal` — HJ-optimal attacker derived from the solved value functions
  (worst case for the defender).
- `human` — teleop via keyboard / gamepad for demos.

## 4. Simulation shim (`reach_avoid_sim`)

- `worlds/reach_avoid_arena.sdf` — 45×25×20 m arena with floor, four walls,
  and one static obstacle box, matching `config/game_params.yaml`.
- PX4 SITL bridge via `px4_ros_com` (XRCE-DDS) for velocity-tracking low-level
  control; the ROS node publishes velocity setpoints, PX4 handles attitude.
- Ground-truth relay that republishes Gazebo model poses on ROS topics so the
  defender node sees the same interface as the Vicon stack on hardware.

## 5. Bringup (`reach_avoid_bringup`)

Top-level launch files — the only entry points a user should touch:

| Launch file                | What it starts                                                             |
| -------------------------- | -------------------------------------------------------------------------- |
| `simulation.launch.py`     | Gazebo world, PX4 SITL, both drones, defender + attacker, RViz2.           |
| `full_game.launch.py`      | Same as above plus data logging for post-run analysis.                     |
| `kinematic_game.launch.py` | Pure kinematic simulation (no Gazebo / PX4); fastest feedback loop.        |
| `attacker_only.launch.py`  | Attacker-in-the-loop rig for tuning `attacker_controller` in isolation.    |
| `hardware.launch.py`       | Crazyswarm2 stack + Vicon bridge + safety monitor (no Gazebo).             |

## 6. Hardware shim (`reach_avoid_hw`)

Thin adapter so `defender_node` can run unchanged on the Crazyflie stack:

- Subscribes to Crazyswarm2 pose streams, republishes on the ground-truth
  topic names used in sim.
- Translates `Twist` commands into Crazyswarm2 velocity setpoints.
- Runs an independent watchdog that lands the defender if either drone exits
  a tight sub-arena or if the control loop hangs.

## 7. Data flow summary

```
YAML ─▶ compute_vertical.py ─▶ phi_z, V_z_inf, B_z .npz ──┐
YAML ─▶ compute_horizontal.py ─▶ phi_h, V_h_T, B_h, … .npz┤
                                                          ▼
                              ValueFunctionLoader (defender_node)
                                                          │
ground-truth poses (Gazebo / Vicon) ──────────────────────┤
                                                          ▼
                              Reach-Track control law (Alg. 1)
                                                          │
                                                          ▼
                               Twist / Crazyswarm2 setpoint
```

YAML parameters are baked into the `.npz` file `params` field at write time,
so the runtime loader can cross-check that the value functions it is about to
use were solved against the currently-active config.
