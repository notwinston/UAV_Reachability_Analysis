---
task_type: feature
workflow: feature-development
current_phase: F1
completed_phases: []
uat_fast_forward: false
session_name: attacker-optimal-control
decomposed: false
context_pressure: null
context_budget: null
---

# Attacker Optimal Control — Working File

## Codebase Exploration Findings

### Current State
- **Attacker controller**: `/workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py`
  - 4 modes: scripted (waypoint), keyboard (teleop), optimal (STUB), switchable
  - Scripted mode: navigates 6 hardcoded waypoints at 1.6 m/s (max_speed=2.0 * speed_fraction=0.8)
  - **`_optimal_control()` at line 203-214 is a STUB** — returns zero velocity, logs warning
  - 20Hz control loop
  - Subscribes: `/attacker/state` (PoseStamped), `/attacker/velocity` (TwistStamped)
  - Publishes: `/attacker/cmd_vel` (Twist)

### Attacker in the Paper's Formulation
- Attacker = **single integrator** (3D): velocity set instantaneously, no inertia
- Attacker = **disturbance player** in the game (minimizes value function to escape defender)
- Optimal control: **bang-bang** — `d_x = -U_A_h * sign(dV/dx_A)`, `d_y = -U_A_h * sign(dV/dy_A)`, `d_z = -U_A_z * sign(dV/dz_A)`
- Speed constraints: 3.0 m/s horizontal, 2.0 m/s vertical

### Existing Infrastructure
- **Value function loader**: `/workspace/reach_avoid_ws/src/reach_avoid_controller/reach_avoid_controller/value_function_loader.py`
  - Already provides `get_value()` and `get_gradient()` methods
  - Uses scipy RegularGridInterpolator with central finite differences
  - Currently only used by defender node
- **Precomputed value functions** in `/workspace/data/value_functions/`:
  - `phi_z.npz` (3D) — vertical game value function
  - `phi_h.npz` (6D) — horizontal game value function
  - `phi_A_reach.npz` (2D) — attacker reaching VF for T_goal
  - `V_z_inf.npz`, `V_h_T.npz`, `B_z.npz`, `B_h.npz` — tracking VFs and invariant sets
- **Dynamics with opt_dstb_numpy()**: Already compute optimal attacker controls from gradients
  - `horizontal_game.py:70-73`: `d_x = -U_A_h * sign(dV/dx_A)`, `d_y = -U_A_h * sign(dV/dy_A)`
  - `vertical_game.py:76-78`: `d_z = -U_A_z * sign(dV/dz_A)`
- **Control extraction module**: `/workspace/reach_avoid_game/src/reach_avoid_game/solvers/control_extraction.py`
  - `extract_optimal_disturbance_vertical()` at line 111-149

### Key Architecture Pattern (from Defender)
Defender uses reach-track state machine with 4 modes per axis:
1. reaching (optimal from phi gradient)
2. tracking (optimal from V_inf gradient)
3. pid_deep (PID when deep inside invariant set)
4. pid_pursuit (fallback when outside winning region)

### Topic Architecture
Attacker and defender share the same topic pattern:
- `/{role}/state` (PoseStamped), `/{role}/velocity` (TwistStamped), `/{role}/cmd_vel` (Twist)

### Game Parameters
- Room: 45x25x20m, Target: x=[38,45] y=[10,15], Obstacle: x=[15,20] y=[5,20]
- Defender speed: 6.0h/4.0v, Attacker speed: 3.0h/2.0v (2:1 ratio)
- Capture: d_h=3.0m, d_z=1.0m

## Open Questions for Discovery
1. Should attacker use game-theoretic optimal control (full adversarial) or simpler reaching control?
2. Should attacker use phi_h/phi_z (game VFs where it's the disturbance) or phi_A_reach (pure reaching)?
3. Obstacle avoidance strategy for the attacker?
4. Fallback behavior when outside VF grid bounds?
5. Should the attacker also have a reach-track-like state machine, or simpler?
