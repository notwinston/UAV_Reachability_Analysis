---
task_type: bugfix
workflow: bug-fix
current_phase: B1
completed_phases: []
uat_fast_forward: false
session_name: fix-defender-wall-collision
decomposed: false
context_pressure: low
---

## B1: Bug Understanding

**Bug**: When running the full reach-avoid game, the defender quadcopter flies straight into the arena wall and gets stuck.

**User answers**:
- Launch mode: Both / unsure (affects both kinematic_game and full_game)
- Which drone: Defender (blue, HJ reach-track controller)
- Wall behavior: Flies straight into wall (not oscillating, not recovering)
- Root cause: Not sure — user wants us to investigate

## Codebase Exploration Findings

### Architecture
- **Defender controller**: `DefenderControlLogic` in `defender_node.py` implements Algorithm 1 (vertical reach-track) and Algorithm 2 (horizontal reach-track-avoid)
- **Kinematic sim**: `kinematic_sim_node.py` — double integrator for defender, single integrator for attacker
- **Full game**: PX4 + Gazebo via `px4_adapter_node.py` + `ground_truth_relay_node.py`

### Key dynamics
- Defender: double integrator `v_dot = k*(u - v)`, k_x=k_y=0.7, k_z=1.5
- Max speeds: U_D_h=6.0, U_D_z=4.0
- Arena: [0,45] × [0,25] × [0,20]
- Control: bang-bang (±U_D_h or ±U_D_z) from value function gradient

### Root causes identified
1. **No wall avoidance in HJ controller**: The bang-bang control commands ±6 m/s with no boundary awareness. When gradient points toward wall, defender accelerates at max speed into wall.
2. **Kinematic sim clamps position but not velocity**: Drone gets "stuck" pushing against wall at max velocity.
3. **Double integrator stopping distance is substantial**: At 6 m/s with k=0.7, stopping takes ~1.4s and ~4.3m.
4. **Coarse value function grid**: 9×7×5×5×9×7 = very coarse, gradients near edges unreliable.

## Open Questions for User
- Pending B1 elicitation
