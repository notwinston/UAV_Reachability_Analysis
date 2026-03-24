# Fix Defender Wall Collision — Plan Metadata

## Task Type
Bug fix

## Summary
The defender quadcopter's HJ reach-track controller commands bang-bang velocity (±6 m/s) with no arena boundary awareness, causing the drone to fly straight into walls. Fix by adding a wall-avoidance safety layer that uses dynamic stopping-distance margins.

## Codebase Context
- **Defender controller** (`defender_node.py`): `DefenderControlLogic` implements Algorithm 1+2 from the paper. Control is bang-bang from value function gradients. No wall awareness.
- **Kinematic sim** (`kinematic_sim_node.py`): Double integrator for defender (v_dot = k*(u-v)), clamps position to arena but doesn't zero velocity on wall contact.
- **Numerical sim** (`numerical_sim.py`): Forward Euler integration, same clamping issue.
- **Value functions**: Precomputed 6D/3D/4D/2D arrays in `/workspace/data/value_functions/`. Very coarse grids (9×7×5×5×9×7 for horizontal). Do not encode wall avoidance.
- **Arena**: [0,45] × [0,25] × [0,20] meters.
- **Defender dynamics**: k_x=0.7, k_y=0.7, k_z=1.5, U_D_h=6.0 m/s, U_D_z=4.0 m/s.

## Chosen Approach
Post-processing wall-avoidance safety layer on controller output. Dynamic margin based on stopping distance for double-integrator dynamics. Active push-away when very close to wall (< 1m).

**Rationale**: Modifying the controller output is faster, less risky, and preserves HJ optimality in the arena interior. Recomputing value functions with wall SDFs would be more theoretically correct but prohibitively expensive for an immediate fix.

## Recommended --max-iterations
**4 iterations**. The fix is well-scoped: one main file change (defender_node.py), two secondary changes (kinematic_sim, numerical_sim), and testing. Each iteration should complete 1-2 phases.

## Context Budget Estimate
- **Pressure rating**: low (~15%)
- **File breakdown**: 4 files, all small (< 600 lines)
  - defender_node.py: ~538 lines
  - kinematic_sim_node.py: ~145 lines
  - numerical_sim.py: ~300 lines (estimated)
  - test_defender_logic.py: ~200 lines (estimated)
- **Estimated cost range**: $0.50 - $2.00
- **Disclaimer**: Cost estimate is approximate and depends on model pricing and iteration count.

## Unresolved Warnings
None — validation agents confirmed plan is sound.
