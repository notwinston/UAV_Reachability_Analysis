---
task_type: bugfix
summary: Fix defender vertical game sticking to floor by implementing double integrator dynamics in kinematic sim
---

## Task Type
Bug fix

## Summary
The defender drone's vertical (Y-axis) game is malfunctioning — the defender sticks to the floor instead of properly pursuing/tracking the attacker vertically.

## Root Cause
`kinematic_sim_node.py` uses single integrator dynamics for the defender (cmd_vel applied directly as velocity), but the controller and value functions assume double integrator dynamics (`v_dot = k * (u - v)`). This causes incorrect velocity state feedback, unreliable value function gradient lookups at grid boundaries, and a feedback loop driving the defender to z=0.

## Codebase Context
- The integration test (`test_kinematic_game.py`) already has the correct double integrator implementation in its `KinematicSim` class
- The attacker correctly uses single integrator (matching the paper)
- Horizontal game is more tolerant of this mismatch due to PID fallback modes

## Chosen Approach
Modify `kinematic_sim_node.py` to implement double integrator dynamics for the defender, matching the reference implementation in `test_kinematic_game.py`. Single file change, minimal risk.

## Recommended --max-iterations: 3
Single file fix with a reference implementation already in the codebase. Should complete in 1-2 iterations; 3 gives buffer for any test failures.

## Context Budget
- Files: 1 (kinematic_sim_node.py, 134 lines)
- Pressure: low (<15%)
- Estimated cost: minimal

## Unresolved Warnings
None.