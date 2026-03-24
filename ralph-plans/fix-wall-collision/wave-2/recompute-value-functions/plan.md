# Recompute Value Functions with Walls/Obstacles — Plan

## Task Type
Bug fix (value function computation)

## Summary
The horizontal value function phi_h doesn't encode arena wall avoidance for the defender. Add wall SDFs to the avoid set in the horizontal solver and recompute all value functions with the medium grid preset (~2x resolution).

## Key Changes
1. Add `_make_wall_avoid_set()` to horizontal_solver.py — creates SDFs for arena boundaries
2. Modify `_make_obstacle_avoid_set()` to combine walls + obstacles
3. Add wall penalties to V_h_T_6d initial values
4. Recompute all 7 value functions with medium preset
5. Verify controller works with new VFs

## Recommended --max-iterations
**6 iterations**. Pipeline verification and computation may need debugging.

## Context Budget
- Pressure: moderate (~35%)
- Files: 2 solver files + 7 VF data files
- Estimated cost: $2-$6 (computation time dominates)
