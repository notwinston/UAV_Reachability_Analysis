# Execution Graph: Fix Defender Wall Collision

## Wave Structure

### Wave 1: Controller Safety Layer (run first)
**Purpose:** Immediate defensive fix — adds wall-avoidance post-processing to defender controller
**Files touched:** defender_node.py, kinematic_sim_node.py, numerical_sim.py, test_defender_logic.py
**Iterations:** 4
**Estimated time:** 10-15 min

### Wave 2: Recompute Value Functions (run after Wave 1)
**Purpose:** Root cause fix — add wall/obstacle SDFs to value function avoid set, recompute with medium preset
**Files touched:** horizontal_solver.py, data/value_functions/*.npz
**Iterations:** 6
**Estimated time:** 20-60 min (depends on VF computation time)
**Depends on:** Wave 1 must complete first (tests in Wave 2 rely on Wave 1's test infrastructure)

## Dependency Rationale
- Wave 1 provides an immediate working fix and adds test infrastructure
- Wave 2 fixes the root cause at the value function level
- Wave 2 depends on Wave 1 being committed so tests pass
- After both waves: defender avoids walls via both VF-encoded avoidance AND controller safety layer (defense in depth)

## Commands

### Wave 1
```
/ralph-loop:ralph-loop $(cat ralph-plans/fix-wall-collision/wave-1/controller-safety-layer/prompt.md) --completion-promise "$(cat ralph-plans/fix-wall-collision/wave-1/controller-safety-layer/promise.txt)" --max-iterations=4
```

### Wave 2 (run after Wave 1 completes)
```
/ralph-loop:ralph-loop $(cat ralph-plans/fix-wall-collision/wave-2/recompute-value-functions/prompt.md) --completion-promise "$(cat ralph-plans/fix-wall-collision/wave-2/recompute-value-functions/promise.txt)" --max-iterations=6
```
