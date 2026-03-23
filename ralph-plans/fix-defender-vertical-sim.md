You are iterating on a bugfix for the defender drone's vertical (Y-axis) game in a reach-avoid differential game simulation. The defender sticks to the floor because the kinematic simulation node uses single integrator dynamics for the defender, but the controller and value functions assume double integrator dynamics.

## Cold Start
Before doing anything: run `git log --oneline -5` and `git diff --name-only` to determine current state. Check whether kinematic_sim_node.py already contains double integrator logic by running:
```bash
cd /workspace && grep -c 'defender_cmd\|k_x\|k_y\|k_z' reach_avoid_ws/src/reach_avoid_sim/reach_avoid_sim/kinematic_sim_node.py
```
Then run the Phase 2 test commands. If all tests pass and changes are committed, the task is complete — output the promise and stop. Otherwise, identify which phase to resume from.

Read these files to orient yourself:
- `/workspace/reach_avoid_ws/src/reach_avoid_sim/reach_avoid_sim/kinematic_sim_node.py` — the file to fix
- `/workspace/tests/integration/test_kinematic_game.py` lines 21-58 — the KinematicSim class with correct double integrator dynamics (your reference implementation)
- `/workspace/config/game_params.yaml` — game parameters (k_x=0.7, k_y=0.7, k_z=1.5)

## Requirements

Fix kinematic_sim_node.py to implement double integrator dynamics for the defender drone while keeping single integrator for the attacker.

### Phase 1: Implement double integrator dynamics for defender

Modify `/workspace/reach_avoid_ws/src/reach_avoid_sim/reach_avoid_sim/kinematic_sim_node.py`:

1. Add defender proportional gain constants (matching test_kinematic_game.py lines 31-33):
   - k_x = 0.7
   - k_y = 0.7
   - k_z = 1.5

2. Track defender's actual dynamic velocity separately from commanded velocity:
   - Add `_defender_cmd` to store the latest cmd_vel received (initialized to [0,0,0])
   - `_defender_vel` continues to store the actual dynamic velocity (initialized to [0,0,0])

3. In `_def_cmd_cb`: store the message values into `_defender_cmd` (NOT `_defender_vel`)

4. In `_sim_step`, update order is critical — velocity FIRST, then position:
   - Step 1: Update defender velocity using double integrator:
     `_defender_vel[i] += dt * k[i] * (_defender_cmd[i] - _defender_vel[i])`
   - Step 2: Update defender position using actual velocity:
     `_defender_pos[i] += _defender_vel[i] * dt`
   - Attacker remains unchanged: single integrator (cmd_vel applied directly as velocity)

5. Publish the actual dynamic velocity `_defender_vel` (not `_defender_cmd`) on `/defender/velocity`

Reference: `/workspace/tests/integration/test_kinematic_game.py` KinematicSim.step() method (lines 35-51). Match it exactly — no PID, no extra damping, no additional filtering.

Verify:
```bash
cd /workspace && grep -c 'defender_cmd\|k_x\|k_y\|k_z' reach_avoid_ws/src/reach_avoid_sim/reach_avoid_sim/kinematic_sim_node.py
```
Should return count >= 4.

Commit:
```bash
cd /workspace && git add -A && git commit -m "fix: use double integrator dynamics for defender in kinematic sim"
```

### Phase 2: Run tests and verify

Run all three test suites:
```bash
cd /workspace && python -m pytest reach_avoid_ws/src/reach_avoid_controller/test/test_defender_logic.py -v 2>&1 | tail -20
cd /workspace && python -m pytest reach_avoid_game/tests/test_vertical_solver.py -v 2>&1 | tail -20
cd /workspace && python tests/integration/test_kinematic_game.py 2>&1 | tail -40
```

All tests must pass with zero failures. Before attempting any fix for a test failure, quote the exact error message or traceback in your reasoning.

On success, commit:
```bash
cd /workspace && git add -A && git commit -m "test: verify double integrator fix passes all test suites" --allow-empty
```

Do NOT consider the task complete after only modifying the file. The test suite results are the source of truth for completion, not the code change itself.

## Rules
- Only modify `/workspace/reach_avoid_ws/src/reach_avoid_sim/reach_avoid_sim/kinematic_sim_node.py`
- Do NOT modify any other files including test files, value functions, solver code, controller code, or config files
- Do NOT rewrite kinematic_sim_node.py from scratch — make targeted edits only
- Do NOT add complexity beyond what the reference KinematicSim.step() has
- The attacker MUST remain a single integrator (direct velocity control)
- Double integrator gains MUST be k_x=0.7, k_y=0.7, k_z=1.5
- Update order in _sim_step: (1) velocity first via v += dt*k*(cmd-v), (2) position via pos += dt*v
- Separate commanded velocity (_defender_cmd) from actual velocity (_defender_vel)
- Published /defender/velocity MUST use _defender_vel (post-double-integrator), NOT _defender_cmd
- Do NOT install or remove any Python packages

## Stuck-State Recovery
- If the same test fails after 2 consecutive fix attempts, stop and do a line-by-line comparison of your _sim_step() against KinematicSim.step() in /workspace/tests/integration/test_kinematic_game.py lines 35-58. Print both implementations before attempting another fix.
- If you find yourself reverting a change you made in the previous iteration, STOP and re-read the reference implementation before continuing.
- Do NOT edit the same lines more than 2 times.
- If unable to make progress after 3 iterations total, output <promise>BLOCKED: [describe exact error]</promise>.

## Completion
Output the following promise tag when ALL of these conditions are verified:
1. kinematic_sim_node.py implements v_dot = k*(u-v) for defender velocity
2. Attacker remains single integrator
3. python -m pytest test_defender_logic.py exits with 0 failures
4. python -m pytest test_vertical_solver.py exits with 0 failures
5. python tests/integration/test_kinematic_game.py exits with 0 failures
6. All changes are committed to git

<promise>
The file kinematic_sim_node.py has been modified to use double integrator dynamics for the defender (v_dot = k * (u - v)) while keeping single integrator for the attacker. All three test suites pass: test_defender_logic.py, test_vertical_solver.py, and test_kinematic_game.py. Changes have been committed.
</promise>