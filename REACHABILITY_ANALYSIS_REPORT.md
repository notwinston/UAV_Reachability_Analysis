# Reachability Replication Analysis Report

Generated: 2026-04-20

Paper target: `Reachability.pdf`, arXiv:2512.22793v1, "Reach-Avoid Differential game with Reachability Analysis for UAVs: A decomposition approach" by Minh Bui, Simon Monckton, and Mo Chen.

## Executive Summary

The current repo still does **not** faithfully reproduce the paper behavior in simulation. The most important reason has changed since the previous report: several old sign and threshold problems have been partially fixed, but the checked-in "paper-valid" value functions are now partly **surrogate formulas**, not Hamilton-Jacobi value functions from the paper.

The highest-impact issues are:

1. `V_z_inf`, `V_h_T`, and `V_h_T_6d` are no longer true HJ maximum-distance solves. They are hand-constructed conservative values such as `abs(z_rel) + velocity_penalty` and `distance + speed_penalty`. They may be useful safety heuristics, but they do not prove the paper's invariant-set guarantees.
2. These surrogate artifacts are marked `paper_valid=True` in metadata, so the loader/controller trusts them as if they came from the paper's HJ computations.
3. The attacker "optimal" mode does not primarily optimize direct progress to the goal. It uses `phi_h` gradients as the main horizontal objective and only falls back to goal seeking when gradients are near zero or value functions/state are unavailable.
4. At the launch state, the attacker's `phi_h` gradient implies a command of approximately `(3, 3)` before filtering, while the target direction from `(5, 20)` to `(41.5, 12.5)` is positive x and negative y. This directly explains why the attacker can move away from the intended goal path.
5. The defender's HJ horizontal command at the launch state is approximately `(4.24, 4.24)`, which closes the y gap but adds arbitrary positive x motion even when both drones have the same x coordinate. This comes from coarse/surrogate `phi_h` gradients.
6. The PX4 simulation layer strongly suppresses the game commands. `simulation.launch.py` gives the PX4 adapter caps of `1.5 m/s` horizontal and `0.6 m/s` vertical, far below the game parameters `U_D^h=6`, `U_D^z=4`, `U_A^h=3`, `U_A^z=2`. The adapter can also override vertical commands with altitude hold and zero horizontal motion while below target altitude.
7. The latest recorded Gazebo run did not capture the attacker: `capture_samples=0`, `attacker_target_samples=0`, and `min_horizontal_distance=6.351266782095096`.
8. The environment cannot currently recompute real HJ value functions: `odp`, `jax`, and `heterocl` are absent. The solver tests that require `odp` are skipped, so the most important paper-fidelity tests are not actually exercising the solver.

Bottom line: the observed drone behavior is not primarily a small tuning bug. The repo is mixing paper labels with non-paper value-function construction, very coarse 6D grids, an attacker objective that is not direct goal seeking, and PX4 safety/altitude filters that substantially alter both players' commanded velocities.

## Paper Requirements

The paper decomposes a 9D two-drone game into horizontal and vertical sub-games. The defender is modeled as a double integrator and the attacker as a single integrator:

- Defender: velocity command affects acceleration, `v_dot = k * (u - v)`.
- Attacker: commanded velocity is applied directly.
- Horizontal controls are bounded by Euclidean speed: `sqrt(u_x^2 + u_y^2) <= U^h`.
- Vertical controls are bounded by scalar speed: `abs(u_z) <= U^z`.
- The paper uses a defender that is twice as fast as the attacker:
  - `U_D^h = 6 m/s`, `U_A^h = 3 m/s`
  - `U_D^z = 4 m/s`, `U_A^z = 2 m/s`
  - `k_x = k_y = 0.7`, `k_z = 1.5`

Capture is decomposed into simultaneous horizontal and vertical conditions:

- Horizontal: `sqrt((x_A - x_D)^2 + (y_A - y_D)^2) <= d_h`
- Vertical: `abs(z_A - z_D) <= d_z`
- Paper experiment values: `d_h = 3`, `d_z = 1`

For the horizontal game, the paper's reach and avoid sets are not simply "defender captures attacker." The paper defines:

- `R_h`: attacker reaches the target, or defender hits an obstacle/wall.
- `A_h`: defender captures/tracks the attacker before the attacker is in the target, or attacker hits an obstacle/wall.
- Winning regions:
  - attacker-winning: `W_A,h = {Phi_h <= 0}`
  - defender-winning: `W_D,h = {Phi_h > 0}`

For the vertical game:

- `Phi_z` is a defender-capture value function.
- Defender minimizes the vertical relative-distance value.
- Attacker maximizes it.
- Winning regions:
  - defender-winning: `W_D,z = {Phi_z <= 0}`
  - attacker-winning: `W_A,z = {Phi_z > 0}`

The key paper idea is not just solving `Phi_h` and `Phi_z`. It also computes invariant capture/tracking sets:

- Vertical maximum-distance value:
  - `V_z(x_z_rel, t) = sup_A inf_D max_tau |z_rel(tau)|`
  - `B_z = {x_z_rel | V_z,inf(x_z_rel) <= d_z}`
- Horizontal maximum-distance value:
  - `V_h(x_h, t) = sup_A inf_D max_tau l_h(x_h(tau))`
  - `l_h` includes relative distance and a large obstacle penalty for defender obstacle collisions.
  - `B_h = {x_h | V_h <= d_h}`

The paper explicitly expects the invariant sets to sit inside the original capture sets. In other words, `B_z` is more restrictive than `abs(z_rel) <= d_z`, and `B_h` is more restrictive than horizontal distance `<= d_h`. That subset property is necessary but not sufficient: it must come from the HJ max-distance game to give the paper's tracking guarantee.

The paper's experiments compute:

- `V_z,inf` over `z_rel in [-10,10]`, `v_Dz in [-4,4]` on a `240 x 100` grid.
- Horizontal `V_h,T` over `x_rel,y_rel in [-3,3]`, `v_Dx,v_Dy in [-6,6]` on a `60 x 60 x 75 x 75` grid, using finite horizon `T=2.5 s`.
- `Phi_h` on a 6D grid of size `85 x 45 x 85 x 45 x 8 x 7` for `(x_A, y_A, x_D, y_D, v_Dx, v_Dy)`, with `T=22 s`, taking about 3 hours in OptimizedDP.

## Current Code Path

### Offline Value-Function Pipeline

The offline value-function code lives in `reach_avoid_game/src/reach_avoid_game/solvers`.

Important current behavior:

- `vertical_solver.py` still calls `HJSolver` for `phi_z`, but `solve_vertical_max_distance()` no longer solves an HJ max-distance PDE. It constructs:
  - `v_z_inf = abs(z_rel) + max(0, abs(v_Dz) - speed_margin) / k_z`
- `horizontal_solver.py` still calls `HJSolver` for `phi_h`, but both horizontal max-distance artifacts are formulas:
  - `V_h_T = sqrt(x_rel^2 + y_rel^2) + speed_excess / min(k_x,k_y)`
  - `V_h_T_6d = distance + speed_excess / min(k_x,k_y) + obstacle_penalty`
- `compute_invariant_set_Bz()` and `compute_invariant_set_Bh()` now refuse to expand thresholds. This is an improvement over the previous report.
- `B_z` and `B_h` are now subset-checked against physical capture sets. This is also an improvement.
- However, the source values for those sets are not the paper's HJ maximum-distance values, so the set labels and metadata are misleading.

The current solver wrapper in `reach_avoid_game/src/reach_avoid_game/odp/solver.py` imports the real `odp.solver`. This is also different from the previous report. The current problem is not a hidden `hj_reachability` backend; it is that `odp` is not available in this environment and the most important max-distance artifacts are not solved with `HJSolver` at all.

### Defender Controller

The defender controller is in `reach_avoid_ws/src/reach_avoid_controller/reach_avoid_controller/defender_node.py`.

Important current behavior:

- Vertical Algorithm 1:
  - Checks `B_z` validity.
  - Uses `phi_z <= 0` as the defender vertical winning condition.
  - Uses `phi_z` gradients for reaching if outside `B_z`.
  - Uses `V_z_inf` gradients for near-boundary tracking.
  - Uses PID when deep inside `B_z` or when the HJ command does not close the gap.
- Horizontal Algorithm 2:
  - Requires `V_h_T_6d`.
  - Uses `phi_h > 0` as the paper horizontal defender-winning condition.
  - Uses `V_h_T_6d <= d_h` as the `B_h`/tracking gate.
  - Uses `phi_h` gradients for reaching outside `B_h`.
  - Uses PID pursuit when outside the winning region or when an HJ command does not close the gap.
- `compute_control()` returns zero command when the physical capture condition is met. Some "defender stopped moving" observations are therefore intentional terminal behavior.

The sign conventions in the current defender controller are much better than the previous report described. The remaining issue is that the gradients are coming from coarse or surrogate values, and the simulation adapter later clamps/filters the commands.

### Attacker Controller

The attacker controller is in `reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py`.

Important current behavior:

- Default launch mode is `optimal`.
- In `optimal` mode, if `phi_h`, `phi_z`, defender state, and defender velocity are available, the attacker uses:
  - `phi_h` gradient with respect to attacker x/y for horizontal command.
  - `phi_z` gradient with respect to attacker z for vertical command.
- It only falls back to direct goal seeking when:
  - value functions are unavailable,
  - defender state/velocity is unavailable,
  - horizontal `phi_h` gradient is near zero,
  - or an exception occurs.

That means the attacker is not "optimally going toward the goal" in the simple sense. It is trying to play the game implied by `phi_h`. If `phi_h` is coarse, surrogate-dependent, or not aligned with the target-reaching objective, the attacker may move away from the target even though direct goal seeking would move toward it.

### Launch And PX4 Adapter

The full simulation is launched from `reach_avoid_ws/src/reach_avoid_bringup/launch/full_game.launch.py`, which includes `simulation.launch.py`.

Important current behavior in `simulation.launch.py`:

- Default defender pose: `(5.0, 12.5, 3.0)`.
- Default attacker pose: `(5.0, 20.0, 3.0)`.
- Attacker target: `(41.5, 12.5, 10.0)`.
- The attacker has scripted waypoints available, but default mode is `optimal`.
- The PX4 adapter receives:
  - `max_speed_horizontal = 1.5`
  - `max_speed_vertical = 0.6`
  - `max_accel_horizontal = 0.6`
  - `max_accel_vertical = 0.6`
  - `target_altitude = spawn[2]`

The adapter in `reach_avoid_ws/src/reach_avoid_sim/reach_avoid_sim/px4_adapter_node.py` then:

- clamps game commands to adapter speed caps,
- smooths/rate-limits them,
- applies geofence and obstacle projections,
- applies altitude hold,
- zeros horizontal motion while below target altitude by more than `0.75 m`.

This means the ROS controller may compute a paper-scale command, but PX4 will often receive a much smaller or altered command. For the current run, that is not a minor detail; it changes the closed-loop behavior substantially.

## Value Function Audit

### Artifact Inventory

Current checked-in value functions in `data/value_functions`:

| Artifact | Shape | Min | Max | Notes |
|---|---:|---:|---:|---|
| `phi_h.npz` | `(9, 7, 5, 5, 9, 7)` | `-3.33016` | `4.05963` | `38219/99225` cells are `<= 0`, meaning attacker-winning under paper convention |
| `V_h_T_6d.npz` | `(9, 7, 5, 5, 9, 7)` | `0` | `67.3143` | only `288/99225` cells satisfy `<= d_h=3` |
| `V_h_T.npz` | `(21, 21, 11, 11)` | `0` | `16.3214` | marked `paper_valid=False`; diagnostic only |
| `B_h.npz` | `(9, 7, 5, 5, 9, 7)` | `0` | `1` | `288` cells inside, subset-valid, source `V_h_T_6d` |
| `V_z_inf.npz` | `(51, 31)` | `0` | `11.3333` | formula source, not HJ max-distance solve |
| `B_z.npz` | `(51, 31)` | `0` | `1` | `103` cells inside, subset-valid, source `V_z_inf` |
| `phi_z.npz` | `(24, 10, 24)` | `-1` | `99` | source target from `V_z_inf` |
| `phi_A_reach.npz` | `(21, 13)` | `-2.5` | `8.3134` | 2D attacker reaching artifact |

Subset checks now pass:

- `B_z`: `103` cells inside; `0` outside `abs(z_rel) <= 1`.
- `B_h`: `288` cells inside; `0` outside horizontal distance `<= 3`.

This is better than the stale report, which described threshold-expanded masks outside the physical capture set. But the key issue remains: the values used to build the masks are not the paper's max-distance HJ values.

### Formulas Versus Paper Values

The paper's vertical invariant set requires:

```text
V_z(x_z_rel, t) = sup_{u_A} inf_{u_D} max_{tau in [0,t]} |z_rel(tau)|
B_z = {x_z_rel | V_z,inf(x_z_rel) <= d_z}
```

The current code constructs:

```python
velocity_penalty = max(0, abs(v_Dz) - speed_margin) / k_z
V_z_inf = abs(z_rel) + velocity_penalty
```

The paper's horizontal invariant set requires an obstacle-aware max-distance value:

```text
V_h(x_h,t) = sup_A inf_D max_tau l_h(x_h(tau))
l_h = max(relative_horizontal_distance, obstacle_penalty)
B_h = {x_h | V_h <= d_h}
```

The current code constructs:

```python
speed_excess = max(0, sqrt(v_Dx^2 + v_Dy^2) - speed_margin)
V_h_T = distance + speed_excess / min(k_x, k_y)
V_h_T_6d = distance + speed_excess / min(k_x, k_y) + obstacle_penalty
```

These formulas are not necessarily wrong as engineering heuristics. They are much faster, safer than threshold expansion, and easy to reason about. But they should not be called paper-valid HJ value functions. They do not include adversarial trajectories, finite/infinite horizon max-over-time propagation, or the HJ Hamiltonian solve that gives the invariant guarantee.

### Grid Resolution

The paper-scale horizontal solve uses an `85 x 45 x 85 x 45 x 8 x 7` grid. The current checked-in `phi_h` uses a much smaller `9 x 7 x 5 x 5 x 9 x 7` grid.

That grid is extremely coarse over a `45 m x 25 m` arena:

- x spacing is about `5.625 m`.
- y spacing is about `4.167 m`.
- defender velocity spacing is about `3 m/s`.

At this resolution, gradient-based bang-bang controls are very sensitive to interpolation artifacts. A small local gradient can flip a full-speed direction. This is part of why the defender can get an x command when the launch x error is exactly zero, and why the attacker can get a y command away from the target.

### Metadata Problem

The loader uses metadata such as `paper_valid` and `subset_valid` to decide whether artifacts are acceptable.

Current metadata declares:

- `V_z_inf`: `paper_valid=True`
- `V_h_T_6d`: `paper_valid=True`
- `B_z`: `paper_valid=True`, `subset_valid=True`
- `B_h`: `paper_valid=True`, `subset_valid=True`
- `phi_h`: `paper_valid=True`, convention `paper_horizontal_phi_h`

This metadata is too strong. `B_z` and `B_h` are subset-valid, but their source functions are formula-based approximations. The report should distinguish:

- "safe diagnostic/surrogate value"
- "subset-valid physical capture mask"
- "paper-valid HJ max-distance value"
- "paper-valid HJ reach-avoid value"

Right now the controller cannot tell these apart.

## Behavior Diagnosis

### Attacker: Why It Does Not Optimally Go Toward The Goal

The attacker target is `(41.5, 12.5, 10.0)`. At the default launch state:

- defender: `(5.0, 12.5, 3.0)`
- attacker: `(5.0, 20.0, 3.0)`
- direct horizontal target vector from attacker is `(36.5, -7.5)`, so a direct goal-seeking attacker should move positive x and negative y.

The current `optimal` attacker does not use direct goal seeking as the primary objective. It uses `phi_h` gradient:

```python
h_grad = self._vf_loader.get_gradient("phi_h", h_state)
grad_xa = h_grad[4]
grad_ya = h_grad[5]
cmd.linear.x = -U_A_h if grad_xa >= 0 else U_A_h
cmd.linear.y = -U_A_h if grad_ya >= 0 else U_A_h
```

At the launch state, the measured `phi_h` gradient components were approximately:

```text
dPhi_h/dx_A = -0.0957
dPhi_h/dy_A = -0.0460
```

Because both are negative, the attacker's bang-bang rule commands:

```text
u_A,h ~= (3, 3)
```

That is positive x and positive y. Positive x helps reach the target, but positive y moves away from `y=12.5`. So the attacker can visibly fail to fly optimally toward the goal even while the code calls the mode "optimal."

This is not a simple sign typo in the fallback. The fallback is not active when `phi_h` gradients are nonzero. The problem is that the main online objective is the game value `phi_h`, not the attacker reaching value `phi_A_reach`, and the checked-in `phi_h` is a coarse value influenced by surrogate `B_h`.

### Defender: Why It Does Not Optimally Go Toward The Attacker

At the default launch state, the defender and attacker have the same x coordinate:

```text
defender: (5.0, 12.5, 3.0)
attacker: (5.0, 20.0, 3.0)
horizontal relative position: x_rel = 0, y_rel = -7.5
```

A simple pursuit controller should command mostly positive y and almost zero x.

The defender's HJ reaching gradient at that state produced velocity-gradient components approximately:

```text
dPhi_h/dv_Dx = 0.0568
dPhi_h/dv_Dy = 0.0186
```

The defender reaching rule maximizes `phi_h`, so it chooses full positive x and y before speed-magnitude clamping. After clamping to `U_D_h=6`, the command becomes approximately:

```text
u_D,h ~= (4.24, 4.24)
```

The positive y component closes the gap; the positive x component is arbitrary for the launch geometry and can look wrong in simulation. This is exactly the kind of artifact expected from a very coarse 6D value function and surrogate invariant target.

The controller does have guardrails:

- If `phi_h <= 0`, it treats the horizontal state as attacker-winning and uses PID pursuit.
- If the HJ command does not close the relative gap, it falls back to PID pursuit.
- If physical capture is already true, it returns zero command.

Those guardrails reduce some bad behavior, but they do not make the HJ gradient itself reliable.

### Algorithm 2 Gate Can Be Misleading

Algorithm 2 currently uses:

```text
in_B_h = V_h_T_6d(horizontal_state) <= d_h
```

Because `V_h_T_6d` is formula-based and very coarse, interpolation can produce surprising values near same-position or near-capture states. One probe at same position returned:

```text
V_h_T_6d ~= 6.3333
B_h interpolation ~= 0.3086
```

The physical capture test still correctly declares capture and returns zero command. But for near-capture states, the `B_h`/tracking gate is not a trustworthy paper invariant-set check.

### Simulation Layer Suppresses Both Players

The controller computes game commands using paper-scale values:

- defender horizontal limit: `6 m/s`
- defender vertical limit: `4 m/s`
- attacker horizontal limit from value-function params: `3 m/s`
- attacker vertical limit from value-function params: `2 m/s`

But `simulation.launch.py` configures the PX4 adapter with:

```text
max_speed_horizontal = 1.5
max_speed_vertical = 0.6
max_accel_horizontal = 0.6
max_accel_vertical = 0.6
```

The adapter also applies altitude hold:

```text
if current z is below target altitude by more than 0.75 m:
    horizontal velocity is set to zero
```

This means the game controller can say "move horizontally now," while the PX4 adapter says "climb first, no horizontal motion yet." This creates a mismatch between the model used for reachability and the actual closed-loop simulation.

### Latest Gazebo Run Evidence

The latest trajectory summary in `data/plots/gazebo_runs/full_game_trajectory_latest_summary.json` says:

```json
{
  "attacker_obstacle_samples": 0,
  "attacker_samples": 1476,
  "attacker_target_samples": 0,
  "capture_samples": 0,
  "defender_obstacle_samples": 0,
  "defender_samples": 1476,
  "min_horizontal_distance": 6.351266782095096,
  "min_vertical_distance": 0.0,
  "outside_room_samples": 0
}
```

The run did not capture the attacker and the attacker did not reach the target.

Sample paired trajectory states show the drones spending a long time near the floor/climbing before useful horizontal game motion:

| Time since recording start | Defender | Attacker | h distance | z distance |
|---:|---|---|---:|---:|
| `0.0 s` | `(36.49, 12.46, 0.14)` | `(10.01, 12.54, 0.07)` | `26.48` | `0.07` |
| `15.16 s` | `(36.63, 12.49, 0.63)` | `(10.06, 12.53, 0.14)` | `26.57` | `0.50` |
| `30.16 s` | `(36.53, 12.48, 8.81)` | `(10.02, 12.50, 8.32)` | `26.51` | `0.50` |
| `44.99 s` | `(22.80, 21.25, 9.73)` | `(14.10, 20.98, 9.73)` | `8.70` | `0.00` |
| `47.44 s` | `(20.31, 21.12, 9.74)` | `(13.96, 21.09, 9.72)` | `6.35` | `0.02` |
| `59.92 s` | `(20.48, 21.01, 9.73)` | `(3.40, 20.70, 9.76)` | `17.09` | `0.03` |

The vertical component eventually aligns, but the horizontal game never achieves the `d_h=3` capture threshold.

## Tests And Validation Gaps

### Selected Test Result

Command run:

```bash
PYTHONPATH=reach_avoid_game/src:reach_avoid_ws/src/reach_avoid_controller \
python -m pytest -q \
  reach_avoid_ws/src/reach_avoid_controller/test \
  reach_avoid_game/tests/test_value_function_io.py \
  reach_avoid_game/tests/test_winning_conditions.py
```

Result:

```text
66 passed, 1 failed
```

The failing test was:

```text
TestWallAvoidance.test_wall_avoidance_allows_capture_near_wall
```

That test expects the defender to keep moving toward an attacker near the wall, but the sample state is already physically captured under `d_h=3`, `d_z=1`. Current `compute_control()` intentionally returns zero command for captured states. This test is stale or ambiguous; it is not primary evidence for the flight behavior problem.

### Solver Tests Are Skipped

Several solver tests include:

```python
pytest.importorskip("odp")
```

In the current environment:

```text
odp: missing
jax: missing
heterocl: missing
hj_reachability: missing
```

So the tests that would most directly exercise true HJ solver behavior are skipped. This is a major validation gap.

### Missing Acceptance Tests

The repo needs tests that explicitly distinguish paper-valid HJ artifacts from surrogate artifacts. Recommended checks:

- Assert `V_z_inf` was generated by an HJ max-distance solve, not by the closed-form conservative formula.
- Assert `V_h_T` or `V_h_T_6d` records the actual HJ max-distance formulation and horizon.
- Assert `B_z` and `B_h` are not only physical subsets but are derived from HJ max-distance values.
- Assert launch-state attacker command has positive x and negative y when the intended validation mode is "goal-seeking attacker."
- Assert launch-state defender command has near-zero x for symmetric x geometry unless an obstacle/wall genuinely requires x motion.
- Assert PX4 adapter limits match the modeled speed limits in paper-validation runs, or explicitly tag the run as a slowed hardware-safe variant.
- Assert trajectory summaries include capture within the expected time for the chosen scenario.

## What Is Fixed Since The Previous Report

The previous report from 2026-04-18 is stale. The following issues have improved:

- The current `VerticalGameDynamics` uses defender `uMode="min"` and attacker `dMode="max"`, matching the paper's vertical role convention.
- Horizontal winning convention is now closer to the paper: defender-winning is treated as `phi_h > 0`.
- Threshold expansion for `B_z` and `B_h` has been removed. The code now raises if the requested threshold is empty.
- Current `B_z` and `B_h` masks are physical subsets of the corresponding capture sets.
- The controller now rejects threshold-expanded invariant masks via metadata.
- The controller uses only 6D `V_h_T_6d` for horizontal tracking, not the old 4D relative `V_h_T`.
- The local solver wrapper now imports real `odp` rather than the older `hj_reachability` compatibility layer.

These fixes are good, but they do not complete the replication because the current max-distance artifacts are not HJ solves.

## Recommended Fix Order

### 1. Separate Surrogate Artifacts From Paper Artifacts

Do this first because it prevents the controller and tests from trusting the wrong data.

- Rename or re-label formula artifacts:
  - `V_z_inf_surrogate`
  - `V_h_T_surrogate`
  - `V_h_T_6d_surrogate`
- Set `paper_valid=False` for formula-based values.
- Add metadata fields such as:
  - `construction: formula_surrogate`
  - `hj_solved: false`
  - `guarantee: physical_subset_only`
- Require HJ-generated values for paper-validation launch modes.

### 2. Restore Real HJ Max-Distance Computation

For paper replication, implement or restore:

- Vertical:
  - solve `V_z = sup_A inf_D max_tau |z_rel|`
  - compute `B_z = {V_z,inf <= d_z}`
  - verify `B_z` is nonempty and subset of `C_z`
- Horizontal:
  - solve obstacle-aware `V_h` over a state that can encode defender obstacles
  - use the paper horizon behavior: finite `T=2.5 s` if infinite horizon does not converge
  - compute `B_h = {V_h <= d_h}`

This requires an environment with OptimizedDP/`odp` and HeteroCL available. No real HJ recomputation was performed for this report because those packages are unavailable here.

### 3. Decide The Attacker Validation Mode

The paper's adversarial attacker and a "fly optimally to the goal" attacker are not always the same thing in this code.

For debugging the user-observed issue, add an explicit launch mode:

- `goal_optimal`: use `phi_A_reach` or direct shortest-time target reaching with obstacle avoidance.
- `game_optimal`: use `phi_h`/`phi_z` as the adversarial game policy.
- `scripted`: follow waypoints for repeatable system tests.

Then validate each mode separately. Do not use the label `optimal` without saying optimal for which objective.

### 4. Align Simulation Limits With The Model For Paper Runs

For a paper-replication run, PX4 adapter limits must match the model or the model must be recomputed for the lower limits.

Choose one:

- Paper-scale simulation:
  - PX4 adapter horizontal cap: `6.0` for defender, `3.0` for attacker
  - PX4 adapter vertical cap: `4.0` for defender, `2.0` for attacker
  - acceleration/rate limits high enough to approximate `k_x=0.7`, `k_y=0.7`, `k_z=1.5`
- Hardware-safe slowed simulation:
  - lower PX4 caps
  - recompute all value functions using those lower caps
  - label results as hardware-safe variant, not paper replication

Also avoid zeroing horizontal motion during the game unless the reachability model includes that takeoff/altitude phase.

### 5. Increase Grid Resolution For Meaningful Gradients

The current `9 x 7 x 5 x 5 x 9 x 7` 6D grid is useful for smoke tests, not for stable bang-bang control in a `45 m x 25 m` arena.

Minimum recommendation:

- Use the paper or near-paper grid for final `Phi_h`.
- Keep `dev` grids only for CI and structural tests.
- Add a runtime warning when loading `dev` artifacts into `full_game.launch.py`.

### 6. Add End-To-End Acceptance Criteria

A paper-replication run should not be considered successful unless:

- The value-function metadata proves the artifacts came from HJ solves.
- The controller starts in the expected winning regions for the scenario.
- The attacker mode is explicitly chosen and matches the expected objective.
- The PX4 adapter limits match the value-function parameters.
- The trajectory summary shows capture:
  - `capture_samples > 0`
  - `min_horizontal_distance <= d_h`
  - `min_vertical_distance <= d_z`
  - attacker target samples are zero before capture for defender-win scenarios

## Conclusion

The repo has moved closer to the paper in sign conventions and physical subset checks, but it still does not replicate the paper's core mathematical pipeline. The biggest current mismatch is that the invariant-set values are formula-based surrogates marked as paper-valid. That makes the defender's reach-track logic depend on data that cannot provide the paper's HJ tracking guarantee.

The attacker behavior is also explained by the code: "optimal" mode uses `phi_h` gradients, not direct target-reaching gradients, so it can command motion away from the goal. The defender behavior is explained by the same family of issues: coarse/surrogate `phi_h` gradients can produce arbitrary full-speed components, and PX4 then clamps or overrides those commands.

To make the drones behave like the paper, the next work should be: separate surrogate artifacts from HJ artifacts, recompute real HJ max-distance and reach-avoid values in an `odp` environment, make attacker objective modes explicit, and run paper-validation simulations with adapter limits that match the modeled game.
