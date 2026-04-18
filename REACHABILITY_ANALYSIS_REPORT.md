# Reachability Replication Analysis Report

Generated: 2026-04-18

## Executive Summary

The current repo does **not** faithfully replicate `Reachability.pdf` yet. The drone behavior problems are mainly caused by value-function semantics being inverted or patched around, not by small simulation tuning errors.

The most important issues are:

1. The project is **not actually using the local `optimized_dp` repo**. The local `optimized_dp/` directory is empty, `odp` is not installed in the current environment, and the solver wrapper delegates to `hj_reachability` (`reach_avoid_game/src/reach_avoid_game/odp/solver.py:1-15`, `:183-195`). The package dependency explicitly requires `hj-reachability==0.7.0` (`reach_avoid_game/pyproject.toml:11-18`).
2. The **vertical reach-avoid game has the player roles reversed** relative to the paper. The paper states that in the vertical game the defender minimizes the relative-distance value and the attacker maximizes it. The code sets `control_mode="max"` and `disturbance_mode="min"` for `VerticalGameDynamics` (`vertical_game.py:35-38`), and the defender controller uses the same maximizing sign (`defender_node.py:249-257`).
3. The **horizontal game is not formulated like the paper's horizontal reach-avoid game**. The code uses capture as the HJ target (`horizontal_solver.py:221-222`) and obstacles as avoid constraints (`:224-233`). In the paper's original horizontal RA game, the attacker's target/defender obstacle are the attacker's reach set, while defender capture/attacker obstacle are the avoid set. The paper's horizontal winning convention is also `W_A,h = {Phi_h <= 0}` and `W_D,h = {Phi_h > 0}`, while the code treats `phi_h <= 0` as defender-winning (`winning_conditions.py:185-188`, `defender_node.py:304-309`, `:410-412`).
4. The invariant capture sets `B_z` and `B_h` are being **artificially expanded** because the computed maximum-distance value functions do not reach the requested capture thresholds. `V_z_inf.min()` is about `3.425` while `d_z=1`, so `B_z` is thresholded at `3.596` instead of `1` (`vertical_solver.py:230-232`). `V_h_T.min()` is about `6.156` while `d_h=3`, so `B_h` is thresholded at `6.464` instead of `3` (`horizontal_solver.py:360-362`). This violates the intended invariant-set definition `B = {V <= d}`.
5. The saved `B_z` and `B_h` masks are not subsets of the physical capture sets. In the saved data, `160/293` `B_z` cells are outside `|z_rel| <= 1`, and `68/529` `B_h` cells are outside horizontal distance `<= 3`. The paper explicitly expects invariant capture sets to be subsets of the original capture sets.
6. The online controller often enters "tracking" with a zero gradient and commands full-speed motion in an arbitrary direction. At equal altitude with zero vertical relative speed, `V_z_inf(0,0) ~= 3.425`, the gradient is effectively zero, and `_optimal_tracking_vertical` returns `-U_D_z` because it treats `direction >= 0` as negative full speed (`defender_node.py:259-268`). This explains strange vertical drone behavior even when the drones are already vertically aligned.
7. The horizontal value function is extremely coarse: saved `phi_h` has shape `(9, 7, 5, 5, 9, 7)` and only `32/99225` grid cells are negative. Starting from the launch state, the controller reports `in_W_D_h=False` and falls back to PID pursuit rather than HJ reach-track control.

## Paper Requirements Versus Repo State

`Reachability.pdf` decomposes the 9D game into:

- A 6D horizontal sub-game with state `[x_D, y_D, v_Dx, v_Dy, x_A, y_A]`.
- A 3D vertical sub-game with state `[z_D, v_Dz, z_A]`.
- A 2D vertical maximum-distance value function `V_z,inf(z_rel, v_Dz)`.
- A horizontal maximum-distance value function `V_h,T` or `V_h,inf`, with obstacle-aware tracking if using Algorithm 2.
- Invariant sets:
  - `B_z = {x_rel_z | V_z,inf(x_rel_z) <= d_z}`
  - `B_h = {x_h | V_h,inf(x_h) <= d_h}` or finite-horizon `V_h,T` for the paper's horizontal experiment.

The repo has files for all these concepts, but the implementation semantics diverge:

- `V_z_inf` and `V_h_T` are computed, but their minimum values are already larger than the capture thresholds, so the code expands the thresholds.
- `phi_z` is computed from expanded `B_z`, not the requested `d_z`.
- `phi_h` is not the same object as the paper's horizontal `Phi_h`; it is closer to a defender-capture BRT with obstacle penalties, but the rest of the code still labels it as the paper value function.
- `V_h_T_6d.npz` exists, but `B_h` and the defender controller use the 4D relative `V_h_T`, so online horizontal tracking does not carry the paper's obstacle-aware invariant guarantee.

## OptimizedDP / `hj_reachability` Status

### What the repo claims

The README says the technical stack uses `hj_reachability v0.7.0`, while `reach_avoid_game/pyproject.toml` says "OptimizedDP" in comments and description but depends on `jax[cpu]` and `hj-reachability==0.7.0`.

### What the code actually does

- `reach_avoid_game/src/reach_avoid_game/odp/solver.py` imports `hj_reachability as hj` and calls `hj.solve`.
- `reach_avoid_game/src/reach_avoid_game/odp/grid.py` tries `from odp.Grid import Grid`, but falls back to a local NumPy grid if `odp` is unavailable.
- The local `optimized_dp/` directory is empty.
- In the current environment:
  - `hj_reachability`: not installed
  - `odp`: not installed
  - `jax`: not installed
  - `numpy` and `scipy`: installed

Conclusion: the repo is **not using SFU-MARS/optimized_dp** for value-function computation. It is using an "OptimizedDP-compatible" API name around an `hj_reachability` backend, and in this environment even that backend cannot currently run.

This matters because OptimizedDP and `hj_reachability` have different APIs, defaults, time conventions, and reach-avoid postprocessor expectations. A wrapper can work, but only if every sign convention and target/avoid convention is verified. That verification is currently missing.

## Value Function Findings

### Saved Value Function Inventory

From `data/value_functions`:

| File | Shape | Min | Max | Notes |
|---|---:|---:|---:|---|
| `phi_z.npz` | `(24, 10, 24)` | `-0.2148` | `96.4036` | Vertical reach-track value, computed on dev grid |
| `phi_z_time_slices.npz` | `(101, 24, 10, 24)` | `-0.5892` | `96.4036` | Time slices exist |
| `V_z_inf.npz` | `(51, 31)` | `3.4251` | `12.2580` | Minimum exceeds `d_z=1` |
| `B_z.npz` | `(51, 31)` | `0` | `1` | Uses effective threshold `3.5964` |
| `phi_h.npz` | `(9, 7, 5, 5, 9, 7)` | `-0.3078` | `48.4781` | Only `32/99225` negative cells |
| `V_h_T.npz` | `(21, 21, 11, 11)` | `6.1559` | `14.2377` | Minimum exceeds `d_h=3` |
| `B_h.npz` | `(21, 21, 11, 11)` | `0` | `1` | Uses effective threshold `6.4637` |
| `V_h_T_6d.npz` | `(9, 7, 5, 5, 9, 7)` | `16.0933` | `1852.64` | Exists but not used by controller |
| `phi_A_reach.npz` | `(21, 13)` | `-2.5` | `8.3134` | Used only for analysis/fallback, not the optimal attacker mode's main objective |

### `V_z_inf` Is Not Usable As `B_z = {V_z_inf <= d_z}`

The paper uses `B_z = {V_z,inf <= d_z}` and chooses `d_z = 1` with a nonempty invariant set. In this repo, `V_z_inf.min() = 3.4251`, so the mathematically correct `B_z = {V_z_inf <= 1}` is empty.

The code responds by changing the threshold:

```python
d_z_effective = max(d_z, v_min * 1.05)
```

That makes `B_z` nonempty, but it is no longer the set that guarantees vertical capture within `d_z=1`. This is the central reason the controller can think it is "inside B_z" while physical vertical capture is not guaranteed.

### `V_h_T` Has The Same Problem

The paper's horizontal finite-horizon invariant set is extracted with `d_h=3`. In this repo, `V_h_T.min() = 6.1559`, so the mathematically correct `B_h = {V_h_T <= 3}` is empty.

The code expands to `d_h_effective = 6.4637`, again producing a non-paper set. This makes the online controller switch into tracking/PID modes in states that are outside the actual horizontal capture cylinder.

### `B_z` And `B_h` Are Not Subsets Of The Capture Sets

The paper says `B_z` is generally a subset of the vertical capture set `C_z`, and the same intuition applies to `B_h`.

Saved data contradicts this:

- `B_z`: `293` cells are marked inside; `160` are outside `|z_rel| <= 1`.
- `B_h`: `529` cells are marked inside; `68` are outside `sqrt(x_rel^2 + y_rel^2) <= 3`.

That means the controller's "captured/tracking" state machine is sometimes using an invariant set larger than the physical capture condition.

### `phi_h` Is Too Sparse And Too Coarse For Reliable Control

The active config uses `grid_preset: dev` (`config/game_params.yaml:32`), giving the 6D horizontal grid only:

- `x_D`: 9 points across 45 m
- `y_D`: 7 points across 25 m
- `v_Dx`: 5 points
- `v_Dy`: 5 points
- `x_A`: 9 points
- `y_A`: 7 points

This is far too coarse to extract stable gradients for bang-bang control in a 45 m by 25 m arena. It also explains why almost the entire grid is classified outside `phi_h <= 0`.

The launch initial state is:

- Defender: `(5.0, 12.5, 3.0)`
- Attacker: `(5.0, 20.0, 3.0)`

At that state, `phi_h ~= 4.11`, so the defender controller treats horizontal as not winning and uses PID pursuit instead of HJ reaching.

## Dynamics And Game-Role Issues

### Vertical Role Sign Is Wrong

Paper vertical game:

- Defender minimizes relative-distance value.
- Attacker maximizes it.
- Defender vertical optimal control is an `arg min`.

Code:

- `VerticalGameDynamics` sets `control_mode="max"`, `disturbance_mode="min"`.
- `opt_ctrl_numpy`, `optCtrl_inPython`, and the online controller choose `+U` when `dV/dv_Dz * k_z >= 0`.
- The attacker controller also says the attacker minimizes `phi_z`.

This role inversion can make both players optimize the wrong Hamiltonian. Even if the value function were numerically correct, the online controls would not match Eq. 26 of the paper.

### Horizontal Role/Winning Convention Is Internally Inconsistent With The Paper

The paper's horizontal value function uses:

- `W_A,h = {Phi_h <= 0}`
- `W_D,h = {Phi_h > 0}`

The code uses:

- `in_W_D_h = phi_h <= 0`
- Defender reaching control from `phi_h` only when `phi_h <= 0`
- `get_winning_regions()` globally defines `W_D = values <= 0`

This could be valid only if `phi_h` is intentionally redefined as a defender-capture value function. But then it is not the paper's `Phi_h`, and the code comments/tests are misleading. Right now the code combines paper labels with non-paper value semantics.

### Horizontal Target/Avoid Sets Are Swapped Relative To The Original RA Game

The code constructs:

- target = capture set (`horizontal_solver.py:221-222`)
- obstacle/avoid = defender walls/obstacles plus attacker walls/obstacles (`:224-233`)

The paper's horizontal RA game instead treats the attacker's goal and defender collision as the attacker's reach/loss structure and defender capture as the avoid/capture condition. The later reach-track modification uses `B_h`, but the same caution applies: the sign conventions must be explicit and consistent.

### Maximum-Distance Solves Need Backend-Level Verification

`V_z_inf` and `V_h_T` are intended to solve:

```text
sup attacker inf defender max_tau distance(...)
```

The repo uses `TargetSetMode: maxVWithV0`, `control_mode="min"`, and `disturbance_mode="max"` for relative dynamics. That high-level choice is plausible. The problem is that the `hj_reachability` wrapper is not a proven OptimizedDP replacement here, and the outputs fail the most basic invariant-set sanity check: their minimum values exceed the capture thresholds.

Before using these in ROS, the team should validate the backend on the 2D vertical problem against the paper's Fig. 4 behavior. Specifically, `V_z_inf(0,0)` and nearby states should allow a nonempty `V_z_inf <= 1` set with the paper parameters.

## Controller Behavior Issues

### The Controller Uses Full-Speed Bang-Bang On Zero Gradients

Several control functions use `>= 0` to break ties:

- `_optimal_reaching_vertical`: `return +U_D_z if direction >= 0 else -U_D_z`
- `_optimal_tracking_vertical`: `return -U_D_z if direction >= 0 else +U_D_z`
- horizontal reaching/tracking uses the same pattern per component.

At the exact center of `B_z`, the gradient is approximately zero, so the vertical tracking controller commands `-4 m/s` instead of `0`. Sample result from the saved VFs:

```text
defender=(20,12,10), attacker=(20,12,10), velocity=0
z_mode=tracking
cmd_z=-4.0
captured=True
```

This is a direct cause of drones diving/climbing when they should hold or use a smooth tracking controller.

### The "Deep Inside B" Test Uses A Broken Value Baseline

The code checks:

```python
near_boundary = V_z_inf(state) > (d_z_eff - margin_z)
```

But with the saved values, `V_z_inf(0,0) ~= 3.425`, `d_z_eff ~= 3.596`, and `margin_z = 0.3`. So the center point is considered near the boundary:

```text
3.425 > 3.296
```

This forces optimal tracking at the most benign state. Since the gradient is zero and tie-breaking is full-speed, the controller immediately commands aggressive motion.

### Horizontal Control Falls Back To PID For Most Useful States

At launch:

```text
defender=(5,12.5,3), attacker=(5,20,3)
phi_h ~= 4.11
h_mode=pid_pursuit
```

So the live simulation is mostly not using the horizontal HJ controller. It is using a simple proportional pursuit controller, then a post-hoc wall-avoidance layer. That can look like drones chasing, overshooting, and bouncing around instead of executing reach-track-avoid.

### Attacker "Optimal" Mode Does Not Primarily Seek The Goal

`attacker_node.py` optimal mode computes gradients of `phi_h` and `phi_z` and chooses controls to minimize those values (`attacker_node.py:270-327`). It only goal-seeks when gradients are near zero or VFs are unavailable.

If `phi_h` is actually a defender-capture BRT, minimizing it may not correspond to "attacker reaches target while avoiding defender." If `phi_h` is intended as the paper's horizontal value, then the defender side is wrong. Either way, attacker and defender are not playing the same formally defined game.

## Simulation / ROS Issues

### Launch And Config Are Not Fully Unified

The main mathematical config has target `[38,45] x [10,15]` and obstacle `[15,20] x [5,20]`. The launch file correctly overrides the attacker target to `(41.5,12.5,10)` (`simulation.launch.py:290-300`), but the attacker node defaults are unrelated `(7,4,2)` (`attacker_node.py:37-46`). This is not the main bug, but it makes standalone runs misleading.

### PX4 Adapter Assumes Offboard/Arm Success

`px4_adapter_node.py` sends offboard and arm commands, then assumes success after a fixed count. It does not verify `VehicleStatus`. This can cause simulation behavior that looks like controller failure when PX4 has not actually accepted the mode/arming state. This is secondary to the value-function issues but still worth fixing once the math is corrected.

### Ground Truth Relay Uses PX4 Local Position Plus Spawn

The relay converts PX4 local NED to ENU and adds the spawn offset (`ground_truth_relay_node.py:90-108`). This is correct if PX4 local position is relative to the spawn/home point. If the PX4 topic is already world-relative in the current SITL setup, this would double-count spawn. I did not find proof of double-counting in code alone, so this is a validation item rather than a confirmed bug.

## Test Coverage Problems

The tests currently encode several wrong assumptions:

- `test_horizontal_dynamics.py` explicitly asserts that the defender maximizes and attacker minimizes in horizontal. That matches Eq. 21, but the rest of the repo then treats `phi_h <= 0` as defender-winning, contradicting the paper's horizontal winning-region statement.
- `test_winning_conditions.py` globally defines defender-winning as `phi <= 0`, which is wrong for the paper's horizontal `Phi_h`.
- `test_vertical_solver.py` allows comments like "Phi_z may be positive everywhere" and accepts coarse-grid behavior, which hides the invariant-set failure.
- Tests assert `B_z` is nonempty after the effective-threshold hack, instead of asserting the mathematically important condition `min(V_z_inf) <= d_z`.
- There is no test that `B_z` is a subset of `|z_rel| <= d_z`.
- There is no test that `B_h` is a subset of horizontal distance `<= d_h`.
- There is no test that equal-position/equal-velocity states produce near-zero defender command.
- There is no backend parity test against OptimizedDP on a small known problem.

Also, the current environment cannot run the HJ computation package because `hj_reachability`, `jax`, and `odp` are missing.

## Root Cause Chain For Bad Drone Behavior

The likely runtime chain is:

1. Offline value functions are computed using a non-OptimizedDP `hj_reachability` wrapper with unverified sign conventions.
2. `V_z_inf` and `V_h_T` fail to produce nonempty invariant sets at the paper capture thresholds.
3. The code silently expands capture thresholds to make `B_z` and `B_h` nonempty.
4. The online controller believes states outside real capture are inside invariant capture sets.
5. In those sets, it often chooses tracking mode.
6. Near zero gradient, tie-breaking produces full-speed commands instead of zero/smooth commands.
7. In horizontal states, the controller often falls back to PID because `phi_h` is almost never classified as defender-winning.
8. The final command is then clipped and modified by wall avoidance, further decoupling live behavior from the value functions.

## Recommendations

### 1. Decide And Enforce The Backend

If the requirement is to use `optimized_dp`, install and import the real package:

- Populate the `optimized_dp/` checkout or add it as a submodule.
- Ensure `from odp.Grid import Grid` succeeds.
- Ensure HJ solves call OptimizedDP's `HJSolver`, not `hj.solve`.
- Remove or clearly separate the `hj_reachability` wrapper.

If the team intentionally wants `hj_reachability`, update the project goal and re-derive every value-function convention for that backend. Do not call it OptimizedDP-compatible until parity tests pass.

### 2. Fix Vertical Game Roles

For the paper's vertical game:

- Defender should minimize.
- Attacker should maximize.
- `VerticalGameDynamics` should use `control_mode="min"`, `disturbance_mode="max"`.
- Online vertical reaching control should use the minimizing sign.
- Attacker vertical optimal control should use the maximizing sign.

Then recompute `phi_z`, `V_z_inf`, and `B_z`.

### 3. Remove Effective Capture Threshold Hacks

Do not use:

```python
d_eff = max(d, min(V) * 1.05)
```

Instead:

- If `min(V_z_inf) > d_z`, the invariant set is empty. Treat that as a failed computation/configuration, not as permission to enlarge capture.
- For debugging, save an explicit diagnostic set like `B_z_effective_debug`, but never feed it into the controller as `B_z`.
- Add tests:
  - `assert V_z_inf.min() <= d_z`
  - `assert np.all(B_z_mask <= (abs(z_rel) <= d_z))`
  - `assert V_h_T.min() <= d_h` for the intended finite-horizon set
  - `assert np.all(B_h_mask <= (horizontal_dist <= d_h))`

### 4. Rebuild Horizontal `Phi_h` With A Clear Convention

Choose one:

Option A: replicate paper `Phi_h`.

- Use the paper's target/avoid sets and winning signs.
- In controller/status, use `W_D,h = phi_h > 0`.
- Use Eq. 21 signs for defender/attacker.

Option B: use a defender-capture value function.

- Rename it to avoid confusion, e.g. `phi_h_defender_capture`.
- Document that `<=0` means defender can capture.
- Do not compare it directly to the paper's `Phi_h` or theorem without translating conventions.

Right now the code is halfway between these options.

### 5. Use Obstacle-Aware Horizontal Tracking Or Admit No Guarantee

The controller loads `V_h_T` and `B_h`, both 4D relative objects. They cannot encode defender obstacle avoidance. The optional `V_h_T_6d.npz` is not used.

To match Algorithm 2:

- Compute obstacle-aware `V_h` in the state used by tracking.
- Derive `B_h` from that value function.
- Load that same value function in the controller for tracking gradients.

Otherwise, use only vertical reach-track plus horizontal capture pursuit, as the paper suggests as a possible mitigation.

### 6. Add Zero-Gradient Deadbands

Every bang-bang extraction should have a deadband:

```python
if abs(direction) < eps:
    return 0.0  # or PID/smooth tracking command
```

For horizontal, apply this per axis or normalize the gradient direction before choosing a command. Full-speed commands on zero gradient are a major behavior bug.

### 7. Increase Resolution Only After Semantics Are Correct

Do not spend hours on high-resolution solves until the 2D vertical invariant problem passes. The fastest validation ladder should be:

1. 2D `V_z_inf` on paper parameters: verify `B_z = {V <= 1}` is nonempty and subset of `C_z`.
2. 3D `Phi_z` with `B_z`: verify reaching trajectories enter `B_z`.
3. 4D or 6D horizontal tracking: verify `B_h` subset and no obstacle violation.
4. 6D horizontal RA: verify winning sign convention on obvious states.
5. ROS/PX4 simulation.

### 8. Fix Metadata Portability

The saved `.npz` files were written with NumPy 2.x object metadata. In the current NumPy 1.x environment, loading `params` raises `ModuleNotFoundError: numpy._core`. The ROS loader catches this only in fallback mode and silently uses defaults.

Save params as JSON/YAML strings instead of pickled Python objects. This removes version-dependent behavior and makes the simulation reproducible.

## Immediate Triage Checklist

1. Stop using the current `B_z.npz` and `B_h.npz` as invariant capture sets. They are threshold-expanded.
2. Fix vertical min/max roles and recompute `V_z_inf`.
3. Verify `V_z_inf.min() <= 1.0` before computing `B_z`.
4. Add zero-gradient deadbands to all optimal control extraction.
5. Decide whether `phi_h` is the paper's `Phi_h` or a defender-capture BRT. Update signs and names accordingly.
6. Do not run full Gazebo/PX4 as a correctness test until the 2D and 3D value-function sanity checks pass.

## Verification Performed

- Read repo structure and key files under `reach_avoid_game`, `reach_avoid_ws`, `config`, `tests`, and `data/value_functions`.
- Extracted relevant text from `Reachability.pdf` using `pdfminer`.
- Inspected solver backend imports and dependencies.
- Loaded saved value function arrays and computed shapes, min/max values, negative-cell counts, and invariant-mask subset checks.
- Sampled the live `DefenderControlLogic` against saved value functions.

Could not run the offline tracker directly because `tests/sim_drone_tracker.py` hardcodes `/workspace/config/game_params.yaml`, while this workspace is `/workspaces/UAV_Reachability_Analysis`. Could not recompute value functions in the current environment because `hj_reachability`, `jax`, and `odp` are not installed.
