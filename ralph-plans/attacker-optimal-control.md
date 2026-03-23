You are iterating on implementing game-theoretic optimal control for the attacker drone in a reach-avoid differential game. The attacker currently follows static waypoints; you are replacing the stub `_optimal_control()` method with Hamilton-Jacobi value function-based bang-bang control.

## Cold Start

Read these files to orient yourself, then determine what has already been completed:

1. `/workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py` — The attacker ROS2 node. Check if `_optimal_control()` still contains "STUB" or "zero velocity" (lines ~203-214). If it already has `phi_h` and `phi_z` references, Phase 1 is done.
2. `/workspace/reach_avoid_ws/src/reach_avoid_controller/reach_avoid_controller/defender_node.py` — The defender node. This is your TEMPLATE for how to use ValueFunctionLoader. Mirror its patterns.
3. `/workspace/reach_avoid_ws/src/reach_avoid_controller/reach_avoid_controller/value_function_loader.py` — ValueFunctionLoader class. Import and reuse directly. Do NOT duplicate.
4. `/workspace/config/game_params.yaml` — Game parameters reference.

After reading, run these progress checks:
- `grep -c "STUB\|zero velocity" /workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py` — If 0, Phase 1 may be done.
- `grep -c "reach_avoid_controller" /workspace/reach_avoid_ws/src/attacker_controller/package.xml` — If >= 1, Phase 2 is done.
- `grep -c "value_function_dir\|target_altitude" /workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/simulation.launch.py` — If >= 2, Phase 3a is done.
- `grep -c "AttackerOptimalController" /workspace/tests/integration/test_kinematic_game.py` — If >= 1, Phase 4 is done.

Skip any phase that is already complete. Resume from the first incomplete phase.

## Requirements

### Phase 1: Implement optimal control in attacker_node.py

Modify `/workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py`:

**Step 1a — Add imports** inside the `try` block of `main()`, after the existing imports (line ~18):
```python
import numpy as np
import sys as _sys
import os as _os
_sys.path.insert(0, '/workspace/reach_avoid_ws/src/reach_avoid_controller')
from reach_avoid_controller.value_function_loader import ValueFunctionLoader
```

**Step 1b — Add parameters** in `__init__`, after the existing parameter declarations:
```python
self.declare_parameter('value_function_dir', '/workspace/data/value_functions/')
self.declare_parameter('target_altitude', 10.0)
```

**Step 1c — Add defender state storage** in `__init__`, after `self._velocity = None`:
```python
self._defender_position = None
self._defender_velocity = None
```

**Step 1d — Add defender state subscriptions** in `__init__`, after the existing attacker subscriptions:
```python
self.create_subscription(
    PoseStamped, '/defender/state',
    self._defender_state_callback, 10
)
self.create_subscription(
    TwistStamped, '/defender/velocity',
    self._defender_velocity_callback, 10
)
```

**Step 1e — Add defender state callback methods** (new methods on the class):
```python
def _defender_state_callback(self, msg: PoseStamped):
    self._defender_position = [
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
    ]

def _defender_velocity_callback(self, msg: TwistStamped):
    self._defender_velocity = [
        msg.twist.linear.x,
        msg.twist.linear.y,
        msg.twist.linear.z,
    ]
```

**Step 1f — Add VF loading** in `__init__`, after the timer creation. Only load when mode is 'optimal':
```python
# Value function loading for optimal mode
self._vf_loader = None
self._U_A_h = 3.0  # attacker max horizontal speed
self._U_A_z = 2.0  # attacker max vertical speed
self._target_center = [41.5, 12.5]  # center of target region [38,45]x[10,15]

if self._mode == 'optimal':
    vf_dir = self.get_parameter('value_function_dir').value
    try:
        loader = ValueFunctionLoader(vf_dir)
        if 'phi_h' in loader.loaded_names and 'phi_z' in loader.loaded_names:
            self._vf_loader = loader
            # Extract params from VFs
            h_params = loader.get_params('phi_h')
            z_params = loader.get_params('phi_z')
            self._U_A_h = h_params.get('U_A_h', 3.0)
            self._U_A_z = z_params.get('U_A_z', 2.0)
            self.get_logger().info(
                f'Optimal mode: loaded VFs from {vf_dir}, '
                f'U_A_h={self._U_A_h}, U_A_z={self._U_A_z}'
            )
        else:
            self.get_logger().error(
                f'Optimal mode requires phi_h and phi_z. '
                f'Loaded: {loader.loaded_names}'
            )
    except Exception as e:
        self.get_logger().error(f'Failed to load value functions: {e}')
```

**Step 1g — Replace `_optimal_control()` stub** with the full implementation:
```python
def _optimal_control(self):
    """Game-theoretic optimal control using HJ value function gradients.

    Attacker minimizes the game value function (bang-bang control):
      Horizontal: d_x = -U_A_h * sign(dPhi_h/dx_A), d_y = -U_A_h * sign(dPhi_h/dy_A)
      Vertical:   d_z = -U_A_z * sign(dPhi_z/dz_A)
    Falls back to simple goal-seeking if VFs unavailable or gradient is near-zero.
    """
    cmd = Twist()
    if self._position is None:
        return cmd

    x_A, y_A, z_A = self._position

    # If VFs not loaded or defender state not available, fall back to goal-seeking
    if (self._vf_loader is None
            or self._defender_position is None
            or self._defender_velocity is None):
        return self._goal_seeking_fallback(x_A, y_A, z_A)

    x_D, y_D, z_D = self._defender_position
    vx_D, vy_D, vz_D = self._defender_velocity

    try:
        # --- Horizontal optimal control from phi_h ---
        # 6D state: [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
        h_state = np.array([x_D, y_D, vx_D, vy_D, x_A, y_A])
        h_grad = self._vf_loader.get_gradient('phi_h', h_state)
        # Attacker minimizes: indices 4 (x_A) and 5 (y_A)
        grad_xa = h_grad[4]
        grad_ya = h_grad[5]

        if abs(grad_xa) < 1e-10 and abs(grad_ya) < 1e-10:
            # Near-zero gradient: goal-seeking fallback
            dx = self._target_center[0] - x_A
            dy = self._target_center[1] - y_A
            dist_h = math.sqrt(dx * dx + dy * dy)
            if dist_h > 0.1:
                cmd.linear.x = (dx / dist_h) * self._U_A_h
                cmd.linear.y = (dy / dist_h) * self._U_A_h
        else:
            cmd.linear.x = -self._U_A_h if grad_xa >= 0 else self._U_A_h
            cmd.linear.y = -self._U_A_h if grad_ya >= 0 else self._U_A_h

        # --- Vertical optimal control from phi_z ---
        # 3D state: [z_D, v_D_z, z_A]
        v_state = np.array([z_D, vz_D, z_A])
        v_grad = self._vf_loader.get_gradient('phi_z', v_state)
        # Attacker minimizes: index 2 (z_A)
        grad_za = v_grad[2]

        if abs(grad_za) < 1e-10:
            # Near-zero gradient: hold target altitude
            target_alt = self.get_parameter('target_altitude').value
            dz = target_alt - z_A
            cmd.linear.z = max(-self._U_A_z, min(self._U_A_z, 2.0 * dz))
        else:
            cmd.linear.z = -self._U_A_z if grad_za >= 0 else self._U_A_z

    except Exception as e:
        self.get_logger().warn(
            f'Optimal control error: {e}, using goal-seeking',
            throttle_duration_sec=5.0,
        )
        return self._goal_seeking_fallback(x_A, y_A, z_A)

    return cmd

def _goal_seeking_fallback(self, x_A, y_A, z_A):
    """Simple goal-seeking: fly toward target center at max speed."""
    cmd = Twist()
    dx = self._target_center[0] - x_A
    dy = self._target_center[1] - y_A
    dist_h = math.sqrt(dx * dx + dy * dy)
    if dist_h > 0.1:
        cmd.linear.x = (dx / dist_h) * self._U_A_h
        cmd.linear.y = (dy / dist_h) * self._U_A_h
    target_alt = self.get_parameter('target_altitude').value
    dz = target_alt - z_A
    cmd.linear.z = max(-self._U_A_z, min(self._U_A_z, 2.0 * dz))
    return cmd
```

**Verify Phase 1:**
```bash
grep -c "STUB\|zero velocity" /workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py
```
Expected: 0

```bash
grep -c "phi_h" /workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py
```
Expected: >= 2

```bash
grep -c "/defender/state" /workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py
```
Expected: >= 1

Git: `git add /workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py && git commit -m "feat: implement game-theoretic optimal control for attacker drone"`

### Phase 2: Update package dependency

Edit `/workspace/reach_avoid_ws/src/attacker_controller/package.xml`:
Add this line after the existing `<depend>std_srvs</depend>` (line 12):
```xml
<exec_depend>reach_avoid_controller</exec_depend>
```

**Verify Phase 2:**
```bash
grep -c "reach_avoid_controller" /workspace/reach_avoid_ws/src/attacker_controller/package.xml
```
Expected: >= 1

Git: `git add /workspace/reach_avoid_ws/src/attacker_controller/package.xml && git commit -m "chore: add reach_avoid_controller dependency to attacker package"`

### Phase 3: Update launch files

**3a.** Edit `/workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/simulation.launch.py`:
Find the attacker_controller Node definition (around line 233). Add these parameters to its `parameters` list:
```python
{'value_function_dir': '/workspace/data/value_functions/'},
{'target_altitude': 10.0},
```

**3b.** Edit `/workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/kinematic_game.launch.py`:
Find the attacker_controller Node definition. Add the same two parameters.

**3c.** Edit `/workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/attacker_only.launch.py`:
Find the attacker_controller Node definition. Add the same two parameters.

**Verify Phase 3:**
```bash
grep -c "value_function_dir\|target_altitude" /workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/simulation.launch.py
```
Expected: >= 2

```bash
grep -c "value_function_dir\|target_altitude" /workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/kinematic_game.launch.py
```
Expected: >= 2

```bash
grep -c "value_function_dir\|target_altitude" /workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/attacker_only.launch.py
```
Expected: >= 2

Git: `git add /workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/*.py && git commit -m "chore: add VF dir and target altitude params to attacker launch files"`

### Phase 4: Add test scenarios

Edit `/workspace/tests/integration/test_kinematic_game.py`:

**Step 4a — Add import** at the top (after existing imports around line 13):
```python
_sys_path_added = False
if '/workspace/reach_avoid_ws/src/attacker_controller' not in sys.path:
    sys.path.insert(0, '/workspace/reach_avoid_ws/src/attacker_controller')
```

**Step 4b — Add `AttackerOptimalController` class** after the `AttackerScriptedController` class (after line 88):
```python
class AttackerOptimalController:
    """Game-theoretic optimal attacker using HJ value function gradients."""

    def __init__(self, loader, U_A_h=3.0, U_A_z=2.0, target_center=None):
        self.loader = loader
        self.U_A_h = U_A_h
        self.U_A_z = U_A_z
        self.target_center = target_center or [41.5, 12.5, 10.0]

    def compute(self, position, defender_pos=None, defender_vel=None):
        """Return optimal velocity command.

        Args:
            position: [x_A, y_A, z_A]
            defender_pos: [x_D, y_D, z_D] (required for optimal, ignored by scripted)
            defender_vel: [vx_D, vy_D, vz_D] (required for optimal, ignored by scripted)
        """
        if defender_pos is None or defender_vel is None:
            # Fallback to goal-seeking
            return self._goal_seek(position)

        x_A, y_A, z_A = position
        x_D, y_D, z_D = defender_pos
        vx_D, vy_D, vz_D = defender_vel

        cmd = np.zeros(3)

        try:
            # Horizontal: 6D state [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
            h_state = np.array([x_D, y_D, vx_D, vy_D, x_A, y_A])
            h_grad = self.loader.get_gradient('phi_h', h_state)
            grad_xa, grad_ya = h_grad[4], h_grad[5]

            if abs(grad_xa) < 1e-10 and abs(grad_ya) < 1e-10:
                goal_cmd = self._goal_seek(position)
                cmd[0], cmd[1] = goal_cmd[0], goal_cmd[1]
            else:
                cmd[0] = -self.U_A_h if grad_xa >= 0 else self.U_A_h
                cmd[1] = -self.U_A_h if grad_ya >= 0 else self.U_A_h

            # Vertical: 3D state [z_D, v_D_z, z_A]
            v_state = np.array([z_D, vz_D, z_A])
            v_grad = self.loader.get_gradient('phi_z', v_state)
            grad_za = v_grad[2]

            if abs(grad_za) < 1e-10:
                dz = self.target_center[2] - z_A
                cmd[2] = np.clip(2.0 * dz, -self.U_A_z, self.U_A_z)
            else:
                cmd[2] = -self.U_A_z if grad_za >= 0 else self.U_A_z

        except Exception:
            return self._goal_seek(position)

        return cmd

    def _goal_seek(self, position):
        diff = np.array(self.target_center) - np.array(position)
        dist_h = np.sqrt(diff[0]**2 + diff[1]**2)
        cmd = np.zeros(3)
        if dist_h > 0.1:
            cmd[0] = (diff[0] / dist_h) * self.U_A_h
            cmd[1] = (diff[1] / dist_h) * self.U_A_h
        cmd[2] = np.clip(2.0 * diff[2], -self.U_A_z, self.U_A_z)
        return cmd
```

**Step 4c — Update `AttackerScriptedController.compute()`** to accept optional kwargs for compatibility:
Change its signature from `def compute(self, position):` to:
```python
def compute(self, position, defender_pos=None, defender_vel=None):
```
The body stays the same (it ignores defender_pos/defender_vel).

**Step 4d — Update `run_game()`** to support both controller types. Change line 90 signature to:
```python
def run_game(scenario_name, d_start, a_start, waypoints=None, attacker_ctrl=None, max_time=30.0, dt=0.02):
```
Replace line 102 with:
```python
if attacker_ctrl is not None:
    attacker = attacker_ctrl
else:
    attacker = AttackerScriptedController(waypoints or [[41.5, 12.5, 10.0]], max_speed=2.0, speed_fraction=0.8)
```
Update the attacker control call (line 117) to pass defender state:
```python
a_cmd = attacker.compute(sim.a_pos, defender_pos=sim.d_pos, defender_vel=sim.d_vel)
```

**Step 4e — Add optimal attacker scenario** in `main()`, after the existing 4 scenarios:
```python
# Scenario 5: Optimal attacker vs optimal defender
opt_attacker = AttackerOptimalController(loader)
ok, t = run_game(
    "Optimal attacker vs optimal defender",
    d_start=[5.0, 12.5, 3.0],
    a_start=[5.0, 20.0, 3.0],
    attacker_ctrl=opt_attacker,
)
results.append(("Optimal attacker", ok or True, t))  # ok or True: capture is not expected with optimal attacker
```
Note: `ok or True` because with an optimal attacker, capture may not happen (attacker may reach target). The test validates that the simulation runs without errors, not that capture occurs.

Also need to move `loader = ValueFunctionLoader(...)` outside `run_game` so it can be reused. Move it to the start of `main()`:
```python
loader = ValueFunctionLoader('/workspace/data/value_functions/')
```
And pass it to `run_game` as well. Update `run_game` to accept an optional `loader` parameter, or use the one from the `attacker_ctrl` if provided. Simplest: construct `DefenderControlLogic(loader)` at the start of `run_game` only when `loader` is not already in the `attacker_ctrl`.

Actually, the simplest approach: move the loader construction outside and pass to `run_game`:
```python
def run_game(scenario_name, d_start, a_start, loader, waypoints=None, attacker_ctrl=None, max_time=30.0, dt=0.02):
    ...
    defender = DefenderControlLogic(loader)
    ...
```
Then update ALL existing `run_game` calls to pass `loader` as the 4th positional argument.

**Verify Phase 4:**
```bash
grep -c "AttackerOptimalController" /workspace/tests/integration/test_kinematic_game.py
```
Expected: >= 2 (class definition + usage)

```bash
python /workspace/tests/integration/test_kinematic_game.py
```
Expected: exits with code 0, no Python exceptions. All original scenarios still pass. Optimal attacker scenario runs to completion.

Git: `git add /workspace/tests/integration/test_kinematic_game.py && git commit -m "test: add optimal attacker controller and game-theoretic test scenario"`

## Rules

- Do NOT modify any file outside the 6 files listed above.
- Do NOT modify the defender controller (`defender_node.py`) or value function loader (`value_function_loader.py`).
- Do NOT create new Python packages or ROS2 packages.
- Do NOT change existing scripted/keyboard/switchable modes — only implement the optimal mode.
- Reuse ValueFunctionLoader by import — do NOT copy or duplicate it.
- The attacker's bang-bang formula uses NEGATIVE sign (attacker minimizes): `d = -U_A * sign(dV/dx_A)`.
- Make targeted edits to files — do NOT rewrite entire files from scratch.
- Do NOT delete existing tests or test scenarios to make the test suite pass.
- Do NOT undo working fixes from previous phases.
- If you have edited the same file more than 4 times in a single phase, stop and reassess your approach.
- Git: commit after completing each phase. Do NOT push to remote.
- If a test fails, read the full traceback before attempting a fix. Quote the error in your reasoning.
- Do NOT proceed to the next phase with failing verification checks.

## Stuck-state handling

- **ValueFunctionLoader import fails**: Check that `sys.path.insert(0, '/workspace/reach_avoid_ws/src/reach_avoid_controller')` is present BEFORE the import statement. The loader is at `/workspace/reach_avoid_ws/src/reach_avoid_controller/reach_avoid_controller/value_function_loader.py`.
- **Gradient returns all zeros**: Verify VF files exist: `ls -la /workspace/data/value_functions/phi_h.npz /workspace/data/value_functions/phi_z.npz`. Then check that the state vector has correct dimensionality (6D for phi_h, 3D for phi_z) and values are within grid bounds.
- **State dimensionality mismatch**: phi_h expects 6D `[x_D, y_D, v_D_x, v_D_y, x_A, y_A]`, phi_z expects 3D `[z_D, v_D_z, z_A]`. If you get shape errors, check the state construction.
- **test_kinematic_game.py import errors**: Ensure both paths are in sys.path: `/workspace/reach_avoid_ws/src/reach_avoid_controller` and `/workspace/reach_avoid_ws/src/attacker_controller`.
- **colcon build fails**: Check package.xml dependency is correct XML and reach_avoid_controller package exists.
- **Generic stuck-state**: If you encounter the same error 3 times in a row, try a fundamentally different approach (e.g., different import mechanism, different test structure).
- **BLOCKED**: If after 8 attempts on a single phase you cannot resolve the issue, stop and report the blocker with full error details.

## Completion Signal

The task is complete when ALL of these pass:

```bash
# 1. No stub remaining
grep -c "STUB" /workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py
# Expected: 0

# 2. phi_h and phi_z used in optimal control
grep -c "phi_h\|phi_z" /workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py
# Expected: >= 4

# 3. Defender state subscriptions exist
grep -c "/defender/state\|/defender/velocity" /workspace/reach_avoid_ws/src/attacker_controller/attacker_controller/attacker_node.py
# Expected: >= 2

# 4. Package dependency added
grep -c "reach_avoid_controller" /workspace/reach_avoid_ws/src/attacker_controller/package.xml
# Expected: >= 1

# 5. Launch files updated (check all 3)
grep -c "value_function_dir\|target_altitude" /workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/simulation.launch.py
# Expected: >= 2
grep -c "value_function_dir\|target_altitude" /workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/kinematic_game.launch.py
# Expected: >= 2
grep -c "value_function_dir\|target_altitude" /workspace/reach_avoid_ws/src/reach_avoid_bringup/launch/attacker_only.launch.py
# Expected: >= 2

# 6. Test class exists
grep -c "AttackerOptimalController" /workspace/tests/integration/test_kinematic_game.py
# Expected: >= 2

# 7. Tests pass
python /workspace/tests/integration/test_kinematic_game.py
# Expected: exit code 0

# 8. Git commits exist (at least 4 new commits)
git log --oneline -6
# Expected: at least 4 recent commits from this work
```

Do NOT output the completion promise until you have run every verification command above and confirmed each one passes. If any check fails, fix the issue and re-verify.