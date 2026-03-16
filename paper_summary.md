# Paper Summary: Reach-Avoid Differential Game with Reachability Analysis for UAVs — A Decomposition Approach

**Authors:** Minh Bui (SFU), Simon Monckton (DRDC), Mo Chen (SFU)
**ArXiv:** 2512.22793v1 (Dec 2025)
**Funding:** Canadian Department of National Defence

---

## 1. Problem Statement

The paper addresses **reach-avoid (RA) differential games** in 3D space for UAVs (drones). In these games:

- An **attacker** tries to reach a target goal region while avoiding obstacles and the defender
- A **defender** tries to intercept/capture the attacker before it reaches the goal

The core challenge: **Hamilton-Jacobi (HJ) reachability analysis** is the gold-standard tool for solving these games, but it suffers from the **curse of dimensionality** — it scales exponentially with the number of state dimensions. Two quadrotors together have a **24D joint state space**, making direct HJ analysis impossible (currently limited to ~6-7D).

---

## 2. Key Idea / Contribution

A **dimensionality reduction framework** that decomposes the intractable 3D reach-avoid game into two solvable sub-games:

1. **Horizontal sub-game (6D)** — X-Y plane dynamics (attacker + defender)
2. **Vertical sub-game (3D)** — Z-axis dynamics (attacker + defender)

Each sub-game is solved independently using HJ reachability, then results are recombined using a novel **reach-track control algorithm** with provable capture guarantees.

### Five Core Contributions

| # | Contribution |
|---|---|
| 1 | **Dynamics simplification**: 12D quadrotor → 6D double integrator (defender) + 3D single integrator (attacker) |
| 2 | **Horizontal/Vertical decomposition**: 9D joint state → 6D horizontal + 3D vertical sub-games |
| 3 | **Reach-Track control algorithm**: combines sub-game solutions while maintaining capture guarantees |
| 4 | **Theoretical guarantee analysis**: formal conditions for when defender is guaranteed to win |
| 5 | **Gazebo validation**: physics-based drone-on-drone interception simulation |

---

## 3. Technical Details

### 3.1 Dynamics Modeling (Dimensionality Reduction Step 1)

**Full quadrotor dynamics** = 12D per drone (position, velocity, angular velocity, orientation)

**Simplified models:**
- **Defender** = **Double integrator** (6D): position + velocity in all 3 axes. Velocity commands become acceleration (can't change velocity instantaneously)
- **Attacker** = **Single integrator** (3D): position only, velocity set instantaneously. This *overestimates* the attacker's agility, making the solution more conservative/robust

```
Defender: x_dot = [v_D; k*(v_command - v_D)]   (has inertia)
Attacker: x_dot = v_command                      (no inertia)
```

**Speed constraints:**
- Horizontal: defender max speed U_D^h, attacker max speed U_A^h
- Vertical: defender max speed U_D^z, attacker max speed U_A^z
- Paper uses defender **2x faster** than attacker in both directions

### 3.2 Game Decomposition (Dimensionality Reduction Step 2)

**Capture condition** is decomposed from a sphere into a **cylinder**:
- Original: `||p_A - p_D||_2 <= d_c` (sphere with radius d_c)
- Decomposed into two simultaneous conditions:
  - **Horizontal**: `sqrt((x_A - x_D)^2 + (y_A - y_D)^2) <= d_h`
  - **Vertical**: `|z_A - z_D| <= d_z`

**Key assumption:** Target region T and obstacles depend only on horizontal (x,y) coordinates, not z. This is practical for airspace zones.

### 3.3 Solving the Sub-Games

**Horizontal game (6D):**
- State: `x^h = (x_D, y_D, v_D^x, v_D^y, x_A, y_A)`
- Solve HJ variational inequality (Eq. 10) to get value function Phi_h
- Includes obstacles in reach/avoid sets
- Winning regions: W_{D,h} (defender wins) and W_{A,h} (attacker wins)

**Vertical game (3D):**
- State: `x^z = (z_D, v_D^z, z_A)`
- Simpler — no obstacles in vertical direction
- Solve HJ variational inequality to get value function Phi_z
- Winning regions: W_{D,z} and W_{A,z}

### 3.4 Invariant Capture Sets (Section IV — Critical Concept)

Simply combining the two sub-game solutions doesn't guarantee simultaneous capture in both directions. The defender might overshoot due to inertia.

**Solution: Invariant capture sets** B_z and B_h

- **Vertical invariant set** B_z: once inside, the defender can keep vertical distance <= d_z forever using a **tracking controller**
- **Horizontal invariant set** B_h: once inside, the defender can keep horizontal distance <= d_h forever, while also avoiding obstacles

These are computed using a **maximum distance value function** (different from the game value function):
- V_z tracks the worst-case maximum relative distance over time
- B_z = {states where V_{z,∞} <= d_z}

### 3.5 Reach-Track Control Algorithms

**Algorithm 1 — Vertical Reach-Track:**
```
while joint vertical state is in defender's winning region:
    if NOT yet inside invariant set B_z:
        Apply optimal reaching control (from Phi_z) to get INTO B_z ASAP
    else:
        if deep inside B_z:
            Use any performant tracking controller (e.g., PID)
        else (near boundary):
            Apply optimal tracking control (from V_{z,∞}) to STAY in B_z
```

**Algorithm 2 — Horizontal Reach-Track-Avoid:**
```
while joint horizontal state is in defender's winning region:
    if NOT yet inside invariant set B_h:
        Apply optimal reaching control (from Phi_h) to capture attacker
    else:
        if deep inside B_h:
            Use any performant tracking controller
        else (near boundary):
            Apply optimal tracking control (from V_{h,∞}) to stay in B_h
```

### 3.6 Capture Guarantee Analysis (Section VI)

Three formal results:

| Result | Condition | Outcome |
|--------|-----------|---------|
| **Proposition 1** | T_goal(x_A^h) <= T_capture(x^z) | **Attacker wins** (reaches goal before vertical capture) |
| **Proposition 2** | x_h in W_{D,h} AND x_z^rel in B_z | **Defender guaranteed to win** |
| **Proposition 3** | x_z in W_{D,z}, T_goal > T_capture, x_h in B_h | **Defender guaranteed to win** |
| **Theorem** | x_z in W_{D,z}, T_goal > T_capture, x_h in W_{D,h} | **Defender guaranteed to win** |

Key insight: The defender wins the original 9D game if it wins both sub-games, with timing conditions on vertical capture vs. attacker reaching the goal.

---

## 4. Experiment Parameters

| Parameter | Value |
|-----------|-------|
| k_x, k_y | 0.7 (proportional velocity gain) |
| k_z | 1.5 |
| U_D^h (defender max horizontal speed) | 6 m/s |
| U_A^h (attacker max horizontal speed) | 3 m/s |
| U_D^z (defender max vertical speed) | 4 m/s |
| U_A^z (attacker max vertical speed) | 2 m/s |
| d_z (vertical capture distance) | 1 meter |
| d_h (horizontal capture distance) | 3 meters |
| Vertical grid | 240 x 100, domain z_rel in [-10, 10]m, v_D^z in [-4, 4] m/s |
| Horizontal grid | 85 x 45 x 85 x 45 x 8 x 7 for (x_A, y_A, x_D, y_D, v_D^x, v_D^y) |
| Horizontal domain | (x_A, x_D) in [0, 45]m, (y_A, y_D) in [0, 25]m, velocities in [-6, 6] m/s |
| Solver | OptimizedDP toolbox |
| Horizontal solve time | ~3 hours for T=22s |

### Gazebo Setup
- **Simulator:** Gazebo physics simulator
- **Low-level controller:** PX4 Ardupilot
- **State information:** Ground truth from Gazebo (perfect information assumed)
- **Pipeline:** Value functions computed offline → Game-theoretic controller (online) → PX4 → Gazebo

---

## 5. What You Need to Implement

### Phase 1: Mathematical Foundation
- [ ] Understand Hamilton-Jacobi (HJ) reachability analysis and the HJI variational inequality (Eq. 10)
- [ ] Understand signed distance functions for representing reach/avoid sets (Eqs. 5-7)
- [ ] Understand the value function and optimal control extraction (Eqs. 9, 12, 13)

### Phase 2: Dynamics Setup
- [ ] Implement the **double integrator dynamics** for the defender (Eq. 16 left)
- [ ] Implement the **single integrator dynamics** for the attacker (Eq. 16 right)
- [ ] Define speed constraints (Eqs. 17a-17d)
- [ ] Define the proportional gain constants k_x, k_y, k_z

### Phase 3: Game Decomposition
- [ ] Define the cylindrical capture set decomposition (Eqs. 18a, 18b)
- [ ] Define the horizontal sub-game state space, dynamics, reach/avoid sets (Eqs. 19, 20a, 20b)
- [ ] Define the vertical sub-game state space, dynamics, reach set (Eqs. 23, 24)
- [ ] Define target region T and obstacles (horizontal only, per the key assumption)

### Phase 4: Solving the HJ PDEs (Offline Computation)
- [ ] Install and set up the **OptimizedDP toolbox** (Python-based, uses JAX for GPU acceleration)
  - GitHub: `https://github.com/SFU-MARS/optimized_dp`
- [ ] Alternatively use **helperOC / toolboxLS** (MATLAB-based)
- [ ] Solve for **Phi_z** (vertical reach-avoid value function) — 3D, relatively fast
- [ ] Solve for **V_{z,∞}** (vertical maximum distance value function) — 2D relative dynamics
- [ ] Solve for **Phi_h** (horizontal reach-avoid value function) — 6D, ~3 hours
- [ ] Solve for **V_{h,∞}** (horizontal maximum distance value function) — 4D relative dynamics
- [ ] Extract invariant capture sets B_z and B_h from V_{z,∞} and V_{h,∞}
- [ ] Solve for attacker's reaching value function Phi_{A,h}^{reach} (for T_goal computation)

### Phase 5: Control Algorithm Implementation (Online)
- [ ] Implement **Algorithm 1** (Vertical Reach-Track) with the three modes:
  1. Optimal reaching control (not yet captured vertically)
  2. Optimal tracking control (near boundary of B_z)
  3. Performant tracking / PID (deep inside B_z)
- [ ] Implement **Algorithm 2** (Horizontal Reach-Track-Avoid) with the three modes:
  1. Optimal reaching control with obstacle avoidance (not yet captured horizontally)
  2. Optimal tracking control (near boundary of B_h)
  3. Performant tracking / PID (deep inside B_h)
- [ ] Implement optimal control extraction via spatial derivatives of value functions (gradient lookup + interpolation on the grid)
- [ ] Implement the combined controller that runs both algorithms simultaneously

### Phase 6: Winning Condition Checking
- [ ] Implement T_goal computation (earliest time attacker reaches target, from Phi_{A,h}^{reach})
- [ ] Implement T_capture computation (earliest vertical capture time, from Phi_z)
- [ ] Implement the winning condition check: defender wins if x_h in W_{D,h}, x_z in W_{D,z}, and T_goal > T_capture

### Phase 7: Simulation & Validation
- [ ] Set up numerical simulation (forward Euler integration)
- [ ] Visualize value functions as 2D slices/contour plots
- [ ] Plot winning regions W_{D,h}, W_{A,h}, W_{D,z}, W_{A,z}
- [ ] Simulate trajectories with both players applying optimal control
- [ ] (Optional) Set up Gazebo with PX4 Ardupilot for physics-based validation

---

## 6. Key Software / Tools Needed

| Tool | Purpose |
|------|---------|
| **OptimizedDP** (Python/JAX) | Solve HJ PDEs numerically on GPU — primary tool used in paper |
| **helperOC + toolboxLS** (MATLAB) | Alternative HJ reachability toolbox |
| **NumPy / SciPy** | Grid interpolation for value function lookups, forward Euler simulation |
| **Matplotlib** | Visualization of value functions, winning regions, trajectories |
| **Gazebo + PX4 Ardupilot** | (Optional) Physics-based quadrotor simulation |
| **ROS** | (Optional) Communication layer for Gazebo experiments |

---

## 7. Key Equations Quick Reference

| Eq. | Description |
|-----|-------------|
| (10) | HJI variational inequality — the main PDE to solve |
| (12) | Optimal Hamiltonian |
| (13a,b) | Optimal controls for defender and attacker |
| (16) | Simplified joint dynamics (double + single integrator) |
| (18a,b) | Decomposed capture conditions (horizontal + vertical) |
| (21a,b) | Horizontal optimal controls |
| (25) | Vertical reach-avoid variational inequality |
| (26a,b) | Vertical optimal controls |
| (30) | Maximum distance value function (for invariant sets) |
| (33) | Vertical tracking controller |
| (37) | Horizontal tracking controller |
| (38) | Modified vertical reach set using invariant capture set |

---

## 8. Limitations & Assumptions to Be Aware Of

1. **Perfect information** — defender knows attacker's full state (position + velocity) at all times
2. **Target/obstacles are horizontal-only** — no vertical dependence on goal or obstacles
3. **Single attacker, single defender** — extension to multi-agent requires additional optimization (e.g., mixed integer programming)
4. **Conservative solution** — the decomposition produces winning regions that are *subsets* of the true optimal winning regions
5. **Grid-based computation** — limits the spatial domain and resolution; ~3 hours for the 6D horizontal game
6. **Attacker modeled as single integrator** — overestimates attacker capability, which is conservative but may be unrealistic
7. **Horizontal invariant set may not converge** — in the paper's parameters V_{h,T} doesn't converge as T→∞; they use T=2.5s instead

---

## 9. Glossary of Key Terms

| Term | Definition |
|------|-----------|
| **Reach-Avoid Game** | Differential game where attacker tries to reach a target while defender tries to capture it before that happens |
| **HJ Reachability** | Method that solves Hamilton-Jacobi PDEs to compute optimal strategies and winning regions |
| **Value Function (Phi)** | Function whose sub-zero level set gives the attacker's winning region; computed by solving HJI variational inequality |
| **Winning Region** | Set of initial states from which a player is guaranteed to win regardless of opponent's strategy |
| **Invariant Set** | A set of states that, once entered, can be maintained indefinitely using appropriate control |
| **Double Integrator** | Dynamics model where control input affects acceleration (has inertia) |
| **Single Integrator** | Dynamics model where control input directly sets velocity (no inertia) |
| **Reach-Track** | Two-phase controller: first reach the invariant capture set, then track/maintain capture status |
| **OptimizedDP** | GPU-accelerated Python toolbox for solving HJ PDEs |
| **Signed Distance Function** | Function that is negative inside a set and positive outside (distance to boundary) |
