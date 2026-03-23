---
task_type: feature
summary: Implement game-theoretic optimal control for attacker drone using HJ value functions
codebase_context: |
  - attacker_node.py has a STUB _optimal_control() at line 203-214
  - ValueFunctionLoader in reach_avoid_controller provides get_gradient() and get_value()
  - Precomputed VFs exist: phi_h.npz (6D), phi_z.npz (3D) in /workspace/data/value_functions/
  - Defender node uses proven reach-track pattern with same VF loader
  - Attacker bang-bang: d_x = -U_A_h * sign(dPhi_h/dx_A), d_y = -U_A_h * sign(dPhi_h/dy_A), d_z = -U_A_z * sign(dPhi_z/dz_A)
approach: |
  Mirror defender's proven pattern: load VFs at init, subscribe to defender state,
  extract bang-bang optimal controls from phi_h/phi_z gradients each tick.
  Fallback to simple goal-seeking when gradient is near-zero or state outside grid.
  No reach-track state machine needed (attacker is simpler single integrator).
max_iterations: 4
reasoning: |
  4 phases, each small and well-defined. Low context pressure (24.6%).
  Phase 1 is the only non-trivial phase (~100 lines). Phases 2-4 are mechanical.
context_budget:
  pressure_rating: low
  pressure_pct: 24.6
  peak_iteration_tokens: 49200
  context_window: 200000
  estimated_cost_range: "$2-5"
  file_count: 8
  file_categories:
    small: 8
    medium: 0
    large: 0
unresolved_warnings: none
---
