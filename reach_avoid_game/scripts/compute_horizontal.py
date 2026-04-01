"""CLI script to compute horizontal sub-game value functions.

Runs the full horizontal computation pipeline (Paper order):
1. V_h_T: Maximum distance value function (4D relative)
2. B_h: Invariant capture set derived from V_h_T
3. phi_A_reach: Attacker reaching value function (2D)
4. Phi_h: Horizontal reach-avoid value function (6D) with B_h feedback (Eq. 39)
"""

import argparse
import gc
import time

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.value_function_io import load_value_function
from reach_avoid_game.solvers.horizontal_solver import (
    solve_horizontal_reach_avoid,
    solve_horizontal_max_distance,
    compute_invariant_set_Bh,
    solve_attacker_reaching,
)


def main():
    parser = argparse.ArgumentParser(description="Compute horizontal sub-game value functions")
    parser.add_argument("--preset", default="dev", choices=["dev", "medium", "paper"],
                        help="Grid resolution preset (default: dev)")
    parser.add_argument("--output-dir", default="/workspace/data/value_functions/",
                        help="Output directory for value functions")
    parser.add_argument("--config", default="/workspace/config/game_params.yaml",
                        help="Path to game configuration YAML")
    parser.add_argument("--include-6d", action="store_true",
                        help="Also compute 6D V_h_T extension with obstacles")
    args = parser.parse_args()

    config = GameConfig.from_yaml(args.config)
    config.apply_preset(args.preset)

    print(f"Computing horizontal value functions with '{args.preset}' preset")
    print(f"Output directory: {args.output_dir}")
    print()

    # Step 1: V_h_T (4D relative)
    t0 = time.time()
    v_h_t_path = solve_horizontal_max_distance(config, preset=args.preset, output_dir=args.output_dir)
    t1 = time.time()
    print(f"  V_h_T computed in {t1 - t0:.1f}s")
    print()

    # Step 2: B_h
    b_h_path = compute_invariant_set_Bh(v_h_t_path, d_h=config.capture.d_h, output_dir=args.output_dir)
    t2 = time.time()
    print(f"  B_h computed in {t2 - t1:.1f}s")
    print()

    # Step 3: Attacker reaching (2D) — independent, can be in any order
    phi_a_path = solve_attacker_reaching(config, preset=args.preset, output_dir=args.output_dir)
    t3 = time.time()
    print(f"  phi_A_reach computed in {t3 - t2:.1f}s")
    print()

    # Free JAX/numpy memory from previous solves before the large 6D solve
    gc.collect()

    # Step 4: Phi_h with B_h feedback (Paper Eq. 39)
    v_h_t_data = load_value_function(v_h_t_path)
    phi_h_path = solve_horizontal_reach_avoid(
        config, preset=args.preset, output_dir=args.output_dir,
        v_h_t_data=v_h_t_data,
    )
    t4 = time.time()
    print(f"  Phi_h computed in {t4 - t3:.1f}s")
    print()

    # Optional: 6D V_h extension
    if args.include_6d:
        gc.collect()
        from reach_avoid_game.solvers.horizontal_solver import solve_horizontal_max_distance_6d
        v_h_6d_path = solve_horizontal_max_distance_6d(config, preset=args.preset, output_dir=args.output_dir)
        t5 = time.time()
        print(f"  V_h_T_6d computed in {t5 - t4:.1f}s")
        print()
        t4 = t5

    print(f"Total computation time: {t4 - t0:.1f}s")
    print("All horizontal value functions computed successfully.")


if __name__ == "__main__":
    main()
