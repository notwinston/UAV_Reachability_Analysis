"""CLI script to compute horizontal sub-game value functions.

Runs the full horizontal computation pipeline:
1. Phi_h: Horizontal reach-avoid value function (6D)
2. V_h_T: Maximum distance value function (4D relative)
3. B_h: Invariant capture set derived from V_h_T
4. phi_A_reach: Attacker reaching value function (2D)
"""

import argparse
import time

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.horizontal_solver import (
    solve_horizontal_reach_avoid,
    solve_horizontal_max_distance,
    compute_invariant_set_Bh,
    solve_attacker_reaching,
)


def main():
    parser = argparse.ArgumentParser(description="Compute horizontal sub-game value functions")
    parser.add_argument("--preset", default="dev", choices=["dev", "paper"],
                        help="Grid resolution preset (default: dev)")
    parser.add_argument("--output-dir", default="/workspace/data/value_functions/",
                        help="Output directory for value functions")
    parser.add_argument("--config", default="/workspace/config/game_params.yaml",
                        help="Path to game configuration YAML")
    args = parser.parse_args()

    config = GameConfig.from_yaml(args.config)

    print(f"Computing horizontal value functions with '{args.preset}' preset")
    print(f"Output directory: {args.output_dir}")
    print()

    # Step 1: Phi_h (6D)
    t0 = time.time()
    phi_h_path = solve_horizontal_reach_avoid(config, preset=args.preset, output_dir=args.output_dir)
    t1 = time.time()
    print(f"  Phi_h computed in {t1 - t0:.1f}s")
    print()

    # Step 2: V_h_T (4D relative)
    v_h_t_path = solve_horizontal_max_distance(config, preset=args.preset, output_dir=args.output_dir)
    t2 = time.time()
    print(f"  V_h_T computed in {t2 - t1:.1f}s")
    print()

    # Step 3: B_h
    b_h_path = compute_invariant_set_Bh(v_h_t_path, d_h=config.capture.d_h, output_dir=args.output_dir)
    t3 = time.time()
    print(f"  B_h computed in {t3 - t2:.1f}s")
    print()

    # Step 4: Attacker reaching (2D)
    phi_a_path = solve_attacker_reaching(config, preset=args.preset, output_dir=args.output_dir)
    t4 = time.time()
    print(f"  phi_A_reach computed in {t4 - t3:.1f}s")
    print()

    print(f"Total computation time: {t4 - t0:.1f}s")
    print("All horizontal value functions computed successfully.")


if __name__ == "__main__":
    main()
