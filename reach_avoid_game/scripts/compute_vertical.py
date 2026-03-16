"""CLI script to compute vertical sub-game value functions.

Runs the full vertical computation pipeline:
1. Phi_z: Vertical reach-avoid value function (3D)
2. V_z_inf: Maximum distance value function (2D relative)
3. B_z: Invariant capture set derived from V_z_inf
"""

import argparse
import time

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.vertical_solver import (
    solve_vertical_reach_avoid,
    solve_vertical_max_distance,
    compute_invariant_set_Bz,
)


def main():
    parser = argparse.ArgumentParser(description="Compute vertical sub-game value functions")
    parser.add_argument("--preset", default="dev", choices=["dev", "medium", "paper"],
                        help="Grid resolution preset (default: dev)")
    parser.add_argument("--output-dir", default="/workspace/data/value_functions/",
                        help="Output directory for value functions")
    parser.add_argument("--config", default="/workspace/config/game_params.yaml",
                        help="Path to game configuration YAML")
    args = parser.parse_args()

    config = GameConfig.from_yaml(args.config)

    print(f"Computing vertical value functions with '{args.preset}' preset")
    print(f"Output directory: {args.output_dir}")
    print()

    # Step 1: Phi_z
    t0 = time.time()
    phi_z_path = solve_vertical_reach_avoid(config, preset=args.preset, output_dir=args.output_dir)
    t1 = time.time()
    print(f"  Phi_z computed in {t1 - t0:.1f}s")
    print()

    # Step 2: V_z_inf
    v_z_inf_path = solve_vertical_max_distance(config, preset=args.preset, output_dir=args.output_dir)
    t2 = time.time()
    print(f"  V_z_inf computed in {t2 - t1:.1f}s")
    print()

    # Step 3: B_z
    b_z_path = compute_invariant_set_Bz(v_z_inf_path, d_z=config.capture.d_z, output_dir=args.output_dir)
    t3 = time.time()
    print(f"  B_z computed in {t3 - t2:.1f}s")
    print()

    print(f"Total computation time: {t3 - t0:.1f}s")
    print("All vertical value functions computed successfully.")


if __name__ == "__main__":
    main()
