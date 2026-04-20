"""CLI script to compute vertical sub-game value functions.

Runs the full vertical computation pipeline (Paper order):
1. V_z_inf: Maximum distance value function (2D relative)
2. B_z: Invariant capture set derived from V_z_inf
3. Phi_z: Vertical reach-avoid value function (3D) with B_z as target (Eq. 38)
"""

import argparse
from pathlib import Path
import shutil
import tempfile
import time

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.value_function_io import load_value_function
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
    parser.add_argument("--calibrated-config", default=None,
                        help="Optional calibrated YAML config to use instead of --config")
    args = parser.parse_args()

    config = GameConfig.from_yaml(args.calibrated_config or args.config)
    config.apply_preset(args.preset)

    print(f"Computing vertical value functions with '{args.preset}' preset")
    print(f"Output directory: {args.output_dir}")
    print()

    final_output = Path(args.output_dir)
    final_output.mkdir(parents=True, exist_ok=True)
    temp_root = final_output.parent
    temp_dir = Path(tempfile.mkdtemp(prefix=f".vertical_{args.preset}_", dir=temp_root))

    # Step 1: V_z_inf (2D relative)
    t0 = time.time()
    v_z_inf_path = solve_vertical_max_distance(config, preset=args.preset, output_dir=temp_dir)
    t1 = time.time()
    print(f"  V_z_inf computed in {t1 - t0:.1f}s")
    print()

    # Step 2: B_z
    b_z_path = compute_invariant_set_Bz(v_z_inf_path, d_z=config.capture.d_z, output_dir=temp_dir)
    t2 = time.time()
    print(f"  B_z computed in {t2 - t1:.1f}s")
    print()

    # Step 3: Phi_z with B_z as target (Paper Eq. 38)
    v_z_inf_data = load_value_function(v_z_inf_path)
    phi_z_path = solve_vertical_reach_avoid(
        config, preset=args.preset, output_dir=temp_dir,
        v_z_inf_data=v_z_inf_data,
    )
    t3 = time.time()
    print(f"  Phi_z computed in {t3 - t2:.1f}s")
    print()

    for name in ["V_z_inf.npz", "B_z.npz", "phi_z.npz", "phi_z_time_slices.npz"]:
        shutil.move(str(temp_dir / name), str(final_output / name))
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"Total computation time: {t3 - t0:.1f}s")
    print("All vertical value functions computed and promoted successfully.")


if __name__ == "__main__":
    main()
