"""CLI script for visualization of value functions and trajectories.

Supports:
- vertical: Plot vertical value function slice
- horizontal: Plot horizontal value function slice
- winning_regions: Plot winning regions overlay
- trajectory: Plot saved trajectory
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from reach_avoid_game.solvers.value_function_io import load_value_function
from reach_avoid_game.visualization import (
    plot_value_function_2d,
    plot_winning_regions,
    plot_trajectory_2d,
)


def parse_slice_args(slice_dims_str, slice_values_str):
    """Parse --slice-dims and --slice-values from CLI strings."""
    if slice_dims_str is None:
        return None, None
    dims = [int(x) for x in slice_dims_str.split(",")]
    values = [float(x) for x in slice_values_str.split(",")]
    return dims, values


def main():
    parser = argparse.ArgumentParser(description="Visualize reach-avoid game data")
    parser.add_argument("--type", required=True,
                        choices=["vertical", "horizontal", "winning_regions", "trajectory",
                                 "attacker_reaching"],
                        help="What to visualize")
    parser.add_argument("--save", default=None,
                        help="Output file path for the plot")
    parser.add_argument("--slice-dims", default=None,
                        help="Comma-separated dimension indices to slice (e.g., '1' or '2,3,4,5')")
    parser.add_argument("--slice-values", default=None,
                        help="Comma-separated values for sliced dimensions (e.g., '0.0' or '0,0,20,12')")
    parser.add_argument("--vf-dir", default="/workspace/data/value_functions/",
                        help="Directory containing value function files")
    parser.add_argument("--traj-file", default="/workspace/data/simulations/vertical_sim.npz",
                        help="Trajectory file for --type trajectory")
    args = parser.parse_args()

    vf_dir = Path(args.vf_dir)
    slice_dims, slice_values = parse_slice_args(args.slice_dims, args.slice_values)

    if args.type == "vertical":
        vf = load_value_function(vf_dir / "phi_z.npz")
        save_path = args.save or "/workspace/data/plots/vertical_vf.png"
        if slice_dims is None:
            # Default: slice v_D_z at 0
            slice_dims = [1]
            slice_values = [0.0]
        plot_value_function_2d(vf, slice_dims=slice_dims, slice_values=slice_values,
                              save_path=save_path, title="Vertical Value Function Phi_z")

    elif args.type == "horizontal":
        vf = load_value_function(vf_dir / "phi_h.npz")
        save_path = args.save or "/workspace/data/plots/horizontal_vf.png"
        if slice_dims is None:
            # Default: slice all but first 2 dims at midpoints
            slice_dims = [2, 3, 4, 5]
            slice_values = [0.0, 0.0, 22.5, 12.5]
        plot_value_function_2d(vf, slice_dims=slice_dims, slice_values=slice_values,
                              save_path=save_path, title="Horizontal Value Function Phi_h")

    elif args.type == "winning_regions":
        # Use attacker reaching (2D, has both positive and negative regions)
        vf = load_value_function(vf_dir / "phi_A_reach.npz")
        save_path = args.save or "/workspace/data/plots/winning_regions.png"
        plot_winning_regions(vf, slice_dims=slice_dims, slice_values=slice_values,
                            save_path=save_path, title="Attacker Reachable Region")

    elif args.type == "attacker_reaching":
        vf = load_value_function(vf_dir / "phi_A_reach.npz")
        save_path = args.save or "/workspace/data/plots/attacker_reaching.png"
        plot_value_function_2d(vf, save_path=save_path,
                              title="Attacker Reaching Value Function")

    elif args.type == "trajectory":
        import numpy as np
        traj_file = Path(args.traj_file)
        save_path = args.save or "/workspace/data/plots/trajectory.png"

        with np.load(traj_file) as data:
            traj_data = {k: data[k] for k in data.files}

        plot_trajectory_2d(traj_data, save_path=save_path, title="Game Trajectory")

    print("Done.")


if __name__ == "__main__":
    main()
