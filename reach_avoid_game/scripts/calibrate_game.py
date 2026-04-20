"""Deterministically choose a paper-valid calibrated game config.

The calibration pass keeps d_z=1 and d_h=3 by default. It searches defender
gains/presets in a fixed order and accepts the first config whose conservative
invariant values produce nonempty paper-threshold sets at useful near-capture
states.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.horizontal_solver import compute_invariant_set_Bh, solve_horizontal_max_distance_6d
from reach_avoid_game.solvers.value_function_io import load_value_function
from reach_avoid_game.solvers.vertical_solver import compute_invariant_set_Bz, solve_vertical_max_distance


K_Z_CANDIDATES = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
K_H_CANDIDATES = [0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
PRESET_CANDIDATES = ["dev", "medium"]


def _candidate_configs(base_path: str | Path):
    for preset in PRESET_CANDIDATES:
        for k_z in K_Z_CANDIDATES:
            for k_h in K_H_CANDIDATES:
                cfg = GameConfig.from_yaml(base_path)
                cfg.apply_preset(preset)
                cfg.defender.k_z = k_z
                cfg.defender.k_x = k_h
                cfg.defender.k_y = k_h
                yield cfg, {
                    "preset": preset,
                    "k_z": k_z,
                    "k_x": k_h,
                    "k_y": k_h,
                    "vertical_horizon": "conservative_invariant",
                    "horizontal_tracking_horizon": "conservative_invariant",
                }


def _is_candidate_valid(config: GameConfig, temp_dir: Path) -> bool:
    v_z_path = solve_vertical_max_distance(config, preset=config.grid_preset, output_dir=temp_dir)
    b_z_path = compute_invariant_set_Bz(v_z_path, d_z=config.capture.d_z, output_dir=temp_dir)
    v_h_path = solve_horizontal_max_distance_6d(config, preset=config.grid_preset, output_dir=temp_dir)
    b_h_path = compute_invariant_set_Bh(v_h_path, d_h=config.capture.d_h, output_dir=temp_dir)

    b_z = load_value_function(b_z_path)
    b_h = load_value_function(b_h_path)
    z_center = b_z.values[b_z.values.shape[0] // 2, b_z.values.shape[1] // 2] > 0.5
    h_center = b_h.values[
        b_h.values.shape[0] // 2,
        b_h.values.shape[1] // 2,
        b_h.values.shape[2] // 2,
        b_h.values.shape[3] // 2,
        b_h.values.shape[4] // 2,
        b_h.values.shape[5] // 2,
    ] > 0.5
    return bool(z_center and h_center)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate paper-valid reachability config")
    parser.add_argument("--config", default="config/game_params.yaml")
    parser.add_argument("--output", default="config/generated_calibrated_game_params.yaml")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="reach_calibration_") as tmp:
        temp_dir = Path(tmp)
        for config, calibration in _candidate_configs(args.config):
            candidate_dir = temp_dir / f"{calibration['preset']}_kz{calibration['k_z']}_kh{calibration['k_x']}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            if _is_candidate_valid(config, candidate_dir):
                config.grid_preset = calibration["preset"]
                config.to_yaml(args.output)
                print(f"Selected calibrated config: {calibration}")
                print(f"Wrote {args.output}")
                return

    raise SystemExit("No calibrated config produced nonempty paper-valid B_z and B_h.")


if __name__ == "__main__":
    main()
