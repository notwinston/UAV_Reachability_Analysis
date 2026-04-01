"""Game configuration dataclasses loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RoomConfig:
    x_min: float = 0.0
    x_max: float = 45.0
    y_min: float = 0.0
    y_max: float = 25.0
    z_min: float = 0.0
    z_max: float = 20.0


@dataclass
class DefenderConfig:
    max_speed_horizontal: float = 6.0
    max_speed_vertical: float = 4.0
    k_x: float = 0.7
    k_y: float = 0.7
    k_z: float = 1.5


@dataclass
class AttackerConfig:
    max_speed_horizontal: float = 3.0
    max_speed_vertical: float = 2.0


@dataclass
class CaptureConfig:
    d_h: float = 3.0
    d_z: float = 1.0


@dataclass
class TargetRegionConfig:
    type: str = "box"
    x_min: float = 38.0
    x_max: float = 45.0
    y_min: float = 10.0
    y_max: float = 15.0


@dataclass
class ObstacleConfig:
    type: str = "box"
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0


@dataclass
class VerticalGridConfig:
    z_rel_points: int = 81
    v_dz_points: int = 41
    z_rel_range: list[float] = field(default_factory=lambda: [-10.0, 10.0])
    v_dz_range: list[float] = field(default_factory=lambda: [-4.0, 4.0])


@dataclass
class Vertical3DGridConfig:
    z_d_points: int = 41
    v_dz_points: int = 21
    z_a_points: int = 41


@dataclass
class HorizontalGridConfig:
    # 4D relative grid (V_h_T, B_h): [x_rel, y_rel, v_D_x, v_D_y]
    rel_pos_points: int = 21       # x_rel and y_rel resolution
    rel_vel_points: int = 11       # v_D_x and v_D_y resolution
    rel_pos_range: float = 6.0     # x_rel, y_rel domain: [-range, range] meters
    # 6D absolute grid (Phi_h): [x_D, y_D, v_D_x, v_D_y, x_A, y_A]
    game_x_points: int = 9         # x_D and x_A resolution
    game_y_points: int = 7         # y_D and y_A resolution
    game_vel_x_points: int = 5     # v_D_x resolution
    game_vel_y_points: int = 5     # v_D_y resolution
    # 2D attacker reaching grid: [x_A, y_A]
    reach_x_points: int = 21       # x_A resolution
    reach_y_points: int = 13       # y_A resolution


@dataclass
class SolverConfig:
    accuracy: str = "medium"
    time_horizon: float = 15.0
    time_steps: int = 100


@dataclass
class GridConfig:
    vertical: VerticalGridConfig = field(default_factory=VerticalGridConfig)
    vertical_3d: Vertical3DGridConfig = field(default_factory=Vertical3DGridConfig)
    horizontal: HorizontalGridConfig = field(default_factory=HorizontalGridConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)


@dataclass
class SimulationConfig:
    dt: float = 0.1
    max_time: float = 60.0


@dataclass
class GameConfig:
    room: RoomConfig = field(default_factory=RoomConfig)
    defender: DefenderConfig = field(default_factory=DefenderConfig)
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    target_region: TargetRegionConfig = field(default_factory=TargetRegionConfig)
    obstacles: list[ObstacleConfig] = field(default_factory=list)
    grid: GridConfig = field(default_factory=GridConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    grid_preset: str = "dev"
    grid_presets: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GameConfig:
        """Load game configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()

        if "room" in data:
            config.room = RoomConfig(**data["room"])
        if "defender" in data:
            config.defender = DefenderConfig(**data["defender"])
        if "attacker" in data:
            config.attacker = AttackerConfig(**data["attacker"])
        if "capture" in data:
            config.capture = CaptureConfig(**data["capture"])
        if "target_region" in data:
            config.target_region = TargetRegionConfig(**data["target_region"])
        if "obstacles" in data:
            config.obstacles = [ObstacleConfig(**obs) for obs in data["obstacles"]]
        if "grid" in data:
            gd = data["grid"]
            grid = GridConfig()
            if "vertical" in gd:
                grid.vertical = VerticalGridConfig(**gd["vertical"])
            if "vertical_3d" in gd:
                grid.vertical_3d = Vertical3DGridConfig(**gd["vertical_3d"])
            if "horizontal" in gd:
                grid.horizontal = HorizontalGridConfig(**gd["horizontal"])
            if "solver" in gd:
                grid.solver = SolverConfig(**gd["solver"])
            config.grid = grid
        if "simulation" in data:
            config.simulation = SimulationConfig(**data["simulation"])
        if "grid_preset" in data:
            config.grid_preset = data["grid_preset"]
        if "grid_presets" in data:
            config.grid_presets = data["grid_presets"]

        # Apply preset overrides to grid config
        config.apply_preset(config.grid_preset)

        return config

    def apply_preset(self, preset_name: str) -> None:
        """Apply a grid preset, overriding current grid settings.

        Args:
            preset_name: Name of preset (e.g., "dev", "medium", "paper")
        """
        if preset_name and preset_name in self.grid_presets:
            preset = self.grid_presets[preset_name]
            if "vertical" in preset:
                for k, v in preset["vertical"].items():
                    setattr(self.grid.vertical, k, v)
            if "vertical_3d" in preset:
                for k, v in preset["vertical_3d"].items():
                    setattr(self.grid.vertical_3d, k, v)
            if "horizontal" in preset:
                for k, v in preset["horizontal"].items():
                    setattr(self.grid.horizontal, k, v)
            self.grid_preset = preset_name
