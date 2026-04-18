"""Dynamics models for the reach-avoid differential game."""

__all__ = [
    "DefenderDynamics",
    "AttackerDynamics",
    "VerticalGameDynamics",
    "HorizontalGameDynamics",
]


def __getattr__(name):
    if name == "DefenderDynamics":
        from reach_avoid_game.dynamics.defender import DefenderDynamics
        return DefenderDynamics
    if name == "AttackerDynamics":
        from reach_avoid_game.dynamics.attacker import AttackerDynamics
        return AttackerDynamics
    if name == "VerticalGameDynamics":
        from reach_avoid_game.dynamics.vertical_game import VerticalGameDynamics
        return VerticalGameDynamics
    if name == "HorizontalGameDynamics":
        from reach_avoid_game.dynamics.horizontal_game import HorizontalGameDynamics
        return HorizontalGameDynamics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
