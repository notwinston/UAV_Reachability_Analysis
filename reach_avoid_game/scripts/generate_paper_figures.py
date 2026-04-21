"""Generate paper figures (Bui et al., arXiv:2512.22793) from computed value functions.

Reproduces Figures 4-12 from the paper using the computed value functions.
Each figure function loads the required data and saves a publication-quality PNG.

Figure catalogue
----------------
Paper figures (static value-function visualizations):

  fig_4.png  — B_z invariant capture set in the (z_rel, v_Dz) plane.
               The filled contour shows V_z_inf; the black boundary marks the
               set where the defender can guarantee vertical capture (V_z_inf = d_z).

  fig_5.png  — Phi_z value-function slices at three representative defender
               vertical velocities (v_Dz = -2, 0, +2 m/s).  Each panel shows
               the zero-level set that separates defender-winning from
               attacker-winning initial conditions in the (z_D, z_A) plane.

  fig_6.png  — Side-by-side view of the vertical sub-game: left panel is Phi_z
               at v_Dz = 0 with the diagonal capture band; right panel is V_z_inf
               with the B_z boundary overlaid.

  fig_7.png  — Obstacle-aware B_h horizontal capture set.  Sliced at v_D = 0 with
               the attacker fixed at the room centre.  Shows how obstacles deform
               the reachable capture region in the (x_D, y_D) plane.

  fig_8.png  — Phi_h value-function slice at v_D = 0, attacker at room centre.
               Zero contour separates W_D (defender wins) from W_A (attacker wins).
               Obstacles and the target region are overlaid.

  fig_10.png — Horizontal winning regions W_D (blue) and W_A (salmon) derived from
               the sign of Phi_h, sliced at v_D = 0 with attacker at room centre.
               Obstacles and target region overlaid for spatial context.

  fig_attacker_reaching.png — Attacker's own reach value function phi_A_reach in the
               (x_A, y_A) plane.  Zero contour marks the boundary from which the
               attacker can reach the target despite the defender.

  fig_vertical_winning.png  — Vertical winning regions W_Dz (blue) and W_Az (salmon)
               derived from Phi_z at v_Dz = 0.  Shows the altitude pairs from which
               each agent prevails.

Simulation figures (40 s forward-Euler trajectories, defender at (35, 20, 8),
                    attacker at (5, 3, 3) — opposite corners, attacker behind obstacle):

  fig_11.png — Three-panel simulation summary: top-down (x, y) trajectories,
               side-view (x, z) trajectories, and altitude z_D / z_A vs time
               with the vertical capture band and mode annotations.

  fig_12.png — Combined trajectory layout with a full 3D view plus companion
               top-down and side-view projections for easier inspection.

Analysis figures (derived from the same simulation run):

  fig_distance_over_time.png — Three stacked panels showing 3D Euclidean distance,
               horizontal distance, and vertical distance between the agents over
               time.  Dashed green lines mark the respective capture thresholds
               (d_h, d_z) so convergence rate is immediately visible.

  fig_control_effort.png — 2x3 grid of control and disturbance signals over time.
               Top row: defender inputs u_x, u_y, u_z.  Bottom row: attacker
               disturbances d_x, d_y, d_z.  Dotted lines show saturation limits.

  fig_speed_profiles.png — Left panel overlays horizontal and vertical speed
               components for both agents; right panel shows full 3D speed
               magnitudes.  Useful for checking whether agents saturate their
               speed limits and for comparing agility.

  fig_mode_timeline.png — Two colour-coded timeline strips (vertical sub-game on
               top, horizontal on bottom).  Red = Reach (optimising Phi), blue =
               Track boundary (optimising V_h_T / V_z_inf), green = PID (deep
               inside invariant set).  Shows exactly when and for how long each
               control regime is active.

  fig_altitude_phase_portrait.png — Trajectory in the (z_D, z_A) phase plane,
               coloured by elapsed time (viridis).  The green band marks the
               vertical capture zone (|z_D - z_A| <= d_z).  Reveals whether the
               defender converges to the diagonal and whether altitude capture
               precedes or follows horizontal capture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.value_function_io import is_paper_valid, load_value_function


SIMULATION_HORIZON_SECONDS = 40.0


PAPER_REQUIRED = {"B_z.npz", "B_h.npz", "phi_z.npz", "phi_h.npz"}
FIGURE_OUTPUTS = {
    "Fig 4": "fig_4.png",
    "Fig 5": "fig_5.png",
    "Fig 6": "fig_6.png",
    "Fig 7": "fig_7.png",
    "Fig 8": "fig_8.png",
    "Fig 10": "fig_10.png",
    "Fig 11": "fig_11.png",
    "Fig 12": "fig_12.png",
    "Attacker Reaching": "fig_attacker_reaching.png",
    "Vertical Winning": "fig_vertical_winning.png",
    "Distance Over Time": "fig_distance_over_time.png",
    "Control Effort": "fig_control_effort.png",
    "Speed Profiles": "fig_speed_profiles.png",
    "Mode Timeline": "fig_mode_timeline.png",
    "Altitude Phase Portrait": "fig_altitude_phase_portrait.png",
}


def _diagnostic_figure(output_dir: Path, filename: str, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.35, message, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(str(output_dir / filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _invalid_paper_deps(vf_dir: Path, deps: list[str]) -> list[str]:
    invalid = []
    for dep in deps:
        if dep not in PAPER_REQUIRED:
            continue
        try:
            if not is_paper_valid(load_value_function(vf_dir / dep)):
                invalid.append(dep)
        except Exception:
            invalid.append(dep)
    return invalid


def _build_axes(vf_data, dim_indices):
    """Build axes for given dimensions from value function data."""
    axes = []
    for d in dim_indices:
        axes.append(np.linspace(
            float(vf_data.grid_min[d]), float(vf_data.grid_max[d]),
            vf_data.values.shape[d],
        ))
    return axes


def _slice_nd(vf_data, fix_dims, fix_values):
    """Slice an N-D value function at given dimension values.

    Returns the 2D slice and (x_axis, y_axis) arrays for the free dimensions.
    """
    values = vf_data.values
    ndim = values.ndim
    free_dims = sorted(set(range(ndim)) - set(fix_dims))

    idx = [slice(None)] * ndim
    for d, v in zip(fix_dims, fix_values):
        axis = np.linspace(
            float(vf_data.grid_min[d]), float(vf_data.grid_max[d]),
            values.shape[d],
        )
        closest = int(np.argmin(np.abs(axis - v)))
        idx[d] = closest

    sliced = values[tuple(idx)]
    x_axis = np.linspace(
        float(vf_data.grid_min[free_dims[0]]), float(vf_data.grid_max[free_dims[0]]),
        values.shape[free_dims[0]],
    )
    y_axis = np.linspace(
        float(vf_data.grid_min[free_dims[1]]), float(vf_data.grid_max[free_dims[1]]),
        values.shape[free_dims[1]],
    )
    return sliced, x_axis, y_axis


def _contour_if_in_range(ax, x_axis, y_axis, values_t, level, **kwargs):
    """Draw a contour level only when it lies inside the plotted value range."""
    v_min = float(np.nanmin(values_t))
    v_max = float(np.nanmax(values_t))
    if v_min <= float(level) <= v_max:
        ax.contour(x_axis, y_axis, values_t, levels=[level], **kwargs)
        return True
    return False


def fig4_bz_invariant_set(vf_dir, output_dir, config):
    """Fig 4 — B_z invariant set: 2D contour of V_z_inf with B_z boundary."""
    v_z_inf = load_value_function(vf_dir / "V_z_inf.npz")

    z_rel_axis, v_dz_axis = _build_axes(v_z_inf, [0, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(z_rel_axis, v_dz_axis, v_z_inf.values.T, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax, label=r"$V_{z,\infty}$ (m)")

    # B_z boundary: paper threshold V_z_inf = d_z.
    drawn = _contour_if_in_range(
        ax, z_rel_axis, v_dz_axis, v_z_inf.values.T, config.capture.d_z,
        colors="black", linewidths=2.5, linestyles="solid",
    )

    ax.set_xlabel(r"$z_{rel}$ (m)", fontsize=12)
    ax.set_ylabel(r"$v_{D,z}$ (m/s)", fontsize=12)
    ax.set_title(r"Vertical Invariant Capture Set $B_z$", fontsize=14)

    legend = [
        plt.Line2D([0], [0], color="black", linewidth=2.5,
                   label=f"$B_z$ boundary ($d_z={config.capture.d_z}$" + (")" if drawn else ", outside range)")),
    ]
    ax.legend(handles=legend, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_4.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_4.png (B_z invariant set)")


def fig5_phi_z_slices(vf_dir, output_dir, config):
    """Fig 5 — Phi_z value function slices at multiple v_D_z values."""
    phi_z = load_value_function(vf_dir / "phi_z.npz")
    n_v = phi_z.values.shape[1]
    v_dz_axis = np.linspace(float(phi_z.grid_min[1]), float(phi_z.grid_max[1]), n_v)

    # Pick 3 representative velocities
    v_targets = [-2.0, 0.0, 2.0]
    v_indices = [int(np.argmin(np.abs(v_dz_axis - vt))) for vt in v_targets]
    v_actual = [v_dz_axis[vi] for vi in v_indices]

    z_d_axis = np.linspace(float(phi_z.grid_min[0]), float(phi_z.grid_max[0]), phi_z.values.shape[0])
    z_a_axis = np.linspace(float(phi_z.grid_min[2]), float(phi_z.grid_max[2]), phi_z.values.shape[2])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, vi, va in zip(axes, v_indices, v_actual):
        data = phi_z.values[:, vi, :]
        cf = ax.contourf(z_d_axis, z_a_axis, data.T, levels=20, cmap="RdBu_r")
        ax.contour(z_d_axis, z_a_axis, data.T, levels=[0.0], colors="black", linewidths=2)
        plt.colorbar(cf, ax=ax, label=r"$\Phi_z$")
        ax.set_xlabel(r"$z_D$ (m)", fontsize=11)
        ax.set_ylabel(r"$z_A$ (m)", fontsize=11)
        ax.set_title(rf"$v_{{D,z}} = {va:.1f}$ m/s", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle(r"Vertical Reach-Avoid $\Phi_z$ Slices", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_5.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_5.png (Phi_z slices)")


def fig6_vertical_3d(vf_dir, output_dir, config):
    """Fig 6 — 3D view of vertical game: 2D slice with trajectory overlay."""
    phi_z = load_value_function(vf_dir / "phi_z.npz")
    v_z_inf = load_value_function(vf_dir / "V_z_inf.npz")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Phi_z at v_Dz=0
    n_v = phi_z.values.shape[1]
    v_idx = n_v // 2
    z_d_axis = np.linspace(float(phi_z.grid_min[0]), float(phi_z.grid_max[0]), phi_z.values.shape[0])
    z_a_axis = np.linspace(float(phi_z.grid_min[2]), float(phi_z.grid_max[2]), phi_z.values.shape[2])

    cf = axes[0].contourf(z_d_axis, z_a_axis, phi_z.values[:, v_idx, :].T, levels=20, cmap="RdBu_r")
    axes[0].contour(z_d_axis, z_a_axis, phi_z.values[:, v_idx, :].T, levels=[0.0], colors="black", linewidths=2)
    plt.colorbar(cf, ax=axes[0], label=r"$\Phi_z$")

    # Add diagonal capture zone lines
    z_vals = np.linspace(0, 20, 100)
    axes[0].plot(z_vals, z_vals, "g--", alpha=0.5, label=r"$z_D = z_A$")
    axes[0].plot(z_vals, z_vals - config.capture.d_z, "g:", alpha=0.4)
    axes[0].plot(z_vals, z_vals + config.capture.d_z, "g:", alpha=0.4)
    axes[0].set_xlabel(r"$z_D$ (m)", fontsize=12)
    axes[0].set_ylabel(r"$z_A$ (m)", fontsize=12)
    axes[0].set_title(r"$\Phi_z$ at $v_{D,z} = 0$", fontsize=13)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Right: V_z_inf with B_z
    z_rel_axis = np.linspace(float(v_z_inf.grid_min[0]), float(v_z_inf.grid_max[0]), v_z_inf.values.shape[0])
    v_dz_axis = np.linspace(float(v_z_inf.grid_min[1]), float(v_z_inf.grid_max[1]), v_z_inf.values.shape[1])

    cf2 = axes[1].contourf(z_rel_axis, v_dz_axis, v_z_inf.values.T, levels=20, cmap="viridis")
    plt.colorbar(cf2, ax=axes[1], label=r"$V_{z,\infty}$")
    drawn = _contour_if_in_range(
        axes[1], z_rel_axis, v_dz_axis, v_z_inf.values.T, config.capture.d_z,
        colors="black", linewidths=2.5,
    )
    axes[1].set_xlabel(r"$z_{rel}$ (m)", fontsize=12)
    axes[1].set_ylabel(r"$v_{D,z}$ (m/s)", fontsize=12)
    axes[1].set_title(r"$V_{z,\infty}$ with paper $B_z$ boundary", fontsize=13)
    axes[1].legend(handles=[
        plt.Line2D([0], [0], color="black", linewidth=2.5,
                   label=rf"$d_z={config.capture.d_z}$" + ("" if drawn else " (outside range)")),
    ], loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Vertical Sub-Game Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_6.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_6.png (vertical 3D view)")


def fig7_bh_invariant_set(vf_dir, output_dir, config):
    """Fig 7 — B_h invariant set from obstacle-aware 6D V_h_T."""
    v_h_t = load_value_function(vf_dir / "V_h_T_6d.npz")

    x_a_mid = (config.room.x_min + config.room.x_max) / 2
    y_a_mid = (config.room.y_min + config.room.y_max) / 2
    sliced, x_axis, y_axis = _slice_nd(v_h_t, [2, 3, 4, 5], [0.0, 0.0, x_a_mid, y_a_mid])

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(x_axis, y_axis, sliced.T, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax, label=r"$V_{h,T}$ (m)")

    # B_h boundary: paper threshold V_h = d_h.
    drawn = _contour_if_in_range(
        ax, x_axis, y_axis, sliced.T, config.capture.d_h,
        colors="black", linewidths=2.5,
    )

    # d_h circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x_a_mid + config.capture.d_h * np.cos(theta), y_a_mid + config.capture.d_h * np.sin(theta),
            "w--", linewidth=1, alpha=0.5, label=f"$d_h={config.capture.d_h}$m circle")

    ax.set_xlabel(r"$x_D$ (m)", fontsize=12)
    ax.set_ylabel(r"$y_D$ (m)", fontsize=12)
    ax.set_title(r"Obstacle-aware $B_h$ slice ($v_D=0$, attacker at center)", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", linewidth=2.5))
    labels.append(f"$B_h$ boundary ($d_h={config.capture.d_h}$" + (")" if drawn else ", outside range)"))
    ax.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_7.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_7.png (B_h invariant set)")


def fig8_phi_h_slices(vf_dir, output_dir, config):
    """Fig 8 — Phi_h value function slices: fix velocity and attacker position."""
    phi_h = load_value_function(vf_dir / "phi_h.npz")

    # Fix v_Dx=0, v_Dy=0, x_A=mid, y_A=mid → show (x_D, y_D)
    x_a_mid = (config.room.x_min + config.room.x_max) / 2
    y_a_mid = (config.room.y_min + config.room.y_max) / 2

    sliced, x_d_axis, y_d_axis = _slice_nd(phi_h, [2, 3, 4, 5], [0.0, 0.0, x_a_mid, y_a_mid])

    fig, ax = plt.subplots(figsize=(10, 6))
    cf = ax.contourf(x_d_axis, y_d_axis, sliced.T, levels=20, cmap="RdBu_r")
    ax.contour(x_d_axis, y_d_axis, sliced.T, levels=[0.0], colors="black", linewidths=2)
    plt.colorbar(cf, ax=ax, label=r"$\Phi_h$")

    # Overlay obstacles
    for obs in config.obstacles:
        rect = Rectangle((obs.x_min, obs.y_min), obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                         linewidth=2, edgecolor="red", facecolor="red", alpha=0.3)
        ax.add_patch(rect)

    # Target region
    tr = config.target_region
    rect_t = Rectangle((tr.x_min, tr.y_min), tr.x_max - tr.x_min, tr.y_max - tr.y_min,
                       linewidth=2, edgecolor="green", facecolor="green", alpha=0.3)
    ax.add_patch(rect_t)

    # Attacker position marker
    ax.plot(x_a_mid, y_a_mid, "r*", markersize=15, label=f"Attacker ({x_a_mid:.0f}, {y_a_mid:.0f})")

    ax.set_xlabel(r"$x_D$ (m)", fontsize=12)
    ax.set_ylabel(r"$y_D$ (m)", fontsize=12)
    ax.set_title(r"$\Phi_h$ at $v_{D}=0$, attacker at room center", fontsize=14)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_8.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_8.png (Phi_h slices)")


def fig10_winning_regions(vf_dir, output_dir, config):
    """Fig 10 — Winning regions: W_D and W_A for horizontal game."""
    phi_h = load_value_function(vf_dir / "phi_h.npz")

    # Slice at v_Dx=0, v_Dy=0, x_A=mid, y_A=mid
    x_a_mid = (config.room.x_min + config.room.x_max) / 2
    y_a_mid = (config.room.y_min + config.room.y_max) / 2

    sliced, x_d_axis, y_d_axis = _slice_nd(phi_h, [2, 3, 4, 5], [0.0, 0.0, x_a_mid, y_a_mid])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Paper horizontal convention: W_A,h = {Phi_h <= 0}, W_D,h = {Phi_h > 0}.
    w_d = (sliced > 0).astype(float)
    w_a = (sliced <= 0).astype(float)

    ax.contourf(x_d_axis, y_d_axis, w_d.T, levels=[0.5, 1.5], colors=["steelblue"], alpha=0.5)
    ax.contourf(x_d_axis, y_d_axis, w_a.T, levels=[0.5, 1.5], colors=["salmon"], alpha=0.5)
    ax.contour(x_d_axis, y_d_axis, sliced.T, levels=[0.0], colors="black", linewidths=2)

    # Overlay obstacles
    for obs in config.obstacles:
        rect = Rectangle((obs.x_min, obs.y_min), obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                         linewidth=2, edgecolor="red", facecolor="red", alpha=0.3)
        ax.add_patch(rect)

    # Target region
    tr = config.target_region
    rect_t = Rectangle((tr.x_min, tr.y_min), tr.x_max - tr.x_min, tr.y_max - tr.y_min,
                       linewidth=2, edgecolor="green", facecolor="green", alpha=0.3)
    ax.add_patch(rect_t)

    legend = [
        mpatches.Patch(facecolor="steelblue", alpha=0.5, label=r"$W_D$ (defender wins)"),
        mpatches.Patch(facecolor="salmon", alpha=0.5, label=r"$W_A$ (attacker wins)"),
        mpatches.Patch(facecolor="red", alpha=0.3, edgecolor="red", label="Obstacle"),
        mpatches.Patch(facecolor="green", alpha=0.3, edgecolor="green", label="Target"),
    ]
    ax.legend(handles=legend, loc="upper left")

    ax.set_xlabel(r"$x_D$ (m)", fontsize=12)
    ax.set_ylabel(r"$y_D$ (m)", fontsize=12)
    ax.set_title("Winning Regions (horizontal plane)", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_10.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_10.png (winning regions)")


def _load_numerical_sim():
    """Import run_combined_sim from numerical_sim script."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "numerical_sim", Path(__file__).parent / "numerical_sim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_combined_sim


def _parse_start_pose(value):
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"start pose must be formatted as x,y,z, got {value!r}"
        )
    return parts


def _run_combined_for_fig(config, vf_dir, sim_start=None):
    """Run the shared paper simulation with optional CLI-provided starts."""
    run_combined_sim = _load_numerical_sim()
    kwargs = {}
    if sim_start is not None:
        kwargs = {
            "initial_defender_pos": sim_start["defender"],
            "initial_attacker_pos": sim_start["attacker"],
        }
    return run_combined_sim(
        config,
        str(vf_dir),
        dt=0.01,
        T=SIMULATION_HORIZON_SECONDS,
        **kwargs,
    )


def _plot_top_view(ax, traj, config):
    ax.plot(traj["x_d"], traj["y_d"], "b-", linewidth=2, label="Defender")
    ax.plot(traj["x_a"], traj["y_a"], "r-", linewidth=2, label="Attacker")
    ax.plot(traj["x_d"][0], traj["y_d"][0], "bs", markersize=10)
    ax.plot(traj["x_a"][0], traj["y_a"][0], "rs", markersize=10)
    ax.plot(traj["x_d"][-1], traj["y_d"][-1], "bo", markersize=8)
    ax.plot(traj["x_a"][-1], traj["y_a"][-1], "ro", markersize=8)

    for obs in config.obstacles:
        rect = Rectangle(
            (obs.x_min, obs.y_min),
            obs.x_max - obs.x_min,
            obs.y_max - obs.y_min,
            linewidth=2,
            edgecolor="red",
            facecolor="red",
            alpha=0.2,
        )
        ax.add_patch(rect)
    tr = config.target_region
    rect_t = Rectangle(
        (tr.x_min, tr.y_min),
        tr.x_max - tr.x_min,
        tr.y_max - tr.y_min,
        linewidth=2,
        edgecolor="green",
        facecolor="green",
        alpha=0.2,
    )
    ax.add_patch(rect_t)

    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title("Top View (x-y)", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(config.room.x_min - 1, config.room.x_max + 1)
    ax.set_ylim(config.room.y_min - 1, config.room.y_max + 1)
    ax.set_aspect("equal", adjustable="box")


def _plot_side_view(ax, traj, config):
    ax.plot(traj["x_d"], traj["z_d"], "b-", linewidth=2, label="Defender")
    ax.plot(traj["x_a"], traj["z_a"], "r-", linewidth=2, label="Attacker")
    ax.plot(traj["x_d"][0], traj["z_d"][0], "bs", markersize=10)
    ax.plot(traj["x_a"][0], traj["z_a"][0], "rs", markersize=10)
    ax.plot(traj["x_d"][-1], traj["z_d"][-1], "bo", markersize=8)
    ax.plot(traj["x_a"][-1], traj["z_a"][-1], "ro", markersize=8)

    for obs in config.obstacles:
        rect = Rectangle(
            (obs.x_min, 0.0),
            obs.x_max - obs.x_min,
            config.room.z_max,
            linewidth=2,
            edgecolor="red",
            facecolor="red",
            alpha=0.15,
        )
        ax.add_patch(rect)
    tr = config.target_region
    rect_t = Rectangle(
        (tr.x_min, 0.0),
        tr.x_max - tr.x_min,
        0.25,
        linewidth=2,
        edgecolor="green",
        facecolor="green",
        alpha=0.2,
    )
    ax.add_patch(rect_t)

    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("z (m)", fontsize=12)
    ax.set_title("Side View (x-z)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(config.room.x_min - 1, config.room.x_max + 1)
    ax.set_ylim(config.room.z_min - 0.5, config.room.z_max + 0.5)


def fig11_simulation_trajectories(vf_dir, output_dir, config, sim_start=None):
    """Fig 11 — Simulation trajectories: top view, side view, and altitude timeline."""
    traj = _run_combined_for_fig(config, vf_dir, sim_start)
    if traj.get("obstacle_violation", False):
        raise ValueError("combined simulation trajectory enters an obstacle")

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    ax = axes[0]
    _plot_top_view(ax, traj, config)

    ax_side = axes[1]
    _plot_side_view(ax_side, traj, config)

    ax2 = axes[2]
    dt = traj["dt"]
    n_pts = len(traj["z_d"])
    t = np.arange(n_pts) * dt

    ax2.plot(t, traj["z_d"], "b-", linewidth=2, label=r"$z_D$")
    ax2.plot(t, traj["z_a"], "r-", linewidth=2, label=r"$z_A$")
    ax2.fill_between(t, traj["z_a"] - config.capture.d_z, traj["z_a"] + config.capture.d_z,
                     alpha=0.15, color="green", label=f"Capture zone ($d_z={config.capture.d_z}$m)")

    # Mode annotations
    mode_names = {0: "Reach", 1: "Track", 2: "PID"}
    mode_colors = {0: "orange", 1: "purple", 2: "cyan"}
    n_modes = len(traj["mode_z"])
    for i in range(min(n_modes - 1, n_pts - 2)):
        if traj["mode_z"][i] != traj["mode_z"][min(i + 1, n_modes - 1)]:
            mode_name = mode_names.get(int(traj["mode_z"][i + 1]), "?")
            ax2.axvline(x=t[i + 1], color=mode_colors.get(int(traj["mode_z"][i + 1]), "gray"),
                       alpha=0.3, linestyle="--")

    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Altitude (m)", fontsize=12)
    ax2.set_title("Vertical Trajectories", fontsize=13)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Combined Simulation — {traj['outcome']}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_11.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_11.png (simulation trajectories)")


def fig12_combined_3d(vf_dir, output_dir, config, sim_start=None):
    """Fig 12 — Combined 3D game view with companion top and side projections."""
    traj = _run_combined_for_fig(config, vf_dir, sim_start)
    if traj.get("obstacle_violation", False):
        raise ValueError("combined simulation trajectory enters an obstacle")

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], hspace=0.18, wspace=0.18)
    ax = fig.add_subplot(grid[:, 0], projection="3d")

    ax.plot(traj["x_d"], traj["y_d"], traj["z_d"], "b-", linewidth=2, label="Defender")
    ax.plot(traj["x_a"], traj["y_a"], traj["z_a"], "r-", linewidth=2, label="Attacker")

    # Start markers
    ax.scatter([traj["x_d"][0]], [traj["y_d"][0]], [traj["z_d"][0]], c="blue", s=100, marker="s")
    ax.scatter([traj["x_a"][0]], [traj["y_a"][0]], [traj["z_a"][0]], c="red", s=100, marker="s")

    # End markers
    ax.scatter([traj["x_d"][-1]], [traj["y_d"][-1]], [traj["z_d"][-1]], c="blue", s=80, marker="o")
    ax.scatter([traj["x_a"][-1]], [traj["y_a"][-1]], [traj["z_a"][-1]], c="red", s=80, marker="o")

    # Obstacle box (project as vertical walls)
    for obs in config.obstacles:
        xs = [obs.x_min, obs.x_max, obs.x_max, obs.x_min, obs.x_min]
        ys = [obs.y_min, obs.y_min, obs.y_max, obs.y_max, obs.y_min]
        ax.plot(xs, ys, [0] * 5, "r-", linewidth=1, alpha=0.5)
        ax.plot(xs, ys, [config.room.z_max] * 5, "r-", linewidth=1, alpha=0.5)
        for x, y in zip(xs[:-1], ys[:-1]):
            ax.plot([x, x], [y, y], [0, config.room.z_max], "r-", linewidth=0.5, alpha=0.3)

    # Target region
    tr = config.target_region
    xs_t = [tr.x_min, tr.x_max, tr.x_max, tr.x_min, tr.x_min]
    ys_t = [tr.y_min, tr.y_min, tr.y_max, tr.y_max, tr.y_min]
    ax.plot(xs_t, ys_t, [0] * 5, "g-", linewidth=2, alpha=0.5)

    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.set_zlabel("z (m)", fontsize=11)
    ax.set_title("3D Trajectory", fontsize=13)
    ax.legend(loc="upper left")
    ax.set_xlim(config.room.x_min, config.room.x_max)
    ax.set_ylim(config.room.y_min, config.room.y_max)
    ax.set_zlim(config.room.z_min, config.room.z_max)

    ax_top = fig.add_subplot(grid[0, 1])
    _plot_top_view(ax_top, traj, config)

    ax_side = fig.add_subplot(grid[1, 1])
    _plot_side_view(ax_side, traj, config)

    fig.suptitle(f"Game Trajectories — {traj['outcome']}", fontsize=14, y=0.98)
    fig.savefig(str(output_dir / "fig_12.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_12.png (3D game view)")


def fig_attacker_reaching(vf_dir, output_dir, config):
    """Extra figure — Attacker reaching value function."""
    phi_a = load_value_function(vf_dir / "phi_A_reach.npz")
    x_axis, y_axis = _build_axes(phi_a, [0, 1])

    fig, ax = plt.subplots(figsize=(10, 6))
    cf = ax.contourf(x_axis, y_axis, phi_a.values.T, levels=20, cmap="RdBu_r")
    ax.contour(x_axis, y_axis, phi_a.values.T, levels=[0.0], colors="black", linewidths=2)
    plt.colorbar(cf, ax=ax, label=r"$\phi_{A,reach}$")

    # Obstacles and target
    for obs in config.obstacles:
        rect = Rectangle((obs.x_min, obs.y_min), obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                         linewidth=2, edgecolor="red", facecolor="red", alpha=0.3)
        ax.add_patch(rect)
    tr = config.target_region
    rect_t = Rectangle((tr.x_min, tr.y_min), tr.x_max - tr.x_min, tr.y_max - tr.y_min,
                       linewidth=2, edgecolor="green", facecolor="green", alpha=0.3)
    ax.add_patch(rect_t)

    ax.set_xlabel(r"$x_A$ (m)", fontsize=12)
    ax.set_ylabel(r"$y_A$ (m)", fontsize=12)
    ax.set_title("Attacker Reaching Value Function", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_attacker_reaching.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_attacker_reaching.png")


def fig_vertical_winning_regions(vf_dir, output_dir, config):
    """Extra figure — Vertical winning regions from Phi_z."""
    phi_z = load_value_function(vf_dir / "phi_z.npz")

    # Slice at v_Dz=0
    n_v = phi_z.values.shape[1]
    v_idx = n_v // 2
    z_d_axis = np.linspace(float(phi_z.grid_min[0]), float(phi_z.grid_max[0]), phi_z.values.shape[0])
    z_a_axis = np.linspace(float(phi_z.grid_min[2]), float(phi_z.grid_max[2]), phi_z.values.shape[2])
    sliced = phi_z.values[:, v_idx, :]

    fig, ax = plt.subplots(figsize=(8, 6))
    w_d = (sliced <= 0).astype(float)
    w_a = (sliced > 0).astype(float)

    if w_d.any():
        ax.contourf(z_d_axis, z_a_axis, w_d.T, levels=[0.5, 1.5], colors=["steelblue"], alpha=0.5)
    ax.contourf(z_d_axis, z_a_axis, w_a.T, levels=[0.5, 1.5], colors=["salmon"], alpha=0.5)
    ax.contour(z_d_axis, z_a_axis, sliced.T, levels=[0.0], colors="black", linewidths=2)

    # Diagonal
    z_vals = np.linspace(0, 20, 100)
    ax.plot(z_vals, z_vals, "g--", alpha=0.5, label=r"$z_D = z_A$")

    legend = [
        mpatches.Patch(facecolor="steelblue", alpha=0.5, label=r"$W_{D,z}$"),
        mpatches.Patch(facecolor="salmon", alpha=0.5, label=r"$W_{A,z}$"),
    ]
    ax.legend(handles=legend, loc="upper left")
    ax.set_xlabel(r"$z_D$ (m)", fontsize=12)
    ax.set_ylabel(r"$z_A$ (m)", fontsize=12)
    ax.set_title(r"Vertical Winning Regions at $v_{D,z}=0$", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_vertical_winning.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_vertical_winning.png")


def fig_distance_over_time(vf_dir, output_dir, config, sim_start=None):
    """Analysis — 3D, horizontal, and vertical inter-agent distances vs time."""
    traj = _run_combined_for_fig(config, vf_dir, sim_start)

    dt = traj["dt"]
    n = len(traj["x_d"])
    t = np.arange(n) * dt

    h_dist = np.sqrt((traj["x_d"] - traj["x_a"])**2 + (traj["y_d"] - traj["y_a"])**2)
    z_dist = np.abs(traj["z_d"] - traj["z_a"])
    dist_3d = np.sqrt(h_dist**2 + z_dist**2)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t, dist_3d, "k-", linewidth=2, label="3D distance")
    axes[0].axhline(np.sqrt(config.capture.d_h**2 + config.capture.d_z**2),
                    color="green", linestyle="--", linewidth=1.5, label="Approx 3D capture threshold")
    axes[0].set_ylabel("Distance (m)", fontsize=11)
    axes[0].set_title("3D Euclidean Distance Between Agents", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, h_dist, "b-", linewidth=2, label="Horizontal distance")
    axes[1].axhline(config.capture.d_h, color="green", linestyle="--", linewidth=1.5,
                    label=f"$d_h = {config.capture.d_h}$m")
    axes[1].set_ylabel("Distance (m)", fontsize=11)
    axes[1].set_title("Horizontal Distance", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, z_dist, "r-", linewidth=2, label="Vertical distance")
    axes[2].axhline(config.capture.d_z, color="green", linestyle="--", linewidth=1.5,
                    label=f"$d_z = {config.capture.d_z}$m")
    axes[2].set_ylabel("Distance (m)", fontsize=11)
    axes[2].set_xlabel("Time (s)", fontsize=11)
    axes[2].set_title("Vertical Distance", fontsize=12)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Inter-Agent Distances (3D capture: {traj['captured_3d']})", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_distance_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_distance_over_time.png")


def fig_control_effort(vf_dir, output_dir, config, sim_start=None):
    """Analysis — Defender control inputs (u_x, u_y, u_z) and attacker disturbances over time."""
    traj = _run_combined_for_fig(config, vf_dir, sim_start)

    dt = traj["dt"]
    n = len(traj["u_x"])
    t = np.arange(n) * dt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ctrl_labels = [
        ("u_x", "Defender $u_x$", "blue"),
        ("u_y", "Defender $u_y$", "blue"),
        ("u_z", "Defender $u_z$", "blue"),
    ]
    dist_labels = [
        ("d_x", "Attacker $d_x$", "red"),
        ("d_y", "Attacker $d_y$", "red"),
        ("d_z_ctrl", "Attacker $d_z$", "red"),
    ]
    speed_limits = [
        config.defender.max_speed_horizontal,
        config.defender.max_speed_horizontal,
        config.defender.max_speed_vertical,
    ]
    att_limits = [
        config.attacker.max_speed_horizontal,
        config.attacker.max_speed_horizontal,
        config.attacker.max_speed_vertical,
    ]

    for col, ((ctrl_key, ctrl_lbl, _), (dist_key, dist_lbl, _), u_max, d_max) in enumerate(
        zip(ctrl_labels, dist_labels, speed_limits, att_limits)
    ):
        ax = axes[0, col]
        ax.plot(t, traj[ctrl_key], "b-", linewidth=1.5, label=ctrl_lbl)
        ax.axhline(u_max, color="gray", linestyle=":", linewidth=1)
        ax.axhline(-u_max, color="gray", linestyle=":", linewidth=1)
        ax.set_title(ctrl_lbl, fontsize=11)
        ax.set_ylabel("Speed (m/s)", fontsize=10)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax2 = axes[1, col]
        ax2.plot(t, traj[dist_key], "r-", linewidth=1.5, label=dist_lbl)
        ax2.axhline(d_max, color="gray", linestyle=":", linewidth=1)
        ax2.axhline(-d_max, color="gray", linestyle=":", linewidth=1)
        ax2.set_title(dist_lbl, fontsize=11)
        ax2.set_ylabel("Speed (m/s)", fontsize=10)
        ax2.set_xlabel("Time (s)", fontsize=10)
        ax2.grid(True, alpha=0.3)

    fig.suptitle("Control Inputs and Disturbances Over Time", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_control_effort.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_control_effort.png")


def fig_speed_profiles(vf_dir, output_dir, config, sim_start=None):
    """Analysis — Speed magnitude profiles for defender and attacker over time."""
    traj = _run_combined_for_fig(config, vf_dir, sim_start)

    dt = traj["dt"]
    n = len(traj["v_dx"])
    t = np.arange(n) * dt

    speed_d_h = np.sqrt(traj["v_dx"]**2 + traj["v_dy"]**2)
    speed_d_3d = np.sqrt(traj["v_dx"]**2 + traj["v_dy"]**2 + traj["v_dz"]**2)

    # Attacker speed from integrated disturbance (first-order)
    n_u = len(traj["d_x"])
    t_u = np.arange(n_u) * dt
    speed_a_h = np.sqrt(traj["d_x"]**2 + traj["d_y"]**2)
    speed_a_3d = np.sqrt(traj["d_x"]**2 + traj["d_y"]**2 + traj["d_z_ctrl"]**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t, speed_d_h, "b-", linewidth=2, label="Defender horizontal")
    axes[0].plot(t, np.abs(traj["v_dz"]), "b--", linewidth=1.5, label="Defender vertical")
    axes[0].plot(t_u, speed_a_h, "r-", linewidth=2, label="Attacker horizontal cmd")
    axes[0].plot(t_u, np.abs(traj["d_z_ctrl"]), "r--", linewidth=1.5, label="Attacker vertical cmd")
    axes[0].axhline(config.defender.max_speed_horizontal, color="blue", linestyle=":", alpha=0.5,
                    label=f"Defender $u_{{h,max}}$={config.defender.max_speed_horizontal}")
    axes[0].axhline(config.attacker.max_speed_horizontal, color="red", linestyle=":", alpha=0.5,
                    label=f"Attacker $u_{{h,max}}$={config.attacker.max_speed_horizontal}")
    axes[0].set_xlabel("Time (s)", fontsize=11)
    axes[0].set_ylabel("Speed (m/s)", fontsize=11)
    axes[0].set_title("Component Speed Profiles", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, speed_d_3d, "b-", linewidth=2, label="Defender 3D speed")
    axes[1].plot(t_u, speed_a_3d, "r-", linewidth=2, label="Attacker 3D speed")
    axes[1].set_xlabel("Time (s)", fontsize=11)
    axes[1].set_ylabel("Speed (m/s)", fontsize=11)
    axes[1].set_title("3D Speed Magnitudes", fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Agent Speed Profiles", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_speed_profiles.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_speed_profiles.png")


def fig_mode_timeline(vf_dir, output_dir, config, sim_start=None):
    """Analysis — Color-coded control mode timeline for horizontal and vertical sub-games."""
    traj = _run_combined_for_fig(config, vf_dir, sim_start)

    dt = traj["dt"]
    n = len(traj["mode_z"])
    t = np.arange(n) * dt

    mode_colors = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}
    mode_names = {0: "Reach (Phi)", 1: "Track (V_h_T)", 2: "PID"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

    for ax, mode_key, title in [
        (axes[0], "mode_z", "Vertical Control Mode"),
        (axes[1], "mode_h", "Horizontal Control Mode"),
    ]:
        modes = traj[mode_key]
        for i in range(n):
            color = mode_colors.get(int(modes[i]), "gray")
            ax.axvspan(t[i], t[i] + dt, alpha=0.8, color=color, linewidth=0)
        ax.set_yticks([])
        ax.set_title(title, fontsize=12)
        for t_val in np.diff(modes).nonzero()[0]:
            ax.axvline(x=t[t_val + 1], color="black", linewidth=0.8, alpha=0.5)

    axes[1].set_xlabel("Time (s)", fontsize=11)

    legend_patches = [mpatches.Patch(color=mode_colors[k], label=f"{k}: {v}")
                      for k, v in mode_names.items()]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=10,
               bbox_to_anchor=(0.99, 0.95))

    fig.suptitle("Control Mode Transitions Over Time", fontsize=14)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_mode_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_mode_timeline.png")


def fig_altitude_phase_portrait(vf_dir, output_dir, config, sim_start=None):
    """Analysis — z_D vs z_A phase portrait showing altitude trajectory and capture zone."""
    traj = _run_combined_for_fig(config, vf_dir, sim_start)

    z_d = traj["z_d"]
    z_a = traj["z_a"]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Color trajectory by time
    n = len(z_d)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    for i in range(n - 1):
        ax.plot(z_d[i:i+2], z_a[i:i+2], color=colors[i], linewidth=1.5)

    # Capture band: |z_D - z_A| <= d_z
    z_vals = np.linspace(config.room.z_min, config.room.z_max, 200)
    ax.fill_between(z_vals, z_vals - config.capture.d_z, z_vals + config.capture.d_z,
                    alpha=0.2, color="green", label=f"Capture zone ($d_z={config.capture.d_z}$m)")
    ax.plot(z_vals, z_vals, "g--", linewidth=1.5, alpha=0.6, label=r"$z_D = z_A$")

    # Start and end markers
    ax.scatter([z_d[0]], [z_a[0]], c="blue", s=120, zorder=5, marker="s", label="Start")
    ax.scatter([z_d[-1]], [z_a[-1]], c="red", s=120, zorder=5, marker="o", label="End")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, traj["T"]))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Time (s)")

    ax.set_xlabel(r"$z_D$ (m)", fontsize=12)
    ax.set_ylabel(r"$z_A$ (m)", fontsize=12)
    ax.set_title("Altitude Phase Portrait", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_altitude_phase_portrait.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_altitude_phase_portrait.png")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output-dir", default="/workspace/data/plots/paper_figures/",
                        help="Output directory for figures")
    parser.add_argument("--vf-dir", default="/workspace/data/value_functions/",
                        help="Directory containing value function files")
    parser.add_argument("--config", default="/workspace/config/game_params.yaml",
                        help="Path to game configuration YAML")
    parser.add_argument("--preset", default="dev", help="Preset name (for info only)")
    parser.add_argument("--defender-start", type=_parse_start_pose, default=None,
                        help="Initial defender pose for simulation figures as x,y,z")
    parser.add_argument("--attacker-start", type=_parse_start_pose, default=None,
                        help="Initial attacker pose for simulation figures as x,y,z")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vf_dir = Path(args.vf_dir)
    config = GameConfig.from_yaml(args.config)
    config.apply_preset(args.preset)
    sim_start = None
    if args.defender_start is not None or args.attacker_start is not None:
        sim_start = {
            "defender": args.defender_start or [35.0, 20.0, 8.0],
            "attacker": args.attacker_start or [5.0, 3.0, 3.0],
        }

    print(f"Generating paper figures from {vf_dir}")
    print(f"Output directory: {output_dir}")
    if sim_start is not None:
        print(
            "Simulation starts: "
            f"defender={sim_start['defender']}, attacker={sim_start['attacker']}"
        )
    print()

    # Check prerequisites
    required_files = ["V_z_inf.npz", "B_z.npz", "phi_z.npz", "V_h_T.npz", "B_h.npz",
                     "phi_h.npz", "phi_A_reach.npz"]
    missing = [f for f in required_files if not (vf_dir / f).exists()]
    if missing:
        print(f"WARNING: Missing value function files: {missing}")
        print("Some figures may be skipped.")
        print()

    # Generate each figure with error handling
    figure_funcs = [
        ("Fig 4", fig4_bz_invariant_set, ["V_z_inf.npz", "B_z.npz"]),
        ("Fig 5", fig5_phi_z_slices, ["phi_z.npz"]),
        ("Fig 6", fig6_vertical_3d, ["phi_z.npz", "V_z_inf.npz"]),
        ("Fig 7", fig7_bh_invariant_set, ["V_h_T_6d.npz", "B_h.npz"]),
        ("Fig 8", fig8_phi_h_slices, ["phi_h.npz"]),
        ("Fig 10", fig10_winning_regions, ["phi_h.npz"]),
        ("Fig 11", fig11_simulation_trajectories, ["phi_z.npz", "V_z_inf.npz", "B_z.npz",
                                                    "phi_h.npz", "V_h_T_6d.npz"]),
        ("Fig 12", fig12_combined_3d, ["phi_z.npz", "V_z_inf.npz", "B_z.npz",
                                       "phi_h.npz", "V_h_T_6d.npz"]),
        ("Attacker Reaching", fig_attacker_reaching, ["phi_A_reach.npz"]),
        ("Vertical Winning", fig_vertical_winning_regions, ["phi_z.npz"]),
        ("Distance Over Time", fig_distance_over_time,
         ["phi_z.npz", "V_z_inf.npz", "B_z.npz", "phi_h.npz", "V_h_T_6d.npz"]),
        ("Control Effort", fig_control_effort,
         ["phi_z.npz", "V_z_inf.npz", "B_z.npz", "phi_h.npz", "V_h_T_6d.npz"]),
        ("Speed Profiles", fig_speed_profiles,
         ["phi_z.npz", "V_z_inf.npz", "B_z.npz", "phi_h.npz", "V_h_T_6d.npz"]),
        ("Mode Timeline", fig_mode_timeline,
         ["phi_z.npz", "V_z_inf.npz", "B_z.npz", "phi_h.npz", "V_h_T_6d.npz"]),
        ("Altitude Phase Portrait", fig_altitude_phase_portrait,
         ["phi_z.npz", "V_z_inf.npz", "B_z.npz", "phi_h.npz", "V_h_T_6d.npz"]),
    ]
    simulation_figures = {
        "Fig 11",
        "Fig 12",
        "Distance Over Time",
        "Control Effort",
        "Speed Profiles",
        "Mode Timeline",
        "Altitude Phase Portrait",
    }

    generated = 0
    for name, func, deps in figure_funcs:
        dep_missing = [d for d in deps if not (vf_dir / d).exists()]
        if dep_missing:
            print(f"  Skipping {name}: missing {dep_missing}")
            _diagnostic_figure(
                output_dir, FIGURE_OUTPUTS[name], f"{name} skipped",
                f"Missing value-function artifacts: {', '.join(dep_missing)}",
            )
            continue
        invalid = _invalid_paper_deps(vf_dir, deps)
        if invalid:
            print(f"  Skipping {name}: invalid paper artifacts {invalid}")
            _diagnostic_figure(
                output_dir, FIGURE_OUTPUTS[name], f"{name} skipped",
                f"Invalid or stale paper artifacts: {', '.join(invalid)}",
            )
            continue
        try:
            if name in simulation_figures:
                func(vf_dir, output_dir, config, sim_start)
            else:
                func(vf_dir, output_dir, config)
            generated += 1
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")

    print()
    print(f"Generated {generated} figures in {output_dir}")


if __name__ == "__main__":
    main()
