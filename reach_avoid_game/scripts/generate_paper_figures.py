"""Generate paper figures (Bui et al., arXiv:2512.22793) from computed value functions.

Reproduces Figures 4-12 from the paper using the computed value functions.
Each figure function loads the required data and saves a publication-quality PNG.
"""

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
from reach_avoid_game.solvers.value_function_io import load_value_function


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


def fig4_bz_invariant_set(vf_dir, output_dir, config):
    """Fig 4 — B_z invariant set: 2D contour of V_z_inf with B_z boundary."""
    v_z_inf = load_value_function(vf_dir / "V_z_inf.npz")

    z_rel_axis, v_dz_axis = _build_axes(v_z_inf, [0, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(z_rel_axis, v_dz_axis, v_z_inf.values.T, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax, label=r"$V_{z,\infty}$ (m)")

    # B_z boundary: V_z_inf = d_z (or d_z_effective)
    b_z = load_value_function(vf_dir / "B_z.npz")
    d_z_eff = b_z.params.get("d_z_effective", config.capture.d_z) if isinstance(b_z.params, dict) else config.capture.d_z
    ax.contour(z_rel_axis, v_dz_axis, v_z_inf.values.T, levels=[d_z_eff],
               colors="black", linewidths=2.5, linestyles="solid")
    ax.contour(z_rel_axis, v_dz_axis, v_z_inf.values.T, levels=[config.capture.d_z],
               colors="red", linewidths=1.5, linestyles="dashed")

    ax.set_xlabel(r"$z_{rel}$ (m)", fontsize=12)
    ax.set_ylabel(r"$v_{D,z}$ (m/s)", fontsize=12)
    ax.set_title(r"Vertical Invariant Capture Set $B_z$", fontsize=14)

    legend = [
        plt.Line2D([0], [0], color="black", linewidth=2.5, label=f"$B_z$ boundary ($d_z^{{eff}}={d_z_eff:.3f}$)"),
        plt.Line2D([0], [0], color="red", linewidth=1.5, linestyle="dashed", label=f"$d_z={config.capture.d_z}$"),
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
    axes[1].contour(z_rel_axis, v_dz_axis, v_z_inf.values.T, levels=[config.capture.d_z],
                    colors="red", linewidths=2, linestyles="dashed")
    axes[1].set_xlabel(r"$z_{rel}$ (m)", fontsize=12)
    axes[1].set_ylabel(r"$v_{D,z}$ (m/s)", fontsize=12)
    axes[1].set_title(r"$V_{z,\infty}$ with $B_z$ boundary", fontsize=13)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Vertical Sub-Game Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_6.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_6.png (vertical 3D view)")


def fig7_bh_invariant_set(vf_dir, output_dir, config):
    """Fig 7 — B_h invariant set: 2D contour of V_h_T at v_Dx=0, v_Dy=0."""
    v_h_t = load_value_function(vf_dir / "V_h_T.npz")

    # Slice at v_Dx=0, v_Dy=0 (dims 2, 3)
    sliced, x_rel_axis, y_rel_axis = _slice_nd(v_h_t, [2, 3], [0.0, 0.0])

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(x_rel_axis, y_rel_axis, sliced.T, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax, label=r"$V_{h,T}$ (m)")

    # B_h boundary
    b_h = load_value_function(vf_dir / "B_h.npz")
    d_h_eff = b_h.params.get("d_h_effective", config.capture.d_h) if isinstance(b_h.params, dict) else config.capture.d_h
    ax.contour(x_rel_axis, y_rel_axis, sliced.T, levels=[d_h_eff],
               colors="black", linewidths=2.5)
    ax.contour(x_rel_axis, y_rel_axis, sliced.T, levels=[config.capture.d_h],
               colors="red", linewidths=1.5, linestyles="dashed")

    # d_h circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(config.capture.d_h * np.cos(theta), config.capture.d_h * np.sin(theta),
            "w--", linewidth=1, alpha=0.5, label=f"$d_h={config.capture.d_h}$m circle")

    ax.set_xlabel(r"$x_{rel}$ (m)", fontsize=12)
    ax.set_ylabel(r"$y_{rel}$ (m)", fontsize=12)
    ax.set_title(r"Horizontal Invariant Capture Set $B_h$ ($v_{D,x}=0, v_{D,y}=0$)", fontsize=14)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

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

    w_d = (sliced <= 0).astype(float)
    w_a = (sliced > 0).astype(float)

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


def fig11_simulation_trajectories(vf_dir, output_dir, config):
    """Fig 11 — Simulation trajectories: x-y and z-t side by side."""
    run_combined_sim = _load_numerical_sim()

    traj = run_combined_sim(config, str(vf_dir), dt=0.01, T=10.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: x-y trajectories
    ax = axes[0]
    ax.plot(traj["x_d"], traj["y_d"], "b-", linewidth=2, label="Defender")
    ax.plot(traj["x_a"], traj["y_a"], "r-", linewidth=2, label="Attacker")
    ax.plot(traj["x_d"][0], traj["y_d"][0], "bs", markersize=10)
    ax.plot(traj["x_a"][0], traj["y_a"][0], "rs", markersize=10)
    ax.plot(traj["x_d"][-1], traj["y_d"][-1], "bo", markersize=8)
    ax.plot(traj["x_a"][-1], traj["y_a"][-1], "ro", markersize=8)

    # Obstacles and target
    for obs in config.obstacles:
        rect = Rectangle((obs.x_min, obs.y_min), obs.x_max - obs.x_min, obs.y_max - obs.y_min,
                         linewidth=2, edgecolor="red", facecolor="red", alpha=0.2)
        ax.add_patch(rect)
    tr = config.target_region
    rect_t = Rectangle((tr.x_min, tr.y_min), tr.x_max - tr.x_min, tr.y_max - tr.y_min,
                       linewidth=2, edgecolor="green", facecolor="green", alpha=0.2)
    ax.add_patch(rect_t)

    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.set_title("Horizontal Trajectories", fontsize=13)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(config.room.x_min - 1, config.room.x_max + 1)
    ax.set_ylim(config.room.y_min - 1, config.room.y_max + 1)

    # Right: z vs t
    ax2 = axes[1]
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

    fig.suptitle(f"Combined Simulation (3D capture: {traj['captured_3d']})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_dir / "fig_11.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_11.png (simulation trajectories)")


def fig12_combined_3d(vf_dir, output_dir, config):
    """Fig 12 — Combined 3D game view: 3D plot of both drone paths."""
    run_combined_sim = _load_numerical_sim()

    traj = run_combined_sim(config, str(vf_dir), dt=0.01, T=10.0)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

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
    ax.set_title("3D Game Trajectories", fontsize=14)
    ax.legend(loc="upper left")

    fig.tight_layout()
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


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--output-dir", default="/workspace/data/plots/paper_figures/",
                        help="Output directory for figures")
    parser.add_argument("--vf-dir", default="/workspace/data/value_functions/",
                        help="Directory containing value function files")
    parser.add_argument("--config", default="/workspace/config/game_params.yaml",
                        help="Path to game configuration YAML")
    parser.add_argument("--preset", default="dev", help="Preset name (for info only)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vf_dir = Path(args.vf_dir)
    config = GameConfig.from_yaml(args.config)

    print(f"Generating paper figures from {vf_dir}")
    print(f"Output directory: {output_dir}")
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
        ("Fig 7", fig7_bh_invariant_set, ["V_h_T.npz", "B_h.npz"]),
        ("Fig 8", fig8_phi_h_slices, ["phi_h.npz"]),
        ("Fig 10", fig10_winning_regions, ["phi_h.npz"]),
        ("Fig 11", fig11_simulation_trajectories, ["phi_z.npz", "V_z_inf.npz", "B_z.npz",
                                                    "phi_h.npz", "V_h_T.npz", "B_h.npz"]),
        ("Fig 12", fig12_combined_3d, ["phi_z.npz", "V_z_inf.npz", "B_z.npz",
                                       "phi_h.npz", "V_h_T.npz", "B_h.npz"]),
        ("Attacker Reaching", fig_attacker_reaching, ["phi_A_reach.npz"]),
        ("Vertical Winning", fig_vertical_winning_regions, ["phi_z.npz"]),
    ]

    generated = 0
    for name, func, deps in figure_funcs:
        dep_missing = [d for d in deps if not (vf_dir / d).exists()]
        if dep_missing:
            print(f"  Skipping {name}: missing {dep_missing}")
            continue
        try:
            func(vf_dir, output_dir, config)
            generated += 1
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")

    print()
    print(f"Generated {generated} figures in {output_dir}")


if __name__ == "__main__":
    main()
