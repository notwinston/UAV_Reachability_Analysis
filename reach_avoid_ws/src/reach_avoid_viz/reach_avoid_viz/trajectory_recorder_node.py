"""Record full-game trajectories and save a 3D matplotlib plot on shutdown."""

from __future__ import annotations

import os
import signal
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


DEFAULT_TARGET = {"x_min": 38.0, "x_max": 45.0, "y_min": 10.0, "y_max": 15.0}
DEFAULT_OBSTACLES = [
    {"x_min": 15.0, "x_max": 20.0, "y_min": 5.0, "y_max": 20.0, "z_min": 0.0, "z_max": 20.0},
]
DEFAULT_D_H = 3.0
DEFAULT_D_Z = 1.0


def _nearest_time_pairs(defender_samples, attacker_samples):
    """Pair samples by nearest timestamp instead of list index.

    The defender and attacker callbacks are not guaranteed to interleave at the
    same rate, so zip()-pairing can miss true capture events and distort the
    minimum-distance summary.
    """
    if not defender_samples or not attacker_samples:
        return []

    pairs = []
    j = 0
    for defender in defender_samples:
        t_def = defender[0]
        while (
            j + 1 < len(attacker_samples)
            and abs(attacker_samples[j + 1][0] - t_def) <= abs(attacker_samples[j][0] - t_def)
        ):
            j += 1
        pairs.append((defender, attacker_samples[j]))
    return pairs


def _load_game_params(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _ensure_obstacle_height(obs: dict, default_z_max: float = 20.0) -> dict:
    result = dict(obs)
    result.setdefault("z_min", 0.0)
    result.setdefault("z_max", default_z_max)
    return result


def main(args=None):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from std_msgs.msg import String

        class TrajectoryRecorderNode(Node):
            """Records defender and attacker state samples and writes a 3D PNG."""

            def __init__(self):
                super().__init__("trajectory_recorder")

                self.declare_parameter(
                    "game_params_file", "/workspace/config/game_params.yaml"
                )
                self.declare_parameter(
                    "output_dir", "/workspaces/UAV_Reachability_Analysis/data/plots/gazebo_runs"
                )
                self.declare_parameter("output_name", "")
                self.declare_parameter("sample_stride", 1)
                self.declare_parameter("autosave_period_sec", 10.0)
                self.declare_parameter("capture_distance_horizontal", -1.0)
                self.declare_parameter("capture_distance_vertical", -1.0)

                self._game_params_file = self.get_parameter("game_params_file").value
                self._output_dir = Path(self.get_parameter("output_dir").value)
                self._output_name = self.get_parameter("output_name").value
                self._sample_stride = max(1, int(self.get_parameter("sample_stride").value))
                self._autosave_period_sec = max(
                    1.0, float(self.get_parameter("autosave_period_sec").value)
                )
                self._defender = []
                self._attacker = []
                self._defender_count = 0
                self._attacker_count = 0
                self._saved = False
                self._autosave_count = 0
                self._latest_defender = None
                self._latest_attacker = None
                self._terminal_status = ""
                self._terminal_time = None
                self._shutdown_timer = None
                self._output_dir.mkdir(parents=True, exist_ok=True)

                gp = _load_game_params(self._game_params_file)
                self._room = gp.get("room", {})
                self._target = gp.get("target_region", DEFAULT_TARGET)
                capture = gp.get("capture", {})
                self._d_h = float(capture.get("d_h", DEFAULT_D_H))
                self._d_z = float(capture.get("d_z", DEFAULT_D_Z))
                capture_d_h = float(self.get_parameter("capture_distance_horizontal").value)
                capture_d_z = float(self.get_parameter("capture_distance_vertical").value)
                if capture_d_h > 0.0:
                    self._d_h = capture_d_h
                if capture_d_z > 0.0:
                    self._d_z = capture_d_z
                self._obstacles = [
                    _ensure_obstacle_height(obs, self._room.get("z_max", 20.0))
                    for obs in gp.get("obstacles", DEFAULT_OBSTACLES)
                ]

                self.create_subscription(PoseStamped, "/defender/state", self._defender_cb, 10)
                self.create_subscription(PoseStamped, "/attacker/state", self._attacker_cb, 10)
                self.create_subscription(String, "/game/status", self._status_cb, 10)
                self._status_pub = self.create_publisher(String, "/game/status", 10)
                self.create_timer(self._autosave_period_sec, self._autosave_plot)

                signal.signal(signal.SIGTERM, self._handle_signal)
                signal.signal(signal.SIGINT, self._handle_signal)

                self.get_logger().info(
                    f"Trajectory recorder started, output_dir={self._output_dir}"
                )

            def _defender_cb(self, msg: PoseStamped):
                self._latest_defender = self._sample(msg)
                self._defender_count += 1
                if self._defender_count % self._sample_stride == 0:
                    self._defender.append(self._latest_defender)
                self._maybe_finish_from_state()

            def _attacker_cb(self, msg: PoseStamped):
                self._latest_attacker = self._sample(msg)
                self._attacker_count += 1
                if self._attacker_count % self._sample_stride == 0:
                    self._attacker.append(self._latest_attacker)
                self._maybe_finish_from_state()

            def _status_cb(self, msg: String):
                status = msg.data.split("|", 1)[0].strip()
                if status in ("CAPTURED", "ATTACKER_REACHED_TARGET"):
                    self._finish_game(status)

            def _sample(self, msg: PoseStamped):
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                return (
                    stamp,
                    float(msg.pose.position.x),
                    float(msg.pose.position.y),
                    float(msg.pose.position.z),
                )

            def _maybe_finish_from_state(self):
                if self._terminal_status:
                    return
                if self._latest_defender is None or self._latest_attacker is None:
                    return
                d = self._latest_defender
                a = self._latest_attacker
                h_dist = float(np.hypot(d[1] - a[1], d[2] - a[2]))
                z_dist = abs(d[3] - a[3])
                if h_dist <= self._d_h and z_dist <= self._d_z:
                    self._finish_game("CAPTURED", max(d[0], a[0]))
                    return
                if self._point_in_target(a):
                    self._finish_game("ATTACKER_REACHED_TARGET", a[0])

            def _point_in_target(self, sample):
                _, x, y, _ = sample
                target = self._target
                return (
                    target.get("x_min", 38.0) <= x <= target.get("x_max", 45.0)
                    and target.get("y_min", 10.0) <= y <= target.get("y_max", 15.0)
                )

            def _finish_game(self, status, event_time=None):
                if self._terminal_status:
                    return
                self._terminal_status = status
                self._terminal_time = event_time
                msg = String()
                msg.data = status
                self._status_pub.publish(msg)
                self.get_logger().info(
                    f"Terminal game status {status}"
                    + (
                        f" at t={event_time:.2f}s"
                        if event_time is not None
                        else ""
                    )
                )
                self.save_plot()
                self._schedule_shutdown()

            def _schedule_shutdown(self):
                if self._shutdown_timer is not None:
                    return
                self._shutdown_timer = self.create_timer(0.5, self._shutdown_launch)

            def _shutdown_launch(self):
                if self._shutdown_timer is not None:
                    self._shutdown_timer.cancel()
                    self._shutdown_timer = None
                self.save_plot()
                try:
                    os.kill(os.getppid(), signal.SIGINT)
                except ProcessLookupError:
                    pass

            def _handle_signal(self, signum, frame):
                self.save_plot()
                raise SystemExit(0)

            def _autosave_plot(self):
                if not self._defender and not self._attacker:
                    return
                self._autosave_count += 1
                output_path = self._output_dir / "full_game_trajectory_latest.png"
                self._write_plot(output_path)
                self._write_data_sidecars("full_game_trajectory_latest")
                self.get_logger().info(
                    f"Autosaved trajectory views to {output_path} "
                    f"(defender_samples={len(self._defender)}, attacker_samples={len(self._attacker)})"
                )

            def save_plot(self):
                if self._saved:
                    return
                self._saved = True
                if not self._defender and not self._attacker:
                    self.get_logger().warn("No trajectory samples recorded; skipping plot")
                    return

                if self._output_name:
                    output_path = self._output_dir / self._output_name
                else:
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    output_path = self._output_dir / f"full_game_trajectory_{stamp}.png"

                self._write_plot(output_path)
                self._write_data_sidecars(output_path.stem)
                latest_path = self._output_dir / "full_game_trajectory_latest.png"
                if latest_path != output_path:
                    self._write_plot(latest_path)
                    self._write_data_sidecars("full_game_trajectory_latest")
                self.get_logger().info(
                    f"Saved trajectory views to {output_path} "
                    f"(defender_samples={len(self._defender)}, attacker_samples={len(self._attacker)})"
                )

            def _write_data_sidecars(self, stem):
                csv_path = self._output_dir / f"{stem}.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["vehicle", "stamp", "x", "y", "z"])
                    for sample in self._defender:
                        writer.writerow(["defender", *sample])
                    for sample in self._attacker:
                        writer.writerow(["attacker", *sample])

                summary_path = self._output_dir / f"{stem}_summary.json"
                summary = self._compute_summary()
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, sort_keys=True)

            def _compute_summary(self):
                def inside_room(sample):
                    _, x, y, z = sample
                    room = self._room
                    return (
                        room.get("x_min", 0.0) <= x <= room.get("x_max", 45.0)
                        and room.get("y_min", 0.0) <= y <= room.get("y_max", 25.0)
                        and room.get("z_min", 0.0) <= z <= room.get("z_max", 20.0)
                    )

                def inside_obstacle(sample):
                    _, x, y, z = sample
                    for obs in self._obstacles:
                        if (
                            obs.get("x_min", 15.0) <= x <= obs.get("x_max", 20.0)
                            and obs.get("y_min", 5.0) <= y <= obs.get("y_max", 20.0)
                            and obs.get("z_min", 0.0) <= z <= obs.get("z_max", 20.0)
                        ):
                            return True
                    return False

                def in_target(sample):
                    _, x, y, _ = sample
                    target = self._target
                    return (
                        target.get("x_min", 38.0) <= x <= target.get("x_max", 45.0)
                        and target.get("y_min", 10.0) <= y <= target.get("y_max", 15.0)
                    )

                paired = _nearest_time_pairs(self._defender, self._attacker)
                min_horizontal = None
                min_vertical = None
                capture_count = 0
                for d, a in paired:
                    h = float(np.hypot(d[1] - a[1], d[2] - a[2]))
                    z = abs(d[3] - a[3])
                    min_horizontal = h if min_horizontal is None else min(min_horizontal, h)
                    min_vertical = z if min_vertical is None else min(min_vertical, z)
                    if h <= self._d_h and z <= self._d_z:
                        capture_count += 1

                all_samples = self._defender + self._attacker
                return {
                    "defender_samples": len(self._defender),
                    "attacker_samples": len(self._attacker),
                    "min_horizontal_distance": min_horizontal,
                    "min_vertical_distance": min_vertical,
                    "capture_samples": capture_count,
                    "attacker_target_samples": sum(1 for s in self._attacker if in_target(s)),
                    "defender_obstacle_samples": sum(1 for s in self._defender if inside_obstacle(s)),
                    "attacker_obstacle_samples": sum(1 for s in self._attacker if inside_obstacle(s)),
                    "outside_room_samples": sum(1 for s in all_samples if not inside_room(s)),
                    "terminal_status": self._terminal_status or None,
                    "terminal_time": self._terminal_time,
                }

            def _write_plot(self, output_path):
                fig = plt.figure(figsize=(16, 10), constrained_layout=True)
                grid = fig.add_gridspec(2, 2, width_ratios=[1.6, 1.0], hspace=0.18, wspace=0.18)

                ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
                self._plot_geometry_3d(ax_3d)
                self._plot_path_3d(ax_3d, self._defender, "Defender", "#2563eb")
                self._plot_path_3d(ax_3d, self._attacker, "Attacker", "#dc2626")
                self._set_axes_3d(ax_3d)
                ax_3d.legend(loc="upper left")
                ax_3d.set_title("3D Trajectory")

                ax_top = fig.add_subplot(grid[0, 1])
                self._plot_geometry_top(ax_top)
                self._plot_path_2d(ax_top, self._defender, "Defender", "#2563eb", 1, 2)
                self._plot_path_2d(ax_top, self._attacker, "Attacker", "#dc2626", 1, 2)
                self._set_axes_top(ax_top)
                ax_top.set_title("Top View (x-y)")

                ax_side = fig.add_subplot(grid[1, 1])
                self._plot_geometry_side(ax_side)
                self._plot_path_2d(ax_side, self._defender, "Defender", "#2563eb", 1, 3)
                self._plot_path_2d(ax_side, self._attacker, "Attacker", "#dc2626", 1, 3)
                self._set_axes_side(ax_side)
                ax_side.set_title("Side View (x-z)")

                terminal = self._terminal_status or "IN_PROGRESS"
                fig.suptitle(f"Full Game Gazebo Trajectory — {terminal}", fontsize=14, y=0.98)
                fig.savefig(output_path, dpi=180)
                plt.close(fig)

            def _plot_path_3d(self, ax, samples, label, color):
                if not samples:
                    return
                arr = np.asarray(samples, dtype=float)
                ax.plot(arr[:, 1], arr[:, 2], arr[:, 3], color=color, linewidth=2.0, label=label)
                ax.scatter(arr[0, 1], arr[0, 2], arr[0, 3], color=color, marker="o", s=35)
                ax.scatter(arr[-1, 1], arr[-1, 2], arr[-1, 3], color=color, marker="x", s=55)

            def _plot_path_2d(self, ax, samples, label, color, x_idx, y_idx):
                if not samples:
                    return
                arr = np.asarray(samples, dtype=float)
                ax.plot(arr[:, x_idx], arr[:, y_idx], color=color, linewidth=2.0, label=label)
                ax.scatter(arr[0, x_idx], arr[0, y_idx], color=color, marker="o", s=35)
                ax.scatter(arr[-1, x_idx], arr[-1, y_idx], color=color, marker="x", s=55)

            def _plot_geometry_3d(self, ax):
                target = self._target
                self._plot_box(
                    ax,
                    target.get("x_min", 38.0),
                    target.get("x_max", 45.0),
                    target.get("y_min", 10.0),
                    target.get("y_max", 15.0),
                    0.0,
                    0.2,
                    color="#22c55e",
                    alpha=0.18,
                )
                for obs in self._obstacles:
                    self._plot_box(
                        ax,
                        obs.get("x_min", 15.0),
                        obs.get("x_max", 20.0),
                        obs.get("y_min", 5.0),
                        obs.get("y_max", 20.0),
                        obs.get("z_min", 0.0),
                        obs.get("z_max", self._room.get("z_max", 20.0)),
                        color="#6b7280",
                        alpha=0.16,
                    )

            def _plot_geometry_top(self, ax):
                import matplotlib.patches as mpatches

                for obs in self._obstacles:
                    rect = mpatches.Rectangle(
                        (obs.get("x_min", 15.0), obs.get("y_min", 5.0)),
                        obs.get("x_max", 20.0) - obs.get("x_min", 15.0),
                        obs.get("y_max", 20.0) - obs.get("y_min", 5.0),
                        linewidth=1.5,
                        edgecolor="#6b7280",
                        facecolor="#6b7280",
                        alpha=0.18,
                    )
                    ax.add_patch(rect)
                target = self._target
                rect = mpatches.Rectangle(
                    (target.get("x_min", 38.0), target.get("y_min", 10.0)),
                    target.get("x_max", 45.0) - target.get("x_min", 38.0),
                    target.get("y_max", 15.0) - target.get("y_min", 10.0),
                    linewidth=1.5,
                    edgecolor="#22c55e",
                    facecolor="#22c55e",
                    alpha=0.18,
                )
                ax.add_patch(rect)

            def _plot_geometry_side(self, ax):
                import matplotlib.patches as mpatches

                room_z_max = self._room.get("z_max", 20.0)
                for obs in self._obstacles:
                    rect = mpatches.Rectangle(
                        (obs.get("x_min", 15.0), obs.get("z_min", 0.0)),
                        obs.get("x_max", 20.0) - obs.get("x_min", 15.0),
                        obs.get("z_max", room_z_max) - obs.get("z_min", 0.0),
                        linewidth=1.5,
                        edgecolor="#6b7280",
                        facecolor="#6b7280",
                        alpha=0.18,
                    )
                    ax.add_patch(rect)
                target = self._target
                rect = mpatches.Rectangle(
                    (target.get("x_min", 38.0), 0.0),
                    target.get("x_max", 45.0) - target.get("x_min", 38.0),
                    0.25,
                    linewidth=1.5,
                    edgecolor="#22c55e",
                    facecolor="#22c55e",
                    alpha=0.18,
                )
                ax.add_patch(rect)

            def _plot_box(self, ax, x0, x1, y0, y1, z0, z1, color, alpha):
                import numpy as np

                xx, yy = np.meshgrid([x0, x1], [y0, y1])
                for z in [z0, z1]:
                    ax.plot_surface(xx, yy, np.full_like(xx, z), color=color, alpha=alpha, shade=False)
                yy, zz = np.meshgrid([y0, y1], [z0, z1])
                for x in [x0, x1]:
                    ax.plot_surface(np.full_like(yy, x), yy, zz, color=color, alpha=alpha, shade=False)
                xx, zz = np.meshgrid([x0, x1], [z0, z1])
                for y in [y0, y1]:
                    ax.plot_surface(xx, np.full_like(xx, y), zz, color=color, alpha=alpha, shade=False)

            def _set_axes_3d(self, ax):
                room = self._room
                ax.set_xlim(room.get("x_min", 0.0), room.get("x_max", 45.0))
                ax.set_ylim(room.get("y_min", 0.0), room.get("y_max", 25.0))
                ax.set_zlim(room.get("z_min", 0.0), room.get("z_max", 20.0))
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.set_zlabel("z [m]")
                ax.view_init(elev=28, azim=-58)

            def _set_axes_top(self, ax):
                room = self._room
                ax.set_xlim(room.get("x_min", 0.0), room.get("x_max", 45.0))
                ax.set_ylim(room.get("y_min", 0.0), room.get("y_max", 25.0))
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)

            def _set_axes_side(self, ax):
                room = self._room
                ax.set_xlim(room.get("x_min", 0.0), room.get("x_max", 45.0))
                ax.set_ylim(room.get("z_min", 0.0), room.get("z_max", 20.0))
                ax.set_xlabel("x [m]")
                ax.set_ylabel("z [m]")
                ax.grid(True, alpha=0.3)

        rclpy.init(args=args)
        node = TrajectoryRecorderNode()
        try:
            rclpy.spin(node)
        finally:
            node.save_plot()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    except ImportError as exc:
        print(f"trajectory_recorder: required dependency unavailable: {exc}")


if __name__ == "__main__":
    main()
