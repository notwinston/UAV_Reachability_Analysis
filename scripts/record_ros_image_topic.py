#!/usr/bin/env python3.10

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageTopicRecorder(Node):
    def __init__(
        self,
        topic: str,
        output_path: Path,
        fps: float,
        first_frame_path: Path | None,
        max_frames: int | None,
        idle_timeout_sec: float,
    ) -> None:
        super().__init__("overview_camera_recorder")
        self._bridge = CvBridge()
        self._topic = topic
        self._output_path = output_path
        self._fps = fps
        self._first_frame_path = first_frame_path
        self._max_frames = max_frames
        self._idle_timeout_sec = idle_timeout_sec
        self._video_writer = None
        self._frame_count = 0
        self._saved_first_frame = False
        self._last_frame_time = time.monotonic()
        self._shutdown_requested = False

        self.create_subscription(Image, self._topic, self._image_cb, 10)
        self.create_timer(0.5, self._watchdog_cb)

        self.get_logger().info(
            f"Recording {self._topic} to {self._output_path} at {self._fps:.1f} fps"
        )

    def _image_cb(self, msg: Image) -> None:
        if self._shutdown_requested:
            return

        frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._last_frame_time = time.monotonic()

        if self._video_writer is None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            height, width = frame.shape[:2]
            self._video_writer = cv2.VideoWriter(
                str(self._output_path), fourcc, self._fps, (width, height)
            )
            if not self._video_writer.isOpened():
                raise RuntimeError(f"Failed to open video output {self._output_path}")
            self.get_logger().info(f"Opened writer at {width}x{height}")

        self._video_writer.write(frame)
        self._frame_count += 1

        if self._first_frame_path and not self._saved_first_frame:
            self._first_frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self._first_frame_path), frame)
            self._saved_first_frame = True
            self.get_logger().info(f"Saved first frame to {self._first_frame_path}")

        if self._max_frames is not None and self._frame_count >= self._max_frames:
            self.get_logger().info(f"Reached max frame count {self._max_frames}")
            self.request_shutdown()

    def _watchdog_cb(self) -> None:
        if self._shutdown_requested:
            return
        if self._frame_count == 0:
            return
        if time.monotonic() - self._last_frame_time > self._idle_timeout_sec:
            self.get_logger().info(
                f"No frame received for {self._idle_timeout_sec:.1f}s, stopping recorder"
            )
            self.request_shutdown()

    def request_shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self._release_writer()

    def _release_writer(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a ROS image topic to MP4.")
    parser.add_argument("--topic", required=True, help="ROS image topic to subscribe to")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=float, default=20.0, help="Target video FPS")
    parser.add_argument(
        "--first-frame-output",
        default="",
        help="Optional PNG path for the first received frame",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame cap; 0 means unlimited",
    )
    parser.add_argument(
        "--idle-timeout-sec",
        type=float,
        default=5.0,
        help="Stop after this long without receiving a frame",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    output_path = Path(args.output)
    first_frame_path = Path(args.first_frame_output) if args.first_frame_output else None
    max_frames = args.max_frames if args.max_frames > 0 else None

    rclpy.init(args=None)
    node = ImageTopicRecorder(
        topic=args.topic,
        output_path=output_path,
        fps=args.fps,
        first_frame_path=first_frame_path,
        max_frames=max_frames,
        idle_timeout_sec=args.idle_timeout_sec,
    )

    def _handle_signal(signum, frame):
        node.get_logger().info(f"Received signal {signum}, stopping recorder")
        node.request_shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        while rclpy.ok() and not node._shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        node.request_shutdown()
    finally:
        node._release_writer()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
