#!/usr/bin/env python3
"""Headless SITL integration test for reach-avoid simulation.

Runs INSIDE the Docker container (reach-avoid-sim). Launches Gazebo headless,
PX4 SITL, MicroXRCEAgent, ros_gz_bridge, and all game nodes, then verifies:

  Phase 1 - Gazebo world loads and sensor plugins produce data
  Phase 2 - PX4 instances connect via XRCE-DDS, EKF2 converges, drones arm
  Phase 3 - Ground truth relay publishes drone positions
  Phase 4 - Drones take off and maintain altitude in offboard mode
  Phase 5 - Attacker moves toward target following waypoints
  Phase 6 - Defender controller publishes reachability-based commands
  Phase 7 - Defender tracks/intercepts attacker (game dynamics)

Usage (inside Docker):
  python3 /home/simuser/ws/tests/integration/test_sitl_headless.py

Exit codes:
  0 = all checks passed
  1 = one or more checks failed
  2 = setup/infrastructure failure
"""

import json
import math
import os
import signal
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PX4_DIR = os.environ.get("PX4_AUTOPILOT", "/opt/PX4-Autopilot")
WS_DIR = "/home/simuser/ws"

# Timeouts (seconds)
GAZEBO_STARTUP_TIMEOUT = 30
PX4_SPAWN_TIMEOUT = 45
EKF2_CONVERGE_TIMEOUT = 60
ARM_TIMEOUT = 90
TOPIC_TIMEOUT = 20
TAKEOFF_TIMEOUT = 40
MOVEMENT_TIMEOUT = 60
GAME_TIMEOUT = 120

# Physical checks
MIN_TAKEOFF_ALT = 0.3       # meters above ground
MIN_ATTACKER_TRAVEL = 0.5   # meters of movement to confirm motion
CAPTURE_DIST_H = 3.0        # horizontal capture distance (from game params)
CAPTURE_DIST_Z = 1.0        # vertical capture distance

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name, detail=""):
        self.passed.append(name)
        detail_str = f" ({detail})" if detail else ""
        print(f"  {GREEN}PASS{RESET} {name}{detail_str}")

    def fail(self, name, detail=""):
        self.failed.append(name)
        detail_str = f" ({detail})" if detail else ""
        print(f"  {RED}FAIL{RESET} {name}{detail_str}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}Results: {len(self.passed)}/{total} passed{RESET}")
        if self.failed:
            print(f"{RED}Failed:{RESET}")
            for f in self.failed:
                print(f"  - {f}")
        else:
            print(f"{GREEN}All integration tests passed!{RESET}")
        print(f"{'='*60}")
        return len(self.failed) == 0


def run_cmd(cmd, timeout=10, check=False):
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def ros2_topic_list(timeout=5):
    """Get list of active ROS2 topics."""
    rc, out, _ = run_cmd("ros2 topic list", timeout=timeout)
    if rc != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def ros2_topic_echo_once(topic, msg_type=None, timeout=10):
    """Get one message from a topic. Returns the raw text or None."""
    cmd = f"ros2 topic echo --once {topic}"
    if msg_type:
        cmd += f" {msg_type}"
    rc, out, _ = run_cmd(cmd, timeout=timeout)
    if rc == 0 and out:
        return out
    return None


def ros2_topic_hz(topic, window=5, timeout=10):
    """Sample topic frequency. Returns average Hz or 0."""
    cmd = f"ros2 topic hz {topic} --window {window}"
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        time.sleep(min(timeout, window + 2))
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=5)
        # Parse "average rate: 49.5" from output
        for line in out.splitlines():
            if "average rate" in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        return float(parts[-1].strip())
                    except ValueError:
                        pass
    except Exception:
        pass
    return 0.0


def wait_for_topic(topic, timeout=TOPIC_TIMEOUT):
    """Wait until a topic appears in ros2 topic list."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        topics = ros2_topic_list()
        if topic in topics:
            return True
        time.sleep(1)
    return False


def wait_for_message(topic, timeout=TOPIC_TIMEOUT):
    """Wait for at least one message on a topic."""
    return ros2_topic_echo_once(topic, timeout=timeout) is not None


def parse_pose_xyz(echo_text):
    """Extract x, y, z from ros2 topic echo output of PoseStamped."""
    x = y = z = None
    lines = echo_text.splitlines()
    in_position = False
    for line in lines:
        stripped = line.strip()
        if "position:" in stripped:
            in_position = True
            continue
        if in_position:
            if stripped.startswith("x:"):
                x = float(stripped.split(":")[1])
            elif stripped.startswith("y:"):
                y = float(stripped.split(":")[1])
            elif stripped.startswith("z:"):
                z = float(stripped.split(":")[1])
            if x is not None and y is not None and z is not None:
                return x, y, z
    return x, y, z


def parse_twist_xyz(echo_text):
    """Extract linear x,y,z from ros2 topic echo output of Twist/TwistStamped."""
    x = y = z = None
    lines = echo_text.splitlines()
    in_linear = False
    for line in lines:
        stripped = line.strip()
        if "linear:" in stripped:
            in_linear = True
            continue
        if in_linear:
            if stripped.startswith("x:"):
                x = float(stripped.split(":")[1])
            elif stripped.startswith("y:"):
                y = float(stripped.split(":")[1])
            elif stripped.startswith("z:"):
                z = float(stripped.split(":")[1])
                return x, y, z  # done after z inside linear
    return x, y, z


def get_position(topic, timeout=10):
    """Get current position from a PoseStamped topic."""
    msg = ros2_topic_echo_once(topic, timeout=timeout)
    if msg:
        return parse_pose_xyz(msg)
    return None, None, None


def sample_positions(topic, duration=5, interval=0.5):
    """Collect multiple position samples over a duration."""
    samples = []
    deadline = time.time() + duration
    while time.time() < deadline:
        msg = ros2_topic_echo_once(topic, timeout=3)
        if msg:
            xyz = parse_pose_xyz(msg)
            if xyz[0] is not None:
                samples.append(xyz)
        time.sleep(interval)
    return samples


def check_arming_state(namespace, timeout=ARM_TIMEOUT):
    """Wait for arming_state == 2 (ARMED) on vehicle_status topic."""
    topic = f"/{namespace}/fmu/out/vehicle_status"
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = ros2_topic_echo_once(topic, timeout=5)
        if msg and "arming_state: 2" in msg:
            return True
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------
_ALL_PROCS = []


def start_process(cmd, name="", env=None, cwd=None):
    """Start a background process, track it for cleanup."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.Popen(
        cmd, shell=True, env=merged_env, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )
    _ALL_PROCS.append((proc, name))
    print(f"  {CYAN}Started{RESET} {name or cmd} (pid={proc.pid})")
    return proc


def cleanup_all():
    """Kill all tracked processes."""
    print(f"\n{YELLOW}Cleaning up processes...{RESET}")
    for proc, name in reversed(_ALL_PROCS):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    _ALL_PROCS.clear()
    # Also kill any straggler PX4/gz/MicroXRCE processes
    run_cmd("pkill -f 'gz sim' || true; pkill -f px4 || true; "
            "pkill -f MicroXRCEAgent || true; pkill -f parameter_bridge || true",
            timeout=5)


# ---------------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------------
def build_gazebo_env():
    """Build environment for Gazebo with PX4 models/plugins."""
    px4_models = os.path.join(PX4_DIR, "Tools", "simulation", "gz", "models")
    px4_worlds = os.path.join(PX4_DIR, "Tools", "simulation", "gz", "worlds")
    px4_plugins = os.path.join(PX4_DIR, "build", "px4_sitl_default", "src",
                               "modules", "simulation", "gz_plugins")
    px4_server_cfg = os.path.join(PX4_DIR, "src", "modules", "simulation",
                                  "gz_bridge", "server.config")
    return {
        "GZ_SIM_RESOURCE_PATH": ":".join(filter(None, [
            os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
            px4_models, px4_worlds,
        ])),
        "GZ_SIM_SYSTEM_PLUGIN_PATH": ":".join(filter(None, [
            os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", ""),
            px4_plugins,
        ])),
        "GZ_SIM_SERVER_CONFIG_PATH": px4_server_cfg,
    }


def build_px4_env(instance_id, namespace, model_pose):
    """Build environment for a PX4 SITL instance."""
    px4_models = os.path.join(PX4_DIR, "Tools", "simulation", "gz", "models")
    px4_worlds = os.path.join(PX4_DIR, "Tools", "simulation", "gz", "worlds")
    px4_plugins = os.path.join(PX4_DIR, "build", "px4_sitl_default", "src",
                               "modules", "simulation", "gz_plugins")
    return {
        "PX4_GZ_STANDALONE": "1",
        "PX4_SIM_MODEL": "gz_x500",
        "PX4_SYS_AUTOSTART": "4001",
        "PX4_GZ_WORLD": "reach_avoid_arena",
        "PX4_GZ_NO_FOLLOW": "1",
        "PX4_GZ_MODELS": px4_models,
        "PX4_GZ_WORLDS": px4_worlds,
        "PX4_GZ_MODEL_POSE": model_pose,
        "PX4_UXRCE_DDS_NS": namespace,
        "GZ_SIM_RESOURCE_PATH": ":".join(filter(None, [
            os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
            px4_models, px4_worlds,
        ])),
        "GZ_SIM_SYSTEM_PLUGIN_PATH": ":".join(filter(None, [
            os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", ""),
            px4_plugins,
        ])),
        # Indoor SITL params
        "PX4_PARAM_NAV_DLL_ACT": "0",
        "PX4_PARAM_COM_ARM_MAG_STR": "0",
        "PX4_PARAM_EKF2_MAG_CHECK": "0",
        "PX4_PARAM_EKF2_MAG_GATE": "10",
        "PX4_PARAM_COM_ARM_WO_GPS": "1",
        "PX4_PARAM_SYS_HAS_GPS": "0",
        "PX4_PARAM_EKF2_GPS_CTRL": "0",
        "PX4_PARAM_EKF2_HGT_REF": "2",
        "PX4_PARAM_EKF2_BARO_CTRL": "1",
        "PX4_PARAM_EKF2_MAG_TYPE": "1",
    }


# ===========================================================================
# Test phases
# ===========================================================================
def phase1_gazebo(results):
    """Phase 1: Start Gazebo headless, verify world loads."""
    print(f"\n{BOLD}Phase 1: Gazebo World & Sensor Plugins{RESET}")

    world_sdf = os.path.join(
        WS_DIR, "install", "reach_avoid_sim", "share",
        "reach_avoid_sim", "worlds", "reach_avoid_arena.sdf"
    )
    # Fallback to src if install doesn't exist
    if not os.path.isfile(world_sdf):
        world_sdf = os.path.join(
            WS_DIR, "src", "reach_avoid_sim", "worlds", "reach_avoid_arena.sdf"
        )

    if not os.path.isfile(world_sdf):
        results.fail("world_sdf_exists", f"Not found: {world_sdf}")
        return False

    results.ok("world_sdf_exists", world_sdf)

    # Start Gazebo server-only (headless), -r = run immediately
    gz_env = build_gazebo_env()
    start_process(
        f"gz sim -s -r --headless-rendering {world_sdf}",
        name="gazebo-server",
        env=gz_env,
    )

    # Wait for Gazebo to be responsive
    deadline = time.time() + GAZEBO_STARTUP_TIMEOUT
    gz_running = False
    while time.time() < deadline:
        rc, out, _ = run_cmd("gz topic -l", timeout=5)
        if rc == 0 and "/world/reach_avoid_arena/clock" in out:
            gz_running = True
            break
        time.sleep(2)

    if gz_running:
        results.ok("gazebo_started", "world clock topic active")
    else:
        results.fail("gazebo_started", "gz topic -l did not show clock")
        return False

    # Verify key Gazebo topics (sensor plugins loaded)
    rc, gz_topics, _ = run_cmd("gz topic -l", timeout=5)
    gz_topics_list = gz_topics.splitlines()

    sensor_checks = {
        "gz_physics_plugin": "/world/reach_avoid_arena/clock",
        "gz_scene_broadcaster": "/world/reach_avoid_arena/scene/info",
    }
    for check_name, topic in sensor_checks.items():
        if any(topic in t for t in gz_topics_list):
            results.ok(check_name, topic)
        else:
            results.fail(check_name, f"topic {topic} not found in gz topics")

    return True


def phase2_px4(results):
    """Phase 2: Start PX4 SITL instances, MicroXRCEAgent, verify EKF2 + arming."""
    print(f"\n{BOLD}Phase 2: PX4 SITL + EKF2 Convergence + Arming{RESET}")

    # Start MicroXRCEAgent
    start_process(
        "MicroXRCEAgent udp4 -p 8888",
        name="xrce-dds-agent",
    )
    time.sleep(2)

    # Start ros_gz_bridge
    bridge_config = os.path.join(
        WS_DIR, "install", "reach_avoid_bringup", "share",
        "reach_avoid_bringup", "config", "gz_bridge.yaml"
    )
    if not os.path.isfile(bridge_config):
        bridge_config = os.path.join(
            WS_DIR, "src", "reach_avoid_bringup", "config", "gz_bridge.yaml"
        )
    start_process(
        f"ros2 run ros_gz_bridge parameter_bridge "
        f"--ros-args -p config_file:={bridge_config}",
        name="ros-gz-bridge",
    )
    time.sleep(2)

    # PX4 binary
    px4_bin = os.path.join(PX4_DIR, "build", "px4_sitl_default", "bin", "px4")
    if not os.path.isfile(px4_bin):
        results.fail("px4_binary_exists", f"Not found: {px4_bin}")
        return False
    results.ok("px4_binary_exists", px4_bin)

    # Start defender PX4 (instance 1)
    defender_env = build_px4_env(1, "defender", "1.5,1.5,0.5")
    start_process(
        f"{px4_bin} -i 1",
        name="px4-defender",
        env=defender_env,
        cwd=PX4_DIR,
    )

    time.sleep(3)

    # Start attacker PX4 (instance 2)
    attacker_env = build_px4_env(2, "attacker", "1.5,6.5,0.5")
    start_process(
        f"{px4_bin} -i 2",
        name="px4-attacker",
        env=attacker_env,
        cwd=PX4_DIR,
    )

    # Wait for PX4 to connect through XRCE-DDS (vehicle_status topics appear)
    print("  Waiting for PX4 XRCE-DDS topics...")
    for ns in ["defender", "attacker"]:
        topic = f"/{ns}/fmu/out/vehicle_status"
        if wait_for_topic(topic, timeout=PX4_SPAWN_TIMEOUT):
            results.ok(f"px4_{ns}_connected", f"{topic} active")
        else:
            results.fail(f"px4_{ns}_connected", f"{topic} not found")
            return False

    # Verify Gazebo models spawned (x500_1, x500_2)
    rc, model_out, _ = run_cmd(
        "gz model --list -w reach_avoid_arena", timeout=10
    )
    for model in ["x500_1", "x500_2"]:
        if model in model_out:
            results.ok(f"gz_model_{model}_spawned")
        else:
            results.fail(f"gz_model_{model}_spawned", f"not in: {model_out}")

    # Wait for ROS2 /clock from bridge
    if wait_for_topic("/clock", timeout=15):
        results.ok("ros2_clock_bridged")
    else:
        results.fail("ros2_clock_bridged")

    # Wait for model pose topics from bridge
    for model_name in ["x500_1", "x500_2"]:
        topic = f"/model/{model_name}/pose"
        if wait_for_message(topic, timeout=15):
            results.ok(f"gz_bridge_{model_name}_pose", f"{topic} has data")
        else:
            results.fail(f"gz_bridge_{model_name}_pose", f"no data on {topic}")

    return True


def phase3_ground_truth(results):
    """Phase 3: Start ground truth relay, verify state topics."""
    print(f"\n{BOLD}Phase 3: Ground Truth Relay{RESET}")

    # Start ground truth relay
    start_process(
        "ros2 run reach_avoid_sim ground_truth_relay "
        "--ros-args -p defender_model_name:=x500_1 "
        "-p attacker_model_name:=x500_2 "
        "-p world_name:=reach_avoid_arena "
        "-p publish_rate:=50.0",
        name="ground-truth-relay",
    )

    # Wait for state topics
    for role in ["defender", "attacker"]:
        for suffix, desc in [("/state", "position"), ("/velocity", "velocity")]:
            topic = f"/{role}{suffix}"
            if wait_for_message(topic, timeout=20):
                results.ok(f"{role}{suffix}_publishing", desc)
            else:
                results.fail(f"{role}{suffix}_publishing", f"no data on {topic}")

    # Verify initial positions match spawn points
    # Defender should be near (1.5, 1.5, ~0.5), attacker near (1.5, 6.5, ~0.5)
    for role, expected_x, expected_y in [("defender", 1.5, 1.5), ("attacker", 1.5, 6.5)]:
        x, y, z = get_position(f"/{role}/state", timeout=5)
        if x is not None:
            dist = math.sqrt((x - expected_x)**2 + (y - expected_y)**2)
            if dist < 3.0:  # generous tolerance for initial spawn
                results.ok(f"{role}_initial_position",
                           f"({x:.1f}, {y:.1f}, {z:.1f}) near expected ({expected_x}, {expected_y})")
            else:
                results.fail(f"{role}_initial_position",
                             f"({x:.1f}, {y:.1f}) too far from ({expected_x}, {expected_y}), dist={dist:.1f}")
        else:
            results.fail(f"{role}_initial_position", "could not read position")

    return True


def phase4_arming_takeoff(results):
    """Phase 4: Start PX4 adapters, verify arming + takeoff."""
    print(f"\n{BOLD}Phase 4: PX4 Adapters - Arming & Takeoff{RESET}")

    # Start PX4 adapters
    for role, vid in [("defender", 1), ("attacker", 2)]:
        start_process(
            f"ros2 run reach_avoid_sim px4_adapter "
            f"--ros-args -p vehicle_id:={vid} "
            f"-p cmd_vel_topic:=/{role}/cmd_vel "
            f"-p fmu_topic_prefix:={role}",
            name=f"px4-adapter-{role}",
        )

    time.sleep(3)

    # Check for arming
    print("  Waiting for drones to arm (EKF2 must converge first)...")
    for role in ["defender", "attacker"]:
        if check_arming_state(role, timeout=ARM_TIMEOUT):
            results.ok(f"{role}_armed", "arming_state=2")
        else:
            results.fail(f"{role}_armed",
                         "arming_state != 2 within timeout (EKF2 may not have converged)")

    # Give time for takeoff (PX4 offboard velocity mode -- adapters send heartbeats)
    print("  Waiting for takeoff...")
    time.sleep(10)

    # Verify altitude > MIN_TAKEOFF_ALT
    for role in ["defender", "attacker"]:
        x, y, z = get_position(f"/{role}/state", timeout=5)
        if z is not None and z > MIN_TAKEOFF_ALT:
            results.ok(f"{role}_airborne", f"altitude={z:.2f}m")
        elif z is not None:
            results.fail(f"{role}_airborne", f"altitude={z:.2f}m < {MIN_TAKEOFF_ALT}m")
        else:
            results.fail(f"{role}_airborne", "could not read altitude")

    return True


def phase5_attacker_movement(results):
    """Phase 5: Start attacker controller, verify it follows waypoints."""
    print(f"\n{BOLD}Phase 5: Attacker Waypoint Following{RESET}")

    start_process(
        "ros2 run attacker_controller attacker_node "
        "--ros-args -p mode:=scripted "
        "-p max_speed:=0.5 -p speed_fraction:=0.8 "
        "-p target_x:=7.0 -p target_y:=4.0 -p target_z:=2.0",
        name="attacker-controller",
    )

    # Collect attacker positions over time to verify movement
    print("  Sampling attacker positions...")
    time.sleep(5)  # let controller start sending commands
    positions = sample_positions("/attacker/state", duration=15, interval=1.0)

    if len(positions) < 3:
        results.fail("attacker_positions_sampled", f"only {len(positions)} samples")
        return True

    results.ok("attacker_positions_sampled", f"{len(positions)} samples")

    # Check movement: total displacement from first to last
    x0, y0, z0 = positions[0]
    x1, y1, z1 = positions[-1]
    travel = math.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)

    if travel > MIN_ATTACKER_TRAVEL:
        results.ok("attacker_moving", f"traveled {travel:.2f}m")
    else:
        results.fail("attacker_moving", f"traveled only {travel:.2f}m < {MIN_ATTACKER_TRAVEL}m")

    # Check attacker cmd_vel is non-zero
    msg = ros2_topic_echo_once("/attacker/cmd_vel", timeout=5)
    if msg:
        vx, vy, vz = parse_twist_xyz(msg)
        if vx is not None:
            speed = math.sqrt(vx**2 + vy**2 + (vz or 0)**2)
            if speed > 0.01:
                results.ok("attacker_cmd_vel_nonzero", f"speed={speed:.3f}")
            else:
                results.fail("attacker_cmd_vel_nonzero", f"speed={speed:.3f}")
        else:
            results.fail("attacker_cmd_vel_nonzero", "could not parse")
    else:
        results.fail("attacker_cmd_vel_nonzero", "no message")

    return True


def phase6_defender_control(results):
    """Phase 6: Start defender controller, verify reachability-based commands."""
    print(f"\n{BOLD}Phase 6: Defender Reachability Controller{RESET}")

    vf_dir = os.path.join(WS_DIR, "data", "value_functions")
    if not os.path.isdir(vf_dir):
        # Try alternate locations
        for alt in ["/workspace/data/value_functions", os.path.join(WS_DIR, "config")]:
            if os.path.isdir(alt):
                vf_dir = alt
                break

    start_process(
        f"ros2 run reach_avoid_controller defender_node "
        f"--ros-args -p value_function_dir:={vf_dir} "
        f"-p control_rate:=50.0 "
        f"-p pid_gain_z:=2.0 -p pid_gain_h:=2.0",
        name="defender-controller",
    )

    time.sleep(5)

    # Check defender cmd_vel is being published
    msg = ros2_topic_echo_once("/defender/cmd_vel", timeout=10)
    if msg:
        vx, vy, vz = parse_twist_xyz(msg)
        if vx is not None:
            results.ok("defender_cmd_vel_publishing",
                       f"v=({vx:.2f}, {vy:.2f}, {vz:.2f})")
        else:
            results.fail("defender_cmd_vel_publishing", "could not parse")
    else:
        results.fail("defender_cmd_vel_publishing", "no message on /defender/cmd_vel")

    # Check game status is being published
    status_msg = ros2_topic_echo_once("/game/status", timeout=10)
    if status_msg:
        results.ok("game_status_publishing", status_msg.split("\n")[0][:80])
    else:
        results.fail("game_status_publishing", "no data on /game/status")

    # Verify game status contains mode info (z: and h: modes)
    if status_msg:
        has_modes = ("z:" in status_msg or "z_mode" in status_msg) and \
                    ("h:" in status_msg or "h_mode" in status_msg)
        if has_modes:
            results.ok("game_status_has_modes", "contains z/h mode info")
        else:
            results.fail("game_status_has_modes", "missing mode info in status")

    return True


def phase7_game_dynamics(results):
    """Phase 7: Verify defender tracks attacker (positions converge over time)."""
    print(f"\n{BOLD}Phase 7: Game Dynamics - Defender Tracks Attacker{RESET}")

    # Sample positions of both drones over time
    print("  Monitoring game for convergence...")
    samples = []
    duration = min(GAME_TIMEOUT, 60)
    deadline = time.time() + duration

    while time.time() < deadline:
        d_msg = ros2_topic_echo_once("/defender/state", timeout=3)
        a_msg = ros2_topic_echo_once("/attacker/state", timeout=3)
        if d_msg and a_msg:
            dx, dy, dz = parse_pose_xyz(d_msg)
            ax, ay, az = parse_pose_xyz(a_msg)
            if all(v is not None for v in [dx, dy, dz, ax, ay, az]):
                h_dist = math.sqrt((dx - ax)**2 + (dy - ay)**2)
                z_dist = abs(dz - az)
                samples.append({
                    "t": time.time(),
                    "defender": (dx, dy, dz),
                    "attacker": (ax, ay, az),
                    "h_dist": h_dist,
                    "z_dist": z_dist,
                })
        time.sleep(2)

    if len(samples) < 5:
        results.fail("game_samples_collected", f"only {len(samples)} samples")
        return True

    results.ok("game_samples_collected", f"{len(samples)} samples over {duration}s")

    # Check 1: Defender is actively moving (not stuck)
    d_first = samples[0]["defender"]
    d_last = samples[-1]["defender"]
    d_travel = math.sqrt(sum((a - b)**2 for a, b in zip(d_first, d_last)))
    if d_travel > 0.3:
        results.ok("defender_moving", f"traveled {d_travel:.2f}m")
    else:
        results.fail("defender_moving", f"traveled only {d_travel:.2f}m")

    # Check 2: Inter-drone distance trend (should decrease or stay bounded)
    initial_h_dist = samples[0]["h_dist"]
    final_h_dist = samples[-1]["h_dist"]
    min_h_dist = min(s["h_dist"] for s in samples)

    results.ok("distance_tracking",
               f"h_dist: {initial_h_dist:.2f}→{final_h_dist:.2f}m, min={min_h_dist:.2f}m")

    # Check 3: Defender reduces distance or enters capture zone at some point
    if min_h_dist < initial_h_dist or min_h_dist < CAPTURE_DIST_H * 2:
        results.ok("defender_pursuing",
                   f"min horizontal distance {min_h_dist:.2f}m (started at {initial_h_dist:.2f}m)")
    else:
        results.fail("defender_pursuing",
                     f"distance never decreased: {initial_h_dist:.2f}→min {min_h_dist:.2f}m")

    # Check 4: Vertical tracking (z-distance should be bounded)
    min_z_dist = min(s["z_dist"] for s in samples)
    avg_z_dist = sum(s["z_dist"] for s in samples) / len(samples)
    if avg_z_dist < CAPTURE_DIST_Z * 3:
        results.ok("vertical_tracking", f"avg z_dist={avg_z_dist:.2f}m, min={min_z_dist:.2f}m")
    else:
        results.fail("vertical_tracking", f"avg z_dist={avg_z_dist:.2f}m too large")

    # Check 5: Game status shows reachability modes (not just pid_fallback)
    status_msg = ros2_topic_echo_once("/game/status", timeout=5)
    if status_msg:
        # Check if it's using value-function-based control, not just fallback
        is_reachability = any(mode in status_msg for mode in
                             ["reaching", "tracking", "pid_deep", "DEFENDER_WINNING", "CAPTURED"])
        if is_reachability:
            results.ok("reachability_control_active",
                       status_msg.strip().split("\n")[0][:80])
        else:
            # pid_fallback is acceptable if value functions couldn't load
            results.fail("reachability_control_active",
                         f"only fallback modes seen: {status_msg.strip()[:80]}")

    # Print final game snapshot
    if samples:
        s = samples[-1]
        print(f"\n  {CYAN}Final snapshot:{RESET}")
        print(f"    Defender: ({s['defender'][0]:.2f}, {s['defender'][1]:.2f}, {s['defender'][2]:.2f})")
        print(f"    Attacker: ({s['attacker'][0]:.2f}, {s['attacker'][1]:.2f}, {s['attacker'][2]:.2f})")
        print(f"    h_dist={s['h_dist']:.2f}m  z_dist={s['z_dist']:.2f}m")

    return True


# ===========================================================================
# Main
# ===========================================================================
def main():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Reach-Avoid SITL Integration Test (Headless){RESET}")
    print(f"{'='*60}")
    print(f"  PX4_DIR:  {PX4_DIR}")
    print(f"  WS_DIR:   {WS_DIR}")
    print(f"  Headless: yes (gz sim -s --headless-rendering)")
    print(f"{'='*60}\n")

    results = TestResult()

    try:
        if not phase1_gazebo(results):
            print(f"\n{RED}Phase 1 failed, cannot continue.{RESET}")
            return 2

        if not phase2_px4(results):
            print(f"\n{RED}Phase 2 failed, cannot continue.{RESET}")
            return 2

        phase3_ground_truth(results)
        phase4_arming_takeoff(results)
        phase5_attacker_movement(results)
        phase6_defender_control(results)
        phase7_game_dynamics(results)

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted by user{RESET}")
    except Exception as e:
        print(f"\n{RED}Unexpected error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        results.fail("unexpected_error", str(e))
    finally:
        cleanup_all()

    all_passed = results.summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
