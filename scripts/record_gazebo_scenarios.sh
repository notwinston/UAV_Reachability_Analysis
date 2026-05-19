#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="${REPO_ROOT}/reach_avoid_ws"
OUTPUT_DIR="${REPO_ROOT}/docs/videos"
FRAME_RATE="12"
SIM_TIMEOUT_SEC="420"
CAMERA_TOPIC="/overview_camera"

source /opt/ros/humble/setup.bash
source /opt/px4_ros_ws/install/setup.bash
source "${WS_DIR}/install/setup.bash"

set -u

export LIBGL_ALWAYS_SOFTWARE=1
export QT_X11_NO_MITSHM=1
unset DISPLAY || true

LAUNCH_PID=""
IMAGE_BRIDGE_PID=""
RECORDER_PID=""

cleanup_runtime() {
    if [[ -n "${RECORDER_PID}" ]] && kill -0 "${RECORDER_PID}" 2>/dev/null; then
        kill -INT "${RECORDER_PID}" 2>/dev/null || true
        wait "${RECORDER_PID}" 2>/dev/null || true
    fi
    RECORDER_PID=""

    if [[ -n "${IMAGE_BRIDGE_PID}" ]] && kill -0 "${IMAGE_BRIDGE_PID}" 2>/dev/null; then
        kill -INT "${IMAGE_BRIDGE_PID}" 2>/dev/null || true
        wait "${IMAGE_BRIDGE_PID}" 2>/dev/null || true
    fi
    IMAGE_BRIDGE_PID=""

    if [[ -n "${LAUNCH_PID}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
        kill -INT "${LAUNCH_PID}" 2>/dev/null || true
        wait "${LAUNCH_PID}" 2>/dev/null || true
    fi
    LAUNCH_PID=""

    pkill -f "ros2 launch reach_avoid_bringup full_game.launch.py" 2>/dev/null || true
    pkill -f "ros2 run ros_gz_image image_bridge ${CAMERA_TOPIC}" 2>/dev/null || true
    pkill -f "record_ros_image_topic.py" 2>/dev/null || true
    pkill -f "MicroXRCEAgent udp4 -p 8888" 2>/dev/null || true
    pkill -f "parameter_bridge.*gz_bridge.yaml" 2>/dev/null || true
    pkill -f "build/px4_sitl_default/bin/px4 -i 1" 2>/dev/null || true
    pkill -f "build/px4_sitl_default/bin/px4 -i 2" 2>/dev/null || true
    pkill -f "gz sim -r" 2>/dev/null || true
    sleep 3
}

shutdown_all() {
    cleanup_runtime
}

trap shutdown_all EXIT

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

start_image_bridge() {
    ros2 run ros_gz_image image_bridge "${CAMERA_TOPIC}" \
        >"${OUTPUT_DIR}/image_bridge.log" 2>&1 &
    IMAGE_BRIDGE_PID=$!
    sleep 2
    if ! kill -0 "${IMAGE_BRIDGE_PID}" 2>/dev/null; then
        echo "Failed to start image bridge for ${CAMERA_TOPIC}" >&2
        exit 1
    fi
}

start_video_recorder() {
    local video_path="$1"
    python3.10 "${REPO_ROOT}/scripts/record_ros_image_topic.py" \
        --topic "${CAMERA_TOPIC}" \
        --output "${video_path}" \
        --fps "${FRAME_RATE}" \
        --idle-timeout-sec 5.0 \
        >"${video_path%.mp4}_recorder.log" 2>&1 &
    RECORDER_PID=$!
    sleep 2
    if ! kill -0 "${RECORDER_PID}" 2>/dev/null; then
        echo "Failed to start recorder for ${video_path}" >&2
        exit 1
    fi
}

run_launch() {
    local defender_pose="$1"
    local attacker_pose="$2"
    local plot_name="$3"
    local launch_log="$4"

    timeout --signal=SIGINT "${SIM_TIMEOUT_SEC}" \
        ros2 launch reach_avoid_bringup full_game.launch.py \
        attacker_mode:=optimal \
        defender_pose:="${defender_pose}" \
        attacker_pose:="${attacker_pose}" \
        trajectory_output_dir:="${OUTPUT_DIR}" \
        trajectory_output_name:="${plot_name}" \
        >"${launch_log}" 2>&1 &
    LAUNCH_PID=$!
}

wait_for_launch() {
    local launch_rc=0
    set +e
    wait "${LAUNCH_PID}"
    launch_rc=$?
    set -e
    LAUNCH_PID=""
    return "${launch_rc}"
}

stop_auxiliary_recorders() {
    if [[ -n "${IMAGE_BRIDGE_PID}" ]] && kill -0 "${IMAGE_BRIDGE_PID}" 2>/dev/null; then
        kill -INT "${IMAGE_BRIDGE_PID}" 2>/dev/null || true
        sleep 1
        if kill -0 "${IMAGE_BRIDGE_PID}" 2>/dev/null; then
            kill -TERM "${IMAGE_BRIDGE_PID}" 2>/dev/null || true
            sleep 1
        fi
        if kill -0 "${IMAGE_BRIDGE_PID}" 2>/dev/null; then
            kill -KILL "${IMAGE_BRIDGE_PID}" 2>/dev/null || true
        fi
        wait "${IMAGE_BRIDGE_PID}" 2>/dev/null || true
    fi
    IMAGE_BRIDGE_PID=""

    if [[ -n "${RECORDER_PID}" ]] && kill -0 "${RECORDER_PID}" 2>/dev/null; then
        kill -INT "${RECORDER_PID}" 2>/dev/null || true
        wait "${RECORDER_PID}" 2>/dev/null || true
    fi
    RECORDER_PID=""
}

run_scenario() {
    local name="$1"
    local defender_pose="$2"
    local attacker_pose="$3"

    local video_path="${OUTPUT_DIR}/${name}.mp4"
    local plot_name="${name}_trajectory.png"
    local launch_log="${OUTPUT_DIR}/${name}_launch.log"

    echo
    echo "=== Running ${name} ==="
    echo "Defender: ${defender_pose}"
    echo "Attacker: ${attacker_pose}"

    cleanup_runtime
    start_image_bridge
    start_video_recorder "${video_path}"
    run_launch "${defender_pose}" "${attacker_pose}" "${plot_name}" "${launch_log}"

    local launch_rc=0
    set +e
    wait_for_launch
    launch_rc=$?
    set -e

    sleep 2
    stop_auxiliary_recorders

    rm -f \
        "${OUTPUT_DIR}/full_game_trajectory_latest.png" \
        "${OUTPUT_DIR}/full_game_trajectory_latest.csv" \
        "${OUTPUT_DIR}/full_game_trajectory_latest_summary.json"

    if [[ ${launch_rc} -ne 0 ]]; then
        echo "Launch exited with code ${launch_rc} for ${name}" >&2
        return "${launch_rc}"
    fi

    if [[ ! -f "${OUTPUT_DIR}/${plot_name}" ]]; then
        echo "Missing trajectory plot for ${name}" >&2
        return 1
    fi
    if [[ ! -s "${video_path}" ]]; then
        echo "Missing or empty video for ${name}" >&2
        return 1
    fi
}

SCENARIOS=(
    "goal_guard_north 36.0,19.0,3.0 3.0,3.0,3.0"
    "goal_guard_south 36.0,6.0,3.0 3.0,22.0,3.0"
    "goal_guard_center 34.0,12.5,3.0 2.5,2.5,3.0"
)

for scenario in "${SCENARIOS[@]}"; do
    run_scenario ${scenario}
done

rm -f "${OUTPUT_DIR}"/*.log

echo
echo "Artifacts saved in ${OUTPUT_DIR}"
