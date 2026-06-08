#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-nsysu_drone_vnc}"
GPU_ID="${GPU_ID:-0}"
VNC_PORT="${VNC_PORT:-5901}"

# Snap-packaged shells can leak environment variables that confuse system
# terminal wrappers, so launch the GUI terminal with a cleaned environment.
unset_snap_env() {
    env \
        -u SNAP \
        -u SNAP_NAME \
        -u SNAP_INSTANCE_NAME \
        -u SNAP_REVISION \
        -u SNAP_ARCH \
        -u SNAP_COMMON \
        -u SNAP_DATA \
        -u SNAP_USER_COMMON \
        -u SNAP_USER_DATA \
        "$@"
}

launch_terminal() {
    local terminal="$1"
    shift

    case "${terminal}" in
        gnome-terminal)
            unset_snap_env "${terminal}" -- "$@"
            ;;
        *)
            unset_snap_env "${terminal}" -e "$@"
            ;;
    esac
}

launch_with_fallback() {
    local candidates=()
    local terminal

    if [[ -n "${TERMINAL_EMULATOR:-}" ]]; then
        candidates+=("${TERMINAL_EMULATOR}")
    fi

    candidates+=(gnome-terminal x-terminal-emulator konsole xterm)

    for terminal in "${candidates[@]}"; do
        [[ -n "${terminal}" ]] || continue
        if ! command -v "${terminal}" >/dev/null 2>&1; then
            continue
        fi

        if launch_terminal "${terminal}" "$@"; then
            TERMINAL_USED="${terminal}"
            return 0
        fi
    done

    return 1
}

docker_cmd="cd '${PROJECT_DIR}' && GPU_ID='${GPU_ID}' VNC_PORT='${VNC_PORT}' CONTAINER_NAME='${CONTAINER_NAME}' ./run_docker.sh"
ros_cmd="echo 'Waiting for ${CONTAINER_NAME}...'; until docker ps --format '{{.Names}}' | grep -qx '${CONTAINER_NAME}'; do sleep 2; done; docker exec -it '${CONTAINER_NAME}' bash -lc 'source /ros2_ws/install/setup.bash; echo ROS workspace sourced.; echo Run launch_drone here when you are ready.; exec bash'"

if ! launch_with_fallback bash -lc "${docker_cmd}"; then
    echo "No supported terminal emulator could launch the Docker terminal."
    echo "Try running this manually:"
    echo "  GPU_ID=${GPU_ID} VNC_PORT=${VNC_PORT} ./run_docker.sh"
    exit 1
fi

if ! launch_with_fallback bash -lc "${ros_cmd}"; then
    echo "No supported terminal emulator could launch the ROS shell."
    echo "If the Docker container started, open another terminal and run:"
    echo "  docker exec -it ${CONTAINER_NAME} bash"
    echo "  source /ros2_ws/install/setup.bash"
    exit 1
fi
