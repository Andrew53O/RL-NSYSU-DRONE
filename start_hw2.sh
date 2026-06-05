#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-nsysu_drone_vnc}"
GPU_ID="${GPU_ID:-0}"
VNC_PORT="${VNC_PORT:-5901}"

if command -v gnome-terminal >/dev/null 2>&1; then
    TERMINAL=(gnome-terminal --)
elif command -v x-terminal-emulator >/dev/null 2>&1; then
    TERMINAL=(x-terminal-emulator -e)
elif command -v konsole >/dev/null 2>&1; then
    TERMINAL=(konsole -e)
elif command -v xterm >/dev/null 2>&1; then
    TERMINAL=(xterm -e)
else
    echo "No supported terminal emulator found."
    echo "Run this manually instead:"
    echo "  GPU_ID=${GPU_ID} VNC_PORT=${VNC_PORT} ./run_docker.sh"
    exit 1
fi

"${TERMINAL[@]}" bash -lc "cd '${PROJECT_DIR}' && GPU_ID='${GPU_ID}' VNC_PORT='${VNC_PORT}' CONTAINER_NAME='${CONTAINER_NAME}' ./run_docker.sh"

"${TERMINAL[@]}" bash -lc "echo 'Waiting for ${CONTAINER_NAME}...'; until docker ps --format '{{.Names}}' | grep -qx '${CONTAINER_NAME}'; do sleep 2; done; docker exec -it '${CONTAINER_NAME}' bash -lc 'source /ros2_ws/install/setup.bash; echo ROS workspace sourced.; echo Run launch_drone here when you are ready.; exec bash'"
