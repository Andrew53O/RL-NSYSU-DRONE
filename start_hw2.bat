@echo off
setlocal

rem Start the NSYSU Drone Docker container in one terminal, then open a
rem second terminal inside the container with the ROS workspace sourced.
set "WSL_DISTRO="
set "PROJECT_DIR=/home/surya/HW2/nsysu_drone"
set "CONTAINER_NAME=nsysu_drone_vnc"
set "GPU_ID=0"
set "VNC_PORT=5901"

if defined WSL_DISTRO (
    set "WSL=wsl -d %WSL_DISTRO%"
) else (
    set "WSL=wsl"
)

start "NSYSU Drone Docker" cmd /k "%WSL% bash -lc ""cd '%PROJECT_DIR%' && GPU_ID=%GPU_ID% VNC_PORT=%VNC_PORT% ./run_docker.sh"""

start "NSYSU Drone ROS Shell" cmd /k "%WSL% bash -lc ""echo Waiting for %CONTAINER_NAME%...; until docker ps --format '{{.Names}}' | grep -qx '%CONTAINER_NAME%'; do sleep 2; done; docker exec -it %CONTAINER_NAME% bash -lc 'source /ros2_ws/install/setup.bash; echo ROS workspace sourced.; echo Run launch_drone here when you are ready.; exec bash'"""

endlocal
