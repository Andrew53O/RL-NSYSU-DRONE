# HW2 Command Runbook

Use this file as the quick command reference for running the NSYSU Drone RL HW2 experiment.

Main goal: **Part 2, Task D - sonar-based obstacle avoidance with PPO**.

Important idea:

- Run Docker from the host.
- Launch Gazebo/RViz inside Docker.
- Keep Gazebo running while training/testing.
- Use a second Docker terminal for sonar checks, training, and testing.
- Rebuild the ROS workspace after changing files in `nsysu_drone_description`, `nsysu_drone_bringup`, or `nsysu_drone_control`.
- Do not use the old `ppo_drone.zip` after observation-space changes; train a new model.

## 0. Host Terminal: Start Docker

Run this on the host, not inside Docker:

```bash
cd ~/HW2/nsysu_drone
GPU_ID=0 ./run_docker.sh
```

Expected result:

- Container name: `nsysu_drone_vnc`
- VNC port: `127.0.0.1:5901`
- VNC password: `nsysudrone`
- You should land in a shell like `root@...:/ros2_ws#`

The script mounts:

```text
~/HW2/nsysu_drone/HW2_Work -> /workspace/HW2_Work
~/HW2/nsysu_drone/nsysu_drone_description -> /ros2_ws/src/nsysu_drone_description
~/HW2/nsysu_drone/nsysu_drone_bringup -> /ros2_ws/src/nsysu_drone_bringup
~/HW2/nsysu_drone/nsysu_drone_control -> /ros2_ws/src/nsysu_drone_control
```

Because of these mounts, host edits are visible inside Docker automatically.

## 1. Container Terminal 1: Rebuild ROS Workspace

Use this after changing URDF/Xacro, launch files, worlds, or ROS package files:

```bash
cd /ros2_ws
colcon build --symlink-install --packages-select nsysu_drone_description nsysu_drone_bringup nsysu_drone_control
source install/setup.bash
```

You do **not** need to rebuild the Docker image for these homework edits.

You do **not** need this rebuild for pure Python edits under:

```text
/workspace/HW2_Work/part2
```

## 2. Container Terminal 1: Launch Gazebo/RViz

Still inside Docker:

```bash
launch_drone
```

Keep this terminal running.

Open VNC:

```text
127.0.0.1:5901
password: nsysudrone
```

Expected visual result:

- Gazebo Classic opens.
- RViz opens.
- Drone appears in the world.
- Sonar/ray cones may appear as blue/purple fan shapes if sensor visualization is enabled.

## 3. Host Terminal 2: Enter The Same Container

Open a second host terminal:

```bash
docker exec -it nsysu_drone_vnc bash
```

Then source the workspace:

```bash
source /ros2_ws/install/setup.bash
```

Use this second Docker terminal for sonar checks, training, and testing.

## 4. Install Python RL Dependencies

Inside Docker terminal 2:

```bash
python3 -m pip --version
```

If pip is missing:

```bash
apt update
apt install -y python3-pip
```

Install the RL packages:

```bash
python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas
```

Why `"numpy<2"`:

- Some installed compiled packages, especially OpenCV/CV2 dependencies, may not work cleanly with NumPy 2 inside this container.
- Pinning NumPy below 2 avoids the `_ARRAY_API not found` warning/crash path.

## 5. Verify Six Sonar Topics

Inside Docker terminal 2:

```bash
ros2 topic list | grep sonar
```

Expected sonar topics:

```text
/simple_drone/sonar/out
/simple_drone/front_sonar_left/out
/simple_drone/front_sonar_center/out
/simple_drone/front_sonar_right/out
/simple_drone/front_sonar_up/out
/simple_drone/front_sonar_down/out
```

Echo each topic once:

```bash
ros2 topic echo --once /simple_drone/sonar/out
ros2 topic echo --once /simple_drone/front_sonar_left/out
ros2 topic echo --once /simple_drone/front_sonar_center/out
ros2 topic echo --once /simple_drone/front_sonar_right/out
ros2 topic echo --once /simple_drone/front_sonar_up/out
ros2 topic echo --once /simple_drone/front_sonar_down/out
```

If any front sonar topic is missing:

1. Stop `launch_drone`.
2. Rebuild the ROS workspace.
3. Source `install/setup.bash`.
4. Launch again.

## 6. Manual Drone Movement For Sonar Check

Take off:

```bash
ros2 topic pub /simple_drone/takeoff std_msgs/msg/Empty {} --once
```

Move forward slowly:

```bash
ros2 topic pub /simple_drone/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" --rate 5
```

Stop:

```bash
ros2 topic pub /simple_drone/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" --once
```

Move sideways:

```bash
ros2 topic pub /simple_drone/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.3, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" --rate 5
```

Move upward:

```bash
ros2 topic pub /simple_drone/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.0, z: 0.3}, angular: {x: 0.0, y: 0.0, z: 0.0}}" --rate 5
```

Reset:

```bash
ros2 topic pub /simple_drone/reset std_msgs/msg/Empty {} --once
```

Use this only to verify that the sonar readings react to the world. Training uses `train.py`, not manual commands.

## 7. Smoke Train PPO

Keep Gazebo running in Docker terminal 1.

In Docker terminal 2:

```bash
cd /workspace/HW2_Work/part2
python3 train.py --smoke
```

Expected outputs:

```text
/workspace/HW2_Work/part2/models/ppo_drone.zip
/workspace/HW2_Work/part2/logs/training_curve.png
```

A smoke run only proves the pipeline works. It does not prove good obstacle avoidance yet.

## 8. Test The Current Model

Inside Docker terminal 2:

```bash
cd /workspace/HW2_Work/part2
python3 test.py
```

Expected printed fields:

```text
status: ...
final_distance_to_target: ...
minimum_front_sonar_range: ...
minimum_down_sonar_range: ...
safety_filter_overrides: ...
total_episode_reward: ...
```

Useful interpretation:

- `success`: reached the target.
- `timeout`: did not reach target in time.
- `unsafe_front_sonar`: obstacle got too close.
- `unsafe_down_sonar` or `crash`: flew too low.
- `out_of_bounds`: left the safe flight area.
- Many `safety_filter_overrides` means PPO is still choosing risky actions.

## 9. Longer Training

Run this after smoke training works:

```bash
cd /workspace/HW2_Work/part2
python3 train.py --timesteps 50000
```

For a better final result, try:

```bash
python3 train.py --timesteps 100000
```

Training is slow because Gazebo is the environment. A longer run is more meaningful than a smoke run.

## 10. Collect Report Evidence

Save or screenshot these:

```bash
ros2 topic list | grep sonar
```

```bash
ros2 topic echo --once /simple_drone/front_sonar_center/out
ros2 topic echo --once /simple_drone/front_sonar_up/out
ros2 topic echo --once /simple_drone/front_sonar_down/out
```

Also save:

```text
HW2_Work/part2/logs/training_curve.png
```

Run final test and copy the printed result into the report:

```bash
cd /workspace/HW2_Work/part2
python3 test.py
```

## 11. Part 1 Evidence If Time Allows

Run the simple controller:

```bash
cd /ros2_ws
source install/setup.bash
ros2 run nsysu_drone_control fly_straight
```

If `ros2 run` does not find it, use the source script directly:

```bash
python3 /ros2_ws/src/nsysu_drone_control/fly_straight.py
```

Capture:

- Gazebo/RViz screenshot.
- Terminal output.
- Notes for several target positions if time allows.

## 12. Git Checkpoint And Push

From the host repo:

```bash
cd ~/HW2/nsysu_drone
git status
git log --oneline -5
git push
```

If push asks for a password, use a GitHub token, not your GitHub account password.
