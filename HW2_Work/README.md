# HW2 Work: Task D Sonar Obstacle Avoidance

This folder contains the Part 2 implementation for NSYSU Drone RL HW2.

Task: **Task D - Autonomous Obstacle Avoidance using sonar, not camera**.

The goal is a minimal working PPO pipeline using ROS 2 + Gazebo Classic, Gymnasium, Stable-Baselines3, and the simulator's sonar range topic.

## Environment

- Host: Ubuntu 22.04
- Docker image: `nsysu_drone_vnc:iron`
- Container name: `nsysu_drone_vnc`
- ROS 2 distro inside Docker: Iron
- Simulator: Gazebo Classic + RViz through VNC
- VNC: `127.0.0.1:5901`
- VNC password: `nsysudrone`

## Start Docker

From the host repo:

```bash
cd ~/HW2/nsysu_drone
GPU_ID=0 ./run_docker.sh
```

This mounts:

```text
~/HW2/nsysu_drone/HW2_Work -> /workspace/HW2_Work
~/HW2/nsysu_drone/nsysu_drone_description -> /ros2_ws/src/nsysu_drone_description
~/HW2/nsysu_drone/nsysu_drone_bringup -> /ros2_ws/src/nsysu_drone_bringup
~/HW2/nsysu_drone/nsysu_drone_control -> /ros2_ws/src/nsysu_drone_control
```

Because the ROS source packages are mounted, you do not need `docker cp` after editing package files on the host. If you edit URDF/Xacro, launch files, package files, or C++ code, rebuild the ROS workspace inside Docker:

```bash
cd /ros2_ws
colcon build --symlink-install --packages-select nsysu_drone_description nsysu_drone_bringup nsysu_drone_control
source install/setup.bash
```

For pure `HW2_Work/part2/*.py` edits, no ROS rebuild is needed.

Open another shell into the container:

```bash
docker exec -it nsysu_drone_vnc bash
```

Install RL Python dependencies once inside the container:

```bash
python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas
```

## Launch Gazebo/RViz

Inside the container:

```bash
launch_drone
```

Keep Gazebo running while training or testing.

## Train

In a second container shell:

```bash
cd /workspace/HW2_Work/part2
python3 train.py --smoke
```

If the smoke test works, run longer training:

```bash
python3 train.py --timesteps 50000
```

Expected outputs:

```text
HW2_Work/part2/models/ppo_drone.zip
HW2_Work/part2/logs/monitor.csv
HW2_Work/part2/logs/training_curve.png
```

## Test

With Gazebo still running:

```bash
cd /workspace/HW2_Work/part2
python3 test.py
```

The test script prints:

- status: success, timeout, crash, out_of_bounds, unsafe_front_sonar, unsafe_down_sonar, or invalid_sensor
- final distance to target
- minimum front sonar range
- minimum downward sonar range
- total episode reward

## ROS Topics

The environment uses:

| Topic | Type | Use |
| --- | --- | --- |
| `/simple_drone/cmd_vel` | `geometry_msgs/Twist` | PPO action output |
| `/simple_drone/gt_pose` | `geometry_msgs/Pose` | Position observation |
| `/simple_drone/gt_vel` | `geometry_msgs/Twist` | Velocity observation |
| `/simple_drone/takeoff` | `std_msgs/Empty` | Takeoff after reset |
| `/simple_drone/reset` | `std_msgs/Empty` | Episode reset |
| `/simple_drone/sonar/out` | `sensor_msgs/Range` | Downward sonar, altitude/ground safety cue |
| `/simple_drone/front_sonar_left/out` | `sensor_msgs/Range` | Left-front obstacle cue |
| `/simple_drone/front_sonar_center/out` | `sensor_msgs/Range` | Center-front obstacle cue |
| `/simple_drone/front_sonar_right/out` | `sensor_msgs/Range` | Right-front obstacle cue |
| `/simple_drone/front_sonar_up/out` | `sensor_msgs/Range` | Upward-pitched front obstacle cue |
| `/simple_drone/front_sonar_down/out` | `sensor_msgs/Range` | Downward-pitched front obstacle cue |

## Design Notes

Observation vector:

```text
[x, y, z,
 vx, vy, vz,
 target_x, target_y, target_z,
 dx_to_target, dy_to_target, dz_to_target,
 distance_to_target,
 down_sonar_range,
 front_sonar_left, front_sonar_center, front_sonar_right,
 front_sonar_up, front_sonar_down,
 previous_front_left, previous_front_center, previous_front_right,
 previous_front_up, previous_front_down,
 min_recent_front_sonar_range,
 front_risk_trend]
```

Action vector:

```text
[vx_cmd, vy_cmd, vz_cmd]
```

Limits:

```text
vx_cmd: [-1.0, 1.0]
vy_cmd: [-1.0, 1.0]
vz_cmd: [-0.5, 0.5]
```

Reward terms:

- progress toward target
- small distance penalty
- small action penalty
- obstacle/sonar proximity penalty
- success bonus
- crash, out-of-bounds, unsafe-sonar, and invalid-sensor penalties

Termination conditions:

- target reached
- timeout
- altitude too low
- out of bounds
- sonar range too small
- invalid sensor state

## Sonar Limitation

The stock simulator originally provided a single downward `sensor_msgs/Range` sonar output. For Task D, this workspace uses six total sonar sensors: one downward safety sonar plus five front-facing sectors. The front sectors distinguish left, center, right, upward-pitched, and downward-pitched obstacle risk.

The vertical front sectors are important because the PPO action includes `vz_cmd`. If the center or lower front sonar reports danger, the policy has state information that can support climbing behavior instead of only steering left or right.

The literature notes in `Homework-files/papers/literature_design_notes.md` support this design choice: process sonar into risk features, add short-term memory, and keep explicit safety logic around the learned policy.
