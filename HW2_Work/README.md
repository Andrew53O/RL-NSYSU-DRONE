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
python3 train.py --smoke --stage 1
```

If the smoke test works, run longer training:

```bash
python3 train.py --stage 1 --timesteps 50000
python3 train.py --stage 2 --timesteps 50000
python3 train.py --stage 3 --timesteps 50000
python3 train.py --stage 4 --timesteps 50000
```

Expected outputs:

```text
HW2_Work/part2/models/ppo_drone.zip
HW2_Work/part2/models/ppo_drone_stage1.zip
HW2_Work/part2/logs/monitor_stage1.csv
HW2_Work/part2/logs/training_curve_stage1.png
```

## Test

With Gazebo still running:

```bash
cd /workspace/HW2_Work/part2
python3 test.py --episodes 5 --csv logs/eval_metrics.csv
```

The test script prints:

- success, crash, and timeout rates
- average return
- average minimum obstacle sonar distance
- average steps to target
- safety filter activation count
- side sonar near-miss count
- per-episode CSV rows in `logs/eval_metrics.csv`

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
| `/simple_drone/side_sonar_left/out` | `sensor_msgs/Range` | Left side-wall obstacle cue |
| `/simple_drone/side_sonar_right/out` | `sensor_msgs/Range` | Right side-wall obstacle cue |

Verify all sonar topics after rebuilding and launching Gazebo:

```bash
ros2 topic list | grep sonar
ros2 topic echo --once /simple_drone/side_sonar_left/out
ros2 topic echo --once /simple_drone/side_sonar_right/out
```

Run a local syntax smoke check before container training:

```bash
cd /workspace/HW2_Work/part2
python3 -m py_compile drone_env.py train.py test.py
```

## Design Notes

Observation vector:

```text
[x/8, y/8, z/5,
 vx, vy, vz/0.5,
 dx_to_target/8, dy_to_target/8, dz_to_target/5,
 distance_to_target/12,
 front_left_range/10, front_center_range/10, front_right_range/10,
 front_up_range/10, front_down_range/10,
 side_left_range/10, side_right_range/10,
 front_left_risk, front_center_risk, front_right_risk,
 front_up_risk, front_down_risk, side_left_risk, side_right_risk,
 previous_front_left_range/10, previous_front_center_range/10,
 previous_front_right_range/10, previous_front_up_range/10,
 previous_front_down_range/10, previous_side_left_range/10,
 previous_side_right_range/10,
 front_left_trend, front_center_trend, front_right_trend,
 front_up_trend, front_down_trend, side_left_trend, side_right_trend,
 min_recent_obstacle_range/10,
 down_sonar_range/10,
 down_sonar_risk,
 left_right_risk_balance,
 up_down_risk_balance]
```

The observation has 43 values. Sonar risk is computed from the caution distance, so `0` means clear and `1` means close/unsafe.

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

- weighted progress toward target
- distance-to-target penalty
- mean and maximum obstacle-sonar risk penalties
- obstacle-sonar approach-trend penalty
- downward-sonar risk penalty
- action magnitude and action-smoothness penalties
- small penalty when the emergency safety filter overrides an action
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

The stock simulator originally provided a single downward `sensor_msgs/Range` sonar output. For Task D, this workspace uses eight total sonar sensors: one downward safety sonar, five front-facing sectors, and two side-facing sectors. The front sectors distinguish left, center, right, upward-pitched, and downward-pitched obstacle risk. The side sectors reduce the wall-observability blind spot during lateral motion.

The vertical front sectors are important because the PPO action includes `vz_cmd`. If the center or lower front sonar reports danger, the policy has state information that can support climbing behavior instead of only steering left or right.

The learned PPO action is treated as a nominal command. A small emergency safety filter slows or redirects commands when front sonar is dangerously close, a side sonar is near a wall, or the downward sonar is too near the ground.

The literature notes in `Homework-files/papers/literature_design_notes.md` support this design choice: process sonar into risk features, add short-term memory, and keep explicit safety logic around the learned policy.
