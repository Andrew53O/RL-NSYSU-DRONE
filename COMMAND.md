# Commands

Quick commands for running Task D sonar obstacle avoidance.

## Host terminal

From the repository root:

```bash
cd ~/HW2/nsysu_drone
GPU_ID=0 ./run_docker.sh
```

Open another shell inside the running container:

```bash
docker exec -it nsysu_drone_vnc bash
```

## Container terminal 1: build and launch simulation

Rebuild after editing ROS package files:

```bash
cd /ros2_ws
colcon build --symlink-install --packages-select nsysu_drone_description nsysu_drone_bringup nsysu_drone_control
source install/setup.bash
```

Launch Gazebo/RViz:

```bash
launch_drone
```

## Container terminal 2: install, train, test

Install RL dependencies once:

```bash
python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas
```

Smoke train:

```bash
cd /workspace/HW2_Work/part2
python3 train.py --smoke
```

Test:

```bash
cd /workspace/HW2_Work/part2
python3 test.py
```

Longer training:

```bash
cd /workspace/HW2_Work/part2
python3 train.py --timesteps 50000
```

## Sonar checks

List sonar topics:

```bash
ros2 topic list | grep sonar
```

Echo sonar topics:

```bash
ros2 topic echo --once /simple_drone/sonar/out
ros2 topic echo --once /simple_drone/front_sonar_left/out
ros2 topic echo --once /simple_drone/front_sonar_center/out
ros2 topic echo --once /simple_drone/front_sonar_right/out
ros2 topic echo --once /simple_drone/front_sonar_up/out
ros2 topic echo --once /simple_drone/front_sonar_down/out
```

## Manual drone commands

Take off:

```bash
ros2 topic pub /simple_drone/takeoff std_msgs/msg/Empty {} --once
```

Land:

```bash
ros2 topic pub /simple_drone/land std_msgs/msg/Empty {} --once
```

Reset:

```bash
ros2 topic pub /simple_drone/reset std_msgs/msg/Empty {} --once
```

Move forward:

```bash
ros2 topic pub /simple_drone/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" --once
```
