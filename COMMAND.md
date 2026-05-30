# HW2 Command Runbook

This is the current command reference for the Part 3 PPO curriculum. The final report focuses on **Task D: sonar-based obstacle avoidance**.

## Host: Start Docker

```bash
cd ~/HW2/nsysu_drone
GPU_ID=0 ./run_docker.sh
```

Open a second terminal:

```bash
docker exec -it nsysu_drone_vnc bash
source /ros2_ws/install/setup.bash
```

## Rebuild ROS Workspace

Run this after changing worlds, launch files, URDF/Xacro, or ROS package files:

```bash
cd /ros2_ws
colcon build --symlink-install --packages-select nsysu_drone_description nsysu_drone_bringup nsysu_drone_control
source install/setup.bash
```

No rebuild is needed for pure `HW2_Work/part3/*.py` edits.

## Launch Gazebo

Default world:

```bash
launch_drone
```

Stage 4 one-obstacle world:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage4_obstacle.world
```

Stage 5 multi-obstacle world:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage5_obstacle.world
```

Important: `train.py` and `test.py` do not load a world file. Start the correct world first, then run training or testing in another terminal.

## Verify Sonar

```bash
ros2 topic list | grep sonar
ros2 topic echo --once /simple_drone/front_sonar_center/out
ros2 topic echo --once /simple_drone/front_sonar_left/out
ros2 topic echo --once /simple_drone/front_sonar_right/out
```

Expected obstacle-stage sonar topics include:

```text
/simple_drone/sonar/out
/simple_drone/front_sonar_left/out
/simple_drone/front_sonar_center/out
/simple_drone/front_sonar_right/out
/simple_drone/front_sonar_up/out
/simple_drone/front_sonar_down/out
/simple_drone/side_sonar_left/out
/simple_drone/side_sonar_right/out
```

## Syntax Check

```bash
cd /workspace/HW2_Work/part3
python3 -m py_compile drone_env.py train.py test.py
```

## Train Stage 5

Launch `stage5_obstacle.world` first, then:

```bash
cd /workspace/HW2_Work/part3
python3 train.py \
  --stage 5 \
  --resume-from models/stage4/run004/best/best_precision_model.zip \
  --success-distance 0.25 \
  --max-steps 2200 \
  --timesteps 80000 \
  --step-dt 0.05 \
  --log-position-every 100 \
  --early-stop-plateau \
  --plateau-window 50 \
  --plateau-patience 60 \
  --plateau-min-delta 0.5
```

## Test Stage 4

Launch `stage4_obstacle.world` first, then:

```bash
cd /workspace/HW2_Work/part3
python3 test.py \
  --stage 4 \
  --model models/stage4/run004/best/best_precision_model.zip \
  --success-distance 0.25 \
  --max-steps 1800 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

## Test Stage 5

Launch `stage5_obstacle.world` first, then replace `runXXX` with the trained run:

```bash
cd /workspace/HW2_Work/part3
python3 test.py \
  --stage 5 \
  --model models/stage5/runXXX/best/best_precision_model.zip \
  --success-distance 0.25 \
  --max-steps 2200 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

## Current Report Results

| Stage | Result |
| --- | --- |
| 1A | 100% success |
| 1B | 100% success |
| 2A | 100% success |
| 2B | 100% success |
| 3A | 100% success |
| 3B | 90% success |
| 4 | 80% success, 20% unsafe sonar |
| 5 | In progress |

## Output Locations

Training:

```text
HW2_Work/part3/models/
HW2_Work/part3/logs/
```

Evaluation:

```text
HW2_Work/part3/logs/eval/
```
