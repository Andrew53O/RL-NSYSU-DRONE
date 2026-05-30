# HW2 Work: Part 3 Sonar RL Curriculum

This folder contains the working reinforcement-learning implementation for NSYSU Drone RL HW2.

The current report direction is **Task D: autonomous obstacle avoidance using sonar**. The earlier Part 2 experiments are kept for reference, but the final working pipeline is in:

```text
HW2_Work/part3
```

Part 3 uses a six-stage PPO curriculum:

1. Stage 1A/1B: vertical altitude control on Gazebo `z`
2. Stage 2A/2B: horizontal movement on Gazebo `x`
3. Stage 3A/3B: combined navigation and sequential targets without sonar
4. Stage 4: long-distance target at `(10, 0, 1)` with one sonar obstacle
5. Stage 5: same long target with multiple sonar obstacles
6. Stage 6: planned extension for sequential targets plus obstacles

Sonar slots are present in the observation for every stage, but Stages 1-3 mask them to safe values. Real sonar is enabled from Stage 4 onward.

## Docker

From the host:

```bash
cd ~/HW2/nsysu_drone
GPU_ID=0 ./run_docker.sh
```

Open another shell into the same container:

```bash
docker exec -it nsysu_drone_vnc bash
source /ros2_ws/install/setup.bash
```

If ROS package files, launch files, URDF/Xacro, or world files change, rebuild inside Docker:

```bash
cd /ros2_ws
colcon build --symlink-install --packages-select nsysu_drone_description nsysu_drone_bringup nsysu_drone_control
source install/setup.bash
```

Python files under `/workspace/HW2_Work/part3` do not need a ROS rebuild.

## Launch Worlds

Default world:

```bash
launch_drone
```

Stage 4 world:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage4_obstacle.world
```

Stage 5 world:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage5_obstacle.world
```

Keep Gazebo running while training or testing. `train.py` and `test.py` reset the current Gazebo world; they do not choose or load the world file themselves.

## TA Test Script

The assignment requires a test script that loads a trained model and demonstrates the agent in Gazebo. Use `HW2_Work/part3/test.py`.

First launch the correct Gazebo world. For the main Stage 4 Task D result:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage4_obstacle.world
```

Then run:

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

Expected behavior:

- The drone moves toward the mission goal `(10, 0, 1)`.
- The obstacle near `x=5` activates the front/side sonar.
- The policy usually bends around the obstacle and reaches the far target.
- Stage 4 evaluation result: 8/10 successes, 2/10 unsafe sonar terminations.

Evaluation outputs are saved under:

```text
HW2_Work/part3/logs/eval/
```

For optional Stage 5 demonstration, launch `stage5_obstacle.world` and run:

```bash
cd /workspace/HW2_Work/part3
python3 test.py \
  --stage 5 \
  --model models/stage5/run006/best/best_precision_model.zip \
  --success-distance 0.25 \
  --max-steps 2200 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

## Current Results

The final report uses the Part 3 results:

| Stage | Result Summary |
| --- | --- |
| 1A | 100% success, fixed altitude target |
| 1B | 100% success, random altitude targets |
| 2A | 100% success, fixed horizontal target |
| 2B | 100% success, random horizontal targets |
| 3A | 100% success, random combined x/z target |
| 3B | 90% success, three sequential targets |
| 4 | 80% success, 20% unsafe sonar, one obstacle near x=5 |
| 5 | Multiple-obstacle extension in progress |

Stage 4 is report-worthy because most episodes reach the far mission goal `(10, 0, 1)` while using active sonar and avoiding the obstacle, but the unsafe episodes show that the policy is not yet perfectly robust.

## Key Files

```text
HW2_Work/part3/drone_env.py
HW2_Work/part3/train.py
HW2_Work/part3/test.py
HW2_Work/part3/README.md
RL-DESIGN.md
TRAIN-DESIGN.md
REPORT.md
SECTION3-LITERATURE-REVIEW.md
nsysu_drone_description/worlds/stage4_obstacle.world
nsysu_drone_description/worlds/stage5_obstacle.world
```
