# Part 2 / Final RL Submission

This folder contains the final RL implementation used for the report. The original project folder was `HW2_Work/part3`, but it is placed here as `part2/` to match the homework submission format.

## Main Files

- `drone_env.py`: Gymnasium environment for the ROS 2/Gazebo drone.
- `train.py`: PPO training script.
- `test.py`: TA evaluation script.
- `models/ppo_drone.zip`: shortcut copy of the main Stage 4 model, copied from `models/stage4/run004/best/best_precision_model.zip`.
- `models/`: full staged model folders for Stages 1-5.
- `logs/`: training curves, monitor files, and evaluation CSV files.
- `logs/training_curve.png`: shortcut copy of the Stage 4 training curve.
- `worlds/stage4_obstacle.world`: one-obstacle evaluation world.
- `worlds/stage5_obstacle.world`: multi-obstacle extension world.
- `requirements.txt`: Python packages expected by the RL scripts, excluding ROS packages that come from the ROS 2 workspace.

## Recommended TA Demo

Use the provided ROS 2/Gazebo homework environment. In each terminal, source ROS and the built workspace if they are not already sourced:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
```

Start Gazebo with the Stage 4 obstacle world:

```bash
cd /workspace/HW2_StudentID_Name/part2
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=$(pwd)/worlds/stage4_obstacle.world
```

Then run the trained PPO policy:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
cd /workspace/HW2_StudentID_Name/part2
python3 test.py \
  --stage 4 \
  --model models/ppo_drone.zip \
  --success-distance 0.25 \
  --max-steps 1800 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

The full original Stage 4 model path is also included:

```text
models/stage4/run004/best/best_precision_model.zip
```

## Optional Stage 5 Extension

Stage 5 was attempted as a harder multi-obstacle extension. It is included for completeness, but the report explains that it currently fails with unsafe sonar terminations.

```bash
cd /workspace/HW2_StudentID_Name/part2
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=$(pwd)/worlds/stage5_obstacle.world
```

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
cd /workspace/HW2_StudentID_Name/part2
python3 test.py \
  --stage 5 \
  --model models/stage5/run006/best/best_precision_model.zip \
  --success-distance 0.25 \
  --max-steps 2200 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

## Notes

`train.py` and `test.py` do not launch Gazebo by themselves. They connect to whichever ROS 2/Gazebo world is already running.

If the folder was unzipped somewhere other than `/workspace`, replace `/workspace/HW2_StudentID_Name` with the actual path.
