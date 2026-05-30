# Training Design: Part 3 Curriculum

This document describes the current training workflow for `HW2_Work/part3`.

## Why A Curriculum

Directly training a drone to fly to a far goal while avoiding obstacles is unstable. The drone first needs to learn:

1. how `vz_cmd` changes altitude,
2. how `vx_cmd` moves forward/backward,
3. how to combine x and z control,
4. how to stop near a target,
5. how to react to sonar risk.

The curriculum trains these skills in order and resumes PPO checkpoints from the previous stage.

## Stages

| Stage | Variant | Description | Target |
| --- | --- | --- | --- |
| 1 | A | Fixed vertical target | `(0, 0, 1.2)` |
| 1 | B | Random vertical target | `z in [0.7, 1.8]` |
| 2 | A | Fixed horizontal target | `(1, 0, 0.8)` |
| 2 | B | Random horizontal target | `x in [-1, 2]` |
| 3 | A | Random x-z target | `x in [-1, 2.5]`, `z in [0.7, 1.8]` |
| 3 | B | Three sequential random targets | A, B, C |
| 4 | A | One-obstacle sonar avoidance | `(10, 0, 1)` |
| 5 | A | Multi-obstacle sonar avoidance | `(10, 0, 1)` |

## PPO Settings

Typical settings:

```text
policy = MlpPolicy
learning_rate = 0.0003
n_steps = 512
batch_size = 64
gamma = 0.99
device = cpu
step_dt = 0.05
```

`step_dt=0.05` is used as a deadline-friendly speed-up. The same value should be used during evaluation.

## Checkpoints

Each run saves:

```text
ppo_drone.zip
best/best_episode_model.zip
best/best_average_model.zip
best/best_success_model.zip
best/best_precision_model.zip
run_config.json
monitor.csv
training_curve.csv
training_curve.png
```

For curriculum transfer, prefer:

```text
best/best_precision_model.zip
```

It is selected using success status and final distance/error, so it is usually better than reward alone.

## Early Plateau Stop

Training supports plateau stopping:

```bash
--early-stop-plateau \
--plateau-window 50 \
--plateau-patience 60 \
--plateau-min-delta 0.5
```

Meaning:

- `plateau-window`: average reward over the latest N episodes,
- `plateau-patience`: how many checks to wait without enough improvement,
- `plateau-min-delta`: minimum moving-average reward improvement counted as real progress.

## Stage 4 Command

Launch the Stage 4 world first:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage4_obstacle.world
```

Then train:

```bash
cd /workspace/HW2_Work/part3

python3 train.py \
  --stage 4 \
  --resume-from models/stage3/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.25 \
  --max-steps 1800 \
  --timesteps 120000 \
  --step-dt 0.05 \
  --log-position-every 50 \
  --early-stop-plateau \
  --plateau-window 50 \
  --plateau-patience 80 \
  --plateau-min-delta 1.0
```

## Stage 5 Command

Launch the Stage 5 world first:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage5_obstacle.world
```

Then train:

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

Important: `train.py` does not load Gazebo worlds. The world must already be launched.
