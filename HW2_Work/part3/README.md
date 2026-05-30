# HW2 Part 3: Six-Stage PPO Curriculum

Part 3 is the current working implementation for the homework. It replaces the earlier Part 2 experiments with a clean staged curriculum and keeps one fixed PPO interface across stages.

## Design

Gazebo axes:

```text
x = forward/back
y = left/right
z = altitude
```

Action:

```text
[vx_cmd, vy_cmd, vz_cmd]
vx, vy in [-1.0, 1.0]
vz in [-0.5, 0.5]
```

Observation stays fixed at 41 values. It includes position, velocity, relative target vector, distance, target progress, sonar slots, sonar trends, and a sonar-enabled flag.

Stages 1-3 do not use sonar. Their sonar values are masked to safe constants so checkpoints can transfer into later stages. Stages 4-6 activate real sonar topics and sonar-risk rewards.

## Curriculum

| Stage | Purpose | Target Setup | Sonar |
| --- | --- | --- | --- |
| 1A | Fixed altitude control | `(0, 0, 1.2)` | masked |
| 1B | Random altitude control | `z in [0.7, 1.8]` | masked |
| 2A | Fixed x movement | `(1, 0, 0.8)` | masked |
| 2B | Random x movement | `x in [-1, 2]` | masked |
| 3A | Random x/z navigation | random x/z target | masked |
| 3B | Sequential navigation | 3 random targets | masked |
| 4 | One-obstacle avoidance | mission goal `(10, 0, 1)` | active |
| 5 | Multi-obstacle avoidance | mission goal `(10, 0, 1)` | active |
| 6 | Sequential obstacle mission | future extension | active |

Stage 4 and Stage 5 use an internal dynamic local subgoal about `1 m` ahead in `x` to help long-distance progress. This local subgoal is not a hand-authored avoidance path. The visible Gazebo ball marks the final mission target.

## Current Evaluation Summary

| Stage | Evaluation |
| --- | --- |
| 1A | 100% success, average distance about `0.015 m` |
| 1B | 100% success, average distance about `0.109 m` |
| 2A | 100% success, average distance about `0.097 m` |
| 2B | 100% success, average distance about `0.082 m` |
| 3A | 100% success, average distance about `0.104 m` |
| 3B | 90% success, average distance about `0.155 m` |
| 4 | 80% success, 20% unsafe sonar |

Stage 5 is prepared as the multi-obstacle continuation.

## Syntax Check

```bash
cd /workspace/HW2_Work/part3
python3 -m py_compile drone_env.py train.py test.py
```

## Training Commands

Stage 1A:

```bash
python3 train.py --stage 1 --variant A --timesteps 30000 --step-dt 0.05
```

Stage 1B:

```bash
python3 train.py \
  --stage 1 \
  --variant B \
  --resume-from models/stage1/variantA/runXXX/best/best_precision_model.zip \
  --timesteps 50000 \
  --step-dt 0.05
```

Stage 2A:

```bash
python3 train.py \
  --stage 2 \
  --variant A \
  --resume-from models/stage1/variantB/run001/best/best_precision_model.zip \
  --timesteps 50000 \
  --step-dt 0.05
```

Stage 2B:

```bash
python3 train.py \
  --stage 2 \
  --variant B \
  --resume-from models/stage2/variantA/run001/best/best_precision_model.zip \
  --timesteps 50000 \
  --step-dt 0.05
```

Stage 3A:

```bash
python3 train.py \
  --stage 3 \
  --variant A \
  --resume-from models/stage2/variantB/run001/best/best_precision_model.zip \
  --timesteps 40000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 3B:

```bash
python3 train.py \
  --stage 3 \
  --variant B \
  --resume-from models/stage3/variantA/run002/best/best_precision_model.zip \
  --timesteps 80000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 4:

```bash
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

Stage 5:

```bash
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

## Evaluation Example

```bash
python3 test.py \
  --stage 4 \
  --model models/stage4/run004/best/best_precision_model.zip \
  --success-distance 0.25 \
  --max-steps 1800 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

## Gazebo Worlds

Launch the correct world before training/testing obstacle stages.

Stage 4:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage4_obstacle.world
```

Stage 5:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage5_obstacle.world
```

`train.py` does not load worlds. It uses whatever Gazebo world is already running.

## Outputs

Training:

```text
models/stageN/.../runXXX/
logs/stageN/.../runXXX/
```

Evaluation:

```text
logs/eval/stageN/.../runXXX/evalXXX.csv
logs/eval/stageN/.../runXXX/evalXXX_config.json
```

Important saved models:

```text
best/best_episode_model.zip
best/best_average_model.zip
best/best_success_model.zip
best/best_precision_model.zip
```
