# HW2 Part 3+Y: Lateral Stage PPO Curriculum

Part 3+Y is a copy of Part 3 with one new sideway movement stage inserted between the old Stage 2 and old Stage 3.

The copied models and logs intentionally stop at Stage 2B. This lets the new curriculum continue from the trained Stage 2B checkpoint, then learn `y` movement before going back to random x/z navigation and obstacle stages.

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

Observation stays fixed at 41 values. Sonar fields are present from Stage 1, but they are masked until Stage 5.

## Curriculum

| Stage | Purpose | Target Setup | Sonar |
| --- | --- | --- | --- |
| 1A | Fixed altitude control | `(0, 0, 1.2)` | masked |
| 1B | Random altitude control | `z in [0.7, 1.8]` | masked |
| 2A | Fixed x movement | `(1, 0, 0.8)` | masked |
| 2B | Random x movement | `x in [-1, 2]` | masked |
| 3A | Fixed sideway movement | `(0, 1, 0.8)` | masked |
| 3B | Random sideway movement | `y in [-1.5, 1.5]`, `x = 0`, `z = 0.8` | masked |
| 4A | Random x/z navigation | random x/z target | masked |
| 4B | Sequential navigation | 3 random x/z targets | masked |
| 5 | One-obstacle avoidance | mission goal `(10, 0, 1)` | active |
| 6 | Multi-obstacle avoidance | mission goal `(10, 0, 1)` | active |
| 7 | Sequential obstacle mission | future extension | active |

Stage 5 uses an internal dynamic local subgoal about `1 m` ahead in `x` to help long-distance progress. This local subgoal is not a hand-authored avoidance path. The visible Gazebo ball marks the final mission target.

## Syntax Check

```bash
cd /workspace/HW2_Work/part3_plusY
python3 -m py_compile drone_env.py train.py test.py
```

## Training Commands

Stage 1 and Stage 2 artifacts were copied from `part3`, so the first new training run should start at Stage 3A.

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
  --resume-from models/stage3/variantA/run001/best/best_precision_model.zip \
  --timesteps 60000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 4A:

```bash
python3 train.py \
  --stage 4 \
  --variant A \
  --resume-from models/stage3/variantB/run001/best/best_precision_model.zip \
  --timesteps 40000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 4B:

```bash
python3 train.py \
  --stage 4 \
  --variant B \
  --resume-from models/stage4/variantA/run001/best/best_precision_model.zip \
  --timesteps 80000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 5:

```bash
python3 train.py \
  --stage 5 \
  --resume-from models/stage4/variantB/run001/best/best_precision_model.zip \
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

Stage 6:

```bash
python3 train.py \
  --stage 6 \
  --resume-from models/stage5/run001/best/best_precision_model.zip \
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
  --stage 3 \
  --variant A \
  --model models/stage3/variantA/run001/best/best_precision_model.zip \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

## Gazebo Worlds

Launch the correct world before training/testing obstacle stages.

Stage 5:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage4_obstacle.world
```

Stage 6:

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
