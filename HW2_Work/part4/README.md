# HW2 Part 4: Expanded Six-Stage PPO Curriculum

Part 4 is a fresh fork of the Part 3 PPO environment. The main change is a new
y-axis curriculum stage between x-axis movement and full x/y/z navigation. This
should make the final navigation policy less surprised by lateral commands.

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

Observation stays fixed at 41 values. It includes position, velocity, relative
target vector, distance, target progress, sonar slots, sonar trends, and a
sonar-enabled flag. Stages 1-4 mask sonar to safe constants. Stages 5-6 use real
sonar for obstacle avoidance.

## Curriculum

| Stage | Variant | Purpose | Target Setup | Sonar |
| --- | --- | --- | --- | --- |
| 1 | A | Fixed altitude control | `(0, 0, 1.2)` | masked |
| 1 | B | Random altitude control | `z in [0.7, 1.8]` | masked |
| 2 | A | Fixed x movement | `(1, 0, 0.8)` | masked |
| 2 | B | Random x movement | `x in [-1, 2]` | masked |
| 3 | A | Fixed y movement | `(0, 1, 0.8)` | masked |
| 3 | B | Random y movement | `y in [-1.5, 1.5]` | masked |
| 4 | A | Random x/y/z navigation | random single target | masked |
| 4 | B | Sequential x/y/z navigation | 3 random targets | masked |
| 5 | A | One-obstacle avoidance | mission goal `(10, 0, 1)` | active |
| 6 | A | Multi-obstacle avoidance | mission goal `(10, 0, 1)` | active |

Stages 5 and 6 use an internal dynamic local subgoal about `1 m` ahead in `x`.
The local subgoal only encourages forward progress. It is not a fixed avoidance
trajectory; sideways or vertical avoidance should come from sonar risk.

## Syntax Check

```bash
cd /workspace/HW2_Work/part4
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
  --resume-from models/stage1/variantA/run002/best/best_precision_model.zip \
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

Stage 3A, new fixed y-axis stage:

```bash
python3 train.py \
  --stage 3 \
  --variant A \
  --resume-from models/stage2/variantB/run001/best/best_precision_model.zip \
  --timesteps 30000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 3B, new random y-axis stage:

```bash
python3 train.py \
  --stage 3 \
  --variant B \
  --resume-from models/stage3/variantA/run001/best/best_precision_model.zip \
  --timesteps 50000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 4A, random x/y/z:

```bash
python3 train.py \
  --stage 4 \
  --variant A \
  --resume-from models/stage3/variantB/run001/best/best_precision_model.zip \
  --timesteps 50000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 4B, sequential x/y/z:

```bash
python3 train.py \
  --stage 4 \
  --variant B \
  --resume-from models/stage4/variantA/run001/best/best_precision_model.zip \
  --timesteps 80000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 5, one-obstacle sonar avoidance:

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

Stage 6, multi-obstacle sonar avoidance:

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

## Evaluation Examples

Test the new y-axis stage:

```bash
python3 test.py \
  --stage 3 \
  --variant B \
  --model models/stage3/variantB/run001/best/best_precision_model.zip \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

Test the one-obstacle stage:

```bash
python3 test.py \
  --stage 5 \
  --model models/stage5/run001/best/best_precision_model.zip \
  --success-distance 0.25 \
  --max-steps 1800 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

## Gazebo Worlds

Launch the correct world before training/testing obstacle stages. `train.py`
does not load worlds. It uses whatever Gazebo world is already running.

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
