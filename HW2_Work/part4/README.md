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

## Train And Test Commands

Use `--success-distance 0.10` for Stages 1-4. These stages are basic
navigation lessons, so the drone should learn fairly precise target reaching.
Use `--success-distance 0.20` for Stages 5-6 because the far obstacle mission is
longer and the avoidance path can naturally finish with more lateral error.

Run the stages in order. If you retrain a stage and the run number changes,
replace `run001` or `run002` in the next command with your newest successful
run.

### Stage 1: Altitude Control

Stage 1A train:

```bash
python3 train.py \
  --stage 1 \
  --variant A \
  --success-distance 0.10 \
  --timesteps 30000 \
  --step-dt 0.05
```

Stage 1A test:

```bash
python3 test.py \
  --stage 1 \
  --variant A \
  --model models/stage1/variantA/run002/best/best_precision_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 1B train:

```bash
python3 train.py \
  --stage 1 \
  --variant B \
  --resume-from models/stage1/variantA/run002/best/best_precision_model.zip \
  --success-distance 0.10 \
  --timesteps 50000 \
  --step-dt 0.05
```

Stage 1B test:

```bash
python3 test.py \
  --stage 1 \
  --variant B \
  --model models/stage1/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

### Stage 2: X-Axis Movement

Stage 2A train:

```bash
python3 train.py \
  --stage 2 \
  --variant A \
  --resume-from models/stage1/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --timesteps 50000 \
  --step-dt 0.05
```

Stage 2A test:

```bash
python3 test.py \
  --stage 2 \
  --variant A \
  --model models/stage2/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 2B train:

```bash
python3 train.py \
  --stage 2 \
  --variant B \
  --resume-from models/stage2/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --timesteps 50000 \
  --step-dt 0.05
```

Stage 2B test:

```bash
python3 test.py \
  --stage 2 \
  --variant B \
  --model models/stage2/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

### Stage 3: Y-Axis Movement

Stage 3A train:

```bash
python3 train.py \
  --stage 3 \
  --variant A \
  --resume-from models/stage2/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --timesteps 30000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 3A test:

```bash
python3 test.py \
  --stage 3 \
  --variant A \
  --model models/stage3/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 3B train:

```bash
python3 train.py \
  --stage 3 \
  --variant B \
  --resume-from models/stage3/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --timesteps 50000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 3B test:

```bash
python3 test.py \
  --stage 3 \
  --variant B \
  --model models/stage3/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

### Stage 4: Combined X/Y/Z Navigation

Stage 4A train:

```bash
python3 train.py \
  --stage 4 \
  --variant A \
  --resume-from models/stage3/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --timesteps 50000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 4A test:

```bash
python3 test.py \
  --stage 4 \
  --variant A \
  --model models/stage4/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 4B train:

```bash
python3 train.py \
  --stage 4 \
  --variant B \
  --resume-from models/stage4/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --timesteps 80000 \
  --step-dt 0.05 \
  --early-stop-plateau
```

Stage 4B test:

```bash
python3 test.py \
  --stage 4 \
  --variant B \
  --model models/stage4/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

### Stage 5: One-Obstacle Sonar Avoidance

Launch the Stage 5 world first:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage4_obstacle.world
```

Stage 5 train:

```bash
python3 train.py \
  --stage 5 \
  --resume-from models/stage4/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.20 \
  --max-steps 1800 \
  --timesteps 120000 \
  --step-dt 0.05 \
  --log-position-every 50 \
  --early-stop-plateau \
  --plateau-window 50 \
  --plateau-patience 80 \
  --plateau-min-delta 1.0
```

Stage 5 test:

```bash
python3 test.py \
  --stage 5 \
  --model models/stage5/run001/best/best_precision_model.zip \
  --success-distance 0.20 \
  --max-steps 1800 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

### Stage 6: Multi-Obstacle Sonar Avoidance

Launch the Stage 6 world first:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage5_obstacle.world
```

Stage 6 train:

```bash
python3 train.py \
  --stage 6 \
  --resume-from models/stage5/run001/best/best_precision_model.zip \
  --success-distance 0.20 \
  --max-steps 2200 \
  --timesteps 80000 \
  --step-dt 0.05 \
  --log-position-every 100 \
  --early-stop-plateau \
  --plateau-window 50 \
  --plateau-patience 60 \
  --plateau-min-delta 0.5
```

Stage 6 test:

```bash
python3 test.py \
  --stage 6 \
  --model models/stage6/run001/best/best_precision_model.zip \
  --success-distance 0.20 \
  --max-steps 2200 \
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
