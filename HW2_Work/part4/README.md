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
| 3 | B | Random y movement | `y in [-1.0, 1.0]` | masked |
| 4 | A | Random x/y/z navigation | random single target | masked |
| 4 | B | Sequential x/y/z navigation | 3 random targets | masked |
| 5 | A | One-obstacle avoidance | mission goal `(10, 0, 1)` | active |
| 5 | B | Random-goal one-obstacle avoidance | random x/y goal, generated obstacle | active |
| 6 | A | Multi-obstacle avoidance | mission goal `(10, 0, 1)` | active |

Stages 5 and 6 use an internal dynamic local subgoal about `1 m` toward the
final goal in 3D. It is not a fixed avoidance trajectory; sideways or vertical
avoidance should come from sonar risk.

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

Episode length is also a curriculum hyperparameter. Shorter early episodes keep
failed exploration from drifting too far, while later stages need more steps for
multi-axis navigation and obstacle avoidance.

| Stage | Variant | `--max-steps` |
| --- | --- | ---: |
| 1 | A/B | 150 |
| 2 | A/B | 200 |
| 3 | A/B | 200 |
| 4 | A | 300 |
| 4 | B | 500 |
| 5 | A/B | 600 |

Early stopping is controlled by the plateau settings in `train.py`. The Part 4
defaults are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--plateau-window` | 30 | Average reward over the latest 30 episodes |
| `--plateau-patience` | 30 | Stop after 30 plateau checks without enough improvement |
| `--plateau-min-delta` | 1.0 | Required reward improvement to reset patience |

Use `--early-stop-plateau` when you want training to finish automatically after
the recent reward curve becomes flat. The default plateau settings are usually
fine for the simple Stage 1-4 curriculum runs:

```bash
--early-stop-plateau \
--plateau-window 30 \
--plateau-patience 30 \
--plateau-min-delta 1.0
```

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
  --max-steps 150 \
  --timesteps 30000 \
  --step-dt 0.05
```

Stage 1A test:

```bash
python3 test.py \
  --stage 1 \
  --variant A \
  --model models/stage1/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --max-steps 150 \
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
  --max-steps 150 \
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
  --max-steps 150 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

### Stage 2: X-Axis Movement

Stage 2 uses target height `z=0.5`, matching the takeoff height, so this stage
isolates x-axis movement instead of asking the drone to climb while moving
forward.

Stage 2A train:

```bash
python3 train.py \
  --stage 2 \
  --variant A \
  --resume-from models/stage1/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --max-steps 200 \
  --timesteps 50000 \
  --step-dt 0.05
  --early-stop-plateau
```

Stage 2A test:

```bash
python3 test.py \
  --stage 2 \
  --variant A \
  --model models/stage2/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --max-steps 200 \
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
  --max-steps 200 \
  --timesteps 50000 \
  --step-dt 0.05
  --early-stop-plateau
```

Stage 2B test:

```bash
python3 test.py \
  --stage 2 \
  --variant B \
  --model models/stage2/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.10 \
  --max-steps 200 \
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
  --max-steps 200 \
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
  --max-steps 200 \
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
  --max-steps 200 \
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
  --max-steps 200 \
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
  --max-steps 300 \
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
  --max-steps 300 \
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
  --max-steps 500 \
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
  --max-steps 500 \
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
  --variant A \
  --resume-from models/stage4/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.20 \
  --max-steps 600 \
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
  --variant A \
  --model models/stage5/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.20 \
  --max-steps 600 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

Stage 5B uses one generated cylinder obstacle on the straight line to a random
mission goal with `x in [5, 10]`, `y in [-1, 1]`, and `z = 1`.

```bash
python3 train.py \
  --stage 5 \
  --variant B \
  --resume-from models/stage5/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.20 \
  --max-steps 600 \
  --timesteps 120000 \
  --step-dt 0.05 \
  --log-position-every 50
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
