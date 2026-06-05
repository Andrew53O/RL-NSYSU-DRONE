# HW2 Part 3+Y: Lateral Stage PPO Curriculum

Part 3+Y is a copy of Part 3 with one new sideway movement stage inserted between the old Stage 2 and old Stage 3.

The copied models and logs intentionally stop at Stage 2B. This lets the new curriculum continue from the trained Stage 2B checkpoint, then learn `y` movement before going back to random 3D navigation and obstacle stages.

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
| 4A | Random x/y/z navigation | random `x,y in [-1, 1]`, `z in [0.5, 2.0]` | masked |
| 4B | Sequential navigation | 3 random x/y/z targets with the same range as 4A | masked |
| 5A | One-obstacle avoidance | fixed mission goal `(10, 0, 1)` | active |
| 5B | Random radial one-obstacle avoidance | mission goal `(X, Y, 1)`, where `(X, Y)` is sampled on a radius-10 circle | active |

Stage 5 uses an internal dynamic local subgoal about `1 m` along the vector toward the mission goal to help long-distance progress. This local subgoal is not a hand-authored avoidance path. The visible Gazebo ball marks the final mission target.

Stage 5B samples the final target on a circle around the start position:

```text
X = 10 * cos(theta)
Y = 10 * sin(theta)
Z = 1
```

For Stage 5B, the environment also spawns one generated `Construction Cone` at the midpoint between the drone's reset position and the sampled mission goal. This makes the obstacle depend on the target direction instead of using one fixed world obstacle.

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

Stage 5A:

```bash
python3 train.py \
  --stage 5 \
  --variant A \
  --resume-from models/stage4/variantB/run001/best/best_precision_model.zip \
  --max-steps 1800 \
  --timesteps 120000 \
  --step-dt 0.05 \
  --log-position-every 50 \
  --early-stop-plateau \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0
```

Stage 5B:

```bash
python3 train.py \
  --stage 5 \
  --variant B \
  --resume-from models/stage5/variantA/run001/best/best_precision_model.zip \
  --max-steps 1800 \
  --timesteps 160000 \
  --step-dt 0.05 \
  --log-position-every 50 \
  --early-stop-plateau \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0
```

## Testing Commands

Run tests from inside the Part 3+Y folder:

```bash
cd /workspace/HW2_Work/part3_plusY
```

Stage 1A:

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

Stage 1B:

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

Stage 2A:

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

Stage 2B:

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

Stage 3A:

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

Stage 3B:

```bash
python3 test.py \
  --stage 3 \
  --variant B \
  --model models/stage3/variantB/run003/best/best_precision_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 25
```

If you want to test an earlier Stage 3B run, change the model path to `run001` or `run002`.

Stage 4A:

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

Stage 4B:

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

Stage 5A:

```bash
python3 test.py \
  --stage 5 \
  --variant A \
  --model models/stage5/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.25 \
  --episodes 10 \
  --step-dt 0.05 \
  --max-steps 1800 \
  --log-position-every 50
```

Stage 5B:

```bash
python3 test.py \
  --stage 5 \
  --variant B \
  --model models/stage5/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.25 \
  --episodes 10 \
  --step-dt 0.05 \
  --max-steps 1800 \
  --log-position-every 50
```

## Gazebo Worlds

Launch the correct world before training/testing obstacle stages.

Stage 5A uses the fixed one-obstacle world:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage4_obstacle.world
```

Stage 5B generates its own cone in the environment, so use the cleaned playground world:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/playground.world
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
