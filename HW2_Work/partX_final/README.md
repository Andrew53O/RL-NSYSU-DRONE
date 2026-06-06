# Part X Final

This folder is the clean final fork for retraining the Part 3+Y curriculum from
Stage 1.

The main correction is model selection: continue from `best_average_model.zip`
between stages. Do not continue from `best_precision_model.zip`; that checkpoint
can be the closest single episode but not the best overall policy.

## Files

```text
drone_env.py
train.py
test.py
models/
logs/
```

`drone_env.py` is copied from the anti-oscillation environment and keeps the
same curriculum as Part 3+Y:

| Stage | Variant | Purpose | Target Setup | Sonar |
| --- | --- | --- | --- | --- |
| 1 | A | Fixed altitude control | fixed target `(0, 0, 1.2)` so the policy first learns stable takeoff and vertical stopping | masked |
| 1 | B | Random altitude control | random `z in [0.7, 1.8]` with `x = 0`, `y = 0` to generalize vertical control | masked |
| 2 | A | Fixed forward/backward x movement | fixed target `(1, 0, 0.8)` to add horizontal x translation after altitude control | masked |
| 2 | B | Random x movement | random `x in [-1, 2]` with fixed `y = 0`, `z = 0.8` | masked |
| 3 | A | Fixed sideways y movement | fixed target `(0, 1, 0.8)` to teach lateral motion in the Gazebo y direction | masked |
| 3 | B | Random y movement | random `y in [-1.5, 1.5]` with fixed `x = 0`, `z = 0.8` | masked |
| 4 | A | Random single-target 3D navigation | one random target with `x,y in [-1, 1]`, `z in [0.5, 2.0]` | masked |
| 4 | B | Sequential 3D navigation | three random x/y/z targets using the same range as Stage 4A | masked |
| 5 | A | Fixed one-obstacle sonar mission | final mission goal `(10, 0, 1)` with one obstacle world and active sonar observations | active |
| 5 | B | Random radial one-obstacle mission | final mission goal `(X, Y, 1)` sampled on a radius-10 circle, with the env generating a midpoint cone | active |

Stage 5 uses an internal dynamic local subgoal about `1 m` along the vector
toward the final mission goal. This helps the long obstacle mission produce
useful progress rewards without giving the drone a hand-authored avoidance
path. The visible Gazebo ball still marks the final mission target.

## World Setup

`train.py` and `test.py` do not launch Gazebo worlds. They use whatever world is
already running.

Use `playground.world` for Stage 1-4:

```bash
ros2 launch nsysu_drone_description launch_drone.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/playground.world
```

Use `stage4_obstacle.world` for Stage 5A:

```bash
ros2 launch nsysu_drone_description launch_drone.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage4_obstacle.world
```

Use `playground.world` again for Stage 5B, because the environment generates
its own midpoint cone.

## Training Defaults

The final fork uses shorter deadline-friendly defaults:

```text
timesteps: 50000
max_steps: 800
success_distance: 0.10
step_dt: 0.05
early_stop_plateau: enabled
plateau_window: 30
plateau_patience: 30
plateau_min_delta: 1.0
```

PPO defaults:

```text
policy: MlpPolicy
learning_rate: 0.0003
n_steps: 512
batch_size: 64
gamma: 0.99
device: cpu
checkpoint_freq: 10000
best_window: 20
```

Reward/action penalty defaults:

```text
near_target_action_penalty: 0.3
action_penalty: 0.03
action_smoothness_penalty: 0.09
```

All of these values are saved in both `models/.../run_config.json` and
`logs/.../run_config.json`.

Full command template:

```bash
python3 train.py \
  --stage 4 \
  --variant A \
  --resume-from models/stage3/variantB/run001/best/best_average_model.zip \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

`--early-stop-plateau` is already enabled by default. Use
`--no-early-stop-plateau` only when you want the run to always consume all
timesteps.

For a quick script check, use:

```bash
python3 train.py --stage 1 --variant A --smoke
```

## Best Model Rule

Only this best model is saved:

```text
best/best_average_model.zip
```

The training summary is saved in both places:

```text
models/stageX/variantY/runXXX/best/best_summary.csv
logs/stageX/variantY/runXXX/best_summary.csv
```

This makes the important checkpoint easier to find in the log folder while
keeping the model folder compatible with the old layout.

## Recommended Retraining Order

Run from this folder inside the Docker container:

```bash
cd /workspace/HW2_Work/partX_final
```

Stage 1A train:

```bash
python3 train.py \
  --stage 1 \
  --variant A \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

Stage 1A test:

```bash
python3 test.py \
  --stage 1 \
  --variant A \
  --model models/stage1/variantA/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 1B train:

```bash
python3 train.py \
  --stage 1 \
  --variant B \
  --resume-from models/stage1/variantA/run001/best/best_average_model.zip \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

Stage 1B test:

```bash
python3 test.py \
  --stage 1 \
  --variant B \
  --model models/stage1/variantB/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 2A train:

```bash
python3 train.py \
  --stage 2 \
  --variant A \
  --resume-from models/stage1/variantB/run001/best/best_average_model.zip \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

Stage 2A test:

```bash
python3 test.py \
  --stage 2 \
  --variant A \
  --model models/stage2/variantA/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 2B train:

```bash
python3 train.py \
  --stage 2 \
  --variant B \
  --resume-from models/stage2/variantA/run001/best/best_average_model.zip \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

Stage 2B test:

```bash
python3 test.py \
  --stage 2 \
  --variant B \
  --model models/stage2/variantB/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 3A train:

```bash
python3 train.py \
  --stage 3 \
  --variant A \
  --resume-from models/stage2/variantB/run001/best/best_average_model.zip \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

Stage 3A test:

```bash
python3 test.py \
  --stage 3 \
  --variant A \
  --model models/stage3/variantA/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 3B train:

```bash
python3 train.py \
  --stage 3 \
  --variant B \
  --resume-from models/stage3/variantA/run001/best/best_average_model.zip \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

Stage 3B test:

```bash
python3 test.py \
  --stage 3 \
  --variant B \
  --model models/stage3/variantB/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 4A train:

```bash
python3 train.py \
  --stage 4 \
  --variant A \
  --resume-from models/stage3/variantB/run001/best/best_average_model.zip \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

Stage 4A test:

```bash
python3 test.py \
  --stage 4 \
  --variant A \
  --model models/stage4/variantA/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 4B train:

```bash
python3 train.py \
  --stage 4 \
  --variant B \
  --resume-from models/stage4/variantA/run001/best/best_average_model.zip \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

Stage 4B test:

```bash
python3 test.py \
  --stage 4 \
  --variant B \
  --model models/stage4/variantB/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 5A train:

```bash
python3 train.py \
  --stage 5 \
  --variant A \
  --resume-from models/stage4/variantB/run001/best/best_average_model.zip \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

Stage 5A test:

```bash
python3 test.py \
  --stage 5 \
  --variant A \
  --model models/stage5/variantA/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

Stage 5B train:

```bash
python3 train.py \
  --stage 5 \
  --variant B \
  --resume-from models/stage5/variantA/run001/best/best_average_model.zip \
  --timesteps 50000 \
  --success-distance 0.10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --learning-rate 3e-4 \
  --n-steps 512 \
  --batch-size 64 \
  --gamma 0.99 \
  --checkpoint-freq 10000 \
  --best-window 20 \
  --plateau-window 30 \
  --plateau-patience 30 \
  --plateau-min-delta 1.0 \
  --near-target-action-penalty 0.3 \
  --action-penalty 0.03 \
  --action-smoothness-penalty 0.09
```

Stage 5B test:

```bash
python3 test.py \
  --stage 5 \
  --variant B \
  --model models/stage5/variantB/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

If you rerun a stage, adjust the `runXXX` part to the actual previous run.

## Testing

Example:

```bash
python3 test.py \
  --stage 4 \
  --variant A \
  --model models/stage4/variantA/run001/best/best_average_model.zip \
  --success-distance 0.10 \
  --episodes 10 \
  --max-steps 800 \
  --step-dt 0.05 \
  --log-position-every 25
```

Evaluation saves:

```text
logs/eval/stageX/variantY/runXXX/eval001.csv
logs/eval/stageX/variantY/runXXX/eval001_config.json
logs/eval/stageX/variantY/runXXX/summary.txt
```

Use success rate, average return, final distance, and average steps together.
When success rate and distance are similar, prefer the model with better average
return and fewer steps.
