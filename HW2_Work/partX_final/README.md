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

| Stage | Variant | Task |
| --- | --- | --- |
| 1 | A | fixed vertical target |
| 1 | B | random vertical target |
| 2 | A | fixed x target |
| 2 | B | random x target |
| 3 | A | fixed y target |
| 3 | B | random y target |
| 4 | A | random x/y/z target |
| 4 | B | three random x/y/z targets |
| 5 | A | fixed one-obstacle sonar mission |
| 5 | B | random radial mission with generated midpoint cone |

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
step_dt: 0.05
early_stop_plateau: enabled
plateau_window: 30
plateau_patience: 30
plateau_min_delta: 1.0
```

The action penalties are also saved in every `run_config.json`:

```text
near_target_action_penalty: 0.3
action_penalty: 0.03
action_smoothness_penalty: 0.09
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

Stage 1A:

```bash
python3 train.py --stage 1 --variant A
```

Stage 1B:

```bash
python3 train.py \
  --stage 1 \
  --variant B \
  --resume-from models/stage1/variantA/run001/best/best_average_model.zip
```

Stage 2A:

```bash
python3 train.py \
  --stage 2 \
  --variant A \
  --resume-from models/stage1/variantB/run001/best/best_average_model.zip
```

Stage 2B:

```bash
python3 train.py \
  --stage 2 \
  --variant B \
  --resume-from models/stage2/variantA/run001/best/best_average_model.zip
```

Stage 3A:

```bash
python3 train.py \
  --stage 3 \
  --variant A \
  --resume-from models/stage2/variantB/run001/best/best_average_model.zip
```

Stage 3B:

```bash
python3 train.py \
  --stage 3 \
  --variant B \
  --resume-from models/stage3/variantA/run001/best/best_average_model.zip
```

Stage 4A:

```bash
python3 train.py \
  --stage 4 \
  --variant A \
  --resume-from models/stage3/variantB/run001/best/best_average_model.zip
```

Stage 4B:

```bash
python3 train.py \
  --stage 4 \
  --variant B \
  --resume-from models/stage4/variantA/run001/best/best_average_model.zip
```

Stage 5A:

```bash
python3 train.py \
  --stage 5 \
  --variant A \
  --resume-from models/stage4/variantB/run001/best/best_average_model.zip
```

Stage 5B:

```bash
python3 train.py \
  --stage 5 \
  --variant B \
  --resume-from models/stage5/variantA/run001/best/best_average_model.zip
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
