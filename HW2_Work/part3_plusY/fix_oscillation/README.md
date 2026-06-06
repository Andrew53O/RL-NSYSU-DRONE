# Part 3+Y Anti-Oscillation Experiment

This folder keeps an isolated fix for the Stage 4/5 behavior where the drone
passes through or circles around the target instead of arriving cleanly.

The original `../drone_env.py` and `../train.py` are not changed. Use these
files only when you want to train the anti-oscillation version:

```text
drone_env_oscillation.py
train_oscillation.py
test_oscillation.py
```

## What Changed

The observation space and action space stay the same, so old checkpoints can
still be resumed.

The environment changes only the reward and success logic:

| Change | Why |
| --- | --- |
| Stronger near-target velocity penalty | Slows the drone near the ball instead of flying through it |
| Tangential-speed penalty near target | Penalizes circular motion around the target |
| Moving-away penalty near target | Discourages the repeated orbit pattern |
| Balanced x/y/z precision in combined Stage 4 | Prevents y drift from being treated as less important |
| Direct Stage 4 success at the target ball | Gives PPO a clear success signal once the drone reaches the random x/y/z target |
| Stable success for Stage 5 | Keeps the longer obstacle mission stricter after Stage 4 has learned target reaching |

Stage 1-3 success logic is kept simple, because those stages are still basic
skill learning.

## Recommended Stage 4A Retraining

Run this from the normal Part 3+Y folder:

```bash
cd /workspace/HW2_Work/part3_plusY

python3 fix_oscillation/train_oscillation.py \
  --stage 4 \
  --variant A \
  --resume-from models/stage3/variantB/run003/best/best_precision_model.zip \
  --success-distance 0.15 \
  --timesteps 50000 \
  --step-dt 0.05 \
  --log-position-every 25 \
  --early-stop-plateau
```

The outputs are saved separately:

```text
fix_oscillation/models/
fix_oscillation/logs/
```

## Testing

After training, test with the matching oscillation test script:

```bash
python3 fix_oscillation/test_oscillation.py \
  --stage 4 \
  --variant A \
  --model fix_oscillation/models/stage4/variantA/run010/best/best_average_model.zip \
  --success-distance 0.15 \
  --episodes 5 \
  --step-dt 0.05 \
  --log-position-every 2
```

## Recommended Model

For the Stage 4A anti-oscillation result, use:

```text
fix_oscillation/models/stage4/variantA/run010/best/best_average_model.zip
```

`best_precision_model.zip` only means the closest single training episode. In
the run010 evaluation, `best_average_model.zip` had the same success rate and
same average final distance, but reached the target faster and had better
average return:

| Model | Success rate | Average distance | Average steps | Average return |
| --- | ---: | ---: | ---: | ---: |
| best_average_model.zip | 1.000 | 0.138 | 118.100 | 14.165 |
| best_precision_model.zip | 1.000 | 0.138 | 157.100 | -7.906 |

Because this reward design penalizes extra motion, action changes, and
near-target circling, the higher average return is the better overall model
choice when success rate and final distance are tied.

The main expected improvement is visible in the pose log: near the target,
distance should keep decreasing and the drone should slow down instead of making
large repeated loops around the ball.
