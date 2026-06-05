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
| Stable success for Stage 4/5 | Requires the drone to stay inside the success radius with low speed for 3 steps |

Stage 1-3 success logic is kept simple, because those stages are still basic
skill learning. The stable-arrival requirement is applied only from Stage 4
onward.

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
  --model fix_oscillation/models/stage4/variantA/run001/best/best_precision_model.zip \
  --success-distance 0.15 \
  --episodes 5 \
  --step-dt 0.05 \
  --log-position-every 2
```

The main expected improvement is visible in the pose log: near the target,
distance should keep decreasing and the drone should slow down instead of making
large repeated loops around the ball.
