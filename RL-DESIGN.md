# RL Design: Task D Sonar PPO

This document describes the current reinforcement-learning design for Task D:
autonomous obstacle avoidance using sonar. The implementation uses PPO with an
MLP policy. It does not use camera input, LSTM, or RNN state.

## Task Setup

The Gymnasium environment is `DroneSonarAvoidEnv` in
`HW2_Work/part2/drone_env.py`.

The policy controls the drone by publishing a `geometry_msgs/Twist` command to:

```text
/simple_drone/cmd_vel
```

The environment reads:

```text
/simple_drone/gt_pose
/simple_drone/gt_vel
/simple_drone/sonar/out
/simple_drone/front_sonar_left/out
/simple_drone/front_sonar_center/out
/simple_drone/front_sonar_right/out
/simple_drone/front_sonar_up/out
/simple_drone/front_sonar_down/out
/simple_drone/side_sonar_left/out
/simple_drone/side_sonar_right/out
```

Each episode reset publishes `/simple_drone/reset` and `/simple_drone/takeoff`
before PPO starts selecting actions.

## Observation Space

The observation length is fixed at `43`.

Base navigation terms:

```text
x / 8
y / 8
z / 5
vx
vy
vz / 0.5
dx / 3
dy / 3
dz / 1.5
distance_to_target / 3
```

The target-delta and distance terms use smaller normalization constants than
the arena-scale pose terms. This makes the Stage 1 target error easier for PPO
to see while keeping the same observation length.

Obstacle-facing sonar sectors:

```text
front_left
front_center
front_right
front_up
front_down
side_left
side_right
```

For each of the 7 obstacle-facing sonar sectors, the observation includes:

```text
normalized range
risk
previous normalized range
trend
```

Risk is computed as:

```text
clip((caution_distance - sonar_range) / caution_distance, 0.0, 1.0)
```

The final safety summary terms are:

```text
recent minimum obstacle sonar range
downward sonar range
downward sonar risk
left-right risk balance
up-down front risk balance
```

The downward sonar is treated separately as ground/altitude safety, not as a
front obstacle sensor.

## Action Space

The action is continuous:

```text
[vx_cmd, vy_cmd, vz_cmd]
```

Action bounds:

```text
vx_cmd: [-1.0, 1.0]
vy_cmd: [-1.0, 1.0]
vz_cmd: [-0.5, 0.5]
```

The action space is intentionally unchanged across all curriculum stages so
PPO checkpoints can continue training without shape mismatch.

## Reward Function

The reward combines target progress, precision shaping, sonar safety, and
control smoothness.

Target-navigation terms:

```text
distance progress reward
axis-specific progress reward for reducing abs(dx), abs(dy), abs(dz)
direction reward toward the target vector
forward direction reward toward target x
vertical direction reward toward target z
small reward for descending when above the target, or climbing when below it
```

Precision and altitude terms:

```text
distance penalty
forward x-error penalty
altitude error penalty
above-target altitude penalty
near-target precision penalty
near-target axis penalty
near-target above-target penalty
near-target altitude-band penalty
near-target velocity penalty
```

The near-target altitude-band penalty activates when the drone is close to the
target and `abs(z - target_z)` exceeds the altitude tolerance. This was added
because earlier policies learned forward motion but drifted upward near the
target.

Sonar and safety terms:

```text
mean obstacle risk penalty
max obstacle risk penalty
obstacle approach trend penalty
downward sonar risk penalty
safety-filter activation penalty
```

Control regularization:

```text
action magnitude penalty
action smoothness penalty
```

Terminal rewards and penalties:

```text
success bonus
invalid sensor penalty
crash penalty
out-of-bounds penalty
unsafe front sonar penalty
unsafe side sonar penalty
unsafe downward sonar penalty
timeout penalty
```

## Success Criteria

Stage 1 target:

```text
(1.0, 0.0, 0.8)
```

Strict Stage 1 precision uses:

```text
success_distance = 0.1
```

A looser `0.4 m` success distance is useful only as a sanity check. It is not
used as the main precision criterion because PPO can learn to stop near the
edge of the success sphere instead of flying to the target center.

## Checkpoints And Logs

Each training run writes outputs under:

```text
models/stageN/runXXX/
logs/stageN/runXXX/
```

Important model checkpoints:

```text
ppo_drone.zip
best/best_episode_model.zip
best/best_average_model.zip
best/best_success_model.zip
best/best_precision_model.zip
```

For strict Stage 1 evaluation, `best_success_model.zip` is the preferred
checkpoint because it is selected using actual episode `status == "success"`
instead of reward alone.

`best_precision_model.zip` is stricter for Stage 1 debugging: it ranks
episodes by success first, then lower final distance, lower altitude error, and
lower x error. This helps when the reward curve looks good but deterministic
evaluation still times out near the target.

Run016 showed the main Stage 1 failure mode was not sonar interference:
obstacle ranges stayed safe and the safety filter did not activate. The policy
was short in `x` and high in `z`, so the reward now penalizes high-altitude
drift more strongly and evaluation logs average commanded velocity plus final
velocity for diagnosis.

Each run also saves:

```text
run_config.json
monitor.csv
training_curve.csv
training_curve.png
```

`run_config.json` records the target, success distance, PPO hyperparameters,
resume checkpoint, output paths, and observation normalization constants.

Evaluation writes a matching config next to each CSV:

```text
logs/eval/stageN/runXXX/eval001.csv
logs/eval/stageN/runXXX/eval001_config.json
```

The eval config records the model path, target, success distance, episode
count, max steps, and metrics so strict and loose evaluation runs are easy to
compare later.

## Training Notes

Current Stage 1 command:

```bash
cd /workspace/HW2_Work/part2
python3 train.py \
  --stage 1 \
  --timesteps 70000 \
  --learning-rate 0.0003 \
  --checkpoint-freq 10000 \
  --log-position-every 100
```

Strict Stage 1 evaluation:

```bash
python3 test.py \
  --model models/stage1/runXXX/best/best_success_model.zip \
  --target 1.0 0.0 0.8 \
  --episodes 10
```

After observation-scaling changes, old PPO checkpoints should not be resumed
because the observation meaning changed even though the observation length is
still `43`.
