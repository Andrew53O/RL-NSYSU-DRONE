# Part 4 RL Design Notes

## Overview

Part 4 is an expanded PPO curriculum for the NSYSU drone reinforcement learning task. It is based on the Part 3 environment, but it adds one important missing skill before combined navigation: lateral movement in the y direction. In Part 3, the drone learned vertical movement, x-axis movement, combined x/z movement, and then obstacle avoidance. That worked, but the drone sometimes looked unstable or wavy because it had not been explicitly trained to move and correct itself along the y axis. Part 4 fixes that by inserting a dedicated y-axis stage between x movement and full x/y/z navigation.

The final goal is still Task D: autonomous obstacle avoidance using sonar. The curriculum teaches the drone basic target navigation first, then activates sonar only in the obstacle stages. This keeps the learning problem manageable. The agent does not need to learn takeoff, motor physics, or attitude stabilization from scratch. The Gazebo drone plugin handles low-level stabilization, while PPO learns high-level velocity commands.

The policy publishes velocity commands to:

```text
/simple_drone/cmd_vel
```

The action sent by PPO is:

```text
[vx_cmd, vy_cmd, vz_cmd]
```

This means the policy learns how fast to move forward/backward, left/right, and up/down. The drone's internal controller then changes the body attitude to achieve those commanded velocities.

## PPO Algorithm

The algorithm used in Part 4 is Proximal Policy Optimization, implemented with Stable-Baselines3:

```python
PPO("MlpPolicy", env, ...)
```

PPO was chosen because it is relatively stable for continuous-control simulation tasks. The drone environment is noisy and slow because every RL step interacts with ROS 2 and Gazebo. A highly sensitive algorithm would be harder to tune under the deadline. PPO is also on-policy, which means it updates using recent rollout data from the current policy. That is useful here because the curriculum changes the task from simple movement to harder obstacle avoidance, and the policy is resumed stage by stage.

The default training hyperparameters in `train.py` are:

| Hyperparameter | Value | Meaning |
| --- | ---: | --- |
| Policy | `MlpPolicy` | Fully connected neural network policy |
| Learning rate | `3e-4` | Step size for PPO updates |
| `n_steps` | `512` | Rollout length before each PPO update |
| Batch size | `64` | Minibatch size for optimization |
| Discount factor `gamma` | `0.99` | Future reward weighting |
| Device | `cpu` | Training runs on CPU for compatibility |
| Default `step_dt` | `0.1 s` | Time each action is held in Gazebo |
| Fast `step_dt` often used | `0.05 s` | Faster training and more frequent control |

During training, the script saves multiple best checkpoints:

| Checkpoint | Selection Rule |
| --- | --- |
| `best_episode_model.zip` | Best single episode reward |
| `best_average_model.zip` | Best recent average reward |
| `best_success_model.zip` | Best recent success rate |
| `best_precision_model.zip` | Best success/target-reaching/final-distance score |

For testing, `best_precision_model.zip` is usually the most useful because the homework cares about actually reaching the target, not only getting high reward.

## MDP Formulation

The Part 4 task is modeled as a Markov Decision Process:

```text
MDP = (S, A, R, P, gamma)
```

where:

| Symbol | Meaning in This Project |
| --- | --- |
| `S` | Observation state from pose, velocity, target error, progress, and sonar |
| `A` | Continuous velocity command `[vx, vy, vz]` |
| `R` | Dense reward for progress, precision, smoothness, success, and safety |
| `P` | Gazebo/ROS drone dynamics and sensor updates |
| `gamma` | Discount factor, set to `0.99` |

The true simulator state includes many hidden details such as motor dynamics, attitude, and Gazebo physics. The policy does not observe all of those directly. Instead, it receives a compact observation vector that contains the important navigation information needed for this homework task.

## Observation Space

The observation space has a fixed size of 41 values:

```text
OBSERVATION_DIM = 12 + (4 * 7) + 1 = 41
```

The observation is fixed across all stages. This is important because PPO checkpoints cannot be resumed if the observation shape changes between stages. Even before sonar is used, the sonar slots still exist, but they are masked to safe values.

| Observation Group | Count | Description |
| --- | ---: | --- |
| Normalized pose | 3 | Drone position `x`, `y`, `z` |
| Velocity | 3 | Drone velocity `vx`, `vy`, normalized `vz` |
| Relative target vector | 3 | `dx`, `dy`, `dz` from drone to active target |
| Target distance | 1 | Euclidean distance to active target |
| Target progress | 1 | Sequence progress, or x-progress in long obstacle stages |
| Total target count | 1 | Number of targets normalized by 3 |
| Sonar ranges | 7 | Normalized front/side sonar distances |
| Sonar risks | 7 | Risk values computed from sonar ranges |
| Previous sonar ranges | 7 | One-step sonar memory |
| Sonar trends | 7 | Whether obstacles are getting closer or farther |
| Sonar enabled flag | 1 | `0` before obstacle stages, `1` in obstacle stages |

The first 12 values are:

```text
x / xy_limit
y / xy_limit
z / max_altitude
vx
vy
vz / 0.5
dx / dx_norm
dy / dy_norm
dz / 1.5
distance / distance_norm
target_progress
total_targets / 3.0
```

For Stages 1-4, sonar is masked:

```text
sonar ranges = 1.0
sonar risks = 0.0
previous sonar ranges = 1.0
sonar trends = 0.0
sonar_enabled = 0.0
```

For Stages 5-6, sonar is active:

```text
sonar ranges = real normalized sonar readings
sonar risks = risk computed from distance
previous sonar ranges = previous step readings
sonar trends = previous - current
sonar_enabled = 1.0
```

The sonar sectors are:

```text
front_left
front_center
front_right
front_up
front_down
side_left
side_right
```

This design lets early stages learn normal movement without being distracted by obstacle sensors. Later stages reuse the same policy interface, but the sonar fields become meaningful.

## Action Space

The action space is a continuous 3D velocity command:

```text
action = [vx_cmd, vy_cmd, vz_cmd]
```

The bounds are:

```text
vx_cmd in [-1.0, 1.0]
vy_cmd in [-1.0, 1.0]
vz_cmd in [-0.5, 0.5]
```

The meaning is:

| Action | Meaning |
| --- | --- |
| `vx_cmd` | Move forward/backward along x |
| `vy_cmd` | Move left/right along y |
| `vz_cmd` | Move up/down along z |

The vertical command range is smaller because altitude control is sensitive. Large vertical commands caused overshoot in earlier experiments, especially when the drone was close to the target.

A continuous action space is better than discrete actions for this task because the drone needs smooth control. It should be able to slow down near the target, make small lateral corrections, and move gently around obstacles. Discrete commands like "forward", "left", and "up" would be easier to implement, but they would make the path more jerky.

## Reward Function

The reward is dense. A sparse reward that only gives points at success is too hard for this task because the drone needs many correct small movements before reaching the target. The reward is built from several parts.

In one compact form, the reward can be written as:

```text
R =
  scale * (previous_distance - distance)
+ final_goal_progress
+ dot(axis_weights, previous_abs_error - current_abs_error)
- 0.05 * distance
- precision_penalty(x_error, y_error, z_error)
- 0.01 * norm(filtered_action)
- 0.02 * norm(filtered_action - previous_action)
- near_target_braking_penalty
- sonar_risk_penalty
- safety_filter_penalty
+ success_bonus
+ intermediate_target_bonus
- failure_penalty
- timeout_penalty
```

Where the optional terms only apply in the correct situation:

```text
final_goal_progress =
    8.0 * (previous_final_distance - mission_goal_distance), in Stages 5 and 6

near_target_braking_penalty =
    0.12 * velocity_norm + 0.08 * norm(filtered_action), if distance < 0.6

sonar_risk_penalty =
    2.0 * obstacle_mean_risk^2 + 4.0 * obstacle_max_risk^2, if sonar is enabled

safety_filter_penalty =
    0.25, if the safety filter changes the action

success_bonus =
    80.0, if the final target is reached

intermediate_target_bonus =
    30.0, if a non-final target in a sequence is reached

failure_penalty =
    100.0 for invalid sensor, crash, or unsafe sonar
    80.0 for out of bounds

timeout_penalty =
    5.0 + 20.0 * min(distance, 2.0)
```

This formula is useful for understanding the reward as one signal: the drone is rewarded for getting closer to the target and final mission goal, while being penalized for distance, poor precision, rough motion, obstacle risk, unsafe actions, and failure.

### Distance Progress Reward

The main reward encourages the drone to reduce distance to the active target:

```python
reward += scale * (previous_distance - distance)
```

The scale is:

```text
10.0 when distance >= 0.5
18.0 when distance < 0.5
```

The larger near-target scale encourages precision near the goal. Without this, the drone may move roughly toward the target but stop too far away.

### Long-Mission Progress Reward

For Stages 5 and 6, the final mission target is far away:

```text
(10.0, 0.0, 1.0)
```

The environment uses a local subgoal about 1 meter ahead in x, but it also rewards progress toward the true final mission goal:

```python
reward += 8.0 * (previous_final_distance - mission_goal_distance)
```

This prevents the policy from only chasing a short local target. It still needs to make real progress toward x = 10.

### Axis Progress Reward

The reward also checks whether the absolute error on each axis is shrinking:

```python
delta = previous_abs_error - current_abs_error
reward += dot(weights, delta)
```

The weights depend on the curriculum focus:

| Focus | Weights `[x, y, z]` | Why |
| --- | --- | --- |
| Vertical | `[2, 2, 12]` | Strongly prioritize altitude |
| Horizontal x | `[12, 3, 6]` | Strongly prioritize x movement |
| Lateral y | `[4, 12, 6]` | Strongly prioritize y movement |
| Combined/Obstacle | `[9, 4, 7]` | Balance x, y, and z navigation |

The new Part 4 y-axis stage uses the lateral weights:

```text
[4.0, 12.0, 6.0]
```

This is the main reward change that makes the drone explicitly learn y-direction movement instead of treating y only as a small drift correction.

### Distance and Precision Penalties

The reward penalizes remaining far from the target:

```python
reward -= 0.05 * distance
```

It also penalizes axis errors:

| Focus | Precision Penalty |
| --- | --- |
| Vertical | `0.45*x_error + 0.45*y_error + 0.65*z_error` |
| Horizontal x | `0.45*x_error + 0.20*y_error + 0.45*z_error` |
| Lateral y | `0.30*x_error + 0.55*y_error + 0.45*z_error` |
| Combined/Obstacle | `0.35*x_error + 0.25*y_error + 0.35*z_error` |

For the lateral stage, the y error penalty is highest. This tells PPO that y accuracy matters, not only x and z.

### Near-Target Braking Penalty

When the drone is close to the target, the reward penalizes high velocity and large actions:

```python
if distance < 0.6:
    reward -= 0.12 * velocity_norm
    reward -= 0.08 * norm(filtered_action)
```

This helps reduce overshoot. Without this term, the drone can fly through the target area too fast and miss the success threshold.

### Smoothness Penalties

The reward includes two smoothness terms:

```python
reward -= 0.01 * norm(filtered_action)
reward -= 0.02 * norm(filtered_action - previous_action)
```

The first term discourages unnecessarily large commands. The second term discourages sudden changes between actions. This matters because sudden changes in velocity command make the drone tilt and wobble.

In practice, this penalty is still fairly small. That means the drone may still look wavy if the policy discovers that aggressive command changes reach the target faster. If smoother flight is more important than raw success rate, this action-change penalty can be increased.

### Sonar Risk Penalty

For Stages 5 and 6, sonar is active. The environment converts sonar range into a risk value:

```python
risk = (sonar_caution_distance - range) / sonar_caution_distance
risk = clip(risk, 0.0, 1.0)
```

The caution distance is:

```text
1.5 m
```

The reward penalizes sonar risk:

```python
reward -= 2.0 * obstacle_mean_risk**2
reward -= 4.0 * obstacle_max_risk**2
```

The maximum risk penalty is stronger because one dangerous close obstacle is enough to crash. The mean risk penalty encourages generally staying away from obstacles.

There is also a stuck-near-obstacle penalty:

```python
if obstacle_max_risk > 0.2 and abs(vx) < 0.05:
    reward -= 0.2
```

This discourages the drone from freezing in front of an obstacle. It should either avoid the obstacle or continue making safe progress.

### Safety Filter and Unsafe Termination

The environment has a simple safety filter in obstacle stages. If an obstacle is very close in front, it prevents forward motion and forces slight upward motion:

```python
filtered[0] = min(filtered[0], 0.0)
filtered[2] = max(filtered[2], 0.1)
```

If side sonar is too close, it pushes lateral command away from the obstacle. This is a last-resort protection, not the main policy. The reward penalizes using the safety filter:

```python
reward -= 0.25
```

If sonar distance becomes too small, the episode terminates:

```text
unsafe_sonar if min_obstacle < 0.25 m
```

with a large penalty:

```python
reward -= 100.0
```

This teaches the policy that obstacle contact or near-contact is a serious failure.

### Success and Failure Rewards

When the target is reached:

```python
reward += 80.0
```

If the stage has multiple targets, each intermediate target gives:

```python
reward += 30.0
```

Failures receive large penalties:

| Failure | Penalty |
| --- | ---: |
| Invalid sensor | `-100` |
| Crash / too low altitude | `-100` |
| Out of bounds | `-80` |
| Unsafe sonar | `-100` |
| Timeout | `-5 - 20 * min(distance, 2.0)` |

The timeout penalty depends on final distance. Timing out close to the target is less bad than timing out far from the target.

## Six-Stage Curriculum

Part 4 uses six learning stages:

| Stage | Variant | Skill | Target Setup | Sonar |
| --- | --- | --- | --- | --- |
| 1 | A | Fixed altitude control | `(0, 0, 1.2)` | Masked |
| 1 | B | Random altitude control | `z in [0.7, 1.8]` | Masked |
| 2 | A | Fixed x movement | `(1, 0, 0.8)` | Masked |
| 2 | B | Random x movement | `x in [-1.0, 2.0]` | Masked |
| 3 | A | Fixed y movement | `(0, 1, 0.8)` | Masked |
| 3 | B | Random y movement | `y in [-1.0, 1.0]` | Masked |
| 4 | A | Random x/y/z navigation | random x, y, z | Masked |
| 4 | B | Sequential x/y/z targets | 3 random targets | Masked |
| 5 | A | Single-obstacle sonar avoidance | final goal `(10, 0, 1)` | Active |
| 5 | B | Random-goal single-obstacle sonar avoidance | random goal, generated obstacle on path | Active |
| 6 | A | Multi-obstacle sonar avoidance | final goal `(10, 0, 1)` | Active |

The curriculum is designed as divide and conquer:

1. Stage 1 teaches vertical control.
2. Stage 2 teaches forward/backward x control.
3. Stage 3 teaches left/right y control.
4. Stage 4 combines x, y, and z navigation.
5. Stage 5 activates sonar for one obstacle.
6. Stage 6 increases obstacle complexity.

This is easier than training obstacle avoidance from scratch. If the drone fails in Stage 5 or 6, the earlier stages help identify whether the problem is basic navigation or obstacle reaction.

## Dynamic Local Subgoal in Obstacle Stages

Stages 5 and 6 use a far final goal. Stage 5A and Stage 6A use:

```text
(10.0, 0.0, 1.0)
```

Stage 5B randomizes the final goal:

```text
x in [5.0, 10.0]
y in [-1.0, 1.0]
z = 1.0
```

Training directly on a 10-meter target is difficult because the reward becomes sparse and the drone may not know how to make steady forward progress. To solve this, the environment creates a dynamic local subgoal:

```text
direction = final_goal - current_position
distance = norm(direction)

if distance <= 1.0:
    local_subgoal = final_goal
else:
    local_subgoal = current_position + 1.0 * direction / distance
```

This means the active target is always a short step toward the final goal instead of the full long-distance goal. Unlike the older x-only version, this local subgoal considers x, y, and z together. For example, if the drone is currently at `(3.2, 0.2, 0.8)`, the local subgoal is placed about one meter along the 3D direction from the drone to the final goal.

The final mission goal is still the true success target. The local subgoal only gives PPO an easier immediate navigation target. The reward still includes final-goal progress, so the drone cannot succeed by only hovering near short subgoals.

In Stage 5B, the environment also generates one cylinder obstacle on the straight line between the takeoff position and the randomized final goal. This creates a repeatable "one obstacle blocks the obvious path" task without manually editing the Gazebo world for every random target.

This local subgoal is not a preplanned obstacle-avoidance path. It does not tell the drone to climb or move sideways around cones. It only says, "continue roughly forward." The sonar reward and sonar observations decide when the drone should move sideways or upward.

This is important because the goal is sonar-based local avoidance, not hard-coded trajectory tracking.

## Why the Drone Body Tilts

The drone body tilting is normal and expected. In this project, PPO does not directly control roll, pitch, yaw, or motor thrust. PPO controls velocity:

```text
vx_cmd, vy_cmd, vz_cmd
```

The Gazebo drone plugin receives those velocity commands and internally changes the drone's attitude to achieve them.

Attitude means the drone body's orientation:

| Attitude Term | Meaning | Visual Effect |
| --- | --- | --- |
| Roll | Tilt left/right | Drone leans sideways |
| Pitch | Tilt forward/backward | Drone nose goes down/up |
| Yaw | Rotate around vertical axis | Drone turns left/right |

To move forward, the drone usually pitches forward. To move backward, it pitches backward. To move sideways, it rolls left or right. So if PPO commands forward velocity, the drone body should not stay perfectly flat. A tilted body is how the simulated drone creates horizontal motion.

The "drunk" or wavy look happens when the velocity command changes direction often. For example:

```text
step 1: vx = 0.8, vy = 0.2
step 2: vx = 0.7, vy = -0.2
step 3: vx = 0.9, vy = 0.1
```

The drone tries to follow these changing commands, so the body tilts forward, then slightly left, then right again. It may still be successful, but visually it looks unstable.

In Part 4, the y-axis curriculum should reduce some of this behavior because the policy learns lateral movement as a real skill instead of only correcting y drift late. However, the current reward still prioritizes reaching the target more than perfectly smooth flight. If smoother visual flight is required, the next improvements should be:

| Improvement | Expected Effect |
| --- | --- |
| Increase action-change penalty | Less sudden command switching |
| Add action smoothing before publishing `/cmd_vel` | Smoother body attitude |
| Reduce action bounds | Less aggressive tilt |
| Use same `step_dt` in training and testing | Less overshoot and oscillation |
| Add stronger near-target braking | Less target overshoot |

The important interpretation is: tilting itself is not failure. Constant rapid tilt changes are the problem. Those rapid changes usually mean the policy is making aggressive corrections or the reward does not punish command jitter strongly enough.

## Practical Training Notes

The `step_dt` value is important. It controls how long each PPO action is held in Gazebo:

```text
--step-dt 0.05
```

means every action is applied for about 0.05 seconds. If training uses `0.05` but testing uses `0.1`, then every test action is held twice as long as the policy learned. This can make the drone overshoot and wave more. Training and testing should use the same `step_dt`.

The success distances recommended in the Part 4 README are:

| Stage | Success Distance |
| --- | ---: |
| Stages 1-4 | `0.10 m` |
| Stages 5-6 | `0.20 m` |

The obstacle stages use a looser threshold because the mission is longer and the policy must also avoid obstacles safely.
