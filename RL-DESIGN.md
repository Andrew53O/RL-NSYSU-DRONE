# RL Design: Part 3 PPO Sonar Curriculum

This document records the current design used in `HW2_Work/part3`. Older Part 2 experiments were replaced by a cleaner curriculum because the reward and evaluation history became too complex to debug under the homework deadline.

## Task

The selected assignment task is **Task D: Autonomous Obstacle Avoidance**. The final evaluated obstacle stage is Stage 4: the drone starts near the origin, flies toward `(10.0, 0.0, 1.0)`, and must avoid a cone obstacle around `x=5.0` using sonar. Stage 5 extends this to multiple cones.

## ROS Interface

The policy publishes:

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

Each reset calls `/reset_world`, publishes `/simple_drone/reset`, lands briefly, then publishes `/simple_drone/takeoff`.

## Observation Space

Part 3 keeps a fixed observation shape across all stages. This allows checkpoints from easier stages to continue training in harder stages.

The observation has 41 values:

```text
x, y, z
vx, vy, vz
dx, dy, dz
distance_to_active_target
target_progress
front/side sonar ranges
sonar risks
previous sonar ranges
sonar trends
sonar_enabled flag
```

Stages 1-3 mask sonar:

```text
sonar ranges = max safe value
sonar risks = 0
sonar trends = 0
sonar_enabled = 0
```

Stages 4-5 activate sonar:

```text
sonar ranges = real ROS Range readings
sonar risk = clipped proximity risk
sonar trend = previous_range - current_range
sonar_enabled = 1
```

This design is inspired by UAV obstacle-avoidance literature that uses compact range/risk features instead of raw camera input.

## Action Space

The action is continuous:

```text
[vx_cmd, vy_cmd, vz_cmd]
```

Bounds:

```text
vx_cmd: [-1.0, 1.0]
vy_cmd: [-1.0, 1.0]
vz_cmd: [-0.5, 0.5]
```

The same action space is used in every stage. PPO therefore learns one control interface: velocity commands in the world-frame-like simulator convention.

## Reward Function

The reward combines:

- target-distance progress,
- axis-specific progress for reducing `abs(dx)`, `abs(dy)`, and `abs(dz)`,
- distance penalty,
- drift and stability penalties,
- near-target braking penalty,
- action magnitude and action smoothness penalties,
- success bonus,
- timeout, crash, out-of-bounds, invalid-sensor penalties,
- sonar mean-risk and max-risk penalties in obstacle stages,
- unsafe-sonar termination in obstacle stages.

Stage 4 and Stage 5 use the far mission goal:

```text
(10.0, 0.0, 1.0)
```

Stage 4 additionally uses an internal dynamic local subgoal about `1 m` ahead in `x`. This helps long-distance progress but is not shown as a Gazebo marker. Success and reporting use the final mission goal, not the local subgoal.

## Safety

Obstacle stages terminate as `unsafe_sonar` if the obstacle sonar range becomes too small. A limited safety filter also prevents obviously dangerous forward or side commands. The filter is not intended to solve the task; it is a guardrail. Evaluation logs include safety-filter override counts so reliance on the filter is visible.

## Current Results

The best available deterministic evaluation results are:

| Stage | Success Rate | Notes |
| --- | ---: | --- |
| 1A | 100% | Fixed altitude |
| 1B | 100% | Random altitude |
| 2A | 100% | Fixed x target |
| 2B | 100% | Random x target |
| 3A | 100% | Random x-z target |
| 3B | 90% | Three sequential targets |
| 4 | 80% | One-obstacle sonar avoidance |

Stage 4 achieved the main Task D behavior but still has a 20% unsafe-sonar failure rate, so it is successful but not fully robust.
