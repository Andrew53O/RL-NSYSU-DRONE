# HW2 Task D Report Notes

## Why Obstacles Matter

For Task D, the agent must experience obstacles during training. If the training world only rewards flying to a target, PPO may learn target navigation but not obstacle avoidance.

However, do not begin with many obstacles. A cluttered world makes early learning noisy and hard to debug. Use a curriculum: start simple, verify the sensors and RL loop, then gradually increase obstacle difficulty.

## Important Sonar Limitation

The original simulator publishes downward sonar on:

```text
/simple_drone/sonar/out
```

This is a `sensor_msgs/Range` topic. In the stock drone model, this sonar behaves like a downward/coarse proximity sensor, not a full forward-facing obstacle detector. For the Task D design, we keep that downward sonar for ground/altitude safety and add five front sonar sectors:

```text
/simple_drone/front_sonar_left/out
/simple_drone/front_sonar_center/out
/simple_drone/front_sonar_right/out
/simple_drone/front_sonar_up/out
/simple_drone/front_sonar_down/out
```

The left/center/right sectors give horizontal obstacle awareness. The up/down sectors give vertical obstacle awareness, which matters because the action includes `vz_cmd`; the policy can learn to climb when front-low or center sectors show risk.

Before claiming strong obstacle avoidance, verify whether the sonar range changes near cones or walls.

With `launch_drone` running, open another container terminal:

```bash
ros2 topic echo /simple_drone/sonar/out
```

Then move the drone near obstacles using teleop or velocity commands. Watch whether the `range` value changes.

If the front sonar ranges change near obstacles, the setup can support real sonar-based obstacle avoidance.

If only the downward sonar changes with altitude or ground distance, then adding more cones in front will not help much. In that case, document the limitation and either:

- treat downward sonar as a safety/proximity cue, or
- use the added front sonar sectors for obstacle avoidance.

## Curriculum For Training

### Stage 0: Pipeline Smoke Test

Goal: prove ROS, Gazebo, Gymnasium, PPO, logging, and model saving work.

Run:

```bash
cd /workspace/HW2_Work/part2
python3 train.py --smoke
python3 test.py
```

Success criteria:

- `ppo_drone.zip` is created.
- `training_curve.png` is created.
- `test.py` prints a status, final distance, minimum sonar range, and total reward.

This stage does not prove obstacle avoidance yet.

### Stage 1: Simple Target Navigation

Goal: confirm the drone can learn to move toward a target without immediately crashing or timing out.

Use:

```bash
python3 train.py --timesteps 50000
python3 test.py
```

Success criteria:

- Episode length is not constantly 1-5 steps.
- Reward is not dominated by immediate `-50` crash penalties.
- Test status sometimes reaches `success` or gets closer to the target before timeout.

### Stage 2: Sonar Behavior Check

Goal: prove whether the front sonar topics can detect obstacle proximity.

Run:

```bash
ros2 topic echo /simple_drone/front_sonar_center/out
ros2 topic echo /simple_drone/front_sonar_up/out
ros2 topic echo /simple_drone/front_sonar_down/out
```

Then move the drone:

- near the ground,
- near a wall,
- near cones,
- away from obstacles.

Record observations:

| Situation | Expected useful sonar behavior |
|---|---|
| Near obstacle in front | front sonar `range` decreases |
| Away from obstacle | front sonar `range` increases |
| Near ground only | downward sonar may represent altitude/ground clearance |

If the front sonar does not react to forward obstacles, use this as a report limitation.

### Stage 3: One-Obstacle Task

Goal: train on one simple obstacle before adding clutter.

Recommended scenario:

- Drone starts near `(0, 0, 1)`.
- Target is near `(5, 0, 1.5)`.
- One obstacle is between start and target.

Do not manually drag obstacles every episode. Manual changes are fine for early inspection, but training needs repeatable conditions.

Better options:

- use the existing `playground.world` obstacles if they already sit between the drone and target,
- or edit the world/URDF setup once and keep it fixed,
- or later add code/scripts to randomize obstacle positions.

Success criteria:

- Fewer unsafe sonar terminations over time.
- Higher minimum sonar range during test.
- Drone still makes progress toward the target.

### Stage 4: Multiple Obstacles

Goal: increase difficulty after one-obstacle behavior works.

Add:

- two or three obstacles,
- different target locations,
- longer episodes,
- stricter minimum-clearance evaluation.

Success criteria:

- test success rate improves,
- collision/unsafe-sonar rate decreases,
- final distance decreases,
- minimum clearance stays above the safety threshold.

### Stage 5: Report-Ready Evaluation

For the final report, evaluate:

- success rate,
- timeout rate,
- crash or unsafe-sonar rate,
- final distance to target,
- minimum sonar range,
- total episode reward,
- representative failure cases.

## Do I Need To Manually Change Obstacles In Gazebo?

For quick visual experiments: yes, you can manually inspect/move around Gazebo and watch sonar readings.

For RL training: no, manual changes are not ideal. Training should be repeatable. Prefer a fixed world first, then code-based randomization if time allows.

Recommended path for this homework:

1. Use the existing `playground.world` first.
2. Check if sonar reacts to existing obstacles.
3. If useful, choose a target that forces the drone near an existing obstacle.
4. Train and test on that fixed setup.
5. If time remains, add one controlled obstacle setup or simple randomization.

## Literature-Based Design Rationale

The current design is supported by the collected papers:

- Yuan et al.: process sonar into compact state features for RL.
- Mane et al.: add short-term memory and safety filtering for partially observable sonar.
- Li et al.: track risk before collision, not only after collision.
- Zhao et al. and Barreto-Cubero et al.: treat ultrasonic/sonar data as imperfect proximity cues that should be interpreted with context.

Therefore, the planned algorithm is:

```text
PPO policy
  input: pose, velocity, target vector, downward sonar, front-left/center/right/up/down sonar, previous front sonar, recent minimum front sonar
  output: continuous velocity command
  reward: target progress - distance/action/sonar-risk penalties + success/crash terms
  safety: terminate or strongly penalize unsafe sonar, low altitude, out of bounds, invalid sensor state
```

This is not just "PPO flies to a target." It is PPO over a processed sonar-risk state, with short-term sonar memory and explicit safety conditions.
