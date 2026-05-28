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
  input: normalized pose, velocity, target vector, front sonar ranges/risks/trends, previous front sonar, recent minimum front sonar, downward sonar risk
  output: continuous velocity command
  reward: target progress - distance/action/smoothness/sonar-risk/trend penalties + success/crash terms
  safety: terminate unsafe states and apply a small emergency filter before publishing dangerous commands
```

This is not just "PPO flies to a target." It is PPO over a processed sonar-risk state, with short-term sonar memory and explicit safety conditions.

## Final Observation, Action, and Reward Design

### Design Goal

The final RL design is not a camera-based obstacle detector and not a pure waypoint follower. It is a sonar-risk-aware PPO controller. The policy receives a compact state made from position, velocity, target direction, sonar sector distances, sonar risk, recent sonar history, and obstacle-approach trend. The policy outputs continuous velocity commands, while the environment gives shaped rewards for target progress and early obstacle avoidance.

This design follows the strongest repeated idea across the collected papers: sonar should be processed into useful state features before learning. Yuan et al. use processed active-sonar information rather than raw high-dimensional perception. Mane et al. add short-term obstacle memory and a safety layer because forward-looking sonar is partially observable. Li et al. track collision risk before impact rather than waiting for collision. Zhao et al. and Barreto-Cubero et al. support treating ultrasonic/sonar data as imperfect local proximity cues that should be interpreted with task context.

### Observation Space

The observation vector has 35 values:

```text
[
  x/8, y/8, z/5,
  vx, vy, vz/0.5,
  dx_to_target/8, dy_to_target/8, dz_to_target/5,
  distance_to_target/12,

  front_left_range/10,
  front_center_range/10,
  front_right_range/10,
  front_up_range/10,
  front_down_range/10,

  front_left_risk,
  front_center_risk,
  front_right_risk,
  front_up_risk,
  front_down_risk,

  previous_front_left_range/10,
  previous_front_center_range/10,
  previous_front_right_range/10,
  previous_front_up_range/10,
  previous_front_down_range/10,

  front_left_trend,
  front_center_trend,
  front_right_trend,
  front_up_trend,
  front_down_trend,

  min_recent_front_range/10,
  down_sonar_range/10,
  down_sonar_risk,
  left_right_risk_balance,
  up_down_risk_balance
]
```

The pose and target-vector features tell the policy where the drone is and where it should go. I normalize `x` and `y` by the safe flight boundary of 8 m, `z` by the 5 m altitude limit, and target deltas by the same scale. This keeps the observation values in a stable range for PPO and prevents position values from dominating smaller sonar-risk values. The target-vector features are important because obstacle avoidance should remain task-aware: the drone should avoid obstacles while still trying to reach the target, which follows Zhao et al.'s idea that local obstacle behavior should be interpreted with task context.

The five front sonar range features represent the local obstacle layout:

```text
left, center, right, up, down
```

These are normalized by the 10 m sonar maximum range. The left/center/right sectors provide horizontal avoidance information, while the up/down sectors provide vertical avoidance information. This matters for a drone because the action includes vertical velocity. If only a center sonar existed, the agent could know that something is close but not whether climbing is a reasonable escape action. With front-up and front-down sectors, the state can support behavior such as climbing over a lower obstacle or avoiding upward movement when the upper front sector is blocked.

The five front risk values convert distance into danger:

```text
risk = clip((caution_distance - range) / caution_distance, 0, 1)
```

In the current design, `caution_distance = 1.5 m`. A risk value of 0 means the sector is clear enough, while 1 means the sector is at or beyond the close-danger region. This is based on Yuan et al.'s processed sonar-state approach: PPO should not have to infer every useful safety feature from raw range alone. Risk features make the learning problem easier because the network directly receives "how dangerous is this direction?" instead of only a distance number.

The previous front sonar ranges and trend features add short-term memory:

```text
trend = previous_normalized_range - current_normalized_range
```

A positive trend means the obstacle is getting closer. This is important because a single sonar reading is partially observable. The same current distance can be safe or dangerous depending on whether the drone is moving toward the obstacle or away from it. Mane et al. motivate short-term memory for sonar because limited field of view and occlusion make one-frame sensing unreliable. Li et al. motivate this from a collision-risk perspective: the controller should react to increasing risk before collision occurs.

The recent minimum front range stores the closest front-sector reading over a short window. This helps represent near misses and short-term clearance history. If the drone briefly passes close to an obstacle, the policy and report metrics can still account for that risk. This follows the safety-oriented view in Mane et al. and Li et al., where minimum clearance and risk history matter, not only the final state.

The downward sonar features are kept separate from the front sonar features. The original `/simple_drone/sonar/out` is treated as ground/altitude safety, not as a forward obstacle detector. This is important for honesty in the report: the drone uses added front sonar sectors for obstacle avoidance and keeps the original downward sonar for low-altitude safety. The downward risk feature helps prevent the policy from learning aggressive downward movement near the ground.

The left-right and up-down risk-balance features are compact directional cues:

```text
left_right_risk_balance = front_left_risk - front_right_risk
up_down_risk_balance = front_up_risk - front_down_risk
```

These help PPO learn avoidance direction. For example, if left risk is high and right risk is low, moving right is likely safer. If front-down risk is high and front-up risk is low, climbing may be safer. This is similar in spirit to Barreto-Cubero et al.'s sensor-fusion/local-mapping idea: individual range readings become an interpreted local spatial structure.

### Action Space

The action is a continuous 3D velocity command:

```text
[vx_cmd, vy_cmd, vz_cmd]
```

The limits are:

```text
vx_cmd: [-1.0, 1.0]
vy_cmd: [-1.0, 1.0]
vz_cmd: [-0.5, 0.5]
```

This action space is intentionally simple because the simulator already has a lower-level drone controller that converts velocity commands into motion. PPO does not need to output motor forces or attitude commands. It only needs to choose the desired motion direction. This makes the MDP easier to train within the homework deadline.

The action design is also matched to the sonar design. The horizontal front sectors support `vy_cmd` decisions because the policy can choose left or right depending on which side is safer. The vertical front sectors support `vz_cmd` decisions because the policy can choose to climb or descend depending on whether the obstacle risk is higher above or below. The center front sector supports `vx_cmd` decisions because the policy should slow down or stop moving forward when an obstacle is directly ahead.

### Reward Function

The reward is shaped as:

```text
reward =
  5.0 * progress_reward
  - 0.02 * distance_to_target
  - 2.0 * mean_front_risk^2
  - 4.0 * max_front_risk^2
  - 1.5 * max_front_approach_trend
  - 1.0 * down_sonar_risk^2
  - 0.01 * norm(action)
  - 0.02 * norm(action - previous_action)
  - 0.25 * safety_filter_used
  + success_or_failure_terms
```

The progress reward is:

```text
progress_reward = previous_distance_to_target - current_distance_to_target
```

This term rewards the drone for moving closer to the target at each step. Without this term, PPO may only receive sparse success or crash information, which is difficult to learn from. The distance penalty keeps the drone goal-directed even when progress is small. Together, these terms define the navigation part of the task.

The mean front-risk penalty discourages flying through generally cluttered or risky areas. The maximum front-risk penalty is stronger because one very close obstacle is dangerous even if other sectors are clear. This makes the drone care about both overall local risk and the single most dangerous direction. Yuan et al. support this kind of shaped obstacle-avoidance reward because their RL formulation combines path planning with obstacle safety.

The approach-trend penalty is the most direct connection to Li et al. A collision is not only about current distance; it is also about whether the relative state is becoming more dangerous. If the sonar range is decreasing, the drone is approaching an obstacle, so the reward penalizes that trend before the unsafe threshold is crossed. This encourages proactive avoidance rather than late emergency reactions.

The downward-sonar risk penalty discourages unsafe altitude behavior near the ground. This is separate from front obstacle avoidance because the original sonar is downward-facing. It helps keep the drone from solving obstacle avoidance by diving toward the floor.

The action magnitude penalty discourages unnecessarily large velocity commands. The action-smoothness penalty discourages jittery behavior by penalizing large changes from the previous command. Smoothness is important for a drone because rapidly changing commands can cause unstable or visually poor motion in Gazebo. It also makes the learned controller easier to compare with classical control in the final reflection.

The safety-filter penalty is small. It does not punish the agent as strongly as a crash, but it tells PPO that relying on the emergency filter is not ideal. The filter should be a last layer of protection, not the main controller. This matches Mane et al.'s safety-filtering idea: the learned or reactive controller proposes an action, and a safety layer prevents clearly dangerous commands.

### Termination and Safety

The episode ends with success when:

```text
distance_to_target < 0.4 m
```

The episode terminates as unsafe when:

```text
min_front_sonar_range < 0.25 m
down_sonar_range < 0.25 m
z < 0.25 m
abs(x) > 8 m
abs(y) > 8 m
z > 5 m
sensor state is invalid
```

Timeout occurs after the maximum step count. Success gives a positive terminal bonus, while unsafe termination gives a large negative penalty. These conditions make the safety constraints explicit rather than leaving the policy to discover every boundary from reward shaping alone.

The emergency safety filter modifies the PPO action before publishing it only in clearly dangerous cases. If front sonar is dangerously close, it limits forward motion and pushes backward/upward. If the downward sonar is too close to the ground, it forces upward motion. This is important for a real robot-style argument: learned policies can make mistakes, so a small safety layer is reasonable for obstacle avoidance. Mane et al. is the strongest support for this design choice because their sonar obstacle-avoidance system combines reactive planning with a safety layer.

### Paper-To-Design Mapping

| Paper | Design choice supported |
|---|---|
| Yuan et al., 2021 | Use processed sonar range/risk features as the RL state instead of raw camera or raw unstructured sensing. |
| Mane et al., 2024 | Add previous sonar, recent minimum sonar, and an emergency safety filter for partial observability and safety. |
| Li et al., 2024 | Penalize increasing obstacle risk using sonar trend, not only final collision. |
| Zhao et al., 2021 | Combine obstacle cues with target direction so avoidance remains task-aware. |
| Barreto-Cubero et al., 2022 | Treat multiple range sectors as an interpreted local proximity structure, similar to lightweight sensor fusion/local mapping. |

In summary, the controller is designed as PPO over a processed sonar-risk state. The observation gives the policy enough information to decide where the target is, where obstacles are, whether risk is increasing, and which avoidance direction is safer. The action space stays simple by using velocity commands. The reward encourages goal progress while penalizing unsafe proximity, increasing collision risk, ground risk, and unstable commands. This makes the design appropriate for the homework because it is sonar-based, explainable, trainable within the deadline, and directly connected to the collected literature.
