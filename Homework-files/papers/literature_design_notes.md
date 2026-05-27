# Literature Design Notes For Task D

Task D: sonar-based obstacle avoidance for a drone in ROS 2 + Gazebo.

## Cross-Paper Comparison

| Authors and year | Method type | Sensors used | Key contribution | Relevance to my design |
|---|---|---|---|---|
| Zhao et al., 2021 | Classical fusion / task-aware local control | Ultrasonic + camera + IR | Layers ultrasonic avoidance with directional/task cues and pit detection | Best reference for task-aware sonar logic rather than pure collision thresholding |
| Yuan et al., 2021 | Reinforcement learning | Active sonar | Learns obstacle-avoidance behavior from processed, dimensionality-reduced sonar information | Best reference for RL state design and reward-shaping justification |
| Barreto-Cubero et al., 2022 | Learning-based fusion / mapping | Ultrasonic + stereo camera + 2D LiDAR | ANN fusion produces a more reliable distance estimate and occupancy map, especially for glass and low/ambiguous obstacles | Best reference for sensor fusion, beam/sector interpretation, and local mapping |
| Li et al., 2024 | Classical control / dynamic collision-risk planning | AUV state information; relevant to sonar-based dynamic obstacle pipelines | Combines state-tracking collision detection with an improved potential field for dynamic environments | Best reference for dynamic-obstacle risk estimation and proactive collision penalties |
| Mane et al., 2024 preprint | Reactive planning + safety filtering | 2D/2.5D forward-looking sonar | Uses short-term obstacle memory and a control-barrier-function safety layer, with HIL evaluation and code link | Best reference for partial observability, safety filtering, and hardware-aware sonar design |

## Cross-Paper Patterns

Across the five papers, three patterns repeat.

First, acoustic or ultrasonic sensing becomes much more useful once it is converted into state, not left as isolated threshold events. Yuan et al. group active-sonar beams into compact obstacle-distance features before learning, while Zhao et al. combine ultrasonic distance cues with road-sign and infrared context. For my drone, that means the sonar topic should become normalized risk features such as current range, clipped range, recent minimum range, and trend.

Second, because sonar is angularly coarse and partially observable, successful systems add either fusion, short-term memory, or risk tracking. Barreto-Cubero et al. use neural fusion to improve distance estimates and occupancy mapping. Mane et al. explicitly add short-term obstacle memory, while Li et al. use state-tracking collision detection to reason about dynamic risk before collision. For my project, I should include at least a small temporal feature, such as previous sonar reading, recent minimum, or range derivative.

Third, learning-based methods are most convincing when they sit on top of processed acoustic inputs rather than replacing the entire sensing stack blindly. Yuan et al. do not train from raw pixels; they train from processed active-sonar information. Mane et al. also keeps structured reactive modules and safety filtering rather than relying on an unrestricted learned controller. For my PPO drone, the learned policy should receive clean state features and still be constrained by explicit safety termination or action filtering.

## Recommended Design Moves

- Build the controller around processed sonar/risk features, not raw threshold-only events.
- Treat sonar readings as coarse proximity/risk cues rather than perfect laser rays.
- Add short-horizon memory such as previous sonar reading, minimum recent sonar reading, range derivative, or risk trend.
- Use PPO on processed state features: pose, velocity, target vector, sonar range, and recent sonar/risk features.
- Keep simple safety logic around the learned policy: terminate or strongly penalize too-close sonar readings, low altitude, out-of-bounds, invalid sensor data, or stalled progress.
- Evaluate collision rate, success rate, final distance, minimum clearance, episode reward, timeout rate, and representative failure cases.

## Suggested Observation Design

Use a compact observation vector such as:

```text
[
  dx_to_goal, dy_to_goal, dz_to_goal,
  vx, vy, vz,
  normalized_distance_to_goal,
  sonar_range_clipped,
  sonar_risk,
  previous_sonar_risk,
  recent_min_sonar_range,
  sonar_risk_trend
]
```

Where:

- `sonar_range_clipped` is the sonar range clipped to a known safe range.
- `sonar_risk` can be `1 - clipped_range / max_range`, so larger means more dangerous.
- `previous_sonar_risk` gives the policy one-step memory.
- `recent_min_sonar_range` tracks near misses over a short window.
- `sonar_risk_trend` is positive when the obstacle risk is increasing.

## Suggested Reward Design

A practical shaped reward can combine:

```text
reward =
  progress_reward
  + target_bonus
  - distance_to_goal_penalty
  - sonar_risk_penalty
  - near_collision_penalty
  - crash_penalty
  - out_of_bounds_penalty
  - action_smoothness_penalty
  - timeout_penalty
```

Design rationale:

- `progress_reward` and `distance_to_goal_penalty` keep the policy goal-directed.
- `sonar_risk_penalty` teaches early avoidance before collision.
- `near_collision_penalty` makes the policy respect minimum clearance.
- `action_smoothness_penalty` discourages jittery velocity commands.
- `crash_penalty`, `out_of_bounds_penalty`, and `timeout_penalty` make termination conditions explicit.

## Safety Filtering Idea

The learned PPO action can be treated as the nominal command. A simple safety filter can modify it before publishing to `/simple_drone/cmd_vel`.

Example policy:

```text
if sonar_range is invalid:
    stop or slow down and apply penalty
elif sonar_range < emergency_distance:
    override vertical/forward command away from danger
elif sonar_range < caution_distance:
    scale down risky velocity components
else:
    publish PPO action
```

This is consistent with Mane et al.'s safety-filtering idea and with Zhao et al.'s task-aware local obstacle logic. It also makes the project easier to debug because the policy is not the only layer responsible for preventing crashes.

## Report Argument

The report can frame the design as follows:

> My Task D controller treats sonar as a partial and noisy proximity signal. Following sonar-based AUV RL work, I convert the raw range into processed state features before training PPO. Following fusion and risk-tracking papers, I add short-term sonar memory and risk penalties so the agent can react before collision. Following reactive safety-filtering work, I keep explicit termination and safety checks around the learned policy instead of relying on the network alone.

This gives a clean explanation for why the design is not just "PPO with a sonar topic." It is PPO over a processed risk-aware state representation, with shaped rewards and simple safety logic motivated by the literature.
