# Report Notes: Side Sonar Curriculum Rationale

Task D uses sonar instead of camera to keep the implementation deadline-friendly and focused on local obstacle avoidance. The earlier six-sonar layout mostly observed the forward and downward directions: one original downward sonar for ground safety plus five front sectors for obstacle cues. A right-wall crash is consistent with a side-observability blind spot, because the policy may command lateral motion without a direct side range measurement.

The updated design adds left and right side sonar topics:

```text
/simple_drone/side_sonar_left/out
/simple_drone/side_sonar_right/out
```

This keeps the controller sonar-based while improving local proximity observability. The policy receives processed sonar features rather than raw sensor messages: normalized range, risk, previous range, and trend for each obstacle-facing sector. The downward sonar remains separate as a ground/altitude safety cue.

The emergency safety filter is intentionally limited. It prevents obvious front, side, and ground collisions, but it does not solve the navigation task by itself. PPO still chooses the main continuous velocity action `[vx_cmd, vy_cmd, vz_cmd]`, and the reward penalizes reliance on the filter through a small activation penalty.

The four-stage curriculum separates skills:

1. Fixed easy target for takeoff, motion, and target progress.
2. Random open-space targets for general navigation.
3. Semi-random targets near direct-path obstacles for sonar-risk learning.
4. Harder Task D placements for final evaluation and demo.

The observation space, action space, environment class, and MLP PPO policy stay fixed across all stages. This matters because PPO checkpoints from earlier stages can continue training in later stages without shape mismatch.

## Checkpoint Selection

During training, PPO does not always improve monotonically. A policy can reach a good behavior for several episodes and then become worse after later gradient updates. For this reason, each training run saves both the final model and several best-model checkpoints.

`best_episode_model.zip` is the checkpoint with the highest single episode return. It is useful for finding whether the policy ever discovered a good behavior, but it can be noisy because one lucky episode may not repeat reliably.

`best_average_model.zip` is the checkpoint with the highest recent moving-average episode return. This is usually better for evaluation and curriculum transfer because it favors stable behavior over a one-time lucky rollout.

`best_success_model.zip` is the checkpoint with the highest recent strict success rate, where success is the environment terminal status rather than only high reward. This is the most relevant Stage-1 checkpoint because the strict target criterion is `distance < 0.4 m`; a near-target timeout can still have a good reward curve but should not be treated as a solved stage.

For Stage 1, the strict success radius is `0.4 m`. The evaluation logs showed that early policies often reached the target region but timed out around `0.6 m`, first because they hovered too high and later because they stopped short in the forward x direction. The reward was therefore adjusted in small steps to improve altitude precision and forward target precision while keeping the same observation and action spaces.

Literature alignment:

- Yuan et al. supports using processed active-sonar state features and reward shaping instead of raw high-dimensional perception.
- EROAS/Mane supports short-term sonar memory, partial-observability handling, and safety filtering.
- Li et al. supports tracking collision risk and penalizing increasing risk before impact.
- Zhao et al. supports task-aware ultrasonic/sonar logic rather than pure thresholding.
- Barreto-Cubero et al. supports range sensors as local proximity/fusion sectors.
