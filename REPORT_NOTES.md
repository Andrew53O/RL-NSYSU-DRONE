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

Literature alignment:

- Yuan et al. supports using processed active-sonar state features and reward shaping instead of raw high-dimensional perception.
- EROAS/Mane supports short-term sonar memory, partial-observability handling, and safety filtering.
- Li et al. supports tracking collision risk and penalizing increasing risk before impact.
- Zhao et al. supports task-aware ultrasonic/sonar logic rather than pure thresholding.
- Barreto-Cubero et al. supports range sensors as local proximity/fusion sectors.
