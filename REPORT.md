# Reinforcement Learning Homework 2 Report

## Task Definition and Motivation

This project addresses **Task D: Autonomous Obstacle Avoidance** in the NSYSU Drone ROS 2 and Gazebo Classic simulator. The final goal is to train a drone policy that can navigate toward a target while reacting to sonar-based obstacle information. Because the deadline is short, the project intentionally uses sonar instead of camera input. A camera-based policy would require image preprocessing, a larger neural network, and more simulation time. Sonar gives a compact local-obstacle signal that is easier to train with PPO.

The final implementation is in `HW2_Work/part3`. It uses a clean curriculum instead of the earlier, more complex Part 2 experiments. The curriculum first teaches basic motion skills, then introduces sequential targets, then activates sonar for obstacle avoidance:

| Stage | Skill | Sonar Used | Main Goal |
| --- | --- | --- | --- |
| 1A | Fixed vertical control | No | Reach `(0, 0, 1.2)` |
| 1B | Random vertical control | No | Generalize altitude control |
| 2A | Fixed horizontal control | No | Reach `(1, 0, 0.8)` |
| 2B | Random horizontal control | No | Generalize forward/back motion |
| 3A | Random x-z navigation | No | Reach random single target |
| 3B | Three sequential targets | No | Visit A, B, C in order |
| 4 | One-obstacle sonar avoidance | Yes | Reach `(10, 0, 1)` past one cone |
| 5 | Multi-obstacle sonar avoidance | Yes | Reach `(10, 0, 1)` through several cones |

The report focuses on the curriculum through Stage 4 because Stage 4 already demonstrates the core Task D requirement: autonomous obstacle avoidance with sonar. Stage 5 was prepared as an extension world with multiple cone obstacles at approximately `x=5`, `x=6.5`, and `x=8`.

## Pain Points of Existing Methods

Conventional controllers are very useful for low-level stabilization, but the chosen homework task is not only a stabilization problem. The drone must fly toward a far target around `(10, 0, 1)`, pass an obstacle near the direct path, react to sonar risk, and still maintain altitude and smooth motion. This creates a mixed navigation-and-avoidance problem where a fixed controller or a small set of hand-written rules becomes difficult to tune.

A simple proportional or PID controller is the easiest baseline. For example, the drone can command forward velocity from the `x` error and vertical velocity from the `z` error. This works in open space, but it does not solve the obstacle decision by itself. If the drone detects an obstacle, the controller needs extra rules such as “move left when front sonar is close” or “climb when front sonar is dangerous.” Those rules quickly become brittle because the correct reaction depends on the drone position, current velocity, target direction, altitude, and which sonar sector is active.

The main pain point is that this task has several objectives that compete with each other. Moving fast in `x` helps reach the target, but it gives less time to react to the obstacle. Moving sideways avoids the cone, but too much lateral motion increases final distance. Climbing can avoid some obstacles, but too much climb causes altitude error. Slowing down improves safety, but can lead to timeout. A hand-designed controller must manually balance all of these trade-offs.

| Method | Strength | Limitation on This Task | Practical Issue Observed or Expected |
| --- | --- | --- | --- |
| PID / proportional control | Simple, fast, easy to understand for fixed target tracking | Needs separate hand-written obstacle logic; gains that work for short targets may overshoot or drift on the long `(10, 0, 1)` mission | Good for flying straight, but not enough to decide when to move around a sonar obstacle |
| LQR | Good for linear systems near an operating point; can produce smooth control | Assumes a local linear model and a quadratic cost; obstacle avoidance is not naturally represented by sonar threshold events | Would need a separate planner or constraint layer before it becomes useful for this obstacle task |
| MPC | Can handle constraints and optimize a short future trajectory | Requires a reliable dynamics model, obstacle model, cost design, and enough computation at each step | More principled than PID, but heavy for this homework deadline and still sensitive to how sonar constraints are modeled |
| Hand-crafted sonar rules | Easy to implement quickly, such as “turn away if front sonar is small” | Rules become layout-specific and can conflict with target tracking, altitude keeping, and stopping behavior | A rule that avoids one cone may fail when the cone is shifted or when multiple cones are added |
| Fixed waypoint trajectory | Makes long navigation easier by decomposing the route | If waypoints go around the obstacle manually, the task becomes trajectory following rather than learned sonar avoidance | It can hide whether the drone learned avoidance or only followed a pre-planned path |
| PPO reinforcement learning | Learns a direct mapping from pose, target error, velocity, and sonar features to velocity commands | Needs reward design, curriculum, and training time; success is not guaranteed | Plausible because the policy can learn the trade-off between forward progress, sonar risk, altitude, and smooth control |

RL is a plausible alternative because it does not require writing a separate rule for every situation. In this project, the PPO policy receives the drone state, relative target information, and sonar-derived risk features. The reward then defines the behavior we want: reduce distance to the mission goal, keep altitude stable, avoid unsafe sonar ranges, avoid excessive lateral drift, and use smooth actions. Through training, the policy can learn when to keep moving forward and when to adjust sideways or vertically because the obstacle is close.

This does not mean RL is automatically better than classical control. The Stage 4 result still has unsafe sonar failures, so the learned controller is not perfect. However, RL is reasonable for this homework because the main challenge is the decision trade-off between navigation and avoidance. Classical methods are still valuable as low-level stabilizers and baselines, while PPO is used here as the high-level policy that combines target progress and sonar-based obstacle reaction.

## Literature Review

Recent UAV reinforcement learning research supports the main design decisions in this project: PPO for stable continuous control, goal-relative observations, curriculum learning, subgoal decomposition, and explicit obstacle-risk features. The following five papers are all from the past five years and are directly connected to the observation space, action space, and reward function used in `HW2_Work/part3`.

**Kabas, 2022 - PPO for autonomous UAV navigation.** Kabas [1] trained a UAV navigation policy with Proximal Policy Optimization (PPO). The method formulates UAV navigation as an RL problem and uses simulation to train a policy that maps environment observations to control actions. The experiments evaluate whether PPO can guide a UAV through autonomous navigation scenarios rather than relying on a manually tuned controller. This paper inspired the use of Stable-Baselines3 PPO with `MlpPolicy` in this project. It also supports the decision to use continuous velocity commands `[vx_cmd, vy_cmd, vz_cmd]` and dense reward shaping instead of sparse success-only rewards.

**Zhang, Li, and Dong, 2022 - TD3 for UAV navigation in multi-obstacle environments.** Zhang et al. [2] proposed a deep reinforcement learning approach based on Twin Delayed Deep Deterministic Policy Gradient (TD3) for UAV navigation with multiple obstacles. Their method uses actor-critic continuous control and environmental observations so the UAV can avoid obstacles while progressing toward a target. The experiments compare navigation performance in random and dynamic multi-obstacle environments. This work influenced the Stage 4 and Stage 5 sonar design. Instead of giving the policy only a target coordinate, the observation includes obstacle-facing sonar ranges, risk values, previous ranges, and range trends, so the policy can react to local obstacle structure.

**Joshi et al., 2023 - PPO obstacle avoidance under measurement uncertainty.** Joshi et al. [3] studied sim-to-real UAV waypoint navigation and obstacle avoidance using PPO under measurement uncertainty. Their experiments analyze how noisy measurements affect DRL navigation and obstacle-avoidance performance, including transfer from simulation to a real UAV. This paper influenced two parts of the project. First, the observation does not use raw sonar alone; it also includes processed risk and trend features to make short-term obstacle changes easier to learn. Second, evaluation is separated from training reward. The `test.py` script records success status, mission-goal distance, minimum sonar range, unsafe-sonar terminations, safety-filter overrides, and command statistics because a high training reward alone does not prove safe navigation.

**Lee, Kim, and Jang, 2023 - goal-conditioned subgoal path planning.** Lee et al. [4] proposed real-time UAV path planning using goal-conditioned reinforcement learning and user-defined subgoals. Their method trains a UAV agent to reach various goals and then uses subgoals to perform more complex maneuvers in unknown environments. Their experiments include tasks such as high-flying, low-flying, penetrating, and bypassing, showing that subgoals can make a single trained policy more flexible. This paper inspired Stage 3B and the Stage 4 long-distance design. Stage 3B uses three sequential targets, while Stage 4 uses an internal local subgoal about one meter ahead in `x` to help the drone make steady progress toward the far mission goal `(10, 0, 1)`. Importantly, the subgoal is not a hand-authored obstacle path; sonar risk still determines avoidance.

**Kim et al., 2025 - curriculum learning and goal-conditioned UAV control.** Kim et al. [5] combined curriculum learning with goal-conditioned reinforcement learning to train a more controllable UAV agent from simple straight-forward tasks toward more complex missions. Their experiments show that gradually increasing task difficulty helps the UAV learn reusable control behavior. This strongly influenced the Part 3 curriculum. Instead of training obstacle avoidance immediately, the project separates learning into Stage 1 vertical control, Stage 2 horizontal control, Stage 3 combined/sequential navigation, and Stage 4 sonar obstacle avoidance. This also explains why the observation/action shape is kept fixed across stages: checkpoints can transfer from easier tasks to harder tasks without changing the policy interface.

Overall, these papers support the final MDP design. The observation combines goal-relative navigation variables with sonar-derived obstacle-risk information. The action space remains continuous velocity control because UAV navigation is naturally continuous. The reward function combines target progress, axis-specific progress, stability, action smoothness, and sonar safety penalties. The curriculum follows the literature by teaching simple motion skills first, then adding long-distance navigation and obstacle avoidance.

## Proposed Solution

### Overview

The proposed solution is a curriculum-based PPO controller for the ROS 2/Gazebo drone. PPO is used as the high-level policy, while the simulator's internal drone plugin still handles low-level physical stabilization. The learned policy publishes velocity commands to `/simple_drone/cmd_vel`. This makes the RL task focused on navigation decisions: move forward/backward, move laterally, climb/descend, slow down near the target, and avoid sonar-detected obstacles.

The implementation uses **Stable-Baselines3 PPO with `MlpPolicy`**. PPO was selected because it is stable for continuous-control simulation tasks and is easier to tune under the homework deadline than more sensitive off-policy algorithms. The same policy interface is kept across all stages so checkpoints can transfer from simple tasks into harder tasks.

### Algorithm and Rationale

The chosen algorithm is **Proximal Policy Optimization (PPO)**. PPO is an on-policy actor-critic algorithm that updates the policy with a clipped objective, preventing each update from changing the policy too aggressively. This is useful for the drone task because Gazebo training is noisy and unstable policies can easily produce crashes, altitude drift, or unsafe sonar behavior.

PPO was chosen for three practical reasons:

1. It supports continuous control, which is needed for smooth drone velocity commands.
2. It is more stable and easier to tune than many off-policy methods under short training time.
3. It is implemented reliably in Stable-Baselines3 and works with a standard vector observation.

The policy network is Stable-Baselines3's default **`MlpPolicy`**, which uses a multilayer perceptron actor-critic architecture. The actor outputs the continuous velocity-command distribution, and the critic estimates the state value for PPO advantage calculation. I did not use a CNN because the policy does not use camera images, and I did not use an RNN/LSTM because the observation already includes short-term sonar memory through previous sonar ranges and sonar trends.

### MDP Formulation

The task is formulated as a Markov Decision Process:

```text
MDP = (S, A, R, gamma)
```

where:

- **State `S`** is the 41-value observation vector containing pose, velocity, target-relative information, target progress, sonar readings, sonar risks, sonar memory, sonar trends, and a sonar-enabled flag.
- **Action `A`** is the continuous velocity command `[vx_cmd, vy_cmd, vz_cmd]`.
- **Reward `R`** combines target progress, axis-specific progress, stability, action smoothness, sonar safety, success bonus, and terminal penalties.
- **Discount factor `gamma`** is `0.99`, so the policy values long-term progress toward the target while still reacting to immediate obstacle risk.

The episode ends when the drone reaches the target, completes the target sequence, crashes, leaves the workspace, gets too close to an obstacle in sonar stages, produces invalid sensor readings, or reaches the maximum step limit.

### Hyperparameter Settings

The main PPO hyperparameters used for the reported Part 3 runs are:

| Hyperparameter | Value | Rationale |
| --- | ---: | --- |
| Policy | `MlpPolicy` | Vector observation, no camera input |
| Learning rate | `0.0003` | Standard PPO baseline; stable enough for Gazebo |
| `n_steps` | `512` | Collects enough rollout data before each PPO update |
| Batch size | `64` | Standard mini-batch size for PPO updates |
| Discount factor `gamma` | `0.99` | Encourages long-term target progress |
| Device | CPU | Sufficient for MLP policy and ROS/Gazebo bottleneck |
| Step duration `step_dt` | `0.05 s` | Faster than `0.1 s` while remaining stable in Gazebo |
| Stage 4 success distance | `0.25 m` | Reasonable tolerance for far target plus obstacle avoidance |
| Stage 4 max steps | `1800` | Allows enough time for the 10 m mission |

The model selection used `best_precision_model.zip` when available, because reward alone can be misleading. A model may achieve high return while still failing strict evaluation or ending too far from the mission target.

### Observation Space Design

The observation space is fixed at **41 values** for all stages. Keeping the size fixed is important because PPO checkpoints cannot be resumed cleanly if the observation shape changes between stages.

The observation is:

| Group | Count | Meaning |
| --- | ---: | --- |
| Normalized pose | 3 | `x`, `y`, `z` |
| Velocity | 3 | `vx`, `vy`, normalized `vz` |
| Relative target | 3 | `dx`, `dy`, `dz` from drone to active target |
| Distance | 1 | Euclidean distance to active target |
| Target progress | 1 | target index progress, or x-progress for long obstacle stages |
| Total target count | 1 | normalized number of targets |
| Sonar ranges | 7 | front-left, front-center, front-right, front-up, front-down, side-left, side-right |
| Sonar risks | 7 | clipped risk values derived from sonar distance |
| Previous sonar ranges | 7 | one-step memory of sonar readings |
| Sonar trends | 7 | whether each sonar reading is getting closer or farther |
| Sonar enabled flag | 1 | `0` before obstacle stages, `1` from Stage 4 onward |

Total:

```text
3 + 3 + 3 + 1 + 1 + 1 + 7 + 7 + 7 + 7 + 1 = 41
```

For Stages 1-3, sonar is intentionally masked:

```text
sonar ranges = safe maximum values
sonar risks = 0
previous sonar ranges = safe maximum values
sonar trends = 0
sonar_enabled = 0
```

This prevents the early navigation policy from depending on obstacle sensors before obstacles are introduced. From Stage 4 onward, the same observation slots contain real sonar data and `sonar_enabled = 1`. This design lets the policy learn basic motion first, then reuse the same neural network interface for obstacle avoidance.

The relative target terms `dx`, `dy`, `dz`, and distance are central to the design. Earlier experiments showed that reward curves could look good while the drone still stopped short of the target or drifted in altitude. Giving the policy explicit normalized target error makes the target direction easier to learn.

### Action Space Design

The action space is a continuous 3D velocity command:

```text
action = [vx_cmd, vy_cmd, vz_cmd]
```

Bounds:

```text
vx_cmd in [-1.0, 1.0]
vy_cmd in [-1.0, 1.0]
vz_cmd in [-0.5, 0.5]
```

`vx_cmd` controls forward/backward movement, `vy_cmd` controls lateral motion, and `vz_cmd` controls altitude. The vertical command range is smaller because altitude control is more sensitive, and large vertical commands caused overshoot in early experiments.

A continuous action space was chosen instead of discrete actions such as "forward," "left," or "hover" because the drone needs smooth motion. Continuous velocity commands allow the policy to slow down near the target, make small lateral corrections, and climb or descend gradually when sonar risk changes.

### Reward Function Design

The reward function is dense because sparse "success only" reward made early learning unstable. The reward combines navigation progress, precision, smoothness, and safety.

The main reward components are:

| Reward Term | Purpose |
| --- | --- |
| Distance progress reward | Reward the drone when distance to the active target decreases |
| Mission-goal progress reward | In obstacle stages, reward progress toward the final goal `(10, 0, 1)` |
| Axis-progress reward | Reward reductions in `abs(dx)`, `abs(dy)`, and `abs(dz)` |
| Distance penalty | Discourage remaining far from the target |
| Precision penalty | Penalize target-axis error depending on the stage focus |
| Near-target braking penalty | Penalize high velocity and large action when close to target |
| Action magnitude penalty | Encourage efficient, smooth commands |
| Action-change penalty | Discourage jerky command changes |
| Sonar risk penalty | Penalize mean and maximum obstacle risk from Stage 4 onward |
| Safety-filter penalty | Penalize relying on the emergency safety filter |
| Success bonus | Reward finishing the current target or full sequence |
| Terminal penalties | Penalize crash, out-of-bounds, invalid sensor state, timeout, or unsafe sonar |

The axis-progress weights change by stage. In the vertical stage, `dz` progress is weighted most strongly. In the horizontal stage, `dx` progress is weighted most strongly. In combined navigation, all axes matter, with extra emphasis on reaching the target while controlling lateral drift and altitude.

Obstacle stages add sonar-specific reward terms. The environment converts sonar distance into a risk value, where a closer obstacle gives larger risk. The reward penalizes both mean obstacle risk and maximum obstacle risk. If the sonar range becomes dangerously small, the episode terminates as `unsafe_sonar` with a large penalty. This encourages the policy to avoid collision rather than simply rushing toward the goal.

Stage 4 and Stage 5 use a far mission goal:

```text
(10.0, 0.0, 1.0)
```

To avoid making the far goal too sparse, the environment uses an internal dynamic local subgoal about `1 m` ahead in `x`. This helps the drone continue forward progress over a long route. However, it is not a hand-coded obstacle path. The local subgoal stays on the nominal route, while sonar risk determines whether the drone should move sideways or adjust altitude around obstacles. Success and reporting are measured using the final mission goal, not the internal local subgoal.

### Six-Stage Learning Curriculum

The six-stage curriculum was designed because training the full obstacle task from scratch was too difficult. Each stage teaches one part of the final behavior.

| Stage | Variant | Goal | Target Setup | Sonar |
| --- | --- | --- | --- | --- |
| 1 | A | Learn basic altitude control | Fixed target `(0, 0, 1.2)` | Masked |
| 1 | B | Generalize altitude control | Random `z` in `[0.7, 1.8]` | Masked |
| 2 | A | Learn horizontal x movement | Fixed target `(1, 0, 0.8)` | Masked |
| 2 | B | Generalize forward/back movement | Random `x` in `[-1, 2]` | Masked |
| 3 | A | Combine x and z navigation | Random x/z target | Masked |
| 3 | B | Visit multiple targets in order | Three random x/z targets A, B, C | Masked |
| 4 | A | Avoid one obstacle with sonar | Final goal `(10, 0, 1)` with one cone near x=5 | Active |
| 5 | A | Avoid multiple obstacles with sonar | Final goal `(10, 0, 1)` with several cones | Active |
| 6 | A | Full mission behavior | Sequential targets with active sonar | Active |

Stage 1 isolates vertical control so the policy learns how `vz_cmd` affects altitude. Stage 2 isolates horizontal `x` control while keeping altitude stable. Stage 3 combines the two skills and adds target sequencing. Stage 4 activates sonar and introduces the first obstacle. Stage 5 increases obstacle complexity. Stage 6 is the planned full mission stage, combining sequential target navigation with active sonar obstacle avoidance.

This curriculum is also useful for debugging. If Stage 4 fails, the earlier stages show whether the problem is basic flight control or obstacle reaction. In this project, Stages 1-3 worked reliably, so Stage 4 failures could be interpreted as sonar-avoidance failures rather than basic navigation failures.

### Gazebo Obstacle Worlds

For the obstacle-avoidance stages, I created separate Gazebo world files so that the evaluation environment is reproducible.

For **Stage 4**, the world file is:

```text
nsysu_drone_description/worlds/stage4_obstacle.world
```

This world places the main task obstacle between the drone start position and the final target. The designed Stage 4 cone is:

```text
Construction Cone -> (5.0, 0.0, 0.05)
```

The final target is `(10.0, 0.0, 1.0)`, so this cone is directly on the nominal straight path. This makes Stage 4 a clear one-obstacle sonar-avoidance task: the drone should fly toward the target, detect the cone with sonar near `x=5`, avoid it, and continue toward the final goal. The base Gazebo map also contains other background cones, but the obstacle intentionally used for the Stage 4 task is the cone at `(5.0, 0.0, 0.05)`.

For **Stage 5**, the world file is:

```text
nsysu_drone_description/worlds/stage5_obstacle.world
```

This world extends Stage 4 by placing three task-relevant cones along the route to the same final target `(10.0, 0.0, 1.0)`:

```text
Construction_Cone   -> (5.0,  0.0, 0.05)
Construction_Cone_0 -> (6.5, -0.5, 0.05)
Construction_Cone_1 -> (8.0,  0.0, 0.05)
```

The first and third cones are close to the direct line from the start to the target, while the second cone is offset to make the path less trivial. This Stage 5 setup is intended to test whether the sonar policy can handle multiple obstacle encounters rather than only memorizing one avoidance maneuver.

## Results and Discussion

The following table summarizes the deterministic evaluation logs available in `HW2_Work/part3/logs/eval`.

| Stage | Episodes | Success Rate | Unsafe Rate | Avg Final Distance | Avg Steps | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1A | 5 | 100% | 0% | 0.015 | 14.0 | Fixed altitude target solved |
| 1B | 10 | 100% | 0% | 0.109 | 20.2 | Random altitude solved |
| 2A | 10 | 100% | 0% | 0.097 | 53.1 | Fixed x target solved |
| 2B | 10 | 100% | 0% | 0.082 | 26.6 | Random x target solved |
| 3A | 10 | 100% | 0% | 0.104 | 40.8 | Random x-z target solved |
| 3B | 10 | 90% | 0% | 0.155 | 279.8 | Sequential target navigation mostly solved |
| 4 | 10 | 80% | 20% | 0.424 overall | 210.0 | Sonar obstacle avoidance mostly successful |

Stage 4 is the most important Task D result. In 8 of 10 episodes, the policy reached near `(10, 0, 1)` after passing the obstacle region. Successful Stage 4 episodes ended with mission-goal distance around `0.22-0.25 m`, final `x` around `9.79-9.84`, and final `z` around `0.92-0.93`. Two episodes ended with `unsafe_sonar` near the obstacle at `x≈5`, with minimum obstacle sonar range around `0.23-0.24 m`. This shows that the policy learned obstacle avoidance but is not perfectly robust.

The Stage 4 result is report-worthy because it demonstrates the intended behavior: long-distance target navigation with sonar risk influencing the path. However, the unsafe episodes are an important limitation. The policy sometimes reacts too late near the obstacle. A future improvement would train longer on randomized obstacle offsets, increase the penalty for high sonar risk, or use a slightly larger unsafe threshold so the policy learns earlier avoidance.

Stage 5 was prepared as a multiple-obstacle extension. The final Stage 5 world places task cones around:

```text
Construction_Cone   -> (5.0,  0.0, 0.05)
Construction_Cone_0 -> (6.5, -0.5, 0.05)
Construction_Cone_1 -> (8.0,  0.0, 0.05)
```

Stage 5 uses the same final target `(10, 0, 1)`. At the time of this report, Stage 5 is an extension experiment rather than the main evaluated result.

## Comparison With Classical Baseline

The classical `fly_straight.py` style controller moves toward a target using a hand-written position-to-velocity rule. It is simple and predictable in open space, but it does not learn from obstacle encounters and does not automatically trade off target progress against sonar risk. The PPO policy is slower to develop because it requires training, but the final Stage 4 policy combines target progress and sonar avoidance in one learned controller. This is the main advantage of the RL approach for Task D.

## Limitations

The obstacle-avoidance behavior is not fully robust. The Stage 4 policy still produced unsafe sonar terminations in 20% of the evaluation episodes. The obstacle layout is also fixed, so the result does not prove full generalization to arbitrary worlds. Finally, the implementation uses Gazebo ground truth pose and velocity; a real drone would require state estimation and sensor calibration.

## Acknowledgement

AI assistance was used for debugging, code review, report organization, and reward-design discussion. All code decisions, experiment execution, evaluation interpretation, and final report content were reviewed and adapted for this specific homework project.

## References

[1] B. Kabas, "Autonomous UAV Navigation via Deep Reinforcement Learning Using PPO," in *Proc. Signal Processing and Communications Applications Conference*, 2022.

[2] S. Zhang, Y. Li, and Q. Dong, "Autonomous navigation of UAV in multi-obstacle environments based on a Deep Reinforcement Learning approach," *Applied Soft Computing*, vol. 115, 108194, 2022.

[3] B. Joshi et al., "Sim-to-Real Deep Reinforcement Learning based Obstacle Avoidance for UAVs under Measurement Uncertainty," arXiv:2303.07243, 2023.

[4] G. T. Lee, K. J. Kim, and J. Jang, "Real-time path planning of controllable UAV by subgoals using goal-conditioned reinforcement learning," *Applied Soft Computing*, vol. 146, 110660, 2023.

[5] H. Kim, J. Choi, H. Do, and G. T. Lee, "A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning: From Straight Forward to Round Trip Missions," *Drones*, vol. 9, no. 1, 26, 2025.
