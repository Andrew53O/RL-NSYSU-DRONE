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

### Algorithm

The policy is trained with **Proximal Policy Optimization (PPO)** from Stable-Baselines3 using `MlpPolicy`. PPO was selected because it is robust, widely used for continuous-control simulation tasks, and easier to tune under deadline pressure than more fragile off-policy alternatives. The implementation uses CPU training inside the provided ROS 2/Gazebo Docker workflow.

### MDP Formulation

The state contains normalized drone pose, velocity, relative target information, target progress, sonar features, and a sonar-enabled flag. Sonar slots are always present, but they are masked to safe values before Stage 4. This keeps the observation shape fixed, allowing one curriculum checkpoint to continue into the next stage.

The action is a continuous velocity command:

```text
[vx_cmd, vy_cmd, vz_cmd]
vx, vy in [-1.0, 1.0]
vz in [-0.5, 0.5]
```

The reward combines dense progress shaping, axis-specific shaping, stability penalties, action penalties, success bonuses, and terminal penalties. From Stage 4 onward it also includes sonar risk penalties and unsafe-sonar termination. The discount factor is set through the PPO configuration, typically `gamma = 0.99`.

### Observation Space

The Part 3 observation is a fixed 41-dimensional vector:

```text
pose: x, y, z
velocity: vx, vy, vz
relative target: dx, dy, dz
distance to active target
target progress / index information
sonar ranges for 7 obstacle-facing sectors
sonar risk values
previous sonar ranges
sonar trends
sonar_enabled flag
```

For Stages 1-3, sonar ranges are set to safe maximum values, sonar risks and trends are zero, and `sonar_enabled = 0`. For Stages 4-5, sonar is active and `sonar_enabled = 1`.

### Reward Design

The reward has four main groups:

- **Target progress:** reward for reducing distance to the active target or mission goal.
- **Axis progress:** reward for reducing `abs(dx)`, `abs(dy)`, and `abs(dz)`.
- **Stability/control:** penalties for unnecessary lateral drift, high speed near target, large actions, and action changes.
- **Safety:** penalties for sonar risk, near misses, safety-filter overrides, crashes, out-of-bounds states, invalid sensor states, and timeouts.

Stage 4 uses a far mission goal at `(10, 0, 1)` and a dynamic local target about 1 meter ahead in `x`. The local target improves long-distance progress, but success is measured against the final mission goal. The Gazebo marker shows only the mission target so the visual target remains clear.

## Training Procedure

The final pipeline was trained as a curriculum. The important commands are:

```bash
cd /workspace/HW2_Work/part3

python3 train.py --stage 1 --variant A --timesteps 30000 --step-dt 0.05
python3 train.py --stage 1 --variant B --resume-from models/stage1/variantA/run002/best/best_precision_model.zip --timesteps 50000 --step-dt 0.05
python3 train.py --stage 2 --variant A --resume-from models/stage1/variantB/run001/best/best_precision_model.zip --timesteps 50000 --step-dt 0.05
python3 train.py --stage 2 --variant B --resume-from models/stage2/variantA/run001/best/best_precision_model.zip --timesteps 70000 --step-dt 0.05
python3 train.py --stage 3 --variant A --resume-from models/stage2/variantB/run001/best/best_precision_model.zip --timesteps 100000 --step-dt 0.05
python3 train.py --stage 3 --variant B --resume-from models/stage3/variantA/run002/best/best_precision_model.zip --timesteps 120000 --step-dt 0.05
```

Stage 4 uses the saved world:

```text
nsysu_drone_description/worlds/stage4_obstacle.world
```

The world contains a cone obstacle around `x=5` and the target remains `(10, 0, 1)`.

```bash
python3 train.py \
  --stage 4 \
  --resume-from models/stage3/variantB/run001/best/best_precision_model.zip \
  --success-distance 0.25 \
  --max-steps 1800 \
  --timesteps 120000 \
  --step-dt 0.05 \
  --log-position-every 50 \
  --early-stop-plateau \
  --plateau-window 50 \
  --plateau-patience 80 \
  --plateau-min-delta 1.0
```

The training script saves numbered run folders, `run_config.json`, `monitor.csv`, `training_curve.csv`, `training_curve.png`, and several checkpoints: final model, best episode, best average reward, best success, and best precision.

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

Stage 5 was prepared as a multiple-obstacle extension. The final Stage 5 world places cones around:

```text
Construction_Cone   -> (5.0,  0.0, 0.05)
Construction_Cone_0 -> (6.5, -1.6, 0.05)
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
