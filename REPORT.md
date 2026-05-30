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

## Pain Points of Classical Control

A proportional or PID-style controller can move a drone toward a fixed point when the environment is simple. For example, `fly_straight.py` can command velocity proportional to position error. However, this style of controller has three important limitations for the chosen task.

First, fixed gains are sensitive to distance and speed. A high gain can move quickly but overshoot the target; a low gain is stable but slow. Second, a hand-coded controller does not naturally combine several objectives such as target progress, altitude stability, lateral drift reduction, action smoothness, and obstacle clearance. Third, obstacle avoidance requires local decisions that depend on sensor context. A simple rule such as “turn left when sonar is small” may work in one layout but fail when the obstacle is shifted or when the drone also needs to keep moving toward a target.

Reinforcement learning is plausible because the policy can learn a mapping from state features to velocity commands. In this project, PPO receives pose, velocity, target-relative information, and sonar features. The reward function shapes the desired behavior: reduce target distance, avoid obstacles, avoid unsafe sonar range, and keep control smooth. This allows the learned policy to combine navigation and local avoidance in one controller instead of relying on separate hand-written rules.

## Literature Review

Recent UAV reinforcement learning work supports the design choices used here: PPO or actor-critic continuous control, curriculum learning, goal-relative observations, subgoals for long routes, and sensor-based obstacle evaluation.

Kabas used PPO for autonomous UAV navigation in simulation. The important lesson for this project is that PPO can serve as a practical high-level UAV controller when the problem is formulated as an MDP and the reward gives dense navigation feedback. This influenced the choice of Stable-Baselines3 PPO and continuous velocity commands.

Zhang, Li, and Dong studied TD3-based UAV navigation in multi-obstacle environments. Their work emphasizes that obstacle avoidance requires meaningful environmental state, not only a target coordinate. This influenced the Stage 4 and Stage 5 sonar design: obstacle sectors are converted into normalized ranges, risk values, previous values, and trends.

Joshi et al. studied PPO-based UAV obstacle avoidance under measurement uncertainty. This influenced the evaluation method. The project does not trust reward curves alone; `test.py` logs mission goal distance, minimum sonar range, unsafe sonar terminations, safety-filter overrides, and command statistics.

Subgoal-based UAV path planning research shows that long missions can be decomposed into smaller goal-reaching tasks. This inspired Stage 3B sequential targets and the Stage 4 internal local subgoal. The Stage 4 local subgoal is not a fixed avoidance trajectory; it only encourages steady forward progress toward the far goal while sonar determines local avoidance.

Kim et al. used curriculum learning and goal-conditioned reinforcement learning for UAV control. This directly supports the Part 3 structure: vertical control, horizontal control, combined navigation, sequential targets, then obstacle avoidance. The same observation and action shape is preserved across stages so PPO checkpoints can continue training.

Kaufmann et al. demonstrated high-performance drone racing using deep reinforcement learning. Although drone racing is much harder than this homework, it reinforces the importance of simulation-first training, stable timing, and careful evaluation.

Zhou et al. proposed an improved TD3 method for 3D UAV path planning and highlighted the role of reward design. This influenced the use of axis-specific progress terms instead of relying only on Euclidean distance.

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

[2] S. Zhang, Y. Li, and Q. Dong, "Autonomous navigation of UAV in multi-obstacle environments based on a Deep Reinforcement Learning approach," *Applied Soft Computing*, 2022.

[3] B. Joshi et al., "Sim-to-Real Deep Reinforcement Learning based Obstacle Avoidance for UAVs under Measurement Uncertainty," arXiv:2303.07243, 2023.

[4] "Real-time path planning of controllable UAV by subgoals using goal-conditioned reinforcement learning," *Applied Soft Computing*, 2023.

[5] H. Kim, J. Choi, H. Do, and G. T. Lee, "A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning," *Drones*, 2025.

[6] E. Kaufmann et al., "Champion-level drone racing using deep reinforcement learning," *Nature*, 2023.

[7] Y. Zhou et al., "Three-Dimensional Path Planning of UAVs in a Complex Dynamic Environment Based on EE-TD3," *Symmetry*, 2023.
