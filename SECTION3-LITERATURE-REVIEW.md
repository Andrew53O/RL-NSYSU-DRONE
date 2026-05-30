# 3. Literature Review

Recent UAV reinforcement learning research supports the main design decisions in this project: PPO for stable continuous control, goal-relative observations, curriculum learning, subgoal decomposition, and explicit obstacle-risk features. The following five papers are all from the past five years and are directly connected to the observation space, action space, and reward function used in `HW2_Work/part3`.

## Paper 1: Kabas, 2022 - PPO for Autonomous UAV Navigation

Kabas [1] trained a UAV navigation policy with Proximal Policy Optimization (PPO). The method formulates UAV navigation as an RL problem and uses simulation to train a policy that maps environment observations to control actions. The experiments evaluate whether PPO can guide a UAV through autonomous navigation scenarios rather than relying on a manually tuned controller.

How it inspired this project:

- Use Stable-Baselines3 PPO with `MlpPolicy`.
- Use continuous velocity action `[vx_cmd, vy_cmd, vz_cmd]`.
- Use dense navigation reward instead of only terminal success/failure.

## Paper 2: Zhang, Li, and Dong, 2022 - TD3 Multi-Obstacle UAV Navigation

Zhang et al. [2] proposed a DRL approach based on Twin Delayed Deep Deterministic Policy Gradient (TD3) for UAV navigation with multiple obstacles. Their method uses actor-critic continuous control and environmental observations so the UAV can avoid obstacles while progressing toward a target. Their experiments compare performance in random and dynamic multi-obstacle environments.

How it inspired this project:

- Do not give the policy only the target coordinate.
- Add obstacle-facing sonar ranges, risk values, previous ranges, and range trends.
- Treat obstacle avoidance as part of the observation and reward design, not as an afterthought.

## Paper 3: Joshi et al., 2023 - PPO Obstacle Avoidance Under Measurement Uncertainty

Joshi et al. [3] studied sim-to-real UAV waypoint navigation and obstacle avoidance using PPO under measurement uncertainty. Their experiments analyze how noisy measurements affect DRL navigation and obstacle-avoidance performance, including transfer from simulation to a real UAV.

How it inspired this project:

- Process sonar into risk/trend features instead of relying only on raw range.
- Separate deterministic evaluation from training reward.
- Log safety metrics such as minimum sonar range, unsafe-sonar termination, safety-filter overrides, and command statistics.

## Paper 4: Lee, Kim, and Jang, 2023 - Goal-Conditioned Subgoal Planning

Lee et al. [4] proposed real-time UAV path planning using goal-conditioned RL and user-defined subgoals. Their method trains a UAV agent to reach various goals and then uses subgoals to perform more complex maneuvers in unknown environments. Their experiments include tasks such as high-flying, low-flying, penetrating, and bypassing.

How it inspired this project:

- Use Stage 3B sequential targets A, B, and C.
- Include target progress information in the observation.
- Use an internal Stage 4 local subgoal to support long-distance progress toward `(10, 0, 1)` without hard-coding an avoidance trajectory.

## Paper 5: Kim et al., 2025 - Curriculum and Goal-Conditioned UAV Control

Kim et al. [5] combined curriculum learning with goal-conditioned RL to train a more controllable UAV agent from simple straight-forward tasks toward more complex missions. Their experiments show that gradually increasing task difficulty helps the UAV learn reusable control behavior.

How it inspired this project:

- Build the Part 3 curriculum: vertical control, horizontal control, combined navigation, sequential navigation, then sonar obstacle avoidance.
- Keep the same observation/action shape across stages so checkpoints can transfer.
- Avoid training the hardest obstacle task from scratch.

## Connection to This Project

Together, these papers justify the final MDP design:

- **Observation:** pose, velocity, relative target, distance, target progress, sonar ranges, sonar risks, sonar memory, sonar trends.
- **Action:** continuous velocity command `[vx_cmd, vy_cmd, vz_cmd]`.
- **Reward:** target progress, axis-specific progress, stability, action smoothness, sonar-risk penalties, unsafe-sonar termination, and success bonuses.
- **Training:** curriculum learning from easy control tasks toward sonar obstacle avoidance.

## References

[1] B. Kabas, "Autonomous UAV Navigation via Deep Reinforcement Learning Using PPO," in *Proc. Signal Processing and Communications Applications Conference*, 2022.

[2] S. Zhang, Y. Li, and Q. Dong, "Autonomous navigation of UAV in multi-obstacle environments based on a Deep Reinforcement Learning approach," *Applied Soft Computing*, vol. 115, 108194, 2022.

[3] B. Joshi et al., "Sim-to-Real Deep Reinforcement Learning based Obstacle Avoidance for UAVs under Measurement Uncertainty," arXiv:2303.07243, 2023.

[4] G. T. Lee, K. J. Kim, and J. Jang, "Real-time path planning of controllable UAV by subgoals using goal-conditioned reinforcement learning," *Applied Soft Computing*, vol. 146, 110660, 2023.

[5] H. Kim, J. Choi, H. Do, and G. T. Lee, "A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning: From Straight Forward to Round Trip Missions," *Drones*, vol. 9, no. 1, 26, 2025.
