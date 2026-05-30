# 3. Literature Review

Recent UAV reinforcement learning research supports the main design choices used in this project: PPO-based continuous control, goal-relative observations, curriculum learning, subgoal navigation, and range-sensor-based obstacle avoidance.

## PPO for UAV Navigation

Kabas [1] applied PPO to autonomous UAV navigation in simulation. This supports the use of Stable-Baselines3 PPO as a practical high-level controller. The paper influenced the decision to use a continuous action space `[vx_cmd, vy_cmd, vz_cmd]` and dense shaped rewards instead of sparse success-only feedback.

## Multi-Obstacle UAV Navigation With TD3

Zhang, Li, and Dong [2] proposed TD3-based UAV navigation in random and dynamic multi-obstacle environments. Their work supports the idea that obstacle avoidance requires explicit obstacle state, not only target position. This influenced the sonar observation design: ranges are converted into risk, previous range, and trend features.

## PPO Obstacle Avoidance Under Measurement Uncertainty

Joshi et al. [3] studied PPO-based UAV waypoint navigation and obstacle avoidance under measurement uncertainty. This influenced the evaluation design. The project logs mission-goal distance, minimum sonar range, unsafe-sonar status, safety-filter overrides, and command velocities, rather than judging success from reward curves alone.

## Goal-Conditioned Subgoal Navigation

Goal-conditioned subgoal planning research [4] shows that longer UAV routes can be decomposed into smaller target-reaching tasks. This inspired Stage 3B sequential targets and the Stage 4 internal local subgoal. The local subgoal supports long-distance progress, but obstacle avoidance remains sonar-driven rather than a fixed hand-authored trajectory.

## Curriculum Learning for UAV Control

Kim et al. [5] used curriculum learning and goal-conditioned RL for UAV control. This directly supports the Part 3 structure: learn vertical control first, then horizontal control, then combined navigation, then sonar obstacle avoidance. The same observation and action interface is kept across stages so checkpoints can transfer.

## High-Performance Drone RL

Kaufmann et al. [6] demonstrated champion-level drone racing using deep RL. Although the task differs from this homework, it supports simulation-first drone policy training and highlights the importance of consistent simulation timing and careful evaluation.

## 3D UAV Path Planning With TD3 Variants

Zhou et al. [7] proposed an EE-TD3 method for 3D UAV path planning. Their work highlights how reward design affects convergence. This inspired the use of axis-specific progress terms for `dx`, `dy`, and `dz` instead of only Euclidean distance.

## References

[1] B. Kabas, "Autonomous UAV Navigation via Deep Reinforcement Learning Using PPO," 2022.

[2] S. Zhang, Y. Li, and Q. Dong, "Autonomous navigation of UAV in multi-obstacle environments based on a Deep Reinforcement Learning approach," *Applied Soft Computing*, 2022.

[3] B. Joshi et al., "Sim-to-Real Deep Reinforcement Learning based Obstacle Avoidance for UAVs under Measurement Uncertainty," arXiv:2303.07243, 2023.

[4] "Real-time path planning of controllable UAV by subgoals using goal-conditioned reinforcement learning," *Applied Soft Computing*, 2023.

[5] H. Kim, J. Choi, H. Do, and G. T. Lee, "A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning," *Drones*, 2025.

[6] E. Kaufmann et al., "Champion-level drone racing using deep reinforcement learning," *Nature*, 2023.

[7] Y. Zhou et al., "Three-Dimensional Path Planning of UAVs in a Complex Dynamic Environment Based on EE-TD3," *Symmetry*, 2023.
