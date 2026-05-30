# Part 3 Literature Review Notes

These notes support the Part 3 curriculum design. Because the project deadline is close, the implementation focuses on PPO with a simple continuous action space and a staged curriculum up to Stage 3, while keeping the later sonar-obstacle stages documented for extension.

## Design connection to this project

Our Part 3 design uses:

- Continuous actions: `[vx_cmd, vy_cmd, vz_cmd]`.
- Goal-relative observations: current position, velocity, target delta, distance, target index, and masked sonar fields.
- Curriculum learning: vertical control, horizontal control, combined navigation, then optional obstacle stages.
- Reward shaping: distance progress, axis-specific progress, stability penalties, action smoothness, and success bonuses.
- Fixed observation/action shape so checkpoints can continue from easier stages to harder stages.

## Paper 1: Kabas, 2022 - Autonomous UAV Navigation via Deep Reinforcement Learning Using PPO

Source: https://gcris.agu.edu.tr/entities/publication/2081266d-7411-42a0-a921-21174017b4d3

Kabas used PPO as a high-level UAV navigation controller in simulation, with end-to-end training and a continuous reward function. The paper reports autonomous UAV navigation results using Unreal Engine and AirSim, and it is directly relevant because this homework also uses simulation to train a policy before any real-world deployment.

How it inspired this project:

- It supports choosing PPO for deadline-friendly UAV navigation.
- It supports using a shaped continuous reward rather than only sparse success/failure rewards.
- It supports training in simulation first, then evaluating policy behavior through repeatable episodes.
- It motivated keeping the policy network simple instead of adding camera or recurrent models.

## Paper 2: Zhang, Li, and Dong, 2022 - Autonomous Navigation of UAV in Multi-Obstacle Environments Based on a Deep Reinforcement Learning Approach

Source: https://www.sciencedirect.com/science/article/abs/pii/S1568494621010383

Zhang et al. proposed a TD3-based UAV navigation approach for random and dynamic multi-obstacle environments. Their method adds changes in environmental observations into the actor-critic input and evaluates UAV behavior in simulation.

How it inspired this project:

- It supports including both current observations and short-term change/trend features.
- It supports the later sonar design where range trends help identify whether obstacles are approaching.
- It supports using continuous control for UAV navigation instead of discrete movement commands.
- It also shows why obstacle avoidance should be introduced after simpler navigation skills are learned.

## Paper 3: Joshi et al., 2023 - Sim-to-Real Deep Reinforcement Learning Based Obstacle Avoidance for UAVs Under Measurement Uncertainty

Source: https://arxiv.org/abs/2303.07243

Joshi et al. evaluated PPO for UAV waypoint navigation and obstacle avoidance with continuous state/action spaces. Their work studies measurement uncertainty and randomized obstacle scenarios, which is close to the uncertainty in ROS/Gazebo sensor readings.

How it inspired this project:

- It supports PPO for continuous UAV waypoint navigation.
- It supports robustness thinking: sensor readings and pose estimates should be treated as noisy.
- It supports separating training and evaluation under different conditions.
- It motivated logging safety metrics and not trusting reward curves alone.

## Paper 4: Real-Time Path Planning of Controllable UAV by Subgoals Using Goal-Conditioned Reinforcement Learning, 2023

Source: https://www.sciencedirect.com/science/article/pii/S1568494623006786

This paper trains a UAV agent to follow user-defined subgoals, allowing it to perform different maneuvers without retraining from scratch for each mission. The main idea is that a large navigation task can be easier when decomposed into smaller target-reaching tasks.

How it inspired this project:

- It supports the Stage 3B idea of sequential targets A, B, and C.
- It supports using `target_index` and target progress in the observation.
- It supports curriculum learning from single-target control to multi-target routes.
- It explains why long routes may need intermediate goals instead of one far target.

## Paper 5: Jeon, 2025 - A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning

Source: https://www.mdpi.com/2504-446X/9/1/26

This paper combines curriculum learning and goal-conditioned reinforcement learning for UAV path planning. It explicitly uses progressively harder missions and subgoals to make the UAV more controllable over varied paths.

How it inspired this project:

- It strongly supports our staged Part 3 curriculum.
- It supports learning basic vertical and horizontal control before combined navigation.
- It supports using subgoals for longer routes and round-trip or multi-point missions.
- It justifies why Stage 1A, 1B, 2A, 2B, 3A, and 3B are separated rather than training everything at once.

## Paper 6: Kaufmann et al., 2023 - Champion-Level Drone Racing Using Deep Reinforcement Learning

Source: https://www.nature.com/articles/s41586-023-06419-4

Kaufmann et al. demonstrated high-performance drone racing using deep reinforcement learning. Although the task is much harder than this homework, the paper is useful because it shows that learned policies can control agile drones when trained in simulation with careful evaluation and real-world-aware design.

How it inspired this project:

- It supports simulation-first drone RL as a valid engineering approach.
- It highlights the importance of matching simulation timing and dynamics to the policy.
- It reinforces why training configuration, evaluation settings, and model checkpoints must be logged.
- It motivates staged evaluation instead of trusting a single training reward curve.

## Paper 7: Zhou et al., 2023 - Three-Dimensional Path Planning of UAVs in a Complex Dynamic Environment Based on EE-TD3

Source: https://www.mdpi.com/2073-8994/15/7/1371

This paper proposes an Environment Exploration TD3 method for 3D UAV path planning. It compares DDPG, TD3, and an improved TD3 variant, and emphasizes dynamic reward design to improve convergence in 3D environments.

How it inspired this project:

- It supports using axis-aware 3D reward shaping.
- It supports treating UAV path planning as a continuous-control problem.
- It shows that reward design can strongly affect convergence speed and final path quality.
- It motivates our separate x/y/z progress terms instead of only using final distance.

## Summary for the report

The literature supports three main decisions in this project:

1. PPO is a reasonable deadline-friendly algorithm for simulated UAV control, especially with a continuous action space and shaped rewards.
2. Curriculum learning is important because training full navigation and obstacle avoidance at once can produce unstable behavior.
3. Goal-relative observations, target progress, and subgoals are useful for generalizing from simple vertical/horizontal control to longer missions.

For this homework, the practical scope is to complete and evaluate Stages 1-3 first. Stages 4-6 remain aligned with the UAV obstacle-avoidance literature, but they require more training time and careful sonar reward tuning.
