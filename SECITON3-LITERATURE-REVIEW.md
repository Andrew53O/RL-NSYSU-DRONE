# 3. Literature Review

Recent UAV reinforcement learning research supports the main design choices used in this project: PPO-based continuous control, goal-relative observations, curriculum learning, subgoal navigation, and range-sensor-based obstacle avoidance. The following papers were selected because they are from the last five years and are closely related to PPO, TD3, UAV navigation, or UAV obstacle avoidance.

## 3.1 PPO for UAV Navigation

Kabas [1] applied Proximal Policy Optimization (PPO) to autonomous UAV navigation in simulation. The UAV was trained as a deep reinforcement learning agent, and the policy learned navigation behavior through reward feedback instead of hand-coded control rules. Although the paper used visual simulation with Unreal Engine and AirSim, the most relevant idea for this project is that PPO can be used as a practical high-level UAV controller when the environment is formulated as a Markov Decision Process.

This paper influenced the choice of PPO in this project. In Part 3, the drone policy uses a continuous action space:

```text
[vx_cmd, vy_cmd, vz_cmd]
```

This design lets the agent output smooth velocity commands instead of choosing from a small set of discrete actions. The paper also supports using shaped rewards, because sparse success-only rewards are difficult for UAV navigation tasks. For this reason, the Part 3 reward includes distance progress, axis-specific progress, stability penalties, action penalties, and success bonuses.

## 3.2 Multi-Obstacle UAV Navigation With TD3

Zhang, Li, and Dong [2] proposed a TD3-based UAV navigation method for random and dynamic multi-obstacle environments. Their work treated obstacle avoidance as a continuous-control reinforcement learning problem and emphasized that the agent needs meaningful environmental information about obstacles, not only target position. The paper also showed that obstacle avoidance becomes more difficult when the environment contains multiple obstacles or changing obstacle conditions.

This paper influenced the Stage 4 sonar design. In this project, sonar is not used in Stages 1-3 because those stages focus on basic navigation. From Stage 4 onward, sonar fields become active in the observation space. The observation includes sonar ranges, risk values, previous sonar values, and sonar trends. This follows the idea that obstacle avoidance should not depend only on one instantaneous distance measurement. If the sonar distance is decreasing, the drone is moving toward danger and should react earlier.

The paper also supports using continuous velocity actions for obstacle avoidance. In Stage 4, the policy still outputs:

```text
[vx_cmd, vy_cmd, vz_cmd]
```

but sonar reward terms penalize unsafe obstacle proximity and encourage the drone to continue progressing toward the final target while avoiding the obstacle.

## 3.3 PPO Obstacle Avoidance Under Measurement Uncertainty

Joshi et al. [3] studied PPO-based UAV waypoint navigation and obstacle avoidance under measurement uncertainty. Their work is important because UAV sensors are not perfect, even in simulation. Pose, velocity, and obstacle measurements may be noisy, delayed, or inconsistent. The paper evaluated how uncertainty affects learned obstacle avoidance policies and discussed the importance of robust evaluation.

This paper influenced how the project evaluates trained policies. In Part 3, training reward alone is not considered enough to judge success. The evaluation script logs additional metrics such as:

```text
mission_goal_distance
minimum_obstacle_sonar_range
unsafe_sonar
sonar_near_miss_count
safety_filter_overrides
average_cmd_vx, average_cmd_vy, average_cmd_vz
```

This was important in the project because some training curves looked good even when deterministic evaluation exposed weaknesses. For Stage 4, the final evaluation showed that the drone reached the far target in most episodes, but unsafe sonar still occurred in some runs near the obstacle. This kind of safety-focused evaluation follows the lesson from Joshi et al. that UAV RL should be judged using robustness and safety metrics, not only reward.

## 3.4 Goal-Conditioned Subgoal Navigation

The subgoal-based UAV path planning paper [4] proposed decomposing larger UAV navigation missions into smaller goal-reaching tasks. Instead of forcing the UAV to solve a long path directly, the agent can follow intermediate subgoals. This helps the UAV perform longer or more complex missions without needing a different policy for every route.

This idea influenced Stage 3B and Stage 4. In Stage 3B, the drone learns to visit multiple target points in sequence. The observation therefore includes target progress information, so the policy knows which target is active. In Stage 4, the final mission target is far away at approximately:

```text
(10.0, 0.0, 1.0)
```

However, the environment uses an internal dynamic local subgoal to encourage steady forward progress. The local subgoal is not a hand-coded avoidance trajectory; it simply helps break the long route into smaller progress steps. Obstacle avoidance is still driven by sonar readings and sonar risk penalties. This design was chosen to avoid turning the task into pure trajectory tracking while still making the long-distance target reachable.

## 3.5 Curriculum Learning for UAV Control

Kim et al. [5] proposed a UAV framework using curriculum learning and goal-conditioned reinforcement learning. Their work showed that UAV control can become more stable when the agent learns simple tasks first and then gradually moves to harder missions. This matches the main structure of Part 3.

The curriculum in this project was designed as:

```text
Stage 1: vertical movement
Stage 2: horizontal movement
Stage 3: combined navigation and sequential targets
Stage 4: sonar obstacle avoidance
```

This staged structure was strongly influenced by curriculum learning. Instead of training obstacle avoidance immediately, the drone first learned altitude control, then horizontal movement, then combined movement, and finally obstacle avoidance. This made Stage 4 more realistic because the policy already had basic movement skills before sonar was introduced.

The paper also supports keeping the same observation and action interface across stages. In Part 3, the observation shape remains fixed, but sonar values are masked before Stage 4. This allows checkpoints from earlier stages to be reused for later stages.

## Summary of Design Influence

Together, these papers support the main design decisions in this project:

- PPO was selected because it is practical for simulated UAV continuous control and available in Stable-Baselines3.
- The action space was designed as continuous velocity control: `[vx_cmd, vy_cmd, vz_cmd]`.
- The observation space uses goal-relative information such as target delta and distance, because this helps the policy generalize across target positions.
- Sonar information is introduced only in Stage 4, after the drone has already learned basic navigation.
- Sonar observations include range, risk, previous range, and trend information to support local obstacle avoidance.
- Curriculum learning was used to reduce task difficulty and improve training stability.
- Subgoals were used for long-distance navigation, but obstacle avoidance remained based on sonar rather than a fixed hand-coded path.

## References

[1] B. Kabas, "Autonomous UAV Navigation via Deep Reinforcement Learning Using PPO," in *Proc. 30th Signal Processing and Communications Applications Conference (SIU)*, 2022. doi: 10.1109/SIU55565.2022.9864769.

[2] S. Zhang, Y. Li, and Q. Dong, "Autonomous navigation of UAV in multi-obstacle environments based on a Deep Reinforcement Learning approach," *Applied Soft Computing*, vol. 115, Art. no. 108194, 2022. doi: 10.1016/j.asoc.2021.108194.

[3] B. Joshi et al., "Sim-to-Real Deep Reinforcement Learning based Obstacle Avoidance for UAVs under Measurement Uncertainty," arXiv:2303.07243, 2023.

[4] "Real-time path planning of controllable UAV by subgoals using goal-conditioned reinforcement learning," *Applied Soft Computing*, vol. 146, Art. no. 110660, 2023. doi: 10.1016/j.asoc.2023.110660.

[5] H. Kim, J. Choi, H. Do, and G. T. Lee, "A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning: From Straight Forward to Round Trip Missions," *Drones*, vol. 9, no. 1, Art. no. 26, 2025.
