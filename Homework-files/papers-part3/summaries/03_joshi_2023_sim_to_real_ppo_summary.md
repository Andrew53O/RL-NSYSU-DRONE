# Summary: Sim-to-Real Deep Reinforcement Learning Based Obstacle Avoidance for UAVs Under Measurement Uncertainty

Source: https://arxiv.org/abs/2303.07243

Local PDF: `../pdfs/03_joshi_2023_sim_to_real_ppo_uav_obstacle_avoidance.pdf`

## Citation

Joshi, B., et al. (2023). *Sim-to-Real Deep Reinforcement Learning based Obstacle Avoidance for UAVs under Measurement Uncertainty*. arXiv:2303.07243.

## Main idea

This paper investigates PPO-based UAV waypoint navigation and obstacle avoidance when sensor measurements are uncertain. The authors train and test in simulation, then consider how uncertainty affects a learned UAV policy. This is useful because real UAVs and simulators rarely provide perfectly clean state estimates.

## Method

The UAV task is modeled with continuous states and continuous actions. The environment includes varying obstacle layouts and measurement uncertainty. The paper studies how different levels of noise affect policy performance and discusses filtering methods such as low-pass and Kalman filtering.

## Why it matters for this project

In this homework, the drone receives pose, velocity, and sonar information from ROS/Gazebo. Even in simulation, timing, reset behavior, and sensor updates can produce noisy learning. This paper supports treating simulation results carefully and logging evaluation metrics beyond training reward.

## Design inspiration

- Use PPO for continuous UAV navigation.
- Evaluate policy performance under repeatable episode tests, not only training reward.
- Log success rate, timeout rate, safety events, and distance-to-target metrics.
- Keep sensor processing simple and robust.

## Connection to Part 3 stages

The paper is relevant across all stages. For Stages 1-3, it supports stable evaluation of waypoint reaching. For Stages 4-6, it supports using sonar cautiously because obstacle distance readings can be imperfect or delayed.

## Limitation for our work

This homework does not attempt full sim-to-real transfer. The useful lesson is narrower: noisy measurements and simulation timing affect policy behavior, so training and evaluation settings must be logged and kept consistent.
