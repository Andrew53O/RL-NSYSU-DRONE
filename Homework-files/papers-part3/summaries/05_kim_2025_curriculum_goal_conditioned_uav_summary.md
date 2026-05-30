# Summary: A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning

Source: https://www.mdpi.com/2504-446X/9/1/26

Local PDF: `../pdfs/05_kim_2025_fully_controllable_uav_curriculum_goal_conditioned.pdf`

## Citation

Kim, H., Choi, J., Do, H., & Lee, G. T. (2025). *A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning: From Straight Forward to Round Trip Missions*. Drones, 9(1), 26.

## Main idea

This paper combines curriculum learning and goal-conditioned reinforcement learning to make a UAV more controllable across increasingly difficult missions. The authors show that starting with simpler tasks and gradually increasing mission complexity can help the agent learn more reliable behavior.

## Method

The paper structures UAV path planning as a progression from simple target reaching to more complex subgoal and round-trip missions. The policy receives goal information and learns to act based on the current goal. The use of curriculum learning is important because it avoids forcing the UAV to learn all behaviors at once.

## Why it matters for this project

This is one of the closest matches to the Part 3 design. Part 3 was created because the previous Part 2 pipeline became too complex. The new design deliberately separates learning into:

- Stage 1: vertical altitude control
- Stage 2: horizontal x control
- Stage 3: combined navigation and sequential targets
- later stages: sonar obstacle avoidance

## Design inspiration

- Use a curriculum instead of training full navigation immediately.
- Train basic movement skills before multi-target missions.
- Use target-relative state so the same policy can work for different goals.
- Use sequential targets for longer missions.

## Connection to Part 3 stages

This paper supports the entire six-stage Part 3 plan. It especially justifies the detailed split into Stage 1A, 1B, 2A, 2B, 3A, and 3B. If the final report only reaches Stage 3, this paper still strongly supports the reduced scope because it argues that basic controllability is a necessary foundation.

## Limitation for our work

The paper is a more complete research system, while this homework implementation is constrained by ROS/Gazebo runtime, Docker setup, and limited training time. Our implementation keeps the same idea but uses a simpler PPO pipeline.
