# Summary: Autonomous Navigation of UAV in Multi-Obstacle Environments Based on a Deep Reinforcement Learning Approach

Source: https://www.sciencedirect.com/science/article/abs/pii/S1568494621010383

## Citation

Zhang, S., Li, Y., & Dong, Q. (2022). *Autonomous navigation of UAV in multi-obstacle environments based on a Deep Reinforcement Learning approach*. Applied Soft Computing, 115, 108194.

## Main idea

This paper studies UAV autonomous navigation in random and dynamic multi-obstacle environments. The authors propose a TD3-based method and emphasize that UAV path planning is difficult when the environment changes or contains multiple obstacles. Instead of relying on a fixed global planner, the UAV learns a policy from interaction with the environment.

## Method

The paper uses a Twin Delayed Deep Deterministic Policy Gradient (TD3) style framework. One important design idea is that the model uses environmental observation changes as part of the policy input. This helps the agent reason not only about the current obstacle situation but also about how the situation is changing over time.

## Why it matters for this project

Although the current Part 3 deadline scope may stop around Stage 3, the later planned Stages 4-6 are sonar obstacle-avoidance stages. This paper supports the idea that obstacle avoidance should not only use one snapshot of obstacle distance. The trend or change in range matters because a decreasing obstacle distance means collision risk is increasing.

## Design inspiration

- Keep continuous actions for UAV motion control.
- Treat obstacle avoidance as a learned policy rather than a hand-coded path planner.
- Include short-term memory-like features such as previous range and trend when sonar is active.
- Introduce obstacle avoidance after basic navigation skills are learned.

## Connection to Part 3 stages

This paper is most relevant to Stage 4, Stage 5, and Stage 6. It justifies the planned sonar observation fields in the fixed Part 3 observation space:

- sonar normalized ranges
- sonar risks
- previous sonar ranges
- sonar trends

Before Stage 4 those fields are masked, but after Stage 4 they can support obstacle-avoidance learning.

## Limitation for our work

The paper uses TD3, while this project uses PPO. The algorithm differs, but the environment-design lesson still applies: obstacle avoidance needs meaningful local obstacle state and reward terms, not only target distance.
