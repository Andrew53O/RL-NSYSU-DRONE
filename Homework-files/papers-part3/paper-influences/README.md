# Paper Influences for Part 3 Design

This folder explains which papers most influenced the final Part 3 drone RL design.
The project uses a six-stage PPO curriculum for sonar-based obstacle avoidance:

```text
Stage 1: altitude control
Stage 2: horizontal x movement
Stage 3: combined and sequential target navigation
Stage 4: one-obstacle sonar avoidance
Stage 5: multi-obstacle sonar avoidance extension
Stage 6: planned full sequential obstacle mission
```

The two papers with the strongest design influence are:

| Rank | Paper | Main Influence |
| ---: | --- | --- |
| 1 | Kim et al. (2025), *A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning* | Six-stage curriculum, goal-relative observations, sequential targets, and subgoal-based long missions |
| 2 | Zhang, Li, and Dong (2022), *Autonomous Navigation of UAV in Multi-Obstacle Environments Based on a Deep Reinforcement Learning Approach* | Obstacle-aware state design, sonar range/risk/trend features, and obstacle-safety reward terms |

## Files

- `01_kim_2025_curriculum_goal_conditioned_influence.md`
  - Detailed explanation of how curriculum learning and goal-conditioned/subgoal navigation influenced the Part 3 stages.
- `02_zhang_2022_multi_obstacle_influence.md`
  - Detailed explanation of how multi-obstacle UAV DRL influenced the sonar observation and obstacle reward design.
- `summary_for_report.md`
  - Shorter version that can be pasted into the report or presentation.

## Design Mapping

| Project Design Choice | Strongest Paper Influence |
| --- | --- |
| Train simple movement before obstacle avoidance | Kim et al. (2025) |
| Keep target-relative observation terms `dx`, `dy`, `dz`, and distance | Kim et al. (2025) |
| Add target index/progress for sequential targets | Kim et al. (2025) |
| Use internal local subgoals for long Stage 4/5 missions | Kim et al. (2025) |
| Add sonar ranges, risks, previous ranges, and trends | Zhang et al. (2022) |
| Penalize sonar risk and unsafe obstacle proximity | Zhang et al. (2022) |
| Treat obstacle avoidance as learned local decision-making | Zhang et al. (2022) |

