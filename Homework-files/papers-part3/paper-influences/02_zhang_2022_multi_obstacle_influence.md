# Influence Analysis: Zhang, Li, and Dong (2022)

## Citation

Zhang, S., Li, Y., & Dong, Q. (2022). *Autonomous navigation of UAV in multi-obstacle environments based on a Deep Reinforcement Learning approach*. Applied Soft Computing, 115, 108194.

Local access note:

```text
Homework-files/papers-part3/pdfs/02_zhang_2022_td3_multi_obstacle_uav_ACCESS_NOTE.md
```

Official source:

```text
https://www.sciencedirect.com/science/article/abs/pii/S1568494621010383
```

## Why This Is One Of The Two Most Influential Papers

This paper is the strongest influence on the obstacle-avoidance part of the project. My final selected homework task is Task D: autonomous obstacle avoidance using sonar. The paper studies UAV autonomous navigation in multi-obstacle environments using deep reinforcement learning.

Even though the paper uses TD3 and my project uses PPO, the design lesson is still very important:

```text
the policy needs local obstacle information,
and it should learn how to trade off target progress and collision avoidance.
```

That is exactly what my Stage 4 and Stage 5 environments try to do.

## What The Paper Does

Zhang, Li, and Dong study UAV navigation in environments with multiple obstacles. Their work is motivated by the limitation of traditional path-planning methods in complex or changing environments. In simple open space, flying toward a target is enough. In obstacle environments, the UAV must decide when to keep moving toward the goal and when to avoid nearby danger.

The paper proposes a deep reinforcement learning approach based on **Twin Delayed Deep Deterministic Policy Gradient (TD3)**. TD3 is an actor-critic algorithm designed for continuous control. It is related to DDPG but improves stability by using twin critics, delayed policy updates, and target policy smoothing.

The paper's UAV learns a policy for navigation and obstacle avoidance in simulation. The agent receives environmental observations and outputs control actions. The reward is designed so that the UAV is encouraged to reach the target while avoiding obstacles.

The most important idea for my project is that the paper does not treat obstacle avoidance as only a fixed global path. Instead, the agent uses environmental observations to make local decisions.

## Important Design Idea: Observation Changes Matter

One key point from the paper is that environmental observation changes can be useful for obstacle navigation. It is not enough to know only the current obstacle distance. The policy also benefits from knowing whether the situation is getting more dangerous or safer.

For example, these two cases are different:

```text
front obstacle distance = 0.4 m and decreasing
front obstacle distance = 0.4 m and increasing
```

In the first case, the UAV is moving toward danger. In the second case, the UAV may already be avoiding the obstacle successfully.

This idea directly influenced my sonar design.

## Influence On My Sonar Observation Space

Stages 1-3 do not use sonar. In those stages, sonar fields are masked so the policy focuses on target navigation. From Stage 4 onward, sonar becomes active.

The final observation includes:

```text
sonar ranges
sonar risks
previous sonar ranges
sonar trends
sonar_enabled flag
```

This design is strongly connected to Zhang et al.'s idea that obstacle/environment observations and their changes are important.

| My Observation Feature | Why It Exists | Connection To Zhang et al. |
| --- | --- | --- |
| Sonar ranges | Tell the policy current obstacle distance | Current environmental observation |
| Sonar risks | Convert distance into a clearer danger signal | Makes obstacle proximity easier for RL to use |
| Previous sonar ranges | Give one-step memory | Supports observation-change reasoning |
| Sonar trends | Tell whether obstacle distance is increasing or decreasing | Direct adaptation of environmental observation changes |
| Sonar enabled flag | Tells the policy whether sonar is real or masked | Allows one fixed observation shape across curriculum stages |

This is important because raw sonar alone can be ambiguous. A range value tells the drone how close something is now, but not whether the drone is moving into danger or escaping from danger. The previous range and trend features make the observation more informative.

## Influence On Stage 4

Stage 4 is the main Task D demonstration.

The setup is:

```text
final target = (10.0, 0.0, 1.0)
one cone obstacle near x = 5.0
sonar active
success distance = 0.25 m
```

The drone must move toward the far target while reacting to obstacle sonar. The policy cannot simply fly straight because the obstacle is placed near the direct path. It also cannot simply avoid obstacles forever because it still needs to reach the target.

This is the same high-level tradeoff studied by Zhang et al.:

```text
target progress versus obstacle safety
```

My Stage 4 policy reached the target in most episodes, but still had unsafe-sonar failures. This result is realistic because obstacle avoidance is harder than open-space navigation. It also shows why this paper matters: multi-objective obstacle navigation needs careful state and reward design.

## Influence On Stage 5

Stage 5 extends Stage 4 to multiple obstacles.

The setup is:

```text
final target = (10.0, 0.0, 1.0)
multiple cone obstacles along the route
sonar active
```

This stage is even closer to Zhang et al.'s multi-obstacle theme. My Stage 5 result was not successful: the drone usually terminated with unsafe sonar. That failure is still useful because it shows that the policy learned some long-range forward behavior, but the multi-obstacle avoidance strategy was not robust enough.

This matches the paper's motivation: multi-obstacle UAV navigation is harder than simple target reaching. The drone must respond to several local obstacle situations while still maintaining progress toward the goal.

## Influence On Reward Function

The paper supports using reward shaping for obstacle navigation. Sparse reward is not enough because the UAV may crash many times before it ever receives a success reward.

My Stage 4 and Stage 5 reward includes:

```text
mission-goal progress reward
distance progress reward
axis progress reward
sonar mean-risk penalty
sonar max-risk penalty
unsafe-sonar terminal penalty
safety-filter penalty
timeout penalty
success bonus
```

The most important obstacle-related terms are:

| Reward Term | Purpose |
| --- | --- |
| Sonar mean-risk penalty | Avoid staying generally close to obstacles |
| Sonar max-risk penalty | React strongly to the most dangerous sonar reading |
| Unsafe-sonar termination | Stop the episode before collision-like behavior |
| Mission progress reward | Prevent the drone from only avoiding and never reaching the target |
| Stuck/timeout penalty | Discourage freezing near obstacles |

These terms are needed because obstacle avoidance has competing objectives. If the reward only gives target progress, the drone may crash into the obstacle. If the reward only gives sonar safety, the drone may avoid the obstacle but stop moving toward the target. The final reward balances both.

## Influence On Evaluation Metrics

This paper also influenced the way I evaluate obstacle stages. For obstacle avoidance, success rate alone is not enough. A policy can reach the target sometimes but be unsafe in other episodes.

My evaluation logs include:

```text
success_rate
crash_or_unsafe_rate
mission_goal_distance
minimum_obstacle_sonar_range
sonar_near_miss_count
safety_filter_overrides
average commanded velocity
```

These metrics make the obstacle behavior visible. For example, Stage 4 had an 80% success rate but also unsafe-sonar failures. Stage 5 had 0% success and 100% unsafe/crash rate, which clearly shows that the multi-obstacle stage is not solved.

## Why It Matters Even Though I Use PPO, Not TD3

The paper uses TD3, while my project uses PPO. This is a difference in algorithm choice, but the environment design lesson still applies.

I chose PPO because:

- it is stable,
- it is easy to use with Stable-Baselines3,
- it is deadline-friendly,
- it works well with the vector observation and continuous action space.

Zhang et al. influenced the **obstacle task formulation**, not the exact algorithm. The main borrowed ideas are:

```text
use local obstacle observations
include observation changes/trends
shape reward for target reaching and obstacle safety
evaluate in obstacle environments
```

These ideas can be used with PPO, TD3, SAC, or other continuous-control RL algorithms.

## What I Did Differently

My implementation adapts the paper's general multi-obstacle idea into a simpler ROS/Gazebo sonar setting:

- I use sonar instead of richer environment sensors.
- I use PPO instead of TD3.
- I use velocity commands `[vx_cmd, vy_cmd, vz_cmd]`.
- I mask sonar before Stage 4 so basic navigation is learned first.
- I keep one fixed 41-value observation shape across all stages.
- I use unsafe-sonar termination as a practical collision-avoidance guardrail.

These changes fit the homework deadline and the available drone simulator.

## Final Summary

Zhang, Li, and Dong (2022) influenced the **obstacle-avoidance design** of my project. Their work supports the idea that UAV obstacle navigation needs local environmental information and reward terms that balance goal progress with safety. In my implementation, this appears as sonar ranges, sonar risks, previous sonar ranges, sonar trends, sonar-risk penalties, unsafe-sonar termination, and Stage 4/5 obstacle-world evaluation.

Simple version:

```text
Zhang et al. taught me what information the drone needs for obstacle avoidance.
```

