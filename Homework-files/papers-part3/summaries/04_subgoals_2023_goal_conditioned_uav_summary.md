# Summary: Real-Time Path Planning of Controllable UAV by Subgoals Using Goal-Conditioned Reinforcement Learning

Source: https://www.sciencedirect.com/science/article/pii/S1568494623006786

## Citation

*Real-time path planning of controllable UAV by subgoals using goal-conditioned reinforcement learning*. (2023). Applied Soft Computing, 146, 110660.

## Main idea

This paper argues that a UAV trained only for one fixed target may not generalize well to new missions. Instead, it proposes goal-conditioned reinforcement learning where the UAV is controlled by user-defined subgoals. The UAV can then perform different behaviors by following intermediate points rather than needing a new policy for every complete route.

## Method

The agent receives goal or subgoal information and learns to reach different target points. The subgoals can be used to shape a route, allowing the UAV to perform maneuvers such as high-flying, low-flying, bypassing, or penetrating through a route. The important concept is that a complex path can be decomposed into a sequence of smaller target-reaching problems.

## Why it matters for this project

This directly supports the Part 3 decision to include sequential targets in Stage 3B. In our experiments, a very far target can be difficult because the policy may drift or fail to stabilize. Subgoals make the task easier by giving the policy shorter local objectives.

## Design inspiration

- Include target-relative observations such as `dx`, `dy`, `dz`, and distance.
- Add `target_index` and target progress in the observation.
- Implement sequential targets A, B, and C.
- Treat longer routes as a chain of simpler navigation tasks.

## Connection to Part 3 stages

This paper most strongly supports:

- Stage 3A: single random target
- Stage 3B: multiple sequential targets
- Stage 6: sequential targets with obstacles

The current deadline plan may only complete Stage 3, so this paper is especially useful for explaining why Stage 3B is a meaningful final scope.

## Limitation for our work

The paper is more advanced than the current homework implementation. Our version is simpler: we use fixed-size observations, PPO, and a small number of sequential targets instead of a full goal-conditioned research framework.
