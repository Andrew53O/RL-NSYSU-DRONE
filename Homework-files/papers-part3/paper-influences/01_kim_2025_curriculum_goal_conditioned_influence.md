# Influence Analysis: Kim et al. (2025)

## Citation

Kim, H., Choi, J., Do, H., & Lee, G. T. (2025). *A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning: From Straight Forward to Round Trip Missions*. Drones, 9(1), 26.

Local file:

```text
Homework-files/papers-part3/pdfs/05_kim_2025_fully_controllable_uav_curriculum_goal_conditioned.pdf
```

Official source:

```text
https://www.mdpi.com/2504-446X/9/1/26
```

## Why This Is The Most Influential Paper

This paper is the closest match to the final Part 3 training strategy. The most important lesson from this paper is that a UAV agent should not be trained directly on a hard mission from the beginning. Instead, it should first learn simple flight skills, then gradually learn harder missions.

That idea directly matches the reason Part 3 was created. The earlier Part 2 experiments tried to make the drone solve target navigation too directly. The drone had to control both horizontal position and altitude at the same time, and the result was unstable. It could often move in the correct general direction, but it stopped too far away from the target or drifted in altitude. Because of that, Part 3 was redesigned as a curriculum:

```text
first learn z movement
then learn x movement
then combine x and z
then learn sequential targets
then activate sonar for obstacle avoidance
```

This is the same philosophy as Kim et al.: build complex UAV behavior from easier skills.

## What The Paper Does

Kim et al. study UAV path planning using a combination of:

```text
curriculum learning
goal-conditioned reinforcement learning
subgoal navigation
actor-critic reinforcement learning
```

The paper argues that a UAV trained only for one simple target-reaching behavior may not generalize to more complex missions. For example, if a UAV only learns to fly forward to one target, it may fail when the mission requires turning, returning, or visiting several points.

To solve this, the authors use **goal-conditioned reinforcement learning**. In goal-conditioned RL, the policy does not only observe the UAV state. It also receives information about the current goal. This changes the learning problem from:

```text
What should I do in this state?
```

to:

```text
What should I do in this state, given this goal?
```

The paper expresses this by extending the MDP from:

```text
(S, A, T, R, gamma)
```

to:

```text
(S, G, A, T, R, gamma)
```

where `G` is the set of goals or subgoals. This allows the same policy to react differently depending on the current target.

The paper also uses **subgoals**. A complex mission is decomposed into smaller temporary goals. The UAV reaches the first subgoal, then the next subgoal becomes active. The mission is successful only after the UAV reaches all required subgoals and the final goal.

## Curriculum Learning In The Paper

The paper uses curriculum learning because directly training complex behavior is difficult. The UAV first learns basic target-reaching tasks, then progressively learns harder missions. The authors describe this as similar to human learning: simple skills are learned first, then reused for harder tasks.

The paper's curriculum teaches the UAV to become more controllable. Instead of only learning the shortest path to one target, the UAV learns to use subgoals to perform more diverse routes, including missions with multiple subgoals and round-trip behavior.

This is important because obstacle avoidance is also a complex behavior. A drone that cannot reliably reach simple targets will not learn obstacle avoidance well. It may look like the obstacle reward is wrong, but the real problem may be that basic navigation was never learned.

## Influence On My Six-Stage Curriculum

Kim et al. had the strongest influence on the full Part 3 stage design.

| My Stage | Design Connection To Kim et al. |
| --- | --- |
| Stage 1A | Fixed easy target, equivalent to teaching the simplest single skill first |
| Stage 1B | Random altitude target, similar to adding variation so the policy does not memorize one coordinate |
| Stage 2A | Fixed horizontal target, another isolated basic skill |
| Stage 2B | Random x target, teaches generalization to different horizontal goals |
| Stage 3A | Combined x-z target, combines previously learned skills |
| Stage 3B | Three sequential targets, directly inspired by subgoal/goal-conditioned navigation |
| Stage 4 | Long mission with local subgoals and obstacle avoidance |
| Stage 5 | Harder multi-obstacle mission built after Stage 4 |

The paper influenced the decision to make Stage 1 and Stage 2 very simple. At first, this may look too easy, but it was useful. The drone learned how each action dimension affects movement before the environment asked it to solve combined navigation.

This is why the Part 3 curriculum succeeded where the earlier Part 2 design struggled. Part 2 tried to solve the combined target problem too early. Part 3 made the drone learn the pieces separately.

## Influence On Observation Space

The paper strongly influenced the use of **goal-relative observations**.

In my project, the observation includes:

```text
current position: x, y, z
current velocity: vx, vy, vz
relative target: dx, dy, dz
distance to active target
target progress / target index
```

The most important part is:

```text
dx, dy, dz
distance_to_target
```

These values tell the policy where the current goal is relative to the drone. This makes the policy goal-conditioned in a practical way. The policy does not need to memorize one fixed coordinate. It can learn general rules:

```text
if dx is positive, move forward
if dz is positive, climb
if dy is large, reduce lateral drift
if distance is small, slow down
```

This connects directly to Kim et al.'s idea that the policy should act based on the current goal/subgoal.

## Influence On Sequential Target Handling

Stage 3B is the clearest direct influence from this paper.

In Stage 3B, the environment generates three targets:

```text
A -> B -> C
```

The drone must visit them in order. When the drone reaches A, B becomes the active target. When it reaches B, C becomes active. The episode succeeds only after reaching the full sequence.

This matches the paper's subgoal idea:

```text
current subgoal is treated as the temporary goal
after reaching it, the next subgoal becomes active
```

Because of this, my observation includes target progress/target index. This helps the policy know whether it is still early in the sequence or near the end.

## Influence On Stage 4 Dynamic Local Subgoals

The paper also influenced the long-distance Stage 4 design.

Stage 4 has a far final mission goal:

```text
(10.0, 0.0, 1.0)
```

Training directly toward a far goal can be difficult because the reward becomes less precise. To solve this, the environment uses an internal local subgoal about `1 m` ahead in x. This helps the drone continue making forward progress.

However, the local subgoal is not a hand-written obstacle path. It does not tell the drone to climb or move around the obstacle. It only keeps the drone moving toward the final goal. The sonar reward still decides whether the drone should move sideways or adjust altitude.

This is a practical adaptation of the subgoal idea from Kim et al.:

```text
Use smaller goals to make a large mission learnable.
```

## Influence On Reward Function

Kim et al. use rewards for reaching targets and penalties for failure or inefficient behavior. My reward function is more detailed because the Gazebo drone needs stable altitude, smooth velocity, and sonar safety, but the core idea is the same:

```text
make each subtask learnable with useful reward feedback
```

This influenced these reward terms:

| Reward Component | Connection To Kim et al. |
| --- | --- |
| Distance progress reward | Encourages movement toward the current goal/subgoal |
| Axis progress reward | Helps each stage learn its focused movement skill |
| Success bonus | Reinforces reaching the active target |
| Sequential target bonus | Reinforces completing subgoals in order |
| Timeout penalty | Prevents wandering instead of finishing |
| Near-target braking penalty | Helps the drone stabilize around the goal |

Without dense reward, Stage 1 and Stage 2 would learn slowly. Without subgoal-style reward, Stage 3B would be hard because the drone would only receive a useful signal after completing all targets.

## Influence On Action Space

Kim et al.'s paper uses a different action formulation, but it still supports the idea that UAV control needs actions that can express different maneuvers. My implementation uses continuous velocity commands:

```text
[vx_cmd, vy_cmd, vz_cmd]
```

This is simpler and more natural for the ROS/Gazebo drone plugin than the paper's aircraft control inputs. The influence is not the exact action definition; the influence is the idea that the policy must learn reusable movement behavior under changing goals.

## What I Did Differently

My implementation is simpler than the paper:

- I use Stable-Baselines3 PPO, not the paper's full architecture with additional exploration/self-imitation methods.
- I use ROS 2/Gazebo Classic instead of the paper's simulation setup.
- I use continuous velocity commands instead of discrete aircraft control actions.
- I add sonar obstacle features in Stage 4 and Stage 5, while Kim et al. mainly focus on path planning and subgoals.

These differences are reasonable because the homework deadline is short and the drone simulator already has a low-level controller. The learned policy only needs to provide high-level velocity commands.

## Final Summary

Kim et al. (2025) influenced the **training structure** of my project more than any other paper. It taught the key lesson that a UAV should learn basic flight skills before complex missions. This directly shaped the six-stage curriculum, goal-relative observation space, sequential targets, local subgoals, and dense progress reward.

Simple version:

```text
Kim et al. taught me how to train the drone step by step.
```

