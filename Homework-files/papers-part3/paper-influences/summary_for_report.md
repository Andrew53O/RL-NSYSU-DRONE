# Report-Ready Summary: Two Most Influential Papers

The two papers that most influenced my design were **Kim et al. (2025)** and **Zhang, Li, and Dong (2022)**.

## Influence Table

| Paper | Main Idea | What It Influenced In My Project | Where It Appears In My Implementation |
| --- | --- | --- | --- |
| Kim et al. (2025), *A Fully Controllable UAV Using Curriculum Learning and Goal-Conditioned Reinforcement Learning* | Train UAV behavior progressively using curriculum learning and goal-conditioned/subgoal navigation | Six-stage curriculum, target-relative observation, sequential targets, target index/progress, local subgoals for long missions | `HW2_Work/part3/drone_env.py` stage definitions, fixed 41-value observation, Stage 3B target sequence, Stage 4/5 local subgoal logic |
| Zhang, Li, and Dong (2022), *Autonomous Navigation of UAV in Multi-Obstacle Environments Based on a Deep Reinforcement Learning Approach* | Use DRL for UAV navigation in multi-obstacle environments with obstacle/environment observations | Sonar ranges, sonar risks, previous sonar ranges, sonar trends, sonar-risk reward penalties, unsafe-sonar termination | `HW2_Work/part3/drone_env.py` sonar observation fields, sonar risk calculation, Stage 4/5 reward terms, `test.py` unsafe-sonar evaluation logs |

## Kim et al. (2025)

Kim et al. had the strongest influence on the overall training strategy. Their paper combines curriculum learning and goal-conditioned reinforcement learning so that a UAV can first learn simple target-reaching behavior and then progress to harder missions with multiple subgoals and round-trip navigation.

This directly inspired my Part 3 curriculum. Instead of training obstacle avoidance from scratch, I split the problem into smaller stages. Stage 1 teaches altitude control, Stage 2 teaches horizontal x movement, Stage 3 combines x and z navigation and adds sequential targets, and Stage 4 activates sonar for obstacle avoidance. This design was important because my earlier Part 2 experiments tried to learn combined navigation too directly and often stopped too far from the target.

Kim et al. also influenced my observation space. My policy receives goal-relative terms such as `dx`, `dy`, `dz`, and distance to the active target. This makes the policy goal-conditioned in practice because the drone can decide its action based on where the current target is relative to itself. Stage 3B also follows the subgoal idea from the paper: the drone reaches target A, then B, then C, and the active target changes after each success.

The paper also influenced the Stage 4 dynamic local subgoal design. The final Stage 4 target is far away at `(10.0, 0.0, 1.0)`, so the environment uses an internal local subgoal about one meter ahead in x to keep progress learnable. This is not a hand-written avoidance path; sonar still decides whether the drone should move around the obstacle.

## Zhang, Li, and Dong (2022)

Zhang, Li, and Dong had the strongest influence on the obstacle-avoidance part of my project. Their paper studies UAV autonomous navigation in multi-obstacle environments using deep reinforcement learning. The key lesson I used is that obstacle avoidance needs local obstacle/environment observations, not only target position.

This influenced my Stage 4 and Stage 5 sonar observation design. From Stage 4 onward, the observation includes sonar ranges, sonar risk values, previous sonar ranges, and sonar trends. The trend feature is important because a current sonar distance alone does not say whether the obstacle is getting closer or farther. If the range is decreasing, the drone is moving toward danger. If it is increasing, the avoidance maneuver may already be working.

The paper also influenced my reward function. In obstacle stages, the reward combines mission-goal progress with sonar safety. The drone is rewarded for moving toward the final target, but it is penalized for high sonar risk. If sonar becomes dangerously close, the episode terminates as `unsafe_sonar`. This prevents the policy from learning to simply rush forward into the obstacle.

Even though Zhang et al. use TD3 and my project uses PPO, the paper still strongly influenced my design. The algorithm is different, but the task formulation is similar: the UAV must learn how to trade off target progress and obstacle avoidance using local environmental information.

## Short Explanation

In simple words:

```text
Kim et al. taught me how to train the drone step by step.
Zhang et al. taught me what information the drone needs to avoid obstacles.
```

Together, these two papers shaped the final MDP design: goal-relative observations, continuous velocity actions, dense progress reward, curriculum training, sequential targets, sonar-risk features, and obstacle-safety penalties.
