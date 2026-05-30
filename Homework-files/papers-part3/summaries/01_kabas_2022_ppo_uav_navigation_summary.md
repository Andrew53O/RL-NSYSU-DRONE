# Summary: Autonomous UAV Navigation via Deep Reinforcement Learning Using PPO

Source: https://gcris.agu.edu.tr/entities/publication/2081266d-7411-42a0-a921-21174017b4d3

## Citation

Kabas, B. (2022). *Autonomous UAV Navigation via Deep Reinforcement Learning Using PPO*. 30th IEEE Signal Processing and Communications Applications Conference.

## Main idea

This paper applies Proximal Policy Optimization (PPO) to autonomous UAV navigation. The navigation controller is trained in simulation and uses a continuous reward function to learn a high-level control policy. The paper uses visual inputs in Unreal Engine and Microsoft AirSim, but the most relevant point for this project is not the camera input. The useful idea is that PPO can be used as a practical high-level controller for UAV navigation when the environment is modeled as an MDP and the reward gives continuous feedback toward successful navigation.

## Method

The paper formulates autonomous UAV navigation as a deep reinforcement learning problem. PPO is used because it is stable and commonly used for continuous-control style robotic tasks. The agent is trained in simulation and evaluated on navigation success. The paper reports strong navigation performance in the simulated visual-navigation setting, showing that a compact neural policy can be practical for aerial vehicles.

## Why it matters for this project

This homework also uses PPO as the main algorithm, but with ROS 2 and Gazebo instead of Unreal/AirSim. The Part 3 implementation keeps the same spirit: a high-level PPO policy outputs velocity commands, while the simulator provides pose and velocity feedback. Since the assignment deadline is short, PPO is a good choice because it is available in Stable-Baselines3, has robust defaults, and does not require implementing a custom off-policy replay pipeline.

## Design inspiration

- Use PPO with `MlpPolicy` for deadline-friendly UAV control.
- Use a continuous action space rather than discrete commands.
- Use dense reward shaping so the agent receives feedback before final success.
- Train and evaluate in simulation with repeatable target-reaching episodes.

## Connection to Part 3 stages

This paper mainly supports Stages 1-3. Stage 1 learns altitude control, Stage 2 learns horizontal movement, and Stage 3 combines them. PPO is appropriate for this because all stages use the same continuous command interface `[vx_cmd, vy_cmd, vz_cmd]`.

## Limitation for our work

The paper focuses on visual navigation, while this project intentionally avoids camera-based RL because of time limits. Therefore, this project uses pose, velocity, target-relative state, and later sonar instead of image observations.
