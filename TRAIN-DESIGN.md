# PPO Training Design Document (Task D)

This document provides a detailed breakdown of the Reinforcement Learning architecture implemented for Task D (Autonomous Obstacle Avoidance) using the Stable-Baselines3 PPO algorithm, `train.py`, and `drone_env.py`.

## 1. Environment Architecture

The environment bridges Gymnasium and ROS 2 / Gazebo Classic.
Instead of using a camera, the drone uses **sonar sensors** mimicking a 7-sector spatial awareness system to keep the observation space small and the training rapid (MLP policy instead of CNN).

### 1.1 Observation Space (`observation_space`)
The observation space is a flat, continuous 43-dimensional array (`spaces.Box`), where all values are normalized to roughly `[-2.5, 2.5]` to ensure stable neural network gradients.

**Dimensions Breakdown:**
1. **Pose (3):** Normalized Drone Position `[x/8.0, y/8.0, z/5.0]`
2. **Velocity (3):** Drone Linear Velocity `[vx, vy, vz/0.5]`
3. **Relative Target (3):** Normalized distance to target `[dx/8.0, dy/8.0, dz/5.0]`
4. **Euclidean Target Distance (1):** Distance to target `dist / 12.0`
5. **Sonar Ranges (7):** Normalized readings from the 7 sectors (front_left, front_center, front_right, front_up, front_down, side_left, side_right).
6. **Sonar Risks (7):** Processed risk factor for each sector (closer = higher risk).
7. **Previous Sonar Ranges (7):** The ranges from the previous timestep, providing temporal awareness.
8. **Range Trends (7):** Rate of change in obstacle distances (identifying if the drone is actively flying *towards* a wall).
9. **Recent Obstacle Min (1):** Minimum detected obstacle range over the last few frames (memory/hysteresis).
10. **Downward Sonar (1):** Ground proximity `down_sonar / 10.0`.
11. **Downward Sonar Risk (1):** Penalty indicator for flying dangerously close to the ground.
12. **Risk Balances (2):** `left_right_risk_balance` and `up_down_risk_balance`, helping the drone steer away from walls intuitively.

### 1.2 Action Space (`action_space`)
The action space is a continuous 3-dimensional control vector determining the drone's linear velocity in `[x, y, z]` axes.
* **Format:** `spaces.Box(low=[-1.0, -1.0, -0.5], high=[1.0, 1.0, 0.5])`
* **Interpretation:**
  * `action[0]`: Forward/Backward velocity (`vx`) bounded to `[-1.0, 1.0]` m/s.
  * `action[1]`: Left/Right velocity (`vy`) bounded to `[-1.0, 1.0]` m/s.
  * `action[2]`: Up/Down velocity (`vz`) bounded to `[-0.5, 0.5]` m/s.
* **Safety Filter:** The simulation applies a programmatic "safety filter" layer that intercepts unsafe commands (e.g., commanding `vx=1.0` when a wall is 0.1m ahead) and overrides them. The RL agent receives a penalty when its action is overridden so it learns not to rely on the filter.

---

## 2. Reward Function

The reward function relies heavily on dense shaping to guide the drone seamlessly towards the target while dodging obstacles. The goal is to maximize cumulative rewards per episode.

### 2.1 Terminal Rewards (Episode Ending)
* **Success:** `+300.0`. Triggered if `current_distance < 0.4` meters (target reached).
* **Crash:** `-150.0`. Triggered if altitude `z_pos < 0.25` meters (min_altitude hit).
* **Invalid Sensor:** `-150.0`. Triggered if ROS topics die or simulate sensor failures.

### 2.2 Dense Rewards/Penalties (Calculated per step)
Every `step_dt` (0.1 seconds), the following terms are accumulated:

**Positive Incentives (Carrots):**
* **Progress Reward:** Rewards closing the gap to the target. It yields `8.0 * (prev_dist - current_dist)`, and scales up to `14.0 * (prev_dist - current_dist)` if the drone is closer than 1.0m to encourage final precision.
* **Direction Alignment:** Encourages facing the velocity vector directly toward the target. `0.20 * dot_product(action, target_direction)`

**Negative Incentives (Sticks - Continuous Penalties):**
* **Distance Penalty:** Constantly drains `-0.06 * current_dist`. Encourages reaching the target as fast as possible to minimize time-based bleed.
* **Sonar Risk Penalties:**
  * **Mean Risk Penalty:** `-2.0 * obstacle_mean_risk^2`
  * **Max Risk Penalty:** `-4.0 * obstacle_max_risk^2` (harshly penalizes the closest approaching obstacle)
  * **Downward Risk Penalty:** `-1.0 * down_sonar_risk^2`
* **Approach Trend Penalty:** `-1.5 * max_approach_trend`. Punishes actively accelerating into obstacles rather than just static presence near one.
* **Near Target Penalties:** Once within 1.0m, the drone is punished for chaotic movements (`-0.35 * dist`, axis penalties, and high velocity penalties), forcing it to slow down and hover cleanly onto the target point.
* **Control Penalties:**
  * **Action Magnitude:** `-0.01 * norm(action)` penalizes maxing out controls constantly.
  * **Smoothness:** `-0.02 * norm(action - prev_action)` reduces jitter and oscillation.
  * **Filter Penalty:** `-0.25` if the fallback safety filter had to intervene to prevent a crash.

---

## 3. Curriculum Training Stages (`train.py`)

Because attempting to fly perfectly to a distant target while avoiding obstacles is incredibly hard to learn from scratch, `train.py` utilizes a **CurriculumTargetWrapper**. This exposes the exact same RL environment to PPO but dynamically adjusts the spawned target's difficulty.

* **Stage 1 (Easy & Fixed):** Target is very close to origin at low altitude `(1.0, 0.0, 0.8)`. The agent learns the fundamental relationship between X/Y/Z velocity, the target coordinate observation, and the progress reward.
* **Stage 2 (Randomized Open Space):** Target spawns randomly within bounding box `X: [1.0, 3.0], Y: [-1.5, 1.5], Z: [0.8, 1.6]`. The agent learns robust 3D navigation in free space.
* **Stage 3 (Single Path Obstacle):** Target placed at `(5.0, 3.0, 2.0)` near the direct path through obstacles. Agent starts learning sonar correlations.
* **Stage 4 (Final Task D Configuration):** Final evaluation mode `(6.0, -3.0, 2.2)`. Fully randomized and requires threading through the Gazebo playground environment.

### 3.1 PPO Hyperparameters (from `train.py`)
* `learning_rate`: `3e-4`
* `n_steps`: `512` (horizon for policy update calculation)
* `batch_size`: `64`
* `gamma`: `0.99` (discount factor prioritizing future rewards)
* Default total timesteps: `500,000` (allowing adequate exploration for dense network convergence).
