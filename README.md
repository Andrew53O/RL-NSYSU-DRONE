# NSYSU Drone RL Homework 2

This repository contains the ROS 2 + Gazebo Classic drone simulator and the Reinforcement Learning Homework 2 implementation for **Task D: Autonomous Obstacle Avoidance using sonar**.

The original simulator README is preserved at [docs/Original-README.md](docs/Original-README.md).

## Project Goal

The homework trains a PPO policy to control the drone with velocity commands. The final task is to fly toward a target near `(10, 0, 1)` while using sonar readings to avoid obstacles in Gazebo.

The current implementation uses sonar instead of camera input because sonar keeps the observation space compact and is more practical for the assignment deadline.

## Repository Layout

```text
.
├── HW2_Work/part3/                 # Main working RL implementation
│   ├── drone_env.py                # Gymnasium/ROS environment
│   ├── train.py                    # PPO training script
│   ├── test.py                     # Deterministic evaluation script
│   ├── models/                     # Saved PPO models
│   └── logs/                       # Training and evaluation logs
├── HW2_B113040056_洪理川/          # Final submission package copy
├── Homework-files/                 # Assignment PDFs and literature notes
├── docs/                           # Supporting runbooks and design notes
├── nsysu_drone_bringup/            # ROS launch/config package
├── nsysu_drone_control/            # ROS control and teleop package
├── nsysu_drone_description/        # URDF, plugins, models, worlds
├── REPORT.md                       # Final homework report
├── Dockerfile
└── run_docker.sh
```

## Important Files

| File | Purpose |
| --- | --- |
| [REPORT.md](REPORT.md) | Final report for the homework |
| [HW2_Work/part3/README.md](HW2_Work/part3/README.md) | Part 3 implementation notes |
| [docs/commands.md](docs/commands.md) | Docker, Gazebo, training, and testing commands |
| [docs/training-design.md](docs/training-design.md) | Curriculum and PPO training notes |
| [docs/rl-design.md](docs/rl-design.md) | MDP, observation, action, reward, and safety design |
| [docs/literature-review-draft.md](docs/literature-review-draft.md) | Draft literature review notes |
| [docs/Original-README.md](docs/Original-README.md) | Original simulator README/instructions |

## Curriculum Summary

The Part 3 PPO curriculum trains simple navigation skills before obstacle avoidance:

| Stage | Variant | Task |
| --- | --- | --- |
| 1 | A | Fixed vertical target `(0, 0, 1.2)` |
| 1 | B | Random vertical target, `z in [0.7, 1.8]` |
| 2 | A | Fixed horizontal target `(1, 0, 0.8)` |
| 2 | B | Random horizontal target, `x in [-1, 2]` |
| 3 | A | Single random x-z target |
| 3 | B | Three sequential random x-z targets |
| 4 | A | One-obstacle sonar avoidance toward `(10, 0, 1)` |
| 5 | A | Multi-obstacle sonar avoidance toward `(10, 0, 1)` |

Stages 1-3 keep sonar masked to safe constant values. Stages 4-5 activate sonar observations and sonar safety rewards.

## Quick Start

Start the Docker container from the host:

```bash
cd ~/HW2/nsysu_drone
GPU_ID=0 ./run_docker.sh
```

Open a second shell inside the running container:

```bash
docker exec -it nsysu_drone_vnc bash
source /ros2_ws/install/setup.bash
```

Launch the Stage 4 one-obstacle Gazebo world:

```bash
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/stage4_obstacle.world
```

In the second container shell, evaluate the trained Stage 4 policy:

```bash
cd /workspace/HW2_Work/part3
python3 test.py \
  --stage 4 \
  --model models/stage4/run004/best/best_precision_model.zip \
  --success-distance 0.25 \
  --max-steps 1800 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

The script prints summary metrics and saves evaluation CSV files under:

```text
HW2_Work/part3/logs/eval/
```

## Training

Training uses Stable-Baselines3 PPO with a continuous velocity action:

```text
[vx_cmd, vy_cmd, vz_cmd]
```

Typical settings:

```text
policy = MlpPolicy
learning_rate = 0.0003
n_steps = 512
batch_size = 64
gamma = 0.99
step_dt = 0.05
device = cpu
```

Detailed training commands are in [docs/training-design.md](docs/training-design.md) and [docs/commands.md](docs/commands.md).

## Current Results

The final report focuses on Stage 4 because it is the main Task D demonstration.

| Stage | Result |
| --- | --- |
| 1A | 100% success |
| 1B | 100% success |
| 2A | 100% success |
| 2B | 100% success |
| 3A | 100% success |
| 3B | 90% success |
| 4 | 80% success, 20% unsafe sonar |
| 5 | Multi-obstacle extension experiment |

See [REPORT.md](REPORT.md) for the full discussion, failure cases, and comparison with classical control.

## Simulator Notes

The ROS/Gazebo simulator is based on the original NSYSU drone package. For the full original Docker, VNC, topic, plugin, native install, and troubleshooting instructions, read:

```text
docs/Original-README.md
```

The most important runtime detail is that GUI ROS commands should use `vglrun` inside the Docker/VNC environment so Gazebo and RViz use GPU rendering.
