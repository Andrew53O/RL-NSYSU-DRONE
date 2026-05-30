# HW2 Submission

This folder is arranged to match the homework submission format.

```text
HW2_StudentID_Name/
├── README.md
├── part1/
│   ├── README.md
│   ├── Part1-Gazebo-Point1.png
│   ├── Part1-Gazebo-Point2.png
│   ├── Part1-Gazebo-Point3.png
│   └── terminal_log.txt
├── part2/
│   ├── drone_env.py
│   ├── train.py
│   ├── test.py
│   ├── README.md
│   ├── models/
│   ├── logs/
│   └── worlds/
└── HW2_StudentID_report.md
```

Before submitting, rename this folder and ZIP file using your real student ID and name:

```text
HW2_StudentID_Name.zip
```

Also export `HW2_StudentID_report.md` to PDF and name it:

```text
HW2_StudentID_report.pdf
```

## Part 1

The Part 1 files are placeholders. Replace them with your actual screenshots and terminal log:

- `part1/Part1-Gazebo-Point1.png`: target `(-5, -3, 2)`
- `part1/Part1-Gazebo-Point2.png`: target `(7, 0, 5)`
- `part1/Part1-Gazebo-Point3.png`: target `(-1, 5, 5)`
- `part1/terminal_log.txt`

## Part 2 / Final RL Demo

The final RL implementation is in `part2/`. It contains the PPO environment, training script, test script, staged models, logs, and obstacle worlds.

Use the provided ROS 2/Gazebo homework environment. In each terminal, source ROS and the workspace if they are not already sourced:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
```

Recommended TA test:

```bash
cd /workspace/HW2_StudentID_Name/part2
vglrun ros2 launch nsysu_drone_bringup nsysu_drone_bringup.launch.py \
  world:=$(pwd)/worlds/stage4_obstacle.world
```

In another terminal:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
cd /workspace/HW2_StudentID_Name/part2
python3 test.py \
  --stage 4 \
  --model models/ppo_drone.zip \
  --success-distance 0.25 \
  --max-steps 1800 \
  --episodes 10 \
  --step-dt 0.05 \
  --log-position-every 100
```

`models/ppo_drone.zip` is a shortcut copy of the main Stage 4 trained policy.

If the folder was unzipped somewhere other than `/workspace`, replace `/workspace/HW2_StudentID_Name` with the actual path.
