# HW2 Work

This folder contains all the homework parts for NSYSU Drone RL HW2.

## Folder Map

| Folder | Meaning |
| --- | --- |
| `part1/` | Homework Part 1 |
| `part2/` | Failed attempt. It uses a 60+ observation space and did not work well. |
| `part3/` | The submitted homework for Tronclass |
| `part4/` | Failed fork of `part3/`. I copied `part3` and changed `drone_env.py` to make the flight less wavy. |
| `partX_final/` | The last try for the final presentation. This is the best case in this repo. |

## Docker

Start the Docker container from the host:

```bash
cd ~/HW2/nsysu_drone
GPU_ID=0 ./run_docker.sh
```

Open another terminal in the same container:

```bash
docker exec -it nsysu_drone_vnc bash
source /ros2_ws/install/setup.bash
```

If you change ROS package files, launch files, URDF/Xacro, or world files, rebuild inside Docker:

```bash
cd /ros2_ws
colcon build --symlink-install --packages-select nsysu_drone_description nsysu_drone_bringup nsysu_drone_control
source install/setup.bash
```

Python files under `HW2_Work/partX_final` do not need a ROS rebuild.

## How To Run

The repo is split by part, so the safest way is to run each part from its own folder.

For `partX_final`, start Gazebo first and then run training or testing from inside the container:

```bash
ros2 launch nsysu_drone_description launch_drone.py \
  world:=/ros2_ws/src/nsysu_drone_description/worlds/playground.world
```

Then run the scripts:

```bash
cd /workspace/HW2_Work/partX_final
python3 train.py --stage 1 --variant A --smoke
python3 test.py --stage 5 --variant A --model models/stage5/variantA/run005/best/best_average_model.zip
```

The detailed train/test commands for each stage are written in `HW2_Work/partX_final/README.md`.

## partX_final Result

`partX_final` is the best final fork in this repo. It keeps the full Stage 1 to Stage 5 curriculum and is the version I used for the final presentation attempt.

Latest evaluation notes from the saved logs:

- `Stage 5A`: 10/10 success in the latest eval, with no unsafe or timeout episodes.
- `Stage 5B`: 3/10 success in the latest eval, with the corridor task still unstable.

So the final folder is the strongest version overall, but `Stage 5B` still needs more tuning if I want a stable multi-obstacle demo.

## Main Files

```text
HW2_Work/part1/
HW2_Work/part2/
HW2_Work/part3/
HW2_Work/part4/
HW2_Work/partX_final/
```
