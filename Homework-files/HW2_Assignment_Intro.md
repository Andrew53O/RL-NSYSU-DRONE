# Homework 2 Assignment Intro

Digitized from `HW2_Assignment_Intro (1).pdf`.

Course: CSE727 Reinforcement Learning, Spring 2026  
Assignment: Homework 2, NSYSU Drone Control with Reinforcement Learning  
Department: Computer Science and Engineering, National Sun Yat-sen University  
Due: May 30, 2026, Saturday, 23:59  
Points: 100 points, plus 5 bonus points  
Format: Individual assignment, 4-8 page PDF reflection report  
Repository: <https://github.com/NSYSU-ARL/Assignment2>

## At A Glance

| Item | Requirement |
| --- | --- |
| Total points | 100 + 5 bonus |
| Parts | 2 parts: environment setup and RL |
| Due date | May 30, 2026, 23:59 |
| Report | 4-8 pages PDF |

## Mission

Use ROS 2 and Gazebo to formulate drone control as an MDP, train an RL agent, and write a reflective report comparing it to classical control.

## Learning Objectives

- Get comfortable with ROS 2 publisher/subscriber patterns and the Gazebo physics simulation loop.
- Compare classical PID control against learned policies in continuous action spaces.
- Express the drone task as `(S, A, R, gamma)` and design an informative reward.
- Train a modern RL algorithm such as PPO, SAC, or TD3 and analyze training curves.
- Write a formal report with motivation, literature review, and honest failure analysis.

## Simulator

The NSYSU Drone environment is a ROS 2 port of `tum_simulator`.

It includes:

- Quadrotor model.
- `plugin_drone`, which contains a built-in PID controller.
- Sensors: IMU, GPS, sonar, and camera.

Recommended setup:

- Ubuntu 22.04.
- Docker 24.x.
- ROS 2 Iron. Humble or Rolling are also acceptable.
- Headless `gzserver` if no NVIDIA GPU is available.

The RL loop:

- Agent policy outputs an action as `/simple_drone/cmd_vel`.
- Gazebo drone environment returns state and reward through ROS topics.
- Gazebo Classic handles physics.
- Your code interacts through ROS 2 topics.
- Suggested software stack: ROS 2 Iron, Gazebo Classic, Gymnasium, Stable-Baselines3, PyTorch.

## Key ROS 2 Topics

| Topic | Message type | Purpose |
| --- | --- | --- |
| `/simple_drone/cmd_vel` | `geometry_msgs/Twist` | Velocity command, used as the action output |
| `/simple_drone/gt_pose` | `geometry_msgs/Pose` | Ground-truth pose, main observation |
| `/simple_drone/gt_vel` | `geometry_msgs/Twist` | Ground-truth velocity |
| `/simple_drone/takeoff` | `std_msgs/Empty` | Take off, required after each reset |
| `/simple_drone/reset` | `std_msgs/Empty` | Reset drone pose at end of episode |
| `/simple_drone/sonar/out` | `sensor_msgs/Range` | Downward range sensor, useful for obstacle cues |

Always publish takeoff after each reset. Otherwise, the drone stays on the ground and the RL episode will not make sense.

## Roadmap

### Part 1: Environment Setup And `fly_straight.py`

Value: 20 points

Tasks:

- Build the Docker image and run the VNC viewer.
- Launch Gazebo and RViz with `launch_drone`.
- Publish takeoff, then run `fly_straight.py`.
- Test at least 3 alternative target points.
- Describe the effect of `Kp` and `max_speed` in Appendix A.

### Part 2: Applying RL Algorithms

Value: 75 points

Tasks:

- Pick one task from A-E, or propose your own.
- Wrap the environment as `gymnasium.Env`.
- Train PPO, SAC, TD3, DDPG, or a justified custom algorithm.
- Design and justify the reward function.
- Produce training curves and a test script.
- Write a 4-8 page reflection report as a PDF.

Copying `rl_fly_to_target.py` with no meaningful extension will not receive credit.

## Part 1 Details

### Build And Launch

```bash
docker build -t nsysu_drone_vnc:iron .
./run_docker.sh
```

### Connect Via VNC

Open:

```text
http://localhost:5901
```

Password:

```text
nsysudrone
```

An XFCE desktop should appear.

### Launch The Drone

```bash
launch_drone
```

Gazebo and RViz should open. Confirm that the quadrotor is visible.

### Take Off And Fly

```bash
python3 fly_straight.py
```

The default target is `(5.0, 3.0, 2.0)`.

### Part 1 Deliverables

- Gazebo screenshot or screenshots.
- Terminal log of a successful run.
- Half-page write-up about `Kp` and `max_speed`, included as Appendix A.

## Part 1 Exercise

Change the target point at least 3 times. Suggested targets:

- `(-2, 4, 1.5)`
- `(0, 0, 3)`
- `(6, -3, 2)`

Record the flight behavior for each case.

Evidence checklist:

- Gazebo screenshots clearly showing the drone and target.
- Terminal log of a successful `fly_straight.py` run.
- Observations table for each of the 3 or more targets.
- Half-page write-up on increasing and decreasing `Kp`.
- Half-page write-up on adjusting `max_speed`.

## Part 2 Minimum Requirements

### Gym Interface

Subclass `gym.Env`. Implement:

- `reset`
- `step`
- `observation_space`
- `action_space`

### RL Algorithm

Use at least one RL algorithm:

- PPO
- SAC
- TD3
- DDPG

Custom algorithms are allowed only if explained in the report.

### Reward Function

Explain the meaning and weight of every term in the reward function.

### Termination

Define explicit episode-end conditions, such as:

- Target reached.
- Timeout.
- Crash.
- Out of bounds.

### Training Curves

Save reward vs. episode or timestep as a figure and embed it in the report.

### Test Script

Provide a script that loads the trained model and demonstrates the agent behavior in Gazebo out of the box.

## Part 2 Task Options

### Task A: Precision Hovering

Hover at an `(x, y, z)` point and stay stable under `motionDriftNoise`.

### Task B: Random Target Navigation

Generate a new target each episode and learn a policy that generalizes.

### Task C: Trajectory Tracking

Follow a time-varying reference trajectory, such as:

- Figure-eight.
- Circle.
- Helix.

### Task D: Obstacle Avoidance

Add obstacles and use sonar or camera to navigate around them.

### Task E: Multi-Waypoint Cruising

Visit an ordered waypoint list while minimizing total time.

## Report

Value: 30 points  
Length: 4-8 pages  
Format: PDF

The report has five required sections.

| Section | Required content | Points |
| --- | --- | --- |
| Task Definition and Motivation | Describe the task and why it is worth studying | 5 |
| Pain Points of Existing Methods | Explain concretely where PID, LQR, MPC, or handcrafted rules struggle | 5 |
| Literature Review | At least 3 recent papers, preferably from the last 5 years; explain how each paper inspired the design | 8 |
| Proposed Solution | Algorithm choice, MDP `(S, A, R, gamma)`, network, hyperparameters, reward rationale | 7 |
| Results and Discussion | Training curves, success rate, failure cases, comparison to P-controller baseline | 5 |

Writing tips:

- Be specific about PID weaknesses.
- Explain how each paper shaped your design.
- Honest failure analysis is better than cherry-picked curves.

## Submission

Package everything in one ZIP:

```text
HW2_StudentID_Name.zip
```

Suggested folder layout:

```text
HW2_StudentID_Name/
|-- README.md                         # How to run
|-- part1/
|   |-- screenshot_01.png
|   |-- screenshot_02.png
|   +-- terminal_log.txt
|-- part2/
|   |-- drone_env.py                  # Gym environment
|   |-- train.py                      # Training script
|   |-- test.py                       # For the TAs
|   |-- models/
|   |   +-- ppo_drone.zip             # Trained model
|   +-- logs/
|       +-- training_curve.png
+-- HW2_StudentID_report.pdf          # 4-8 pages
```

Required contents:

- Source code: Gym environment, training script, and test script.
- Trained model: `.zip` for SB3 or `.pth` for PyTorch.
- Report PDF: `HW2_StudentID_report.pdf`.
- Part 1 screenshots and logs.
- `README.md` with environment, installation, reproduction, and testing instructions.

## Grading

| Category | Points |
| --- | ---: |
| Part 1: setup, `fly_straight.py`, logs, screenshots | 20 |
| Part 2 code: Gym env, reward design, training pipeline | 30 |
| Part 2 model: loads cleanly and reaches reasonable success rate | 20 |
| Part 2 report: required sections and presentation | 30 |
| Bonus: baselines, extra task, sim-to-real discussion | 5 |

Total: 100 points, plus 5 bonus.

## Rubric Details

### Part 1

| Requirement | Points |
| --- | ---: |
| Environment set up and demonstrated | 10 |
| At least 3 alternative targets tested and documented | 6 |
| Written notes on `Kp` and `max_speed` in Appendix A | 4 |

### Part 2 Code

| Requirement | Points |
| --- | ---: |
| Gym env `reset`, `step`, and spaces are correct | 8 |
| Reward design is well reasoned | 6 |
| Training script, checkpointing, and logging | 6 |
| Test script loads model and runs | 5 |
| Code readability and comments | 5 |

### Part 2 Model

| Requirement | Points |
| --- | ---: |
| Loads cleanly, no version conflicts | 5 |
| Reasonable success rate on TA test scenario | 15 |

Presentation quality may adjust the report score by plus or minus 2 points.

## Academic Integrity And AI Usage

Do not:

- Copy another student's code or report.
- Paste paragraphs from published papers.
- Submit a report that is entirely AI-generated.
- Copy figures from the internet as your own.
- Resubmit `rl_fly_to_target.py` as your Part 2 solution.

You must:

- Write your own code and report.
- Cite papers accurately using IEEE or APA.
- Produce your own figures with numbers and captions.
- Disclose AI tool usage in an Acknowledgement section.

Discussion with classmates is allowed, but the code and report must be your own work.

## Resources

- NSYSU Drone README and plugins: <https://github.com/NSYSU-ARL/Assignment2>
- ROS 2 Iron docs: <https://docs.ros.org/en/iron/>
- Gazebo Classic tutorials: <https://classic.gazebosim.org/tutorials>
- Stable-Baselines3: <https://stable-baselines3.readthedocs.io/>
- Gymnasium: <https://gymnasium.farama.org>
- Troubleshooting: check the repository `README.md`.

## Deadline

May 30, 2026, Saturday, 23:59.

No delays allowed. Upload the ZIP to eLearn following the required folder structure.
