# Reinforcement Learning Homework 2: NSYSU Drone Control

Digitized from `HW2_NSYSU_Drone_RL_EN.pdf`.

Course: Reinforcement Learning, Spring 2026  
Assignment: Homework 2, NSYSU Drone Control  
Department: Computer Science and Engineering, NSYSU  
Due: 2026/05/30, Saturday, 23:59  
Total points: 100  
Submission: Upload ZIP files to eLearn  
Work policy: Individual assignment. Discussion is allowed, but code and report must be your own work.

## 1. Learning Objectives

In this assignment, you will use the NSYSU Drone simulation environment, ROS 2, and Gazebo Classic to learn how reinforcement learning can be applied to a continuous-control problem.

You are expected to:

- Become familiar with ROS 2 publisher/subscriber mechanisms and the Gazebo physics simulation workflow.
- Understand trade-offs between PID control and RL-based control in continuous action spaces.
- Formulate a drone control task as a Markov Decision Process: state, action, reward, and discount factor.
- Train an RL agent using a modern library and analyze its training curves.
- Communicate your motivation, literature review, and experimental results in a formal report.

## 2. Environment

This assignment uses the NSYSU Drone simulation environment, a ROS 2 port derived from `tum_simulator`.

It includes:

- Quadrotor model.
- PID control plugin, `plugin_drone`.
- Sensors: IMU, GPS, sonar, and camera.

### 2.1 Recommended Setup

- Host: Ubuntu 22.04, Docker 24.x.
- If you do not have an NVIDIA GPU, you may use lightweight headless mode with `gzserver` only for training. The Gazebo GUI will not be available in that mode.
- ROS 2 Iron is recommended. Humble and Rolling are also supported.

### 2.2 Key ROS 2 Topics

| Topic | Message type | Purpose |
| --- | --- | --- |
| `/simple_drone/cmd_vel` | `geometry_msgs/Twist` | Velocity command, action output |
| `/simple_drone/gt_pose` | `geometry_msgs/Pose` | Ground-truth pose, observation |
| `/simple_drone/gt_vel` | `geometry_msgs/Twist` | Ground-truth velocity |
| `/simple_drone/takeoff` | `std_msgs/Empty` | Take off, required after each reset |
| `/simple_drone/reset` | `std_msgs/Empty` | Reset drone pose at end of episode |
| `/simple_drone/sonar/out` | `sensor_msgs/Range` | Downward range sensor, useful for obstacle avoidance |

## 3. Part 1: Environment Setup And `fly_straight.py`

Value: 20 points

The goal of Part 1 is to confirm that you can run the simulator end to end and to give you intuition for the simple position-to-velocity control loop. It is a prerequisite for Part 2.

### 3.1 Steps

1. Follow the Quick Start section of `README.md` to build the Docker image and launch the container with `run_docker.sh`. Native installation is acceptable, but the TAs will not debug native installs.
2. Use `launch_drone` to start Gazebo and RViz. Confirm that the drone model is visible.
3. Open a second terminal and publish a takeoff message so the drone lifts off.
4. Run `fly_straight.py` and confirm that the drone flies from the origin to the default target `(5.0, 3.0, 2.0)`.
5. Modify the target point at least three times and record the flight behavior for each case.

Suggested target points:

- `(-2, 4, 1.5)`
- `(0, 0, 3)`
- `(6, -3, 2)`

### 3.2 What To Submit

- At least one Gazebo screenshot that clearly shows the drone and its target location.
- A terminal log of a successful `fly_straight.py` run. This can be a screenshot or `.txt` file.
- A short write-up, about half a page, describing how the behavior changes when you increase or decrease `Kp` or the maximum speed. Include this as Appendix A of your Part 2 report.

## 4. Part 2: Applying RL Algorithms

Value: 75 points

Part 2 is the main assignment component. You must define your own RL task in the drone environment, implement and train an agent, evaluate it, and document everything in a report.

The provided `rl_fly_to_target.py` is only a minimal reference. A direct copy with no meaningful extension will not receive credit.

### 4.1 Task Options

You may choose one of the suggested tasks below or propose your own.

#### Task A: Precision Hovering And Disturbance Rejection

Train the drone to hover at a given `(x, y, z)` pose and maintain stability when `motionDriftNoise` is non-zero.

#### Task B: Random Target Navigation

Generate a different target point in each episode and train a policy that generalizes across targets. This is an extension of `rl_fly_to_target.py`.

#### Task C: Trajectory Tracking

Train the agent to follow a time-varying reference trajectory such as:

- Figure-eight.
- Circle.
- Helix.

#### Task D: Autonomous Obstacle Avoidance

Add obstacles to the Gazebo world and use sonar or camera inputs so the agent can navigate around them toward a goal.

#### Task E: Multi-Waypoint Cruising

Given an ordered list of waypoints, train the agent to visit them in sequence while minimizing total time.

### 4.2 Implementation Requirements

Regardless of the task, your code and model must satisfy the following requirements:

1. Wrap the environment with the Gymnasium interface. Inherit from `gym.Env`, implement `reset` and `step`, and define `observation_space` and `action_space`.
2. Use at least one RL algorithm. Recommended algorithms are PPO, SAC, TD3, and DDPG. If you use a custom algorithm, explain it in the report.
3. Use a well-justified reward function. Explain the meaning and weight of each term in the report.
4. Define explicit episode termination conditions, such as target reached, timeout, crash, or out of bounds.
5. Produce a training curve, such as reward vs. episode or timestep, and save it as an image included in the report.
6. Provide a test script that loads your trained model and demonstrates the agent behavior in Gazebo when run by the TAs.

### 4.3 Reflection Report

Value: 30 points  
Length: 4-8 pages  
Format: PDF  
Suggested formatting: A4, 12 pt font, 1.5 line spacing  
Tools: LaTeX or Word are both acceptable.

Required sections:

| Section | Required content | Points |
| --- | --- | ---: |
| Task Definition and Motivation | Clearly describe the task you want the RL agent to learn and explain why it is worth studying | 5 |
| Pain Points of Existing Methods | Analyze limitations of PID, LQR, MPC, or handcrafted rules on your chosen task and explain why RL is plausible | 5 |
| Literature Review | At least 3 papers from the past 5 years, preferably PPO, SAC, TD3, DDPG, or UAV RL applications; summarize method, experiments, and design influence | 8 |
| Proposed Solution | Choice of algorithm and rationale, MDP formulation `(S, A, R, gamma)`, network architecture, hyperparameters, and reward rationale | 7 |
| Results and Discussion | Training curves, test-time success rate, failure-case analysis, and comparison with a baseline such as the P controller in `fly_straight.py` | 5 |

Writing tips:

- Do not make vague claims about motivation or pain points. Point out specifically why PID struggles in your chosen task and give an example.
- A literature review is not a summary of abstracts. For each paper, explain what it inspired in your own design.
- Submissions without failure-case analysis will lose points. Honest failure analysis is more valuable than cherry-picked success curves.
- All figures must be produced by you with proper figure numbers and captions.
- Copying figures from the internet is considered plagiarism.

## 5. Submission Guidelines

### 5.1 Required Files

Package everything in a single ZIP archive:

```text
HW2_StudentID_Name.zip
```

Example:

```text
HW2_M12345678_Wang_Xiaoming.zip
```

The archive must contain:

- Source code, at least the Gym environment, training script, and test script for Part 2.
- Trained model file. `.zip` for Stable-Baselines3 or `.pth` for PyTorch is recommended. The filename should correspond to your training logs.
- Reflection report as a PDF named `HW2_StudentID_report.pdf`.
- Part 1 screenshots and logs in a subfolder named `part1/`.
- `README.md` describing the execution environment, installation commands, how to reproduce your training run, and how to load and test the trained model.

### 5.2 Suggested Folder Layout

```text
HW2_StudentID_Name/
|-- README.md                # How to run
|-- part1/
|   |-- screenshot_01.png
|   |-- screenshot_02.png
|   +-- terminal_log.txt
|-- part2/
|   |-- drone_env.py         # Gym environment
|   |-- train.py             # Training script
|   |-- test.py              # Test script
|   |-- models/
|   |   +-- ppo_drone.zip    # Trained model
|   +-- logs/
|       +-- training_curve.png
+-- HW2_StudentID_report.pdf
```

### 5.3 Deadline

Submissions are due by May 30, 2026, Saturday, 23:59.

No delay allowed.

### 5.4 Academic Integrity

- Copying another student's code or report is strictly prohibited. If substantial similarity is detected, both parties receive 0 for this assignment.
- AI tools such as ChatGPT and Claude may be used as assistants, but you must disclose how they were used in an Acknowledgement section at the end of the report. Examples: debugging help, syntax suggestions, brainstorming reward designs.
- Reports that are entirely AI-generated without personal understanding will be penalized.
- Citations must be accurate.
- Copying paragraphs from published papers is considered plagiarism.

## 6. Grading

The assignment is worth 100 points, with up to 5 additional bonus points.

| Item | Description | Points | Deliverable |
| --- | --- | ---: | --- |
| Part 1 | Set up the environment and successfully run `fly_straight.py`; provide screenshots, terminal logs, and Gazebo views | 20 | Screenshots and log |
| Part 2 Code | RL implementation: Gym environment, reward design, training script; evaluated on quality, readability, and reproducibility | 30 | `.py` source files |
| Part 2 Model | Trained model file that passes the TA test script and runs in the simulator without version conflicts | 15 | `.zip` or `.pth` |
| Part 2 Report | Reflection report covering task definition, motivation, pain points, literature review, proposed solution, and discussion | 30 | PDF |
| Bonus | Extra contributions such as multi-target navigation, obstacle avoidance, trajectory tracking, sim-to-real discussion, or quantitative comparison with PID/open-loop baselines | 5 | Included in report |

Total: 100 points, plus 5 bonus.

### 6.1 Part 1 Rubric

| Requirement | Points |
| --- | ---: |
| Environment is set up successfully and demonstrated | 10 |
| At least three alternative target points are tested and documented with screenshots | 6 |
| Written observations on the effect of `Kp` and `max_speed`, included as Appendix A | 4 |

### 6.2 Part 2 Code Rubric

| Requirement | Points |
| --- | ---: |
| Gym environment is correctly structured: `reset`, `step`, and space definitions are reasonable | 8 |
| Reward design is well reasoned and aligned with the chosen task | 6 |
| Training script runs out of the box and provides checkpointing and logging | 6 |
| Test script loads the model and demonstrates agent behavior | 5 |
| Code readability and quality of comments | 5 |

### 6.3 Part 2 Model Rubric

| Requirement | Points |
| --- | ---: |
| Model file loads cleanly without version conflicts | 5 |
| Model achieves a reasonable success rate on the fixed test scenario defined for your task | 10 |

### 6.4 Part 2 Report Rubric

Section-level allocation is given in section 4.3. Overall presentation, including layout, clarity of writing, and figure quality, may adjust the score by up to plus or minus 2 points.

## 7. Resources

### 7.1 Tools And Environment

- NSYSU Drone `README.md` and the `plugin_drone` / `pid_controller` source files.
- ROS 2 documentation: <https://docs.ros.org/en/iron/>
- Gazebo Classic tutorials: <https://classic.gazebosim.org/tutorials>
- Stable-Baselines3 documentation: <https://stable-baselines3.readthedocs.io/>
- Gymnasium documentation: <https://gymnasium.farama.org/>

## 8. Getting Help

If you run into problems, check the Troubleshooting section of `README.md`.

Good luck, and may your reward curves only go up.
