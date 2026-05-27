# AGENTS.md

## Project Identity

This repository is for Reinforcement Learning Homework 2: NSYSU Drone Control with Reinforcement Learning.

The assignment uses ROS 2 + Gazebo Classic to formulate drone control as a Markov Decision Process, train a reinforcement learning agent, and write a reflection report comparing learned control with classical control.

The main implementation goal for this project is:

**Task D — Autonomous Obstacle Avoidance using sonar.**

Use sonar instead of camera because the deadline is short and camera-based RL would require much more implementation, training, and debugging time.

## Important Reference Files

The following assignment/reference files will be placed in this repository so Codex can read them:

```text
Homework-files/HW2_Assignment_Intro (1).pdf
Homework-files/HW2_NSYSU_Drone_RL_EN.pdf
Homework-files/Github-boilercode-README.txt


## Git Workflow

After making meaningful changes, help me create clear Git checkpoints.

For every prompt I give, consider whether the work should be committed. If the prompt causes multiple logically separate changes, create multiple Git commits instead of one large commit.

For each checkpoint:

1. Run `git status` first.
2. Review the changed files.
3. Use `git add` only for the files related to that checkpoint.
4. Write a clear commit message that explains the purpose of the change.
5. Commit the change.
6. Push the commit to the remote repository.

Use concise commit messages, preferably in this style:

```text
type: short description