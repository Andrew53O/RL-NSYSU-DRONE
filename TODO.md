# HW2 Fast-Finish TODO

Priority: get the Part 2 RL pipeline working first. Part 1 evidence is still required for the final submission, but defer screenshots/logs until the Part 2 code can train and test.

## Docker/mount setup

- [x] Confirm `run_docker.sh` launches `nsysu_drone_vnc:iron`.
- [x] Mount host `HW2_Work` into Docker at `/workspace/HW2_Work`.
- [ ] Start container from host with `GPU_ID=0 ./run_docker.sh`.
- [ ] Inside container, install RL dependencies once:
  `python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas`
- [ ] Verify `/workspace/HW2_Work` shows the host files.

## Repository inspection

- [x] Read `AGENTS.md`.
- [x] Inspect assignment PDFs/Markdown references.
- [x] Inspect `README.md`.
- [x] Inspect `fly_straight.py` and `rl_fly_to_target.py`.
- [x] Inspect launch files, `drone.yaml`, URDF/Xacro, and world file.
- [x] Confirm original sonar topic is `/simple_drone/sonar/out`.
- [ ] Confirm added front sonar topics after ROS rebuild.

## Part 2 environment implementation

- [x] Create `HW2_Work/part2/drone_env.py`.
- [x] Implement `DroneSonarAvoidEnv`.
- [x] Subscribe to pose, velocity, and sonar topics.
- [x] Publish reset, takeoff, and velocity commands.
- [x] Include processed sonar features in the observation vector.
- [x] Implement shaped reward and explicit termination conditions.
- [ ] Run a simulator smoke test.

## Training script

- [x] Create `HW2_Work/part2/train.py`.
- [x] Train Stable-Baselines3 PPO.
- [x] Save model to `HW2_Work/part2/models/ppo_drone.zip`.
- [x] Save logs under `HW2_Work/part2/logs/`.
- [x] Add a `--smoke` mode for short test runs.
- [ ] Run smoke training inside Docker with Gazebo running.
- [ ] Run longer training if smoke training works.

## Test script

- [x] Create `HW2_Work/part2/test.py`.
- [x] Load trained PPO model.
- [x] Reset, take off, and run deterministic policy in Gazebo.
- [x] Print status, final distance, minimum sonar range, and total reward.
- [ ] Run test inside Docker with Gazebo running.

## Training curve generation

- [x] Generate `HW2_Work/part2/logs/training_curve.png` from Monitor logs.
- [ ] Confirm curve is created after smoke training.
- [ ] Use the final training curve in the report.

## Minimal report/README support

- [x] Create `HW2_Work/README.md`.
- [x] Document environment setup, train/test commands, outputs, and sonar limitation.
- [ ] Add final model result numbers after testing.
- [ ] Use `Homework-files/papers/paper_metadata.md` for report citations.

## Deferred Part 1 evidence

- [ ] Launch Gazebo/RViz and capture at least one screenshot.
- [ ] Run `fly_straight.py` to default target and save terminal log.
- [ ] Test at least 3 alternative targets.
- [ ] Write Appendix A notes on `Kp` and `max_speed`.
