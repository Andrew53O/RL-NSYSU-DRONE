# HW2 Progress TODO

Priority: finish **Part 2, Task D - sonar obstacle avoidance** first. Part 1 evidence is deferred until the Part 2 training/test pipeline is working, but it is still needed for the final submission if time allows.

## Completed

- [x] Read `AGENTS.md` and assignment reference files.
- [x] Digitalized the assignment PDFs into Markdown for easier reading.
- [x] Collected Task D literature papers and notes in `Homework-files/papers/`.
- [x] Set up Git/fork workflow and checkpoint commits.
- [x] Confirmed Docker launches `nsysu_drone_vnc:iron`.
- [x] Mounted `HW2_Work` into Docker at `/workspace/HW2_Work`.
- [x] Mounted ROS source packages into `/ros2_ws/src` so host edits sync automatically.
- [x] Created `HW2_Work/part2/drone_env.py`.
- [x] Created `HW2_Work/part2/train.py`.
- [x] Created `HW2_Work/part2/test.py`.
- [x] Created `HW2_Work/README.md`.
- [x] Created `REPORT.md` with obstacle curriculum and sonar design notes.
- [x] Implemented PPO with continuous action `[vx_cmd, vy_cmd, vz_cmd]`.
- [x] Implemented processed sonar state, short-term sonar memory, reward shaping, and safety termination.
- [x] Confirmed the earlier 4-sonar layout was visible in Gazebo.
- [x] Upgraded the design to 6 total sonar sensors: 1 downward safety sonar plus 5 front sectors.

## Active Now: Six-Sonar Runtime Verification

- [ ] Restart container with the updated `run_docker.sh`.
- [ ] Rebuild ROS workspace inside Docker:
  `colcon build --symlink-install --packages-select nsysu_drone_description nsysu_drone_bringup nsysu_drone_control`
- [ ] Source the rebuilt workspace:
  `source /ros2_ws/install/setup.bash`
- [ ] Relaunch simulator with `launch_drone`.
- [ ] Confirm the expected 6 sonar topics:
  `/simple_drone/sonar/out`
  `/simple_drone/front_sonar_left/out`
  `/simple_drone/front_sonar_center/out`
  `/simple_drone/front_sonar_right/out`
  `/simple_drone/front_sonar_up/out`
  `/simple_drone/front_sonar_down/out`
- [ ] Echo each sonar topic once with `ros2 topic echo --once <topic>`.
- [ ] Visually confirm the extra front/up/down range cones in Gazebo or RViz Range displays.
- [ ] Move near a wall/cone and verify the front sonar ranges react.

## Part 2 RL Training

- [ ] Reinstall RL dependencies if the container was recreated:
  `python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas`
- [ ] Delete or ignore the old `ppo_drone.zip` because the observation shape changed.
- [ ] Run smoke training:
  `cd /workspace/HW2_Work/part2 && python3 train.py --smoke`
- [ ] Confirm `models/ppo_drone.zip` and `logs/training_curve.png` are created.
- [ ] Run one deterministic test:
  `python3 test.py`
- [ ] If smoke/test works, run longer training:
  `python3 train.py --timesteps 50000`
- [ ] Record status, final distance, minimum front sonar, minimum down sonar, and total reward.

## Report Evidence

- [ ] Capture screenshot of Gazebo/RViz with the six sonar cones visible.
- [ ] Capture `ros2 topic list | grep sonar` output showing all six topics.
- [ ] Capture example `ros2 topic echo --once` output for front center/up/down sonar.
- [ ] Save final training curve from `HW2_Work/part2/logs/training_curve.png`.
- [ ] Save test output after the final trained model.
- [ ] Explain the MDP: state, action, reward, termination, and PPO setup.
- [ ] Cite the literature papers for processed sonar state, short-term memory, risk tracking, and safety filtering.
- [ ] Describe limitations: sonar is coarse, sector-based, and not camera/LiDAR.

## Deferred Part 1 Evidence

- [ ] Launch Gazebo/RViz and capture the required Part 1 screenshot.
- [ ] Run `fly_straight.py` to the default target and save terminal output.
- [ ] Test at least 3 alternative targets if time allows.
- [ ] Write short notes comparing classical proportional control against learned PPO behavior.
- [ ] Add Appendix A notes on `Kp` and `max_speed` if needed.

## Final Packaging

- [ ] Confirm `HW2_Work/README.md` has correct run commands.
- [ ] Confirm model, logs, and report images are present.
- [ ] Check `git status` for accidental unrelated files.
- [ ] Make final Git commit.
- [ ] Push to the fork when ready.
