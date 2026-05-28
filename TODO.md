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

## Active Now: Side-Sonar Task D Improvements

- [ ] Add left and right side sonar sensors to the drone xacro/URDF:
  `/simple_drone/side_sonar_left/out`
  `/simple_drone/side_sonar_right/out`
- [ ] Rebuild the ROS workspace inside Docker after sensor model edits:
  `colcon build --symlink-install --packages-select nsysu_drone_description nsysu_drone_bringup nsysu_drone_control`
- [ ] Verify the expected 8 sonar topics:
  `/simple_drone/sonar/out`
  `/simple_drone/front_sonar_left/out`
  `/simple_drone/front_sonar_center/out`
  `/simple_drone/front_sonar_right/out`
  `/simple_drone/front_sonar_up/out`
  `/simple_drone/front_sonar_down/out`
  `/simple_drone/side_sonar_left/out`
  `/simple_drone/side_sonar_right/out`
- [ ] Echo the two new side sonar topics once:
  `ros2 topic echo --once /simple_drone/side_sonar_left/out`
  `ros2 topic echo --once /simple_drone/side_sonar_right/out`
- [ ] Update environment subscriptions for the two side sonar topics.
- [ ] Expand the observation vector from 35 to 43 values with 7 obstacle-facing sonar sectors.
- [ ] Keep the continuous action space fixed as `[vx_cmd, vy_cmd, vz_cmd]`.
- [ ] Update reward terms so obstacle risk includes front and side sonar sectors.
- [ ] Add side-sonar unsafe termination for left/right wall proximity.
- [ ] Update the emergency safety filter so side sonar only prevents obvious side-wall collisions.
- [ ] Add 4 curriculum stages while keeping one environment class, one observation shape, and one action shape.
- [ ] Add evaluation metrics: success rate, crash rate, timeout rate, average return, average minimum obstacle sonar distance, average steps, safety filter activation count, and side near-miss count.
- [ ] Document validation commands for side sonar topics, smoke tests, staged training, and evaluation.
- [ ] Document report rationale for side sonar, processed sonar features, emergency filtering, frozen spaces, and curriculum training.

## Part 2 RL Training

- [ ] Reinstall RL dependencies if the container was recreated:
  `python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas`
- [ ] Delete or ignore the old `ppo_drone.zip` because the observation shape changed to 43.
- [ ] Run smoke training:
  `cd /workspace/HW2_Work/part2 && python3 train.py --smoke --stage 1`
- [ ] Confirm `models/ppo_drone_stage1.zip` and a training curve are created.
- [ ] Run deterministic evaluation:
  `python3 test.py --episodes 3 --csv logs/eval_metrics.csv`
- [ ] If smoke/eval works, train stages 1-4 in order.
- [ ] Record status, final distance, minimum obstacle sonar, minimum down sonar, side near-misses, safety filter count, and total reward.

## Report Evidence

- [ ] Capture screenshot of Gazebo/RViz with the eight sonar cones visible.
- [ ] Capture `ros2 topic list | grep sonar` output showing all eight topics.
- [ ] Capture example `ros2 topic echo --once` output for the two side sonar topics.
- [ ] Save final training curves from `HW2_Work/part2/logs/`.
- [ ] Save evaluation CSV from `HW2_Work/part2/logs/eval_metrics.csv`.
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
