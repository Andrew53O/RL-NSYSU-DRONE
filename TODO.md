# HW2 Current TODO

The active implementation is `HW2_Work/part3`. The report focus is **Task D: sonar obstacle avoidance**.

## Completed

- [x] Created clean Part 3 PPO curriculum.
- [x] Kept fixed action space `[vx_cmd, vy_cmd, vz_cmd]`.
- [x] Kept fixed 41-value observation across stages.
- [x] Masked sonar for Stages 1-3.
- [x] Enabled sonar observations and sonar risk reward from Stage 4.
- [x] Trained/evaluated Stage 1A, 1B, 2A, 2B, 3A, and 3B.
- [x] Prepared and evaluated Stage 4 one-obstacle world.
- [x] Prepared Stage 5 multi-obstacle world.
- [x] Rewrote report and design documentation for the current Part 3 work.

## Active Next Steps

- [ ] Continue Stage 5 training from `models/stage4/run004/best/best_precision_model.zip`.
- [ ] Evaluate Stage 5 with `success-distance 0.25`.
- [ ] Record Stage 5 success rate, unsafe sonar rate, mission distance, and minimum sonar range.
- [ ] Capture Gazebo screenshots for Stage 4 or Stage 5.
- [ ] Add final training/evaluation screenshots to the report if time allows.
- [ ] Decide whether Stage 5 is strong enough to include as a main result or only as an extension.

## Final Submission Checks

- [ ] Confirm `REPORT.md` is the final report.
- [ ] Confirm `SECTION3-LITERATURE-REVIEW.md` matches the report references.
- [ ] Confirm `COMMAND.md` has the commands needed to reproduce Stage 4/5.
- [ ] Confirm accidental model/log folders are not committed unless intentionally required.
- [ ] Make final Git checkpoint.
