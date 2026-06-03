# Team Presentation Comparison Notes

Use `team_reward_curves_normalized.svg` for the required side-by-side reward curve. The axes are comparable:

- x-axis: normalized training progress, from 0% to 100%
- y-axis: normalized moving-average reward, from 0 to 1

This avoids misleading raw-reward comparison because the two tasks use different reward scales.

## Member Results

| Member | Task | Algorithm | Network | Training budget | Success / completion | Convergence speed |
| --- | --- | --- | --- | ---: | ---: | --- |
| 洪理川 | Task D: sonar obstacle avoidance, Stage 4 | PPO | `MlpPolicy` MLP actor-critic | 120,000 timesteps | 8/10 eval episodes = 80% | first success at episode 17, about 3,616 timesteps; 90% of final moving-average improvement by episode 134 |
| 王宸澤 | Trajectory tracking, full Step 4 path | PPO | `MlpPolicy` MLP actor-critic | 350,000 timesteps | proxy from training log: 15/24 positive non-short episodes = 62.5% | 90% of best 5-episode moving-average reward by episode 9 |

Important caveat: 王宸澤's available CSV contains `episode`, `reward`, `length`, and `reward_per_step`, but no explicit evaluation `status` column. The 62.5% value is therefore a training-log completion proxy, not a formal evaluation success rate. For a stricter presentation result, run 10 test episodes and record whether `info["Success"]` is true.

## RL Design Comparison

| Design item | 洪理川: sonar obstacle avoidance | 王宸澤: trajectory tracking |
| --- | --- | --- |
| MDP objective | Reach the mission goal `(10, 0, 1)` while avoiding an obstacle detected by sonar. | Follow a sequence of trajectory waypoints loaded from `trajectory1_noFace.txt` or `trajectory2_noFace_new.txt`. |
| Curriculum | Six-stage curriculum: fixed vertical, random vertical, fixed horizontal, random horizontal, random/sequence x-z navigation, then sonar obstacle stages. Stage 4 is the reported result; Stage 5 is a harder multi-obstacle extension. | Four-step curriculum: Step 1 single point, Step 2 random point / simpler path, Step 3 shorter trajectory, Step 4 full trajectory. Each step resumes from the previous model. |
| Observation | 41 values: normalized pose, velocity, relative target, distance, target progress, target count, 7 sonar ranges, 7 sonar risks, 7 previous sonar ranges, 7 sonar trends, sonar-enabled flag. Sonar fields are masked before Stage 4. | 15 values: drone pose, drone velocity, vector to current target, vector to previous/next stored target, and vector to another stored upcoming target slot. No sonar or image input. |
| Action | Continuous velocity command `[vx, vy, vz]` with bounds `[-1, 1]`, `[-1, 1]`, and `[-0.5, 0.5]`. | Continuous velocity command `[vx, vy, vz]` with bounds `[-1, 1]` for all three axes. |
| Episode success | Stage 4 success is measured by final mission-goal distance below `0.25 m`. | Success is `info["Success"] = True` when all trajectory waypoints are completed. The training CSV does not log this directly. |
| Termination / failure | Invalid sensor, crash below minimum altitude, out-of-bounds, unsafe sonar below `0.25 m`, success, or timeout. | Out-of-bounds, no waypoint update for more than 2000 steps, all waypoints completed, or max-step truncation. |
| Safety design | Uses sonar risk penalties and an action safety filter. The filter slows forward motion when front sonar is close and limits lateral motion when side sonar is close. | No obstacle/safety sensing; the task assumes the reference path is safe and focuses on tracking. |

## Reward Design

| Reward component | 洪理川: obstacle avoidance | 王宸澤: trajectory tracking |
| --- | --- | --- |
| Progress reward | Rewards reduction in distance to the active target. Stage 4 also rewards reduction in final mission-goal distance. | Rewards reduction in distance to the current trajectory waypoint: `100 * (prev_dist - curr_dist)`. |
| Precision penalty | Penalizes remaining distance and axis-specific position error. The axis weights change by curriculum focus: vertical, horizontal, or combined navigation. | Penalizes current waypoint position error: `-0.5 * position_error`. |
| Smoothness penalty | Penalizes action magnitude and change from previous action. Adds stronger braking penalties near the target. | Penalizes action change: `-0.1 * ||last_action - action||`. |
| Safety penalty | Penalizes mean and max sonar risk; terminates with `-100` if sonar distance is unsafe. Also penalizes reliance on the safety filter. | Penalizes out-of-bounds and no-progress failures by `-10`. No obstacle penalty. |
| Completion bonus | `+80` for reaching the final target; `+30` for intermediate targets in sequence stages. | `+10` when a waypoint is reached; `+10` when the full trajectory is completed. |

This reward-design difference is the main reason the curves have different shapes. The obstacle task has sudden large negative returns from unsafe sonar, so the reward curve remains noisy even after the policy learns some successes. The trajectory task has dense waypoint-progress feedback and smaller failure penalties, so early improvement can appear faster and smoother.

## Algorithm, Hyperparameters, And Architecture

| Item | 洪理川 | 王宸澤 |
| --- | ---: | ---: |
| Algorithm | PPO | PPO |
| Policy | `MlpPolicy` | `MlpPolicy` |
| Learning rate | `0.0003` | `0.0003` |
| `n_steps` | `512` | `1024` |
| Batch size | `64` | `128` |
| `gamma` | `0.99` | SB3 PPO default, likely `0.99` |
| Control step | `0.05 s` in reported Stage 4 config | `0.025 s` |
| Max episode length | `1800` steps for reported Stage 4 | `10000` or `20000` steps depending on selected trajectory |

Both members used PPO with an MLP actor-critic policy, so algorithm choice does not explain most of the difference. The larger rollout and batch size in 王宸澤's setup can make updates more stable over long trajectory episodes, while 洪理川's shorter rollout updates more frequently and is more reactive to the obstacle curriculum.

The network architecture is appropriate for both tasks because both use vector observations rather than camera images. 洪理川's MLP must learn from state plus sonar-risk features, so the policy balances goal progress against obstacle safety. 王宸澤's MLP receives target-vector information for the trajectory, so it mainly learns local waypoint-following behavior.

## Analysis Attribution

The difference is not simply that one model is better. The tasks optimize different behaviors:

- Reward design: obstacle avoidance has strong safety penalties and hard unsafe-sonar terminations; trajectory tracking has dense progress and tracking-error shaping.
- Algorithm choice: both use PPO, so algorithm choice is mostly controlled. The comparison mainly reflects task and design differences.
- Hyperparameters: 王宸澤 uses larger `n_steps` and batch size, which suits long trajectories. 洪理川 uses smaller rollouts and batches, which gives more frequent updates for a shorter obstacle episode.
- Network architecture: both use MLPs, but the input features differ. Sonar features make 洪理川's policy safety-aware; target-vector features make 王宸澤's policy path-following-oriented.
- Task difficulty: obstacle avoidance includes hidden danger from sensor thresholds and local obstacle geometry; trajectory tracking follows known reference points and does not need collision reasoning.

## Presentation Script

Our curves are plotted with comparable axes using normalized moving-average reward. Raw rewards are not directly comparable because obstacle avoidance and trajectory tracking use different reward scales. In my sonar obstacle-avoidance task, unsafe sonar events create sudden large penalties, so the curve has sharp drops. In the trajectory-tracking task, the reward is denser because every step measures progress toward the next trajectory waypoint, so improvement can appear earlier.

Both agents use PPO and MLP policies, so the main differences come from reward design, task structure, hyperparameters, and observation design. My obstacle agent observes sonar ranges, risk, memory, and trend features, then must trade off forward progress against safety. 王宸澤's trajectory agent observes pose, velocity, and target vectors, then mainly minimizes tracking error along a known path. This explains why my Stage 4 policy reached 80% evaluation success but still had noisy returns, while the trajectory policy shows faster normalized reward improvement but needs a separate explicit evaluation log for a strict success rate.
