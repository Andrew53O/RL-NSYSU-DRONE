# Team Presentation Comparison Notes

Use `team_reward_curves_normalized.svg` for the required side-by-side reward curve. The axes are comparable:

- x-axis: normalized training progress, from 0% to 100%
- y-axis: normalized moving-average reward, from 0 to 1

This avoids misleading raw-reward comparison because the two tasks use different reward scales.

## Member Results

| Member | Task | Algorithm | Network | Training budget | Success / completion | Convergence speed |
| --- | --- | --- | --- | ---: | ---: | --- |
| 洪理川 | Sonar obstacle avoidance, Stage 4 | PPO | `MlpPolicy` MLP actor-critic | 120,000 timesteps | 8/10 eval episodes = 80% | first success at episode 17, about 3,616 timesteps; 90% of final moving-average improvement by episode 134 |
| 王宸澤 | Full trajectory tracking, Step 4 | PPO | `MlpPolicy` MLP actor-critic | 350,000 timesteps | proxy from training log: 15/24 positive non-short episodes = 62.5% | 90% of best 5-episode moving-average reward by episode 9 |

Important caveat: 王宸澤's available CSV contains `episode`, `reward`, `length`, and `reward_per_step`, but no explicit evaluation `status` column. The 62.5% value is therefore a training-log completion proxy, not a formal evaluation success rate. For a stricter presentation result, run 10 test episodes and record whether `info["Success"]` is true.

## Hyperparameters

| Member | Learning rate | `n_steps` | Batch size | `gamma` |
| --- | ---: | ---: | ---: | ---: |
| 洪理川 | 0.0003 | 512 | 64 | 0.99 |
| 王宸澤 | 0.0003 | 1024 | 128 | SB3 PPO default if not changed |

## Analysis Attribution

The reward curves differ mainly because the reward designs create different learning signals. The obstacle-avoidance reward includes target progress, distance precision, smoothness, and strong sonar safety penalties. Unsafe sonar events cause abrupt negative returns, so the curve is noisy even after the policy learns some successful behavior. The trajectory-tracking reward is denser: it rewards progress toward the next trajectory point and penalizes position error and action changes. This gives the policy feedback at every step and can create faster early improvement.

Both members used PPO, so algorithm choice does not explain most of the difference. The comparison is mainly about task structure, reward design, and hyperparameters. 王宸澤 used a larger PPO rollout and batch size (`n_steps=1024`, `batch_size=128`), which can stabilize updates over long trajectory episodes. 洪理川 used shorter rollouts (`n_steps=512`, `batch_size=64`), which update more often but see less long-horizon information per update.

The network architecture was similar for both tasks: an MLP policy, not a CNN or RNN. This is appropriate because both agents use vector observations rather than camera images. The obstacle-avoidance policy observes drone state plus sonar-derived risk features, while the trajectory policy observes drone state and target/next-target vectors. A recurrent network might help trajectory tracking remember path phase, but the current implementation instead exposes current and upcoming target vectors directly.

## Presentation Script

Our curves are plotted with comparable axes using normalized moving-average reward. Raw rewards are not directly comparable because obstacle avoidance and trajectory tracking use different reward scales. In my sonar obstacle-avoidance task, unsafe sonar events create sudden large penalties, so the curve has sharp drops. In the trajectory-tracking task, the reward is denser because every step measures progress toward the next trajectory point, so improvement can appear earlier.

The difference is not simply that one model is better. It comes from reward design, task difficulty, PPO hyperparameters, and observation design. Both agents use PPO with an MLP policy, but the obstacle agent must balance target progress against safety, while the trajectory agent mainly minimizes tracking error along a known reference path.
