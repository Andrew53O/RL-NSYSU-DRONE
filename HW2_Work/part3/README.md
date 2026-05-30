# HW2 Part 3: Clean Six-Stage PPO Curriculum

Part 3 is a clean curriculum pipeline for drone navigation. It avoids the
complex Part 2 reward history and trains skills in order:

1. Vertical altitude control on Gazebo `z`
2. Horizontal movement on Gazebo `x`
3. Combined navigation without sonar
4. Single-obstacle avoidance with sonar
5. Multiple-obstacle avoidance with sonar
6. Sequential targets with sonar obstacle avoidance

Gazebo axes:

```text
x = forward/back
y = left/right
z = altitude
```

Sonar is not used for policy decisions or reward before Stage 4. The observation
still contains sonar slots in Stages 1-3, but they are masked to safe constants
so PPO checkpoints can continue training across all stages.

## Report Section 1: Task Definition and Motivation

The selected task for this homework is Task B: Random Target Navigation. The evaluated part of this project focuses on learning goal-directed drone navigation in open space. Task D, Autonomous Obstacle Avoidance, is treated as a future extension and is prepared in later curriculum stages, but the main report evaluation is based on Task B unless otherwise stated.

The task is formulated as a Markov Decision Process (MDP) \((S, A, R, \gamma)\). The state \(S\) contains the drone position, linear velocity, relative target vector, Euclidean distance to the current target, target progress information, and sonar-related slots. In Stages 1-3, sonar is intentionally masked to safe constant values, so the policy learns navigation without obstacle cues. From Stage 4 onward, sonar values become active for obstacle-avoidance extensions. The action \(A\) is a continuous velocity command published to `/simple_drone/cmd_vel`, represented as `[vx, vy, vz]`, with limits of `vx, vy ∈ [-1.0, 1.0]` and `vz ∈ [-0.5, 0.5]`. The reward \(R\) encourages reduction of target distance and axis-specific error, penalizes excessive distance, unstable velocity, large or jerky actions, unsafe sonar readings in obstacle stages, crashes, timeouts, and out-of-bounds behavior. The discount factor \(\gamma\) is set in the PPO training configuration and represents the importance of long-term progress rather than only immediate movement.

The target sampling range depends on the curriculum stage. Stage 1A uses a fixed vertical target `(0.0, 0.0, 1.2)`, while Stage 1B samples random altitude targets with `z ∈ [0.7, 1.8]`. Stage 2A uses a fixed horizontal target `(1.0, 0.0, 0.8)`, while Stage 2B samples `x ∈ [-1.0, 2.0]` with stable altitude. Stage 3A samples a single random navigation target with `x ∈ [-1.0, 2.5]`, `y = 0.0`, and `z ∈ [0.7, 1.8]`. Stage 3B extends this to three sequential random `x-z` targets. Later stages add sonar-based obstacle conditions: Stage 4 uses a single obstacle, Stage 5 uses multiple obstacles, and Stage 6 combines sequential targets with active sonar obstacle avoidance.

An episode is considered successful when the drone reaches the current target within the success threshold. The default success distance is `0.15 m`. For vertical-only control, success requires the altitude error to be below this threshold while keeping lateral drift within tolerance. For horizontal and combined navigation, success is based on Euclidean distance to the target. Episodes terminate when the target sequence is completed, the drone crashes below the minimum altitude, leaves the allowed workspace, reaches an unsafe sonar distance in obstacle stages, produces invalid sensor values, or exceeds the maximum step limit.

Curriculum learning is necessary because directly training a drone on fully random multi-axis navigation with obstacle sensing is difficult and unstable. The policy must simultaneously learn altitude control, horizontal translation, stopping behavior near the target, and later obstacle reaction. Training these skills all at once creates sparse success signals and can lead to policies that only rise, drift, or pass through the target without stabilizing. The six-stage curriculum reduces this difficulty by first teaching simple vertical control, then horizontal movement, then combined navigation, and only after that adding sonar-based obstacle avoidance. This makes each stage a continuation of a previously learned skill instead of a completely new problem.

Reinforcement learning is preferable to a fixed controller for this task because the desired behavior is not only to move toward one known point, but to generalize across changing targets and eventually react to sensor-based obstacle information. A PID or hand-crafted proportional controller can work for a fixed target if the gains and speed limits are tuned carefully, but its behavior depends strongly on manual tuning and may overshoot, oscillate, drift, or fail when the target distribution or environment changes. PPO learns a policy from interaction data, allowing the controller to combine position error, velocity, target progress, and later sonar risk into one decision rule. This makes RL a plausible approach for reusable autonomous navigation instead of a manually tuned rule for one scenario.

## Quick Checks

```bash
cd /workspace/HW2_Work/part3
python3 -m py_compile drone_env.py train.py test.py
```

## Smoke Tests

```bash
python3 train.py --stage 1 --variant A --smoke
python3 train.py --stage 1 --variant B --smoke
python3 train.py --stage 2 --variant A --smoke
python3 train.py --stage 2 --variant B --smoke
python3 train.py --stage 3 --variant A --smoke
python3 train.py --stage 3 --variant B --smoke
python3 train.py --stage 4 --smoke
python3 train.py --stage 5 --smoke
python3 train.py --stage 6 --smoke
```

## Training Order

For deadline runs, `--step-dt 0.05` is a reasonable speed-up. Keep the same
`--step-dt` when evaluating a model. The default is `0.1`, which is slower but
more conservative for Gazebo physics.

```bash
python3 train.py --stage 1 --variant A --timesteps 30000 --step-dt 0.05
python3 train.py --stage 1 --variant B --resume-from models/stage1/variantA/runXXX/best/best_success_model.zip --timesteps 50000 --step-dt 0.05
python3 train.py --stage 2 --variant A --resume-from models/stage1/variantB/runXXX/best/best_success_model.zip --timesteps 50000 --step-dt 0.05
python3 train.py --stage 2 --variant B --resume-from models/stage2/variantA/runXXX/best/best_success_model.zip --timesteps 70000
python3 train.py --stage 3 --variant A --resume-from models/stage2/variantB/runXXX/best/best_success_model.zip --timesteps 100000
python3 train.py --stage 3 --variant B --resume-from models/stage3/variantA/runXXX/best/best_success_model.zip --timesteps 120000
python3 train.py --stage 4 --resume-from models/stage3/variantB/runXXX/best/best_success_model.zip --timesteps 120000
python3 train.py --stage 5 --resume-from models/stage4/runXXX/best/best_success_model.zip --timesteps 150000
python3 train.py --stage 6 --resume-from models/stage5/runXXX/best/best_success_model.zip --timesteps 200000
```

## Evaluation

```bash
python3 test.py \
  --stage 1 \
  --variant A \
  --model models/stage1/variantA/runXXX/best/best_success_model.zip \
  --episodes 10 \
  --step-dt 0.05
```

Each training run saves `run_config.json`; each evaluation saves
`evalXXX.csv` plus `evalXXX_config.json`.
