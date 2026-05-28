#!/usr/bin/env python3
"""Train PPO for the NSYSU Drone sonar obstacle-avoidance task.

This file is the reinforcement-learning equivalent of "start training."
It is intentionally not a full ROS 2 controller like fly_straight.py.

Difference from fly_straight.py:
- fly_straight.py directly subscribes to pose and publishes velocity commands
  with a hand-written proportional-control rule.
- train.py creates a Gymnasium environment. The environment talks to ROS 2,
  computes observations/rewards, and lets Stable-Baselines3 PPO learn the
  velocity-command policy from trial and error.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "Missing RL dependencies. Inside the Docker container, run:\n"
        'python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas'
    ) from exc

from drone_env import DroneSonarAvoidEnv


ROOT = Path(__file__).resolve().parents[1]
PART2_DIR = Path(__file__).resolve().parent
MODEL_DIR = PART2_DIR / "models"
LOG_DIR = PART2_DIR / "logs"
MODEL_PATH = MODEL_DIR / "ppo_drone.zip"
CURVE_PATH = LOG_DIR / "training_curve.png"
MONITOR_PATH = LOG_DIR / "monitor.csv"


def parse_args() -> argparse.Namespace:
    """Read command-line options for the training run.

    Examples:
    - python3 train.py --smoke
      Runs a tiny training job to check that ROS, Gazebo, Gym, PPO, logging,
      and model saving all work.

    - python3 train.py --timesteps 50000
      Runs a longer training job that can start learning useful behavior.
    """
    parser = argparse.ArgumentParser(description="Train PPO for NSYSU Drone Task D.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--smoke", action="store_true", help="Run a short training smoke test.")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--target", nargs=3, type=float, default=(5.0, 3.0, 2.0))
    return parser.parse_args()


def build_training_curve() -> None:
    """Create a PNG reward curve from Stable-Baselines3 Monitor logs.

    Monitor writes one row per finished episode into monitor.csv. The column
    named "r" is the total reward for that episode. This function reads those
    episode rewards and plots them so the report can show whether training
    improved over time.
    """
    rewards: list[float] = []
    if not MONITOR_PATH.exists():
        return

    with MONITOR_PATH.open("r", newline="") as fp:
        # Stable-Baselines3 writes a comment/header line starting with "#".
        # csv.DictReader should only see the actual CSV header and data rows.
        reader = csv.DictReader(row for row in fp if not row.startswith("#"))
        for row in reader:
            try:
                rewards.append(float(row["r"]))
            except (KeyError, ValueError):
                continue

    if not rewards:
        return

    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(rewards) + 1), rewards, label="episode reward")
    if len(rewards) >= 5:
        # A moving average makes the learning trend easier to see than raw,
        # noisy episode rewards alone.
        window = min(20, max(5, len(rewards) // 5))
        moving_avg = [
            sum(rewards[max(0, i - window + 1): i + 1])
            / len(rewards[max(0, i - window + 1): i + 1])
            for i in range(len(rewards))
        ]
        plt.plot(range(1, len(moving_avg) + 1), moving_avg, label=f"{window}-episode avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()

    # Smoke mode is only a pipeline check. It is not enough for final behavior.
    # Longer training should use --timesteps 50000 or more.
    total_timesteps = 1_000 if args.smoke else args.timesteps

    # These folders live in the mounted HW2_Work directory, so models/logs
    # remain visible on the host after training inside Docker.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # DroneSonarAvoidEnv is where ROS 2 communication happens:
    # - publishes /simple_drone/reset, /takeoff, and /cmd_vel
    # - subscribes to pose, velocity, and all sonar topics
    # - converts simulator state into a Gym observation vector
    # - computes reward and termination conditions
    env = DroneSonarAvoidEnv(target=tuple(args.target), max_steps=args.max_steps)

    # Monitor wraps the environment and records episode reward/length statistics
    # to monitor.csv. PPO also prints those statistics during training.
    env = Monitor(env, filename=str(MONITOR_PATH))

    try:
        # PPO is the RL algorithm. "MlpPolicy" means a small neural network that
        # maps the observation vector to a continuous action [vx, vy, vz].
        #
        # This is different from fly_straight.py:
        # - fly_straight.py computes velocity with vx = Kp * error_x.
        # - PPO learns its velocity rule from reward feedback over many episodes.
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            # n_steps is how many environment steps PPO collects before one
            # policy update. A smaller value makes smoke tests finish faster.
            n_steps=256 if args.smoke else 512,
            batch_size=64,
            # gamma controls how much future reward matters. 0.99 is common for
            # navigation tasks because reaching the target may require many steps.
            gamma=0.99,
            # For this MLP policy, CPU is usually faster and avoids GPU warnings.
            device="cpu",
        )

        # This is the actual training loop. Stable-Baselines3 repeatedly calls:
        # obs = env.reset()
        # action = policy(obs)
        # obs, reward, terminated, truncated, info = env.step(action)
        # Then PPO updates the policy to increase expected future reward.
        model.learn(total_timesteps=total_timesteps)

        # Save the learned neural-network policy. test.py loads this file later.
        model.save(MODEL_PATH)

        # Convert monitor.csv into logs/training_curve.png for the report.
        build_training_curve()
        print(f"Saved model: {MODEL_PATH}")
        print(f"Saved training curve: {CURVE_PATH}")
    finally:
        # Always close the environment so the ROS node stops publishing velocity
        # and shuts down cleanly, even if training is interrupted.
        env.close()


if __name__ == "__main__":
    main()
