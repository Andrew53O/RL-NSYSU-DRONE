#!/usr/bin/env python3
"""Train PPO for the NSYSU Drone sonar obstacle-avoidance task."""

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
    parser = argparse.ArgumentParser(description="Train PPO for NSYSU Drone Task D.")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--smoke", action="store_true", help="Run a short training smoke test.")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--target", nargs=3, type=float, default=(5.0, 3.0, 2.0))
    return parser.parse_args()


def build_training_curve() -> None:
    rewards: list[float] = []
    if not MONITOR_PATH.exists():
        return

    with MONITOR_PATH.open("r", newline="") as fp:
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
    total_timesteps = 1_000 if args.smoke else args.timesteps
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = DroneSonarAvoidEnv(target=tuple(args.target), max_steps=args.max_steps)
    env = Monitor(env, filename=str(MONITOR_PATH))

    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=256 if args.smoke else 512,
            batch_size=64,
            gamma=0.99,
            device="cpu",
        )
        model.learn(total_timesteps=total_timesteps)
        model.save(MODEL_PATH)
        build_training_curve()
        print(f"Saved model: {MODEL_PATH}")
        print(f"Saved training curve: {CURVE_PATH}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
