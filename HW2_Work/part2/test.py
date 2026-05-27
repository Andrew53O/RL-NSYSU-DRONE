#!/usr/bin/env python3
"""Run a trained PPO policy in Gazebo and print evaluation metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "Missing Stable-Baselines3. Inside the Docker container, run:\n"
        "python3 -m pip install gymnasium stable-baselines3 matplotlib pandas"
    ) from exc

from drone_env import DroneSonarAvoidEnv


PART2_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PART2_DIR / "models" / "ppo_drone.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PPO for NSYSU Drone Task D.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--target", nargs=3, type=float, default=(5.0, 3.0, 2.0))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    env = DroneSonarAvoidEnv(target=tuple(args.target), max_steps=args.max_steps)
    model = PPO.load(args.model)

    total_reward = 0.0
    min_sonar = float("inf")
    status = "running"

    try:
        obs, info = env.reset()
        min_sonar = min(min_sonar, float(info["sonar_range"]))

        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            min_sonar = min(min_sonar, float(info["sonar_range"]))
            status = str(info["status"])
            if terminated or truncated:
                break

        print(f"status: {status}")
        print(f"final_distance_to_target: {info['distance_to_target']:.3f}")
        print(f"minimum_sonar_range: {min_sonar:.3f}")
        print(f"total_episode_reward: {total_reward:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
