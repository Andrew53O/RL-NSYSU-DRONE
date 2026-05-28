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
        'python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas'
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
    min_front_sonar = float("inf")
    min_down_sonar = float("inf")
    safety_filter_overrides = 0
    status = "running"

    try:
        obs, info = env.reset()
        min_front_sonar = min(
            min_front_sonar,
            float(info["front_sonar_left"]),
            float(info["front_sonar_center"]),
            float(info["front_sonar_right"]),
            float(info["front_sonar_up"]),
            float(info["front_sonar_down"]),
        )
        min_down_sonar = min(min_down_sonar, float(info["down_sonar_range"]))

        for _ in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            min_front_sonar = min(
                min_front_sonar,
                float(info["front_sonar_left"]),
                float(info["front_sonar_center"]),
                float(info["front_sonar_right"]),
                float(info["front_sonar_up"]),
                float(info["front_sonar_down"]),
            )
            min_down_sonar = min(min_down_sonar, float(info["down_sonar_range"]))
            if bool(info.get("action_was_filtered", False)):
                safety_filter_overrides += 1
            status = str(info["status"])
            if terminated or truncated:
                break

        print(f"status: {status}")
        print(f"final_distance_to_target: {info['distance_to_target']:.3f}")
        print(f"minimum_front_sonar_range: {min_front_sonar:.3f}")
        print(f"minimum_down_sonar_range: {min_down_sonar:.3f}")
        print(f"safety_filter_overrides: {safety_filter_overrides}")
        print(f"total_episode_reward: {total_reward:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
