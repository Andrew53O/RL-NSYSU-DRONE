#!/usr/bin/env python3
"""Run a trained PPO policy in Gazebo and print evaluation metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "Missing Stable-Baselines3. Inside the Docker container, run:\n"
        'python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas'
    ) from exc

from drone_env import DroneSonarAvoidEnv


PART2_DIR = Path(__file__).resolve().parent
LOG_DIR = PART2_DIR / "logs"
DEFAULT_MODEL_PATH = PART2_DIR / "models" / "ppo_drone.zip"
DEFAULT_EVAL_CSV = LOG_DIR / "eval_metrics.csv"
SIDE_NEAR_MISS_DISTANCE = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test PPO for NSYSU Drone Task D.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument(
        "--success-distance",
        type=float,
        default=0.4,
        help="Distance in meters that counts as reaching the target.",
    )
    parser.add_argument("--target", nargs=3, type=float, default=(1.0, 0.0, 0.8))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--csv", type=Path, default=DEFAULT_EVAL_CSV)
    parser.add_argument(
        "--log-position-every",
        type=int,
        default=25,
        help="Print drone position, target, and distance every N env steps. 0 disables it.",
    )
    return parser.parse_args()


def run_episode(env: DroneSonarAvoidEnv, model: PPO, max_steps: int) -> dict[str, float | int | str]:
    total_reward = 0.0
    min_obstacle_sonar = float("inf")
    min_down_sonar = float("inf")
    safety_filter_overrides = 0
    side_near_misses = 0
    status = "running"
    info = {}
    steps = 0

    obs, info = env.reset()
    min_obstacle_sonar = min(min_obstacle_sonar, float(info["min_obstacle_sonar_range"]))
    min_down_sonar = min(min_down_sonar, float(info["down_sonar_range"]))

    for steps in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        min_obstacle_sonar = min(
            min_obstacle_sonar,
            float(info["min_obstacle_sonar_range"]),
        )
        min_down_sonar = min(min_down_sonar, float(info["down_sonar_range"]))
        side_min = min(float(info["side_sonar_left"]), float(info["side_sonar_right"]))
        if side_min < SIDE_NEAR_MISS_DISTANCE:
            side_near_misses += 1
        if bool(info.get("action_was_filtered", False)):
            safety_filter_overrides += 1
        status = str(info["status"])
        if terminated or truncated:
            break

    return {
        "status": status,
        "final_distance_to_target": float(info["distance_to_target"]),
        "final_x": float(info["x"]),
        "final_y": float(info["y"]),
        "final_z": float(info["z"]),
        "episode_return": total_reward,
        "minimum_obstacle_sonar_range": min_obstacle_sonar,
        "minimum_down_sonar_range": min_down_sonar,
        "steps": steps,
        "safety_filter_overrides": safety_filter_overrides,
        "side_near_miss_count": side_near_misses,
    }


def write_eval_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "status",
        "final_distance_to_target",
        "final_x",
        "final_y",
        "final_z",
        "episode_return",
        "minimum_obstacle_sonar_range",
        "minimum_down_sonar_range",
        "steps",
        "safety_filter_overrides",
        "side_near_miss_count",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    total = len(rows)
    success_rows = [row for row in rows if row["status"] == "success"]
    crash_statuses = {
        "crash",
        "out_of_bounds",
        "unsafe_front_sonar",
        "unsafe_side_sonar",
        "unsafe_down_sonar",
        "invalid_sensor",
    }
    crash_rows = [row for row in rows if row["status"] in crash_statuses]
    timeout_rows = [row for row in rows if row["status"] == "timeout"]
    avg_steps_to_target = mean(float(row["steps"]) for row in success_rows) if success_rows else 0.0

    print(f"episodes: {total}")
    print(f"success_rate: {len(success_rows) / total:.3f}")
    print(f"crash_rate: {len(crash_rows) / total:.3f}")
    print(f"timeout_rate: {len(timeout_rows) / total:.3f}")
    print(f"average_return: {mean(float(row['episode_return']) for row in rows):.3f}")
    print(
        "average_minimum_obstacle_sonar_distance: "
        f"{mean(float(row['minimum_obstacle_sonar_range']) for row in rows):.3f}"
    )
    print(f"average_steps_to_target: {avg_steps_to_target:.3f}")
    print(
        "safety_filter_activation_count: "
        f"{sum(int(row['safety_filter_overrides']) for row in rows)}"
    )
    print(f"side_sonar_near_miss_count: {sum(int(row['side_near_miss_count']) for row in rows)}")


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")

    env = DroneSonarAvoidEnv(
        target=tuple(args.target),
        max_steps=args.max_steps,
        log_position_every=args.log_position_every,
        success_distance=args.success_distance,
    )
    model = PPO.load(args.model, device="cpu")
    rows: list[dict[str, float | int | str]] = []

    try:
        for episode in range(1, args.episodes + 1):
            row = run_episode(env, model, args.max_steps)
            row["episode"] = episode
            rows.append(row)
            print(
                f"episode {episode}: status={row['status']} "
                f"return={float(row['episode_return']):.3f} "
                f"min_obstacle={float(row['minimum_obstacle_sonar_range']):.3f}"
            )
    finally:
        env.close()

    write_eval_csv(args.csv, rows)
    print_summary(rows)
    print(f"saved_eval_csv: {args.csv}")


if __name__ == "__main__":
    main()
