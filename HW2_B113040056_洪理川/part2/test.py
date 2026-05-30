#!/usr/bin/env python3
"""Evaluate a trained Part 3 PPO policy."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from stable_baselines3 import PPO
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing Stable-Baselines3. Inside the Docker container, run:\n"
        'python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas'
    ) from exc

from drone_env import DroneCurriculumEnv, normalize_variant


PART3_DIR = Path(__file__).resolve().parent
LOG_DIR = PART3_DIR / "logs"
DEFAULT_MODEL = PART3_DIR / "models" / "latest" / "ppo_drone.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Part 3 drone PPO.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stage", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--variant", choices=("A", "B", "a", "b"), default="A")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--target", nargs=3, type=float, default=None)
    parser.add_argument("--success-distance", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument(
        "--step-dt",
        type=float,
        default=0.1,
        help="Seconds of ROS/Gazebo time to wait after each action. Match the value used during training.",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-position-every", type=int, default=25)
    return parser.parse_args()


def infer_run(model_path: Path) -> tuple[str, str, str]:
    parts = model_path.resolve().parts
    for i, part in enumerate(parts):
        if part.startswith("stage") and i + 1 < len(parts):
            stage = part
            if parts[i + 1].startswith("variant") and i + 2 < len(parts):
                return stage, parts[i + 1], parts[i + 2]
            return stage, "variantA", parts[i + 1]
    return "stage_unknown", "variantA", model_path.stem


def next_eval_path(model_path: Path, stage: int, variant: str) -> Path:
    model_stage, model_variant, model_run = infer_run(model_path)
    stage_name = f"stage{stage}" if model_stage == "stage_unknown" else model_stage
    variant_name = f"variant{variant}" if stage <= 3 else model_variant
    eval_dir = LOG_DIR / "eval" / stage_name / variant_name / model_run
    index = 1
    while True:
        candidate = eval_dir / f"eval{index:03d}.csv"
        if not candidate.exists():
            return candidate
        index += 1


def explicit_eval_path(path: Path, overwrite: bool) -> Path:
    if overwrite or not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def run_episode(env: DroneCurriculumEnv, model: PPO, max_steps: int) -> dict[str, float | int | str]:
    total_reward = 0.0
    min_obstacle = float("inf")
    safety_overrides = 0
    near_misses = 0
    command_sum = np.zeros(3, dtype=np.float64)
    command_count = 0
    obs, info = env.reset()
    status = "running"
    steps = 0
    for steps in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=True)
        command_sum += np.asarray(action, dtype=np.float64)
        command_count += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        min_obstacle = min(min_obstacle, float(info["min_obstacle_sonar_range"]))
        if bool(info.get("action_was_filtered", False)):
            safety_overrides += 1
        if float(info["sonar_enabled"]) > 0.5 and float(info["min_obstacle_sonar_range"]) < 0.5:
            near_misses += 1
        status = str(info["status"])
        if terminated or truncated:
            break
    avg_cmd = command_sum / command_count if command_count else np.zeros(3, dtype=np.float64)
    target_total = int(info.get("total_targets", 1))
    targets_reached = int(info.get("targets_reached", 0))
    return {
        "status": status,
        "final_distance_to_target": float(info["distance_to_target"]),
        "final_x": float(info["x"]),
        "final_y": float(info["y"]),
        "final_z": float(info["z"]),
        "final_dx": float(info["dx"]),
        "final_dy": float(info["dy"]),
        "final_dz": float(info["dz"]),
        "mission_goal_x": float(info.get("mission_goal_x", 0.0)),
        "mission_goal_y": float(info.get("mission_goal_y", 0.0)),
        "mission_goal_z": float(info.get("mission_goal_z", 0.0)),
        "mission_goal_distance": float(info.get("mission_goal_distance", info["distance_to_target"])),
        "episode_return": total_reward,
        "steps": steps,
        "targets_reached": targets_reached,
        "total_targets": target_total,
        "sequence_completion_rate": targets_reached / max(target_total, 1),
        "minimum_obstacle_sonar_range": min_obstacle,
        "safety_filter_overrides": safety_overrides,
        "sonar_near_miss_count": near_misses,
        "average_cmd_vx": float(avg_cmd[0]),
        "average_cmd_vy": float(avg_cmd[1]),
        "average_cmd_vz": float(avg_cmd[2]),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "status",
        "final_distance_to_target",
        "final_x",
        "final_y",
        "final_z",
        "final_dx",
        "final_dy",
        "final_dz",
        "mission_goal_x",
        "mission_goal_y",
        "mission_goal_z",
        "mission_goal_distance",
        "episode_return",
        "steps",
        "targets_reached",
        "total_targets",
        "sequence_completion_rate",
        "minimum_obstacle_sonar_range",
        "safety_filter_overrides",
        "sonar_near_miss_count",
        "average_cmd_vx",
        "average_cmd_vy",
        "average_cmd_vz",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_config(path: Path, args: argparse.Namespace, csv_path: Path) -> None:
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "csv": str(csv_path),
        "stage": args.stage,
        "variant": normalize_variant(args.stage, args.variant),
        "episodes": args.episodes,
        "target_override": list(args.target) if args.target else None,
        "success_distance": args.success_distance,
        "max_steps": args.max_steps,
        "step_dt": args.step_dt,
        "log_position_every": args.log_position_every,
    }
    with path.open("w") as fp:
        json.dump(config, fp, indent=2, sort_keys=True)
        fp.write("\n")


def print_summary(rows: list[dict[str, float | int | str]]) -> None:
    total = len(rows)
    success_rows = [row for row in rows if row["status"] == "success"]
    timeout_rows = [row for row in rows if row["status"] == "timeout"]
    unsafe_rows = [row for row in rows if row["status"] in {"crash", "unsafe_sonar", "out_of_bounds"}]
    print(f"episodes: {total}")
    print(f"success_rate: {len(success_rows) / total:.3f}")
    print(f"timeout_rate: {len(timeout_rows) / total:.3f}")
    print(f"crash_or_unsafe_rate: {len(unsafe_rows) / total:.3f}")
    print(f"average_return: {mean(float(row['episode_return']) for row in rows):.3f}")
    print(f"average_distance_to_target: {mean(float(row['final_distance_to_target']) for row in rows):.3f}")
    print(f"average_mission_goal_distance: {mean(float(row['mission_goal_distance']) for row in rows):.3f}")
    print(f"average_steps_to_target: {mean(float(row['steps']) for row in success_rows) if success_rows else 0.0:.3f}")
    print(f"average_sequence_completion_rate: {mean(float(row['sequence_completion_rate']) for row in rows):.3f}")
    print(f"average_cmd_vx: {mean(float(row['average_cmd_vx']) for row in rows):.3f}")
    print(f"average_cmd_vy: {mean(float(row['average_cmd_vy']) for row in rows):.3f}")
    print(f"average_cmd_vz: {mean(float(row['average_cmd_vz']) for row in rows):.3f}")
    print(f"safety_filter_activation_count: {sum(int(row['safety_filter_overrides']) for row in rows)}")
    print(f"sonar_near_miss_count: {sum(int(row['sonar_near_miss_count']) for row in rows)}")


def main() -> None:
    args = parse_args()
    if args.step_dt <= 0.0:
        raise SystemExit("--step-dt must be greater than 0.0")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    variant = normalize_variant(args.stage, args.variant)
    csv_path = (
        explicit_eval_path(args.csv, args.overwrite)
        if args.csv
        else next_eval_path(args.model, args.stage, variant)
    )
    env = DroneCurriculumEnv(
        stage=args.stage,
        variant=variant,
        target_override=tuple(args.target) if args.target else None,
        max_steps=args.max_steps,
        success_distance=args.success_distance,
        step_dt=args.step_dt,
        log_position_every=args.log_position_every,
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
                f"distance={float(row['final_distance_to_target']):.3f}"
            )
    finally:
        env.close()
    write_csv(csv_path, rows)
    config_path = csv_path.with_name(f"{csv_path.stem}_config.json")
    write_config(config_path, args, csv_path)
    print_summary(rows)
    print(f"saved_eval_csv: {csv_path}")
    print(f"saved_eval_config: {config_path}")


if __name__ == "__main__":
    main()
