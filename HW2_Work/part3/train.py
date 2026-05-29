#!/usr/bin/env python3
"""Train PPO for the clean Part 3 six-stage drone curriculum."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing RL dependencies. Inside the Docker container, run:\n"
        'python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas'
    ) from exc

from drone_env import DroneCurriculumEnv, get_stage_spec, normalize_variant


PART3_DIR = Path(__file__).resolve().parent
MODEL_DIR = PART3_DIR / "models"
LOG_DIR = PART3_DIR / "logs"


class BestTrainingModelCallback(BaseCallback):
    """Save best episode, average, success-rate, and precision checkpoints."""

    def __init__(self, best_model_dir: Path, window: int = 20, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.best_model_dir = best_model_dir
        self.window = max(1, window)
        self.rewards: list[float] = []
        self.successes: list[int] = []
        self.best_episode_reward = float("-inf")
        self.best_average_reward = float("-inf")
        self.best_success_rate = float("-inf")
        self.best_precision_key: tuple[int, float, int] | None = None
        self.summary_path = best_model_dir / "best_summary.csv"

    def _on_training_start(self) -> None:
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "episode",
                    "timesteps",
                    "reward",
                    "status",
                    "recent_average_reward",
                    "recent_success_rate",
                    "distance_to_target",
                    "targets_reached",
                    "saved",
                ],
            )
            writer.writeheader()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode_info = info.get("episode")
            if not episode_info:
                continue
            reward = float(episode_info["r"])
            status = str(info.get("status", "unknown"))
            success = 1 if status == "success" else 0
            distance = float(info.get("distance_to_target", float("inf")))
            targets_reached = int(info.get("targets_reached", 0))
            self.rewards.append(reward)
            self.successes.append(success)
            recent_rewards = self.rewards[-self.window :]
            recent_successes = self.successes[-self.window :]
            recent_average = sum(recent_rewards) / len(recent_rewards)
            recent_success_rate = sum(recent_successes) / len(recent_successes)
            precision_key = (success, targets_reached, -distance)
            saved: list[str] = []

            if reward > self.best_episode_reward:
                self.best_episode_reward = reward
                self.model.save(self.best_model_dir / "best_episode_model.zip")
                saved.append("best_episode_model")
            if recent_average > self.best_average_reward:
                self.best_average_reward = recent_average
                self.model.save(self.best_model_dir / "best_average_model.zip")
                saved.append("best_average_model")
            if recent_success_rate > self.best_success_rate:
                self.best_success_rate = recent_success_rate
                self.model.save(self.best_model_dir / "best_success_model.zip")
                saved.append("best_success_model")
            if self.best_precision_key is None or precision_key > self.best_precision_key:
                self.best_precision_key = precision_key
                self.model.save(self.best_model_dir / "best_precision_model.zip")
                saved.append("best_precision_model")

            if saved:
                with self.summary_path.open("a", newline="") as fp:
                    writer = csv.DictWriter(
                        fp,
                        fieldnames=[
                            "episode",
                            "timesteps",
                            "reward",
                            "status",
                            "recent_average_reward",
                            "recent_success_rate",
                            "distance_to_target",
                            "targets_reached",
                            "saved",
                        ],
                    )
                    writer.writerow(
                        {
                            "episode": len(self.rewards),
                            "timesteps": self.num_timesteps,
                            "reward": reward,
                            "status": status,
                            "recent_average_reward": recent_average,
                            "recent_success_rate": recent_success_rate,
                            "distance_to_target": distance,
                            "targets_reached": targets_reached,
                            "saved": "+".join(saved),
                        }
                    )
                if self.verbose:
                    print(f"Saved {', '.join(saved)} at episode {len(self.rewards)}")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Part 3 drone curriculum PPO.")
    parser.add_argument("--stage", type=int, choices=range(1, 7), default=1)
    parser.add_argument("--variant", choices=("A", "B", "a", "b"), default="A")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--success-distance", type=float, default=0.15)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--best-window", type=int, default=20)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target", nargs=3, type=float, default=None)
    parser.add_argument("--log-position-every", type=int, default=0)
    return parser.parse_args()


def run_root(base: Path, stage: int, variant: str) -> Path:
    if stage <= 3:
        return base / f"stage{stage}" / f"variant{variant}"
    return base / f"stage{stage}"


def next_run_name(model_root: Path, log_root: Path, smoke: bool) -> str:
    prefix = "smoke_run" if smoke else "run"
    index = 1
    while True:
        candidate = f"{prefix}{index:03d}"
        if not (model_root / candidate).exists() and not (log_root / candidate).exists():
            return candidate
        index += 1


def build_training_curve(monitor_path: Path, png_path: Path, csv_path: Path) -> bool:
    rewards: list[float] = []
    if not monitor_path.exists():
        return False
    with monitor_path.open("r", newline="") as fp:
        reader = csv.DictReader(row for row in fp if not row.startswith("#"))
        for row in reader:
            try:
                rewards.append(float(row["r"]))
            except (KeyError, ValueError):
                continue
    if not rewards:
        return False
    window = min(20, max(5, len(rewards) // 5)) if len(rewards) >= 5 else 1
    averages = [
        sum(rewards[max(0, i - window + 1): i + 1])
        / len(rewards[max(0, i - window + 1): i + 1])
        for i in range(len(rewards))
    ]
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["episode", "reward", "moving_average_reward", "moving_average_window"],
        )
        writer.writeheader()
        for episode, (reward, average) in enumerate(zip(rewards, averages), start=1):
            writer.writerow(
                {
                    "episode": episode,
                    "reward": reward,
                    "moving_average_reward": average,
                    "moving_average_window": window,
                }
            )
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(rewards) + 1), rewards, label="episode reward")
    if len(rewards) >= 5:
        plt.plot(range(1, len(averages) + 1), averages, label=f"{window}-episode avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Part 3 PPO Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    return True


def write_run_config(path: Path, args: argparse.Namespace, spec_name: str, run_name: str, model_dir: Path, log_dir: Path) -> None:
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "variant": normalize_variant(args.stage, args.variant),
        "stage_spec": spec_name,
        "run_name": run_name,
        "target_override": list(args.target) if args.target else None,
        "success_distance": args.success_distance,
        "max_steps": args.max_steps,
        "timesteps": 1_000 if args.smoke else args.timesteps,
        "smoke": args.smoke,
        "resume_from": str(args.resume_from) if args.resume_from else None,
        "ppo": {
            "policy": "MlpPolicy",
            "learning_rate": args.learning_rate,
            "n_steps": 256 if args.smoke else args.n_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "device": "cpu",
        },
        "outputs": {"model_dir": str(model_dir), "log_dir": str(log_dir)},
        "notes": "Sonar observation fields are masked before Stage 4.",
    }
    with path.open("w") as fp:
        json.dump(config, fp, indent=2, sort_keys=True)
        fp.write("\n")


def main() -> None:
    args = parse_args()
    variant = normalize_variant(args.stage, args.variant)
    spec = get_stage_spec(args.stage, variant)
    total_timesteps = 1_000 if args.smoke else args.timesteps
    effective_n_steps = 256 if args.smoke else args.n_steps

    model_root = run_root(MODEL_DIR, args.stage, variant)
    log_root = run_root(LOG_DIR, args.stage, variant)
    run_name = args.run_name or next_run_name(model_root, log_root, args.smoke)
    model_dir = model_root / run_name
    log_dir = log_root / run_name
    best_dir = model_dir / "best"
    checkpoint_dir = model_dir / "checkpoints"
    model_path = model_dir / "ppo_drone.zip"
    monitor_path = log_dir / "monitor.csv"
    curve_png = log_dir / "training_curve.png"
    curve_csv = log_dir / "training_curve.csv"
    config_paths = [model_dir / "run_config.json", log_dir / "run_config.json"]

    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and any(path.exists() for path in [model_path, monitor_path, *config_paths]):
        raise SystemExit("Run outputs already exist. Use --run-name or --overwrite.")

    for config_path in config_paths:
        write_run_config(config_path, args, spec.name, run_name, model_dir, log_dir)

    env = DroneCurriculumEnv(
        stage=args.stage,
        variant=variant,
        target_override=tuple(args.target) if args.target else None,
        max_steps=args.max_steps,
        success_distance=args.success_distance,
        log_position_every=args.log_position_every,
    )
    env = Monitor(env, filename=str(monitor_path))

    try:
        if args.resume_from:
            if not args.resume_from.exists():
                raise SystemExit(f"Resume checkpoint not found: {args.resume_from}")
            model = PPO.load(
                args.resume_from,
                env=env,
                device="cpu",
                learning_rate=args.learning_rate,
                n_steps=effective_n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
            )
            print(f"Resuming PPO from: {args.resume_from}")
        else:
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=args.learning_rate,
                n_steps=effective_n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                device="cpu",
            )

        print(f"Training Part 3 {spec.name}: {spec.description}")
        callbacks = [
            BestTrainingModelCallback(best_model_dir=best_dir, window=args.best_window, verbose=1)
        ]
        if args.checkpoint_freq > 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                CheckpointCallback(
                    save_freq=args.checkpoint_freq,
                    save_path=str(checkpoint_dir),
                    name_prefix="ppo_drone",
                )
            )
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
        model.save(model_path)
        if build_training_curve(monitor_path, curve_png, curve_csv):
            print(f"Saved training curve: {curve_png}")
            print(f"Saved training curve CSV: {curve_csv}")
        print(f"Saved model: {model_path}")
        print(f"Saved best models under: {best_dir}")
        latest_dir = model_root / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, latest_dir / "ppo_drone.zip")
    finally:
        env.close()


if __name__ == "__main__":
    main()
