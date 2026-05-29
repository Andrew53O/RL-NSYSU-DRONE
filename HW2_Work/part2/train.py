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
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "Missing RL dependencies. Inside the Docker container, run:\n"
        'python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas'
    ) from exc

from drone_env import DroneSonarAvoidEnv


# File layout:
# - PART2_DIR is /workspace/HW2_Work/part2 inside Docker.
# - models/stageN/runXXX/ stores Stable-Baselines3 PPO .zip checkpoints.
# - logs/stageN/runXXX/ stores Monitor CSV files and training-curve PNGs.
ROOT = Path(__file__).resolve().parents[1]
PART2_DIR = Path(__file__).resolve().parent
MODEL_DIR = PART2_DIR / "models"
LOG_DIR = PART2_DIR / "logs"
MODEL_PATH = MODEL_DIR / "latest" / "ppo_drone.zip"


# Curriculum stages are intentionally simple. They do not change the neural
# network, observation space, action space, or environment class. They only
# change the target selection. This lets a PPO checkpoint from an easier stage
# continue training in a harder stage without shape mismatch.
#
# Target coordinates use the Gazebo/world frame in meters:
# - x: forward/back direction in the world. Positive x is farther "forward"
#   from the origin in the usual fly_straight.py task.
# - y: left/right direction in the world. Positive y is one side of the arena,
#   negative y is the other side.
# - z: altitude above the ground in meters.
#
# Example: target=(1.0, 0.0, 0.8) means "fly to x=1.0 m, y=0 m,
# altitude z=0.8 m." Stage 1 is deliberately very close to the takeoff
# altitude because PPO first needs to learn basic target-directed velocity
# control before obstacle avoidance or high-altitude navigation is meaningful.
CURRICULUM_STAGES = {
    1: {
        "name": "stage1_fixed_easy",
        "target": (1.0, 0.0, 0.8),
        "random_target": False,
        "description": "fixed start/reset and very close low-altitude target",
    },
    2: {
        "name": "stage2_random_open",
        "target": (2.5, 0.0, 1.2),
        "random_target": True,
        "target_bounds": ((1.0, 3.0), (-1.5, 1.5), (0.8, 1.6)),
        "description": "randomized target in open space away from walls",
    },
    3: {
        "name": "stage3_single_path_obstacle",
        "target": (5.0, 3.0, 2.0),
        "random_target": True,
        "target_bounds": ((4.0, 6.0), (1.5, 4.5), (1.5, 3.0)),
        "description": "semi-random target near the direct path through obstacles",
    },
    4: {
        "name": "stage4_final_task_d",
        "target": (6.0, -3.0, 2.2),
        "random_target": True,
        "target_bounds": ((3.5, 7.0), (-5.0, 5.0), (1.4, 3.2)),
        "description": "harder final Task D placements in the playground",
    },
}


class CurriculumTargetWrapper(gym.Wrapper):
    """Select fixed or randomized targets without changing the base env API."""

    def __init__(self, env: DroneSonarAvoidEnv, stage_config: dict) -> None:
        super().__init__(env)
        self.stage_config = stage_config

    def reset(self, **kwargs):
        # Gymnasium reset accepts an "options" dictionary. DroneSonarAvoidEnv
        # already knows how to read options["target"], so the wrapper can choose
        # a target here without modifying the base environment.
        #
        # Important: DroneSonarAvoidEnv.reset() is also where /reset and
        # /takeoff are published. In other words, takeoff already happens before
        # every learning episode. If you see "Takeoff wait timed out", the issue
        # is that the drone did not reach the requested takeoff altitude in time,
        # not that train.py forgot to publish /takeoff.
        options = dict(kwargs.pop("options", {}) or {})
        options["target"] = self._sample_target()
        return self.env.reset(options=options, **kwargs)

    def _sample_target(self) -> tuple[float, float, float]:
        if not self.stage_config.get("random_target", False):
            return tuple(self.stage_config["target"])

        # target_bounds is ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
        # One random coordinate is sampled from each interval. This makes Stage
        # 2+ generalize while preserving the same observation/action shapes.
        bounds = self.stage_config["target_bounds"]
        return tuple(random.uniform(low, high) for low, high in bounds)


class BestTrainingModelCallback(BaseCallback):
    """Save the best policy seen during training, not only the final policy.

    PPO can temporarily learn a good behavior and then drift worse later. This
    callback watches finished Monitor episodes and saves two extra checkpoints:
    - best_episode_model.zip: highest single episode reward
    - best_average_model.zip: highest recent moving-average reward
    - best_success_model.zip: highest recent strict success rate
    - best_precision_model.zip: best strict target-error ranking
    """

    def __init__(self, best_model_dir: Path, window: int = 20, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.best_model_dir = best_model_dir
        self.window = max(1, window)
        self.episode_rewards: list[float] = []
        self.episode_successes: list[int] = []
        self.best_episode_reward = float("-inf")
        self.best_average_reward = float("-inf")
        self.best_success_rate = float("-inf")
        self.best_success_average_reward = float("-inf")
        self.best_precision_key: tuple[int, float, float, float] | None = None
        self.best_summary_path = best_model_dir / "best_summary.csv"

    def _on_training_start(self) -> None:
        self.best_model_dir.mkdir(parents=True, exist_ok=True)
        with self.best_summary_path.open("w", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "episode",
                    "timesteps",
                    "reward",
                    "status",
                    "recent_average_reward",
                    "recent_success_rate",
                    "final_distance_to_target",
                    "abs_final_dx",
                    "abs_final_dz",
                    "precision_success",
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
            self.episode_rewards.append(reward)
            self.episode_successes.append(success)
            recent_rewards = self.episode_rewards[-self.window :]
            recent_average = sum(recent_rewards) / len(recent_rewards)
            recent_successes = self.episode_successes[-self.window :]
            recent_success_rate = sum(recent_successes) / len(recent_successes)
            final_distance = float(info.get("distance_to_target", float("inf")))
            final_x = float(info.get("x", float("nan")))
            final_z = float(info.get("z", float("nan")))
            target = info.get("target")
            if target is not None:
                abs_final_dx = abs(float(target[0]) - final_x)
                abs_final_dz = abs(float(target[2]) - final_z)
            else:
                abs_final_dx = float("inf")
                abs_final_dz = float("inf")
            precision_key = (
                success,
                -final_distance,
                -abs_final_dz,
                -abs_final_dx,
            )
            saved: list[str] = []

            if reward > self.best_episode_reward:
                self.best_episode_reward = reward
                self.model.save(self.best_model_dir / "best_episode_model.zip")
                saved.append("best_episode_model")

            if recent_average > self.best_average_reward:
                self.best_average_reward = recent_average
                self.model.save(self.best_model_dir / "best_average_model.zip")
                saved.append("best_average_model")

            # Reward is useful but imperfect: a loose near-target success can
            # score well while still failing the strict 0.1 m Stage-1 precision
            # criterion. This
            # checkpoint prioritizes actual episode status == "success".
            has_recent_success = sum(recent_successes) > 0
            better_success_rate = recent_success_rate > self.best_success_rate
            tied_rate_better_reward = (
                recent_success_rate == self.best_success_rate
                and recent_average > self.best_success_average_reward
            )
            if has_recent_success and (better_success_rate or tied_rate_better_reward):
                self.best_success_rate = recent_success_rate
                self.best_success_average_reward = recent_average
                self.model.save(self.best_model_dir / "best_success_model.zip")
                saved.append("best_success_model")

            if self.best_precision_key is None or precision_key > self.best_precision_key:
                self.best_precision_key = precision_key
                self.model.save(self.best_model_dir / "best_precision_model.zip")
                saved.append("best_precision_model")

            if saved:
                with self.best_summary_path.open("a", newline="") as fp:
                    writer = csv.DictWriter(
                        fp,
                        fieldnames=[
                            "episode",
                            "timesteps",
                            "reward",
                            "status",
                            "recent_average_reward",
                            "recent_success_rate",
                            "final_distance_to_target",
                            "abs_final_dx",
                            "abs_final_dz",
                            "precision_success",
                            "saved",
                        ],
                    )
                    writer.writerow(
                        {
                            "episode": len(self.episode_rewards),
                            "timesteps": self.num_timesteps,
                            "reward": reward,
                            "status": status,
                            "recent_average_reward": recent_average,
                            "recent_success_rate": recent_success_rate,
                            "final_distance_to_target": final_distance,
                            "abs_final_dx": abs_final_dx,
                            "abs_final_dz": abs_final_dz,
                            "precision_success": success,
                            "saved": "+".join(saved),
                        }
                    )

                if self.verbose:
                    print(
                        "Saved "
                        f"{', '.join(saved)} at episode {len(self.episode_rewards)} "
                        f"(status={status}, reward={reward:.3f}, "
                        f"avg={recent_average:.3f}, "
                        f"success_rate={recent_success_rate:.3f})"
                    )

        return True


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
    # A timestep is one call to env.step(action), not a full episode.
    # max_steps is the episode time limit. Stage-1 evals showed the drone can
    # command max forward velocity but still need more than 800 control steps
    # to settle at a strict 0.1 m target radius.
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--smoke", action="store_true", help="Run a short training smoke test.")
    # max_steps is the episode time limit. If rollout/ep_len_mean stays at 400,
    # the policy is usually timing out instead of reaching the target.
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument(
        "--success-distance",
        type=float,
        default=0.1,
        help=(
            "Distance in meters that counts as reaching the target. "
            "Stage 1 precision training defaults to 0.1; use 0.4 only as a "
            "loose sanity check."
        ),
    )
    parser.add_argument("--stage", type=int, choices=sorted(CURRICULUM_STAGES), default=1)
    parser.add_argument(
        "--target",
        nargs=3,
        type=float,
        default=None,
        help=(
            "Override the stage target as x y z in meters. This disables random "
            "target sampling for the run."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional PPO checkpoint to continue training from.",
    )
    parser.add_argument(
        "--log-position-every",
        type=int,
        default=0,
        help="Print drone position, target, and distance every N env steps. 0 disables it.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Save a periodic PPO checkpoint every N timesteps. Use 0 to disable.",
    )
    parser.add_argument(
        "--best-window",
        type=int,
        default=20,
        help="Episode window used when saving best_average_model.zip.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run label for model/log filenames. Auto-numbered if omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reusing an existing run-name and overwriting its outputs.",
    )
    parser.add_argument(
        "--update-latest",
        action="store_true",
        help="Also update ppo_drone_stageN.zip and ppo_drone.zip convenience copies.",
    )
    return parser.parse_args()


def build_training_curve(monitor_path: Path, curve_path: Path, csv_path: Path) -> bool:
    """Create PNG and CSV reward curves from Stable-Baselines3 Monitor logs.

    Monitor writes one row per finished episode into monitor.csv. The column
    named "r" is the total reward for that episode. This function reads those
    episode rewards and saves:
    - a PNG plot for the report
    - a CSV table that is easy to inspect with tail, pandas, or a spreadsheet
    """
    rewards: list[float] = []
    if not monitor_path.exists():
        return False

    with monitor_path.open("r", newline="") as fp:
        # Stable-Baselines3 writes a comment/header line starting with "#".
        # csv.DictReader should only see the actual CSV header and data rows.
        reader = csv.DictReader(row for row in fp if not row.startswith("#"))
        for row in reader:
            try:
                rewards.append(float(row["r"]))
            except (KeyError, ValueError):
                continue

    if not rewards:
        return False

    window = min(20, max(5, len(rewards) // 5)) if len(rewards) >= 5 else 1
    moving_avg = [
        sum(rewards[max(0, i - window + 1): i + 1])
        / len(rewards[max(0, i - window + 1): i + 1])
        for i in range(len(rewards))
    ]

    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "episode",
                "reward",
                "moving_average_reward",
                "moving_average_window",
            ],
        )
        writer.writeheader()
        for episode, (reward, avg_reward) in enumerate(zip(rewards, moving_avg), start=1):
            writer.writerow(
                {
                    "episode": episode,
                    "reward": reward,
                    "moving_average_reward": avg_reward,
                    "moving_average_window": window,
                }
            )

    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(rewards) + 1), rewards, label="episode reward")
    if len(rewards) >= 5:
        # A moving average makes the learning trend easier to see than raw,
        # noisy episode rewards alone.
        plt.plot(range(1, len(moving_avg) + 1), moving_avg, label=f"{window}-episode avg")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()
    return True


def _sanitize_run_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return safe or "run"


def _next_stage_run_name(stage: int, smoke: bool) -> str:
    stage_model_dir = MODEL_DIR / f"stage{stage}"
    stage_log_dir = LOG_DIR / f"stage{stage}"
    prefix = "smoke_run" if smoke else "run"
    index = 1
    while True:
        candidate = f"{prefix}{index:03d}"
        paths = [
            stage_model_dir / candidate / "ppo_drone.zip",
            stage_log_dir / candidate / "monitor.csv",
            stage_log_dir / candidate / "training_curve.png",
            stage_log_dir / candidate / "training_curve.csv",
        ]
        if not any(path.exists() for path in paths):
            return candidate
        index += 1


def _latest_stage_model(stage: int) -> Path | None:
    candidates = list((MODEL_DIR / f"stage{stage}").glob("*/ppo_drone.zip"))
    legacy_path = MODEL_DIR / f"ppo_drone_stage{stage}.zip"
    if legacy_path.exists():
        candidates.append(legacy_path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_resume_path(args: argparse.Namespace) -> Path | None:
    # Manual resume has priority. Use this when you want to keep training the
    # same stage after a 10k-timestep chunk:
    # python3 train.py --stage 1 --resume-from models/stage1/run001/ppo_drone.zip ...
    if args.resume_from is not None:
        return args.resume_from

    # For Stage 2+, automatically continue from the previous stage if it exists.
    # This is the curriculum handoff: Stage 2 starts from Stage 1, Stage 3 from
    # Stage 2, and Stage 4 from Stage 3.
    if args.stage > 1:
        return _latest_stage_model(args.stage - 1)
    return None


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def write_run_config(
    path: Path,
    args: argparse.Namespace,
    stage_config: dict,
    run_name: str,
    total_timesteps: int,
    effective_n_steps: int,
    resume_path: Path | None,
    run_model_dir: Path,
    run_log_dir: Path,
) -> None:
    """Persist the exact training settings used for this PPO run."""
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "stage": args.stage,
        "stage_config": _json_ready(stage_config),
        "target": list(stage_config["target"]),
        "success_distance": args.success_distance,
        "total_timesteps": total_timesteps,
        "smoke": args.smoke,
        "max_steps": args.max_steps,
        "ppo": {
            "policy": "MlpPolicy",
            "learning_rate": args.learning_rate,
            "n_steps": effective_n_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "device": "cpu",
        },
        "callbacks": {
            "best_window": args.best_window,
            "checkpoint_freq": args.checkpoint_freq,
            "best_models": [
                "best_episode_model.zip",
                "best_average_model.zip",
                "best_success_model.zip",
                "best_precision_model.zip",
            ],
        },
        "observation_normalization": {
            "pose_xy_norm": 8.0,
            "pose_z_norm": 5.0,
            "target_xy_delta_norm": 3.0,
            "target_z_delta_norm": 1.5,
            "target_distance_norm": 3.0,
            "sonar_range_norm": 10.0,
        },
        "resume_from": str(resume_path) if resume_path is not None else None,
        "outputs": {
            "model_dir": str(run_model_dir),
            "log_dir": str(run_log_dir),
        },
        "notes": (
            "Stage 1 precision training uses success_distance=0.1 by default; "
            "0.4 is only a loose sanity metric."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(config, fp, indent=2, sort_keys=True)
        fp.write("\n")


def main() -> None:
    args = parse_args()
    stage_config = dict(CURRICULUM_STAGES[args.stage])
    if args.target:
        # A user-supplied --target is useful for deterministic evaluation-style
        # training experiments. It turns even random stages into fixed-target
        # runs for this process.
        stage_config["target"] = tuple(args.target)
        stage_config["random_target"] = False

    # Smoke mode is only a pipeline check. It is not enough for final behavior.
    # Longer training should use --timesteps 50000 or more.
    total_timesteps = 1_000 if args.smoke else args.timesteps
    # Each run gets numbered outputs by default so experiments are not
    # overwritten. Use --run-name for a human label and --overwrite only when
    # you intentionally want to replace that run's files.
    run_name = (
        _sanitize_run_name(args.run_name)
        if args.run_name
        else _next_stage_run_name(args.stage, args.smoke)
    )
    run_model_dir = MODEL_DIR / f"stage{args.stage}" / run_name
    run_log_dir = LOG_DIR / f"stage{args.stage}" / run_name
    model_path = run_model_dir / "ppo_drone.zip"
    best_model_dir = run_model_dir / "best"
    checkpoint_dir = run_model_dir / "checkpoints"
    monitor_path = run_log_dir / "monitor.csv"
    curve_path = run_log_dir / "training_curve.png"
    curve_csv_path = run_log_dir / "training_curve.csv"
    model_config_path = run_model_dir / "run_config.json"
    log_config_path = run_log_dir / "run_config.json"

    # These folders live in the mounted HW2_Work directory, so models/logs
    # remain visible on the host after training inside Docker.
    run_model_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        existing_outputs = [
            path
            for path in (
                model_path,
                monitor_path,
                curve_path,
                curve_csv_path,
                model_config_path,
                log_config_path,
            )
            if path.exists()
        ]
        if existing_outputs:
            paths = "\n".join(str(path) for path in existing_outputs)
            raise SystemExit(
                "Refusing to overwrite existing run outputs. Use a different "
                f"--run-name or pass --overwrite.\n{paths}"
            )

    # DroneSonarAvoidEnv is where ROS 2 communication happens:
    # - publishes /simple_drone/reset, /takeoff, and /cmd_vel
    # - subscribes to pose, velocity, and all sonar topics
    # - converts simulator state into a Gym observation vector
    # - computes reward and termination conditions
    #
    # Takeoff is not controlled by PPO. The env reset routine publishes takeoff
    # first, waits for a safe starting altitude, and then PPO begins choosing
    # velocity commands. So Stage 1 trains "fly from the reset/takeoff state to
    # the target", not "learn the takeoff command itself."
    base_target = tuple(stage_config["target"])
    env = DroneSonarAvoidEnv(
        target=base_target,
        max_steps=args.max_steps,
        log_position_every=args.log_position_every,
        success_distance=args.success_distance,
    )
    env = CurriculumTargetWrapper(env, stage_config)

    # Monitor wraps the environment and records episode reward/length statistics
    # to monitor.csv. PPO also prints those statistics during training.
    env = Monitor(env, filename=str(monitor_path))

    resume_path = resolve_resume_path(args)
    effective_n_steps = 256 if args.smoke else args.n_steps
    write_run_config(
        path=model_config_path,
        args=args,
        stage_config=stage_config,
        run_name=run_name,
        total_timesteps=total_timesteps,
        effective_n_steps=effective_n_steps,
        resume_path=resume_path,
        run_model_dir=run_model_dir,
        run_log_dir=run_log_dir,
    )
    write_run_config(
        path=log_config_path,
        args=args,
        stage_config=stage_config,
        run_name=run_name,
        total_timesteps=total_timesteps,
        effective_n_steps=effective_n_steps,
        resume_path=resume_path,
        run_model_dir=run_model_dir,
        run_log_dir=run_log_dir,
    )

    try:
        if resume_path is not None:
            if not resume_path.exists():
                raise SystemExit(f"Resume checkpoint not found: {resume_path}")
            # PPO checkpoints store their old optimizer settings. Passing these
            # values into load() makes --learning-rate/--n-steps/--batch-size/
            # --gamma actually apply when continuing a run.
            model = PPO.load(
                resume_path,
                env=env,
                device="cpu",
                learning_rate=args.learning_rate,
                n_steps=effective_n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
            )
            print(f"Resuming PPO from: {resume_path}")
            print(
                "Applied PPO settings after resume: "
                f"learning_rate={args.learning_rate}, n_steps={effective_n_steps}, "
                f"batch_size={args.batch_size}, gamma={args.gamma}"
            )
        else:
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
                learning_rate=args.learning_rate,
                # n_steps is how many environment steps PPO collects before one
                # policy update. A smaller value makes smoke tests finish faster.
                n_steps=effective_n_steps,
                batch_size=args.batch_size,
                # gamma controls how much future reward matters. 0.99 is common for
                # navigation tasks because reaching the target may require many steps.
                gamma=args.gamma,
                # For this MLP policy, CPU is usually faster and avoids GPU warnings.
                device="cpu",
            )

        print(
            f"Training curriculum stage {args.stage}: "
            f"{stage_config['name']} ({stage_config['description']})"
        )

        # This is the actual training loop. Stable-Baselines3 repeatedly calls:
        # obs = env.reset()
        # action = policy(obs)
        # obs, reward, terminated, truncated, info = env.step(action)
        # Then PPO updates the policy to increase expected future reward.
        callbacks = [
            BestTrainingModelCallback(
                best_model_dir=best_model_dir,
                window=args.best_window,
                verbose=1,
            )
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
        callback = CallbackList(callbacks)

        model.learn(total_timesteps=total_timesteps, callback=callback)

        # Save the learned neural-network policy. test.py loads this file later.
        model.save(model_path)
        if args.update_latest:
            latest_stage_dir = MODEL_DIR / f"stage{args.stage}" / "latest"
            latest_stage_dir.mkdir(parents=True, exist_ok=True)
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            model.save(latest_stage_dir / "ppo_drone.zip")
            model.save(MODEL_PATH)

        # Convert monitor.csv into report PNG and readable CSV training curves.
        curve_saved = build_training_curve(monitor_path, curve_path, curve_csv_path)
        print(f"Run name: {run_name}")
        print(f"Saved stage model: {model_path}")
        print(f"Saved run config: {model_config_path}")
        print(f"Saved best models under: {best_model_dir}")
        if args.checkpoint_freq > 0:
            print(f"Saved periodic checkpoints under: {checkpoint_dir}")
        if args.update_latest:
            print(f"Updated stage latest model: {latest_stage_dir / 'ppo_drone.zip'}")
            print(f"Updated default model copy: {MODEL_PATH}")
        if curve_saved:
            print(f"Saved training curve: {curve_path}")
            print(f"Saved training curve CSV: {curve_csv_path}")
    finally:
        # Always close the environment so the ROS node stops publishing velocity
        # and shuts down cleanly, even if training is interrupted.
        env.close()


if __name__ == "__main__":
    main()
