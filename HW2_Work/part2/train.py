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
import random
from pathlib import Path

try:
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
except ImportError as exc:  # pragma: no cover - helpful runtime message
    raise SystemExit(
        "Missing RL dependencies. Inside the Docker container, run:\n"
        'python3 -m pip install "numpy<2" gymnasium stable-baselines3 matplotlib pandas'
    ) from exc

from drone_env import DroneSonarAvoidEnv


# File layout:
# - PART2_DIR is /workspace/HW2_Work/part2 inside Docker.
# - models/ stores Stable-Baselines3 PPO .zip checkpoints.
# - logs/ stores Monitor CSV files and training-curve PNGs for the report.
ROOT = Path(__file__).resolve().parents[1]
PART2_DIR = Path(__file__).resolve().parent
MODEL_DIR = PART2_DIR / "models"
LOG_DIR = PART2_DIR / "logs"
MODEL_PATH = MODEL_DIR / "ppo_drone.zip"


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
# Example: target=(2.5, 0.0, 1.5) means "fly to x=2.5 m, y=0 m,
# altitude z=1.5 m." Stage 1 is fixed because PPO first needs to learn basic
# target-directed velocity control before obstacle avoidance is meaningful.
CURRICULUM_STAGES = {
    1: {
        "name": "stage1_fixed_easy",
        "target": (2.5, 0.0, 1.5),
        "random_target": False,
        "description": "fixed start/reset and fixed easy target",
    },
    2: {
        "name": "stage2_random_open",
        "target": (4.0, 2.0, 2.0),
        "random_target": True,
        "target_bounds": ((2.0, 5.5), (-4.5, 4.5), (1.3, 3.0)),
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
    # With max_steps=400 and step_dt=0.1 in DroneSonarAvoidEnv, one full timeout
    # episode is about 400 timesteps, or roughly 40 simulated seconds.
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--smoke", action="store_true", help="Run a short training smoke test.")
    # max_steps is the episode time limit. If rollout/ep_len_mean stays at 400,
    # the policy is usually timing out instead of reaching the target.
    parser.add_argument("--max-steps", type=int, default=400)
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
    return parser.parse_args()


def build_training_curve(monitor_path: Path, curve_path: Path) -> None:
    """Create a PNG reward curve from Stable-Baselines3 Monitor logs.

    Monitor writes one row per finished episode into monitor.csv. The column
    named "r" is the total reward for that episode. This function reads those
    episode rewards and plots them so the report can show whether training
    improved over time.
    """
    rewards: list[float] = []
    if not monitor_path.exists():
        return

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
    plt.savefig(curve_path, dpi=150)
    plt.close()


def resolve_resume_path(args: argparse.Namespace) -> Path | None:
    # Manual resume has priority. Use this when you want to keep training the
    # same stage after a 10k-timestep chunk:
    # python3 train.py --stage 1 --resume-from models/ppo_drone_stage1.zip ...
    if args.resume_from is not None:
        return args.resume_from

    # For Stage 2+, automatically continue from the previous stage if it exists.
    # This is the curriculum handoff: Stage 2 starts from Stage 1, Stage 3 from
    # Stage 2, and Stage 4 from Stage 3.
    previous_stage_path = MODEL_DIR / f"ppo_drone_stage{args.stage - 1}.zip"
    if args.stage > 1 and previous_stage_path.exists():
        return previous_stage_path
    return None


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
    # Each stage gets its own checkpoint and curve. ppo_drone.zip is also saved
    # as a convenience "latest model" path for test.py.
    model_path = MODEL_DIR / f"ppo_drone_stage{args.stage}.zip"
    monitor_path = LOG_DIR / f"stage{args.stage}.monitor.csv"
    curve_path = LOG_DIR / f"training_curve_stage{args.stage}.png"

    # These folders live in the mounted HW2_Work directory, so models/logs
    # remain visible on the host after training inside Docker.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

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
    env = DroneSonarAvoidEnv(target=base_target, max_steps=args.max_steps)
    env = CurriculumTargetWrapper(env, stage_config)

    # Monitor wraps the environment and records episode reward/length statistics
    # to monitor.csv. PPO also prints those statistics during training.
    env = Monitor(env, filename=str(monitor_path))

    try:
        resume_path = resolve_resume_path(args)
        if resume_path is not None:
            if not resume_path.exists():
                raise SystemExit(f"Resume checkpoint not found: {resume_path}")
            model = PPO.load(resume_path, env=env, device="cpu")
            print(f"Resuming PPO from: {resume_path}")
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

        print(
            f"Training curriculum stage {args.stage}: "
            f"{stage_config['name']} ({stage_config['description']})"
        )

        # This is the actual training loop. Stable-Baselines3 repeatedly calls:
        # obs = env.reset()
        # action = policy(obs)
        # obs, reward, terminated, truncated, info = env.step(action)
        # Then PPO updates the policy to increase expected future reward.
        model.learn(total_timesteps=total_timesteps)

        # Save the learned neural-network policy. test.py loads this file later.
        model.save(model_path)
        model.save(MODEL_PATH)

        # Convert monitor.csv into logs/training_curve.png for the report.
        build_training_curve(monitor_path, curve_path)
        print(f"Saved stage model: {model_path}")
        print(f"Saved default model copy: {MODEL_PATH}")
        print(f"Saved training curve: {curve_path}")
    finally:
        # Always close the environment so the ROS node stops publishing velocity
        # and shuts down cleanly, even if training is interrupted.
        env.close()


if __name__ == "__main__":
    main()
