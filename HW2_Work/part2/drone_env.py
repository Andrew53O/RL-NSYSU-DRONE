#!/usr/bin/env python3
"""Gymnasium environment for NSYSU Drone Task D.

This file intentionally keeps the RL interface small:
- observation: ground-truth pose/velocity, target vector, and processed sonar
- action: continuous velocity command [vx, vy, vz]
- reward: target progress with safety penalties

The current simulator publishes one sonar Range message. In the stock model this
is best treated as a coarse proximity/safety cue, not a full forward obstacle map.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, Twist
from gymnasium import spaces
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Empty


class DroneRosBridge(Node):
    """Thin ROS 2 bridge for the topics used by the Gym environment."""

    def __init__(self, namespace: str = "/simple_drone") -> None:
        super().__init__("drone_sonar_avoid_env")
        ns = namespace.rstrip("/")

        self.pose: np.ndarray | None = None
        self.velocity = np.zeros(3, dtype=np.float32)
        self.sonar_range: float | None = None
        self.sonar_min_range = 0.02
        self.sonar_max_range = 10.0
        self.last_pose_time: float | None = None

        self.cmd_pub = self.create_publisher(Twist, f"{ns}/cmd_vel", 10)
        self.takeoff_pub = self.create_publisher(Empty, f"{ns}/takeoff", 10)
        self.reset_pub = self.create_publisher(Empty, f"{ns}/reset", 10)

        self.create_subscription(Pose, f"{ns}/gt_pose", self._pose_cb, 10)
        self.create_subscription(Twist, f"{ns}/gt_vel", self._vel_cb, 10)
        self.create_subscription(Range, f"{ns}/sonar/out", self._sonar_cb, 10)

    def _pose_cb(self, msg: Pose) -> None:
        self.pose = np.array(
            [msg.position.x, msg.position.y, msg.position.z],
            dtype=np.float32,
        )
        self.last_pose_time = time.monotonic()

    def _vel_cb(self, msg: Twist) -> None:
        self.velocity = np.array(
            [msg.linear.x, msg.linear.y, msg.linear.z],
            dtype=np.float32,
        )

    def _sonar_cb(self, msg: Range) -> None:
        self.sonar_min_range = float(msg.min_range)
        self.sonar_max_range = float(msg.max_range)
        self.sonar_range = float(msg.range)

    def publish_velocity(self, action: np.ndarray) -> None:
        msg = Twist()
        msg.linear.x = float(action[0])
        msg.linear.y = float(action[1])
        msg.linear.z = float(action[2])
        self.cmd_pub.publish(msg)

    def stop(self) -> None:
        self.publish_velocity(np.zeros(3, dtype=np.float32))

    def reset_and_takeoff(self, takeoff_altitude: float = 0.8, timeout_sec: float = 8.0) -> None:
        self.stop()
        self.pose = None
        self.sonar_range = None
        self.reset_pub.publish(Empty())
        rclpy.spin_once(self, timeout_sec=0.5)

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.takeoff_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.pose is not None and self.pose[2] >= takeoff_altitude:
                self.stop()
                return

        self.get_logger().warning(
            f"Takeoff wait timed out before reaching {takeoff_altitude:.2f} m"
        )


class DroneSonarAvoidEnv(gym.Env):
    """ROS 2 + Gazebo drone environment for sonar-based obstacle avoidance."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        target: tuple[float, float, float] = (5.0, 3.0, 2.0),
        max_steps: int = 400,
        namespace: str = "/simple_drone",
        step_dt: float = 0.1,
    ) -> None:
        super().__init__()

        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        self.ros = DroneRosBridge(namespace=namespace)
        self.target = np.array(target, dtype=np.float32)
        self.max_steps = int(max_steps)
        self.step_dt = float(step_dt)

        self.target_reached_distance = 0.4
        self.min_altitude = 0.25
        self.max_altitude = 5.0
        self.xy_limit = 8.0
        self.sonar_unsafe_distance = 0.25
        self.sonar_caution_distance = 1.0
        self.takeoff_altitude = 0.8

        self.step_count = 0
        self.previous_distance: float | None = None
        self.previous_sonar_range = 10.0
        self.recent_sonar = deque(maxlen=10)
        self.last_status = "not_started"

        # Action: velocity command in m/s, published to /simple_drone/cmd_vel.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -0.5], dtype=np.float32),
            high=np.array([1.0, 1.0, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation:
        # [x, y, z, vx, vy, vz, target_x, target_y, target_z,
        #  dx, dy, dz, distance, sonar_range, previous_sonar, min_recent_sonar]
        self.observation_space = spaces.Box(
            low=np.array(
                [-20, -20, 0, -5, -5, -5, -20, -20, 0, -20, -20, -10, 0, 0, 0, 0],
                dtype=np.float32,
            ),
            high=np.array(
                [20, 20, 10, 5, 5, 5, 20, 20, 10, 20, 20, 10, 50, 10, 10, 10],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and "target" in options:
            self.target = np.array(options["target"], dtype=np.float32)

        self.step_count = 0
        self.last_status = "running"
        self.recent_sonar.clear()
        self.previous_sonar_range = self._safe_sonar_range()

        self.ros.reset_and_takeoff(takeoff_altitude=self.takeoff_altitude)
        self._wait_for_initial_state(min_altitude=self.takeoff_altitude)

        obs = self._get_obs()
        self.previous_distance = float(obs[12])
        return obs, self._info(obs)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.ros.publish_velocity(action)
        rclpy.spin_once(self.ros, timeout_sec=self.step_dt)
        self.step_count += 1

        obs = self._get_obs()
        current_distance = float(obs[12])
        sonar_range = float(obs[13])

        progress_reward = 0.0
        if self.previous_distance is not None and math.isfinite(current_distance):
            progress_reward = self.previous_distance - current_distance
        self.previous_distance = current_distance

        distance_penalty = 0.01 * current_distance
        action_penalty = 0.01 * float(np.linalg.norm(action))
        obstacle_penalty = self._obstacle_proximity_penalty(sonar_range)

        reward = progress_reward - distance_penalty - action_penalty - obstacle_penalty

        terminated = False
        truncated = False
        status = "running"

        if not self._sensor_state_valid(obs):
            reward -= 50.0
            terminated = True
            status = "invalid_sensor"
        elif current_distance < self.target_reached_distance:
            reward += 50.0
            terminated = True
            status = "success"
        elif obs[2] < self.min_altitude:
            reward -= 50.0
            terminated = True
            status = "crash"
        elif abs(obs[0]) > self.xy_limit or abs(obs[1]) > self.xy_limit or obs[2] > self.max_altitude:
            reward -= 50.0
            terminated = True
            status = "out_of_bounds"
        elif sonar_range < self.sonar_unsafe_distance:
            reward -= 50.0
            terminated = True
            status = "unsafe_sonar"
        elif self.step_count >= self.max_steps:
            reward -= 5.0
            truncated = True
            status = "timeout"

        self.last_status = status
        if terminated or truncated:
            self.ros.stop()

        return obs, float(reward), terminated, truncated, self._info(obs, status=status)

    def close(self) -> None:
        self.ros.stop()
        self.ros.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def _wait_for_initial_state(
        self,
        timeout_sec: float = 5.0,
        min_altitude: float | None = None,
    ) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self.ros, timeout_sec=0.1)
            pose_ready = self.ros.pose is not None
            sonar_ready = self.ros.sonar_range is not None
            altitude_ready = min_altitude is None or (
                self.ros.pose is not None and self.ros.pose[2] >= min_altitude
            )
            if pose_ready and sonar_ready and altitude_ready:
                return

    def _get_obs(self) -> np.ndarray:
        pose = self.ros.pose
        if pose is None:
            pose = np.full(3, np.nan, dtype=np.float32)

        velocity = self.ros.velocity.astype(np.float32)
        delta = self.target - pose
        distance = float(np.linalg.norm(delta)) if np.all(np.isfinite(delta)) else math.nan

        sonar_range = self._safe_sonar_range()
        self.recent_sonar.append(sonar_range)
        min_recent = min(self.recent_sonar) if self.recent_sonar else sonar_range

        obs = np.concatenate(
            [
                pose.astype(np.float32),
                velocity,
                self.target,
                delta.astype(np.float32),
                np.array(
                    [distance, sonar_range, self.previous_sonar_range, min_recent],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        self.previous_sonar_range = sonar_range
        return obs

    def _safe_sonar_range(self) -> float:
        raw = self.ros.sonar_range
        max_range = max(self.ros.sonar_max_range, 0.1)
        if raw is None or not math.isfinite(raw):
            return max_range
        return float(np.clip(raw, self.ros.sonar_min_range, max_range))

    def _obstacle_proximity_penalty(self, sonar_range: float) -> float:
        if sonar_range >= self.sonar_caution_distance:
            return 0.0
        risk = (self.sonar_caution_distance - sonar_range) / self.sonar_caution_distance
        return 2.0 * risk

    def _sensor_state_valid(self, obs: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(obs)))

    def _info(self, obs: np.ndarray, status: str | None = None) -> dict[str, Any]:
        return {
            "status": status or self.last_status,
            "distance_to_target": float(obs[12]),
            "sonar_range": float(obs[13]),
            "min_recent_sonar_range": float(obs[15]),
            "step_count": self.step_count,
            "target": self.target.copy(),
        }
