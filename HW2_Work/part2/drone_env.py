#!/usr/bin/env python3
"""Gymnasium environment for NSYSU Drone Task D.

This file intentionally keeps the RL interface small:
- observation: normalized pose/velocity/target and processed sonar-risk sectors
- action: continuous velocity command [vx, vy, vz]
- reward: target progress with sonar risk, trend, smoothness, and safety penalties

The original simulator publishes one downward Range message. For Task D, this
workspace adds forward and side sonar sectors and treats the downward sonar as
a separate altitude/proximity safety cue.
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
from std_srvs.srv import Empty as EmptySrv

try:
    from gazebo_msgs.srv import DeleteEntity, SpawnEntity
except ImportError:  # pragma: no cover - available inside the ROS/Gazebo container
    DeleteEntity = None
    SpawnEntity = None


OBSTACLE_SONAR_SECTORS = (
    "front_left",
    "front_center",
    "front_right",
    "front_up",
    "front_down",
    "side_left",
    "side_right",
)
OBSTACLE_SONAR_COUNT = len(OBSTACLE_SONAR_SECTORS)
OBSERVATION_DIM = 10 + (4 * OBSTACLE_SONAR_COUNT) + 5
TARGET_MARKER_NAME = "rl_target_marker"
TARGET_MARKER_SDF = """
<sdf version="1.6">
  <model name="rl_target_marker">
    <static>true</static>
    <link name="target_link">
      <visual name="target_visual">
        <geometry>
          <sphere>
            <radius>0.18</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0.0 1.0 0.1 1.0</ambient>
          <diffuse>0.0 1.0 0.1 1.0</diffuse>
          <emissive>0.0 0.7 0.05 1.0</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


class DroneRosBridge(Node):
    """Thin ROS 2 bridge for the topics used by the Gym environment."""

    def __init__(self, namespace: str = "/simple_drone") -> None:
        super().__init__("drone_sonar_avoid_env")
        ns = namespace.rstrip("/")

        self.pose: np.ndarray | None = None
        self.velocity = np.zeros(3, dtype=np.float32)
        self.down_sonar_range: float | None = None
        self.front_sonar_ranges: dict[str, float | None] = {
            "left": None,
            "center": None,
            "right": None,
            "up": None,
            "down": None,
        }
        self.side_sonar_ranges: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self.sonar_min_range = 0.02
        self.sonar_max_range = 10.0
        self.last_pose_time: float | None = None
        self.target_marker_enabled = SpawnEntity is not None and DeleteEntity is not None
        self.target_marker_warning_logged = False
        self.target_marker_spawned = False
        self.reset_world_warning_logged = False

        self.cmd_pub = self.create_publisher(Twist, f"{ns}/cmd_vel", 10)
        # Reset publishes /takeoff before PPO starts controlling each episode.
        self.takeoff_pub = self.create_publisher(Empty, f"{ns}/takeoff", 10)
        self.reset_pub = self.create_publisher(Empty, f"{ns}/reset", 10)
        if self.target_marker_enabled:
            self.spawn_entity_client = self.create_client(SpawnEntity, "/spawn_entity")
            self.delete_entity_client = self.create_client(DeleteEntity, "/delete_entity")
        self.reset_world_client = self.create_client(EmptySrv, "/reset_world")

        self.create_subscription(Pose, f"{ns}/gt_pose", self._pose_cb, 10)
        self.create_subscription(Twist, f"{ns}/gt_vel", self._vel_cb, 10)
        self.create_subscription(Range, f"{ns}/sonar/out", self._down_sonar_cb, 10)
        self.create_subscription(
            Range, f"{ns}/front_sonar_left/out", self._front_sonar_cb("left"), 10
        )
        self.create_subscription(
            Range, f"{ns}/front_sonar_center/out", self._front_sonar_cb("center"), 10
        )
        self.create_subscription(
            Range, f"{ns}/front_sonar_right/out", self._front_sonar_cb("right"), 10
        )
        self.create_subscription(
            Range, f"{ns}/front_sonar_up/out", self._front_sonar_cb("up"), 10
        )
        self.create_subscription(
            Range, f"{ns}/front_sonar_down/out", self._front_sonar_cb("down"), 10
        )
        self.create_subscription(
            Range, f"{ns}/side_sonar_left/out", self._side_sonar_cb("left"), 10
        )
        self.create_subscription(
            Range, f"{ns}/side_sonar_right/out", self._side_sonar_cb("right"), 10
        )

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

    def _down_sonar_cb(self, msg: Range) -> None:
        self.sonar_min_range = float(msg.min_range)
        self.sonar_max_range = float(msg.max_range)
        self.down_sonar_range = float(msg.range)

    def _front_sonar_cb(self, sector: str):
        def callback(msg: Range) -> None:
            self.sonar_min_range = float(msg.min_range)
            self.sonar_max_range = float(msg.max_range)
            self.front_sonar_ranges[sector] = float(msg.range)

        return callback

    def _side_sonar_cb(self, sector: str):
        def callback(msg: Range) -> None:
            self.sonar_min_range = float(msg.min_range)
            self.sonar_max_range = float(msg.max_range)
            self.side_sonar_ranges[sector] = float(msg.range)

        return callback

    def publish_velocity(self, action: np.ndarray) -> None:
        msg = Twist()
        msg.linear.x = float(action[0])
        msg.linear.y = float(action[1])
        msg.linear.z = float(action[2])
        self.cmd_pub.publish(msg)

    def stop(self) -> None:
        self.publish_velocity(np.zeros(3, dtype=np.float32))

    def reset_gazebo_world(self) -> None:
        """Reset Gazebo model poses so each RL episode starts near the origin."""
        if not self.reset_world_client.wait_for_service(timeout_sec=0.5):
            if not self.reset_world_warning_logged:
                self.get_logger().warning(
                    "/reset_world is unavailable; episode reset will only reset controller state"
                )
                self.reset_world_warning_logged = True
            return

        future = self.reset_world_client.call_async(EmptySrv.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        if not future.done() and not self.reset_world_warning_logged:
            self.get_logger().warning("/reset_world did not complete before timeout")
            self.reset_world_warning_logged = True

    def update_target_marker(self, target: np.ndarray) -> None:
        """Show the current RL target in Gazebo as a small green sphere."""
        if not self.target_marker_enabled:
            if not self.target_marker_warning_logged:
                self.get_logger().warning(
                    "gazebo_msgs is unavailable; Gazebo target marker is disabled"
                )
                self.target_marker_warning_logged = True
            return

        if not self.spawn_entity_client.wait_for_service(timeout_sec=0.2):
            if not self.target_marker_warning_logged:
                self.get_logger().warning(
                    "/spawn_entity is unavailable; Gazebo target marker is disabled"
                )
                self.target_marker_warning_logged = True
            return

        if self.target_marker_spawned and self.delete_entity_client.wait_for_service(
            timeout_sec=0.2
        ):
            delete_req = DeleteEntity.Request()
            delete_req.name = TARGET_MARKER_NAME
            delete_future = self.delete_entity_client.call_async(delete_req)
            rclpy.spin_until_future_complete(self, delete_future, timeout_sec=0.5)

        spawn_req = SpawnEntity.Request()
        spawn_req.name = TARGET_MARKER_NAME
        spawn_req.xml = TARGET_MARKER_SDF
        spawn_req.robot_namespace = ""
        spawn_req.reference_frame = "world"
        spawn_req.initial_pose.position.x = float(target[0])
        spawn_req.initial_pose.position.y = float(target[1])
        spawn_req.initial_pose.position.z = float(target[2])
        spawn_req.initial_pose.orientation.w = 1.0
        spawn_future = self.spawn_entity_client.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, spawn_future, timeout_sec=0.8)

        if spawn_future.done() and spawn_future.result() is not None:
            self.target_marker_spawned = True
        elif not self.target_marker_warning_logged:
            self.get_logger().warning("Gazebo target marker spawn did not complete")
            self.target_marker_warning_logged = True

    def reset_and_takeoff(self, takeoff_altitude: float = 0.5, timeout_sec: float = 12.0) -> bool:
        self.stop()
        self.reset_gazebo_world()
        self.pose = None
        self.down_sonar_range = None
        for sector in self.front_sonar_ranges:
            self.front_sonar_ranges[sector] = None
        for sector in self.side_sonar_ranges:
            self.side_sonar_ranges[sector] = None
        self.reset_pub.publish(Empty())
        rclpy.spin_once(self, timeout_sec=0.5)

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.takeoff_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.pose is not None and self.pose[2] >= takeoff_altitude:
                self.stop()
                return True

        if self.pose is not None:
            self.get_logger().warning(
                "Takeoff wait timed out before reaching "
                f"{takeoff_altitude:.2f} m; last_z={self.pose[2]:.2f} m"
            )
        else:
            self.get_logger().warning(
                "Takeoff wait timed out before receiving pose at "
                f"{takeoff_altitude:.2f} m"
            )
        return False


class DroneSonarAvoidEnv(gym.Env):
    """ROS 2 + Gazebo drone environment for sonar-based obstacle avoidance."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        target: tuple[float, float, float] = (5.0, 3.0, 2.0),
        max_steps: int = 400,
        namespace: str = "/simple_drone",
        step_dt: float = 0.1,
        log_position_every: int = 0,
        success_distance: float = 0.4,
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
        self.log_position_every = max(0, int(log_position_every))

        self.target_reached_distance = float(success_distance)
        self.min_altitude = 0.25
        self.max_altitude = 5.0
        self.xy_limit = 8.0
        self.sonar_unsafe_distance = 0.25
        self.down_sonar_lift_distance = 0.35
        self.side_sonar_push_distance = 0.45
        self.sonar_caution_distance = 1.5
        # Obstacle-facing sonar needs a wider caution distance. The downward
        # sonar is only ground-safety, so it should not punish normal low
        # Stage-1 flight at 0.8 m altitude.
        self.down_sonar_caution_distance = 0.45
        # The simulator takeoff helper sometimes stabilizes below 0.8 m or
        # takes longer than expected after reset. PPO does not learn takeoff;
        # reset only needs a reliable airborne start above the crash threshold.
        self.takeoff_altitude = 0.5
        self.max_sonar_range = 10.0
        self.distance_norm = 12.0

        self.step_count = 0
        self.previous_distance: float | None = None
        self.previous_obstacle_sonar = np.full(
            OBSTACLE_SONAR_COUNT,
            self.max_sonar_range,
            dtype=np.float32,
        )
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.recent_obstacle_min = deque(maxlen=10)
        self.last_status = "not_started"
        self.last_observation_info: dict[str, Any] = {}
        self.last_action_was_filtered = False

        # Action: velocity command in m/s, published to /simple_drone/cmd_vel.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -0.5], dtype=np.float32),
            high=np.array([1.0, 1.0, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation:
        # [x/8, y/8, z/5, vx, vy, vz/0.5, dx/8, dy/8, dz/5, distance/12,
        #  seven normalized obstacle ranges, seven obstacle risks,
        #  seven previous normalized obstacle ranges, seven obstacle range trends,
        #  min_recent_obstacle/10, down_sonar/10, down_sonar_risk,
        #  left_right_risk_balance, up_down_risk_balance]
        self.observation_space = spaces.Box(
            low=np.array([-2.5] * OBSERVATION_DIM, dtype=np.float32),
            high=np.array([2.5] * OBSERVATION_DIM, dtype=np.float32),
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
        self.ros.update_target_marker(self.target)

        self.step_count = 0
        self.last_status = "running"
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.last_action_was_filtered = False
        self.recent_obstacle_min.clear()
        self.previous_obstacle_sonar = self._safe_obstacle_sonar_ranges()

        for attempt in range(3):
            takeoff_ok = self.ros.reset_and_takeoff(takeoff_altitude=self.takeoff_altitude)
            self._wait_for_initial_state(min_altitude=self.takeoff_altitude)
            if takeoff_ok and self.ros.pose is not None and self.ros.pose[2] >= self.min_altitude:
                break
            self.ros.get_logger().warning(
                f"Retrying reset/takeoff after low start attempt {attempt + 1}/3"
            )

        obs = self._get_obs()
        self._log_position_if_needed(force=True)
        self.previous_distance = float(self.last_observation_info["distance_to_target"])
        return obs, self._info(obs)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        filtered_action, action_was_filtered = self._apply_safety_filter(action)
        self.last_action_was_filtered = action_was_filtered

        self.ros.publish_velocity(filtered_action)
        rclpy.spin_once(self.ros, timeout_sec=self.step_dt)
        self.step_count += 1

        obs = self._get_obs()
        self._log_position_if_needed()
        info = self.last_observation_info
        current_distance = float(info["distance_to_target"])
        down_sonar_range = float(info["down_sonar_range"])
        min_front_range = float(info["min_front_sonar_range"])
        side_left_range = float(info["side_sonar_left"])
        side_right_range = float(info["side_sonar_right"])
        obstacle_mean_risk = float(info["obstacle_mean_risk"])
        obstacle_max_risk = float(info["obstacle_max_risk"])
        max_approach_trend = max(0.0, float(info["max_obstacle_approach_trend"]))
        down_sonar_risk = float(info["down_sonar_risk"])
        x_pos = float(info["x"])
        y_pos = float(info["y"])
        z_pos = float(info["z"])
        reward_distance = current_distance if math.isfinite(current_distance) else self.distance_norm

        progress_reward = 0.0
        if self.previous_distance is not None and math.isfinite(current_distance):
            progress_reward = 8.0 * (self.previous_distance - current_distance)
        self.previous_distance = current_distance

        direction_reward = 0.0
        if math.isfinite(current_distance) and current_distance > 1e-6:
            target_direction = (self.target - np.array([x_pos, y_pos, z_pos])) / current_distance
            command_alignment = float(np.dot(filtered_action, target_direction))
            direction_reward = 0.10 * float(np.clip(command_alignment, -1.0, 1.0))

        distance_penalty = 0.02 * reward_distance
        near_target_precision_penalty = 0.0
        near_target_velocity_penalty = 0.0
        if math.isfinite(current_distance) and current_distance < 1.0:
            near_target_precision_penalty = 0.25 * current_distance
            near_target_velocity_penalty = 0.05 * float(np.linalg.norm(self.ros.velocity))
        mean_risk_penalty = 2.0 * obstacle_mean_risk**2
        max_risk_penalty = 4.0 * obstacle_max_risk**2
        trend_penalty = 1.5 * max_approach_trend
        down_risk_penalty = 1.0 * down_sonar_risk**2
        action_penalty = 0.01 * float(np.linalg.norm(filtered_action))
        smoothness_penalty = 0.02 * float(np.linalg.norm(filtered_action - self.previous_action))
        filter_penalty = 0.25 if action_was_filtered else 0.0

        reward = (
            progress_reward
            + direction_reward
            - distance_penalty
            - near_target_precision_penalty
            - near_target_velocity_penalty
            - mean_risk_penalty
            - max_risk_penalty
            - trend_penalty
            - down_risk_penalty
            - action_penalty
            - smoothness_penalty
            - filter_penalty
        )
        self.previous_action = filtered_action.copy()

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
        elif z_pos < self.min_altitude:
            reward -= 50.0
            terminated = True
            status = "crash"
        elif abs(x_pos) > self.xy_limit or abs(y_pos) > self.xy_limit or z_pos > self.max_altitude:
            reward -= 50.0
            terminated = True
            status = "out_of_bounds"
        elif min_front_range < self.sonar_unsafe_distance:
            reward -= 50.0
            terminated = True
            status = "unsafe_front_sonar"
        elif (
            side_left_range < self.sonar_unsafe_distance
            or side_right_range < self.sonar_unsafe_distance
        ):
            reward -= 50.0
            terminated = True
            status = "unsafe_side_sonar"
        elif down_sonar_range < self.sonar_unsafe_distance:
            reward -= 50.0
            terminated = True
            status = "unsafe_down_sonar"
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
            sonar_ready = self.ros.down_sonar_range is not None
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

        down_sonar_range = self._safe_sonar_range(self.ros.down_sonar_range)
        obstacle_sonar = self._safe_obstacle_sonar_ranges()
        obstacle_norm = self._normalize_ranges(obstacle_sonar)
        previous_obstacle_norm = self._normalize_ranges(self.previous_obstacle_sonar)
        obstacle_risk = self._ranges_to_risk(obstacle_sonar)
        obstacle_trend = previous_obstacle_norm - obstacle_norm
        down_norm = self._normalize_range(down_sonar_range)
        down_sonar_risk = float(self._down_range_to_risk(down_sonar_range))

        front_sonar = obstacle_sonar[:5]
        front_risk = obstacle_risk[:5]
        front_trend = obstacle_trend[:5]
        side_left_risk = float(obstacle_risk[5])
        side_right_risk = float(obstacle_risk[6])
        min_front = float(np.min(front_sonar))
        min_obstacle = float(np.min(obstacle_sonar))
        self.recent_obstacle_min.append(min_obstacle)
        min_recent_obstacle = (
            min(self.recent_obstacle_min) if self.recent_obstacle_min else min_obstacle
        )
        min_recent_norm = self._normalize_range(min_recent_obstacle)
        left_risk = max(float(front_risk[0]), side_left_risk)
        right_risk = max(float(front_risk[2]), side_right_risk)
        left_right_risk_balance = left_risk - right_risk
        up_down_risk_balance = float(front_risk[3] - front_risk[4])

        obs = np.concatenate(
            [
                np.array(
                    [
                        pose[0] / self.xy_limit,
                        pose[1] / self.xy_limit,
                        pose[2] / self.max_altitude,
                        velocity[0],
                        velocity[1],
                        velocity[2] / 0.5,
                        delta[0] / self.xy_limit,
                        delta[1] / self.xy_limit,
                        delta[2] / self.max_altitude,
                        distance / self.distance_norm,
                    ],
                    dtype=np.float32,
                ),
                obstacle_norm,
                obstacle_risk,
                previous_obstacle_norm,
                obstacle_trend,
                np.array(
                    [
                        min_recent_norm,
                        down_norm,
                        down_sonar_risk,
                        left_right_risk_balance,
                        up_down_risk_balance,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

        self.last_observation_info = {
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
            "distance_to_target": distance,
            "down_sonar_range": down_sonar_range,
            "down_sonar_risk": down_sonar_risk,
            "front_sonar_left": float(front_sonar[0]),
            "front_sonar_center": float(front_sonar[1]),
            "front_sonar_right": float(front_sonar[2]),
            "front_sonar_up": float(front_sonar[3]),
            "front_sonar_down": float(front_sonar[4]),
            "side_sonar_left": float(obstacle_sonar[5]),
            "side_sonar_right": float(obstacle_sonar[6]),
            "front_risk_left": float(front_risk[0]),
            "front_risk_center": float(front_risk[1]),
            "front_risk_right": float(front_risk[2]),
            "front_risk_up": float(front_risk[3]),
            "front_risk_down": float(front_risk[4]),
            "side_left_risk": side_left_risk,
            "side_right_risk": side_right_risk,
            "min_front_sonar_range": min_front,
            "min_obstacle_sonar_range": min_obstacle,
            "min_recent_front_sonar_range": min_recent_obstacle,
            "min_recent_obstacle_sonar_range": min_recent_obstacle,
            "mean_front_risk": float(np.mean(front_risk)),
            "max_front_risk": float(np.max(front_risk)),
            "max_front_approach_trend": float(np.max(front_trend)),
            "obstacle_mean_risk": float(np.mean(obstacle_risk)),
            "obstacle_max_risk": float(np.max(obstacle_risk)),
            "max_obstacle_approach_trend": float(np.max(obstacle_trend)),
            "left_right_risk_balance": left_right_risk_balance,
            "up_down_risk_balance": up_down_risk_balance,
        }
        self.previous_obstacle_sonar = obstacle_sonar
        return obs

    def _safe_sonar_range(self, raw: float | None) -> float:
        max_range = max(min(self.ros.sonar_max_range, self.max_sonar_range), 0.1)
        if raw is None or not math.isfinite(raw):
            return max_range
        return float(np.clip(raw, self.ros.sonar_min_range, max_range))

    def _normalize_range(self, sonar_range: float) -> float:
        return float(np.clip(sonar_range / self.max_sonar_range, 0.0, 1.0))

    def _normalize_ranges(self, sonar_ranges: np.ndarray) -> np.ndarray:
        return np.clip(sonar_ranges / self.max_sonar_range, 0.0, 1.0).astype(np.float32)

    def _range_to_risk(self, sonar_range: float) -> float:
        risk = (self.sonar_caution_distance - sonar_range) / self.sonar_caution_distance
        return float(np.clip(risk, 0.0, 1.0))

    def _down_range_to_risk(self, sonar_range: float) -> float:
        risk = (
            self.down_sonar_caution_distance - sonar_range
        ) / self.down_sonar_caution_distance
        return float(np.clip(risk, 0.0, 1.0))

    def _ranges_to_risk(self, sonar_ranges: np.ndarray) -> np.ndarray:
        risk = (self.sonar_caution_distance - sonar_ranges) / self.sonar_caution_distance
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def _safe_front_sonar_ranges(self) -> np.ndarray:
        return np.array(
            [
                self._safe_sonar_range(self.ros.front_sonar_ranges["left"]),
                self._safe_sonar_range(self.ros.front_sonar_ranges["center"]),
                self._safe_sonar_range(self.ros.front_sonar_ranges["right"]),
                self._safe_sonar_range(self.ros.front_sonar_ranges["up"]),
                self._safe_sonar_range(self.ros.front_sonar_ranges["down"]),
            ],
            dtype=np.float32,
        )

    def _safe_side_sonar_ranges(self) -> np.ndarray:
        return np.array(
            [
                self._safe_sonar_range(self.ros.side_sonar_ranges["left"]),
                self._safe_sonar_range(self.ros.side_sonar_ranges["right"]),
            ],
            dtype=np.float32,
        )

    def _safe_obstacle_sonar_ranges(self) -> np.ndarray:
        return np.concatenate(
            [self._safe_front_sonar_ranges(), self._safe_side_sonar_ranges()]
        ).astype(np.float32)

    def _apply_safety_filter(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        filtered = action.copy()
        front_sonar = self._safe_front_sonar_ranges()
        side_sonar = self._safe_side_sonar_ranges()
        front_risk = self._ranges_to_risk(front_sonar)
        min_front = float(np.min(front_sonar))
        side_left_range = float(side_sonar[0])
        side_right_range = float(side_sonar[1])
        down_sonar_range = self._safe_sonar_range(self.ros.down_sonar_range)
        was_filtered = False

        if min_front < self.sonar_unsafe_distance:
            filtered[0] = min(filtered[0], -0.2)
            filtered[2] = max(filtered[2], 0.2)
            was_filtered = True
        elif front_risk[1] > 0.6 or front_risk[4] > 0.6:
            filtered[0] = min(filtered[0], 0.0)
            was_filtered = True

        if down_sonar_range < self.down_sonar_lift_distance:
            filtered[2] = max(filtered[2], 0.2)
            was_filtered = True

        left_too_close = side_left_range < self.side_sonar_push_distance
        right_too_close = side_right_range < self.side_sonar_push_distance
        if left_too_close and right_too_close:
            filtered[1] = 0.0
            was_filtered = True
        elif left_too_close:
            # Teleop uses +y for left and -y for right, so push right.
            filtered[1] = min(filtered[1], -0.2)
            was_filtered = True
        elif right_too_close:
            # Teleop uses +y for left and -y for right, so push left.
            filtered[1] = max(filtered[1], 0.2)
            was_filtered = True

        return np.clip(filtered, self.action_space.low, self.action_space.high), was_filtered

    def _sensor_state_valid(self, obs: np.ndarray) -> bool:
        return bool(np.all(np.isfinite(obs)))

    def _log_position_if_needed(self, force: bool = False) -> None:
        if not force and (
            self.log_position_every <= 0
            or self.step_count % self.log_position_every != 0
        ):
            return
        info = self.last_observation_info
        if not info:
            return
        print(
            "[pose] "
            f"step={self.step_count} "
            f"pos=({info['x']:.2f}, {info['y']:.2f}, {info['z']:.2f}) "
            f"target=({self.target[0]:.2f}, {self.target[1]:.2f}, {self.target[2]:.2f}) "
            f"distance={info['distance_to_target']:.2f}",
            flush=True,
        )

    def _info(self, obs: np.ndarray, status: str | None = None) -> dict[str, Any]:
        info = dict(self.last_observation_info)
        info.update(
            {
                "status": status or self.last_status,
                "step_count": self.step_count,
                "target": self.target.copy(),
                "action_was_filtered": self.last_action_was_filtered,
            }
        )
        return {
            key: (float(value) if isinstance(value, np.floating) else value)
            for key, value in info.items()
        }
