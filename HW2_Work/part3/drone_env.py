#!/usr/bin/env python3
"""Clean six-stage Gymnasium environment for NSYSU drone PPO.

Part 3 keeps one fixed observation/action interface across all stages. Sonar
fields exist from Stage 1, but they are masked to safe constants until Stage 4
so early checkpoints can continue training in obstacle stages.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
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
except ImportError:  # pragma: no cover - available in the ROS/Gazebo container
    DeleteEntity = None
    SpawnEntity = None


SONAR_SECTORS = (
    "front_left",
    "front_center",
    "front_right",
    "front_up",
    "front_down",
    "side_left",
    "side_right",
)
SONAR_COUNT = len(SONAR_SECTORS)
OBSERVATION_DIM = 12 + (4 * SONAR_COUNT) + 1
TARGET_MARKER_NAME = "part3_rl_target_marker"
TARGET_MARKER_SDF = """
<sdf version="1.6">
  <model name="part3_rl_target_marker">
    <static>true</static>
    <link name="target_link">
      <visual name="target_visual">
        <geometry><sphere><radius>0.18</radius></sphere></geometry>
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


@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    fixed_targets: tuple[tuple[float, float, float], ...]
    x_bounds: tuple[float, float] | None = None
    z_bounds: tuple[float, float] | None = None
    sequence_count: int = 1
    sonar_enabled: bool = False
    focus: str = "navigation"


STAGE_SPECS: dict[tuple[int, str], StageSpec] = {
    (1, "A"): StageSpec(
        name="stage1A_fixed_vertical",
        description="fixed altitude target on Gazebo z",
        fixed_targets=((0.0, 0.0, 1.2),),
        focus="vertical",
    ),
    (1, "B"): StageSpec(
        name="stage1B_random_vertical",
        description="random altitude target on Gazebo z",
        fixed_targets=((0.0, 0.0, 1.2),),
        z_bounds=(0.7, 1.8),
        focus="vertical",
    ),
    (2, "A"): StageSpec(
        name="stage2A_fixed_horizontal",
        description="fixed x target with stable altitude",
        fixed_targets=((1.0, 0.0, 0.8),),
        focus="horizontal",
    ),
    (2, "B"): StageSpec(
        name="stage2B_random_horizontal",
        description="random x target with stable altitude",
        fixed_targets=((1.0, 0.0, 0.8),),
        x_bounds=(-1.0, 2.0),
        focus="horizontal",
    ),
    (3, "A"): StageSpec(
        name="stage3A_random_xz",
        description="single random x/z target without sonar",
        fixed_targets=((1.0, 0.0, 1.0),),
        x_bounds=(-1.0, 2.5),
        z_bounds=(0.7, 1.8),
        focus="combined",
    ),
    (3, "B"): StageSpec(
        name="stage3B_sequence_xz",
        description="three sequential random x/z targets without sonar",
        fixed_targets=((1.0, 0.0, 1.0),),
        x_bounds=(-1.0, 2.5),
        z_bounds=(0.7, 1.8),
        sequence_count=3,
        focus="combined",
    ),
    (4, "A"): StageSpec(
        name="stage4_single_obstacle",
        description="single-obstacle sonar avoidance",
        fixed_targets=((2.5, 0.0, 0.9),),
        sonar_enabled=True,
        focus="obstacle",
    ),
    (5, "A"): StageSpec(
        name="stage5_multi_obstacle",
        description="multi-obstacle sonar avoidance",
        fixed_targets=((3.0, 0.8, 1.0),),
        x_bounds=(1.5, 3.5),
        z_bounds=(0.8, 1.6),
        sonar_enabled=True,
        focus="obstacle",
    ),
    (6, "A"): StageSpec(
        name="stage6_sequence_obstacle",
        description="sequential targets with active sonar avoidance",
        fixed_targets=((1.5, -0.8, 1.0), (2.4, 0.8, 1.1), (3.0, 0.0, 1.0)),
        sequence_count=3,
        sonar_enabled=True,
        focus="mission",
    ),
}


def normalize_variant(stage: int, variant: str) -> str:
    variant = variant.upper()
    if stage >= 4:
        return "A"
    if variant not in ("A", "B"):
        raise ValueError("variant must be A or B")
    return variant


def get_stage_spec(stage: int, variant: str) -> StageSpec:
    key = (stage, normalize_variant(stage, variant))
    if key not in STAGE_SPECS:
        raise ValueError(f"Unsupported stage/variant: stage={stage}, variant={variant}")
    return STAGE_SPECS[key]


class DroneRosBridge(Node):
    """ROS 2 bridge for pose, velocity, reset/takeoff, target marker, and sonar."""

    def __init__(self, namespace: str = "/simple_drone") -> None:
        super().__init__("part3_drone_curriculum_env")
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
        self.side_sonar_ranges: dict[str, float | None] = {"left": None, "right": None}
        self.sonar_min_range = 0.02
        self.sonar_max_range = 10.0
        self.target_marker_enabled = SpawnEntity is not None and DeleteEntity is not None
        self.target_marker_spawned = False
        self.target_marker_warning_logged = False
        self.reset_world_warning_logged = False

        self.cmd_pub = self.create_publisher(Twist, f"{ns}/cmd_vel", 10)
        self.takeoff_pub = self.create_publisher(Empty, f"{ns}/takeoff", 10)
        self.land_pub = self.create_publisher(Empty, f"{ns}/land", 10)
        self.reset_pub = self.create_publisher(Empty, f"{ns}/reset", 10)
        self.reset_world_client = self.create_client(EmptySrv, "/reset_world")
        if self.target_marker_enabled:
            self.spawn_entity_client = self.create_client(SpawnEntity, "/spawn_entity")
            self.delete_entity_client = self.create_client(DeleteEntity, "/delete_entity")

        self.create_subscription(Pose, f"{ns}/gt_pose", self._pose_cb, 10)
        self.create_subscription(Twist, f"{ns}/gt_vel", self._vel_cb, 10)
        self.create_subscription(Range, f"{ns}/sonar/out", self._down_sonar_cb, 10)
        self.create_subscription(Range, f"{ns}/front_sonar_left/out", self._front_cb("left"), 10)
        self.create_subscription(Range, f"{ns}/front_sonar_center/out", self._front_cb("center"), 10)
        self.create_subscription(Range, f"{ns}/front_sonar_right/out", self._front_cb("right"), 10)
        self.create_subscription(Range, f"{ns}/front_sonar_up/out", self._front_cb("up"), 10)
        self.create_subscription(Range, f"{ns}/front_sonar_down/out", self._front_cb("down"), 10)
        self.create_subscription(Range, f"{ns}/side_sonar_left/out", self._side_cb("left"), 10)
        self.create_subscription(Range, f"{ns}/side_sonar_right/out", self._side_cb("right"), 10)

    def _pose_cb(self, msg: Pose) -> None:
        self.pose = np.array([msg.position.x, msg.position.y, msg.position.z], dtype=np.float32)

    def _vel_cb(self, msg: Twist) -> None:
        self.velocity = np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float32)

    def _down_sonar_cb(self, msg: Range) -> None:
        self.sonar_min_range = float(msg.min_range)
        self.sonar_max_range = float(msg.max_range)
        self.down_sonar_range = float(msg.range)

    def _front_cb(self, sector: str):
        def callback(msg: Range) -> None:
            self.sonar_min_range = float(msg.min_range)
            self.sonar_max_range = float(msg.max_range)
            self.front_sonar_ranges[sector] = float(msg.range)

        return callback

    def _side_cb(self, sector: str):
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

    def reset_world(self) -> None:
        if not self.reset_world_client.wait_for_service(timeout_sec=0.5):
            if not self.reset_world_warning_logged:
                self.get_logger().warning("/reset_world unavailable; using topic reset only")
                self.reset_world_warning_logged = True
            return
        future = self.reset_world_client.call_async(EmptySrv.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

    def reset_and_takeoff(self, takeoff_altitude: float = 0.5, timeout_sec: float = 12.0) -> bool:
        self.stop()
        self.reset_world()
        self.pose = None
        self.down_sonar_range = None
        for key in self.front_sonar_ranges:
            self.front_sonar_ranges[key] = None
        for key in self.side_sonar_ranges:
            self.side_sonar_ranges[key] = None

        for _ in range(3):
            self.reset_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.2)

        land_deadline = time.monotonic() + 1.3
        while time.monotonic() < land_deadline:
            self.land_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.1)

        for _ in range(3):
            self.reset_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.2)

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            self.takeoff_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.pose is not None and self.pose[2] >= takeoff_altitude:
                self.stop()
                return True
        return False

    def update_target_marker(self, target: np.ndarray) -> None:
        if not self.target_marker_enabled:
            return
        if not self.spawn_entity_client.wait_for_service(timeout_sec=0.2):
            return
        if self.target_marker_spawned and self.delete_entity_client.wait_for_service(timeout_sec=0.2):
            req = DeleteEntity.Request()
            req.name = TARGET_MARKER_NAME
            future = self.delete_entity_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
        req = SpawnEntity.Request()
        req.name = TARGET_MARKER_NAME
        req.xml = TARGET_MARKER_SDF
        req.reference_frame = "world"
        req.initial_pose.position.x = float(target[0])
        req.initial_pose.position.y = float(target[1])
        req.initial_pose.position.z = float(target[2])
        req.initial_pose.orientation.w = 1.0
        future = self.spawn_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.8)
        self.target_marker_spawned = future.done() and future.result() is not None


class DroneCurriculumEnv(gym.Env):
    """Six-stage drone curriculum with masked sonar before obstacle stages."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        stage: int = 1,
        variant: str = "A",
        target_override: tuple[float, float, float] | None = None,
        max_steps: int = 800,
        success_distance: float = 0.15,
        namespace: str = "/simple_drone",
        step_dt: float = 0.1,
        log_position_every: int = 0,
    ) -> None:
        super().__init__()
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        self.stage = int(stage)
        self.variant = normalize_variant(self.stage, variant)
        self.stage_spec = get_stage_spec(self.stage, self.variant)
        self.target_override = target_override
        self.max_steps = int(max_steps)
        self.success_distance = float(success_distance)
        self.step_dt = float(step_dt)
        self.log_position_every = max(0, int(log_position_every))
        self.ros = DroneRosBridge(namespace=namespace)

        self.xy_limit = 8.0
        self.max_altitude = 5.0
        self.min_altitude = 0.25
        self.takeoff_altitude = 0.5
        self.max_sonar_range = 10.0
        self.sonar_caution_distance = 1.5
        self.sonar_unsafe_distance = 0.25
        self.near_miss_distance = 0.5

        self.targets = np.zeros((1, 3), dtype=np.float32)
        self.target_index = 0
        self.step_count = 0
        self.previous_distance: float | None = None
        self.previous_abs_error: np.ndarray | None = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_sonar = np.full(SONAR_COUNT, self.max_sonar_range, dtype=np.float32)
        self.last_info: dict[str, Any] = {}
        self.last_action_was_filtered = False
        self.targets_reached = 0

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -0.5], dtype=np.float32),
            high=np.array([1.0, 1.0, 0.5], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-3.0] * OBSERVATION_DIM, dtype=np.float32),
            high=np.array([3.0] * OBSERVATION_DIM, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def sonar_enabled(self) -> bool:
        return bool(self.stage_spec.sonar_enabled)

    @property
    def current_target(self) -> np.ndarray:
        return self.targets[self.target_index]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if options and "target" in options:
            self.target_override = tuple(float(v) for v in options["target"])
        self.targets = self._sample_targets()
        self.target_index = 0
        self.targets_reached = 0
        self.step_count = 0
        self.previous_distance = None
        self.previous_abs_error = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_sonar = np.full(SONAR_COUNT, self.max_sonar_range, dtype=np.float32)
        self.last_action_was_filtered = False
        self.ros.update_target_marker(self.current_target)

        takeoff_ok = False
        for attempt in range(3):
            takeoff_ok = self.ros.reset_and_takeoff(self.takeoff_altitude)
            self._wait_for_state(min_altitude=self.takeoff_altitude)
            if takeoff_ok and self.ros.pose is not None and self.ros.pose[2] >= self.min_altitude:
                break
            self.ros.get_logger().warning(f"Retrying Part 3 reset/takeoff {attempt + 1}/3")
        if not takeoff_ok or self.ros.pose is None:
            raise RuntimeError("Part 3 reset/takeoff failed; restart Gazebo and try again.")

        obs = self._get_obs()
        self.previous_distance = float(self.last_info["distance_to_target"])
        self.previous_abs_error = self._abs_error()
        self._log_position(force=True)
        return obs, self._info("running")

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        filtered_action, was_filtered = self._apply_safety_filter(action)
        self.last_action_was_filtered = was_filtered
        self.ros.publish_velocity(filtered_action)
        self._spin_for_step()
        self.step_count += 1

        obs = self._get_obs()
        self._log_position()
        info = self.last_info
        distance = float(info["distance_to_target"])
        dx = float(info["dx"])
        dy = float(info["dy"])
        dz = float(info["dz"])
        x_error = abs(dx)
        y_error = abs(dy)
        z_error = abs(dz)
        target_reached = self._target_reached(x_error, y_error, z_error, distance)
        velocity_norm = float(np.linalg.norm(self.ros.velocity))

        reward = 0.0
        if self.previous_distance is not None and math.isfinite(distance):
            scale = 10.0 if distance >= 0.5 else 18.0
            reward += scale * (self.previous_distance - distance)
        self.previous_distance = distance

        current_abs_error = np.array([x_error, y_error, z_error], dtype=np.float32)
        if self.previous_abs_error is not None:
            delta = self.previous_abs_error - current_abs_error
            reward += self._axis_progress_reward(delta)
        self.previous_abs_error = current_abs_error.copy()

        reward -= 0.05 * distance
        reward -= self._stage_precision_penalty(x_error, y_error, z_error)
        if distance < 0.6:
            reward -= 0.12 * velocity_norm
            reward -= 0.08 * float(np.linalg.norm(filtered_action))
        reward -= 0.01 * float(np.linalg.norm(filtered_action))
        reward -= 0.02 * float(np.linalg.norm(filtered_action - self.previous_action))
        if was_filtered:
            reward -= 0.25
        self.previous_action = filtered_action.copy()

        obstacle_max_risk = float(info["obstacle_max_risk"])
        obstacle_mean_risk = float(info["obstacle_mean_risk"])
        min_obstacle = float(info["min_obstacle_sonar_range"])
        if self.sonar_enabled:
            reward -= 2.0 * obstacle_mean_risk**2
            reward -= 4.0 * obstacle_max_risk**2

        terminated = False
        truncated = False
        status = "running"
        if not np.all(np.isfinite(obs)):
            reward -= 100.0
            terminated = True
            status = "invalid_sensor"
        elif float(info["z"]) < self.min_altitude:
            reward -= 100.0
            terminated = True
            status = "crash"
        elif abs(float(info["x"])) > self.xy_limit or abs(float(info["y"])) > self.xy_limit:
            reward -= 80.0
            terminated = True
            status = "out_of_bounds"
        elif self.sonar_enabled and min_obstacle < self.sonar_unsafe_distance:
            reward -= 100.0
            terminated = True
            status = "unsafe_sonar"
        elif target_reached:
            reward += 80.0
            self.targets_reached += 1
            if self.target_index + 1 >= len(self.targets):
                terminated = True
                status = "success"
            else:
                reward += 30.0
                self.target_index += 1
                self.previous_distance = None
                self.previous_abs_error = None
                self.ros.update_target_marker(self.current_target)
                status = "target_reached"
        elif self.step_count >= self.max_steps:
            reward -= 5.0 + 20.0 * min(distance, 2.0)
            truncated = True
            status = "timeout"

        if status == "timeout":
            self._log_position(force=True)
        if terminated or truncated:
            self.ros.stop()
        return obs, float(reward), terminated, truncated, self._info(status)

    def close(self) -> None:
        self.ros.stop()
        self.ros.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def _sample_targets(self) -> np.ndarray:
        if self.target_override is not None:
            return np.array([self.target_override], dtype=np.float32)
        if self.stage_spec.sequence_count > 1:
            targets = [self._sample_one_target(index) for index in range(self.stage_spec.sequence_count)]
            return np.array(targets, dtype=np.float32)
        return np.array([self._sample_one_target(0)], dtype=np.float32)

    def _sample_one_target(self, index: int) -> tuple[float, float, float]:
        if self.stage_spec.x_bounds is None and self.stage_spec.z_bounds is None:
            fixed = self.stage_spec.fixed_targets[min(index, len(self.stage_spec.fixed_targets) - 1)]
            return tuple(float(v) for v in fixed)
        base = self.stage_spec.fixed_targets[0]
        x = random.uniform(*self.stage_spec.x_bounds) if self.stage_spec.x_bounds else base[0]
        z = random.uniform(*self.stage_spec.z_bounds) if self.stage_spec.z_bounds else base[2]
        y = 0.0
        if self.stage >= 6:
            y = random.uniform(-1.0, 1.0)
        return (float(x), float(y), float(z))

    def _axis_progress_reward(self, delta: np.ndarray) -> float:
        if self.stage_spec.focus == "vertical":
            weights = np.array([2.0, 2.0, 12.0], dtype=np.float32)
        elif self.stage_spec.focus == "horizontal":
            weights = np.array([12.0, 3.0, 6.0], dtype=np.float32)
        else:
            weights = np.array([9.0, 4.0, 7.0], dtype=np.float32)
        return float(np.dot(weights, delta))

    def _stage_precision_penalty(self, x_error: float, y_error: float, z_error: float) -> float:
        if self.stage_spec.focus == "vertical":
            return 0.45 * x_error + 0.45 * y_error + 0.65 * z_error
        if self.stage_spec.focus == "horizontal":
            return 0.45 * x_error + 0.20 * y_error + 0.45 * z_error
        return 0.35 * x_error + 0.25 * y_error + 0.35 * z_error

    def _target_reached(self, x_error: float, y_error: float, z_error: float, distance: float) -> bool:
        if self.stage_spec.focus == "vertical":
            lateral_error = math.hypot(x_error, y_error)
            lateral_tolerance = max(0.20, 1.5 * self.success_distance)
            return z_error < self.success_distance and lateral_error < lateral_tolerance
        return distance < self.success_distance

    def _wait_for_state(self, timeout_sec: float = 5.0, min_altitude: float | None = None) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self.ros, timeout_sec=0.1)
            if self.ros.pose is None:
                continue
            if min_altitude is not None and self.ros.pose[2] < min_altitude:
                continue
            return

    def _spin_for_step(self) -> None:
        """Hold each action for the configured control period."""
        deadline = time.monotonic() + self.step_dt
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            rclpy.spin_once(self.ros, timeout_sec=min(0.02, remaining))

    def _get_obs(self) -> np.ndarray:
        pose = self.ros.pose
        if pose is None:
            pose = np.full(3, np.nan, dtype=np.float32)
        velocity = self.ros.velocity.astype(np.float32)
        target = self.current_target
        delta = target - pose
        distance = float(np.linalg.norm(delta)) if np.all(np.isfinite(delta)) else math.nan

        sonar = self._safe_sonar_ranges()
        sonar_norm = np.clip(sonar / self.max_sonar_range, 0.0, 1.0).astype(np.float32)
        prev_sonar_norm = np.clip(self.previous_sonar / self.max_sonar_range, 0.0, 1.0)
        if self.sonar_enabled:
            sonar_risk = self._ranges_to_risk(sonar)
            sonar_trend = prev_sonar_norm - sonar_norm
            sonar_enabled = 1.0
        else:
            sonar_norm = np.ones(SONAR_COUNT, dtype=np.float32)
            prev_sonar_norm = np.ones(SONAR_COUNT, dtype=np.float32)
            sonar_risk = np.zeros(SONAR_COUNT, dtype=np.float32)
            sonar_trend = np.zeros(SONAR_COUNT, dtype=np.float32)
            sonar_enabled = 0.0

        total_targets = max(len(self.targets), 1)
        target_progress = self.target_index / max(total_targets - 1, 1)
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
                        delta[0] / 3.0,
                        delta[1] / 3.0,
                        delta[2] / 1.5,
                        distance / 4.0,
                        target_progress,
                        total_targets / 3.0,
                    ],
                    dtype=np.float32,
                ),
                sonar_norm,
                sonar_risk,
                prev_sonar_norm,
                sonar_trend,
                np.array([sonar_enabled], dtype=np.float32),
            ]
        ).astype(np.float32)

        min_obstacle = float(np.min(sonar)) if self.sonar_enabled else self.max_sonar_range
        self.last_info = {
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
            "vx": float(velocity[0]),
            "vy": float(velocity[1]),
            "vz": float(velocity[2]),
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "dz": float(delta[2]),
            "distance_to_target": distance,
            "target_index": self.target_index,
            "total_targets": total_targets,
            "targets_reached": self.targets_reached,
            "sonar_enabled": sonar_enabled,
            "min_obstacle_sonar_range": min_obstacle,
            "obstacle_mean_risk": float(np.mean(sonar_risk)),
            "obstacle_max_risk": float(np.max(sonar_risk)),
            "side_sonar_left": float(sonar[5]),
            "side_sonar_right": float(sonar[6]),
        }
        self.previous_sonar = sonar
        return obs

    def _abs_error(self) -> np.ndarray | None:
        if self.ros.pose is None:
            return None
        return np.abs(self.current_target - self.ros.pose).astype(np.float32)

    def _safe_sonar(self, raw: float | None) -> float:
        max_range = max(min(self.ros.sonar_max_range, self.max_sonar_range), 0.1)
        if raw is None or not math.isfinite(raw):
            return max_range
        return float(np.clip(raw, self.ros.sonar_min_range, max_range))

    def _safe_sonar_ranges(self) -> np.ndarray:
        return np.array(
            [
                self._safe_sonar(self.ros.front_sonar_ranges["left"]),
                self._safe_sonar(self.ros.front_sonar_ranges["center"]),
                self._safe_sonar(self.ros.front_sonar_ranges["right"]),
                self._safe_sonar(self.ros.front_sonar_ranges["up"]),
                self._safe_sonar(self.ros.front_sonar_ranges["down"]),
                self._safe_sonar(self.ros.side_sonar_ranges["left"]),
                self._safe_sonar(self.ros.side_sonar_ranges["right"]),
            ],
            dtype=np.float32,
        )

    def _ranges_to_risk(self, ranges: np.ndarray) -> np.ndarray:
        risk = (self.sonar_caution_distance - ranges) / self.sonar_caution_distance
        return np.clip(risk, 0.0, 1.0).astype(np.float32)

    def _apply_safety_filter(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.sonar_enabled:
            return action.copy(), False
        filtered = action.copy()
        sonar = self._safe_sonar_ranges()
        was_filtered = False
        if float(np.min(sonar[:5])) < 0.45:
            filtered[0] = min(filtered[0], 0.0)
            filtered[2] = max(filtered[2], 0.1)
            was_filtered = True
        if float(sonar[5]) < 0.45:
            filtered[1] = min(filtered[1], -0.2)
            was_filtered = True
        if float(sonar[6]) < 0.45:
            filtered[1] = max(filtered[1], 0.2)
            was_filtered = True
        return np.clip(filtered, self.action_space.low, self.action_space.high), was_filtered

    def _log_position(self, force: bool = False) -> None:
        if not force and (
            self.log_position_every <= 0
            or self.step_count % self.log_position_every != 0
        ):
            return
        info = self.last_info
        if not info:
            return
        target = self.current_target
        print(
            "[pose] "
            f"step={self.step_count} "
            f"target_index={self.target_index + 1}/{len(self.targets)} "
            f"pos=({info['x']:.2f}, {info['y']:.2f}, {info['z']:.2f}) "
            f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) "
            f"distance={info['distance_to_target']:.2f}",
            flush=True,
        )

    def _info(self, status: str) -> dict[str, Any]:
        info = dict(self.last_info)
        info.update(
            {
                "status": status,
                "step_count": self.step_count,
                "stage": self.stage,
                "variant": self.variant,
                "target": self.current_target.copy(),
                "action_was_filtered": self.last_action_was_filtered,
            }
        )
        return {
            key: (float(value) if isinstance(value, np.floating) else value)
            for key, value in info.items()
        }
