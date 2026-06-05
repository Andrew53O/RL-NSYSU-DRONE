#!/usr/bin/env python3
"""Five-stage Gymnasium environment for NSYSU drone PPO with lateral Stage 3.

Part 3+Y keeps one fixed observation/action interface across all stages. Sonar
fields exist from Stage 1, but they are masked to safe constants until Stage 5
so early checkpoints can continue training in obstacle stages.

This copy adds anti-oscillation shaping for Stage 4/5. It is intentionally kept
separate from the normal Part 3+Y environment so the original experiments remain
reproducible.
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
STAGE5_FINAL_GOAL = np.array([10.0, 0.0, 1.0], dtype=np.float32)
STAGE5_GOAL_RADIUS = 10.0
STAGE5_LOCAL_GOAL_STEP = 1.0
GENERATED_OBSTACLE_NAME = "part3_plusy_generated_cone"
TARGET_MARKER_COLORS = (
    (0.0, 1.0, 0.1, 1.0),  # first target: green
    (0.0, 0.25, 1.0, 1.0),  # second target: blue
    (1.0, 0.0, 0.0, 1.0),  # third target: red
    (1.0, 0.85, 0.0, 1.0),
)
TARGET_MARKER_EMISSIVE_SCALE = 0.7
TARGET_MARKER_SDF_TEMPLATE = """
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="target_link">
      <visual name="target_visual">
        <geometry><sphere><radius>0.18</radius></sphere></geometry>
        <material>
          <ambient>{ambient}</ambient>
          <diffuse>{diffuse}</diffuse>
          <emissive>{emissive}</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
GENERATED_CONE_SDF = """
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <mesh>
            <scale>10 10 10</scale>
            <uri>model://construction_cone/meshes/construction_cone.dae</uri>
          </mesh>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <mesh>
            <scale>10 10 10</scale>
            <uri>model://construction_cone/meshes/construction_cone.dae</uri>
          </mesh>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
"""

# build the ball marker in the world 
def target_marker_sdf(name: str, color: tuple[float, float, float, float]) -> str:
    emissive = (
        color[0] * TARGET_MARKER_EMISSIVE_SCALE,
        color[1] * TARGET_MARKER_EMISSIVE_SCALE,
        color[2] * TARGET_MARKER_EMISSIVE_SCALE,
        color[3],
    )
    return TARGET_MARKER_SDF_TEMPLATE.format(
        name=name,
        ambient=" ".join(f"{value:.3f}" for value in color),
        diffuse=" ".join(f"{value:.3f}" for value in color),
        emissive=" ".join(f"{value:.3f}" for value in emissive),
    )


# Environment Stage configuration specification 
@dataclass(frozen=True)
class StageSpec:
    name: str
    description: str
    fixed_targets: tuple[tuple[float, float, float], ...]
    x_bounds: tuple[float, float] | None = None
    y_bounds: tuple[float, float] | None = None
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
        name="stage3A_fixed_lateral",
        description="fixed sideway y target with stable altitude",
        fixed_targets=((0.0, 1.0, 0.8),),
        focus="lateral",
    ),
    (3, "B"): StageSpec(
        name="stage3B_random_lateral",
        description="random sideway y target with stable altitude",
        fixed_targets=((0.0, 1.0, 0.8),),
        y_bounds=(-1.5, 1.5),
        focus="lateral",
    ),
    (4, "A"): StageSpec(
        name="stage4A_random_xyz",
        description="single random x/y/z target without sonar",
        fixed_targets=((1.0, 0.0, 1.0),),
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        z_bounds=(0.5, 2.0),
        focus="combined",
    ),
    (4, "B"): StageSpec(
        name="stage4B_sequence_xyz",
        description="three sequential random x/y/z targets without sonar",
        fixed_targets=((1.0, 0.0, 1.0),),
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        z_bounds=(0.5, 2.0),
        sequence_count=3,
        focus="combined",
    ),
    (5, "A"): StageSpec(
        name="stage5_single_obstacle",
        description="long-goal single-obstacle sonar avoidance with dynamic local subgoal",
        fixed_targets=((10.0, 0.0, 1.0),),
        sonar_enabled=True,
        focus="obstacle",
    ),
    (5, "B"): StageSpec(
        name="stage5B_random_radial_obstacle",
        description="random radius-10 x/y mission target with generated midpoint cone",
        fixed_targets=((10.0, 0.0, 1.0),),
        sonar_enabled=True,
        focus="obstacle",
    ),
}


def normalize_variant(stage: int, variant: str) -> str:
    variant = variant.upper()
    if stage == 5:
        if variant not in ("A", "B"):
            raise ValueError("stage 5 variant must be A or B")
        return variant
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
        self.target_marker_names: set[str] = set()
        self.generated_obstacle_spawned = False
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

    def _target_marker_name(self, index: int, target_count: int) -> str:
        if target_count == 1:
            return TARGET_MARKER_NAME
        return f"{TARGET_MARKER_NAME}_{index + 1}"

    def _delete_target_marker(self, name: str) -> None:
        if not self.target_marker_enabled:
            return
        if not self.delete_entity_client.wait_for_service(timeout_sec=0.2):
            return
        req = DeleteEntity.Request()
        req.name = name
        future = self.delete_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)

    def clear_target_markers(self) -> None:
        if not self.target_marker_enabled:
            return
        names = set(self.target_marker_names)
        names.add(TARGET_MARKER_NAME)
        for index in range(4):
            names.add(self._target_marker_name(index, 4))
        for name in sorted(names):
            self._delete_target_marker(name)
        self.target_marker_names.clear()
        self.target_marker_spawned = False

    def update_target_markers(self, targets: np.ndarray) -> None:
        if not self.target_marker_enabled:
            return
        if not self.spawn_entity_client.wait_for_service(timeout_sec=0.2):
            return
        self.clear_target_markers()
        target_count = len(targets)
        for index, target in enumerate(targets):
            marker_name = self._target_marker_name(index, target_count)
            color = TARGET_MARKER_COLORS[index % len(TARGET_MARKER_COLORS)]
            if target_count == 2 and index == 1:
                color = TARGET_MARKER_COLORS[2]
            req = SpawnEntity.Request()
            req.name = marker_name
            req.xml = target_marker_sdf(marker_name, color)
            req.reference_frame = "world"
            req.initial_pose.position.x = float(target[0])
            req.initial_pose.position.y = float(target[1])
            req.initial_pose.position.z = float(target[2])
            req.initial_pose.orientation.w = 1.0
            future = self.spawn_entity_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.8)
            if future.done() and future.result() is not None:
                self.target_marker_names.add(marker_name)
        self.target_marker_spawned = bool(self.target_marker_names)

    def update_target_marker(self, target: np.ndarray) -> None:
        self.update_target_markers(np.array([target], dtype=np.float32))

    def clear_generated_obstacle(self) -> None:
        if not self.target_marker_enabled:
            return
        self._delete_target_marker(GENERATED_OBSTACLE_NAME)
        self.generated_obstacle_spawned = False

    def spawn_generated_cone(self, position: np.ndarray) -> None:
        if not self.target_marker_enabled:
            return
        if not self.spawn_entity_client.wait_for_service(timeout_sec=0.5):
            return
        self.clear_generated_obstacle()
        req = SpawnEntity.Request()
        req.name = GENERATED_OBSTACLE_NAME
        req.xml = GENERATED_CONE_SDF.format(name=GENERATED_OBSTACLE_NAME)
        req.reference_frame = "world"
        req.initial_pose.position.x = float(position[0])
        req.initial_pose.position.y = float(position[1])
        req.initial_pose.position.z = float(position[2])
        req.initial_pose.orientation.w = 1.0
        future = self.spawn_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.8)
        self.generated_obstacle_spawned = bool(future.done() and future.result() is not None)


class DroneCurriculumEnv(gym.Env):
    """Five-stage drone curriculum with masked sonar before obstacle stages."""

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

        self.xy_limit = 12.0 if self.stage >= 5 else 8.0
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
        self.previous_final_distance: float | None = None
        self.previous_abs_error: np.ndarray | None = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_sonar = np.full(SONAR_COUNT, self.max_sonar_range, dtype=np.float32)
        self.last_info: dict[str, Any] = {}
        self.last_action_was_filtered = False
        self.targets_reached = 0
        self.stable_success_steps = 0
        self.stable_success_required = 3
        self.stable_success_velocity = 0.22

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
        if self.stage == 5:
            # Stage 5 trains on a moving local subgoal while success is checked
            # against the far mission goal. This keeps the target vector local.
            return self._stage5_local_target()
        return self.targets[self.target_index]

    @property
    def mission_goal(self) -> np.ndarray:
        if self.stage == 5:
            if len(self.targets) > 0:
                return self.targets[-1]
            return STAGE5_FINAL_GOAL
        return self.targets[-1]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if options and "target" in options:
            self.target_override = tuple(float(v) for v in options["target"])
        self.targets = self._sample_targets()
        self.target_index = 0
        self.targets_reached = 0
        self.step_count = 0
        self.previous_distance = None
        self.previous_final_distance = None
        self.previous_abs_error = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.previous_sonar = np.full(SONAR_COUNT, self.max_sonar_range, dtype=np.float32)
        self.last_action_was_filtered = False
        self.stable_success_steps = 0

        takeoff_ok = False
        for attempt in range(3):
            takeoff_ok = self.ros.reset_and_takeoff(self.takeoff_altitude)
            self._wait_for_state(min_altitude=self.takeoff_altitude)
            if takeoff_ok and self.ros.pose is not None and self.ros.pose[2] >= self.min_altitude:
                break
            self.ros.get_logger().warning(f"Retrying Part 3 reset/takeoff {attempt + 1}/3")
        if not takeoff_ok or self.ros.pose is None:
            raise RuntimeError("Part 3 reset/takeoff failed; restart Gazebo and try again.")

        self._update_stage_obstacle()
        obs = self._get_obs()
        self._update_stage_markers(force=True)
        self.previous_distance = float(self.last_info["distance_to_target"])
        self.previous_final_distance = float(self.last_info["mission_goal_distance"])
        self.previous_abs_error = self._abs_error()
        self._log_position(force=True)
        return obs, self._info("running")

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # The safety filter only blocks clearly unsafe commands. The policy is
        # still penalized when the filter intervenes, so it learns avoidance.
        filtered_action, was_filtered = self._apply_safety_filter(action)
        self.last_action_was_filtered = was_filtered

        self.ros.publish_velocity(filtered_action)

        self._spin_for_step()
        self.step_count += 1

        obs = self._get_obs()
        self._log_position()
        info = self.last_info
        distance = float(info["distance_to_target"])
        mission_goal_distance = float(info["mission_goal_distance"])
        dx = float(info["dx"])
        dy = float(info["dy"])
        dz = float(info["dz"])
        x_error = abs(dx)
        y_error = abs(dy)
        z_error = abs(dz)
        velocity_norm = float(np.linalg.norm(self.ros.velocity))
        target_reached = self._target_reached(x_error, y_error, z_error, distance, velocity_norm)

        reward = 0.0
        # Dense progress reward: positive when the active target gets closer.
        if self.previous_distance is not None and math.isfinite(distance):
            scale = 10.0 if distance >= 0.5 else 18.0
            reward += scale * (self.previous_distance - distance)
        self.previous_distance = distance

        # Stage 5 uses a local target, so also reward progress toward the final
        # long-range goal to prevent the drone from hovering around subgoals.
        if self.stage == 5 and self.previous_final_distance is not None:
            reward += 8.0 * (self.previous_final_distance - mission_goal_distance)
        self.previous_final_distance = mission_goal_distance

        # Axis-specific shaping matches the current curriculum focus, e.g.
        # vertical accuracy matters most in Stage 1.
        current_abs_error = np.array([x_error, y_error, z_error], dtype=np.float32)
        if self.previous_abs_error is not None:
            delta = self.previous_abs_error - current_abs_error
            reward += self._axis_progress_reward(delta)
        self.previous_abs_error = current_abs_error.copy()

        # Small penalties discourage drifting, overspeed near the target, and
        # sharp action changes that make Gazebo control unstable.
        reward -= 0.05 * distance
        reward -= self._stage_precision_penalty(x_error, y_error, z_error)
        if distance < 0.6:
            reward -= 0.18 * velocity_norm
            reward -= 0.10 * float(np.linalg.norm(filtered_action))
        if distance < 0.45:
            reward -= self._near_target_motion_penalty(
                distance=distance,
                target_vector=np.array([dx, dy, dz], dtype=np.float32),
                velocity=self.ros.velocity,
            )
        reward -= 0.01 * float(np.linalg.norm(filtered_action))
        reward -= 0.02 * float(np.linalg.norm(filtered_action - self.previous_action))
        if was_filtered:
            reward -= 0.25
        self.previous_action = filtered_action.copy()

        obstacle_max_risk = float(info["obstacle_max_risk"])
        obstacle_mean_risk = float(info["obstacle_mean_risk"])
        min_obstacle = float(info["min_obstacle_sonar_range"])
        if self.sonar_enabled:
            # Sonar risk is zero in free space and approaches one near obstacles.
            reward -= 2.0 * obstacle_mean_risk**2
            reward -= 4.0 * obstacle_max_risk**2
            if self.stage == 5 and obstacle_max_risk > 0.2 and abs(float(info["vx"])) < 0.05:
                reward -= 0.2

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
                # Move to the next waypoint and reset progress baselines so the
                # next step is not punished for the target suddenly changing.
                reward += 30.0
                self.target_index += 1
                self.previous_distance = None
                self.previous_abs_error = None
                self.stable_success_steps = 0
                status = "target_reached"
        elif self.step_count >= self.max_steps:
            reward -= 5.0 + 20.0 * min(distance, 2.0)
            truncated = True
            status = "timeout"
        elif self.stage >= 4 and distance < self.success_distance:
            # The drone entered the target ball but was still moving too fast,
            # so this is a near miss rather than a completed arrival.
            reward -= 0.5 * velocity_norm

        if status in {"success", "timeout"}:
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
        if self.stage == 5:
            if self.variant == "B":
                angle = random.uniform(-math.pi, math.pi)
                return np.array(
                    [
                        (
                            STAGE5_GOAL_RADIUS * math.cos(angle),
                            STAGE5_GOAL_RADIUS * math.sin(angle),
                            1.0,
                        )
                    ],
                    dtype=np.float32,
                )
            return np.array([STAGE5_FINAL_GOAL], dtype=np.float32)
        if self.stage_spec.sequence_count > 1:
            targets = [self._sample_one_target(index) for index in range(self.stage_spec.sequence_count)]
            return np.array(targets, dtype=np.float32)
        return np.array([self._sample_one_target(0)], dtype=np.float32)

    def _stage5_local_target(self) -> np.ndarray:
        mission_goal = self.mission_goal
        if self.ros.pose is None:
            direction = mission_goal.copy()
            direction[2] = 0.0
            norm = float(np.linalg.norm(direction[:2]))
            if norm < 1e-6:
                return mission_goal.copy()
            local_xy = direction[:2] / norm * min(STAGE5_LOCAL_GOAL_STEP, norm)
            return np.array([local_xy[0], local_xy[1], float(mission_goal[2])], dtype=np.float32)
        # The local target advances one meter along the direct vector to the
        # mission goal. Sonar, not a preplanned path, decides any avoidance.
        pose = self.ros.pose.astype(np.float32)
        vector = mission_goal - pose
        distance = float(np.linalg.norm(vector))
        if distance <= STAGE5_LOCAL_GOAL_STEP:
            return mission_goal.copy()
        return (pose + vector / distance * STAGE5_LOCAL_GOAL_STEP).astype(np.float32)

    def _update_stage_obstacle(self) -> None:
        if self.stage == 5 and self.variant == "B" and self.ros.pose is not None:
            midpoint = (self.ros.pose + self.mission_goal) / 2.0
            midpoint[2] = 0.05
            self.ros.spawn_generated_cone(midpoint)
        else:
            self.ros.clear_generated_obstacle()

    def _update_stage_markers(self, force: bool = False) -> None:
        if not force:
            return
        if self.stage == 5:
            self.ros.update_target_marker(self.mission_goal)
        else:
            self.ros.update_target_markers(self.targets)

    def _sample_one_target(self, index: int) -> tuple[float, float, float]:
        if (
            self.stage_spec.x_bounds is None
            and self.stage_spec.y_bounds is None
            and self.stage_spec.z_bounds is None
        ):
            fixed = self.stage_spec.fixed_targets[min(index, len(self.stage_spec.fixed_targets) - 1)]
            return tuple(float(v) for v in fixed)
        base = self.stage_spec.fixed_targets[0]
        x = random.uniform(*self.stage_spec.x_bounds) if self.stage_spec.x_bounds else base[0]
        y = random.uniform(*self.stage_spec.y_bounds) if self.stage_spec.y_bounds else base[1]
        z = random.uniform(*self.stage_spec.z_bounds) if self.stage_spec.z_bounds else base[2]
        return (float(x), float(y), float(z))

    def _axis_progress_reward(self, delta: np.ndarray) -> float:
        if self.stage_spec.focus == "vertical":
            weights = np.array([2.0, 2.0, 12.0], dtype=np.float32)
        elif self.stage_spec.focus == "horizontal":
            weights = np.array([12.0, 3.0, 6.0], dtype=np.float32)
        elif self.stage_spec.focus == "lateral":
            weights = np.array([3.0, 12.0, 6.0], dtype=np.float32)
        else:
            weights = np.array([9.0, 4.0, 7.0], dtype=np.float32)
        return float(np.dot(weights, delta))

    def _stage_precision_penalty(self, x_error: float, y_error: float, z_error: float) -> float:
        if self.stage_spec.focus == "vertical":
            return 0.45 * x_error + 0.45 * y_error + 0.65 * z_error
        if self.stage_spec.focus == "horizontal":
            return 0.45 * x_error + 0.20 * y_error + 0.45 * z_error
        if self.stage_spec.focus == "lateral":
            return 0.20 * x_error + 0.45 * y_error + 0.45 * z_error
        return 0.35 * x_error + 0.35 * y_error + 0.35 * z_error

    def _near_target_motion_penalty(
        self,
        distance: float,
        target_vector: np.ndarray,
        velocity: np.ndarray,
    ) -> float:
        """Penalize orbiting and fly-through behavior near the active target."""
        if distance <= 1e-6 or not math.isfinite(distance):
            return 0.0
        if not np.all(np.isfinite(target_vector)) or not np.all(np.isfinite(velocity)):
            return 0.0

        direction_to_target = target_vector / max(distance, 1e-6)
        radial_speed = float(np.dot(velocity, direction_to_target))
        tangential_velocity = velocity - radial_speed * direction_to_target
        tangential_speed = float(np.linalg.norm(tangential_velocity))
        total_speed = float(np.linalg.norm(velocity))

        penalty = 0.18 * tangential_speed + 0.10 * total_speed
        if radial_speed < -0.03:
            penalty += 0.25 * abs(radial_speed)
        return penalty

    def _target_reached(
        self,
        x_error: float,
        y_error: float,
        z_error: float,
        distance: float,
        velocity_norm: float,
    ) -> bool:
        if self.stage == 5 and self.ros.pose is not None:
            reached = float(np.linalg.norm(self.mission_goal - self.ros.pose)) < self.success_distance
            return self._stable_target_reached(reached, velocity_norm)
        if self.stage_spec.focus == "vertical":
            lateral_error = math.hypot(x_error, y_error)
            lateral_tolerance = max(0.20, 1.5 * self.success_distance)
            return z_error < self.success_distance and lateral_error < lateral_tolerance
        reached = distance < self.success_distance
        if self.stage >= 4:
            return self._stable_target_reached(reached, velocity_norm)
        return reached

    def _stable_target_reached(self, reached: bool, velocity_norm: float) -> bool:
        if reached and velocity_norm <= self.stable_success_velocity:
            self.stable_success_steps += 1
        else:
            self.stable_success_steps = 0
        return self.stable_success_steps >= self.stable_success_required

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
        if self.stage == 5 and pose is not None and math.isfinite(float(pose[0])):
            # Obstacle stages use mission course progress instead of waypoint
            # index progress because the active target is a moving local subgoal.
            mission_distance_for_progress = float(np.linalg.norm(self.mission_goal - pose))
            target_progress = float(
                np.clip(
                    (STAGE5_GOAL_RADIUS - mission_distance_for_progress) / STAGE5_GOAL_RADIUS,
                    0.0,
                    1.0,
                )
            )
        dx_norm = 10.0 if self.stage >= 5 else 3.0
        dy_norm = 5.0 if self.stage >= 5 else 3.0
        distance_norm = 12.0 if self.stage >= 5 else 4.0
        # Keep this layout synchronized with OBSERVATION_DIM and RL-DESIGN.md.
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
                        delta[0] / dx_norm,
                        delta[1] / dy_norm,
                        delta[2] / 1.5,
                        distance / distance_norm,
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
        mission_delta = self.mission_goal - pose
        mission_distance = (
            float(np.linalg.norm(mission_delta)) if np.all(np.isfinite(mission_delta)) else math.nan
        )
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
            "mission_goal_x": float(self.mission_goal[0]),
            "mission_goal_y": float(self.mission_goal[1]),
            "mission_goal_z": float(self.mission_goal[2]),
            "mission_goal_distance": mission_distance,
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
        # If something is very close in the front fan, stop moving forward and
        # add a small climb command to avoid immediate collision.
        if float(np.min(sonar[:5])) < 0.45:
            filtered[0] = min(filtered[0], 0.0)
            filtered[2] = max(filtered[2], 0.1)
            was_filtered = True
        # Side sonars push the drone away from nearby lateral obstacles.
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
