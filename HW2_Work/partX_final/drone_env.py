#!/usr/bin/env python3
"""Final five-stage Gymnasium environment for NSYSU drone PPO.

Part X Final keeps the Part 3+Y curriculum and one fixed observation/action
interface across all stages. Sonar fields exist from Stage 1, but they are
masked to safe constants until Stage 5 so early checkpoints can continue
training in obstacle stages.

This final fork includes the anti-oscillation reward shaping used to reduce
passing through or circling around the target.
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
# Observation layout:
#   12 navigation fields
#   7 current sonar ranges
#   7 current sonar risk values
#   7 previous sonar ranges
#   7 sonar trend values
#   1 sonar-enabled flag
OBSERVATION_DIM = 12 + (4 * SONAR_COUNT) + 1
TARGET_MARKER_NAME = "part3_rl_target_marker"
STAGE5_FINAL_GOAL = np.array([10.0, 0.0, 1.0], dtype=np.float32)
STAGE5_LOCAL_GOAL_STEP = 1.0
GENERATED_OBSTACLE_NAME = "part3_plusy_generated_cone"
STAGE5_CORRIDOR_X_MIN = 0.0
STAGE5_CORRIDOR_X_MAX = 10.0
STAGE5_CORRIDOR_Y_MIN = -3.0
STAGE5_CORRIDOR_Y_MAX = 3.0
STAGE5B_CONE_X_MIN = 2.0
STAGE5B_CONE_X_MAX = 8.0
STAGE5B_CONE_Y_MIN = -2.0
STAGE5B_CONE_Y_MAX = 2.0
STAGE5B_TARGET_Z = 1.0
STAGE5_CONE_Z = 0.05
STAGE5B_MIN_CONES = 2
STAGE5B_MAX_CONES = 10
STAGE5B_CONE_MIN_SPACING = 1.0
STAGE5B_START_CLEARANCE = 1.5
STAGE5B_GOAL_CLEARANCE = 1.5
STAGE5B_CONE_SAMPLE_ATTEMPTS = 200
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

def target_marker_sdf(name: str, color: tuple[float, float, float, float]) -> str:
    """Build the Gazebo SDF string for a colored target ball."""
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


@dataclass(frozen=True)
class StageSpec:
    """Static curriculum settings for one stage/variant.

    fixed_targets is the default target list. Optional x/y/z bounds replace
    only the axes that should be randomized for that stage. The focus field is
    used by the reward helpers so earlier stages emphasize the single skill the
    agent is supposed to learn first.
    """

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
    # Stage 1: learn altitude control before asking the policy to translate.
    (1, "A"): StageSpec(
        name="stage1A_fixed_vertical",
        description="fixed altitude target on Gazebo z",
        fixed_targets=((0.0, 0.0, 1.0),),
        focus="vertical",
    ),
    (1, "B"): StageSpec(
        name="stage1B_random_vertical",
        description="random altitude target on Gazebo z",
        fixed_targets=((0.0, 0.0, 1.0),),
        z_bounds=(0.8, 1.5),
        focus="vertical",
    ),
    # Stage 2: learn forward/backward x movement while keeping altitude stable.
    (2, "A"): StageSpec(
        name="stage2A_fixed_horizontal",
        description="fixed x target with stable altitude",
        fixed_targets=((1.0, 0.0, 1.0),),
        focus="horizontal",
    ),
    (2, "B"): StageSpec(
        name="stage2B_random_horizontal",
        description="random x target with stable altitude",
        fixed_targets=((1.0, 0.0, 1.0),),
        x_bounds=(-1.0, 1.5),
        focus="horizontal",
    ),
    # Stage 3: add y-axis lateral movement from the Part 3+Y fork.
    (3, "A"): StageSpec(
        name="stage3A_fixed_lateral",
        description="fixed sideway y target with stable altitude",
        fixed_targets=((0.0, 1.0, 1.0),),
        focus="lateral",
    ),
    (3, "B"): StageSpec(
        name="stage3B_random_lateral",
        description="random sideway y target with stable altitude",
        fixed_targets=((0.0, 1.0, 1.0),),
        y_bounds=(-1.0, 1.0),
        focus="lateral",
    ),
    # Stage 4: combine x/y/z target reaching without obstacle avoidance.
    (4, "A"): StageSpec(
        name="stage4A_random_xyz",
        description="single random x/y/z target without sonar",
        fixed_targets=((1.0, 0.0, 1.0),),
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        z_bounds=(0.8, 1.5),
        focus="combined",
    ),
    (4, "B"): StageSpec(
        name="stage4B_sequence_xyz",
        description="three sequential random x/y/z targets without sonar",
        fixed_targets=((1.0, 0.0, 1.0),),
        x_bounds=(-1.0, 1.0),
        y_bounds=(-1.0, 1.0),
        z_bounds=(0.8, 1.5),
        sequence_count=3,
        focus="combined",
    ),
    # Stage 5: keep the same action/observation interface, but unmask sonar and
    # train the final obstacle-avoidance task.
    (5, "A"): StageSpec(
        name="stage5_single_obstacle",
        description="fixed long-goal sonar mission with generated midpoint cone",
        fixed_targets=((10.0, 0.0, 1.0),),
        sonar_enabled=True,
        focus="obstacle",
    ),
    (5, "B"): StageSpec(
        name="stage5B_corridor_obstacles",
        description="forward corridor target with 2-10 random generated cones",
        fixed_targets=((10.0, 0.0, 1.0),),
        sonar_enabled=True,
        focus="obstacle",
    ),
}


def get_stage_spec(stage: int, variant: str) -> StageSpec:
    variant = variant.upper()
    if variant not in ("A", "B"):
        raise ValueError("variant must be A or B")
    key = (stage, variant)
    if key not in STAGE_SPECS:
        raise ValueError(f"Unsupported stage/variant: stage={stage}, variant={variant}")
    return STAGE_SPECS[key]


class DroneRosBridge(Node):
    """ROS 2 bridge for pose, velocity, reset/takeoff, target marker, and sonar."""

    def __init__(self, namespace: str = "/simple_drone") -> None:
        super().__init__("part3_drone_curriculum_env")
        ns = namespace.rstrip("/")
        # The Gym environment reads these latest cached sensor values after
        # every control step. ROS callbacks update them asynchronously during
        # rclpy.spin_once().
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
        self.generated_obstacle_names: set[str] = set()
        self.generated_obstacle_spawned = False
        self.target_marker_warning_logged = False
        self.reset_world_warning_logged = False

        # The action vector maps directly to /cmd_vel linear x/y/z. No angular
        # command is sent because this homework controls translational motion.
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
        # Return a tiny sector-specific callback so each Range topic can update
        # the matching entry while sharing the same sanitizing logic later.
        def callback(msg: Range) -> None:
            self.sonar_min_range = float(msg.min_range)
            self.sonar_max_range = float(msg.max_range)
            self.front_sonar_ranges[sector] = float(msg.range)

        return callback

    def _side_cb(self, sector: str):
        # Side sonar is stored separately from the front fan but is merged into
        # the fixed SONAR_SECTORS order before entering the observation vector.
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
        # Gazebo reset is best effort. If the service is missing, the simple
        # drone reset/land/takeoff sequence below still gives a usable episode.
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
        # Clear cached state so the next episode waits for fresh Gazebo/ROS
        # messages instead of accidentally using the previous episode's pose.
        self.pose = None
        self.down_sonar_range = None
        for key in self.front_sonar_ranges:
            self.front_sonar_ranges[key] = None
        for key in self.side_sonar_ranges:
            self.side_sonar_ranges[key] = None

        # Publish several reset messages because the simulator may miss a single
        # topic message while Gazebo plugins are still recovering.
        for _ in range(3):
            self.reset_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.2)

        # A short land phase helps force the simple_drone plugin back into a
        # grounded state before the new takeoff command starts.
        land_deadline = time.monotonic() + 1.3
        while time.monotonic() < land_deadline:
            self.land_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.1)

        for _ in range(3):
            self.reset_pub.publish(Empty())
            rclpy.spin_once(self, timeout_sec=0.2)

        # Keep sending takeoff until the reported z height reaches the requested
        # altitude. This is more reliable than publishing takeoff only once.
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
        # Delete both remembered marker names and possible stale names from
        # older runs, otherwise Gazebo can keep old balls in the world.
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
            # Multi-target Stage 4B gets different colors so it is clear which
            # waypoint order the policy is trying to follow.
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
        names = set(self.generated_obstacle_names)
        names.add(GENERATED_OBSTACLE_NAME)
        for index in range(STAGE5B_MAX_CONES):
            names.add(self._generated_obstacle_name(index, STAGE5B_MAX_CONES))
        for name in sorted(names):
            self._delete_target_marker(name)
        self.generated_obstacle_names.clear()
        self.generated_obstacle_spawned = False

    def _generated_obstacle_name(self, index: int, obstacle_count: int) -> str:
        if obstacle_count == 1:
            return GENERATED_OBSTACLE_NAME
        return f"{GENERATED_OBSTACLE_NAME}_{index + 1}"

    def spawn_generated_cone(self, position: np.ndarray) -> None:
        self.spawn_generated_cones(np.array([position], dtype=np.float32))

    def spawn_generated_cones(self, positions: np.ndarray) -> None:
        if not self.target_marker_enabled:
            return
        if not self.spawn_entity_client.wait_for_service(timeout_sec=0.5):
            return
        # Clear once before spawning so old cones from previous episodes do not
        # accumulate in the corridor.
        self.clear_generated_obstacle()
        obstacle_count = len(positions)
        for index, position in enumerate(positions):
            obstacle_name = self._generated_obstacle_name(index, obstacle_count)
            req = SpawnEntity.Request()
            req.name = obstacle_name
            req.xml = GENERATED_CONE_SDF.format(name=obstacle_name)
            req.reference_frame = "world"
            req.initial_pose.position.x = float(position[0])
            req.initial_pose.position.y = float(position[1])
            req.initial_pose.position.z = float(position[2])
            req.initial_pose.orientation.w = 1.0
            future = self.spawn_entity_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.8)
            if future.done() and future.result() is not None:
                self.generated_obstacle_names.add(obstacle_name)
        self.generated_obstacle_spawned = bool(self.generated_obstacle_names)


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
        near_target_action_penalty: float = 0.3,
        action_penalty: float = 0.03,
        action_smoothness_penalty: float = 0.09,
    ) -> None:
        super().__init__()
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True

        self.stage = int(stage)
        self.variant = variant.upper()
        self.stage_spec = get_stage_spec(self.stage, self.variant)
        self.target_override = target_override
        self.max_steps = int(max_steps)
        self.success_distance = float(success_distance)
        self.step_dt = float(step_dt)
        self.log_position_every = max(0, int(log_position_every))
        self.near_target_action_penalty = float(near_target_action_penalty)
        self.action_penalty = float(action_penalty)
        self.action_smoothness_penalty = float(action_smoothness_penalty)
        self.ros = DroneRosBridge(namespace=namespace)

        # Stage 5 has a 10 m mission goal, so it needs a wider xy safety box
        # than the earlier short-distance curriculum stages.
        self.xy_limit = 12.0 if self.stage >= 5 else 8.0
        self.max_altitude = 5.0
        self.min_altitude = 0.25
        self.takeoff_altitude = 0.5
        # Sonar distances below caution_distance become a smooth risk penalty.
        # Distances below unsafe_distance terminate the episode as a collision
        # or near-collision.
        self.max_sonar_range = 10.0
        self.sonar_caution_distance = 1.5
        self.sonar_unsafe_distance = 0.25
        self.near_miss_distance = 0.5

        # Episode state. The previous_* fields are reward baselines; they let
        # each step reward improvement instead of only rewarding the final goal.
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
        # Later stages can require the drone to be inside the goal while moving
        # slowly. The current setting of 1 keeps success permissive but still
        # routes Stage 5 through the stability check.
        self.stable_success_steps = 0
        self.stable_success_required = 1
        self.stable_success_velocity = 0.35

        # Action is desired linear velocity in Gazebo/world x, y, z. z is
        # capped lower because vertical motion is more sensitive in this drone.
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
        # Sample the target list first so markers, reward baselines, and info
        # fields all describe the same episode.
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

        # Reset/takeoff sometimes fails transiently in Gazebo. Retrying here is
        # cheaper than aborting a long training run because one episode started
        # before the simulator settled.
        takeoff_ok = False
        for attempt in range(3):
            takeoff_ok = self.ros.reset_and_takeoff(self.takeoff_altitude)
            self._wait_for_state(min_altitude=self.takeoff_altitude)
            if takeoff_ok and self.ros.pose is not None and self.ros.pose[2] >= self.min_altitude:
                break
            self.ros.get_logger().warning(f"Retrying Part X reset/takeoff {attempt + 1}/3")
        if not takeoff_ok or self.ros.pose is None:
            raise RuntimeError("Part X reset/takeoff failed; restart Gazebo and try again.")

        self._update_stage_obstacle()
        obs = self._get_obs()
        self._update_stage_markers(force=True)
        # Initialize progress baselines after the first observation so step()
        # does not count reset motion as policy progress.
        self.previous_distance = float(self.last_info["distance_to_target"])
        self.previous_final_distance = float(self.last_info["mission_goal_distance"])
        self.previous_abs_error = self._abs_error()
        self._log_position(force=True)
        return obs, self._info("running")

    def step(self, action: np.ndarray):
        # Gym gives raw policy output. Clip it before publishing so the ROS
        # command always stays inside the declared action_space contract.
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
        # Dense progress reward: positive when the active target gets closer and
        # negative when it moves away. The gain is lower very near the target so
        # the policy does not learn to overshoot just to collect progress.
        if self.previous_distance is not None and math.isfinite(distance):
            scale = 10.0 if distance >= 0.5 else 4.0
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
        # sharp action changes that make Gazebo control unstable. These terms
        # are intentionally smaller than success/crash rewards, so they shape
        # the route without replacing the main task objective.
        reward -= 0.05 * distance
        reward -= self._stage_precision_penalty(x_error, y_error, z_error)
        if distance < 0.6:
            reward -= 0.18 * velocity_norm
            reward -= self.near_target_action_penalty * float(np.linalg.norm(filtered_action))
        if distance < 0.45:
            # Once inside the close target region, split velocity into radial
            # and tangential parts. This catches two bad behaviors:
            #   1. circling around the ball
            #   2. flying through the ball instead of settling
            reward -= self._near_target_motion_penalty(
                distance=distance,
                target_vector=np.array([dx, dy, dz], dtype=np.float32),
                velocity=self.ros.velocity,
            )
        reward -= self.action_penalty * float(np.linalg.norm(filtered_action))
        reward -= self.action_smoothness_penalty * float(np.linalg.norm(filtered_action - self.previous_action))
        if was_filtered:
            reward -= 0.25
        self.previous_action = filtered_action.copy()

        obstacle_max_risk = float(info["obstacle_max_risk"])
        obstacle_mean_risk = float(info["obstacle_mean_risk"])
        min_obstacle = float(info["min_obstacle_sonar_range"])
        if self.sonar_enabled:
            # Sonar risk is zero in free space and approaches one near obstacles.
            # mean risk discourages staying in generally crowded space, while
            # max risk reacts strongly to the closest detected obstacle.
            reward -= 2.0 * obstacle_mean_risk**2
            reward -= 4.0 * obstacle_max_risk**2
            if self.stage == 5 and obstacle_max_risk > 0.2 and abs(float(info["vx"])) < 0.05:
                reward -= 0.2

        terminated = False
        truncated = False
        status = "running"
        # Terminal checks are ordered from sensor/safety failures to successful
        # completion. Safety failures use large penalties so PPO learns them as
        # hard constraints, not minor route preferences.
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
        # target_override is used by evaluation scripts to replay a specific
        # target instead of sampling a new random one.
        if self.target_override is not None:
            return np.array([self.target_override], dtype=np.float32)
        if self.stage == 5:
            if self.variant == "B":
                # Variant B keeps the mission mostly forward for the front sonar
                # while randomizing lateral goal position inside the corridor.
                y = random.uniform(STAGE5_CORRIDOR_Y_MIN, STAGE5_CORRIDOR_Y_MAX)
                return np.array(
                    [(STAGE5_CORRIDOR_X_MAX, y, STAGE5B_TARGET_Z)],
                    dtype=np.float32,
                )
            return np.array([STAGE5_FINAL_GOAL], dtype=np.float32)
        if self.stage_spec.sequence_count > 1:
            # Stage 4B is a waypoint sequence. All targets are sampled at reset
            # so the policy can observe total_targets and target_progress.
            targets = [self._sample_one_target(index) for index in range(self.stage_spec.sequence_count)]
            return np.array(targets, dtype=np.float32)
        return np.array([self._sample_one_target(0)], dtype=np.float32)

    def _stage5_local_target(self) -> np.ndarray:
        mission_goal = self.mission_goal
        if self.ros.pose is None:
            # Before the first pose message arrives, approximate the local goal
            # from the origin so observation construction can still run.
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
        if self.stage == 5 and self.ros.pose is not None:
            if self.variant == "B":
                
                self.ros.clear_generated_obstacle()
                #self.ros.spawn_generated_cones(self._sample_stage5b_cones())
            else:
                # Stage 5A stays the simple bridge task: one cone halfway
                # between takeoff position and the fixed forward goal.
                midpoint = (self.ros.pose + self.mission_goal) / 2.0
                midpoint[2] = STAGE5_CONE_Z
                self.ros.spawn_generated_cone(midpoint)
        else:
            self.ros.clear_generated_obstacle()

    def _sample_stage5b_cones(self) -> np.ndarray:
        cone_count = random.randint(STAGE5B_MIN_CONES, STAGE5B_MAX_CONES)
        positions: list[np.ndarray] = []
        if self.ros.pose is not None:
            start_pose = self.ros.pose.astype(np.float32)
        else:
            start_pose = np.zeros(3, dtype=np.float32)
        goal = self.mission_goal.astype(np.float32)

        for _ in range(STAGE5B_CONE_SAMPLE_ATTEMPTS):
            if len(positions) >= cone_count:
                break
            candidate = np.array(
                [
                    random.uniform(STAGE5B_CONE_X_MIN, STAGE5B_CONE_X_MAX),
                    random.uniform(STAGE5B_CONE_Y_MIN, STAGE5B_CONE_Y_MAX),
                    STAGE5_CONE_Z,
                ],
                dtype=np.float32,
            )
            if self._stage5b_cone_position_valid(candidate, positions, start_pose, goal):
                positions.append(candidate)

        if len(positions) < cone_count:
            self.ros.get_logger().warning(
                f"Stage 5B requested {cone_count} cones but only sampled {len(positions)} valid cones"
            )
        return np.array(positions, dtype=np.float32)

    def _stage5b_cone_position_valid(
        self,
        candidate: np.ndarray,
        positions: list[np.ndarray],
        start_pose: np.ndarray,
        goal: np.ndarray,
    ) -> bool:
        candidate_xy = candidate[:2]
        if float(np.linalg.norm(candidate_xy - start_pose[:2])) < STAGE5B_START_CLEARANCE:
            return False
        if float(np.linalg.norm(candidate_xy - goal[:2])) < STAGE5B_GOAL_CLEARANCE:
            return False
        for position in positions:
            if float(np.linalg.norm(candidate_xy - position[:2])) < STAGE5B_CONE_MIN_SPACING:
                return False
        return True

    def _update_stage_markers(self, force: bool = False) -> None:
        if not force:
            return
        if self.stage == 5:
            self.ros.update_target_marker(self.mission_goal)
        else:
            self.ros.update_target_markers(self.targets)

    def _sample_one_target(self, index: int) -> tuple[float, float, float]:
        # If no random bounds are configured, use the fixed target. For sequence
        # stages, clamp the fixed target index so short fixed_targets lists are
        # still valid.
        if (
            self.stage_spec.x_bounds is None
            and self.stage_spec.y_bounds is None
            and self.stage_spec.z_bounds is None
        ):
            fixed = self.stage_spec.fixed_targets[min(index, len(self.stage_spec.fixed_targets) - 1)]
            return tuple(float(v) for v in fixed)
        base = self.stage_spec.fixed_targets[0]
        # Only axes with bounds are randomized; the other coordinates stay at
        # the base target so each curriculum stage isolates one new skill.
        x = random.uniform(*self.stage_spec.x_bounds) if self.stage_spec.x_bounds else base[0]
        y = random.uniform(*self.stage_spec.y_bounds) if self.stage_spec.y_bounds else base[1]
        z = random.uniform(*self.stage_spec.z_bounds) if self.stage_spec.z_bounds else base[2]
        return (float(x), float(y), float(z))

    def _axis_progress_reward(self, delta: np.ndarray) -> float:
        # delta is previous_abs_error - current_abs_error. A positive value means
        # that axis improved on this step. Weights match the stage focus.
        if self.stage_spec.focus == "vertical":
            weights = np.array([2.0, 2.0, 12.0], dtype=np.float32)
        elif self.stage_spec.focus == "horizontal":
            weights = np.array([12.0, 3.0, 6.0], dtype=np.float32)
        elif self.stage_spec.focus == "lateral":
            weights = np.array([3.0, 12.0, 6.0], dtype=np.float32)
        else:
            weights = np.array([7.0, 7.0, 7.0], dtype=np.float32)
        return float(np.dot(weights, delta))

    def _stage_precision_penalty(self, x_error: float, y_error: float, z_error: float) -> float:
        # This is the distance penalty split by axis. The focused axis gets the
        # strongest pressure, while non-focused axes still prevent sloppy drift.
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
        # Bad or zero-length geometry cannot produce a meaningful direction
        # vector, so return no extra penalty instead of creating NaN rewards.
        if distance <= 1e-6 or not math.isfinite(distance):
            return 0.0
        if not np.all(np.isfinite(target_vector)) or not np.all(np.isfinite(velocity)):
            return 0.0

        # direction_to_target points from the drone toward the target. Projecting
        # velocity onto this unit vector gives radial_speed:
        #   positive: moving toward the target
        #   near zero: sliding sideways around the target
        #   negative: moving away after passing/overshooting the target
        direction_to_target = target_vector / max(distance, 1e-6)
        radial_speed = float(np.dot(velocity, direction_to_target))
        # The leftover velocity after removing the radial component is sideways
        # motion around the target, which is the circling behavior we want to
        # discourage near the goal.
        tangential_velocity = velocity - radial_speed * direction_to_target
        tangential_speed = float(np.linalg.norm(tangential_velocity))
        total_speed = float(np.linalg.norm(velocity))

        # Always penalize sideways/fast motion near the target. Add an extra
        # penalty only when radial_speed is negative enough to show fly-through
        # or retreat from the target.
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
            # Stage 5 observes a moving local target, but success is the final
            # mission goal. Otherwise the drone could "succeed" at a subgoal.
            reached = float(np.linalg.norm(self.mission_goal - self.ros.pose)) < self.success_distance
            return self._stable_target_reached(reached, velocity_norm)
        if self.stage_spec.focus == "vertical":
            # Stage 1 mainly checks z error but also limits lateral drift so the
            # policy cannot pass altitude training while far from the origin.
            lateral_error = math.hypot(x_error, y_error)
            lateral_tolerance = max(0.20, 1.5 * self.success_distance)
            return z_error < self.success_distance and lateral_error < lateral_tolerance
        reached = distance < self.success_distance
        if self.stage == 4:
            return reached
        if self.stage >= 5:
            return self._stable_target_reached(reached, velocity_norm)
        return reached

    def _stable_target_reached(self, reached: bool, velocity_norm: float) -> bool:
        # Require the drone to be both inside the target ball and slow enough.
        # This avoids counting high-speed fly-through as a clean arrival.
        if reached and velocity_norm <= self.stable_success_velocity:
            self.stable_success_steps += 1
        else:
            self.stable_success_steps = 0
        return self.stable_success_steps >= self.stable_success_required

    def _wait_for_state(self, timeout_sec: float = 5.0, min_altitude: float | None = None) -> None:
        # Spin ROS until a pose arrives. reset() uses min_altitude so it waits
        # for actual takeoff instead of only seeing a ground-level pose.
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
        # Pose, velocity, target delta, sonar, and progress are folded into one
        # fixed-size vector so PPO checkpoints can be reused across stages.
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
            # Trend is positive when the normalized range shrinks, meaning an
            # obstacle is getting closer compared with the previous step.
            sonar_risk = self._ranges_to_risk(sonar)
            sonar_trend = prev_sonar_norm - sonar_norm
            sonar_enabled = 1.0
        else:
            # Keep sonar channels present but masked before Stage 5. This avoids
            # changing the observation shape while preventing early stages from
            # learning noise from irrelevant obstacle sensors.
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
            mission_start = np.array([0.0, 0.0, self.takeoff_altitude], dtype=np.float32)
            mission_distance_start = max(float(np.linalg.norm(self.mission_goal - mission_start)), 1e-6)
            target_progress = float(
                np.clip(
                    (mission_distance_start - mission_distance_for_progress) / mission_distance_start,
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
            # last_info feeds logs, evaluation CSVs, reward calculations, and
            # Gym info output. Keep names stable so existing scripts keep
            # working.
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
        # Missing or invalid sonar means "no obstacle seen" instead of NaN. This
        # keeps training running during brief ROS sensor dropouts.
        max_range = max(min(self.ros.sonar_max_range, self.max_sonar_range), 0.1)
        if raw is None or not math.isfinite(raw):
            return max_range
        return float(np.clip(raw, self.ros.sonar_min_range, max_range))

    def _safe_sonar_ranges(self) -> np.ndarray:
        # The order must match SONAR_SECTORS and the observation layout.
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
        # Convert meters into a bounded risk score. At caution_distance or
        # farther risk is 0; as range approaches 0, risk approaches 1.
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
        # Optional lightweight console trace for debugging a live Gazebo run.
        # It is disabled by default to avoid slowing down training logs.
        if not force and (
            self.log_position_every <= 0
            or self.step_count % self.log_position_every != 0
        ):
            return
        info = self.last_info
        if not info:
            return
        target = self.current_target
        velocity_norm = float(np.linalg.norm(self.ros.velocity))
        print(
            "[pose] "
            f"step={self.step_count} "
            f"target_index={self.target_index + 1}/{len(self.targets)} "
            f"pos=({info['x']:.2f}, {info['y']:.2f}, {info['z']:.2f}) "
            f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) "
            f"distance={info['distance_to_target']:.2f} "
            f"velocity={velocity_norm:.2f}",
            flush=True,
        )

    def _info(self, status: str) -> dict[str, Any]:
        # Gymnasium expects info to be a plain dict. Convert numpy scalar values
        # to Python floats so CSV/json logging does not need special handling.
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
