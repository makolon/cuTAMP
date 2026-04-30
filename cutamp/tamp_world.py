# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
from __future__ import annotations

import itertools
import logging
from functools import cached_property
from typing import Dict, List, Literal, Union

import torch
from jaxtyping import Float

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Obstacle
from curobo.types import DeviceCfg, JointState, Pose
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.envs import TAMPEnvironment
from cutamp.robots import RobotContainer
from cutamp.tamp_domain import get_initial_state
from cutamp.task_planning import State
from cutamp.utils.collision import get_world_collision_cost
from cutamp.utils.common import (
    approximate_goal_aabb,
    get_world_cfg,
    sample_between_bounds,
    transform_spheres,
)
from cutamp.utils.shapes import sample_greedy_surface_spheres

_log = logging.getLogger(__name__)


class TAMPWorld:
    """TAMP world with cuRobo v2 FK, IK, collision, and motion planning backends."""

    def __init__(
        self,
        env: TAMPEnvironment,
        device_cfg: DeviceCfg,
        robot: RobotContainer,
        q_init: Float[torch.Tensor, "dof"],
        ik_batch_size: int = 64,
        collision_activation_distance: float = 0.0,
        coll_n_spheres: int = 50,
        coll_sphere_radius: float = 0.005,
        motion_refinement_mode: Literal["ee_strict", "joint"] = "ee_strict",
        use_cuda_graph: bool = True,
    ):
        self.env = env
        self.device_cfg = device_cfg
        self.robot_container = robot
        self.robot_name = robot.name
        self.q_init = self.device_cfg.to_device(q_init)
        self.ik_solver_batch_size = ik_batch_size
        self.motion_refinement_mode = motion_refinement_mode
        self.collision_activation_distance = collision_activation_distance
        self.use_cuda_graph = use_cuda_graph

        self._movable_names = {obj.name for obj in env.movables}
        self._name_to_obj = {obj.name: obj for obj in env.movables + env.statics}
        self._obj_to_aabb: dict[str, torch.Tensor] = {}
        self._attached_object_name: str | None = None
        self._attached_joint_state: JointState | None = None

        self._obj_to_spheres: Dict[str, Float[torch.Tensor, "n 4"]] = {}
        for obj in self.movables:
            spheres = sample_greedy_surface_spheres(
                obj,
                n_spheres=coll_n_spheres,
                sphere_radius=coll_sphere_radius,
            )
            self._obj_to_spheres[obj.name] = spheres.to(self.device)

        self.world_cfg = get_world_cfg(env, include_movables=True)
        self.collision_fn = get_world_collision_cost(
            self.world_cfg,
            self.device_cfg,
            self.collision_activation_distance,
        )

        self.robot_collision_checker = self.create_robot_collision_checker()
        self.ik_solver = self.create_ik_solver()
        self.motion_planner = self.create_motion_planner()

    @property
    def movables(self) -> List[Obstacle]:
        return self.env.movables

    @property
    def is_movable(self, obj: Obstacle | str) -> bool:
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        return obj_name in self._movable_names

    @property
    def statics(self) -> List[Obstacle]:
        return self.env.statics

    @property
    def kinematics(self):
        return self.robot_container.kinematics

    @property
    def tool_frame(self) -> str:
        return self.robot_container.tool_frame

    @property
    def tool_from_ee(self) -> Float[torch.Tensor, "4 4"]:
        return self.robot_container.tool_from_ee

    @property
    def device(self) -> torch.device:
        return self.device_cfg.device

    @property
    def initial_state(self) -> State:
        return get_initial_state(
            movables=self.get_objects_by_type("Movable", return_name=True),
            surfaces=self.get_objects_by_type("Surface", return_name=True),
            sticks=self.get_objects_by_type("Stick", return_name=True),
            buttons=self.get_objects_by_type("Button", return_name=True),
        )

    @property
    def goal_state(self) -> State:
        return self.env.goal_state

    def get_objects_by_type(self, obj_type: str, return_name: bool = True) -> List[Union[Obstacle, str]]:
        if obj_type not in self.env.type_to_objects:
            return []

        objects = self.env.type_to_objects[obj_type]
        if return_name:
            return [obj.name for obj in objects]
        return objects

    def get_object(self, name: str) -> Obstacle:
        if name not in self._name_to_obj:
            raise ValueError(f"Object '{name}' not found in environment")
        return self._name_to_obj[name]

    def has_object(self, name: str) -> bool:
        return name in self._name_to_obj

    def get_object_pose(self, obj: Obstacle | str) -> Float[torch.Tensor, "4 4"]:
        obstacle = obj if isinstance(obj, Obstacle) else self.get_object(obj)
        return Pose.from_list(obstacle.pose, self.device_cfg).get_matrix()[0]

    def get_collision_spheres(self, obj: Obstacle | str) -> Float[torch.Tensor, "n 4"]:
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        return self._obj_to_spheres[obj_name]

    def get_aabb(self, obj: Obstacle | str) -> Float[torch.Tensor, "2 3"]:
        obj_name = obj.name if isinstance(obj, Obstacle) else obj

        if obj_name not in self._obj_to_aabb:
            self._obj_to_aabb[obj_name] = approximate_goal_aabb(self.get_object(obj_name)).to(self.device)

        return self._obj_to_aabb[obj_name]

    @cached_property
    def world_aabb(self) -> Float[torch.Tensor, "2 3"]:
        aabbs = [self.get_aabb(obj) for obj in self.movables]
        aabbs += [self.get_aabb(obj) for obj in self.statics]
        stacked = torch.stack(aabbs)
        lower = stacked[:, 0].min(dim=0).values
        upper = stacked[:, 1].max(dim=0).values
        return torch.stack([lower, upper])

    def warmup_ik_solver(self, num_particles: int) -> None:
        """Warm up cuRobo v2 IK solver."""
        q = JointState.from_position(
            sample_between_bounds(
                num_particles,
                bounds=self.robot_container.joint_limits,
            ).to(device=self.device, dtype=self.device_cfg.dtype)
        )

        world_from_ee = self.kinematics.compute_kinematics(q)
        goal_tool_poses = world_from_ee.tool_poses.as_goal(
            ordered_tool_frames=self.ik_solver.kinematics.tool_frames,
        )

        _ = self.ik_solver.solve_pose(goal_tool_poses)

    def warmup_motion_gen(self):
        self.motion_planner.warmup(enable_graph=True)

    def create_robot_collision_checker(self) -> RobotCollisionChecker:
        num_meshes = len(self.world_cfg.mesh)
        num_obstacles = len(self.movables) + len(self.statics)

        cache = {
            "mesh": max(16, num_meshes),
            "primitive": max(64, num_obstacles + 8),
        }

        checker_cfg = RobotCollisionCheckerCfg.load_from_config(
            robot_config=self.robot_container.robot_cfg,
            scene_model=self.world_cfg,
            device_cfg=self.device_cfg,
            collision_activation_distance=self.collision_activation_distance,
            n_meshes=cache["mesh"],
            n_cuboids=cache["primitive"],
        )
        return RobotCollisionChecker(checker_cfg)

    def create_ik_solver(self, num_seeds: int = 12) -> InverseKinematics:
        ik_cfg = InverseKinematicsCfg.create(
            robot=self.robot_container.robot_cfg,
            scene_model=self.world_cfg,
            device_cfg=self.device_cfg,
            num_seeds=num_seeds,
            max_batch_size=self.ik_solver_batch_size,
            use_cuda_graph=False,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            optimizer_collision_activation_distance=self.collision_activation_distance,
        )
        return InverseKinematics(ik_cfg)

    def create_motion_planner(self) -> MotionPlanner:
        motion_cfg = MotionPlannerCfg.create(
            robot=self.robot_container.robot_cfg,
            scene_model=self.world_cfg,
            device_cfg=self.device_cfg,
            use_cuda_graph=self.use_cuda_graph,
            num_ik_seeds=12,
            num_trajopt_seeds=4,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            optimizer_collision_activation_distance=self.collision_activation_distance,
        )
        return MotionPlanner(motion_cfg)


def check_tamp_world_not_in_collision(
    world: TAMPWorld,
    collision_tol: float = 1e-6,
    movable_activation_dist: float = 0.0
):
    """Check that the initial state of the movable objects are not in collision."""
    for obj in world.movables:
        # Transform spheres to world frame
        mat4x4 = Pose.from_list(obj.pose).get_matrix()[0]
        spheres = transform_spheres(world.get_collision_spheres(obj), mat4x4)  # [n, 4]
        spheres = spheres[None, None].contiguous()  # [1, 1, n, 4]

        coll_cost = world.collision_fn(spheres).sum()
        if coll_cost > collision_tol:
            _log.warning(f"Initial state in collision for object '{obj.name}' with cost {coll_cost}")
            # raise ValueError(f"Initial state in collision for object '{obj.name}' with cost {coll_cost}")

    # Catch collisions between spheres for movable objects
    obj_to_spheres = {}
    for idx, obj in enumerate(world.movables):
        obj_spheres = transform_spheres(world.get_collision_spheres(obj), world.get_object_pose(obj))
        obj_to_spheres[obj.name] = obj_spheres

    for obj_1, obj_2 in itertools.combinations(world.movables, 2):
        obj_1_spheres = obj_to_spheres[obj_1.name]
        obj_2_spheres = obj_to_spheres[obj_2.name]
        coll_cost = sphere_to_sphere_overlap(
            obj_1_spheres,
            obj_2_spheres,
            activation_distance=movable_activation_dist,
            use_aabb_check=True,
        )
        if coll_cost > collision_tol:
            _log.warning(f"Initial state in collision between {obj_1.name} and {obj_2.name} with cost {coll_cost}")
