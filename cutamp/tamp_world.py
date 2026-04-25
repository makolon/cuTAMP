# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import copy
import itertools
import logging
from functools import cached_property
from typing import Dict, List, Literal, Union

import torch
from jaxtyping import Float

from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.motion_retargeter import ToolPoseCriteria
from curobo.scene import Obstacle, Scene
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose
from cutamp.costs import sphere_to_sphere_overlap
from cutamp.envs import TAMPEnvironment
from cutamp.robots import RobotContainer
from cutamp.tamp_domain import get_initial_state
from cutamp.task_planning import State
from cutamp.utils.collision import get_world_collision_cost
from cutamp.utils.common import (
    approximate_goal_aabb,
    filter_valid_spheres,
    get_world_cfg,
    sample_between_bounds,
    transform_spheres,
)
from cutamp.utils.shapes import sample_greedy_surface_spheres

_log = logging.getLogger(__name__)


class TAMPWorld:
    """TAMP world with cuRobo v2 backends for FK, IK, motion planning, and collision."""

    def __init__(
        self,
        env: TAMPEnvironment,
        device_cfg: DeviceCfg,
        robot: RobotContainer,
        q_init: Float[torch.Tensor, "dof"],
        ik_max_batch_size: int = 1,
        collision_activation_distance: float = 0.0,
        coll_n_spheres: int = 50,
        coll_sphere_radius: float = 0.005,
        motion_refinement_mode: Literal["ee_strict", "joint"] = "ee_strict",
    ):
        self.env = env
        self.device_cfg = device_cfg

        self._movable_names = {obj.name for obj in env.movables}
        self._name_to_obj = {obj.name: obj for obj in env.movables + env.statics}

        self.robot_container = robot
        self.robot_name = self.robot_container.name
        self.q_init = self.device_cfg.to_device(q_init)
        self.ik_max_batch_size = max(1, int(ik_max_batch_size))
        self.ik_solver_batch_size = self.ik_max_batch_size
        self.motion_refinement_mode = motion_refinement_mode
        self.collision_activation_distance = collision_activation_distance

        self._obj_to_spheres: Dict[str, Float[torch.Tensor, "n 4"]] = {}
        for obj in self.movables:
            spheres = sample_greedy_surface_spheres(
                obj,
                n_spheres=coll_n_spheres,
                sphere_radius=coll_sphere_radius,
            )
            self._obj_to_spheres[obj.name] = spheres.to(self.device_cfg.device)

        self._static_scene = get_world_cfg(env, include_movables=False)
        self._planning_scene = get_world_cfg(env, include_movables=True)
        self._runtime_scene = self._planning_scene.clone()
        self.collision_fn = get_world_collision_cost(
            self._static_scene,
            self.device_cfg,
            collision_activation_distance,
        )

        self.robot_collision_checker = self.get_robot_collision_checker()
        self.ik_solver = self.get_ik_solver()
        self.motion_planner = self.get_motion_planner()
        self._default_tool_pose_criteria = {
            tool_frame: ToolPoseCriteria(device_cfg=self.device_cfg)
            for tool_frame in self.motion_planner.tool_frames
        }
        self.motion_planner.update_tool_pose_criteria(self._default_tool_pose_criteria)

        self._attached_object_name: str | None = None
        self._attached_joint_state: JointState | None = None
        self._obj_to_aabb: dict[str, torch.Tensor] = {}

    @property
    def movables(self) -> List[Obstacle]:
        return self.env.movables

    def is_movable(self, obj: Obstacle | str) -> bool:
        if isinstance(obj, Obstacle):
            obj = obj.name
        return obj in self._movable_names

    @property
    def statics(self) -> List[Obstacle]:
        return self.env.statics

    @property
    def kinematics(self):
        return self.robot_container.kinematics

    @property
    def tool_from_ee(self) -> Float[torch.Tensor, "4 4"]:
        return self.robot_container.tool_from_ee

    @cached_property
    def ee_from_tool(self) -> Float[torch.Tensor, "4 4"]:
        return torch.linalg.inv(self.tool_from_ee)

    @property
    def tool_frame(self) -> str:
        return self.robot_container.tool_frame

    @property
    def joint_names(self) -> list[str]:
        return list(self.robot_container.joint_names)

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

    def get_object_pose(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "4 4"]:
        obj = obj if isinstance(obj, Obstacle) else self.get_object(obj)
        pose = Pose.from_list(obj.pose, self.device_cfg)
        return pose.get_matrix()[0]

    def get_collision_spheres(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "n 4"]:
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        return self._obj_to_spheres[obj_name]

    def get_aabb(self, obj: Union[Obstacle, str]) -> Float[torch.Tensor, "2 3"]:
        obj_name = obj.name if isinstance(obj, Obstacle) else obj
        if obj_name not in self._obj_to_aabb:
            self._obj_to_aabb[obj_name] = approximate_goal_aabb(self.get_object(obj_name)).to(self.device)
        return self._obj_to_aabb[obj_name]

    @cached_property
    def world_aabb(self) -> Float[torch.Tensor, "2 3"]:
        aabbs = [self.get_aabb(obj) for obj in self.movables] + [self.get_aabb(obj) for obj in self.statics]
        stacked = torch.stack(aabbs)
        union_lower = stacked[:, 0].min(dim=0).values
        union_upper = stacked[:, 1].max(dim=0).values
        return torch.stack([union_lower, union_upper])

    def joint_state_from_position(self, position: torch.Tensor) -> JointState:
        position = self.device_cfg.to_device(position)
        return JointState.from_position(position, joint_names=self.joint_names)

    def ensure_joint_state(self, joint_state: JointState | torch.Tensor) -> JointState:
        if isinstance(joint_state, JointState):
            if joint_state.joint_names == self.joint_names:
                return joint_state
            return JointState.from_position(joint_state.position, joint_names=self.joint_names)
        return self.joint_state_from_position(joint_state)

    def compute_kinematics(self, joint_state: JointState | torch.Tensor):
        return self.kinematics.compute_kinematics(self.ensure_joint_state(joint_state))

    def compute_tool_pose(self, joint_state: JointState | torch.Tensor) -> Pose:
        state = self.compute_kinematics(joint_state)
        return state.tool_poses.get_link_pose(self.tool_frame)

    def compute_ee_matrix(self, joint_state: JointState | torch.Tensor) -> torch.Tensor:
        matrix = self.compute_tool_pose(joint_state).get_matrix() @ self.ee_from_tool
        return matrix[0] if matrix.shape[0] == 1 else matrix

    def compute_robot_spheres(self, joint_state: JointState | torch.Tensor) -> torch.Tensor:
        return filter_valid_spheres(self.compute_kinematics(joint_state).robot_spheres)

    def compute_robot_scene_and_self_collision(self, confs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        confs = self.device_cfg.to_device(confs)
        squeeze_horizon = False
        if confs.ndim == 2:
            confs = confs[:, None, :]
            squeeze_horizon = True
        elif confs.ndim != 3:
            raise ValueError(f"Expected joint tensor with shape [batch, dof] or [batch, horizon, dof], got {tuple(confs.shape)}")

        batch, horizon, dof = confs.shape
        self.robot_collision_checker.setup_batch_tensors(batch, horizon)

        state = self.robot_collision_checker.get_kinematics(confs)
        robot_spheres = state.get_link_spheres()
        num_spheres = robot_spheres.shape[2]

        collision_cost = self.robot_collision_checker.collision_cost
        if collision_cost is not None and collision_cost.config.num_spheres != num_spheres:
            collision_cost.update_num_spheres(num_spheres, batch_size=batch, horizon=horizon)

        collision_constraint = self.robot_collision_checker.collision_constraint
        if collision_constraint is not None and collision_constraint.config.num_spheres != num_spheres:
            collision_constraint.update_num_spheres(num_spheres, batch_size=batch, horizon=horizon)

        robot_to_world = self.robot_collision_checker.get_collision_distance(state)
        if robot_to_world.ndim == 3:
            if collision_cost is not None:
                robot_to_world = collision_cost.jit_weight_distance(
                    robot_to_world, collision_cost.config.sum_distance
                )
            else:
                robot_to_world = robot_to_world.sum(dim=-1)

        robot_to_self = torch.zeros(
            (batch, horizon),
            device=robot_spheres.device,
            dtype=robot_spheres.dtype,
        )

        if squeeze_horizon:
            return robot_to_world[:, 0], robot_to_self[:, 0]
        return robot_to_world, robot_to_self

    def warmup_ik_solver(self, num_particles: int):
        q = sample_between_bounds(num_particles, bounds=self.robot_container.joint_limits)
        world_from_ee = self.compute_ee_matrix(self.joint_state_from_position(q))
        _ = self.solve_pose(world_from_ee, return_seeds=1)

    def warmup_motion_gen(self):
        self.motion_planner.warmup(enable_graph=True)

    def _planner_robot_cfg(self) -> dict:
        robot_cfg = copy.deepcopy(self.robot_container.robot_cfg)
        kinematics_cfg = robot_cfg.setdefault("robot_cfg", {}).setdefault("kinematics", {})
        extra_collision_spheres = kinematics_cfg.setdefault("extra_collision_spheres", {})
        if self._obj_to_spheres:
            extra_collision_spheres["attached_object"] = max(len(spheres) for spheres in self._obj_to_spheres.values())
        else:
            extra_collision_spheres["attached_object"] = 0
        return robot_cfg

    def get_robot_collision_checker(self) -> RobotCollisionChecker:
        cache = {"mesh": max(16, len(getattr(self._planning_scene, "mesh", ()) or ())), "primitive": max(64, len(self.movables) + len(self.statics) + 8)}
        checker_cfg = RobotCollisionCheckerCfg.load_from_config(
            robot_config=self._planner_robot_cfg(),
            scene_model=self._runtime_scene,
            device_cfg=self.device_cfg,
            collision_activation_distance=self.collision_activation_distance,
            n_meshes=cache["mesh"],
            n_cuboids=cache["primitive"],
        )
        return RobotCollisionChecker(checker_cfg)

    def get_ik_solver(self, num_seeds: int = 12) -> InverseKinematics:
        ik_cfg = InverseKinematicsCfg.create(
            robot=self._planner_robot_cfg(),
            scene_model=self._runtime_scene,
            device_cfg=self.device_cfg,
            num_seeds=num_seeds,
            max_batch_size=self.ik_solver_batch_size,
            use_cuda_graph=False,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            optimizer_collision_activation_distance=self.collision_activation_distance,
        )
        return InverseKinematics(ik_cfg)

    def get_motion_planner(self, use_cuda_graph: bool = True) -> MotionPlanner:
        cache = {"mesh": max(16, len(getattr(self._planning_scene, "mesh", ()) or ())), "primitive": max(64, len(self.movables) + len(self.statics) + 8)}
        motion_cfg = MotionPlannerCfg.create(
            robot=self._planner_robot_cfg(),
            scene_model=self._runtime_scene,
            device_cfg=self.device_cfg,
            collision_cache=cache,
            use_cuda_graph=use_cuda_graph,
            num_ik_seeds=12,
            num_trajopt_seeds=4,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            optimizer_collision_activation_distance=self.collision_activation_distance,
        )
        return MotionPlanner(motion_cfg)

    def plan_result_success(self, result: object | None) -> bool:
        success = getattr(result, "success", None)
        if isinstance(success, bool):
            return success
        return isinstance(success, torch.Tensor) and bool(success.any().item())

    def _goal_tool_pose(self, desired_world_from_ee: torch.Tensor) -> GoalToolPose:
        desired_world_from_tool = desired_world_from_ee @ self.tool_from_ee
        desired_pose = Pose.from_matrix(
            desired_world_from_tool.to(device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        )
        return GoalToolPose.from_poses(
            {self.tool_frame: desired_pose},
            ordered_tool_frames=[self.tool_frame],
            num_goalset=1,
        )

    def solve_pose(
        self,
        desired_world_from_ee: torch.Tensor,
        *,
        current_state: JointState | None = None,
        seed_config: torch.Tensor | None = None,
        return_seeds: int = 1,
    ):
        desired_world_from_ee = self.device_cfg.to_device(desired_world_from_ee)
        if desired_world_from_ee.ndim == 2:
            desired_world_from_ee = desired_world_from_ee.unsqueeze(0)
        current_state = None if current_state is None else self.ensure_joint_state(current_state)
        if seed_config is not None:
            seed_config = self.device_cfg.to_device(seed_config)
            if seed_config.ndim == 1:
                seed_config = seed_config.view(1, 1, -1)
            elif seed_config.ndim == 2:
                seed_config = seed_config[:, None, :]
            elif seed_config.ndim != 3:
                raise ValueError(
                    f"Expected seed_config with shape [dof], [batch, dof], or [batch, num_seeds, dof], got {tuple(seed_config.shape)}"
                )

        return self.ik_solver.solve_pose(
            self._goal_tool_pose(desired_world_from_ee),
            current_state=current_state,
            seed_config=seed_config,
            return_seeds=return_seeds,
        )

    def _set_linear_motion_criteria(self, axis: str | None):
        if axis is None:
            self.motion_planner.update_tool_pose_criteria(self._default_tool_pose_criteria)
            return
        criteria = {
            tool_frame: ToolPoseCriteria.linear_motion(axis=axis)
            for tool_frame in self.motion_planner.tool_frames
        }
        self.motion_planner.update_tool_pose_criteria(criteria)

    def plan_pose(
        self,
        start_js: JointState,
        desired_world_from_ee: torch.Tensor,
        *,
        linear_axis: str | None = None,
        allow_detached_retry: bool = False,
        obstacle_name: str | None = None,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ):
        start_js = self.ensure_joint_state(start_js)

        def run_once():
            with self.disabled_scene_obstacle(obstacle_name):
                self._set_linear_motion_criteria(linear_axis)
                try:
                    return self.motion_planner.plan_pose(
                        self._goal_tool_pose(desired_world_from_ee),
                        start_js,
                        max_attempts=max_attempts,
                        enable_graph_attempt=enable_graph_attempt,
                    )
                finally:
                    self._set_linear_motion_criteria(None)

        result = run_once()
        if not self.plan_result_success(result) and allow_detached_retry:
            with self.detached_attached_object():
                result = run_once()
        return result

    def plan_cspace(
        self,
        start_js: JointState,
        goal_js: JointState,
        *,
        allow_detached_retry: bool = False,
        obstacle_name: str | None = None,
        max_attempts: int = 5,
        enable_graph_attempt: int = 1,
    ):
        start_js = self.ensure_joint_state(start_js)
        goal_js = self.ensure_joint_state(goal_js)

        def run_once():
            with self.disabled_scene_obstacle(obstacle_name):
                return self.motion_planner.plan_cspace(
                    goal_js,
                    start_js,
                    max_attempts=max_attempts,
                    enable_graph_attempt=enable_graph_attempt,
                )

        result = run_once()
        if not self.plan_result_success(result) and allow_detached_retry:
            with self.detached_attached_object():
                result = run_once()
        return result

    def update_world(self, scene: Scene):
        self._runtime_scene = scene.clone()
        self.motion_planner.update_world(self._runtime_scene)
        self.ik_solver.update_world(self._runtime_scene)
        self.robot_collision_checker.update_world(self._runtime_scene)

    def reset_runtime_scene(self, obj_to_pose: dict[str, torch.Tensor]):
        scene = self._planning_scene.clone()
        for obj_name, obj_pose in obj_to_pose.items():
            obstacle = scene.get_obstacle(obj_name)
            if obstacle is None:
                continue
            obstacle.pose = Pose.from_matrix(obj_pose).tolist()
        self.motion_planner.attachment_manager.detach()
        self._attached_object_name = None
        self._attached_joint_state = None
        self.update_world(scene)

    def update_object_pose(self, obj_name: str, obj_pose: torch.Tensor):
        updated_scene = self._runtime_scene.clone()
        obstacle = updated_scene.get_obstacle(obj_name)
        if obstacle is None:
            raise ValueError(f"Obstacle '{obj_name}' not found in runtime scene")
        obstacle.pose = Pose.from_matrix(obj_pose).tolist()
        self.update_world(updated_scene)

    def attach_scene_object(self, joint_state: JointState, object_name: str):
        self.motion_planner.attachment_manager.attach_from_scene(
            joint_states=self.ensure_joint_state(joint_state),
            obstacle_names=[object_name],
            link_name="attached_object",
            num_spheres=int(self.get_collision_spheres(object_name).shape[0]),
            surface_radius=0.005,
        )
        self._attached_object_name = object_name
        self._attached_joint_state = self.ensure_joint_state(joint_state).clone()

    def detach_attached_object(self, enable_obstacle_names: list[str] | None = None):
        self.motion_planner.attachment_manager.detach(enable_obstacle_names=enable_obstacle_names)
        self._attached_object_name = None
        self._attached_joint_state = None


def check_tamp_world_not_in_collision(
    world: TAMPWorld,
    collision_tol: float = 1e-6,
    movable_activation_dist: float = 0.0,
):
    """Check that movable objects are not initially in collision."""
    for obj in world.movables:
        obj_pose = Pose.from_list(obj.pose, world.device_cfg).get_matrix()[0]
        spheres = transform_spheres(world.get_collision_spheres(obj), obj_pose)[None, None].contiguous()
        coll_cost = world.collision_fn(spheres).sum()
        if coll_cost > collision_tol:
            _log.warning("Initial state in collision for object '%s' with cost %s", obj.name, coll_cost)

    obj_to_spheres = {
        obj.name: transform_spheres(world.get_collision_spheres(obj), world.get_object_pose(obj))
        for obj in world.movables
    }
    for obj_1, obj_2 in itertools.combinations(world.movables, 2):
        coll_cost = sphere_to_sphere_overlap(
            obj_to_spheres[obj_1.name],
            obj_to_spheres[obj_2.name],
            activation_distance=movable_activation_dist,
            use_aabb_check=True,
        )
        if coll_cost > collision_tol:
            _log.warning(
                "Initial state in collision between %s and %s with cost %s",
                obj_1.name,
                obj_2.name,
                coll_cost,
            )
