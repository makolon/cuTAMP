# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import List, Dict, TypedDict

import torch
from jaxtyping import Float

from curobo.types import Pose
from cutamp.utils.common import (
    Particles,
    action_6dof_to_mat4x4,
    action_4dof_to_mat4x4,
    filter_valid_spheres,
)
from cutamp.config import TAMPConfiguration
from cutamp.tamp_domain import MoveFree, MoveHolding, Pick, Place, Push, Conf
from cutamp.tamp_world import (
    TAMPWorld,
)
from cutamp.task_planning import PlanSkeleton


def get_conf_parameters(plan_skeleton: PlanSkeleton, ignore_initial: bool) -> List[str]:
    """Get the parameters of the plan skeletons that are of type Conf. Returns a list of unique parameter names."""
    conf_params = []
    for ground_op in plan_skeleton:
        conf_idxs = [idx for idx, param in enumerate(ground_op.operator.parameters) if param.type == Conf]
        op_conf_params = [ground_op.values[idx] for idx in conf_idxs]
        conf_params.extend(op_conf_params)
    conf_params = list(dict.fromkeys(conf_params))  # remove duplicates

    if ignore_initial:
        assert conf_params[0] == "q0", "Expected first configuration to be q0"
        conf_params = conf_params[1:]  # remove the initial configuration
    return conf_params


class Rollout(TypedDict):
    """Dict that stores results of a rollout."""

    num_particles: int
    confs: Float[torch.Tensor, "num_particles *h d"]
    conf_params: List[str]
    robot_spheres: Float[torch.Tensor, "num_particles *h 4"]
    collision_robot_spheres: Float[torch.Tensor, "num_particles *h 4"]
    ee_position: Float[torch.Tensor, "num_particles t 3"]
    ee_quaternion: Float[torch.Tensor, "num_particles t 4"]
    world_from_tool_desired: Float[torch.Tensor, "num_particles *h 4 4"]
    world_from_ee_desired: Float[torch.Tensor, "num_particles *h 4 4"]
    gripper_close: List[bool]
    action_params: List[str]
    grasp_confidences: Dict[str, Float[torch.Tensor, "num_particles"]]
    obj_to_pose: Dict[str, Float[torch.Tensor, "num_particles *h 4 4"]]
    action_to_ts: Dict[str, int]
    action_to_pose_ts: Dict[str, int]
    ts_to_pose_ts: Dict[int, int]


class RolloutFunction:
    """Rollout function that rolls out the plan skeleton. Only supports robot and object kinematics right now."""

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectories are not supported in rollouts yet")
        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self.conf_params = get_conf_parameters(plan_skeleton, ignore_initial=True)
        self.obj_to_initial_pose = {obj.name: self.world.get_object_pose(obj) for obj in self.world.movables}

        # Grasp to 4x4 matrix function
        if config.grasp_dof == 4:
            self.grasp_to_mat4x4_fn = action_4dof_to_mat4x4
        elif config.grasp_dof == 6:
            self.grasp_to_mat4x4_fn = action_6dof_to_mat4x4
        else:
            raise ValueError(f"Unsupported {config.grasp_dof=}")

        # Place to 4x4 matrix function
        if config.place_dof == 4:
            self.place_to_mat4x4_fn = action_4dof_to_mat4x4
        elif config.place_dof == 6:
            self.place_to_mat4x4_fn = action_6dof_to_mat4x4
        else:
            raise ValueError(f"Unsupported {config.place_dof=}")
        
        # Push to 4x4 matrix function
        if config.push_dof == 4:
            self.push_to_mat4x4_fn = action_4dof_to_mat4x4
        elif config.push_dof == 6:
            self.push_to_mat4x4_fn = action_6dof_to_mat4x4
        else:
            raise ValueError(f"Unsupported {config.push_dof=}")

        # Flag for first rollout, used to apply a runtime check
        self._is_first_rollout = True

    def __call__(self, particles: Particles) -> Rollout:
        """
        Rollout particles given the plan skeleton through the world. We keep the rollout information minimal to avoid
        unnecessary computations and backward passes.
        """
        num_particles = particles["q0"].shape[0]

        # Forward kinematics and collision spheres from the same collision-model state.
        with torch.profiler.record_function("rollout::forward_kinematics"):
            confs = torch.stack([particles[conf] for conf in self.conf_params], dim=1)
            self.world.robot_collision_checker.setup_batch_tensors(num_particles, confs.shape[1])
            robot_state = self.world.robot_collision_checker.get_kinematics(confs)

            # cuRobo exposes the configured tool/TCP pose; convert it back to the execution
            # end-effector frame that the grasp/place streams target.
            tool_pose = robot_state.tool_poses.get_link_pose(self.world.tool_frame)
            ee_pose = Pose.from_matrix(tool_pose.get_matrix() @ self.world.ee_from_tool)
            ee_position = ee_pose.position.view(num_particles, confs.shape[1], 3)
            ee_quaternion = ee_pose.quaternion.view(num_particles, confs.shape[1], 4)

            robot_spheres = robot_state.get_link_spheres()
            collision_robot_spheres = filter_valid_spheres(robot_spheres)

        # Stores the desired actions
        world_from_ee_desired = []
        gripper_close: List[bool] = []
        action_params: List[str] = []
        action_to_ts: Dict[str, int] = {}
        grasp_confidences: Dict[str, Float[torch.Tensor, "num_particles"]] = {}

        # For pose timestamp (pose_ts), we only accumulate the poses if the operator causes a change in the object pose.
        # These dicts are used to map actions and timestamps to their corresponding pose timestamps.
        action_to_pose_ts: Dict[str, int] = {}
        ts_to_pose_ts: Dict[int, int] = {}

        # 4x4 transformation matrices for grasp parameters (if any)
        grasp_to_mat4x4: Dict[str, Float[torch.Tensor, "num_particles 4 4"]] = {}

        def get_grasp_mat4x4(grasp_name_: str) -> Float[torch.Tensor, "num_particles 4 4"]:
            if grasp_name_ not in grasp_to_mat4x4:
                grasp_ = particles[grasp_name_]
                # grasp_ has shape (n, 4, 4) or (n, 4) or (n, 6)
                if grasp_.shape[1:3] == (4, 4):
                    grasp_to_mat4x4[grasp_name_] = grasp_
                else:
                    grasp_to_mat4x4[grasp_name_] = self.grasp_to_mat4x4_fn(grasp_)
            return grasp_to_mat4x4[grasp_name_]

        # Object poses in world frame for every timestep
        obj_to_pose = {
            obj.name: [self.obj_to_initial_pose[obj.name].expand(num_particles, -1, -1)] for obj in self.world.movables
        }

        def current_pose(obj: str) -> Float[torch.Tensor, "num_particles 4 4"]:
            return obj_to_pose[obj][-1]

        # Timestep of all actionable operators, and the corresponding timestep in the obj_to_pose list
        ts, pose_ts = 0, 0

        # Rollout each ground operator in the plan skeleton
        for ground_op in self.plan_skeleton:
            op_name = ground_op.operator.name

            # Skip MoveFree and MoveHolding as we don't support trajectories yet
            if op_name == MoveFree.name or op_name == MoveHolding.name:
                continue

            # Pick
            elif op_name == Pick.name:
                obj_name, grasp_name, _ = ground_op.values
                confidence_key = f"{grasp_name}_confidences"
                if confidence_key in particles:
                    grasp_confidences[grasp_name] = particles[confidence_key]
                # Grasp is in object frame
                obj_from_grasp = get_grasp_mat4x4(grasp_name)

                # Grasp poses are planner ee poses expressed in the object frame.
                world_from_obj = current_pose(obj_name)
                world_from_ee = world_from_obj @ obj_from_grasp

                world_from_ee_desired.append(world_from_ee)
                gripper_close.append(True)  # closing gripper at Pick
                action_params.append(grasp_name)
                action_to_ts[grasp_name] = ts
                action_to_pose_ts[grasp_name] = pose_ts

            # Place
            elif op_name == Place.name:
                obj_name, grasp_name, place_name, _, _ = ground_op.values

                # Place is desired object pose in world frame
                place_action = particles[place_name]
                world_from_obj = self.place_to_mat4x4_fn(place_action)

                # Grasp poses are planner ee poses expressed in the object frame.
                obj_from_grasp = get_grasp_mat4x4(grasp_name)
                world_from_ee = world_from_obj @ obj_from_grasp

                # Accumulate poses of all the movable objects as we've moved the object
                for obj in self.world.movables:
                    if obj.name == obj_name:
                        obj_to_pose[obj.name].append(world_from_obj)  # use new desired pose
                    else:
                        obj_to_pose[obj.name].append(current_pose(obj.name))
                pose_ts += 1

                world_from_ee_desired.append(world_from_ee)
                gripper_close.append(False)  # opening gripper at Place
                action_params.append(place_name)
                action_to_ts[place_name] = ts
                action_to_pose_ts[place_name] = pose_ts
                ts_to_pose_ts[ts] = pose_ts

            # Push
            elif op_name == Push.name:
                button_name, pose_name, _ = ground_op.values

                # Push poses are sampled directly in the planner ee frame.
                push_action = particles[pose_name]
                world_from_ee = self.push_to_mat4x4_fn(push_action)

                world_from_ee_desired.append(world_from_ee)
                gripper_close.append(True)  # close gripper at Push
                action_params.append(pose_name)
                action_to_ts[pose_name] = ts
                action_to_pose_ts[pose_name] = pose_ts

            # Unknown
            else:
                raise ValueError(f"Unsupported operator {op_name}")

            # Increment time step
            ts_to_pose_ts[ts] = pose_ts
            ts += 1

        # Stack and store in rollout
        world_from_ee_desired = torch.stack(world_from_ee_desired, dim=1)
        world_from_tool_desired = world_from_ee_desired @ self.world.tool_from_ee

        # Object poses for each timestep
        obj_to_pose = {k: torch.stack(v, dim=1) for k, v in obj_to_pose.items()}

        # Sanity check
        if self._is_first_rollout:
            assert (
                confs.shape[1]
                == ee_position.shape[1]
                == ee_quaternion.shape[1]
                == world_from_ee_desired.shape[1]
                == world_from_tool_desired.shape[1]
                == ts
            ), "Number of timesteps do not match"
            for obj, poses in obj_to_pose.items():
                assert poses.shape[1] == pose_ts + 1, f"Number of pose timesteps do not match for {obj}"
            self._is_first_rollout = False

        rollout = Rollout(
            num_particles=num_particles,
            confs=confs,
            conf_params=self.conf_params,
            robot_spheres=robot_spheres,
            collision_robot_spheres=collision_robot_spheres,
            ee_position=ee_position,
            ee_quaternion=ee_quaternion,
            world_from_tool_desired=world_from_tool_desired,
            world_from_ee_desired=world_from_ee_desired,
            gripper_close=gripper_close,
            action_params=action_params,
            grasp_confidences=grasp_confidences,
            obj_to_pose=obj_to_pose,
            action_to_ts=action_to_ts,
            action_to_pose_ts=action_to_pose_ts,
            ts_to_pose_ts=ts_to_pose_ts,
        )
        return rollout
