# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Solving motions with cuRobo."""

import logging
from typing import List

import torch

from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Sphere
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from cutamp.config import TAMPConfiguration
from cutamp.optimize_plan import PlanContainer
from cutamp.tamp_domain import MoveHolding, Push, MoveFree, Place, Pick
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.common import Particles, action_6dof_to_mat4x4, action_4dof_to_mat4x4
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


def _mat4_to_list(mat4: torch.Tensor) -> list[list[float]]:
    mat = mat4.detach().cpu().float().view(4, 4)
    return [[float(value) for value in row] for row in mat.tolist()]


def _pose_error(actual: torch.Tensor, desired: torch.Tensor) -> tuple[float, float]:
    actual_mat = actual.detach().cpu().float().view(4, 4)
    desired_mat = desired.detach().cpu().float().view(4, 4)
    translation_error = float(torch.linalg.norm(actual_mat[:3, 3] - desired_mat[:3, 3]).item())
    delta = actual_mat[:3, :3] @ desired_mat[:3, :3].T
    trace = float(torch.trace(delta).item())
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    rotation_error = float(torch.rad2deg(torch.arccos(torch.tensor(cos_theta))).item())
    return translation_error, rotation_error


def _segment_debug_payload(
    *,
    label: str,
    segment_type: str,
    desired_world_from_ee: torch.Tensor,
    terminal_world_from_ee: torch.Tensor,
    plan_positions: torch.Tensor | None = None,
    waypoint_index_offset: int = 0,
    selected_parameter_name: str | None = None,
    selected_obj_from_grasp: torch.Tensor | None = None,
    object_name: str | None = None,
) -> dict[str, object]:
    translation_error, rotation_error = _pose_error(terminal_world_from_ee, desired_world_from_ee)
    payload: dict[str, object] = {
        "label": str(label),
        "segment_type": str(segment_type),
        "desired_world_from_ee": _mat4_to_list(desired_world_from_ee),
        "terminal_world_from_fk": _mat4_to_list(terminal_world_from_ee),
        "terminal_translation_error_m": float(translation_error),
        "terminal_rotation_error_deg": float(rotation_error),
        "waypoint_index_offset": int(waypoint_index_offset),
    }
    if plan_positions is not None:
        payload["num_waypoints"] = int(plan_positions.shape[0])
    if selected_parameter_name:
        payload["selected_parameter_name"] = str(selected_parameter_name)
    if selected_obj_from_grasp is not None:
        payload["selected_obj_from_grasp"] = _mat4_to_list(selected_obj_from_grasp)
    if object_name:
        payload["object_name"] = str(object_name)
    return payload


def solve_curobo(
    plan_info: PlanContainer,
    best_particle: Particles,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    visualizer: Visualizer,
    obj_to_initial_pose: dict[str, torch.Tensor],
    timeline: str = "curobo",
):
    """
    Solve for full motion plan given a plan skeleton and optimized particles.
    Note that visualization adds non-trivial overhead.
    """
    plan_skeleton = plan_info["plan_skeleton"]
    motion_gen = world.motion_gen
    if config.warmup_motion_gen:
        with timer.time(f"{timeline}_motion_gen_warmup", log_callback=_log.debug):
            world.warmup_motion_gen()

    plan_config = MotionGenPlanConfig(
        timeout=0.25, enable_finetune_trajopt=False, time_dilation_factor=config.time_dilation_factor
    )

    # Log initial state
    ts = 0.0
    obj_to_current_pose = {k: v.clone() for k, v in obj_to_initial_pose.items()}

    # Reset motion gen, clear attachments and reset pose of all objects
    motion_gen.detach_object_from_robot("attached_object")
    for obj, obj_pose in obj_to_current_pose.items():
        motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)
        obj_pose = obj_to_current_pose[obj]
        motion_gen.world_collision.update_obstacle_pose(obj, Pose.from_matrix(obj_pose))

    visualizer.set_time_seconds(timeline, ts)
    visualizer.set_joint_positions(best_particle["q0"])
    for obj, pose in obj_to_current_pose.items():
        visualizer.log_mat4x4(f"world/{obj}", pose)

    last_js = JointState.from_position(best_particle["q0"][None].clone())
    last_q_name = "q0"

    # Fixed approach offset. This could be something we eventually optimize too
    approach_offset = torch.eye(4, device=world.device)
    approach_offset[2, 3] = -0.05

    approach_offsets = torch.eye(4, device=world.device).repeat(4, 1, 1)
    approach_offsets[:, 2, 3] = torch.tensor([-0.05, -0.1, -0.15, -0.2], device=world.device)

    constrained_motion_cost_metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=world.tensor_args.to_device([0.1, 0.1, 0.1, 0.1, 0.1, 0.0]),
        project_to_goal_frame=True,
    )
    constrained_plan_config = plan_config.clone()
    constrained_plan_config.pose_cost_metric = constrained_motion_cost_metric

    # Accumulated plans we return that the real robot can actually execute
    accum_plans = []
    waypoint_index_offset = 0

    # Iterate through skeleton and motion plan
    for idx, ground_op in enumerate(plan_skeleton):
        op_name = ground_op.operator.name
        print(f"{idx + 1}. {ground_op.name}")

        # MoveFree, defer motion planning to pick to use object pose instead of planning from q_start to q_end.
        # This works more reliably and gives higher quality motions.
        if op_name == MoveFree.name:
            q_start, traj, q_end = ground_op.values
            if traj in best_particle:
                raise NotImplementedError("Trajectories not supported yet")
            last_q_name = q_start

        # MoveHolding
        elif op_name == MoveHolding.name:
            obj, grasp, q_start, traj, q_end = ground_op.values
            if traj in best_particle:
                raise NotImplementedError("Trajectories not supported yet")
            last_q_name = q_start

        # Pick
        elif op_name == Pick.name:
            obj, grasp, q = ground_op.values
            assert last_js is not None

            with timer.time(f"{timeline}_planning"):
                start_js = last_js
                world_from_start_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                world_from_retract = world_from_start_ee

                # Get the retract pose and plan to it if it's not q0
                if last_q_name != "q0":
                    world_from_retract = world_from_start_ee @ approach_offset
                    retract_result = motion_gen.plan_single(
                        start_js, Pose.from_matrix(world_from_retract), constrained_plan_config
                    )
                    if not retract_result.success:
                        raise RuntimeError(
                            f"Failed to plan for retract for {ground_op.name}. Status: {retract_result.status}"
                        )
                    retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                else:
                    retract_result = None
                    retract_js = start_js

                # Get the approach pose and plan to it
                world_from_obj = obj_to_current_pose[obj]
                if best_particle[grasp].shape == (4, 4):  # already a 4x4, probably came from M2T2
                    obj_from_grasp = best_particle[grasp].clone()
                elif config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp].clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(best_particle[grasp].clone())
                world_from_grasp = world_from_obj @ obj_from_grasp
                world_from_ee = world_from_grasp @ world.tool_from_ee

                world_from_approach = world_from_ee @ approach_offset
                approach_result = motion_gen.plan_single(retract_js, Pose.from_matrix(world_from_approach), plan_config)
                if not approach_result.success:
                    raise RuntimeError(
                        f"Failed to plan for approach for {ground_op.name}. Status: {approach_result.status}"
                    )

                # Plan to from approach to target EE pose for grasp
                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                end_result = motion_gen.plan_single(
                    approach_js, Pose.from_matrix(world_from_ee), constrained_plan_config
                )
                if not end_result.success:
                    raise RuntimeError(
                        f"Failed to plan from approach to end for {ground_op.name}. Status: {end_result.status}"
                    )

            ik_fk = world.kin_model.get_state(best_particle[q][None]).ee_pose.get_matrix()[0]
            ik_translation_error, ik_rotation_error = _pose_error(ik_fk, world_from_ee)
            ik_debug = {
                "label": ground_op.name,
                "segment_type": "ik_pick",
                "selected_parameter_name": str(grasp),
                "object_name": str(obj),
                "desired_world_from_ee": _mat4_to_list(world_from_ee),
                "ik_world_from_fk": _mat4_to_list(ik_fk),
                "translation_error_m": float(ik_translation_error),
                "rotation_error_deg": float(ik_rotation_error),
                "success": True,
                "selected_obj_from_grasp": _mat4_to_list(obj_from_grasp),
            }

            segment_specs = [
                ("pick_retract", retract_result, world_from_retract),
                ("pick_approach", approach_result, world_from_approach),
                ("pick_grasp", end_result, world_from_ee),
            ]
            for segment_type, result, desired_world_from_ee in segment_specs:
                if result is None:
                    continue
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                terminal_world_from_fk = world.kin_model.get_state(plan.position[-1:]).ee_pose.get_matrix()[0]
                debug_payload = _segment_debug_payload(
                    label=ground_op.name,
                    segment_type=segment_type,
                    desired_world_from_ee=desired_world_from_ee,
                    terminal_world_from_ee=terminal_world_from_fk,
                    plan_positions=plan.position,
                    waypoint_index_offset=waypoint_index_offset,
                    selected_parameter_name=str(grasp),
                    selected_obj_from_grasp=obj_from_grasp,
                    object_name=str(obj),
                )
                if segment_type == "pick_retract":
                    debug_payload["ik_debug"] = ik_debug
                accum_plans.append(
                    {
                        "type": "trajectory",
                        "plan": plan,
                        "dt": dt,
                        "optimized_plan": result.optimized_plan,
                        "optimized_dt": result.optimized_dt,
                        "label": ground_op.name,
                        "debug": debug_payload,
                    }
                )
                last_js = JointState.from_position(plan[-1:].position)
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
                waypoint_index_offset += int(plan.position.shape[0])

            # Temporarily monkey patch get_bounding_spheres to return the spheres we sampled
            obstacle = motion_gen.world_model.get_obstacle(obj)
            obstacle.old_get_bounding_spheres = obstacle.get_bounding_spheres

            def get_bounding_spheres(self, *args, **kwargs) -> List[Sphere]:
                spheres = world.get_collision_spheres(obj)
                pts = spheres[:, :3].cpu().numpy()
                n_radius = spheres[:, 3].cpu().numpy()

                obj_pose = Pose.from_matrix(obj_to_current_pose[obj])
                pre_transform_pose = kwargs["pre_transform_pose"]
                if pre_transform_pose is not None:
                    obj_pose = pre_transform_pose.multiply(obj_pose)  # convert object pose to another frame

                if pts is None or len(pts) == 0:
                    raise ValueError("No points found from the spheres")

                points_cuda = self.tensor_args.to_device(pts)
                pts = obj_pose.transform_points(points_cuda).cpu().view(-1, 3).numpy()

                new_spheres = [
                    Sphere(
                        name=f"{self.name}_sph_{i}",
                        pose=[pts[i, 0], pts[i, 1], pts[i, 2], 1, 0, 0, 0],
                        radius=n_radius[i],
                    )
                    for i in range(pts.shape[0])
                ]
                return new_spheres

            obstacle.get_bounding_spheres = get_bounding_spheres.__get__(obstacle)

            # Attach the object to the robot
            with timer.time(f"{timeline}_planning"):
                motion_gen.attach_objects_to_robot(
                    last_js,
                    object_names=[obj],
                    surface_sphere_radius=0.005,
                    sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                    voxelize_method="subdivide",
                )

            obstacle.get_bounding_spheres = obstacle.old_get_bounding_spheres
            del obstacle.old_get_bounding_spheres

            # TODO: Fix this hardcoding
            # Close the gripper in the visualization
            if "ur5" in config.robot or "robotiq_2f_85" in config.robot:
                end_val = 0.4
                interp = torch.linspace(0.0, end_val, 20)
                interp = interp[:, None]
            else:
                end_val = 0.02
                interp = torch.linspace(0.04, end_val, 20)[:, None]
                interp = interp.repeat(1, 2)
            dt = 0.02
            accum_plans.append({"type": "gripper", "action": "close", "label": ground_op.name})

            all_pos = last_js.position.expand(interp.shape[0], -1).cpu()
            all_pos = torch.cat([all_pos, interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=dt)

        # Place
        elif op_name == Place.name:
            obj, grasp, placement, surface, q = ground_op.values
            assert last_js is not None

            with timer.time(f"{timeline}_planning"):
                start_js = last_js

                # Plan to retract
                world_from_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                world_from_ee_start = world_from_ee
                world_from_retract = world_from_ee @ approach_offset
                retract_result = motion_gen.plan_single(
                    start_js, Pose.from_matrix(world_from_retract), constrained_plan_config
                )
                if not retract_result.success:
                    if (
                        retract_result.status is not None
                        and retract_result.status.name == "INVALID_START_STATE_WORLD_COLLISION"
                    ):
                        kin_config = motion_gen.kinematics.kinematics_config
                        link_name = "attached_object"
                        curr_obj_sphs = kin_config.get_link_spheres(link_name).clone()
                        kin_config.detach_object(link_name)
                        retract_result = motion_gen.plan_single(
                            start_js, Pose.from_matrix(world_from_retract), plan_config
                        )
                        kin_config.attach_object(sphere_tensor=curr_obj_sphs, link_name=link_name)
                        if not retract_result.success:
                            raise RuntimeError(
                                f"Failed to plan for retract for {ground_op.name}. Status: {retract_result.status}"
                            )
                    else:
                        raise RuntimeError(
                            f"Failed to plan for retract for {ground_op.name}. Status: {retract_result.status}"
                        )

                # Plan from retract to approach
                retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                world_from_obj = action_4dof_to_mat4x4(best_particle[placement].clone())
                if best_particle[grasp].shape == (4, 4):  # already a 4x4, probably came from M2T2
                    obj_from_grasp = best_particle[grasp].clone()
                elif config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp].clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(best_particle[grasp].clone())
                world_from_grasp = world_from_obj @ obj_from_grasp
                world_from_ee = world_from_grasp @ world.tool_from_ee
                world_from_approaches = world_from_ee @ approach_offsets
                for app_idx, world_from_approach in enumerate(world_from_approaches):
                    approach_result = motion_gen.plan_single(
                        retract_js, Pose.from_matrix(world_from_approach), plan_config
                    )
                    _log.debug(
                        f"Approach attempt {app_idx + 1}/{len(world_from_approaches)}. {approach_result.success}"
                    )
                    if approach_result.success:
                        break

                if not approach_result.success:
                    raise RuntimeError(
                        f"Failed to plan for approach for {ground_op.name}. Status: {approach_result.status}"
                    )

                # Plan from approach to end js
                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                end_result = motion_gen.plan_single(
                    approach_js, Pose.from_matrix(world_from_ee), constrained_plan_config
                )
                if not end_result.success:
                    raise RuntimeError(
                        f"Failed to plan from approach to end for {ground_op.name}. Status: {end_result.status}"
                    )

            ik_fk = world.kin_model.get_state(best_particle[q][None]).ee_pose.get_matrix()[0]
            ik_translation_error, ik_rotation_error = _pose_error(ik_fk, world_from_ee)
            ik_debug = {
                "label": ground_op.name,
                "segment_type": "ik_place",
                "selected_parameter_name": str(grasp),
                "object_name": str(obj),
                "desired_world_from_ee": _mat4_to_list(world_from_ee),
                "ik_world_from_fk": _mat4_to_list(ik_fk),
                "translation_error_m": float(ik_translation_error),
                "rotation_error_deg": float(ik_rotation_error),
                "success": True,
                "selected_obj_from_grasp": _mat4_to_list(obj_from_grasp),
            }

            # Compute the offset between the object and end-effector at start of plan
            obj_from_ee = torch.inverse(obj_to_current_pose[obj]) @ world_from_ee_start
            ee_from_obj = torch.inverse(obj_from_ee)

            segment_specs = [
                ("place_retract", retract_result, world_from_retract),
                ("place_approach", approach_result, world_from_approach),
                ("place_place", end_result, world_from_ee),
            ]
            for segment_type, result, desired_world_from_ee in segment_specs:
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                terminal_world_from_fk = world.kin_model.get_state(plan.position[-1:]).ee_pose.get_matrix()[0]
                debug_payload = _segment_debug_payload(
                    label=ground_op.name,
                    segment_type=segment_type,
                    desired_world_from_ee=desired_world_from_ee,
                    terminal_world_from_ee=terminal_world_from_fk,
                    plan_positions=plan.position,
                    waypoint_index_offset=waypoint_index_offset,
                    selected_parameter_name=str(grasp),
                    selected_obj_from_grasp=obj_from_grasp,
                    object_name=str(obj),
                )
                if segment_type == "place_retract":
                    debug_payload["ik_debug"] = ik_debug
                accum_plans.append(
                    {
                        "type": "trajectory",
                        "plan": plan,
                        "dt": dt,
                        "optimized_plan": result.optimized_plan,
                        "optimized_dt": result.optimized_dt,
                        "label": ground_op.name,
                        "debug": debug_payload,
                    }
                )
                last_js = JointState.from_position(plan[-1:].position)

                # Forward kinematics to get end-effector pose
                robot_state = world.kin_model.get_state(plan.position)
                world_from_ee = robot_state.ee_pose.get_matrix()
                world_from_obj = world_from_ee @ ee_from_obj
                ts = visualizer.log_joint_trajectory_with_mat4x4(
                    traj=plan.position,
                    mat4x4_key=f"world/{obj}",
                    mat4x4=world_from_obj,
                    timeline=timeline,
                    start_time=ts,
                    dt=dt,
                )
                waypoint_index_offset += int(plan.position.shape[0])

                # Updated pose is the last pose
                obj_to_current_pose[obj] = world_from_obj[-1]

            # Detach object from robot and enable it again
            with timer.time(f"{timeline}_planning"):
                motion_gen.detach_object_from_robot("attached_object")
                motion_gen.world_coll_checker.enable_obstacle(enable=True, name=obj)
                obj_pose = obj_to_current_pose[obj]
                motion_gen.world_collision.update_obstacle_pose(obj, Pose.from_matrix(obj_pose))

            # Open the gripper for visualization purposes
            if "ur5" in config.robot or "robotiq_2f_85" in config.robot:
                end_val = 0.0
                interp = torch.linspace(0.4, end_val, 20)
                interp = interp[:, None]
            else:
                end_val = 0.04
                interp = torch.linspace(0.02, end_val, 20)[:, None]
                interp = interp.repeat(1, 2)
            dt = 0.02
            accum_plans.append({"type": "gripper", "action": "open", "label": ground_op.name})

            all_pos = last_js.position.expand(interp.shape[0], -1).cpu()
            all_pos = torch.cat([all_pos, interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=dt)

        # Push
        elif op_name == Push.name:
            button, pose, _ = ground_op.values
            assert last_js is not None

            if "ur5" in config.robot or "robotiq_2f_85" in config.robot:
                end_val = 0.4
                interp = torch.linspace(0.0, end_val, 20)
                interp = interp[:, None]
            else:
                end_val = 0.02
                interp = torch.linspace(0.04, end_val, 20)[:, None]
                interp = interp.repeat(1, 2)
            dt = 0.02
            accum_plans.append({"type": "gripper", "action": "close", "label": ground_op.name})

            all_pos = last_js.position.expand(interp.shape[0], -1).cpu()
            all_pos = torch.cat([all_pos, interp], dim=1)
            ts = visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=dt)

            with timer.time(f"{timeline}_planning"):
                start_js = last_js

                if last_q_name != "q0":
                    world_from_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
                    world_from_retract = world_from_ee @ approach_offset
                    retract_result = motion_gen.plan_single(
                        start_js, Pose.from_matrix(world_from_retract), constrained_plan_config
                    )
                    if not retract_result.success:
                        raise RuntimeError(
                            f"Failed to plan for retract for {ground_op.name}. Status: {retract_result.status}"
                        )
                    retract_js = JointState.from_position(retract_result.get_interpolated_plan().position[-1:])
                else:
                    retract_result = None
                    retract_js = start_js

                if best_particle[pose].shape == (4, 4):
                    world_from_push = best_particle[pose].clone()
                elif config.push_dof == 4:
                    world_from_push = action_4dof_to_mat4x4(best_particle[pose].clone())
                else:
                    world_from_push = action_6dof_to_mat4x4(best_particle[pose].clone())
                world_from_ee = world_from_push @ world.tool_from_ee

                world_from_approach = world_from_ee @ approach_offset
                approach_result = motion_gen.plan_single(retract_js, Pose.from_matrix(world_from_approach), plan_config)
                if not approach_result.success:
                    raise RuntimeError(
                        f"Failed to plan for approach for {ground_op.name}. Status: {approach_result.status}"
                    )

                approach_js = JointState.from_position(approach_result.get_interpolated_plan().position[-1:])
                motion_gen.world_coll_checker.enable_obstacle(enable=False, name=button)
                try:
                    end_result = motion_gen.plan_single(
                        approach_js, Pose.from_matrix(world_from_ee), constrained_plan_config
                    )
                finally:
                    motion_gen.world_coll_checker.enable_obstacle(enable=True, name=button)
                if not end_result.success:
                    raise RuntimeError(
                        f"Failed to plan from approach to end for {ground_op.name}. Status: {end_result.status}"
                    )

            segment_specs = [
                ("push_retract", retract_result, world_from_retract if retract_result is not None else world_from_ee),
                ("push_approach", approach_result, world_from_approach),
                ("push_execute", end_result, world_from_ee),
            ]
            for segment_type, result, desired_world_from_ee in segment_specs:
                if result is None:
                    continue
                dt = result.interpolation_dt
                plan = result.get_interpolated_plan()
                terminal_world_from_fk = world.kin_model.get_state(plan.position[-1:]).ee_pose.get_matrix()[0]
                accum_plans.append(
                    {
                        "type": "trajectory",
                        "plan": plan,
                        "dt": dt,
                        "optimized_plan": result.optimized_plan,
                        "optimized_dt": result.optimized_dt,
                        "label": ground_op.name,
                        "debug": _segment_debug_payload(
                            label=ground_op.name,
                            segment_type=segment_type,
                            desired_world_from_ee=desired_world_from_ee,
                            terminal_world_from_ee=terminal_world_from_fk,
                            plan_positions=plan.position,
                            waypoint_index_offset=waypoint_index_offset,
                            selected_parameter_name=str(pose),
                        ),
                    }
                )
                last_js = JointState.from_position(plan[-1:].position)
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
                waypoint_index_offset += int(plan.position.shape[0])

        # Unsupported
        else:
            raise NotImplementedError(f"Unsupported operator {op_name}")

    start_js = last_js

    # Plan to retract
    world_from_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
    world_from_retract = world_from_ee @ approach_offset
    retract_result = motion_gen.plan_single(start_js, Pose.from_matrix(world_from_retract), constrained_plan_config)
    if not retract_result.success:
        raise RuntimeError(f"Failed to plan for retract. Status: {retract_result.status}")
    dt = retract_result.interpolation_dt
    plan = retract_result.get_interpolated_plan()
    accum_plans.append(
        {
            "type": "trajectory",
            "plan": plan,
            "dt": dt,
            "optimized_plan": result.optimized_plan,
            "optimized_dt": result.optimized_dt,
            "label": "GoToInitial(q0)",
            "debug": _segment_debug_payload(
                label="GoToInitial(q0)",
                segment_type="return_retract",
                desired_world_from_ee=world_from_retract,
                terminal_world_from_ee=world.kin_model.get_state(plan.position[-1:]).ee_pose.get_matrix()[0],
                plan_positions=plan.position,
                waypoint_index_offset=waypoint_index_offset,
            ),
        }
    )
    last_js = JointState.from_position(plan[-1:].position)
    ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
    waypoint_index_offset += int(plan.position.shape[0])

    # Plan to go home at the end which we'll assume is q0
    q_last = last_js.position[0]
    q_home = best_particle["q0"].clone()
    js_last = JointState.from_position(q_last[None])
    js_home = JointState.from_position(q_home[None])
    with timer.time(f"{timeline}_planning"):
        result = motion_gen.plan_single_js(js_last, js_home, plan_config)
    if not result.success:
        raise RuntimeError("Failed to plan for going home")

    dt = result.interpolation_dt
    plan = result.get_interpolated_plan()
    accum_plans.append(
        {
            "type": "trajectory",
            "plan": plan,
            "dt": dt,
            "optimized_plan": result.optimized_plan,
            "optimized_dt": result.optimized_dt,
            "label": "GoToInitial(q0)",
            "debug": {
                "label": "GoToInitial(q0)",
                "segment_type": "go_home_joint",
                "waypoint_index_offset": int(waypoint_index_offset),
                "num_waypoints": int(plan.position.shape[0]),
            },
        }
    )
    _ = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
    _log.debug("Planned to go home")

    _log.info(f"Motion planning metrics: {timer.get_summary(f'{timeline}_planning')}")
    return accum_plans
