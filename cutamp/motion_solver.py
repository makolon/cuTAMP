# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Solving motions with cuRobo v2.

This solver is aligned with the RobotContainer-based TAMPWorld implementation.
It does not access private scene fields such as `_planning_scene` or
`_runtime_scene`. Scene updates are routed through TAMPWorld methods.
"""

from __future__ import annotations

import logging

import torch

from cutamp.config import TAMPConfiguration
from cutamp.optimize_plan import PlanContainer
from cutamp.tamp_domain import MoveFree, MoveHolding, Pick, Place, Push
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.common import Particles, action_4dof_to_mat4x4, action_6dof_to_mat4x4
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


def solve_curobo(
    plan_info: PlanContainer,
    best_particle: Particles,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    visualizer: Visualizer,
    obj_to_initial_pose: dict[str, torch.Tensor],
    timeline: str = "curobo",
    motion_gen: object | None = None,
):
    """Convert a satisfying TAMP particle into executable robot trajectories."""
    del motion_gen

    if config.warmup_motion_gen:
        with timer.time(f"{timeline}_motion_gen_warmup", log_callback=_log.info):
            world.warmup_motion_gen()

    time_s = 0.0
    object_poses = {name: pose.clone() for name, pose in obj_to_initial_pose.items()}
    world.reset_scene(object_poses)

    visualizer.set_time_seconds(timeline, time_s)
    visualizer.set_joint_positions(best_particle["q0"])
    for obj_name, obj_pose in object_poses.items():
        visualizer.log_mat4x4(f"world/{obj_name}", obj_pose)

    last_js = world.joint_state_from_position(best_particle["q0"][None].clone())
    last_q_name = "q0"
    accum_plans: list[dict] = []

    approach_offset = torch.eye(4, device=world.device)
    approach_offset[2, 3] = -0.05

    place_approach_offsets = torch.eye(4, device=world.device).repeat(4, 1, 1)
    place_approach_offsets[:, 2, 3] = torch.tensor([-0.05, -0.1, -0.15, -0.2], device=world.device)

    for idx, ground_op in enumerate(plan_info["plan_skeleton"]):
        op_name = ground_op.operator.name
        print(f"{idx + 1}. {ground_op.name}")

        if op_name == MoveFree.name:
            q_start, traj, _ = ground_op.values
            if traj in best_particle:
                raise NotImplementedError("Trajectory variables are not supported yet")
            last_q_name = q_start
            continue

        if op_name == MoveHolding.name:
            _, _, q_start, traj, _ = ground_op.values
            if traj in best_particle:
                raise NotImplementedError("Trajectory variables are not supported yet")
            last_q_name = q_start
            continue

        if op_name == Pick.name:
            obj, grasp, q = ground_op.values

            with timer.time(f"{timeline}_planning"):
                retract_result = None
                retract_js = last_js

                if last_q_name != "q0":
                    start_ee = world.compute_ee_matrix(last_js.position)
                    retract_result = world.plan_pose(
                        last_js,
                        start_ee @ approach_offset,
                        linear_axis="z",
                    )

                    if not bool(torch.as_tensor(retract_result.success).any().item()):
                        raise RuntimeError(
                            f"Failed to plan pick retract for {ground_op.name}. "
                            f"Status: {retract_result.status}"
                        )

                    retract_plan = retract_result.get_interpolated_plan()
                    retract_js = world.joint_state_from_position(retract_plan.position[-1:])

                grasp_value = best_particle[grasp]
                if grasp_value.shape == (4, 4):
                    obj_from_grasp = grasp_value.clone()
                elif config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(grasp_value.clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(grasp_value.clone())

                target_ee = object_poses[obj] @ obj_from_grasp

                approach_result = world.plan_pose(
                    retract_js,
                    target_ee @ approach_offset,
                )

                if not bool(torch.as_tensor(approach_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan pick approach for {ground_op.name}. "
                        f"Status: {approach_result.status}"
                    )

                approach_plan = approach_result.get_interpolated_plan()
                approach_js = world.joint_state_from_position(approach_plan.position[-1:])

                grasp_result = world.plan_cspace(
                    approach_js,
                    world.joint_state_from_position(best_particle[q][None].clone()),
                    allow_detached_retry=True,
                    obstacle_name=obj,
                )

                if not bool(torch.as_tensor(grasp_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan pick grasp for {ground_op.name}. "
                        f"Status: {grasp_result.status}"
                    )

            for result in (retract_result, approach_result, grasp_result):
                if result is None:
                    continue

                plan = result.get_interpolated_plan()

                if config.time_dilation_factor is not None and config.time_dilation_factor != 1.0:
                    if config.time_dilation_factor <= 0.0:
                        raise ValueError(
                            f"time_dilation_factor must be positive, not {config.time_dilation_factor}"
                        )
                    plan = plan.clone()
                    plan.dt = plan.dt / config.time_dilation_factor
                    plan.velocity = plan.velocity * config.time_dilation_factor
                    plan.acceleration = plan.acceleration * config.time_dilation_factor**2
                    plan.jerk = plan.jerk * config.time_dilation_factor**3

                dt = float(torch.as_tensor(plan.dt).reshape(-1)[0].item())
                accum_plans.append(
                    {
                        "type": "trajectory",
                        "plan": plan,
                        "dt": dt,
                        "label": ground_op.name,
                    }
                )

                last_js = world.joint_state_from_position(plan.position[-1:])
                time_s = visualizer.log_joint_trajectory(
                    plan.position,
                    timeline=timeline,
                    start_time=time_s,
                    dt=dt,
                )

            world.attach_scene_object(last_js, obj)

            if "ur5" in world.robot_name or "robotiq" in world.robot_name:
                gripper_q = torch.linspace(0.0, 0.4, 20)[:, None]
            else:
                gripper_q = torch.linspace(0.04, 0.02, 20)[:, None].repeat(1, 2)

            accum_plans.append({"type": "gripper", "action": "close", "label": ground_op.name})
            robot_q = last_js.position.expand(gripper_q.shape[0], -1).cpu()
            full_q = torch.cat([robot_q, gripper_q], dim=1)
            time_s = visualizer.log_joint_trajectory(
                full_q,
                timeline=timeline,
                start_time=time_s,
                dt=0.02,
            )
            continue

        if op_name == Place.name:
            obj, grasp, placement, _, q = ground_op.values

            with timer.time(f"{timeline}_planning"):
                start_ee = world.compute_ee_matrix(last_js.position)

                retract_result = world.plan_pose(
                    last_js,
                    start_ee @ approach_offset,
                    linear_axis="z",
                    allow_detached_retry=True,
                    obstacle_name=obj,
                )

                if not bool(torch.as_tensor(retract_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan place retract for {ground_op.name}. "
                        f"Status: {retract_result.status}"
                    )

                retract_plan = retract_result.get_interpolated_plan()
                retract_js = world.joint_state_from_position(retract_plan.position[-1:])

                placement_value = best_particle[placement]
                if placement_value.shape == (4, 4):
                    world_from_obj_goal = placement_value.clone()
                elif config.place_dof == 4:
                    world_from_obj_goal = action_4dof_to_mat4x4(placement_value.clone())
                else:
                    world_from_obj_goal = action_6dof_to_mat4x4(placement_value.clone())

                grasp_value = best_particle[grasp]
                if grasp_value.shape == (4, 4):
                    obj_from_grasp = grasp_value.clone()
                elif config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(grasp_value.clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(grasp_value.clone())

                target_ee = world_from_obj_goal @ obj_from_grasp

                approach_result = None
                for offset in place_approach_offsets:
                    candidate = world.plan_pose(
                        retract_js,
                        target_ee @ offset,
                        allow_detached_retry=True,
                        obstacle_name=obj,
                    )

                    if bool(torch.as_tensor(candidate.success).any().item()):
                        approach_result = candidate
                        break

                if approach_result is None:
                    raise RuntimeError(f"Failed to plan place approach for {ground_op.name}")

                approach_plan = approach_result.get_interpolated_plan()
                approach_js = world.joint_state_from_position(approach_plan.position[-1:])

                place_result = world.plan_cspace(
                    approach_js,
                    world.joint_state_from_position(best_particle[q][None].clone()),
                    allow_detached_retry=True,
                    obstacle_name=obj,
                )

                if not bool(torch.as_tensor(place_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan place final for {ground_op.name}. "
                        f"Status: {place_result.status}"
                    )

            obj_from_ee = torch.inverse(object_poses[obj]) @ start_ee
            ee_from_obj = torch.inverse(obj_from_ee)

            for result in (retract_result, approach_result, place_result):
                plan = result.get_interpolated_plan()

                if config.time_dilation_factor is not None and config.time_dilation_factor != 1.0:
                    if config.time_dilation_factor <= 0.0:
                        raise ValueError(
                            f"time_dilation_factor must be positive, not {config.time_dilation_factor}"
                        )
                    plan = plan.clone()
                    plan.dt = plan.dt / config.time_dilation_factor
                    plan.velocity = plan.velocity * config.time_dilation_factor
                    plan.acceleration = plan.acceleration * config.time_dilation_factor**2
                    plan.jerk = plan.jerk * config.time_dilation_factor**3

                dt = float(torch.as_tensor(plan.dt).reshape(-1)[0].item())
                accum_plans.append(
                    {
                        "type": "trajectory",
                        "plan": plan,
                        "dt": dt,
                        "label": ground_op.name,
                    }
                )

                world_from_ee = world.compute_ee_matrix(plan.position)
                world_from_obj = world_from_ee @ ee_from_obj
                last_js = world.joint_state_from_position(plan.position[-1:])
                time_s = visualizer.log_joint_trajectory_with_mat4x4(
                    traj=plan.position,
                    mat4x4_key=f"world/{obj}",
                    mat4x4=world_from_obj,
                    timeline=timeline,
                    start_time=time_s,
                    dt=dt,
                )
                object_poses[obj] = world_from_obj[-1]

            with timer.time(f"{timeline}_planning"):
                world.update_object_pose(obj, object_poses[obj])

            world.detach_attached_object()

            if "ur5" in world.robot_name or "robotiq" in world.robot_name:
                gripper_q = torch.linspace(0.4, 0.0, 20)[:, None]
            else:
                gripper_q = torch.linspace(0.02, 0.04, 20)[:, None].repeat(1, 2)

            accum_plans.append({"type": "gripper", "action": "open", "label": ground_op.name})
            robot_q = last_js.position.expand(gripper_q.shape[0], -1).cpu()
            full_q = torch.cat([robot_q, gripper_q], dim=1)
            time_s = visualizer.log_joint_trajectory(
                full_q,
                timeline=timeline,
                start_time=time_s,
                dt=0.02,
            )
            continue

        if op_name == Push.name:
            button, pose, q = ground_op.values

            if "ur5" in world.robot_name or "robotiq" in world.robot_name:
                gripper_q = torch.linspace(0.0, 0.4, 20)[:, None]
            else:
                gripper_q = torch.linspace(0.04, 0.02, 20)[:, None].repeat(1, 2)

            accum_plans.append({"type": "gripper", "action": "close", "label": ground_op.name})
            robot_q = last_js.position.expand(gripper_q.shape[0], -1).cpu()
            full_q = torch.cat([robot_q, gripper_q], dim=1)
            time_s = visualizer.log_joint_trajectory(
                full_q,
                timeline=timeline,
                start_time=time_s,
                dt=0.02,
            )

            with timer.time(f"{timeline}_planning"):
                retract_result = None
                retract_js = last_js

                if last_q_name != "q0":
                    start_ee = world.compute_ee_matrix(last_js.position)
                    retract_result = world.plan_pose(
                        last_js,
                        start_ee @ approach_offset,
                        linear_axis="z",
                    )

                    if not bool(torch.as_tensor(retract_result.success).any().item()):
                        raise RuntimeError(
                            f"Failed to plan push retract for {ground_op.name}. "
                            f"Status: {retract_result.status}"
                        )

                    retract_plan = retract_result.get_interpolated_plan()
                    retract_js = world.joint_state_from_position(retract_plan.position[-1:])

                push_value = best_particle[pose]
                if push_value.shape == (4, 4):
                    target_ee = push_value.clone()
                elif config.push_dof == 4:
                    target_ee = action_4dof_to_mat4x4(push_value.clone())
                else:
                    target_ee = action_6dof_to_mat4x4(push_value.clone())

                approach_result = world.plan_pose(
                    retract_js,
                    target_ee @ approach_offset,
                )

                if not bool(torch.as_tensor(approach_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan push approach for {ground_op.name}. "
                        f"Status: {approach_result.status}"
                    )

                approach_plan = approach_result.get_interpolated_plan()
                approach_js = world.joint_state_from_position(approach_plan.position[-1:])

                push_result = world.plan_cspace(
                    approach_js,
                    world.joint_state_from_position(best_particle[q][None].clone()),
                    allow_detached_retry=True,
                    obstacle_name=button,
                )

                if not bool(torch.as_tensor(push_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan push execute for {ground_op.name}. "
                        f"Status: {push_result.status}"
                    )

            for result in (retract_result, approach_result, push_result):
                if result is None:
                    continue

                plan = result.get_interpolated_plan()

                if config.time_dilation_factor is not None and config.time_dilation_factor != 1.0:
                    if config.time_dilation_factor <= 0.0:
                        raise ValueError(
                            f"time_dilation_factor must be positive, not {config.time_dilation_factor}"
                        )
                    plan = plan.clone()
                    plan.dt = plan.dt / config.time_dilation_factor
                    plan.velocity = plan.velocity * config.time_dilation_factor
                    plan.acceleration = plan.acceleration * config.time_dilation_factor**2
                    plan.jerk = plan.jerk * config.time_dilation_factor**3

                dt = float(torch.as_tensor(plan.dt).reshape(-1)[0].item())
                accum_plans.append(
                    {
                        "type": "trajectory",
                        "plan": plan,
                        "dt": dt,
                        "label": ground_op.name,
                    }
                )

                last_js = world.joint_state_from_position(plan.position[-1:])
                time_s = visualizer.log_joint_trajectory(
                    plan.position,
                    timeline=timeline,
                    start_time=time_s,
                    dt=dt,
                )

            continue

        raise NotImplementedError(f"Unsupported operator {op_name}")

    with timer.time(f"{timeline}_planning"):
        start_ee = world.compute_ee_matrix(last_js.position)
        retract_result = world.plan_pose(
            last_js,
            start_ee @ approach_offset,
            linear_axis="z",
            allow_detached_retry=True,
        )

        if not bool(torch.as_tensor(retract_result.success).any().item()):
            raise RuntimeError(f"Failed to plan return retract. Status: {retract_result.status}")

    plan = retract_result.get_interpolated_plan()

    if config.time_dilation_factor is not None and config.time_dilation_factor != 1.0:
        if config.time_dilation_factor <= 0.0:
            raise ValueError(f"time_dilation_factor must be positive, not {config.time_dilation_factor}")
        plan = plan.clone()
        plan.dt = plan.dt / config.time_dilation_factor
        plan.velocity = plan.velocity * config.time_dilation_factor
        plan.acceleration = plan.acceleration * config.time_dilation_factor**2
        plan.jerk = plan.jerk * config.time_dilation_factor**3

    dt = float(torch.as_tensor(plan.dt).reshape(-1)[0].item())
    accum_plans.append(
        {
            "type": "trajectory",
            "plan": plan,
            "dt": dt,
            "label": "GoToInitial(q0)",
        }
    )
    last_js = world.joint_state_from_position(plan.position[-1:])
    time_s = visualizer.log_joint_trajectory(
        plan.position,
        timeline=timeline,
        start_time=time_s,
        dt=dt,
    )

    with timer.time(f"{timeline}_planning"):
        home_result = world.plan_cspace(
            last_js,
            world.joint_state_from_position(best_particle["q0"][None].clone()),
        )

        if not bool(torch.as_tensor(home_result.success).any().item()):
            raise RuntimeError(f"Failed to plan going home. Status: {home_result.status}")

    plan = home_result.get_interpolated_plan()

    if config.time_dilation_factor is not None and config.time_dilation_factor != 1.0:
        if config.time_dilation_factor <= 0.0:
            raise ValueError(f"time_dilation_factor must be positive, not {config.time_dilation_factor}")
        plan = plan.clone()
        plan.dt = plan.dt / config.time_dilation_factor
        plan.velocity = plan.velocity * config.time_dilation_factor
        plan.acceleration = plan.acceleration * config.time_dilation_factor**2
        plan.jerk = plan.jerk * config.time_dilation_factor**3

    dt = float(torch.as_tensor(plan.dt).reshape(-1)[0].item())
    accum_plans.append(
        {
            "type": "trajectory",
            "plan": plan,
            "dt": dt,
            "label": "GoToInitial(q0)",
        }
    )
    visualizer.log_joint_trajectory(
        plan.position,
        timeline=timeline,
        start_time=time_s,
        dt=dt,
    )

    _log.info("Motion planning metrics: %s", timer.get_summary(f"{timeline}_planning"))
    return accum_plans
