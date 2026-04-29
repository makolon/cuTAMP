# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Solving motions with cuRobo v2."""

from __future__ import annotations

import logging

import torch

from curobo._src.state.state_joint_trajectory_ops import get_joint_state_at_horizon_index
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
):
    """Convert a satisfying TAMP particle into executable robot trajectories."""

    if config.warmup_motion_gen:
        with timer.time(f"{timeline}_motion_gen_warmup", log_callback=_log.info):
            world.warmup_motion_gen()

    time_s = 0.0
    object_poses = {name: pose.clone() for name, pose in obj_to_initial_pose.items()}
    world.reset_scene(object_poses)

    if "ur5" in world.robot_name or "robotiq" in world.robot_name:
        gripper_open = torch.tensor([0.0], device=world.device)
        gripper_close = torch.tensor([0.4], device=world.device)
    else:
        gripper_open = torch.tensor([0.04, 0.04], device=world.device)
        gripper_close = torch.tensor([0.02, 0.02], device=world.device)
    gripper_state = gripper_open.clone()
    planning_dof = best_particle["q0"].shape[-1]
    visual_dof = planning_dof + gripper_state.numel()

    def add_visual_gripper(robot_q: torch.Tensor, gripper_q: torch.Tensor) -> torch.Tensor:
        """Append gripper columns only for arm-only trajectories."""
        robot_q = robot_q.detach().cpu()
        gripper_q = gripper_q.detach().cpu()
        squeeze = False
        if robot_q.ndim == 1:
            robot_q = robot_q[None]
            gripper_q = gripper_q[None] if gripper_q.ndim == 1 else gripper_q
            squeeze = True
        if gripper_q.ndim == 1:
            gripper_q = gripper_q[None].expand(robot_q.shape[0], -1)
        elif gripper_q.shape[0] == 1 and robot_q.shape[0] != 1:
            gripper_q = gripper_q.expand(robot_q.shape[0], -1)

        if robot_q.shape[-1] == visual_dof:
            full_q = robot_q
        elif robot_q.shape[-1] == planning_dof and robot_q.shape[-1] + gripper_q.shape[-1] == visual_dof:
            full_q = torch.cat([robot_q, gripper_q], dim=-1)
        else:
            raise ValueError(
                f"Cannot build visual joint trajectory: robot_q={tuple(robot_q.shape)}, "
                f"gripper_q={tuple(gripper_q.shape)}, planning_dof={planning_dof}, visual_dof={visual_dof}"
            )
        return full_q[0] if squeeze else full_q

    visualizer.set_time_seconds(timeline, time_s)
    visualizer.set_joint_positions(add_visual_gripper(best_particle["q0"], gripper_state))
    for obj_name, obj_pose in object_poses.items():
        visualizer.log_mat4x4(f"world/{obj_name}", obj_pose)

    last_js = world.joint_state_from_position(best_particle["q0"][None].clone())
    last_q_name = "q0"
    accum_plans: list[dict] = []

    approach_offset = torch.eye(4, device=world.device)
    approach_offset[2, 3] = -0.05

    approach_offsets = torch.eye(4, device=world.device).repeat(4, 1, 1)
    approach_offsets[:, 2, 3] = torch.tensor([-0.05, -0.1, -0.15, -0.2], device=world.device)

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
            obj, grasp, _ = ground_op.values

            with timer.time(f"{timeline}_planning"):
                start_js = last_js

                if last_q_name != "q0":
                    world_from_ee = world.compute_ee_matrix(last_js.position)
                    world_from_retract = world_from_ee @ approach_offset
                    retract_result = world.plan_pose(last_js, world_from_retract, linear_axis="z")
                    if retract_result is None or not bool(torch.as_tensor(retract_result.success).any().item()):
                        raise RuntimeError(
                            f"Failed to plan pick retract for {ground_op.name}. "
                            f"Status: {retract_result.status if retract_result is not None else 'None'}"
                        )
                    retract_js = get_joint_state_at_horizon_index(retract_result.js_solution, -1).squeeze(0)
                    retract_js = world.ensure_joint_state(retract_js)
                else:
                    retract_result = None
                    retract_js = start_js

                world_from_obj = object_poses[obj]
                if best_particle[grasp].shape == (4, 4):
                    obj_from_grasp = best_particle[grasp].clone()
                elif config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp].clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(best_particle[grasp].clone())

                world_from_grasp = world_from_obj @ obj_from_grasp
                world_from_ee = world_from_grasp @ world.tool_from_ee
                world_from_approach = world_from_ee @ approach_offset

                approach_result = world.plan_pose(retract_js, world_from_approach)
                if approach_result is None or not bool(torch.as_tensor(approach_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan pick approach for {ground_op.name}. "
                        f"Status: {approach_result.status if approach_result is not None else 'None'}"
                    )

                approach_js = get_joint_state_at_horizon_index(approach_result.js_solution, -1).squeeze(0)
                approach_js = world.ensure_joint_state(approach_js)

                end_result = world.plan_pose(approach_js, world_from_ee, linear_axis="z")
                if end_result is None or not bool(torch.as_tensor(end_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan pick grasp for {ground_op.name}. "
                        f"Status: {end_result.status if end_result is not None else 'None'}"
                    )

            for result in (retract_result, approach_result, end_result):
                if result is None:
                    continue

                plan = result.get_interpolated_plan()
                if config.time_dilation_factor is not None and config.time_dilation_factor != 1.0:
                    if config.time_dilation_factor <= 0.0:
                        raise ValueError(f"time_dilation_factor must be positive, not {config.time_dilation_factor}")
                    plan = plan.clone()
                    plan.dt = plan.dt / config.time_dilation_factor
                    plan.velocity = plan.velocity * config.time_dilation_factor
                    plan.acceleration = plan.acceleration * config.time_dilation_factor**2
                    plan.jerk = plan.jerk * config.time_dilation_factor**3

                dt = float(torch.as_tensor(plan.dt).reshape(-1)[0].item())
                accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt, "label": ground_op.name})

                last_js = get_joint_state_at_horizon_index(result.js_solution, -1).squeeze(0)
                last_js = world.ensure_joint_state(last_js)

                vis_position = add_visual_gripper(plan.position, gripper_state)
                time_s = visualizer.log_joint_trajectory(vis_position, timeline=timeline, start_time=time_s, dt=dt)

            gripper_q = torch.linspace(0.0, 1.0, 20, device=world.device)[:, None] * (gripper_close - gripper_state)[None] + gripper_state[None]
            accum_plans.append({"type": "gripper", "action": "close", "label": ground_op.name})
            robot_q = last_js.position.reshape(1, -1).expand(gripper_q.shape[0], -1)
            full_q = add_visual_gripper(robot_q, gripper_q)
            time_s = visualizer.log_joint_trajectory(full_q, timeline=timeline, start_time=time_s, dt=0.02)
            gripper_state = gripper_q[-1]
            continue

        if op_name == Place.name:
            obj, grasp, placement, _, _ = ground_op.values

            with timer.time(f"{timeline}_planning"):
                world_from_ee_start = world.compute_ee_matrix(last_js.position)
                world_from_retract = world_from_ee_start @ approach_offset
                retract_result = world.plan_pose(last_js, world_from_retract, linear_axis="z")
                if retract_result is None or not bool(torch.as_tensor(retract_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan place retract for {ground_op.name}. "
                        f"Status: {retract_result.status if retract_result is not None else 'None'}"
                    )

                retract_js = get_joint_state_at_horizon_index(retract_result.js_solution, -1).squeeze(0)
                retract_js = world.ensure_joint_state(retract_js)

                if best_particle[placement].shape == (4, 4):
                    world_from_obj_goal = best_particle[placement].clone()
                elif config.place_dof == 4:
                    world_from_obj_goal = action_4dof_to_mat4x4(best_particle[placement].clone())
                else:
                    world_from_obj_goal = action_6dof_to_mat4x4(best_particle[placement].clone())

                if best_particle[grasp].shape == (4, 4):
                    obj_from_grasp = best_particle[grasp].clone()
                elif config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp].clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(best_particle[grasp].clone())

                world_from_grasp = world_from_obj_goal @ obj_from_grasp
                world_from_ee = world_from_grasp @ world.tool_from_ee
                world_from_approaches = world_from_ee @ approach_offsets

                approach_result = None
                for world_from_approach in world_from_approaches:
                    candidate = world.plan_pose(retract_js, world_from_approach)
                    if candidate is not None and bool(torch.as_tensor(candidate.success).any().item()):
                        approach_result = candidate
                        break

                if approach_result is None or not bool(torch.as_tensor(approach_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan place approach for {ground_op.name}. "
                        f"Status: {approach_result.status if approach_result is not None else 'None'}"
                    )

                approach_js = get_joint_state_at_horizon_index(approach_result.js_solution, -1).squeeze(0)
                approach_js = world.ensure_joint_state(approach_js)

                end_result = world.plan_pose(approach_js, world_from_ee, linear_axis="z")
                if end_result is None or not bool(torch.as_tensor(end_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan place final for {ground_op.name}. "
                        f"Status: {end_result.status if end_result is not None else 'None'}"
                    )

            obj_from_ee = torch.inverse(object_poses[obj]) @ world_from_ee_start
            ee_from_obj = torch.inverse(obj_from_ee)

            for result in (retract_result, approach_result, end_result):
                if result is None:
                    continue

                plan = result.get_interpolated_plan()
                if config.time_dilation_factor is not None and config.time_dilation_factor != 1.0:
                    if config.time_dilation_factor <= 0.0:
                        raise ValueError(f"time_dilation_factor must be positive, not {config.time_dilation_factor}")
                    plan = plan.clone()
                    plan.dt = plan.dt / config.time_dilation_factor
                    plan.velocity = plan.velocity * config.time_dilation_factor
                    plan.acceleration = plan.acceleration * config.time_dilation_factor**2
                    plan.jerk = plan.jerk * config.time_dilation_factor**3

                dt = float(torch.as_tensor(plan.dt).reshape(-1)[0].item())
                accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt, "label": ground_op.name})

                world_from_ee_traj = world.compute_ee_matrix(plan.position)
                world_from_obj_traj = world_from_ee_traj @ ee_from_obj

                last_js = get_joint_state_at_horizon_index(result.js_solution, -1).squeeze(0)
                last_js = world.ensure_joint_state(last_js)

                vis_position = add_visual_gripper(plan.position, gripper_state)
                time_s = visualizer.log_joint_trajectory_with_mat4x4(
                    traj=vis_position,
                    mat4x4_key=f"world/{obj}",
                    mat4x4=world_from_obj_traj,
                    timeline=timeline,
                    start_time=time_s,
                    dt=dt,
                )
                object_poses[obj] = world_from_obj_traj[-1]

            with timer.time(f"{timeline}_planning"):
                world.update_object_pose(obj, object_poses[obj])

            world.detach_attached_object()

            gripper_q = torch.linspace(0.0, 1.0, 20, device=world.device)[:, None] * (gripper_open - gripper_state)[None] + gripper_state[None]
            accum_plans.append({"type": "gripper", "action": "open", "label": ground_op.name})
            robot_q = last_js.position.reshape(1, -1).expand(gripper_q.shape[0], -1)
            full_q = add_visual_gripper(robot_q, gripper_q)
            time_s = visualizer.log_joint_trajectory(full_q, timeline=timeline, start_time=time_s, dt=0.02)
            gripper_state = gripper_q[-1]
            continue

        if op_name == Push.name:
            button, pose, q = ground_op.values

            gripper_q = torch.linspace(0.0, 1.0, 20, device=world.device)[:, None] * (gripper_close - gripper_state)[None] + gripper_state[None]
            accum_plans.append({"type": "gripper", "action": "close", "label": ground_op.name})
            robot_q = last_js.position.reshape(1, -1).expand(gripper_q.shape[0], -1)
            full_q = add_visual_gripper(robot_q, gripper_q)
            time_s = visualizer.log_joint_trajectory(full_q, timeline=timeline, start_time=time_s, dt=0.02)
            gripper_state = gripper_q[-1]

            with timer.time(f"{timeline}_planning"):
                retract_result = None
                retract_js = last_js

                if last_q_name != "q0":
                    world_from_ee = world.compute_ee_matrix(last_js.position)
                    world_from_retract = world_from_ee @ approach_offset
                    retract_result = world.plan_pose(last_js, world_from_retract, linear_axis="z")
                    if retract_result is None or not bool(torch.as_tensor(retract_result.success).any().item()):
                        raise RuntimeError(
                            f"Failed to plan push retract for {ground_op.name}. "
                            f"Status: {retract_result.status if retract_result is not None else 'None'}"
                        )
                    retract_js = get_joint_state_at_horizon_index(retract_result.js_solution, -1).squeeze(0)
                    retract_js = world.ensure_joint_state(retract_js)

                push_value = best_particle[pose]
                if push_value.shape == (4, 4):
                    world_from_push = push_value.clone()
                elif config.push_dof == 4:
                    world_from_push = action_4dof_to_mat4x4(push_value.clone())
                else:
                    world_from_push = action_6dof_to_mat4x4(push_value.clone())

                world_from_ee = world_from_push @ world.tool_from_ee
                world_from_approach = world_from_ee @ approach_offset

                approach_result = world.plan_pose(retract_js, world_from_approach)
                if approach_result is None or not bool(torch.as_tensor(approach_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan push approach for {ground_op.name}. "
                        f"Status: {approach_result.status if approach_result is not None else 'None'}"
                    )

                approach_js = get_joint_state_at_horizon_index(approach_result.js_solution, -1).squeeze(0)
                approach_js = world.ensure_joint_state(approach_js)

                end_result = world.plan_cspace(
                    approach_js,
                    world.joint_state_from_position(best_particle[q][None].clone()),
                )
                if end_result is None or not bool(torch.as_tensor(end_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan push execute for {ground_op.name}. "
                        f"Status: {end_result.status if end_result is not None else 'None'}"
                    )

            for result in (retract_result, approach_result, end_result):
                if result is None:
                    continue

                plan = result.get_interpolated_plan()
                if config.time_dilation_factor is not None and config.time_dilation_factor != 1.0:
                    if config.time_dilation_factor <= 0.0:
                        raise ValueError(f"time_dilation_factor must be positive, not {config.time_dilation_factor}")
                    plan = plan.clone()
                    plan.dt = plan.dt / config.time_dilation_factor
                    plan.velocity = plan.velocity * config.time_dilation_factor
                    plan.acceleration = plan.acceleration * config.time_dilation_factor**2
                    plan.jerk = plan.jerk * config.time_dilation_factor**3

                dt = float(torch.as_tensor(plan.dt).reshape(-1)[0].item())
                accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt, "label": ground_op.name})

                last_js = get_joint_state_at_horizon_index(result.js_solution, -1).squeeze(0)
                last_js = world.ensure_joint_state(last_js)

                vis_position = add_visual_gripper(plan.position, gripper_state)
                time_s = visualizer.log_joint_trajectory(vis_position, timeline=timeline, start_time=time_s, dt=dt)

            continue

        raise NotImplementedError(f"Unsupported operator {op_name}")

    with timer.time(f"{timeline}_planning"):
        world_from_ee = world.compute_ee_matrix(last_js.position)
        world_from_retract = world_from_ee @ approach_offset
        retract_result = world.plan_pose(last_js, world_from_retract, linear_axis="z")
        if retract_result is None or not bool(torch.as_tensor(retract_result.success).any().item()):
            raise RuntimeError(
                f"Failed to plan return retract. "
                f"Status: {retract_result.status if retract_result is not None else 'None'}"
            )

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
    accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt, "label": "GoToInitial(q0)"})
    last_js = get_joint_state_at_horizon_index(retract_result.js_solution, -1).squeeze(0)
    last_js = world.ensure_joint_state(last_js)

    vis_position = add_visual_gripper(plan.position, gripper_state)
    time_s = visualizer.log_joint_trajectory(vis_position, timeline=timeline, start_time=time_s, dt=dt)

    with timer.time(f"{timeline}_planning"):
        home_result = world.plan_cspace(last_js, world.joint_state_from_position(best_particle["q0"][None].clone()))
        if home_result is None or not bool(torch.as_tensor(home_result.success).any().item()):
            raise RuntimeError(
                f"Failed to plan going home. "
                f"Status: {home_result.status if home_result is not None else 'None'}"
            )

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
    accum_plans.append({"type": "trajectory", "plan": plan, "dt": dt, "label": "GoToInitial(q0)"})

    vis_position = add_visual_gripper(plan.position, gripper_state)
    visualizer.log_joint_trajectory(vis_position, timeline=timeline, start_time=time_s, dt=dt)

    _log.info("Motion planning metrics: %s", timer.get_summary(f"{timeline}_planning"))
    return accum_plans
