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
from curobo.types import GoalToolPose, JointState, Pose
from cutamp.config import TAMPConfiguration
from cutamp.optimize_plan import PlanContainer
from cutamp.tamp_domain import MoveFree, MoveHolding, Pick, Place, Push
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.common import Particles, action_4dof_to_mat4x4, action_6dof_to_mat4x4
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import Visualizer

_log = logging.getLogger(__name__)


def convert_to_goal_tool_pose(
    world_from_ee: torch.Tensor,
    tool_frames: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> GoalToolPose:
    world_from_ee = world_from_ee.to(device=device, dtype=dtype).contiguous()
    if world_from_ee.ndim == 2:
        world_from_ee = world_from_ee.unsqueeze(0)

    ee_pose = Pose.from_matrix(world_from_ee)
    pose_dict = {tool_frame: ee_pose for tool_frame in tool_frames}
    return GoalToolPose.from_poses(
        pose_dict,
        ordered_tool_frames=tool_frames,
        num_goalset=1,
    )


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

    device = world.device_cfg.device
    dtype = world.device_cfg.dtype
    planner_joint_names = list(world.motion_planner.kinematics.joint_names)
    planner_tool_frames = list(world.motion_planner.tool_frames)
    tool_frame = planner_tool_frames[0]

    if "ur5" in world.robot_name or "robotiq" in world.robot_name:
        gripper_open = torch.tensor([0.0], device=world.device, dtype=dtype)
        gripper_close = torch.tensor([0.4], device=world.device, dtype=dtype)
    else:
        gripper_open = torch.tensor([0.04, 0.04], device=world.device, dtype=dtype)
        gripper_close = torch.tensor([0.02, 0.02], device=world.device, dtype=dtype)
    gripper_state = gripper_open.clone()

    q0 = best_particle["q0"][None].clone().to(device=device, dtype=dtype)
    last_js = JointState.from_position(q0, joint_names=planner_joint_names)
    last_q_name = "q0"

    approach_offset = torch.eye(4, device=world.device, dtype=dtype)
    approach_offset[2, 3] = -0.05

    approach_offsets = torch.eye(4, device=world.device, dtype=dtype).repeat(4, 1, 1)
    approach_offsets[:, 2, 3] = torch.tensor(
        [-0.05, -0.1, -0.15, -0.2],
        device=world.device,
        dtype=dtype,
    )

    # Accumulated plans we return that the real robot can actually execute.
    accum_plans: list[dict] = []

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
                    last_js = world.motion_planner.kinematics.get_active_js(last_js)
                    kin_state = world.motion_planner.compute_kinematics(last_js)
                    world_from_ee_start = (
                        kin_state.tool_poses
                        .get_link_pose(tool_frame, make_contiguous=True)
                        .get_matrix()[0]
                    )
                    world_from_retract = world_from_ee_start @ approach_offset
                    world_from_retract = convert_to_goal_tool_pose(
                        world_from_retract,
                        planner_tool_frames,
                        device=device,
                        dtype=dtype,
                    )
                    retract_result = world.motion_planner.plan_pose(world_from_retract, last_js)
                    if retract_result is None or not bool(torch.as_tensor(retract_result.success).any().item()):
                        raise RuntimeError(
                            f"Failed to plan pick retract for {ground_op.name}. "
                            f"Status: {retract_result.status if retract_result is not None else 'None'}"
                        )
                    retract_js = get_joint_state_at_horizon_index(retract_result.js_solution, -1).squeeze(0)
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

                world_from_approach = convert_to_goal_tool_pose(
                    world_from_approach,
                    planner_tool_frames,
                    device=device,
                    dtype=dtype,
                )
                retract_js = world.motion_planner.kinematics.get_active_js(retract_js)
                approach_result = world.motion_planner.plan_pose(world_from_approach, retract_js)
                if approach_result is None or not bool(torch.as_tensor(approach_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan pick approach for {ground_op.name}. "
                        f"Status: {approach_result.status if approach_result is not None else 'None'}"
                    )

                # Plan from approach to target EE pose for grasp.
                approach_js = get_joint_state_at_horizon_index(approach_result.js_solution, -1).squeeze(0)
                world_from_ee_goal = convert_to_goal_tool_pose(
                    world_from_ee,
                    planner_tool_frames,
                    device=device,
                    dtype=dtype,
                )
                approach_js = world.motion_planner.kinematics.get_active_js(approach_js)
                end_result = world.motion_planner.plan_pose(world_from_ee_goal, approach_js)
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

            gripper_q = (
                torch.linspace(0.0, 1.0, 20, device=world.device, dtype=dtype)[:, None]
                * (gripper_close - gripper_state)[None]
                + gripper_state[None]
            )
            accum_plans.append({"type": "gripper", "action": "close", "label": ground_op.name})
            robot_q = last_js.position.reshape(1, -1).expand(gripper_q.shape[0], -1)
            gripper_state = gripper_q[-1]
            continue

        if op_name == Place.name:
            obj, grasp, placement, _, _ = ground_op.values

            with timer.time(f"{timeline}_planning"):
                last_js = world.motion_planner.kinematics.get_active_js(last_js)
                kin_state = world.motion_planner.compute_kinematics(last_js)
                world_from_ee_start = (
                    kin_state.tool_poses
                    .get_link_pose(tool_frame, make_contiguous=True)
                    .get_matrix()[0]
                )
                world_from_retract = world_from_ee_start @ approach_offset
                world_from_retract = convert_to_goal_tool_pose(
                    world_from_retract,
                    planner_tool_frames,
                    device=device,
                    dtype=dtype,
                )
                retract_result = world.motion_planner.plan_pose(world_from_retract, last_js)
                if retract_result is None or not bool(torch.as_tensor(retract_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan place retract for {ground_op.name}. "
                        f"Status: {retract_result.status if retract_result is not None else 'None'}"
                    )

                retract_js = get_joint_state_at_horizon_index(retract_result.js_solution, -1).squeeze(0)

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
                    world_from_approach = convert_to_goal_tool_pose(
                        world_from_approach,
                        planner_tool_frames,
                        device=device,
                        dtype=dtype,
                    )
                    retract_js = world.motion_planner.kinematics.get_active_js(retract_js)
                    approach_result = world.motion_planner.plan_pose(world_from_approach, retract_js)
                    if approach_result is not None and bool(torch.as_tensor(approach_result.success).any().item()):
                        break

                if approach_result is None or not bool(torch.as_tensor(approach_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan place approach for {ground_op.name}. "
                        f"Status: {approach_result.status if approach_result is not None else 'None'}"
                    )

                approach_js = get_joint_state_at_horizon_index(approach_result.js_solution, -1).squeeze(0)
                world_from_ee_goal = convert_to_goal_tool_pose(
                    world_from_ee,
                    planner_tool_frames,
                    device=device,
                    dtype=dtype,
                )
                approach_js = world.motion_planner.kinematics.get_active_js(approach_js)
                end_result = world.motion_planner.plan_pose(world_from_ee_goal, approach_js)
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

                # cuRobo v2 FK requires [batch, horizon, dof].
                # get_interpolated_plan() can return [1, 1, T, dof], so flatten all
                # trajectory dimensions into the horizon dimension before FK.
                plan_for_fk = world.motion_planner.kinematics.get_active_js(plan)
                q_fk = plan_for_fk.position.to(device=device, dtype=dtype)
                q_fk = q_fk.reshape(1, -1, q_fk.shape[-1])
                plan_for_fk = JointState.from_position(q_fk, joint_names=planner_joint_names)

                traj_kin_state = world.motion_planner.compute_kinematics(plan_for_fk)
                world_from_ee_traj = (
                    traj_kin_state.tool_poses
                    .get_link_pose(tool_frame, make_contiguous=True)
                    .get_matrix()[0]
                )
                world_from_obj_traj = world_from_ee_traj @ ee_from_obj

                last_js = get_joint_state_at_horizon_index(result.js_solution, -1).squeeze(0)
                object_poses[obj] = world_from_obj_traj[-1]

            gripper_q = (
                torch.linspace(0.0, 1.0, 20, device=world.device, dtype=dtype)[:, None]
                * (gripper_open - gripper_state)[None]
                + gripper_state[None]
            )
            accum_plans.append({"type": "gripper", "action": "open", "label": ground_op.name})
            robot_q = last_js.position.reshape(1, -1).expand(gripper_q.shape[0], -1)
            gripper_state = gripper_q[-1]
            continue

        if op_name == Push.name:
            button, pose, q = ground_op.values

            gripper_q = (
                torch.linspace(0.0, 1.0, 20, device=world.device, dtype=dtype)[:, None]
                * (gripper_close - gripper_state)[None]
                + gripper_state[None]
            )
            accum_plans.append({"type": "gripper", "action": "close", "label": ground_op.name})
            robot_q = last_js.position.reshape(1, -1).expand(gripper_q.shape[0], -1)
            gripper_state = gripper_q[-1]

            with timer.time(f"{timeline}_planning"):
                retract_result = None
                retract_js = last_js

                if last_q_name != "q0":
                    last_js = world.motion_planner.kinematics.get_active_js(last_js)
                    kin_state = world.motion_planner.compute_kinematics(last_js)
                    world_from_ee_start = (
                        kin_state.tool_poses
                        .get_link_pose(tool_frame, make_contiguous=True)
                        .get_matrix()[0]
                    )
                    world_from_retract = world_from_ee_start @ approach_offset
                    world_from_retract = convert_to_goal_tool_pose(
                        world_from_retract,
                        planner_tool_frames,
                        device=device,
                        dtype=dtype,
                    )
                    retract_result = world.motion_planner.plan_pose(world_from_retract, last_js)
                    if retract_result is None or not bool(torch.as_tensor(retract_result.success).any().item()):
                        raise RuntimeError(
                            f"Failed to plan push retract for {ground_op.name}. "
                            f"Status: {retract_result.status if retract_result is not None else 'None'}"
                        )
                    retract_js = get_joint_state_at_horizon_index(retract_result.js_solution, -1).squeeze(0)

                push_value = best_particle[pose]
                if push_value.shape == (4, 4):
                    world_from_push = push_value.clone()
                elif config.push_dof == 4:
                    world_from_push = action_4dof_to_mat4x4(push_value.clone())
                else:
                    world_from_push = action_6dof_to_mat4x4(push_value.clone())

                world_from_ee = world_from_push @ world.tool_from_ee
                world_from_approach = world_from_ee @ approach_offset

                world_from_approach = convert_to_goal_tool_pose(
                    world_from_approach,
                    planner_tool_frames,
                    device=device,
                    dtype=dtype,
                )
                retract_js = world.motion_planner.kinematics.get_active_js(retract_js)
                approach_result = world.motion_planner.plan_pose(world_from_approach, retract_js)
                if approach_result is None or not bool(torch.as_tensor(approach_result.success).any().item()):
                    raise RuntimeError(
                        f"Failed to plan push approach for {ground_op.name}. "
                        f"Status: {approach_result.status if approach_result is not None else 'None'}"
                    )

                approach_js = get_joint_state_at_horizon_index(approach_result.js_solution, -1).squeeze(0)
                world_from_ee_goal = convert_to_goal_tool_pose(
                    world_from_ee,
                    planner_tool_frames,
                    device=device,
                    dtype=dtype,
                )
                approach_js = world.motion_planner.kinematics.get_active_js(approach_js)
                end_result = world.motion_planner.plan_pose(world_from_ee_goal, approach_js)
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

            continue

        raise NotImplementedError(f"Unsupported operator {op_name}")

    with timer.time(f"{timeline}_planning"):
        last_js = world.motion_planner.kinematics.get_active_js(last_js)
        kin_state = world.motion_planner.compute_kinematics(last_js)
        world_from_ee_start = (
            kin_state.tool_poses
            .get_link_pose(tool_frame, make_contiguous=True)
            .get_matrix()[0]
        )
        world_from_retract = world_from_ee_start @ approach_offset
        world_from_retract = convert_to_goal_tool_pose(
            world_from_retract,
            planner_tool_frames,
            device=device,
            dtype=dtype,
        )
        retract_result = world.motion_planner.plan_pose(world_from_retract, last_js)
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

    with timer.time(f"{timeline}_planning"):
        last_js = world.motion_planner.kinematics.get_active_js(last_js)
        q0 = JointState.from_position(q0, joint_names=planner_joint_names)
        home_goal = world.kinematics.compute_kinematics(q0)
        home_goal = (
            home_goal.tool_poses
            .get_link_pose(tool_frame, make_contiguous=True)
            .get_matrix()[0]
        )
        home_goal = convert_to_goal_tool_pose(
            home_goal,
            planner_tool_frames,
            device=device,
            dtype=dtype,
        )
        home_result = world.motion_planner.plan_pose(home_goal, last_js)
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

    _log.info("Motion planning metrics: %s", timer.get_summary(f"{timeline}_planning"))
    return accum_plans
