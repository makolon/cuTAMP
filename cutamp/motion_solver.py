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
import math
from typing import Callable, List, Literal

import torch

from curobo.types import JointState
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
    motion_refinement_mode: str | None = None,
    segment_target_space: str | None = None,
    approach_candidate_index: int | None = None,
    ik_target_debug: dict[str, object] | None = None,
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
    if motion_refinement_mode:
        payload["motion_refinement_mode"] = str(motion_refinement_mode)
    if segment_target_space:
        payload["segment_target_space"] = str(segment_target_space)
    if approach_candidate_index is not None:
        payload["approach_candidate_index"] = int(approach_candidate_index)
    if ik_target_debug is not None:
        payload.update(ik_target_debug)
    return payload


def _motion_refinement_mode(config: TAMPConfiguration) -> Literal["ee_strict", "joint"]:
    mode = getattr(config, "motion_refinement_mode", "ee_strict")
    if mode not in {"ee_strict", "joint"}:
        raise ValueError(f"Unsupported motion refinement mode: {mode}")
    return mode


def _result_success(result: object | None) -> bool:
    success = getattr(result, "success", None)
    if isinstance(success, bool):
        return success
    return isinstance(success, torch.Tensor) and bool(success.any().item())


def _result_status(result: object | None) -> str:
    if result is None:
        return "result_missing"
    return "success" if _result_success(result) else "planning_failed"


def _first_success_index(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, torch.Tensor):
        return None
    success_indices = torch.nonzero(value.detach(), as_tuple=False)
    if success_indices.numel() == 0:
        return None
    return tuple(int(v) for v in success_indices[0].tolist())


def _scalar_at_index(value: object, index: tuple[int, ...]) -> float:
    if not isinstance(value, torch.Tensor):
        return float("nan")
    selected = value.detach()
    for idx in index:
        selected = selected[idx]
    if selected.numel() != 1:
        return float("nan")
    return float(selected.item())


def _rotation_error_deg(value: object, index: tuple[int, ...]) -> float:
    rotation_error = _scalar_at_index(value, index)
    if math.isnan(rotation_error):
        return float("nan")
    return float(torch.rad2deg(torch.tensor(rotation_error)).item())


def _apply_time_dilation(plan: JointState, time_dilation_factor: float | None) -> JointState:
    if time_dilation_factor is None or time_dilation_factor == 1.0:
        return plan
    if time_dilation_factor <= 0.0:
        raise ValueError(f"time_dilation_factor must be positive, not {time_dilation_factor}")

    scaled = plan.clone()
    inverse_factor = 1.0 / time_dilation_factor
    if scaled.dt is not None:
        scaled.dt = scaled.dt * inverse_factor
    if scaled.velocity is not None:
        scaled.velocity = scaled.velocity * time_dilation_factor
    if scaled.acceleration is not None:
        scaled.acceleration = scaled.acceleration * (time_dilation_factor ** 2)
    if scaled.jerk is not None:
        scaled.jerk = scaled.jerk * (time_dilation_factor ** 3)
    return scaled


def _result_plan(result: object, config: TAMPConfiguration) -> JointState:
    plan = result.get_interpolated_plan()
    return _apply_time_dilation(plan, config.time_dilation_factor)


def _plan_dt(plan: JointState) -> float:
    dt = getattr(plan, "dt", None)
    if dt is None:
        return 0.0
    if isinstance(dt, torch.Tensor):
        dt_flat = dt.reshape(-1)
        return float(dt_flat[0].item())
    return float(dt)


def _plan_pose_target(
    *,
    start_js: JointState,
    desired_world_from_ee: torch.Tensor,
    world: TAMPWorld,
    log_label: str,
    linear_axis: str | None = None,
    allow_detached_retry: bool = False,
    obstacle_name: str | None = None,
):
    result = world.plan_pose(
        start_js,
        desired_world_from_ee,
        linear_axis=linear_axis,
        allow_detached_retry=allow_detached_retry,
        obstacle_name=obstacle_name,
    )
    if not _result_success(result) and allow_detached_retry:
        _log.warning(
            f"{log_label}: pose-goal refinement failed with status {_result_status(result)}; "
            "retrying while temporarily detaching attached-object spheres"
        )
    return result


def _solve_joint_target_for_pose(
    *,
    world: TAMPWorld,
    start_js: JointState,
    desired_world_from_ee: torch.Tensor,
) -> tuple[JointState | None, dict[str, object]]:
    ik_result = world.solve_pose(
        desired_world_from_ee,
        current_state=start_js,
        seed_config=start_js.position,
        return_seeds=1,
    )
    success_index = _first_success_index(getattr(ik_result, "success", None))
    if success_index is None:
        return None, {
            "ik_target_success": False,
            "ik_target_translation_error_m": float("nan"),
            "ik_target_rotation_error_deg": float("nan"),
        }
    solution = getattr(ik_result, "solution", None)
    if not isinstance(solution, torch.Tensor):
        return None, {
            "ik_target_success": False,
            "ik_target_translation_error_m": float("nan"),
            "ik_target_rotation_error_deg": float("nan"),
        }
    target_position = solution[success_index].detach().clone().view(1, -1)
    ik_fk = world.compute_ee_matrix(world.joint_state_from_position(target_position))
    return world.joint_state_from_position(target_position), {
        "ik_target_success": True,
        "ik_target_translation_error_m": _scalar_at_index(getattr(ik_result, "position_error", None), success_index),
        "ik_target_rotation_error_deg": _rotation_error_deg(getattr(ik_result, "rotation_error", None), success_index),
        "ik_target_world_from_fk": _mat4_to_list(ik_fk),
    }


def _plan_joint_target(
    *,
    start_js: JointState,
    goal_js: JointState,
    world: TAMPWorld,
    log_label: str,
    allow_detached_retry: bool = False,
    obstacle_name: str | None = None,
):
    result = world.plan_cspace(
        start_js,
        goal_js,
        allow_detached_retry=allow_detached_retry,
        obstacle_name=obstacle_name,
    )
    if not _result_success(result) and allow_detached_retry:
        _log.warning(
            f"{log_label}: joint-goal refinement failed with status {_result_status(result)}; "
            "retrying while temporarily detaching attached-object spheres"
        )
    return result


def _plan_segment(
    *,
    world: TAMPWorld,
    config: TAMPConfiguration,
    start_js: JointState,
    desired_world_from_ee: torch.Tensor,
    log_label: str,
    desired_goal_js: JointState | None = None,
    linear_axis: str | None = None,
    allow_detached_retry: bool = False,
    obstacle_name: str | None = None,
    approach_candidate_index: int | None = None,
):
    mode = _motion_refinement_mode(config)
    if mode == "ee_strict":
        result = _plan_pose_target(
            start_js=start_js,
            desired_world_from_ee=desired_world_from_ee,
            world=world,
            log_label=log_label,
            linear_axis=linear_axis,
            allow_detached_retry=allow_detached_retry,
            obstacle_name=obstacle_name,
        )
        return result, {
            "goal_mode": "pose_goal",
            "segment_target_space": "pose",
            "approach_candidate_index": approach_candidate_index,
            "ik_target_debug": None,
        }

    joint_goal_js = desired_goal_js
    ik_target_debug = None
    if joint_goal_js is None:
        joint_goal_js, ik_target_debug = _solve_joint_target_for_pose(
            world=world,
            start_js=start_js,
            desired_world_from_ee=desired_world_from_ee,
        )
        if joint_goal_js is None:
            return None, {
                "goal_mode": "joint_goal",
                "segment_target_space": "joint",
                "approach_candidate_index": approach_candidate_index,
                "ik_target_debug": ik_target_debug,
            }

    result = _plan_joint_target(
        start_js=start_js,
        goal_js=joint_goal_js,
        world=world,
        log_label=log_label,
        allow_detached_retry=allow_detached_retry,
        obstacle_name=obstacle_name,
    )
    return result, {
        "goal_mode": "joint_goal",
        "segment_target_space": "joint",
        "approach_candidate_index": approach_candidate_index,
        "ik_target_debug": ik_target_debug,
    }


def _terminal_joint_state(world: TAMPWorld, plan: JointState) -> JointState:
    return world.joint_state_from_position(plan.position[-1:])


def _append_segment_plan(
    *,
    accum_plans: list[dict[str, object]],
    world: TAMPWorld,
    config: TAMPConfiguration,
    label: str,
    segment_type: str,
    result: object,
    desired_world_from_ee: torch.Tensor,
    waypoint_index_offset: int,
    motion_refinement_mode: str,
    segment_meta: dict[str, object] | None = None,
    selected_parameter_name: str | None = None,
    selected_obj_from_grasp: torch.Tensor | None = None,
    object_name: str | None = None,
    extra_debug: dict[str, object] | None = None,
) -> tuple[JointState, JointState, float]:
    plan = _result_plan(result, config)
    dt = _plan_dt(plan)
    terminal_world_from_fk = world.compute_ee_matrix(plan.position[-1:])
    debug_payload = _segment_debug_payload(
        label=label,
        segment_type=segment_type,
        desired_world_from_ee=desired_world_from_ee,
        terminal_world_from_ee=terminal_world_from_fk,
        plan_positions=plan.position,
        waypoint_index_offset=waypoint_index_offset,
        selected_parameter_name=selected_parameter_name,
        selected_obj_from_grasp=selected_obj_from_grasp,
        object_name=object_name,
        motion_refinement_mode=motion_refinement_mode,
        segment_target_space=None if segment_meta is None else segment_meta["segment_target_space"],
        approach_candidate_index=None if segment_meta is None else segment_meta["approach_candidate_index"],
        ik_target_debug=None if segment_meta is None else segment_meta["ik_target_debug"],
    )
    if segment_meta is not None and "goal_mode" in segment_meta:
        debug_payload["goal_mode"] = str(segment_meta["goal_mode"])
    if extra_debug is not None:
        debug_payload.update(extra_debug)
    accum_plans.append(
        {
            "type": "trajectory",
            "plan": plan,
            "dt": dt,
            "label": label,
            "debug": debug_payload,
        }
    )
    return plan, _terminal_joint_state(world, plan), dt


def _append_gripper_step(
    *,
    accum_plans: list[dict[str, object]],
    config: TAMPConfiguration,
    action: Literal["open", "close"],
    label: str,
    last_js: JointState,
    visualizer: Visualizer,
    timeline: str,
    ts: float,
) -> float:
    if "ur5" in config.robot or "robotiq_2f_85" in config.robot:
        start_val, end_val = (0.4, 0.0) if action == "open" else (0.0, 0.4)
        interp = torch.linspace(start_val, end_val, 20)[:, None]
    else:
        start_val, end_val = (0.02, 0.04) if action == "open" else (0.04, 0.02)
        interp = torch.linspace(start_val, end_val, 20)[:, None].repeat(1, 2)
    dt = 0.02
    accum_plans.append({"type": "gripper", "action": action, "label": label})
    all_pos = last_js.position.expand(interp.shape[0], -1).cpu()
    all_pos = torch.cat([all_pos, interp], dim=1)
    return visualizer.log_joint_trajectory(all_pos, timeline=timeline, start_time=ts, dt=dt)


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
    if config.warmup_motion_gen:
        with timer.time(f"{timeline}_motion_gen_warmup", log_callback=_log.debug):
            world.warmup_motion_gen()

    ts = 0.0
    obj_to_current_pose = {name: pose.clone() for name, pose in obj_to_initial_pose.items()}
    world.reset_runtime_scene(obj_to_current_pose)

    visualizer.set_time_seconds(timeline, ts)
    visualizer.set_joint_positions(best_particle["q0"])
    for obj_name, pose in obj_to_current_pose.items():
        visualizer.log_mat4x4(f"world/{obj_name}", pose)

    last_js = world.joint_state_from_position(best_particle["q0"][None].clone())
    last_q_name = "q0"
    motion_refinement_mode = _motion_refinement_mode(config)

    approach_offset = torch.eye(4, device=world.device)
    approach_offset[2, 3] = -0.05

    approach_offsets = torch.eye(4, device=world.device).repeat(4, 1, 1)
    approach_offsets[:, 2, 3] = torch.tensor([-0.05, -0.1, -0.15, -0.2], device=world.device)

    accum_plans: list[dict[str, object]] = []
    waypoint_index_offset = 0

    for idx, ground_op in enumerate(plan_skeleton):
        op_name = ground_op.operator.name
        print(f"{idx + 1}. {ground_op.name}")

        if op_name == MoveFree.name:
            q_start, traj, q_end = ground_op.values
            if traj in best_particle:
                raise NotImplementedError("Trajectories not supported yet")
            last_q_name = q_start
            continue

        if op_name == MoveHolding.name:
            obj, grasp, q_start, traj, q_end = ground_op.values
            if traj in best_particle:
                raise NotImplementedError("Trajectories not supported yet")
            last_q_name = q_start
            continue

        if op_name == Pick.name:
            obj, grasp, q = ground_op.values
            with timer.time(f"{timeline}_planning"):
                start_js = last_js
                world_from_start_ee = world.compute_ee_matrix(start_js.position)
                world_from_retract = world_from_start_ee
                retract_result = None
                retract_meta = None
                retract_js = start_js
                if last_q_name != "q0":
                    world_from_retract = world_from_start_ee @ approach_offset
                    retract_result, retract_meta = _plan_segment(
                        world=world,
                        config=config,
                        start_js=start_js,
                        desired_world_from_ee=world_from_retract,
                        log_label=ground_op.name,
                        linear_axis="z",
                    )
                    if not _result_success(retract_result):
                        raise RuntimeError(
                            f"Failed to plan for retract for {ground_op.name}. "
                            f"Status: {_result_status(retract_result)}"
                        )
                    retract_js = _terminal_joint_state(world, _result_plan(retract_result, config))

                world_from_obj = obj_to_current_pose[obj]
                if best_particle[grasp].shape == (4, 4):
                    obj_from_grasp = best_particle[grasp].clone()
                elif config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp].clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(best_particle[grasp].clone())
                world_from_ee = world_from_obj @ obj_from_grasp
                world_from_approach = world_from_ee @ approach_offset

                approach_result, approach_meta = _plan_segment(
                    world=world,
                    config=config,
                    start_js=retract_js,
                    desired_world_from_ee=world_from_approach,
                    log_label=ground_op.name,
                    approach_candidate_index=0,
                )
                if not _result_success(approach_result):
                    raise RuntimeError(
                        f"Failed to plan for approach for {ground_op.name}. "
                        f"Status: {_result_status(approach_result)}"
                    )
                approach_js = _terminal_joint_state(world, _result_plan(approach_result, config))

                end_result, end_meta = _plan_segment(
                    world=world,
                    config=config,
                    start_js=approach_js,
                    desired_world_from_ee=world_from_ee,
                    desired_goal_js=world.joint_state_from_position(best_particle[q][None].clone()),
                    log_label=ground_op.name,
                    linear_axis="z",
                    allow_detached_retry=True,
                    approach_candidate_index=0,
                )
                if not _result_success(end_result):
                    raise RuntimeError(
                        f"Failed to plan from approach to end for {ground_op.name}. "
                        f"Status: {_result_status(end_result)}"
                    )

            ik_fk = world.compute_ee_matrix(best_particle[q][None].clone())
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

            for segment_type, result, desired_world_from_ee, segment_meta, extra_debug in (
                ("pick_retract", retract_result, world_from_retract, retract_meta, {"ik_debug": ik_debug}),
                ("pick_approach", approach_result, world_from_approach, approach_meta, None),
                ("pick_grasp", end_result, world_from_ee, end_meta, None),
            ):
                if result is None:
                    continue
                plan, last_js, dt = _append_segment_plan(
                    accum_plans=accum_plans,
                    world=world,
                    config=config,
                    label=ground_op.name,
                    segment_type=segment_type,
                    result=result,
                    desired_world_from_ee=desired_world_from_ee,
                    waypoint_index_offset=waypoint_index_offset,
                    motion_refinement_mode=motion_refinement_mode,
                    segment_meta=segment_meta,
                    selected_parameter_name=str(grasp),
                    selected_obj_from_grasp=obj_from_grasp,
                    object_name=str(obj),
                    extra_debug=extra_debug,
                )
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
                waypoint_index_offset += int(plan.position.shape[0])

            with timer.time(f"{timeline}_planning"):
                world.attach_scene_object(last_js, obj)
            ts = _append_gripper_step(
                accum_plans=accum_plans,
                config=config,
                action="close",
                label=ground_op.name,
                last_js=last_js,
                visualizer=visualizer,
                timeline=timeline,
                ts=ts,
            )
            continue

        if op_name == Place.name:
            obj, grasp, placement, surface, q = ground_op.values
            with timer.time(f"{timeline}_planning"):
                start_js = last_js
                place_motion_mode = motion_refinement_mode
                world_from_ee_start = world.compute_ee_matrix(start_js.position)
                world_from_retract = world_from_ee_start @ approach_offset
                world_from_obj = (
                    action_4dof_to_mat4x4(best_particle[placement].clone())
                    if config.place_dof == 4
                    else action_6dof_to_mat4x4(best_particle[placement].clone())
                )
                if best_particle[grasp].shape == (4, 4):
                    obj_from_grasp = best_particle[grasp].clone()
                elif config.grasp_dof == 4:
                    obj_from_grasp = action_4dof_to_mat4x4(best_particle[grasp].clone())
                else:
                    obj_from_grasp = action_6dof_to_mat4x4(best_particle[grasp].clone())
                world_from_ee = world_from_obj @ obj_from_grasp
                world_from_approaches = world_from_ee @ approach_offsets
                world_from_approach = world_from_approaches[0]

                retract_result, retract_meta = _plan_segment(
                    world=world,
                    config=config,
                    start_js=start_js,
                    desired_world_from_ee=world_from_retract,
                    log_label=ground_op.name,
                    linear_axis="z",
                    allow_detached_retry=True,
                )
                if motion_refinement_mode == "ee_strict" and not _result_success(retract_result):
                    _log.warning(
                        "%s: retract pose-goal failed with status %s; retrying without linear-motion criteria",
                        ground_op.name,
                        _result_status(retract_result),
                    )
                    retract_result, retract_meta = _plan_segment(
                        world=world,
                        config=config,
                        start_js=start_js,
                        desired_world_from_ee=world_from_retract,
                        log_label=ground_op.name,
                        allow_detached_retry=True,
                    )
                if not _result_success(retract_result):
                    raise RuntimeError(
                        f"Failed to plan for retract for {ground_op.name}. "
                        f"Status: {_result_status(retract_result)}"
                    )
                retract_js = _terminal_joint_state(world, _result_plan(retract_result, config))

                approach_result = None
                approach_meta = None
                approach_candidate_index = 0
                for app_idx, world_from_approach_candidate in enumerate(world_from_approaches):
                    candidate_result, candidate_meta = _plan_segment(
                        world=world,
                        config=config,
                        start_js=retract_js,
                        desired_world_from_ee=world_from_approach_candidate,
                        log_label=ground_op.name,
                        allow_detached_retry=True,
                        approach_candidate_index=app_idx,
                    )
                    if not _result_success(candidate_result):
                        continue
                    approach_result = candidate_result
                    approach_meta = candidate_meta
                    world_from_approach = world_from_approach_candidate
                    approach_candidate_index = app_idx
                    break
                if not _result_success(approach_result):
                    raise RuntimeError(
                        f"Failed to plan for approach for {ground_op.name}. "
                        f"Status: {_result_status(approach_result)}"
                    )
                approach_js = _terminal_joint_state(world, _result_plan(approach_result, config))

                end_result, end_meta = _plan_segment(
                    world=world,
                    config=config,
                    start_js=approach_js,
                    desired_world_from_ee=world_from_ee,
                    desired_goal_js=world.joint_state_from_position(best_particle[q][None].clone()),
                    log_label=ground_op.name,
                    linear_axis="z",
                    allow_detached_retry=True,
                    approach_candidate_index=approach_candidate_index,
                )
                if not _result_success(end_result):
                    raise RuntimeError(
                        f"Failed to plan from approach to end for {ground_op.name}. "
                        f"Status: {_result_status(end_result)}"
                    )

            ik_fk = world.compute_ee_matrix(best_particle[q][None].clone())
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
            obj_from_ee = torch.inverse(obj_to_current_pose[obj]) @ world_from_ee_start
            ee_from_obj = torch.inverse(obj_from_ee)

            for segment_type, result, desired_world_from_ee, segment_meta, extra_debug in (
                ("place_retract", retract_result, world_from_retract, retract_meta, {"ik_debug": ik_debug}),
                ("place_approach", approach_result, world_from_approach, approach_meta, None),
                (
                    "place_place",
                    end_result,
                    world_from_ee,
                    end_meta,
                    {"place_motion_mode": str(place_motion_mode)},
                ),
            ):
                plan, last_js, dt = _append_segment_plan(
                    accum_plans=accum_plans,
                    world=world,
                    config=config,
                    label=ground_op.name,
                    segment_type=segment_type,
                    result=result,
                    desired_world_from_ee=desired_world_from_ee,
                    waypoint_index_offset=waypoint_index_offset,
                    motion_refinement_mode=motion_refinement_mode,
                    segment_meta=segment_meta,
                    selected_parameter_name=str(grasp),
                    selected_obj_from_grasp=obj_from_grasp,
                    object_name=str(obj),
                    extra_debug=extra_debug,
                )
                world_from_ee_path = world.compute_ee_matrix(plan.position)
                world_from_obj_path = world_from_ee_path @ ee_from_obj
                ts = visualizer.log_joint_trajectory_with_mat4x4(
                    traj=plan.position,
                    mat4x4_key=f"world/{obj}",
                    mat4x4=world_from_obj_path,
                    timeline=timeline,
                    start_time=ts,
                    dt=dt,
                )
                waypoint_index_offset += int(plan.position.shape[0])
                obj_to_current_pose[obj] = world_from_obj_path[-1]

            with timer.time(f"{timeline}_planning"):
                world.update_object_pose(obj, obj_to_current_pose[obj])
            ts = _append_gripper_step(
                accum_plans=accum_plans,
                config=config,
                action="open",
                label=ground_op.name,
                last_js=last_js,
                visualizer=visualizer,
                timeline=timeline,
                ts=ts,
            )
            continue

        if op_name == Push.name:
            button, pose, q = ground_op.values
            ts = _append_gripper_step(
                accum_plans=accum_plans,
                config=config,
                action="close",
                label=ground_op.name,
                last_js=last_js,
                visualizer=visualizer,
                timeline=timeline,
                ts=ts,
            )

            with timer.time(f"{timeline}_planning"):
                start_js = last_js
                world_from_push = (
                    best_particle[pose].clone()
                    if best_particle[pose].shape == (4, 4)
                    else (
                        action_4dof_to_mat4x4(best_particle[pose].clone())
                        if config.push_dof == 4
                        else action_6dof_to_mat4x4(best_particle[pose].clone())
                    )
                )
                world_from_ee = world_from_push
                world_from_retract = world_from_ee
                retract_result = None
                retract_meta = None
                retract_js = start_js
                if last_q_name != "q0":
                    world_from_retract = world.compute_ee_matrix(start_js.position) @ approach_offset
                    retract_result, retract_meta = _plan_segment(
                        world=world,
                        config=config,
                        start_js=start_js,
                        desired_world_from_ee=world_from_retract,
                        log_label=ground_op.name,
                        linear_axis="z",
                    )
                    if not _result_success(retract_result):
                        raise RuntimeError(
                            f"Failed to plan for retract for {ground_op.name}. "
                            f"Status: {_result_status(retract_result)}"
                        )
                    retract_js = _terminal_joint_state(world, _result_plan(retract_result, config))

                world_from_approach = world_from_ee @ approach_offset
                approach_result, approach_meta = _plan_segment(
                    world=world,
                    config=config,
                    start_js=retract_js,
                    desired_world_from_ee=world_from_approach,
                    log_label=ground_op.name,
                    approach_candidate_index=0,
                )
                if not _result_success(approach_result):
                    raise RuntimeError(
                        f"Failed to plan for approach for {ground_op.name}. "
                        f"Status: {_result_status(approach_result)}"
                    )
                approach_js = _terminal_joint_state(world, _result_plan(approach_result, config))

                end_result, end_meta = _plan_segment(
                    world=world,
                    config=config,
                    start_js=approach_js,
                    desired_world_from_ee=world_from_ee,
                    desired_goal_js=world.joint_state_from_position(best_particle[q][None].clone()),
                    log_label=ground_op.name,
                    linear_axis="z",
                    allow_detached_retry=True,
                    obstacle_name=button,
                    approach_candidate_index=0,
                )
                if not _result_success(end_result):
                    raise RuntimeError(
                        f"Failed to plan from approach to end for {ground_op.name}. "
                        f"Status: {_result_status(end_result)}"
                    )

            for segment_type, result, desired_world_from_ee, segment_meta in (
                (
                    "push_retract",
                    retract_result,
                    world_from_retract if retract_result is not None else world_from_ee,
                    retract_meta,
                ),
                ("push_approach", approach_result, world_from_approach, approach_meta),
                ("push_execute", end_result, world_from_ee, end_meta),
            ):
                if result is None:
                    continue
                plan, last_js, dt = _append_segment_plan(
                    accum_plans=accum_plans,
                    world=world,
                    config=config,
                    label=ground_op.name,
                    segment_type=segment_type,
                    result=result,
                    desired_world_from_ee=desired_world_from_ee,
                    waypoint_index_offset=waypoint_index_offset,
                    motion_refinement_mode=motion_refinement_mode,
                    segment_meta=segment_meta,
                    selected_parameter_name=str(pose),
                )
                ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
                waypoint_index_offset += int(plan.position.shape[0])
            continue

        raise NotImplementedError(f"Unsupported operator {op_name}")

    start_js = last_js
    world_from_retract = world.compute_ee_matrix(start_js.position) @ approach_offset
    retract_result, retract_meta = _plan_segment(
        world=world,
        config=config,
        start_js=start_js,
        desired_world_from_ee=world_from_retract,
        log_label="GoToInitial(q0)",
        linear_axis="z",
        allow_detached_retry=True,
    )
    if not _result_success(retract_result):
        raise RuntimeError(f"Failed to plan for retract. Status: {_result_status(retract_result)}")
    plan, last_js, dt = _append_segment_plan(
        accum_plans=accum_plans,
        world=world,
        config=config,
        label="GoToInitial(q0)",
        segment_type="return_retract",
        result=retract_result,
        desired_world_from_ee=world_from_retract,
        waypoint_index_offset=waypoint_index_offset,
        motion_refinement_mode=motion_refinement_mode,
        segment_meta=retract_meta,
    )
    ts = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
    waypoint_index_offset += int(plan.position.shape[0])

    with timer.time(f"{timeline}_planning"):
        result = world.plan_cspace(
            last_js,
            world.joint_state_from_position(best_particle["q0"][None].clone()),
        )
    if not _result_success(result):
        raise RuntimeError("Failed to plan for going home")
    plan = _result_plan(result, config)
    dt = _plan_dt(plan)
    accum_plans.append(
        {
            "type": "trajectory",
            "plan": plan,
            "dt": dt,
            "label": "GoToInitial(q0)",
            "debug": {
                "label": "GoToInitial(q0)",
                "segment_type": "go_home_joint",
                "motion_refinement_mode": motion_refinement_mode,
                "segment_target_space": "joint",
                "goal_mode": "joint_goal",
                "waypoint_index_offset": int(waypoint_index_offset),
                "num_waypoints": int(plan.position.shape[0]),
            },
        }
    )
    _ = visualizer.log_joint_trajectory(plan.position, timeline=timeline, start_time=ts, dt=dt)
    _log.debug("Planned to go home")

    _log.info(f"Motion planning metrics: {timer.get_summary(f'{timeline}_planning')}")
    return accum_plans
