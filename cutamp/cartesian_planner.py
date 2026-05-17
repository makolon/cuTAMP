# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.trajectory import InterpolateType, get_batch_interpolated_trajectory
from curobo.wrap.reacher.motion_gen import MotionGenResult, MotionGenStatus
from cutamp.config import TAMPConfiguration

_log = logging.getLogger(__name__)


def _slerp_quat(q_start: torch.Tensor, q_end: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two unit quaternions in wxyz layout.

    Args:
        q_start: ``(4,)`` tensor for the starting quaternion.
        q_end: ``(4,)`` tensor for the ending quaternion.
        alpha: ``(N, 1)`` interpolation factors in ``[0, 1]``.

    Returns:
        ``(N, 4)`` interpolated quaternions on the same device as ``alpha``.
    """

    q_start = q_start / q_start.norm()
    q_end = q_end / q_end.norm()
    dot = (q_start * q_end).sum()
    if dot.item() < 0.0:
        q_end = -q_end
        dot = -dot
    dot = dot.clamp(-1.0, 1.0)
    if (1.0 - dot.item()) < 1e-6:
        out = q_start.unsqueeze(0) + alpha * (q_end.unsqueeze(0) - q_start.unsqueeze(0))
        return out / out.norm(dim=-1, keepdim=True)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    a = torch.sin((1.0 - alpha) * omega) / sin_omega
    b = torch.sin(alpha * omega) / sin_omega
    return a * q_start.unsqueeze(0) + b * q_end.unsqueeze(0)


def _interpolate_ee_path(
    *,
    world_from_start_ee: torch.Tensor,
    world_from_goal_ee: torch.Tensor,
    num_waypoints: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Pose:
    """Linearly interpolate position and slerp orientation between two EE poses."""

    start_pos = world_from_start_ee[:3, 3]
    goal_pos = world_from_goal_ee[:3, 3]
    start_pose = Pose.from_matrix(world_from_start_ee.unsqueeze(0))
    goal_pose = Pose.from_matrix(world_from_goal_ee.unsqueeze(0))
    q_start = start_pose.quaternion[0]
    q_end = goal_pose.quaternion[0]

    alpha = torch.linspace(0.0, 1.0, num_waypoints, device=device, dtype=dtype).unsqueeze(-1)
    positions = start_pos.unsqueeze(0) + alpha * (goal_pos - start_pos).unsqueeze(0)
    quaternions = _slerp_quat(q_start, q_end, alpha)
    return Pose(position=positions, quaternion=quaternions)


def _build_motion_gen_result(
    *,
    plan: JointState,
    interpolation_dt: float,
    success: bool,
    status: Optional[MotionGenStatus] = None,
) -> MotionGenResult:
    """Wrap a Cartesian plan in the dataclass shape downstream code expects."""

    success_tensor = torch.tensor([bool(success)], device=plan.position.device)
    optimized_dt = torch.tensor([float(interpolation_dt)], device=plan.position.device)
    return MotionGenResult(
        success=success_tensor,
        valid_query=True,
        optimized_plan=plan,
        optimized_dt=optimized_dt,
        interpolated_plan=plan,
        interpolation_dt=float(interpolation_dt),
        status=status,
    )


def _interpolate_with_curobo(
    *,
    positions: torch.Tensor,
    raw_dt: float,
    target_interpolation_dt: float,
    time_dilation_factor: Optional[float],
    kin_model,
    joint_names,
    device: torch.device,
) -> tuple[JointState, float]:
    """Resample IK joint waypoints via cuRobo's batched CUBIC interpolator."""

    n_in = int(positions.shape[0])
    dof = int(positions.shape[1])

    # Finite-difference seed velocity/acceleration/jerk.
    raw_vel = torch.zeros_like(positions)
    raw_acc = torch.zeros_like(positions)
    raw_jerk = torch.zeros_like(positions)
    if n_in >= 2 and raw_dt > 0.0:
        raw_vel[:-1] = (positions[1:] - positions[:-1]) / raw_dt
    if n_in >= 3 and raw_dt > 0.0:
        raw_acc[:-1] = (raw_vel[1:] - raw_vel[:-1]) / raw_dt
    if n_in >= 4 and raw_dt > 0.0:
        raw_jerk[:-1] = (raw_acc[1:] - raw_acc[:-1]) / raw_dt

    raw_plan = JointState(
        position=positions,
        velocity=raw_vel,
        acceleration=raw_acc,
        jerk=raw_jerk,
        joint_names=joint_names,
    )

    if n_in < 2 or raw_dt <= 0.0:
        # Nothing to interpolate; return a stub plan with zero velocities.
        return raw_plan, max(raw_dt, target_interpolation_dt)

    joint_limits = kin_model.get_joint_limits()
    max_vel = joint_limits.velocity[1, :].abs().to(device)
    max_acc = joint_limits.acceleration[1, :].abs().to(device)
    max_jerk_t = joint_limits.jerk[1, :].abs().to(device)

    raw_dt_tensor = torch.tensor([raw_dt], device=device, dtype=torch.float32)
    tensor_args = TensorDeviceType(device=device, dtype=torch.float32)

    out_traj, traj_steps, opt_dt = get_batch_interpolated_trajectory(
        raw_plan,
        raw_dt=raw_dt_tensor,
        interpolation_dt=float(target_interpolation_dt),
        max_vel=max_vel,
        max_acc=max_acc,
        max_jerk=max_jerk_t,
        kind=InterpolateType.CUBIC,
        tensor_args=tensor_args,
        min_dt=float(target_interpolation_dt) * 0.5,
        max_dt=max(float(raw_dt) * 4.0, float(target_interpolation_dt) * 4.0),
        optimize_dt=True,
    )

    length = int(traj_steps[0].item())
    if length < 2:
        return raw_plan, max(raw_dt, target_interpolation_dt)

    dense_positions = out_traj.position[0, :length].to(device)
    final_dt = float(target_interpolation_dt)

    if time_dilation_factor is not None and time_dilation_factor > 0.0:
        final_dt = final_dt / float(time_dilation_factor)

    dense_vel = torch.zeros_like(dense_positions)
    if dense_positions.shape[0] >= 2 and final_dt > 0.0:
        dense_vel[:-1] = (dense_positions[1:] - dense_positions[:-1]) / final_dt

    plan = JointState(
        position=dense_positions,
        velocity=dense_vel,
        acceleration=torch.zeros_like(dense_positions),
        jerk=torch.zeros_like(dense_positions),
        joint_names=joint_names,
    )
    return plan, final_dt


def plan_cartesian_linear(
    *,
    start_js: JointState,
    world_from_goal_ee: torch.Tensor,
    world,
    config: TAMPConfiguration,
) -> MotionGenResult:
    """Plan a Cartesian-linear EE-space motion from ``start_js`` to ``world_from_goal_ee``.

    Returns a :class:`MotionGenResult` with ``success=False`` and a descriptive
    :class:`MotionGenStatus` when IK fails. ``solve_curobo`` raises a ``RuntimeError``
    on this, which surfaces as a normal motion-planning failure to the TAMP loop.
    """

    device = start_js.position.device
    dtype = start_js.position.dtype

    world_from_start_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]

    # Distance-adaptive IK waypoint count.
    ee_distance = float(
        torch.linalg.norm(world_from_goal_ee[:3, 3] - world_from_start_ee[:3, 3]).item()
    )
    target_step = float(config.ee_planning_step_m)
    num_waypoints = max(2, int(math.ceil(ee_distance / target_step)) + 1)

    interpolated_poses = _interpolate_ee_path(
        world_from_start_ee=world_from_start_ee,
        world_from_goal_ee=world_from_goal_ee,
        num_waypoints=num_waypoints,
        device=device,
        dtype=dtype,
    )

    q_start = start_js.position[:1]  # (1, dof)
    positions_list = [q_start]
    prev_q = q_start
    for waypoint_idx in range(1, num_waypoints):
        wp_pose = Pose(
            position=interpolated_poses.position[waypoint_idx : waypoint_idx + 1],
            quaternion=interpolated_poses.quaternion[waypoint_idx : waypoint_idx + 1],
        )
        ik_result = world.ik_solver.solve_single(
            wp_pose,
            retract_config=prev_q,
            seed_config=prev_q.unsqueeze(1),
        )
        if not bool(ik_result.success.view(-1)[0].item()):
            _log.debug(
                "Cartesian-linear IK failed at waypoint %d/%d",
                waypoint_idx,
                num_waypoints,
            )
            stub_positions = start_js.position.expand(num_waypoints, -1).contiguous()
            stub_plan = JointState.from_position(stub_positions, joint_names=start_js.joint_names)
            return _build_motion_gen_result(
                plan=stub_plan,
                interpolation_dt=0.02,
                success=False,
                status=MotionGenStatus.IK_FAIL,
            )
        # solution is shape (1, num_seeds, dof) for solve_single; index 0 is the best.
        prev_q = ik_result.solution[:, 0]
        positions_list.append(prev_q)

    positions = torch.cat(positions_list, dim=0)

    segment_lengths = torch.linalg.norm(
        interpolated_poses.position[1:] - interpolated_poses.position[:-1],
        dim=-1,
    )
    total_length = float(segment_lengths.sum().item())
    target_velocity = max(float(config.ee_planning_velocity), 1e-3)
    total_time = max(total_length / target_velocity, 1e-3)
    raw_dt = total_time / max(num_waypoints - 1, 1)
    if not math.isfinite(raw_dt):
        raw_dt = 0.02

    plan, interpolation_dt = _interpolate_with_curobo(
        positions=positions,
        raw_dt=raw_dt,
        target_interpolation_dt=float(config.ee_planning_interpolation_dt),
        time_dilation_factor=config.time_dilation_factor,
        kin_model=world.kin_model,
        joint_names=start_js.joint_names,
        device=device,
    )

    if _log.isEnabledFor(logging.DEBUG):
        joint_deltas = (plan.position[1:] - plan.position[:-1]).abs()
        max_step = float(joint_deltas.max().item()) if joint_deltas.numel() else 0.0
        mean_step = float(joint_deltas.mean().item()) if joint_deltas.numel() else 0.0
        _log.debug(
            "Cartesian-linear plan: ee_distance=%.3fm, ik_waypoints=%d, "
            "dense_waypoints=%d, interpolation_dt=%.4fs, "
            "max_joint_step=%.4f rad, mean_joint_step=%.4f rad",
            ee_distance,
            num_waypoints,
            int(plan.position.shape[0]),
            interpolation_dt,
            max_step,
            mean_step,
        )

    return _build_motion_gen_result(
        plan=plan,
        interpolation_dt=interpolation_dt,
        success=True,
        status=MotionGenStatus.SUCCESS,
    )
