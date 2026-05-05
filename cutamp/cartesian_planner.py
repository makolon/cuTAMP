# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""End-effector (Cartesian) space motion planning for cuTAMP.

The cuRobo MotionGen-based path planner in :mod:`cutamp.motion_solver` runs joint-space
trajectory optimization. The resulting motions can curve through joint space and look
unnatural when replayed visually. This module provides a complementary planner that
interpolates the end-effector pose linearly between start and goal, then solves IK at
each waypoint with cuRobo's IK solver. The result mimics
:class:`curobo.wrap.reacher.motion_gen.MotionGenResult` so the rest of cuTAMP's plan
emission code can consume it without changes.

IK is solved sequentially with the previous joint configuration provided as both the
seed and the retract configuration. This is the IK-servoing pattern recommended by
cuRobo (see ``IKSolver.solve_single`` docstring) and is what keeps each waypoint on
the same IK branch as the previous one. Solving each waypoint independently with
random seeds (``seed_config=None``) lets adjacent waypoints converge to different
branches (e.g., elbow flip, wrist ±pi rollover), which produces a Cartesian-smooth
EE path but a visibly jerky joint trajectory.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenResult, MotionGenStatus
from cutamp.config import TAMPConfiguration

_log = logging.getLogger(__name__)


def _quat_dot(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    return (q_a * q_b).sum(dim=-1, keepdim=True)


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


def plan_cartesian_linear(
    *,
    start_js: JointState,
    world_from_goal_ee: torch.Tensor,
    world,  # TAMPWorld; typed-as-object to avoid a cyclic import.
    config: TAMPConfiguration,
) -> MotionGenResult:
    """Plan a Cartesian-linear EE-space motion from ``start_js`` to ``world_from_goal_ee``.

    Returns a :class:`MotionGenResult` with ``success=False`` and a descriptive
    :class:`MotionGenStatus` when IK fails. ``solve_curobo`` raises a ``RuntimeError``
    on this, which surfaces as a normal motion-planning failure to the TAMP loop.
    """

    device = start_js.position.device
    dtype = start_js.position.dtype
    num_waypoints = max(int(config.ee_planning_num_waypoints), 2)

    world_from_start_ee = world.kin_model.get_state(start_js.position).ee_pose.get_matrix()[0]
    interpolated_poses = _interpolate_ee_path(
        world_from_start_ee=world_from_start_ee,
        world_from_goal_ee=world_from_goal_ee,
        num_waypoints=num_waypoints,
        device=device,
        dtype=dtype,
    )

    # Solve IK sequentially, seeding each waypoint with the joint configuration of the
    # previous waypoint. This biases the null-space cost toward the prior branch and is
    # what keeps the joint trajectory continuous across adjacent waypoints. The first
    # waypoint corresponds (up to FK precision) to ``start_js.position`` by construction,
    # so we use it directly without re-solving IK to avoid an FK->IK round-trip drift.
    q_start = start_js.position[:1]  # (1, dof)
    positions_list = [q_start]
    prev_q = q_start
    for waypoint_idx in range(1, num_waypoints):
        wp_pose = Pose(
            position=interpolated_poses.position[waypoint_idx : waypoint_idx + 1],
            quaternion=interpolated_poses.quaternion[waypoint_idx : waypoint_idx + 1],
        )
        # ``seed_config`` for ``solve_single`` is shape (batch=1, n_seeds=1, dof). The
        # remaining ``num_seeds - 1`` random seeds are still optimized in parallel and
        # serve as a safety net when the seeded branch is locally infeasible; the
        # null-space cost (regularization=True at IK init) biases selection toward the
        # solution closest to ``retract_config``.
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
        # ``solution`` is shape (1, num_seeds, dof) for solve_single; index 0 is the best.
        prev_q = ik_result.solution[:, 0]
        positions_list.append(prev_q)

    positions = torch.cat(positions_list, dim=0)

    # Time-parametrize so that the end-effector moves at roughly the configured velocity.
    segment_lengths = torch.linalg.norm(
        interpolated_poses.position[1:] - interpolated_poses.position[:-1],
        dim=-1,
    )
    total_length = float(segment_lengths.sum().item())
    target_velocity = max(float(config.ee_planning_velocity), 1e-3)
    total_time = max(total_length / target_velocity, 1e-3)
    interpolation_dt = total_time / max(num_waypoints - 1, 1)

    # Apply the global time-dilation factor consistently with cuRobo's MotionGen output.
    if config.time_dilation_factor is not None and config.time_dilation_factor > 0.0:
        interpolation_dt = interpolation_dt / float(config.time_dilation_factor)

    # Finite-difference velocities; clamped to keep the last sample finite.
    velocities = torch.zeros_like(positions)
    if positions.shape[0] >= 2 and interpolation_dt > 0.0:
        velocities[:-1] = (positions[1:] - positions[:-1]) / interpolation_dt

    plan = JointState(
        position=positions,
        velocity=velocities,
        acceleration=torch.zeros_like(positions),
        jerk=torch.zeros_like(positions),
        joint_names=start_js.joint_names,
    )

    if not math.isfinite(interpolation_dt):
        interpolation_dt = 0.02

    # Smoothness diagnostics. Logs the per-step joint deltas so regressions in IK
    # continuity are visible without requiring downstream replay. ``max_step`` is the
    # most useful single number: a value much larger than ``mean_step`` typically
    # signals an IK-branch jump at one waypoint.
    if _log.isEnabledFor(logging.DEBUG) and positions.shape[0] >= 2:
        joint_deltas = (positions[1:] - positions[:-1]).abs()
        max_step = float(joint_deltas.max().item())
        mean_step = float(joint_deltas.mean().item())
        worst_step_idx = int(joint_deltas.max(dim=-1).values.argmax().item())
        _log.debug(
            "Cartesian-linear plan smoothness: max_joint_step=%.4f rad, "
            "mean_joint_step=%.4f rad, worst_at_waypoint=%d/%d",
            max_step,
            mean_step,
            worst_step_idx + 1,  # +1 because deltas are between i and i+1
            num_waypoints,
        )

    return _build_motion_gen_result(
        plan=plan,
        interpolation_dt=interpolation_dt,
        success=True,
        status=MotionGenStatus.SUCCESS,
    )
