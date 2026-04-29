# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import defaultdict
from typing import Dict

import roma
import torch
from curobo.scene import (
    Capsule,
    Cuboid,
    Cylinder,
    Mesh,
    Obstacle,
    Scene,
    Sphere,
)
from einops import einsum
from jaxtyping import Float

from cutamp.envs import TAMPEnvironment
from cutamp.utils.shapes import MultiSphere

Particles = Dict[str, Float[torch.Tensor, "num_particles *h d"]]


def action_4dof_to_mat4x4(action_4dof: Float[torch.Tensor, "*b 4"]) -> Float[torch.Tensor, "*b 4 4"]:
    """Convert 4-DoF action [x, y, z, yaw] to a 4x4 matrix."""
    if action_4dof.shape[-1] != 4:
        raise ValueError(f"Expected last dimension to be 4, got shape {action_4dof.shape}")

    mat4x4 = torch.eye(4, device=action_4dof.device, dtype=action_4dof.dtype)
    mat4x4 = mat4x4.repeat(*action_4dof.shape[:-1], 1, 1)

    yaw = action_4dof[..., 3]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    mat4x4[..., 0, 0] = cos_yaw
    mat4x4[..., 0, 1] = -sin_yaw
    mat4x4[..., 1, 0] = sin_yaw
    mat4x4[..., 1, 1] = cos_yaw
    mat4x4[..., :3, 3] = action_4dof[..., :3]

    return mat4x4


def action_6dof_to_mat4x4(action_6dof: Float[torch.Tensor, "*b 6"]) -> Float[torch.Tensor, "*b 4 4"]:
    """Convert 6-DoF action [x, y, z, roll, pitch, yaw] to a 4x4 matrix."""
    translation = action_6dof[..., :3]
    rpy = action_6dof[..., 3:]

    mat4x4 = torch.eye(4, device=action_6dof.device, dtype=action_6dof.dtype)
    mat4x4 = mat4x4.repeat(*action_6dof.shape[:-1], 1, 1)
    mat4x4[..., :3, :3] = roma.euler_to_rotmat("XYZ", rpy)
    mat4x4[..., :3, 3] = translation

    return mat4x4


def transform_spheres(
    spheres: Float[torch.Tensor, "num_spheres 4"],
    transform: Float[torch.Tensor, "*b 4 4"],
) -> Float[torch.Tensor, "*b num_spheres 4"]:
    """Transform object-frame spheres by 4x4 transforms."""
    centers = spheres[:, :3]
    centers_hom = torch.cat([centers, torch.ones_like(centers[:, :1])], dim=1)
    radii = spheres[:, 3]

    if transform.ndim > 2:
        transform = transform.unsqueeze(-3)

    out_spheres = einsum(transform, centers_hom, "... i j, ... j -> ... i")
    out_spheres[..., 3] = radii

    return out_spheres.contiguous()


def transform_points(
    points: Float[torch.Tensor, "num_points 3"],
    transform: Float[torch.Tensor, "*b 4 4"],
) -> Float[torch.Tensor, "*b num_points 3"]:
    """Transform object-frame points by 4x4 transforms."""
    points_hom = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)

    if transform.ndim > 2:
        transform = transform.unsqueeze(-3)

    out_points_hom = einsum(transform, points_hom, "... i j, ... j -> ... i")
    return out_points_hom[..., :3]


def sample_between_bounds(num_samples: int, bounds: Float[torch.Tensor, "2 d"]) -> Float[torch.Tensor, "num_samples d"]:
    """Sample uniformly between lower and upper joint limits."""
    lower, upper = bounds
    samples = torch.rand(num_samples, *lower.shape, device=bounds.device, dtype=bounds.dtype)
    return lower + samples * (upper - lower)


def approximate_goal_aabb(goal: Obstacle) -> Float[torch.Tensor, "2 3"]:
    """Approximate an obstacle AABB in world coordinates."""
    pose = torch.as_tensor(goal.pose, dtype=torch.float32)
    pos = pose[:3]
    quat_xyzw = roma.quat_wxyz_to_xyzw(pose[3:])

    mat4x4 = torch.eye(4, dtype=pose.dtype)
    mat4x4[:3, :3] = roma.unitquat_to_rotmat(quat_xyzw)
    mat4x4[:3, 3] = pos

    if isinstance(goal, MultiSphere):
        goal = goal.get_mesh(process=False)

    if isinstance(goal, Cuboid):
        half_extents = torch.as_tensor(goal.dims, dtype=mat4x4.dtype) / 2
        center = mat4x4[:3, 3]
        world_half_extents = torch.abs(mat4x4[:3, :3]) @ half_extents
        return torch.stack([center - world_half_extents, center + world_half_extents])

    if isinstance(goal, Mesh):
        vertices = torch.as_tensor(goal.vertices, dtype=mat4x4.dtype)
        vertices = transform_points(vertices, mat4x4)
        return torch.stack([vertices.min(dim=0).values, vertices.max(dim=0).values])

    if isinstance(goal, (Sphere, Cylinder, Capsule)):
        mesh = goal.get_mesh(process=False)
        vertices = torch.as_tensor(mesh.vertices, dtype=mat4x4.dtype)
        vertices = transform_points(vertices, mat4x4)
        return torch.stack([vertices.min(dim=0).values, vertices.max(dim=0).values])

    raise NotImplementedError(f"Goal type {type(goal)} not supported yet.")


def get_world_cfg(env: TAMPEnvironment, include_movables: bool = False) -> Scene:
    """Build a cuRobo-v2-facing Scene from a TAMPEnvironment.

    cuRobo's collision loaders should receive only supported canonical scene
    objects. MultiSphere is a cuTAMP internal shape and is converted to Mesh.
    Sphere, Cylinder, and Capsule are also converted to Mesh to avoid failures
    in collision backends that only accept Cuboid and Mesh.
    """

    geoms = defaultdict(list)
    obstacles = list(env.movables) if include_movables else []
    obstacles += list(env.statics)

    for obj in obstacles:
        if isinstance(obj, Cuboid):
            geoms["cuboid"].append(obj)
        elif isinstance(obj, Sphere):
            geoms["sphere"].append(obj)
        elif isinstance(obj, Cylinder):
            geoms["cylinder"].append(obj)
        elif isinstance(obj, Capsule):
            geoms["capsule"].append(obj)
        elif isinstance(obj, (MultiSphere, Mesh)):
            # Need to use mesh for MultiSphere
            geoms["mesh"].append(obj.get_mesh(process=False))
        else:
            raise ValueError(f"Unknown object type: {type(obj)}")

    world_cfg = Scene(**geoms)

    return world_cfg
