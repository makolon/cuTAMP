# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional

import torch
from curobo.scene import Cuboid, Mesh, Obstacle
from jaxtyping import Float

from cutamp.utils.common import approximate_goal_aabb, pose_list_to_mat4x4, transform_points
from cutamp.utils.obb import get_object_obb
from cutamp.utils.shapes import MultiSphere

Grasp4DOF = Place4DOF = Float[torch.Tensor, "n 4"]
Grasp6DOF = Place6DOF = Float[torch.Tensor, "n 6"]


def _device_dtype(obj_spheres: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    return obj_spheres.device, obj_spheres.dtype


def sample_yaw(num_samples: int, num_faces: Optional[int], device: torch.device) -> torch.Tensor:
    """Sample continuous yaw, or one of a fixed number of yaw faces."""
    if num_faces is None:
        return torch.rand(num_samples, device=device) * 2 * torch.pi

    if num_faces < 1:
        raise ValueError(f"Expected num_faces >= 1, got {num_faces}")
    two_pi = 2 * torch.pi
    endpoint = two_pi - (two_pi / num_faces)
    yaw_choices = torch.linspace(0, endpoint, num_faces, device=device)
    yaw_idxs = torch.randint(0, num_faces, (num_samples,), device=device)
    return yaw_choices[yaw_idxs]


def sample_stick_grasps(num_samples: int, stick: MultiSphere) -> Grasp4DOF:
    """Sample simple 4-DOF grasps for a stick-like MultiSphere."""
    spheres = stick.spheres
    if not (spheres[:, 1:3] == 0.0).all():
        raise ValueError("Expected stick spheres to have y and z positions of 0")

    sphere_x = spheres[:, 0]
    x_idxs = torch.randint(0, len(sphere_x), (num_samples,), device=spheres.device)
    yaw = sample_yaw(num_samples, num_faces=2, device=spheres.device)

    grasp_4dof = torch.zeros((num_samples, 4), device=spheres.device, dtype=spheres.dtype)
    grasp_4dof[:, 0] = sphere_x[x_idxs]
    grasp_4dof[:, 3] = yaw.to(spheres.dtype)
    return grasp_4dof


def grasp_4dof_sampler(
    num_samples: int,
    obj: Obstacle,
    obj_spheres: Float[torch.Tensor, "n 4"],
    num_faces: Optional[int] = None,
) -> Grasp4DOF:
    """Sample 4-DOF grasps in the object frame."""
    if obj.name == "stick":
        return sample_stick_grasps(num_samples, obj)

    device, dtype = _device_dtype(obj_spheres)
    if isinstance(obj, Cuboid):
        obj_half_z = max(0.0, obj.dims[2] / 2 - 0.02)
    elif isinstance(obj, MultiSphere):
        obj_half_z = 0.0
    else:
        obj_half_z = obj_spheres[:, 2].max() - 0.02

    translation = torch.zeros(num_samples, 3, device=device, dtype=dtype)
    translation[:, 2] = torch.as_tensor(obj_half_z, device=device, dtype=dtype)
    yaw = sample_yaw(num_samples, num_faces, device).to(dtype)
    return torch.cat([translation, yaw.unsqueeze(-1)], dim=1)


def grasp_6dof_sampler(
    num_samples: int,
    obj: Obstacle,
    obj_spheres: Float[torch.Tensor, "n 4"],
    num_faces: Optional[int] = None,
) -> Grasp6DOF:
    """Sample simple 6-DOF grasps in the object frame."""
    if not isinstance(obj, Cuboid):
        raise NotImplementedError("Only Cuboid objects are supported for 6-DOF grasps")

    device, dtype = _device_dtype(obj_spheres)
    roll_choices = torch.tensor(
        [-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, torch.pi / 4, torch.pi / 3, torch.pi / 2],
        device=device,
        dtype=dtype,
    )
    roll = roll_choices[torch.randint(0, len(roll_choices), (num_samples,), device=device)]
    pitch = torch.zeros(num_samples, device=device, dtype=dtype)
    yaw_choices = torch.tensor([-torch.pi / 2, torch.pi / 2], device=device, dtype=dtype)
    yaw = yaw_choices[torch.randint(0, 2, (num_samples,), device=device)]
    rpy = torch.stack([roll, pitch, yaw], dim=1)

    half_extents = torch.tensor([dim / 2 for dim in obj.dims], device=device, dtype=dtype)
    gripper_offset = torch.as_tensor(0.01, device=device, dtype=dtype)
    upper = (half_extents - gripper_offset).clamp(min=0.0)
    lower = torch.full((3,), gripper_offset, device=device, dtype=dtype).clamp(max=upper)
    lower[0] = upper[0] = 0.0

    translation = lower + (upper - lower) * torch.rand(num_samples, 3, device=device, dtype=dtype)
    return torch.cat([translation, rpy], dim=1)


def place_4dof_sampler(
    num_samples: int,
    obj: Obstacle,
    obj_spheres: Float[torch.Tensor, "n 4"],
    surface: Obstacle,
    surface_rep: str,
    shrink_dist: float | None,
    collision_activation_dist: float,
) -> Place4DOF:
    """Sample 4-DOF object placement poses in the world frame."""
    if not isinstance(surface, (Cuboid, Mesh)):
        raise NotImplementedError(f"Only Cuboid or Mesh surfaces supported for now, not {type(surface)}")

    device, dtype = _device_dtype(obj_spheres)
    sph_bottom = obj_spheres[:, 2] - obj_spheres[:, 3]
    obj_z_delta = -sph_bottom.min()
    z_offset = torch.empty(num_samples, 1, device=device, dtype=dtype).uniform_(1e-3, 1e-2)

    if surface_rep == "obb":
        obb = get_object_obb(surface, shrink_dist)
        radial_distances = torch.sqrt(obj_spheres[:, 0] ** 2 + obj_spheres[:, 1] ** 2) + obj_spheres[:, 3]
        max_xy_extent = radial_distances.max()

        sampling_half_extents = (obb.half_extents[:2].to(device=device, dtype=dtype) - max_xy_extent).clamp(min=0.0)
        if (sampling_half_extents == 0.0).any():
            sampling_half_extents = obb.half_extents[:2].to(device=device, dtype=dtype)

        xy_local = (torch.rand(num_samples, 2, device=device, dtype=dtype) * 2 - 1) * sampling_half_extents
        xyz_local = torch.cat([xy_local, torch.zeros(num_samples, 1, device=device, dtype=dtype)], dim=1)
        xy_world = xyz_local @ obb.rot_matrix.to(device=device, dtype=dtype).T
        xy_world = xy_world[:, :2] + obb.center[:2].to(device=device, dtype=dtype)
        surface_z = torch.as_tensor(obb.surface_z, device=device, dtype=dtype)
        z_world = z_offset + obj_z_delta + surface_z + collision_activation_dist
        xyz_world = torch.cat([xy_world, z_world], dim=1)
    elif surface_rep == "aabb" and isinstance(surface, Cuboid):
        if shrink_dist is not None:
            raise ValueError("AABB placement sampling expects shrink_dist=None")
        aabb_xy_local = torch.tensor(
            [[-surface.dims[0] / 2, -surface.dims[1] / 2], [surface.dims[0] / 2, surface.dims[1] / 2]],
            device=device,
            dtype=dtype,
        )
        xy_local = aabb_xy_local[0] + torch.rand(num_samples, 2, device=device, dtype=dtype) * (
            aabb_xy_local[1] - aabb_xy_local[0]
        )
        z_local = z_offset + obj_z_delta + surface.dims[2] / 2 + collision_activation_dist
        xyz_local = torch.cat([xy_local, z_local], dim=1)
        surface_mat4x4 = pose_list_to_mat4x4(surface.pose).to(device=device, dtype=dtype)
        xyz_world = transform_points(xyz_local, surface_mat4x4)
    elif surface_rep == "aabb" and isinstance(surface, Mesh):
        if shrink_dist is not None:
            raise ValueError("AABB placement sampling expects shrink_dist=None")
        aabb_world = approximate_goal_aabb(surface).to(device=device, dtype=dtype)
        aabb_xy_world = aabb_world[:, :2]
        xy_world = aabb_xy_world[0] + torch.rand(num_samples, 2, device=device, dtype=dtype) * (
            aabb_xy_world[1] - aabb_xy_world[0]
        )
        z_world = z_offset + obj_z_delta + aabb_world[1, 2] + collision_activation_dist
        xyz_world = torch.cat([xy_world, z_world], dim=1)
    else:
        raise ValueError(f"Unsupported placement sampler: surface_rep={surface_rep}, surface={type(surface)}")

    yaw = sample_yaw(num_samples, None, device).to(dtype)
    return torch.cat([xyz_world, yaw.unsqueeze(-1)], dim=1)
