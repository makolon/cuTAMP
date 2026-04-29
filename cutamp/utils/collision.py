# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
from collections.abc import Callable

import torch
from curobo.scene import Cuboid, Scene
from curobo.types import DeviceCfg, Pose

_log = logging.getLogger(__name__)


def _obb_cuboids(world_scene: Scene) -> tuple[Cuboid, ...]:
    obb_scene = Scene.create_obb_world(world_scene)
    cuboids = tuple(getattr(obb_scene, "cuboid", ()) or ())
    _log.debug("Created OBB world with %d cuboids", len(cuboids))
    return cuboids


def _sphere_cuboid_signed_distance(
    *,
    sphere_centers_world: torch.Tensor,
    sphere_radii: torch.Tensor,
    cuboid: Cuboid,
    device_cfg: DeviceCfg,
) -> torch.Tensor:
    world_from_cuboid = Pose.from_list(cuboid.pose).get_matrix()[0].to(
        device=sphere_centers_world.device,
        dtype=sphere_centers_world.dtype,
    )
    cuboid_center = world_from_cuboid[:3, 3]
    cuboid_rot = world_from_cuboid[:3, :3]
    cuboid_half_extents = device_cfg.to_device(cuboid.dims).to(
        device=sphere_centers_world.device,
        dtype=sphere_centers_world.dtype,
    ) * 0.5

    centers_local = (sphere_centers_world - cuboid_center) @ cuboid_rot
    delta = centers_local.abs() - cuboid_half_extents
    outside = torch.relu(delta)
    outside_distance = torch.linalg.norm(outside, dim=-1)
    inside_distance = torch.clamp(delta.max(dim=-1).values, max=0.0)
    return outside_distance + inside_distance - sphere_radii


def get_world_collision_cost(
    world_scene: Scene,
    device_cfg: DeviceCfg,
    collision_activation_distance: float,
    weight: float = 1.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a differentiable sphere-vs-static-world collision cost.

    The input tensor must have shape ``[batch, horizon, num_spheres, 4]``.
    The output tensor has shape ``[batch, horizon]`` and sums collision
    penalties across all spheres and all static OBB approximations.
    """

    if collision_activation_distance < 0.0:
        raise ValueError(
            f"Collision activation distance must be >= 0.0, not {collision_activation_distance}"
        )

    cuboids = _obb_cuboids(world_scene)

    def collision_cost(spheres_world: torch.Tensor) -> torch.Tensor:
        if spheres_world.ndim != 4 or spheres_world.shape[-1] != 4:
            raise ValueError(
                "Expected spheres_world with shape [batch, horizon, num_spheres, 4], "
                f"got {tuple(spheres_world.shape)}"
            )
        batch, horizon, _, _ = spheres_world.shape
        if not cuboids:
            return torch.zeros(
                (batch, horizon),
                device=spheres_world.device,
                dtype=spheres_world.dtype,
            )

        sphere_centers = spheres_world[..., :3]
        sphere_radii = spheres_world[..., 3]
        total_cost = torch.zeros(
            (batch, horizon),
            device=spheres_world.device,
            dtype=spheres_world.dtype,
        )
        for cuboid in cuboids:
            signed_distance = _sphere_cuboid_signed_distance(
                sphere_centers_world=sphere_centers,
                sphere_radii=sphere_radii,
                cuboid=cuboid,
                device_cfg=device_cfg,
            )
            penalty = torch.relu(collision_activation_distance - signed_distance) * weight
            total_cost = total_cost + penalty.sum(dim=-1)
        return total_cost

    return collision_cost
