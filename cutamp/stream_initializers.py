from __future__ import annotations

from collections.abc import Mapping
from typing import Iterator

import torch
from jaxtyping import Float

from cutamp.utils.common import action_4dof_to_mat4x4, action_6dof_to_mat4x4
from cutamp.utils.math import rotmat_to_euler_xyz


def as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def get_stream_data(stream_initializers: Mapping[str, object] | None, stream_name: str) -> dict[str, object]:
    if stream_initializers is None:
        return {}
    return as_mapping(stream_initializers.get(stream_name))


def sample_initializer_indices(
    num_candidates: int,
    num_particles: int,
    *,
    device: torch.device,
    scores: torch.Tensor | None = None,
) -> torch.Tensor:
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive")
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")

    if scores is None or scores.numel() != num_candidates:
        if num_candidates >= num_particles:
            return torch.randperm(num_candidates, device=device)[:num_particles]
        return torch.randint(0, num_candidates, (num_particles,), device=device)

    scores = scores.to(device=device, dtype=torch.float32).reshape(-1)
    scores = torch.clamp(scores, min=0.0)
    if torch.all(scores <= 0):
        weights = torch.ones_like(scores) / float(num_candidates)
    else:
        weights = scores / scores.sum()
    replacement = num_candidates < num_particles
    return torch.multinomial(weights, num_samples=num_particles, replacement=replacement)


def grasp_data_to_actions(
    grasps_obj: Float[torch.Tensor, "n 4 4"],
    grasp_dof: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if grasps_obj.ndim != 3 or grasps_obj.shape[-2:] != (4, 4):
        raise ValueError(f"Expected grasp matrices with shape (n, 4, 4), got {tuple(grasps_obj.shape)}")

    if grasp_dof == 4:
        translation = grasps_obj[:, :3, 3]
        yaw = torch.atan2(grasps_obj[:, 1, 0], grasps_obj[:, 0, 0]).unsqueeze(-1)
        actions = torch.cat([translation, yaw], dim=1)
        return actions, action_4dof_to_mat4x4(actions)

    if grasp_dof == 6:
        translation = grasps_obj[:, :3, 3]
        rotation = grasps_obj[:, :3, :3]
        rpy = rotmat_to_euler_xyz(rotation)
        actions = torch.cat([translation, rpy], dim=1)
        return actions, action_6dof_to_mat4x4(actions)

    raise ValueError(f"Unsupported grasp_dof: {grasp_dof}")


def place_data_to_actions(placements_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if placements_world.ndim == 2 and placements_world.shape[-1] == 4:
        return placements_world, action_4dof_to_mat4x4(placements_world)

    if placements_world.ndim == 3 and placements_world.shape[-2:] == (4, 4):
        translation = placements_world[:, :3, 3]
        yaw = torch.atan2(placements_world[:, 1, 0], placements_world[:, 0, 0]).unsqueeze(-1)
        actions = torch.cat([translation, yaw], dim=1)
        return actions, action_4dof_to_mat4x4(actions)

    raise ValueError(
        "Expected placements as either (n, 4) xyz+yaw actions or (n, 4, 4) transforms, "
        f"got {tuple(placements_world.shape)}"
    )


def push_data_to_actions(pushes_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pushes_world.ndim == 2 and pushes_world.shape[-1] == 4:
        return pushes_world, action_4dof_to_mat4x4(pushes_world)

    if pushes_world.ndim == 3 and pushes_world.shape[-2:] == (4, 4):
        translation = pushes_world[:, :3, 3]
        yaw = torch.atan2(pushes_world[:, 1, 0], pushes_world[:, 0, 0]).unsqueeze(-1)
        actions = torch.cat([translation, yaw], dim=1)
        return actions, action_4dof_to_mat4x4(actions)

    raise ValueError(
        "Expected pushes as either (n, 4) xyz+yaw actions or (n, 4, 4) transforms, "
        f"got {tuple(pushes_world.shape)}"
    )
