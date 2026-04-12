from __future__ import annotations

from collections.abc import Mapping
from typing import Iterator

import torch
from jaxtyping import Float

from cutamp.utils.common import action_4dof_to_mat4x4, action_6dof_to_mat4x4


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


def placement_data_to_actions(placements_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def rotmat_to_euler_xyz(rotation: Float[torch.Tensor, "n 3 3"]) -> Float[torch.Tensor, "n 3"]:
    sy = torch.sqrt(rotation[:, 0, 0] ** 2 + rotation[:, 1, 0] ** 2)
    singular = sy < 1e-6

    roll = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2])
    pitch = torch.atan2(-rotation[:, 2, 0], sy)
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])

    roll_singular = torch.atan2(-rotation[:, 1, 2], rotation[:, 1, 1])
    yaw_singular = torch.zeros_like(yaw)

    roll = torch.where(singular, roll_singular, roll)
    yaw = torch.where(singular, yaw_singular, yaw)
    return torch.stack([roll, pitch, yaw], dim=1)


def iter_stream_objects(stream_initializers: Mapping[str, object] | None) -> Iterator[object]:
    if stream_initializers is None:
        return

    seen: set[int] = set()

    def walk(value: object) -> Iterator[object]:
        if value is None:
            return

        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        yield value

        if isinstance(value, Mapping):
            for item in value.values():
                yield from walk(item)
            return

        if isinstance(value, (list, tuple, set)):
            for item in value:
                yield from walk(item)
            return

        attrs = getattr(value, "__dict__", None)
        if attrs is not None and isinstance(attrs, dict):
            for item in attrs.values():
                yield from walk(item)

    runtime_data = get_stream_data(stream_initializers, "runtime")
    if runtime_data:
        yield from walk(runtime_data)
    yield from walk(stream_initializers)


def find_stream_resource_by_methods(
    stream_initializers: Mapping[str, object] | None,
    required_methods: tuple[str, ...],
) -> object | None:
    if not required_methods:
        raise ValueError("required_methods must not be empty")

    for candidate in iter_stream_objects(stream_initializers):
        if all(callable(getattr(candidate, method_name, None)) for method_name in required_methods):
            return candidate
    return None
