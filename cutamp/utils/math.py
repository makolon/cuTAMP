import torch
from jaxtyping import Float


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