from dataclasses import dataclass

import torch
from jaxtyping import Float
from curobo.kinematics import Kinematics


@dataclass(frozen=True)
class RobotContainer:
    name: str
    kinematics: Kinematics
    joint_names: tuple[str, ...]
    tool_frame: str
    joint_limits: Float[torch.Tensor, "2 d"]
    # Note: in tool frame, not end-effector
    gripper_spheres: Float[torch.Tensor, "n 4"]
    # Transformation from tool pose to end-effector (defined in cuRobo config)
    tool_from_ee: Float[torch.Tensor, "4 4"]
    robot_cfg: dict
    rerun_robot: object = None
    gripper_family: str = ""
    visualizer_gripper_open: tuple[float, ...] = ()
    visualizer_gripper_closed: tuple[float, ...] = ()
