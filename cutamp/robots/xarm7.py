import os
from functools import lru_cache
from pathlib import Path

import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from jaxtyping import Float
from yourdfpy import URDF

from cutamp.robots.utils import RerunRobot

xarm7_home = (0.0, -0.698, 0.0, 0.349, 0.0, 1.047, 0.0)
xarm7_gripper_open = 0.85
_ASSETS_DIR = Path(__file__).parent / "assets"


@lru_cache(maxsize=1)
def xarm7_curobo_cfg() -> dict:
    local_cfg = _ASSETS_DIR / "xarm7.yml"
    if local_cfg.exists():
        cfg = load_yaml(str(local_cfg))
        kin_cfg = cfg["robot_cfg"]["kinematics"]
        kin_cfg["external_asset_path"] = str(_ASSETS_DIR)
        kin_cfg["external_robot_configs_path"] = str(_ASSETS_DIR)
        return cfg
    return load_yaml(join_path(get_robot_configs_path(), "xarm7.yml"))


def get_xarm7_kinematics_model() -> CudaRobotModel:
    robot_cfg = RobotConfig.from_dict(xarm7_curobo_cfg()["robot_cfg"])
    return CudaRobotModel(robot_cfg.kinematics)


def get_xarm7_ik_solver(
    world_cfg: WorldConfig,
    num_seeds: int = 12,
    self_collision_opt: bool = False,
    self_collision_check: bool = True,
    use_particle_opt: bool = False,
) -> IKSolver:
    robot_cfg = xarm7_curobo_cfg()["robot_cfg"]
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        num_seeds=num_seeds,
        self_collision_opt=self_collision_opt,
        self_collision_check=self_collision_check,
        use_particle_opt=use_particle_opt,
    )
    return IKSolver(ik_config)


def get_xarm7_gripper_spheres(tensor_args: TensorDeviceType = TensorDeviceType()) -> Float[torch.Tensor, "n 4"]:
    # These spheres are derived from the xArm7 configuration used in
    # makolon/curobo_test:
    # src/curobo/content/configs/robot/spheres/xarm7.yml
    # together with the grasp-frame offsets from the matching xarm7.urdf.
    grasp_z = 0.15
    base_link_spheres = [
        [0.0, 0.0, 0.0, 0.042],
        [0.0, -0.025, 0.04, 0.042],
        [0.0, 0.025, 0.04, 0.042],
        [0.0, -0.025, 0.08, 0.042],
        [0.0, 0.025, 0.08, 0.042],
    ]
    left_finger_offset = [0.0, 0.070465, 0.101137]
    right_finger_offset = [0.0, -0.070465, 0.101137]
    finger_spheres = [
        [0.0, 0.0, 0.0, 0.012],
        [0.0, -0.01, 0.01, 0.012],
        [0.0, -0.02, 0.02, 0.012],
        [0.0, -0.02, 0.035, 0.012],
        [0.0, -0.02, 0.05, 0.012],
    ]

    spheres = []
    for x, y, z, r in base_link_spheres:
        spheres.append([x, y, z - grasp_z, r])
    for x, y, z, r in finger_spheres:
        spheres.append([x + left_finger_offset[0], y + left_finger_offset[1], z + left_finger_offset[2] - grasp_z, r])
        spheres.append(
            [x + right_finger_offset[0], -y + right_finger_offset[1], z + right_finger_offset[2] - grasp_z, r]
        )
    return tensor_args.to_device(spheres)


def load_xarm7_rerun(load_mesh: bool = True) -> RerunRobot:
    robot_cfg = xarm7_curobo_cfg()["robot_cfg"]
    urdf_rel_path = robot_cfg["kinematics"]["urdf_path"]
    external_asset_path = robot_cfg["kinematics"].get("external_asset_path", "")
    urdf_path = os.path.join(external_asset_path, urdf_rel_path) or join_path(get_assets_path(), urdf_rel_path)

    def _locate_asset(fname: str) -> str:
        if fname.startswith("package://"):
            return os.path.join(get_assets_path(), fname.replace("package://", ""))
        return os.path.join(os.path.dirname(urdf_path), fname)

    urdf = URDF.load(urdf_path, filename_handler=_locate_asset)
    return RerunRobot("xarm7", urdf, q_neutral=(*xarm7_home, xarm7_gripper_open), load_mesh=load_mesh)
