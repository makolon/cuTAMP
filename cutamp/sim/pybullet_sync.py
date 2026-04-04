"""State synchronization between TAMPEnvironment and a PyBullet render scene."""

from __future__ import annotations

import pybullet as pb
import torch
from roma import quat_wxyz_to_xyzw

from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import get_openable_state
from cutamp.sim.pybullet_scene import PyBulletScene


def _pose_to_pybullet(pose: list[float]) -> tuple[list[float], list[float]]:
    position = list(pose[:3])
    orientation = quat_wxyz_to_xyzw(torch.tensor(pose[3:], dtype=torch.float32)).tolist()
    return position, orientation


def sync_pybullet_scene_from_env(scene: PyBulletScene, env: TAMPEnvironment) -> None:
    env_objects = {obj.name: obj for obj in env.statics + env.movables if hasattr(obj, "pose")}

    for name, asset_spec in scene.asset_specs.items():
        if not bool(asset_spec.get("sync_pose_from_env", True)):
            continue
        if name not in env_objects:
            continue
        position, orientation = _pose_to_pybullet(list(env_objects[name].pose))
        pb.resetBasePositionAndOrientation(
            scene.body_ids[name],
            position,
            orientation,
            physicsClientId=scene.client_id,
        )

    for openable_name, joint_group in scene.openable_joints.items():
        is_open = get_openable_state(env, openable_name)
        for joint_spec in joint_group["joints"]:
            target_value = joint_spec["open_value"] if is_open else joint_spec["closed_value"]
            pb.resetJointState(
                scene.body_ids[joint_spec["body_name"]],
                int(joint_spec["joint_index"]),
                targetValue=float(target_value),
                physicsClientId=scene.client_id,
            )
