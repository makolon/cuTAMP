"""Scene construction utilities for PyBullet RGB rendering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pybullet as pb
import torch
from roma import quat_wxyz_to_xyzw

from cutamp.envs import TAMPEnvironment


@dataclass
class PyBulletScene:
    client_id: int
    body_ids: dict[str, int]
    render_config: dict[str, Any]
    asset_specs: dict[str, dict[str, Any]]
    openable_joints: dict[str, dict[str, Any]]
    asset_root: Path


def _pose_to_pybullet(pose: list[float]) -> tuple[list[float], list[float]]:
    position = list(pose[:3])
    orientation = quat_wxyz_to_xyzw(torch.tensor(pose[3:], dtype=torch.float32)).tolist()
    return position, orientation


def _resolve_asset_root(render_config: dict[str, Any]) -> Path:
    bundle_name = str(render_config.get("asset_bundle", "mini_kitchen"))
    return Path(__file__).resolve().parent / "assets" / bundle_name


def _get_pose_for_asset(name: str, env_objects: dict[str, Any], asset_spec: dict[str, Any]) -> list[float]:
    if bool(asset_spec.get("sync_pose_from_env", True)):
        if name not in env_objects:
            raise ValueError(f"PyBullet asset '{name}' is configured to sync from env, but no matching object exists")
        return list(env_objects[name].pose)
    if "base_pose" not in asset_spec:
        raise ValueError(f"PyBullet asset '{name}' must define base_pose when sync_pose_from_env is false")
    return list(asset_spec["base_pose"])


def _load_asset_body(
    client_id: int,
    asset_root: Path,
    name: str,
    asset_spec: dict[str, Any],
    pose: list[float],
) -> int:
    urdf_path = asset_root / str(asset_spec["urdf"])
    if not urdf_path.exists():
        raise FileNotFoundError(f"PyBullet asset for '{name}' not found: {urdf_path}")
    position, orientation = _pose_to_pybullet(pose)
    return pb.loadURDF(
        str(urdf_path),
        basePosition=position,
        baseOrientation=orientation,
        globalScaling=float(asset_spec.get("global_scaling", 1.0)),
        useFixedBase=bool(asset_spec.get("fixed_base", True)),
        physicsClientId=client_id,
    )


def _resolve_joint_index(client_id: int, body_id: int, joint_name: str) -> int:
    num_joints = pb.getNumJoints(body_id, physicsClientId=client_id)
    for joint_index in range(num_joints):
        info = pb.getJointInfo(body_id, joint_index, physicsClientId=client_id)
        if info[1].decode("utf-8") == joint_name:
            return joint_index
    raise ValueError(f"Joint '{joint_name}' not found in body {body_id}")


def _normalize_openable_joint_specs(joint_spec: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(joint_spec, list):
        return [dict(spec) for spec in joint_spec]
    return [dict(joint_spec)]


def build_pybullet_scene(env: TAMPEnvironment) -> PyBulletScene:
    if env.name != "mini_kitchen":
        raise ValueError(f"PyBullet RGB rendering only supports mini_kitchen, not {env.name}")

    render_config = dict(env.metadata.get("pybullet_render", {}))
    asset_specs = {
        str(name): dict(spec) for name, spec in dict(render_config.get("object_assets", {})).items()
    }
    if not asset_specs:
        raise ValueError("mini_kitchen PyBullet rendering requires object_assets metadata")

    env_objects = {obj.name: obj for obj in env.statics + env.movables}
    asset_root = _resolve_asset_root(render_config)
    if not asset_root.exists():
        raise FileNotFoundError(f"PyBullet asset bundle not found: {asset_root}")

    client_id = pb.connect(pb.DIRECT)
    pb.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)

    body_ids: dict[str, int] = {}
    for name, asset_spec in asset_specs.items():
        pose = _get_pose_for_asset(name, env_objects, asset_spec)
        body_ids[name] = _load_asset_body(client_id, asset_root, name, asset_spec, pose)

    openable_joints: dict[str, dict[str, Any]] = {}
    for openable_name, joint_spec in dict(render_config.get("openable_joints", {})).items():
        joint_entries = []
        for joint_entry in _normalize_openable_joint_specs(joint_spec):
            body_name = str(joint_entry.get("body_name", openable_name))
            if body_name not in body_ids:
                raise ValueError(f"Openable '{openable_name}' refers to unknown body '{body_name}'")
            joint_name = str(joint_entry["joint_name"])
            joint_index = _resolve_joint_index(client_id, body_ids[body_name], joint_name)
            joint_entries.append(
                {
                    "body_name": body_name,
                    "joint_name": joint_name,
                    "joint_index": joint_index,
                    "closed_value": float(joint_entry["closed_value"]),
                    "open_value": float(joint_entry["open_value"]),
                }
            )
        openable_joints[str(openable_name)] = {"joints": joint_entries}

    return PyBulletScene(
        client_id=client_id,
        body_ids=body_ids,
        render_config=render_config,
        asset_specs=asset_specs,
        openable_joints=openable_joints,
        asset_root=asset_root,
    )


def disconnect_pybullet_scene(scene: PyBulletScene) -> None:
    pb.disconnect(physicsClientId=scene.client_id)
