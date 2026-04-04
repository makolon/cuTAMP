"""PyBullet RGB rendering for VLM query images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as pb
from PIL import Image

from cutamp.envs import TAMPEnvironment
from cutamp.sim.pybullet_scene import build_pybullet_scene, disconnect_pybullet_scene
from cutamp.sim.pybullet_sync import sync_pybullet_scene_from_env


def _render_camera(scene, camera_spec: dict[str, Any], image_size: tuple[int, int]) -> Image.Image:
    width, height = image_size
    if "eye_position" in camera_spec:
        view_matrix = pb.computeViewMatrix(
            cameraEyePosition=list(camera_spec["eye_position"]),
            cameraTargetPosition=list(camera_spec["target_position"]),
            cameraUpVector=list(camera_spec.get("up_vector", [0.0, 0.0, 1.0])),
        )
    else:
        view_matrix = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(camera_spec["target_position"]),
            distance=float(camera_spec["distance"]),
            yaw=float(camera_spec["yaw"]),
            pitch=float(camera_spec["pitch"]),
            roll=float(camera_spec.get("roll", 0.0)),
            upAxisIndex=2,
        )
    projection_matrix = pb.computeProjectionMatrixFOV(
        fov=float(camera_spec["fov"]),
        aspect=float(width) / float(height),
        nearVal=float(camera_spec.get("near", 0.01)),
        farVal=float(camera_spec.get("far", 5.0)),
    )
    _, _, rgba, _, _ = pb.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=pb.ER_TINY_RENDERER,
        shadow=1,
        physicsClientId=scene.client_id,
    )
    rgba_array = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
    return Image.fromarray(rgba_array[..., :3], mode="RGB")


def render_pybullet_query_image(
    env: TAMPEnvironment,
    path: Path,
    open_goal: str | None = None,
    scene_text: str | None = None,
) -> Path:
    if env.name != "mini_kitchen":
        raise ValueError(f"PyBullet RGB rendering only supports mini_kitchen, not {env.name}")

    scene = build_pybullet_scene(env)
    sync_pybullet_scene_from_env(scene, env)

    render_config = scene.render_config
    image_size = tuple(render_config.get("image_size", [640, 480]))
    background_color = tuple(render_config.get("background_color", [246, 247, 249]))
    camera_specs = list(render_config["cameras"])

    view_images = [_render_camera(scene, camera_spec, image_size) for camera_spec in camera_specs]
    mosaic_width = image_size[0] * len(view_images)
    mosaic_height = image_size[1]
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height), background_color)
    for index, view_image in enumerate(view_images):
        mosaic.paste(view_image, (index * image_size[0], 0))

    path.parent.mkdir(parents=True, exist_ok=True)
    for index, view_image in enumerate(view_images):
        view_image.save(path.with_name(f"{path.stem}_view{index}.png"))
    with open(path.with_name(f"{path.stem}_camera_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "env_name": env.name,
                "asset_bundle": render_config.get("asset_bundle"),
                "asset_manifest": render_config.get("asset_manifest"),
                "image_size": list(image_size),
                "background_color": list(background_color),
                "open_goal": open_goal,
                "scene_text_preview": scene_text.splitlines()[:8] if scene_text else [],
                "cameras": camera_specs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    mosaic.save(path)
    disconnect_pybullet_scene(scene)
    return path
