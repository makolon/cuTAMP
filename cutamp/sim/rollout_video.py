from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as pb
import torch
from roma import quat_wxyz_to_xyzw

from cutamp.sim.pybullet_render import _render_camera
from cutamp.sim.pybullet_scene import build_pybullet_scene, disconnect_pybullet_scene
from cutamp.tamp_world import TAMPWorld
from cutamp.utils.common import mat4x4_to_pose_list


def _pose_list_to_pybullet(pose: list[float]) -> tuple[list[float], list[float]]:
	position = list(pose[:3])
	orientation = quat_wxyz_to_xyzw(torch.tensor(pose[3:], dtype=torch.float32)).tolist()
	return position, orientation


def export_rollout_mp4(
	world: TAMPWorld,
	rollout: dict,
	best_idx: int,
	output_path: str | Path,
	fps: int,
) -> bool:
	"""Export rollout frames to mp4 by rendering the world and robot collision spheres in PyBullet."""
	if world.env.name != "mini_kitchen":
		return False

	scene = build_pybullet_scene(world.env)
	render_config = scene.render_config
	image_size = tuple(render_config.get("image_size", [640, 480]))
	background_color = tuple(render_config.get("background_color", [246, 247, 249]))
	camera_specs = list(render_config["cameras"])

	robot_spheres_t0 = rollout["robot_spheres"][best_idx, 0].detach().cpu()
	robot_sphere_ids: list[int] = []
	for sphere in robot_spheres_t0:
		radius = max(float(sphere[3].item()), 1e-4)
		visual_shape = pb.createVisualShape(
			pb.GEOM_SPHERE,
			radius=radius,
			rgbaColor=[0.15, 0.33, 0.85, 0.75],
			physicsClientId=scene.client_id,
		)
		body_id = pb.createMultiBody(
			baseMass=0.0,
			baseCollisionShapeIndex=-1,
			baseVisualShapeIndex=visual_shape,
			basePosition=sphere[:3].tolist(),
			baseOrientation=[0.0, 0.0, 0.0, 1.0],
			physicsClientId=scene.client_id,
		)
		robot_sphere_ids.append(body_id)

	out_path = Path(output_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", format="FFMPEG")

	num_steps = len(rollout["conf_params"])
	for ts in range(num_steps):
		pose_ts_val = rollout["ts_to_pose_ts"][ts]
		if isinstance(pose_ts_val, torch.Tensor):
			pose_ts = int(pose_ts_val.item())
		else:
			pose_ts = int(pose_ts_val)

		for obj in world.movables:
			mat4x4 = rollout["obj_to_pose"][obj.name][best_idx, pose_ts].detach().cpu()
			pose = mat4x4_to_pose_list(mat4x4)
			position, orientation = _pose_list_to_pybullet(pose)
			pb.resetBasePositionAndOrientation(
				scene.body_ids[obj.name],
				position,
				orientation,
				physicsClientId=scene.client_id,
			)

		robot_spheres = rollout["robot_spheres"][best_idx, ts].detach().cpu()
		for sphere_id, sphere in zip(robot_sphere_ids, robot_spheres):
			pb.resetBasePositionAndOrientation(
				sphere_id,
				sphere[:3].tolist(),
				[0.0, 0.0, 0.0, 1.0],
				physicsClientId=scene.client_id,
			)

		view_images = [_render_camera(scene, camera_spec, image_size) for camera_spec in camera_specs]
		mosaic_width = image_size[0] * len(view_images)
		mosaic_height = image_size[1]
		mosaic = np.full((mosaic_height, mosaic_width, 3), background_color, dtype=np.uint8)
		for index, view_image in enumerate(view_images):
			arr = np.asarray(view_image, dtype=np.uint8)
			x0 = index * image_size[0]
			mosaic[:, x0 : x0 + image_size[0], :] = arr

		writer.append_data(mosaic)

	writer.close()
	disconnect_pybullet_scene(scene)
	return out_path.exists()
