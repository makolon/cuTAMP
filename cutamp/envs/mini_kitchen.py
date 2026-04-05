# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

import math
from typing import Iterable

from curobo.geom.types import Cuboid

from cutamp.envs.utils import TAMPEnvironment, unit_quat
from cutamp.tamp_domain import CanOpen, HandEmpty, In, On


COUNTER_SCENE_SCALE = 0.3423306212380528
COUNTER_SCENE_YAW_DEG = -166.70452839355173
COUNTER_SCENE_BASE_XY = [0.8633317843990503, 0.011493352569675097]
COUNTER_SCENE_BASE_Z = 0.495


def _panel(name: str, dims: list[float], pose: list[float], color: list[int]) -> Cuboid:
    return Cuboid(name=name, dims=dims, pose=pose, color=color)


def _yaw_quat(yaw_deg: float) -> list[float]:
    yaw_rad = math.radians(yaw_deg)
    return [math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0)]


def _aabb_to_pose_dims(aabb: tuple[tuple[float, float, float], tuple[float, float, float]]) -> tuple[list[float], list[float]]:
    lower, upper = aabb
    dims = [max(float(upper[i] - lower[i]), 1e-4) for i in range(3)]
    pose = [float((lower[i] + upper[i]) * 0.5) for i in range(3)] + list(unit_quat)
    return pose, dims


def _align_env_with_pybullet_kitchen_world(env: TAMPEnvironment) -> TAMPEnvironment:
    """Align cuboid planning geometry to PyBullet Kitchen World AABBs for tighter environment consistency."""
    import pybullet as pb

    from cutamp.sim.pybullet_scene import build_pybullet_scene, disconnect_pybullet_scene

    scene = build_pybullet_scene(env)
    name_to_obj = {obj.name: obj for obj in env.movables + env.statics}

    # Align all assets that have 1:1 names between TAMP objects and PyBullet bodies.
    for asset_name in scene.asset_specs:
        if asset_name not in name_to_obj:
            continue
        body_id = scene.body_ids[asset_name]
        aabb = pb.getAABB(body_id, -1, physicsClientId=scene.client_id)
        pose, dims = _aabb_to_pose_dims(aabb)
        obj = name_to_obj[asset_name]
        obj.pose = pose
        obj.dims = dims

    # Align openable panels from their articulated joint-link AABBs.
    for openable_name, joint_group in scene.openable_joints.items():
        openable_meta = env.metadata.get("openables", {}).get(openable_name)
        if openable_meta is None:
            continue

        joint_aabbs = []
        for joint_spec in joint_group["joints"]:
            body_id = scene.body_ids[joint_spec["body_name"]]
            link_aabb = pb.getAABB(body_id, joint_spec["joint_index"], physicsClientId=scene.client_id)
            joint_aabbs.append(link_aabb)
        if not joint_aabbs:
            continue

        lower = [min(aabb[0][axis] for aabb in joint_aabbs) for axis in range(3)]
        upper = [max(aabb[1][axis] for aabb in joint_aabbs) for axis in range(3)]
        panel_pose, panel_dims = _aabb_to_pose_dims((tuple(lower), tuple(upper)))

        panel_name = openable_meta.get("panel")
        if panel_name in name_to_obj:
            panel_obj = name_to_obj[panel_name]
            panel_obj.pose = panel_pose
            panel_obj.dims = panel_dims
            openable_meta["closed_pose"] = list(panel_pose)

        # Derive a small handle region on the panel front; this is used by ValidOpen reachability checks.
        handle_name = openable_meta.get("handle")
        if handle_name in name_to_obj:
            handle_obj = name_to_obj[handle_name]
            handle_dims = list(handle_obj.dims)
            thin_axis = min(range(3), key=lambda idx: panel_dims[idx])
            handle_pose = list(panel_pose)
            handle_pose[thin_axis] = upper[thin_axis] + 0.5 * handle_dims[thin_axis]
            handle_obj.pose = handle_pose

    disconnect_pybullet_scene(scene)
    return env


def load_mini_kitchen_env(*, use_pybullet_layout: bool = False) -> TAMPEnvironment:
    """A compact articulated-kitchen environment for long-horizon VLM-TAMP experiments."""
    wood_dark = [185, 145, 102]
    wood_light = [227, 196, 152]
    cabinet_beige = [226, 221, 212]
    cabinet_shadow = [207, 200, 188]
    interior = [247, 245, 239]
    metal = [137, 143, 151]

    big_table = Cuboid(
        name="big_table",
        dims=[1.52, 1.42, 0.02],
        pose=[0.76, -0.04, -0.011, *unit_quat],
        color=wood_dark,
    )
    table = Cuboid(
        name="table",
        dims=[0.36, 0.48, 0.04],
        pose=[0.10, 0.62, 0.292, *unit_quat],
        color=wood_light,
    )
    counter = Cuboid(
        name="counter",
        dims=[0.22, 0.28, 0.02],
        pose=[0.942, -0.409, 0.300, *unit_quat],
        color=[204, 214, 194],
    )

    cabinet_center = [0.946, -0.395, 0.164]
    cabinet_outer_dims = [0.21, 0.26, 0.26]
    cabinet_wall = 0.02
    cabinet_depth_clear = cabinet_outer_dims[1] - 2 * cabinet_wall
    cabinet_proxy = Cuboid(
        name="cabinet",
        dims=[0.01, 0.01, 0.01],
        pose=[cabinet_center[0], cabinet_center[1], -0.30, *unit_quat],
        color=cabinet_shadow,
    )
    cabinet_left = _panel(
        "cabinet_shell_left",
        [cabinet_wall, cabinet_depth_clear, cabinet_outer_dims[2]],
        [cabinet_center[0] - cabinet_outer_dims[0] / 2 + cabinet_wall / 2, cabinet_center[1], cabinet_center[2], *unit_quat],
        cabinet_shadow,
    )
    cabinet_right = _panel(
        "cabinet_shell_right",
        [cabinet_wall, cabinet_depth_clear, cabinet_outer_dims[2]],
        [cabinet_center[0] + cabinet_outer_dims[0] / 2 - cabinet_wall / 2, cabinet_center[1], cabinet_center[2], *unit_quat],
        cabinet_shadow,
    )
    cabinet_bottom = _panel(
        "cabinet_shell_bottom",
        [cabinet_outer_dims[0], cabinet_depth_clear, cabinet_wall],
        [cabinet_center[0], cabinet_center[1], cabinet_center[2] - cabinet_outer_dims[2] / 2 + cabinet_wall / 2, *unit_quat],
        cabinet_shadow,
    )
    cabinet_top = _panel(
        "cabinet_shell_top",
        [cabinet_outer_dims[0], cabinet_depth_clear, cabinet_wall],
        [cabinet_center[0], cabinet_center[1], cabinet_center[2] + cabinet_outer_dims[2] / 2 - cabinet_wall / 2, *unit_quat],
        cabinet_shadow,
    )
    cabinet_interior = Cuboid(
        name="cabinet_interior",
        dims=[0.18, 0.23, 0.014],
        pose=[cabinet_center[0], cabinet_center[1], 0.061, *unit_quat],
        color=interior,
    )
    cabinet_door_closed_pose = [0.946, -0.250, 0.164, *unit_quat]
    cabinet_door_open_pose = [0.850, -0.395, 0.164, *unit_quat]
    cabinet_door = Cuboid(
        name="cabinet_door",
        dims=[0.24, 0.016, 0.26],
        pose=list(cabinet_door_closed_pose),
        color=cabinet_beige,
    )
    cabinet_handle = Cuboid(
        name="cabinet_handle",
        dims=[0.02, 0.016, 0.04],
        pose=[1.000, -0.236, 0.152, *unit_quat],
        color=metal,
    )

    drawer_center = [0.836, 0.139, 0.378]
    drawer_outer_dims = [0.18, 0.24, 0.08]
    drawer_wall = 0.02
    drawer_depth_clear = drawer_outer_dims[1] - 2 * drawer_wall
    drawer_proxy = Cuboid(
        name="drawer",
        dims=[0.01, 0.01, 0.01],
        pose=[drawer_center[0], drawer_center[1], -0.30, *unit_quat],
        color=cabinet_shadow,
    )
    drawer_left = _panel(
        "drawer_shell_left",
        [drawer_wall, drawer_depth_clear, drawer_outer_dims[2]],
        [drawer_center[0] - drawer_outer_dims[0] / 2 + drawer_wall / 2, drawer_center[1], drawer_center[2], *unit_quat],
        cabinet_shadow,
    )
    drawer_right = _panel(
        "drawer_shell_right",
        [drawer_wall, drawer_depth_clear, drawer_outer_dims[2]],
        [drawer_center[0] + drawer_outer_dims[0] / 2 - drawer_wall / 2, drawer_center[1], drawer_center[2], *unit_quat],
        cabinet_shadow,
    )
    drawer_bottom = _panel(
        "drawer_shell_bottom",
        [drawer_outer_dims[0], drawer_depth_clear, drawer_wall],
        [drawer_center[0], drawer_center[1], drawer_center[2] - drawer_outer_dims[2] / 2 + drawer_wall / 2, *unit_quat],
        cabinet_shadow,
    )
    drawer_interior = Cuboid(
        name="drawer_interior",
        dims=[0.15, 0.20, 0.015],
        pose=[drawer_center[0], drawer_center[1], 0.364, *unit_quat],
        color=interior,
    )
    drawer_front_closed_pose = [0.747, 0.118, 0.414, *unit_quat]
    drawer_front_open_pose = [0.687, 0.104, 0.414, *unit_quat]
    drawer_front = Cuboid(
        name="drawer_front",
        dims=[0.18, 0.02, 0.08],
        pose=list(drawer_front_closed_pose),
        color=cabinet_beige,
    )
    drawer_handle = Cuboid(
        name="drawer_handle",
        dims=[0.10, 0.05, 0.08],
        pose=[0.733, 0.115, 0.467, *unit_quat],
        color=metal,
    )

    z_buffer = 0.008
    cabinet_floor_z = cabinet_interior.pose[2] + cabinet_interior.dims[2] / 2
    drawer_floor_z = drawer_interior.pose[2] + drawer_interior.dims[2] / 2
    mug = Cuboid(
        name="mug",
        dims=[0.055, 0.055, 0.09],
        pose=[0.990, -0.385, cabinet_floor_z + 0.045 + z_buffer, *unit_quat],
        color=[63, 102, 136],
    )
    bowl = Cuboid(
        name="bowl",
        dims=[0.09, 0.09, 0.05],
        pose=[0.912, -0.390, cabinet_floor_z + 0.025 + z_buffer, *unit_quat],
        color=[170, 198, 255],
    )
    plate = Cuboid(
        name="plate",
        dims=[0.11, 0.11, 0.022],
        pose=[0.950, -0.495, cabinet_floor_z + 0.011 + z_buffer, *unit_quat],
        color=[246, 246, 243],
    )
    spoon = Cuboid(
        name="spoon",
        dims=[0.12, 0.025, 0.02],
        # Keep the long axis object fully inside the drawer interior AABB at reset.
        pose=[0.826, 0.188, drawer_floor_z + 0.01 + z_buffer, *unit_quat],
        color=[178, 178, 178],
    )
    fork = Cuboid(
        name="fork",
        dims=[0.12, 0.025, 0.02],
        # Keep clear of the closed drawer front panel near y ~= 0.118.
        pose=[0.846, 0.206, drawer_floor_z + 0.01 + z_buffer, *unit_quat],
        color=[162, 162, 162],
    )
    movables = [mug, bowl, plate, spoon, fork]

    initial_atoms = frozenset(
        {
            In.ground("mug", "cabinet"),
            In.ground("bowl", "cabinet"),
            In.ground("plate", "cabinet"),
            In.ground("spoon", "drawer"),
            In.ground("fork", "drawer"),
            CanOpen.ground("cabinet"),
            CanOpen.ground("drawer"),
        }
    )
    goal_state = frozenset({On.ground("mug", "counter"), HandEmpty.ground()})

    pybullet_render = {
        "asset_bundle": "mini_kitchen",
        "asset_manifest": "sources.json",
        "image_size": [704, 512],
        "background_color": [241, 243, 246],
        "object_assets": {
            "table": {"urdf": "table.urdf", "sync_pose_from_env": True},
            "counter": {"urdf": "counter.urdf", "sync_pose_from_env": True},
            "cabinet": {"urdf": "cabinet_unit.urdf", "sync_pose_from_env": True},
            "drawer": {"urdf": "drawer_unit.urdf", "sync_pose_from_env": True},
            "mug": {
                "urdf": "kitchen_models/Drinks/GreyMug/cuTAMP_grey_mug.urdf",
                "sync_pose_from_env": True,
                "global_scaling": 0.68,
            },
            "bowl": {
                "urdf": "kitchen_models/Bowl/0000/cuTAMP_bowl.urdf",
                "sync_pose_from_env": True,
                "global_scaling": 0.63,
            },
            "plate": {
                "urdf": "kitchen_models/Plate/SquarePlate/cuTAMP_square_plate.urdf",
                "sync_pose_from_env": True,
                "global_scaling": 0.0095,
            },
            "spoon": {"urdf": "spoon.urdf", "sync_pose_from_env": True},
            "fork": {"urdf": "fork.urdf", "sync_pose_from_env": True},
        },
        "openable_joints": {
            "cabinet": [
                {
                    "body_name": "cabinet",
                    "joint_name": "door_hinge",
                    "closed_value": 0.0,
                    "open_value": 1.15,
                },
            ],
            "drawer": [
                {
                    "body_name": "drawer",
                    "joint_name": "drawer_slide",
                    "closed_value": 0.0,
                    "open_value": 0.18,
                }
            ],
        },
        "cameras": [
            {
                "name": "front",
                "eye_position": [0.86, 0.62, 0.48],
                "target_position": [0.86, -0.12, 0.20],
                "up_vector": [0.0, 0.0, 1.0],
                "fov": 38.0,
                "near": 0.01,
                "far": 4.0,
            },
        ],
    }

    metadata = {
        "openables": {
            "cabinet": {
                "panel": "cabinet_door",
                "handle": "cabinet_handle",
                "interior": "cabinet_interior",
                "closed_pose": list(cabinet_door_closed_pose),
                "open_pose": list(cabinet_door_open_pose),
                "is_open": False,
            },
            "drawer": {
                "panel": "drawer_front",
                "handle": "drawer_handle",
                "interior": "drawer_interior",
                "closed_pose": list(drawer_front_closed_pose),
                "open_pose": list(drawer_front_open_pose),
                "is_open": False,
            },
        },
        "task_suite": {
            "set_breakfast": "open the cabinet, take out the mug and bowl, and place them on the counter",
            "clear_drawer": "open the drawer, take out the spoon and fork, and place them on the table",
            "set_table": "open the cabinet and drawer, then place the plate, mug, and spoon on the table",
        },
        "vlm_frame_exclude": [
            "cabinet",
            "drawer",
            "cabinet_shell_left",
            "cabinet_shell_right",
            "cabinet_shell_bottom",
            "cabinet_shell_top",
            "drawer_shell_left",
            "drawer_shell_right",
            "drawer_shell_bottom",
        ],
        "pybullet_render": pybullet_render,
    }

    statics = [
        table,
        counter,
        cabinet_proxy,
        cabinet_left,
        cabinet_right,
        cabinet_bottom,
        cabinet_top,
        cabinet_interior,
        cabinet_door,
        cabinet_handle,
        drawer_proxy,
        drawer_left,
        drawer_right,
        drawer_bottom,
        drawer_interior,
        drawer_front,
        drawer_handle,
    ]

    env = TAMPEnvironment(
        name="mini_kitchen",
        movables=movables,
        statics=statics,
        type_to_objects={
            "Movable": movables,
            "Surface": [table, counter, cabinet_interior, drawer_interior],
            "Container": [cabinet_proxy, drawer_proxy],
            "Openable": [cabinet_proxy, drawer_proxy],
        },
        goal_state=goal_state,
        initial_atoms=initial_atoms,
        metadata=metadata,
    )
    if use_pybullet_layout:
        env = _align_env_with_pybullet_kitchen_world(env)
    return env
