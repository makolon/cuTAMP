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


def load_mini_kitchen_env() -> TAMPEnvironment:
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
        dims=[cabinet_outer_dims[0], cabinet_wall, cabinet_outer_dims[2]],
        pose=[cabinet_center[0], cabinet_center[1] + cabinet_outer_dims[1] / 2 - cabinet_wall / 2, cabinet_center[2], *unit_quat],
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
        dims=[0.17, 0.21, 0.012],
        pose=[cabinet_center[0], cabinet_center[1], 0.038, *unit_quat],
        color=interior,
    )
    cabinet_door_closed_pose = [0.839, -0.420, 0.118, *unit_quat]
    cabinet_door_open_pose = [0.786, -0.419, 0.118, *unit_quat]
    cabinet_door = Cuboid(
        name="cabinet_door",
        dims=[0.22, 0.02, 0.26],
        pose=list(cabinet_door_closed_pose),
        color=cabinet_beige,
    )
    cabinet_handle = Cuboid(
        name="cabinet_handle",
        dims=[0.10, 0.20, 0.08],
        pose=[0.823, -0.424, 0.152, *unit_quat],
        color=metal,
    )

    drawer_center = [0.836, 0.139, 0.378]
    drawer_outer_dims = [0.18, 0.24, 0.08]
    drawer_wall = 0.02
    drawer_depth_clear = drawer_outer_dims[1] - 2 * drawer_wall
    drawer_proxy = Cuboid(
        name="drawer",
        dims=[drawer_outer_dims[0], drawer_wall, drawer_outer_dims[2]],
        pose=[drawer_center[0], drawer_center[1] + drawer_outer_dims[1] / 2 - drawer_wall / 2, drawer_center[2], *unit_quat],
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
        pose=[0.926, -0.449, cabinet_floor_z + 0.045 + z_buffer, *unit_quat],
        color=[63, 102, 136],
    )
    bowl = Cuboid(
        name="bowl",
        dims=[0.10, 0.10, 0.05],
        pose=[0.952, -0.396, cabinet_floor_z + 0.025 + z_buffer, *unit_quat],
        color=[170, 198, 255],
    )
    plate = Cuboid(
        name="plate",
        dims=[0.14, 0.14, 0.025],
        pose=[0.966, -0.344, cabinet_floor_z + 0.0125 + z_buffer, *unit_quat],
        color=[246, 246, 243],
    )
    spoon = Cuboid(
        name="spoon",
        dims=[0.12, 0.025, 0.02],
        pose=[0.804, 0.150, drawer_floor_z + 0.01 + z_buffer, *unit_quat],
        color=[178, 178, 178],
    )
    fork = Cuboid(
        name="fork",
        dims=[0.12, 0.025, 0.02],
        pose=[0.854, 0.118, drawer_floor_z + 0.01 + z_buffer, *unit_quat],
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
            "big_table": {"urdf": "big_table.urdf", "sync_pose_from_env": True},
            "table": {"urdf": "table.urdf", "sync_pose_from_env": True},
            "counter_scene": {
                "urdf": "kitchen_worlds_counter/counter/urdf/kitchen_part_right_gen_convex.urdf",
                "sync_pose_from_env": False,
                "base_pose": [
                    COUNTER_SCENE_BASE_XY[0],
                    COUNTER_SCENE_BASE_XY[1],
                    COUNTER_SCENE_BASE_Z,
                    *_yaw_quat(COUNTER_SCENE_YAW_DEG),
                ],
                "global_scaling": COUNTER_SCENE_SCALE,
            },
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
                    "body_name": "counter_scene",
                    "joint_name": "indigo_door_left_joint",
                    "closed_value": 0.0,
                    "open_value": -1.15,
                },
                {
                    "body_name": "counter_scene",
                    "joint_name": "indigo_door_right_joint",
                    "closed_value": 0.0,
                    "open_value": 1.15,
                },
            ],
            "drawer": [
                {
                    "body_name": "counter_scene",
                    "joint_name": "hitman_drawer_top_joint",
                    "closed_value": 0.0,
                    "open_value": 0.18,
                }
            ],
        },
        "cameras": [
            {
                "name": "overview",
                "eye_position": [0.22, 0.72, 0.94],
                "target_position": [0.90, -0.10, 0.18],
                "up_vector": [0.0, 0.0, 1.0],
                "fov": 42.0,
                "near": 0.01,
                "far": 4.0,
            },
            {
                "name": "workspace",
                "eye_position": [0.44, 0.18, 0.50],
                "target_position": [0.90, -0.24, 0.15],
                "up_vector": [0.0, 0.0, 1.0],
                "fov": 34.0,
                "near": 0.01,
                "far": 3.0,
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
            "big_table",
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
        big_table,
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

    return TAMPEnvironment(
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
