# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import warnings
from typing import Any, Dict, List, ClassVar, Iterable, Sequence, Set, Tuple

import torch
import yaml
from curobo.geom.types import Cuboid, Obstacle, Cylinder, Mesh

from cutamp.tamp_domain import all_tamp_fluents
from cutamp.utils.shapes import MultiSphere
from cutamp.task_planning.base_structs import State

unit_quat = [1.0, 0.0, 0.0, 0.0]
unit_pose = [0.0, 0.0, 0.0, *unit_quat]


class TAMPEnvironment:
    _known_types: ClassVar[Set[str]] = {"Movable", "Surface", "Button", "Stick", "Container", "Openable"}

    def __init__(
        self,
        name: str,
        movables: List[Obstacle],
        statics: List[Obstacle],
        type_to_objects: Dict[str, List[Obstacle]],
        goal_state: State,
        initial_atoms: State = frozenset(),
        metadata: Dict[str, Any] | None = None,
    ):
        self.name = name
        self.movables = movables
        self.statics = statics
        self.type_to_objects = type_to_objects
        self.goal_state = goal_state
        self.initial_atoms = initial_atoms
        self.metadata = metadata.copy() if metadata is not None else {}

        # No object (identified by name) should be in both movables and statics
        movable_names = {obj.name for obj in movables}
        static_names = {obj.name for obj in statics}
        intersection = movable_names.intersection(static_names)
        if intersection:
            raise ValueError(f"Objects cannot be both movable and static: {intersection}")

        # Make sure types are all known
        for obj_type in type_to_objects:
            if obj_type not in self._known_types:
                raise ValueError(f"Unknown object type: {obj_type}")

    def __str__(self):
        accum = [f"Environment: {self.name}", "\tMovables:"]
        # Movables
        for obj in self.movables:
            accum.append(f"\t\t{obj.name}")
        # Statics
        accum.append("\tStatics:")
        for obj in self.statics:
            accum.append(f"\t\t{obj.name}")
        # Types
        accum.append("\tTypes:")
        for obj_type, objects in self.type_to_objects.items():
            accum.append(f"\t\t{obj_type}: {[obj.name for obj in objects]}")

        result = "\n".join(accum)
        return result


def _multi_sphere_factory(name: str, spheres: List[List[float]], **kwargs) -> MultiSphere:
    """Create a MultiSphere object from a list of spheres."""
    spheres_tensor = torch.tensor(spheres)
    if spheres_tensor.ndim != 2 or spheres_tensor.shape[1] != 4:
        raise ValueError(f"Spheres should be (n, 4) not {spheres_tensor.shape}")
    return MultiSphere(name=name, spheres=spheres_tensor, **kwargs)


_key_to_factory = {
    "cuboid": Cuboid,
    "cylinder": Cylinder,
    "mesh": Mesh,
    "multi_sphere": _multi_sphere_factory,
}


def load_env(env_path: str) -> TAMPEnvironment:
    """Load the environment from a YAML file and parse into movable and static objects."""
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    with open(env_path, "r") as f:
        env_dict = yaml.load(f, Loader=yaml.SafeLoader)

    return load_env_from_dict(env_dict)


def load_env_from_dict(env_dict: dict) -> TAMPEnvironment:
    """Load the environment from a dictionary representation."""

    # Make sure expected keys are present
    for key in ["name", "geometries", "types"]:
        if key not in env_dict:
            raise ValueError(f"Key {key} not found in environment file")

    # Load the geometries
    name_to_obj = {}
    for key, obj_dicts in env_dict["geometries"].items():
        if key not in _key_to_factory:
            raise ValueError(f"Unknown shape type: {key}")
        cls = _key_to_factory[key]
        for name, obj_dict in obj_dicts.items():
            if name in name_to_obj:
                raise ValueError(f"Cannot have duplicate object names: {name}")
            obj = cls(name=name, **obj_dict)
            name_to_obj[name] = obj

    # Parse the types of the objects. If not specified then just a static object
    type_to_objects = {}
    for obj_type, objects in env_dict["types"].items():
        for obj in objects:
            if obj not in name_to_obj:
                raise ValueError(f"Object {obj} not found in geometries")
        type_to_objects[obj_type] = [name_to_obj[obj] for obj in objects]

    # Determine movables and statics. There is a Movable type which we use to determine movables
    movable_objects = type_to_objects.get("Movable", [])
    movables, statics = [], []
    for obj in name_to_obj.values():
        if obj in movable_objects:
            movables.append(obj)
        else:
            statics.append(obj)

    # Parse goal states
    name_to_fluent = {fluent.name: fluent for fluent in all_tamp_fluents}
    goal_state = set()
    for atom_dict in env_dict["goal"]:
        if not (isinstance(atom_dict, dict) and len(atom_dict) == 1):
            raise ValueError(f"Goal atom should be a dict of length 1, not {atom_dict}")

        for fluent_name, values in atom_dict.items():
            if fluent_name not in name_to_fluent:
                raise ValueError(f"Unknown fluent: {fluent_name}")
            fluent = name_to_fluent[fluent_name]
            if len(values) != len(fluent.parameters):
                raise ValueError(f"Expected {len(fluent.parameters)} values for {fluent_name}, got {values}")
            goal_atom = fluent.ground(*values)
            goal_state.add(goal_atom)
    goal_state = frozenset(goal_state)

    initial_atoms = set()
    for atom_dict in env_dict.get("initial", []):
        if not (isinstance(atom_dict, dict) and len(atom_dict) == 1):
            raise ValueError(f"Initial atom should be a dict of length 1, not {atom_dict}")

        for fluent_name, values in atom_dict.items():
            if fluent_name not in name_to_fluent:
                raise ValueError(f"Unknown fluent: {fluent_name}")
            fluent = name_to_fluent[fluent_name]
            if len(values) != len(fluent.parameters):
                raise ValueError(f"Expected {len(fluent.parameters)} values for {fluent_name}, got {values}")
            initial_atoms.add(fluent.ground(*values))
    initial_atoms = frozenset(initial_atoms)

    env = TAMPEnvironment(
        name=env_dict["name"],
        movables=movables,
        statics=statics,
        type_to_objects=type_to_objects,
        goal_state=goal_state,
        initial_atoms=initial_atoms,
        metadata=env_dict.get("metadata", {}),
    )
    return env


def clone_tamp_environment(env: TAMPEnvironment) -> TAMPEnvironment:
    """Clone an environment through its serializable dictionary representation."""
    return load_env_from_dict(get_env_dict(env))


def replace_goal_state(env: TAMPEnvironment, goal_state: State) -> TAMPEnvironment:
    """Clone an environment and replace its goal state."""
    cloned = clone_tamp_environment(env)
    cloned.goal_state = goal_state
    return cloned


def set_object_pose(env: TAMPEnvironment, obj_name: str, pose: List[float]) -> None:
    """Set the pose of an object in-place."""
    for obj in env.movables + env.statics:
        if obj.name == obj_name:
            obj.pose = list(pose)
            return
    raise ValueError(f"Object {obj_name} not found in environment {env.name}")


def merge_tamp_environment_state(reference_env: TAMPEnvironment, state_env: TAMPEnvironment) -> TAMPEnvironment:
    """Clone the reference environment and overwrite object poses from another environment."""
    merged = clone_tamp_environment(reference_env)
    for obj in state_env.movables + state_env.statics:
        set_object_pose(merged, obj.name, obj.pose)
    _merge_openable_states(merged, state_env)
    return merged


def overlay_tamp_environment_states(template_env: TAMPEnvironment, state_envs: Sequence[TAMPEnvironment]) -> TAMPEnvironment:
    """Clone a template environment and sequentially apply poses from multiple state environments."""
    merged = clone_tamp_environment(template_env)
    for state_env in state_envs:
        for obj in state_env.movables + state_env.statics:
            set_object_pose(merged, obj.name, obj.pose)
        _merge_openable_states(merged, state_env)
    return merged


def reduce_tamp_environment(
    env: TAMPEnvironment, movable_names: Iterable[str], goal_state: State | None = None
) -> TAMPEnvironment:
    """
    Clone an environment while demoting non-relevant movable objects into static obstacles.
    This preserves geometry for collision checking while shrinking the symbolic movable set.
    """
    keep_movable_names = set(movable_names)
    reduced = clone_tamp_environment(env)

    new_movables = [obj for obj in reduced.movables if obj.name in keep_movable_names]
    demoted = [obj for obj in reduced.movables if obj.name not in keep_movable_names]
    reduced.movables = new_movables
    reduced.statics = [*reduced.statics, *demoted]

    new_type_to_objects = {}
    for obj_type, objects in reduced.type_to_objects.items():
        if obj_type in {"Movable", "Stick"}:
            filtered = [obj for obj in objects if obj.name in keep_movable_names]
        else:
            filtered = list(objects)
        if filtered or obj_type == "Movable":
            new_type_to_objects[obj_type] = filtered
    reduced.type_to_objects = new_type_to_objects

    filtered_initial_atoms = set()
    for atom in reduced.initial_atoms:
        keep_atom = True
        for param, value in zip(atom.fluent.parameters, atom.values):
            if param.type == "movable" and value not in keep_movable_names:
                keep_atom = False
                break
        if keep_atom:
            filtered_initial_atoms.add(atom)
    reduced.initial_atoms = frozenset(filtered_initial_atoms)

    if goal_state is not None:
        reduced.goal_state = goal_state
    return reduced


def _get_object_dict(obj: Obstacle) -> Tuple[str, dict]:
    """Get the object as a dictionary, so we can serialize."""
    if obj.scale is not None:
        raise NotImplementedError(f"{obj.name} should not have a scale")
    if obj.texture is not None:
        raise NotImplementedError(f"{obj.name} should not have a texture")
    if isinstance(obj, Cuboid):
        return "cuboid", {
            "dims": obj.dims,
            "pose": obj.pose,
            "color": obj.color,
        }
    elif isinstance(obj, Cylinder):
        return "cylinder", {
            "radius": obj.radius,
            "height": obj.height,
            "pose": obj.pose,
            "color": obj.color,
        }
    elif isinstance(obj, MultiSphere):
        spheres = obj.spheres.tolist()
        return "multi_sphere", {
            "spheres": spheres,
            "pose": obj.pose,
            "color": obj.color,
        }
    elif isinstance(obj, Mesh):
        warnings.warn(f"Serialization of mesh objects are not supported yet. Mesh name: {obj.name}")
        return "mesh", {}
    raise NotImplementedError(f"Unsupported object type: {type(obj)}")


def get_env_dict(env: TAMPEnvironment) -> dict:
    """Get the environment as a dictionary, so we can serialize."""
    geometries = {}
    for obj in env.movables + env.statics:
        geo_type, obj_dict = _get_object_dict(obj)
        if geo_type not in geometries:
            geometries[geo_type] = {}
        geometries[geo_type][obj.name] = obj_dict

    env_dict = {
        "name": env.name,
        "geometries": geometries,
        "types": {obj_type: [obj.name for obj in objs] for obj_type, objs in env.type_to_objects.items()},
        "goal": [{goal_atom.fluent.name: goal_atom.values} for goal_atom in env.goal_state],
        "initial": [{atom.fluent.name: atom.values} for atom in env.initial_atoms],
        "metadata": env.metadata,
    }
    return env_dict


def get_openable_state(env: TAMPEnvironment, openable: str) -> bool:
    """Return whether the named openable is currently open."""
    openables = env.metadata.get("openables", {})
    if openable not in openables:
        raise ValueError(f"Openable '{openable}' not found in environment metadata")
    return bool(openables[openable]["is_open"])


def get_container_interior_name(env: TAMPEnvironment, container: str) -> str:
    """Return the interior geometry name associated with the container."""
    openables = env.metadata.get("openables", {})
    if container not in openables:
        raise ValueError(f"Container '{container}' not found in environment metadata")
    return str(openables[container]["interior"])


def set_openable_state(env: TAMPEnvironment, openable: str, is_open: bool) -> None:
    """Set an openable state and update the corresponding panel geometry pose in-place."""
    openables = env.metadata.get("openables", {})
    if openable not in openables:
        raise ValueError(f"Openable '{openable}' not found in environment metadata")
    openable_info = openables[openable]
    pose_key = "open_pose" if is_open else "closed_pose"
    panel_name = openable_info["panel"]
    set_object_pose(env, panel_name, list(openable_info[pose_key]))
    openable_info["is_open"] = bool(is_open)


def _merge_openable_states(target_env: TAMPEnvironment, source_env: TAMPEnvironment) -> None:
    source_openables = source_env.metadata.get("openables", {})
    if not source_openables:
        return
    for openable, info in source_openables.items():
        if "openables" not in target_env.metadata or openable not in target_env.metadata["openables"]:
            continue
        set_openable_state(target_env, openable, bool(info["is_open"]))


def get_env_dir() -> str:
    """Get the directory containing the environment files."""
    return os.path.join(os.path.dirname(__file__), "assets")


def create_walls_for_cuboid(
    cuboid: Cuboid, wall_height: float, wall_thickness: float, wall_color: List[int]
) -> List[Cuboid]:
    if cuboid.pose[3:] != unit_quat:
        raise NotImplementedError("Only cuboids with unit quaternion are supported.")

    walls = [
        Cuboid(
            name=f"wall_1_{cuboid.name}",
            dims=[cuboid.dims[0], wall_thickness, wall_height],
            pose=[
                cuboid.pose[0],
                cuboid.pose[1] + cuboid.dims[1] / 2 + wall_thickness / 2,
                cuboid.pose[2] + wall_height / 2,
                *unit_quat,
            ],
            color=list(wall_color),
        ),
        Cuboid(
            name=f"wall_2_{cuboid.name}",
            dims=[cuboid.dims[0], 0.02, wall_height],
            pose=[
                cuboid.pose[0],
                cuboid.pose[1] - cuboid.dims[1] / 2 - wall_thickness / 2,
                cuboid.pose[2] + wall_height / 2,
                *unit_quat,
            ],
            color=list(wall_color),
        ),
        Cuboid(
            name=f"wall_3_{cuboid.name}",
            dims=[0.02, cuboid.dims[1], wall_height],
            pose=[
                cuboid.pose[0] - cuboid.dims[0] / 2 - wall_thickness / 2,
                cuboid.pose[1],
                cuboid.pose[2] + wall_height / 2,
                *unit_quat,
            ],
            color=list(wall_color),
        ),
        Cuboid(
            name=f"wall_4_{cuboid.name}",
            dims=[0.02, cuboid.dims[1], wall_height],
            pose=[
                cuboid.pose[0] + cuboid.dims[0] / 2 + wall_thickness / 2,
                cuboid.pose[1],
                cuboid.pose[2] + wall_height / 2,
                *unit_quat,
            ],
            color=list(wall_color),
        ),
    ]
    return walls
