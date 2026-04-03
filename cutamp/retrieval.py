import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.experiment_logger import ExperimentLogger
from cutamp.tamp_domain import Conf, Grasp, MoveFree, MoveHolding, Pick, Place, Pose, Push, PushStick, Traj
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.utils.common import (
    Particles,
    action_4dof_to_mat4x4,
    mat4x4_to_action_4dof,
    pose_list_to_mat4x4,
)
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import get_env_dict

_log = logging.getLogger(__name__)
_PLACEHOLDER_TYPES = {Conf, Grasp, Pose, Traj}


@dataclass
class RetrievalMatch:
    particles: Particles
    score: float
    source_experiment: str
    source_artifact: str
    exact_env_match: bool
    num_particles: int


def _wrap_to_pi(vals: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(vals), torch.cos(vals))


def _pose_to_xyz_yaw(pose: list[float]) -> tuple[list[float], float]:
    mat4x4 = pose_list_to_mat4x4(pose)
    xyz = [float(v) for v in mat4x4[:3, 3].tolist()]
    yaw = float(torch.atan2(mat4x4[1, 0], mat4x4[0, 0]).item())
    return xyz, yaw


def build_env_metadata(env: TAMPEnvironment) -> dict[str, Any]:
    env_dict = get_env_dict(env)
    object_poses = {}
    for geometries in env_dict["geometries"].values():
        for name, obj_dict in geometries.items():
            pose = obj_dict.get("pose")
            if pose is None:
                continue
            xyz, yaw = _pose_to_xyz_yaw(pose)
            object_poses[name] = {"xyz": xyz, "yaw": yaw}
    return {
        "env_name": env.name,
        "goal": sorted(json.dumps(atom, sort_keys=True) for atom in env_dict["goal"]),
        "types": {obj_type: sorted(objs) for obj_type, objs in env_dict["types"].items()},
        "object_poses": object_poses,
    }


def env_distance(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    if query["env_name"] != candidate["env_name"]:
        return float("inf")
    if query["goal"] != candidate["goal"]:
        return float("inf")
    if query["types"] != candidate["types"]:
        return float("inf")

    query_poses = query["object_poses"]
    candidate_poses = candidate["object_poses"]
    if set(query_poses) != set(candidate_poses):
        return float("inf")

    total = 0.0
    for name, query_pose in query_poses.items():
        cand_pose = candidate_poses[name]
        query_xyz = torch.tensor(query_pose["xyz"])
        cand_xyz = torch.tensor(cand_pose["xyz"])
        xyz_dist = torch.linalg.norm(query_xyz - cand_xyz).item()
        yaw_dist = abs(math.atan2(math.sin(query_pose["yaw"] - cand_pose["yaw"]), math.cos(query_pose["yaw"] - cand_pose["yaw"])))
        total += xyz_dist + yaw_dist
    return total / max(len(query_poses), 1)


def build_plan_signature(plan_skeleton: PlanSkeleton) -> list[dict[str, Any]]:
    signature = []
    for ground_op in plan_skeleton:
        bindings = []
        for param, value in zip(ground_op.operator.parameters, ground_op.values):
            if param.type in _PLACEHOLDER_TYPES:
                continue
            bindings.append({"name": param.name, "type": param.type, "value": value})
        signature.append({"op": ground_op.operator.name, "bindings": bindings})
    return signature


def _plan_signature_hash(plan_signature: list[dict[str, Any]]) -> str:
    return json.dumps(plan_signature, sort_keys=True)


def _iter_warmstart_slots(plan_skeleton: PlanSkeleton) -> list[dict[str, str]]:
    slots = []
    for step_idx, ground_op in enumerate(plan_skeleton):
        op_name = ground_op.operator.name
        step_prefix = f"step_{step_idx:03d}:{op_name}"
        if op_name == Pick.name:
            obj, grasp, q = ground_op.values
            slots.append(
                {
                    "slot_key": f"{step_prefix}:{obj}:grasp",
                    "param_name": grasp,
                    "param_type": Grasp,
                    "encoding": "raw",
                    "anchor": obj,
                }
            )
            slots.append(
                {
                    "slot_key": f"{step_prefix}:{obj}:q",
                    "param_name": q,
                    "param_type": Conf,
                    "encoding": "raw",
                    "anchor": obj,
                }
            )
        elif op_name == Place.name:
            obj, _grasp, placement, surface, q = ground_op.values
            slots.append(
                {
                    "slot_key": f"{step_prefix}:{obj}:{surface}:placement",
                    "param_name": placement,
                    "param_type": Pose,
                    "encoding": "surface_local_4d",
                    "anchor": surface,
                }
            )
            slots.append(
                {
                    "slot_key": f"{step_prefix}:{obj}:{surface}:q",
                    "param_name": q,
                    "param_type": Conf,
                    "encoding": "raw",
                    "anchor": surface,
                }
            )
        elif op_name == Push.name:
            button, pose_name, q = ground_op.values
            slots.append(
                {
                    "slot_key": f"{step_prefix}:{button}:push",
                    "param_name": pose_name,
                    "param_type": Pose,
                    "encoding": "button_local_4d",
                    "anchor": button,
                }
            )
            slots.append(
                {
                    "slot_key": f"{step_prefix}:{button}:q",
                    "param_name": q,
                    "param_type": Conf,
                    "encoding": "raw",
                    "anchor": button,
                }
            )
        elif op_name == PushStick.name:
            button, stick_name, _grasp, pose_name, q = ground_op.values
            slots.append(
                {
                    "slot_key": f"{step_prefix}:{button}:{stick_name}:push",
                    "param_name": pose_name,
                    "param_type": Pose,
                    "encoding": "button_local_4d",
                    "anchor": button,
                }
            )
            slots.append(
                {
                    "slot_key": f"{step_prefix}:{button}:{stick_name}:q",
                    "param_name": q,
                    "param_type": Conf,
                    "encoding": "raw",
                    "anchor": button,
                }
            )
        elif op_name not in {MoveFree.name, MoveHolding.name}:
            raise NotImplementedError(f"Unsupported operator for retrieval slots: {op_name}")
    return slots


def _repeat_to_length(tensor: torch.Tensor, count: int) -> torch.Tensor:
    if tensor.shape[0] == count:
        return tensor.clone()
    repeats = math.ceil(count / tensor.shape[0])
    rep_dims = (repeats,) + (1,) * (tensor.ndim - 1)
    return tensor.repeat(rep_dims)[:count].clone()


def _apply_noise(tensor: torch.Tensor, param_type: str, noise_scale: float) -> torch.Tensor:
    if noise_scale <= 0:
        return tensor
    tensor = tensor.clone()
    tensor += noise_scale * torch.randn_like(tensor)
    if param_type == Pose and tensor.shape[-1] >= 4:
        tensor[..., 3] = _wrap_to_pi(tensor[..., 3])
    if param_type == Grasp:
        if tensor.shape[-1] == 4:
            tensor[..., 3] = _wrap_to_pi(tensor[..., 3])
        elif tensor.shape[-1] >= 6:
            tensor[..., 3:] = _wrap_to_pi(tensor[..., 3:])
    return tensor


def _canonicalize_slot_tensor(slot: dict[str, str], values: torch.Tensor, world: TAMPWorld) -> torch.Tensor:
    encoding = slot["encoding"]
    if encoding == "raw":
        return values.detach().cpu().clone()

    if encoding not in {"surface_local_4d", "button_local_4d"}:
        raise ValueError(f"Unsupported slot encoding: {encoding}")

    world_from_anchor = pose_list_to_mat4x4(world.get_object(slot["anchor"]).pose).to(values.device, values.dtype)
    anchor_from_world = torch.linalg.inv(world_from_anchor)
    world_from_action = action_4dof_to_mat4x4(values)
    local = torch.matmul(anchor_from_world.expand(values.shape[0], -1, -1), world_from_action)
    return mat4x4_to_action_4dof(local).detach().cpu().clone()


def _restore_slot_tensor(
    slot_meta: dict[str, Any],
    values: torch.Tensor,
    world: TAMPWorld,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = values.to(device=device, dtype=dtype)
    encoding = slot_meta["encoding"]
    if encoding == "raw":
        return values

    if encoding not in {"surface_local_4d", "button_local_4d"}:
        raise ValueError(f"Unsupported slot encoding: {encoding}")

    world_from_anchor = pose_list_to_mat4x4(world.get_object(slot_meta["anchor"]).pose).to(device=device, dtype=dtype)
    local = action_4dof_to_mat4x4(values)
    world_from_action = torch.matmul(world_from_anchor.expand(values.shape[0], -1, -1), local)
    return mat4x4_to_action_4dof(world_from_action)


def get_elite_particles(
    plan_info: dict,
    config: TAMPConfiguration,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
    max_elites: int = 32,
) -> Optional[Particles]:
    particles = plan_info["particles"]
    rollout_fn = plan_info["rollout_fn"]
    cost_fn = plan_info["cost_fn"]

    with torch.no_grad():
        rollout = rollout_fn(particles)
        cost_dict = cost_fn(rollout)

    satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    if not satisfying_mask.any():
        return None

    consider_types = {"constraint"}
    if config.optimize_soft_costs:
        consider_types.add("cost")
    costs = cost_reducer(cost_dict, consider_types=consider_types)

    satisfying_indices = torch.arange(config.num_particles, device=costs.device)[satisfying_mask]
    elite_local = costs[satisfying_mask].argsort()[:max_elites]
    elite_idx = satisfying_indices[elite_local]
    elite_particles = {name: values[elite_idx].detach().cpu().clone() for name, values in particles.items()}
    return elite_particles


def save_retrieval_artifact(
    exp_logger: ExperimentLogger,
    world: TAMPWorld,
    config: TAMPConfiguration,
    plan_skeleton: PlanSkeleton,
    elite_particles: Particles,
) -> Optional[tuple[Path, Path]]:
    if not elite_particles:
        return None

    slot_metadata = {}
    payload = {}
    for slot in _iter_warmstart_slots(plan_skeleton):
        param_name = slot["param_name"]
        if param_name not in elite_particles:
            continue
        stored = _canonicalize_slot_tensor(slot, elite_particles[param_name], world)
        payload[slot["slot_key"]] = stored
        slot_metadata[slot["slot_key"]] = {
            "param_type": slot["param_type"],
            "encoding": slot["encoding"],
            "anchor": slot["anchor"],
            "shape": list(stored.shape),
        }

    if not payload:
        return None

    plan_signature = build_plan_signature(plan_skeleton)
    manifest = {
        "version": 1,
        "experiment_id": exp_logger.exp_dir.name,
        "env_metadata": build_env_metadata(world.env),
        "robot": config.robot,
        "grasp_dof": config.grasp_dof,
        "place_dof": config.place_dof,
        "plan_signature": plan_signature,
        "plan_signature_hash": _plan_signature_hash(plan_signature),
        "num_saved_particles": next(iter(payload.values())).shape[0],
        "slot_metadata": slot_metadata,
        "particle_file": "particles.pt",
    }
    manifest_path = exp_logger.log_dict("retrieval/artifact", manifest)
    payload_path = exp_logger.log_torch("retrieval/particles", payload)
    return manifest_path, payload_path


class RetrievalWarmStarter:
    def __init__(self, world: TAMPWorld, config: TAMPConfiguration):
        self.world = world
        self.config = config
        self.root = Path(config.retrieval_root or config.experiment_root)
        self._index: Optional[list[dict[str, Any]]] = None
        self.last_match: Optional[RetrievalMatch] = None

    def _load_index(self) -> list[dict[str, Any]]:
        if self._index is not None:
            return self._index

        self._index = []
        if not self.root.exists():
            return self._index

        for manifest_path in self.root.rglob("retrieval/artifact.json"):
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                _log.warning(f"Skipping unreadable retrieval manifest {manifest_path}: {exc}")
                continue

            manifest["_manifest_path"] = str(manifest_path)
            self._index.append(manifest)
        return self._index

    def get_warmstart(self, plan_skeleton: PlanSkeleton) -> Particles:
        self.last_match = None
        manifests = self._load_index()
        if not manifests:
            return {}

        query_signature = build_plan_signature(plan_skeleton)
        query_signature_hash = _plan_signature_hash(query_signature)
        query_env_metadata = build_env_metadata(self.world.env)

        best_manifest = None
        best_score = float("inf")
        for manifest in manifests:
            if manifest.get("plan_signature_hash") != query_signature_hash:
                continue
            if manifest.get("robot") != self.config.robot:
                continue
            if manifest.get("grasp_dof") != self.config.grasp_dof:
                continue
            if manifest.get("place_dof") != self.config.place_dof:
                continue

            score = env_distance(query_env_metadata, manifest["env_metadata"])
            if score < best_score:
                best_score = score
                best_manifest = manifest

        if best_manifest is None or math.isinf(best_score):
            return {}

        manifest_path = Path(best_manifest["_manifest_path"])
        tensor_path = manifest_path.parent / best_manifest["particle_file"]
        try:
            payload = torch.load(tensor_path, map_location="cpu")
        except OSError as exc:
            _log.warning(f"Failed to load retrieval payload {tensor_path}: {exc}")
            return {}

        desired_count = min(self.config.retrieval_num_particles, self.config.num_particles)
        warmstart_particles = {}
        slot_metadata = best_manifest["slot_metadata"]
        for slot in _iter_warmstart_slots(plan_skeleton):
            slot_key = slot["slot_key"]
            if slot_key not in slot_metadata or slot_key not in payload:
                continue
            param_name = slot["param_name"]
            restored = _restore_slot_tensor(
                slot_metadata[slot_key],
                payload[slot_key],
                world=self.world,
                device=self.world.device,
                dtype=self.world.q_init.dtype,
            )
            restored = _repeat_to_length(restored, desired_count)
            restored = _apply_noise(restored, slot["param_type"], self.config.retrieval_noise_scale)
            warmstart_particles[param_name] = restored

        if not warmstart_particles:
            return {}

        self.last_match = RetrievalMatch(
            particles=warmstart_particles,
            score=best_score,
            source_experiment=best_manifest["experiment_id"],
            source_artifact=str(tensor_path),
            exact_env_match=best_score <= self.config.retrieval_exact_env_tol,
            num_particles=desired_count,
        )
        return warmstart_particles
