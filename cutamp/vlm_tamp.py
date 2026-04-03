# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""VLM-guided sequential subgoal planning for cuTAMP."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from cutamp.algorithm import CutampRunResult, run_cutamp
from cutamp.clients.vlm_client import BaseVLMClient, create_vlm_client
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import clone_tamp_environment, get_env_dict, reduce_tamp_environment
from cutamp.task_planning import Atom
from cutamp.tamp_domain import HandEmpty, On
from cutamp.utils.common import approximate_goal_aabb

_log = logging.getLogger(__name__)

_LINE_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*]\s*|\d+[.)]\s*)")
_FORMAL_SUBGOAL_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*?)\))?\s*$")


@dataclass(frozen=True)
class FormalSubgoal:
    atom: Atom
    source_text: str
    canonical: str


@dataclass
class VLMTAMPResult:
    found_solution: bool
    subgoals_raw: list[str]
    subgoals_validated: list[str]
    attempt_trace: list[dict[str, Any]]
    final_env: Optional[TAMPEnvironment]
    final_q_init: Optional[list[float]]
    best_particle: Optional[dict[str, torch.Tensor]]
    experiment_dir: str
    query_trace: list[dict[str, Any]]
    failure_reason: Optional[str] = None


def _normalize_name(name: str) -> str:
    name = name.strip().strip("`\"'")
    name = name.replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def _subgoal_to_string(atom: Atom) -> str:
    if atom.name == HandEmpty.name:
        return "HandEmpty()"
    return f"{atom.name}({', '.join(atom.values)})"


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_jsonify(v) for v in obj)
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(payload), f, indent=2, ensure_ascii=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _save_env_yaml(path: Path, env: TAMPEnvironment) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(get_env_dict(env), f, sort_keys=False)


def _object_name_maps(env: TAMPEnvironment) -> tuple[dict[str, str], dict[str, str]]:
    object_map = {_normalize_name(obj.name): obj.name for obj in env.movables + env.statics}
    surface_map = {_normalize_name(obj.name): obj.name for obj in env.type_to_objects.get("Surface", [])}
    return object_map, surface_map


def _support_relation(
    env: TAMPEnvironment,
    obj_name: str,
    *,
    xy_tol: float = 0.01,
    z_tol: float = 0.02,
) -> Optional[str]:
    obj = next(obj for obj in env.movables if obj.name == obj_name)
    obj_aabb = approximate_goal_aabb(obj)
    obj_center_xy = obj_aabb.mean(dim=0)[:2]
    obj_bottom_z = float(obj_aabb[0, 2].item())

    candidates: list[tuple[float, str]] = []
    for surface in env.type_to_objects.get("Surface", []):
        surface_aabb = approximate_goal_aabb(surface)
        lower_xy = surface_aabb[0, :2] - xy_tol
        upper_xy = surface_aabb[1, :2] + xy_tol
        if not torch.all(obj_center_xy >= lower_xy) or not torch.all(obj_center_xy <= upper_xy):
            continue
        support_gap = abs(obj_bottom_z - float(surface_aabb[1, 2].item()))
        if support_gap <= z_tol:
            candidates.append((support_gap, surface.name))
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1]


def check_subgoal_achieved(env: TAMPEnvironment, atom: Atom, hand_empty: bool = True) -> bool:
    """Check if a supported subgoal is currently achieved in the environment."""
    if atom.name == HandEmpty.name:
        return hand_empty
    if atom.name == On.name:
        obj_name, surface_name = atom.values
        return _support_relation(env, obj_name) == surface_name
    raise ValueError(f"Unsupported subgoal predicate: {atom.name}")


def build_scene_abstraction(env: TAMPEnvironment) -> dict[str, Any]:
    """Build text-friendly scene metadata for the current environment state."""
    support_relations = {}
    for obj in env.movables:
        support_relations[obj.name] = _support_relation(env, obj.name)

    centers = {}
    for obj in env.movables + env.statics:
        aabb = approximate_goal_aabb(obj)
        centers[obj.name] = aabb.mean(dim=0)

    relative_relations = []
    movable_names = [obj.name for obj in env.movables]
    for idx, name_a in enumerate(movable_names):
        for name_b in movable_names[idx + 1 :]:
            center_a, center_b = centers[name_a], centers[name_b]
            dx = float(center_a[0] - center_b[0])
            dy = float(center_a[1] - center_b[1])
            if abs(dy) > 0.05:
                if dy < 0:
                    relative_relations.append(f"{name_a} is left of {name_b}")
                else:
                    relative_relations.append(f"{name_a} is right of {name_b}")
            if abs(dx) > 0.05:
                if dx < 0:
                    relative_relations.append(f"{name_a} is in front of {name_b} relative to the robot")
                else:
                    relative_relations.append(f"{name_a} is behind {name_b} relative to the robot")

    lines = []
    lines.append("Objects by type:")
    for obj_type, objects in env.type_to_objects.items():
        lines.append(f"- {obj_type}: {', '.join(obj.name for obj in objects)}")
    lines.append("Current support relations:")
    for obj_name, surface_name in support_relations.items():
        if surface_name is None:
            lines.append(f"- {obj_name}: not confidently on a known surface")
        else:
            lines.append(f"- {obj_name}: on {surface_name}")
    lines.append("Robot-manipulable objects:")
    lines.append(f"- {', '.join(movable_names)}")
    if relative_relations:
        lines.append("Coarse relative positions:")
        for relation in relative_relations:
            lines.append(f"- {relation}")

    return {
        "object_names": [obj.name for obj in env.movables + env.statics],
        "movables": movable_names,
        "surfaces": [obj.name for obj in env.type_to_objects.get("Surface", [])],
        "support_relations": support_relations,
        "relative_relations": relative_relations,
        "scene_text": "\n".join(lines),
    }


def render_simple_annotated_scene(
    env: TAMPEnvironment,
    path: Path,
    open_goal: Optional[str] = None,
    scene_text: Optional[str] = None,
) -> Path:
    """Render a simple top-down annotated schematic for the VLM query."""
    width, height = 1280, 800
    padding = 40
    panel_width = 360
    view_right = width - panel_width - padding

    image = Image.new("RGB", (width, height), (248, 247, 243))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    objects = env.movables + env.statics
    aabbs = {obj.name: approximate_goal_aabb(obj) for obj in objects}
    min_x = min(float(aabb[0, 0]) for aabb in aabbs.values())
    max_x = max(float(aabb[1, 0]) for aabb in aabbs.values())
    min_y = min(float(aabb[0, 1]) for aabb in aabbs.values())
    max_y = max(float(aabb[1, 1]) for aabb in aabbs.values())

    span_x = max(max_x - min_x, 1e-3)
    span_y = max(max_y - min_y, 1e-3)

    def project(x: float, y: float) -> tuple[float, float]:
        px = padding + ((x - min_x) / span_x) * (view_right - 2 * padding)
        py = height - padding - ((y - min_y) / span_y) * (height - 2 * padding)
        return px, py

    surface_names = {obj.name for obj in env.type_to_objects.get("Surface", [])}
    def color_for(obj_name: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
        obj = next(obj for obj in objects if obj.name == obj_name)
        color = getattr(obj, "color", None)
        if color is None or len(color) < 3:
            return default
        return tuple(int(v) for v in color[:3])

    draw.rounded_rectangle((15, 15, view_right + 15, height - 15), radius=24, outline=(190, 186, 176), width=2)
    for obj in sorted(env.statics, key=lambda item: 0 if item.name in surface_names else 1):
        lower = aabbs[obj.name][0]
        upper = aabbs[obj.name][1]
        x0, y0 = project(float(lower[0]), float(lower[1]))
        x1, y1 = project(float(upper[0]), float(upper[1]))
        fill = color_for(obj.name, (220, 218, 211))
        outline = (120, 118, 112)
        if obj.name in surface_names:
            draw.rectangle((x0, y1, x1, y0), fill=fill, outline=outline, width=3)
        else:
            draw.rectangle((x0, y1, x1, y0), fill=fill, outline=outline, width=1)
        draw.text((x0 + 4, y1 + 4), obj.name, fill=(45, 45, 45), font=font)

    for obj in env.movables:
        lower = aabbs[obj.name][0]
        upper = aabbs[obj.name][1]
        x0, y0 = project(float(lower[0]), float(lower[1]))
        x1, y1 = project(float(upper[0]), float(upper[1]))
        fill = color_for(obj.name, (85, 141, 224))
        draw.rounded_rectangle((x0, y1, x1, y0), radius=8, fill=fill, outline=(25, 25, 25), width=3)
        draw.text((x0 + 4, y1 + 4), obj.name, fill=(0, 0, 0), font=font)

    draw.rounded_rectangle(
        (view_right + 30, 20, width - 20, height - 20),
        radius=18,
        fill=(255, 253, 249),
        outline=(190, 186, 176),
        width=2,
    )
    panel_lines = ["Scene Summary"]
    if open_goal:
        panel_lines.extend(["", "Open goal:", open_goal])
    if scene_text:
        panel_lines.extend(["", *scene_text.splitlines()])

    y_cursor = 40
    for line in panel_lines[:44]:
        draw.text((view_right + 45, y_cursor), line, fill=(35, 35, 35), font=font)
        y_cursor += 16

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def build_stage1_prompt(
    open_goal: str,
    scene: dict[str, Any],
    successful_subgoals: list[str],
    *,
    failure_context: Optional[dict[str, Any]] = None,
) -> str:
    history = "\n".join(f"- {item}" for item in successful_subgoals) or "- none"
    failure_lines = []
    if failure_context is not None:
        failure_lines.append(f"Failed subgoal: {failure_context.get('failed_subgoal')}")
        blockers = failure_context.get("blockers", [])
        if blockers:
            failure_lines.append(f"Likely blockers: {', '.join(blockers)}")
        details = failure_context.get("details")
        if details:
            failure_lines.append(f"Planning failure details: {details}")
    failure_text = "\n".join(failure_lines) if failure_lines else "None."

    return f"""You are helping a robot solve an open-ended manipulation goal by proposing stable intermediate subgoals.

Open goal:
{open_goal}

Scene summary:
{scene["scene_text"]}

Already completed subgoals:
{history}

Current failure / retry context:
{failure_text}

Rules:
- Return an ordered list of simple English subgoals.
- Each line should describe one stable state only.
- Use the exact object names that appear in the scene summary.
- Focus on shelf/table placement states and hand-empty states only.
- Do not mention trajectories, joint motions, or hidden objects.
- Keep the plan short and practical.

Return only the ordered English subgoals, one per line.
"""


def build_stage2_prompt(
    open_goal: str,
    scene: dict[str, Any],
    english_subgoals: str,
    successful_subgoals: list[str],
) -> str:
    history = "\n".join(f"- {item}" for item in successful_subgoals) or "- none"
    movables = ", ".join(scene["movables"])
    surfaces = ", ".join(scene["surfaces"])
    return f"""Convert the English robot subgoals into the formal grammar used by the planner.

Open goal:
{open_goal}

Scene summary:
{scene["scene_text"]}

Already completed formal subgoals:
{history}

English subgoals:
{english_subgoals}

Allowed predicates:
- On(<movable>, <surface>)
- HandEmpty()

Allowed movable objects:
{movables}

Allowed surfaces:
{surfaces}

Rules:
- Output exactly one grounded predicate per line.
- Use the exact object names from the allowed lists.
- Do not output explanations or JSON.
- Do not combine multiple predicates on one line.
- Use HandEmpty() only when it is a standalone stable subgoal.

Return only formal subgoals, one per line.
"""


def _candidate_subgoal_lines(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, list):
        return [str(item).strip() for item in loaded if str(item).strip()]
    lines = []
    for raw_line in stripped.splitlines():
        line = _LINE_PREFIX_PATTERN.sub("", raw_line).strip().strip(",")
        if not line:
            continue
        lines.append(line)
    return lines


def parse_formal_subgoals(
    text: str,
    env: TAMPEnvironment,
    *,
    achieved_atoms: set[Atom],
    hand_empty: bool = True,
) -> tuple[list[FormalSubgoal], list[str]]:
    """Parse and validate VLM-produced formal subgoals."""
    object_map, surface_map = _object_name_maps(env)
    formal_subgoals = []
    errors = []
    seen = set()
    for line in _candidate_subgoal_lines(text):
        match = _FORMAL_SUBGOAL_PATTERN.match(line)
        if not match:
            errors.append(f"Could not parse line: {line}")
            continue
        pred_name = _normalize_name(match.group(1))
        raw_args = match.group(2)
        args = []
        if raw_args is not None and raw_args.strip():
            args = [_normalize_name(arg) for arg in raw_args.split(",") if arg.strip()]

        if pred_name == _normalize_name(HandEmpty.name):
            if args:
                errors.append(f"HandEmpty takes no arguments: {line}")
                continue
            atom = HandEmpty.ground()
        elif pred_name == _normalize_name(On.name):
            if len(args) != 2:
                errors.append(f"On requires exactly 2 arguments: {line}")
                continue
            obj_name = object_map.get(args[0])
            surface_name = surface_map.get(args[1])
            if obj_name is None:
                errors.append(f"Unknown movable object '{args[0]}' in line: {line}")
                continue
            if surface_name is None:
                errors.append(f"Unknown surface '{args[1]}' in line: {line}")
                continue
            if obj_name not in {obj.name for obj in env.movables}:
                errors.append(f"Object '{obj_name}' is not movable in the current planning environment")
                continue
            atom = On.ground(obj_name, surface_name)
        else:
            errors.append(f"Unsupported predicate '{match.group(1)}' in line: {line}")
            continue

        canonical = _subgoal_to_string(atom)
        if canonical in seen:
            continue
        if atom in achieved_atoms or check_subgoal_achieved(env, atom, hand_empty=hand_empty):
            continue
        seen.add(canonical)
        formal_subgoals.append(FormalSubgoal(atom=atom, source_text=line, canonical=canonical))
    return formal_subgoals, errors


def _goal_state_for_subgoal(atom: Atom):
    if atom.name == HandEmpty.name:
        return frozenset({HandEmpty.ground()})
    return frozenset({atom, HandEmpty.ground()})


def _subgoal_movable_names(atom: Atom) -> list[str]:
    if atom.name == On.name:
        return [atom.values[0]]
    return []


def summarize_blockers(collision_summary: dict[str, float], env: TAMPEnvironment, exclude: set[str]) -> list[str]:
    """Summarize likely blocking movable objects from collision metric names."""
    object_names = [obj.name for obj in env.movables]
    blockers = []
    for metric_name, value in sorted(collision_summary.items(), key=lambda item: item[1], reverse=True):
        if value <= 0:
            continue
        hits = [obj_name for obj_name in object_names if obj_name in metric_name and obj_name not in exclude]
        for hit in hits:
            if hit not in blockers:
                blockers.append(hit)
        if len(blockers) >= 3:
            break
    return blockers


def _run_subgoal_attempt(
    *,
    current_env: TAMPEnvironment,
    goal_atom: Atom,
    config: TAMPConfiguration,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    cutamp_root: Path,
    subgoal_idx: int,
    attempt_idx: int,
    mode: str,
    movable_names: list[str],
    q_init: Optional[list[float]],
    planner_fn: Callable[..., CutampRunResult],
) -> CutampRunResult:
    goal_state = _goal_state_for_subgoal(goal_atom)
    if mode == "full":
        planning_env = clone_tamp_environment(current_env)
        planning_env.goal_state = goal_state
    else:
        planning_env = reduce_tamp_environment(current_env, movable_names=movable_names, goal_state=goal_state)

    attempt_config = replace(
        config,
        enable_vlm_tamp=False,
        experiment_root=str(cutamp_root),
    )
    experiment_id = f"subgoal_{subgoal_idx:03d}_attempt_{attempt_idx:02d}_{mode}"
    return planner_fn(
        planning_env,
        attempt_config,
        cost_reducer,
        constraint_checker,
        q_init=q_init,
        experiment_id=experiment_id,
    )


def run_vlm_tamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    *,
    experiment_id: Optional[str] = None,
    vlm_client: Optional[BaseVLMClient] = None,
    planner_fn: Callable[..., CutampRunResult] = run_cutamp,
) -> VLMTAMPResult:
    """Run the VLM-guided sequential TAMP pipeline."""
    if env.name != "book_shelf":
        raise ValueError("The initial VLM-TAMP implementation currently supports only the book_shelf environment.")
    if not config.enable_vlm_tamp:
        raise ValueError("enable_vlm_tamp must be True to run the VLM-TAMP pipeline")
    if not config.open_goal:
        raise ValueError("open_goal must be provided for VLM-TAMP")

    if experiment_id is None:
        experiment_id = datetime.now().isoformat().split(".")[0]

    experiment_dir = Path(config.experiment_root) / experiment_id
    vlm_dir = experiment_dir / "vlm"
    cutamp_root = experiment_dir / "cutamp"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    vlm_dir.mkdir(parents=True, exist_ok=True)
    cutamp_root.mkdir(parents=True, exist_ok=True)

    _write_text(vlm_dir / "open_goal.txt", config.open_goal)
    _write_json(vlm_dir / "config.json", config.__dict__)
    _save_env_yaml(vlm_dir / "initial_env.yml", env)

    client = vlm_client or create_vlm_client(config)
    current_env = clone_tamp_environment(env)
    current_q_init: Optional[list[float]] = None
    hand_empty = True
    successful_atoms: list[Atom] = []
    query_trace: list[dict[str, Any]] = []
    attempt_trace: list[dict[str, Any]] = []
    last_best_particle = None
    last_failure_reason = None

    query_idx = 0
    reprompt_count = 0
    failure_context: Optional[dict[str, Any]] = None

    while reprompt_count <= config.vlm_max_reprompts:
        scene = build_scene_abstraction(current_env)
        image_path = render_simple_annotated_scene(
            current_env,
            vlm_dir / f"query_{query_idx:02d}.png",
            open_goal=config.open_goal,
            scene_text=scene["scene_text"],
        )
        _write_json(vlm_dir / f"query_{query_idx:02d}_scene.json", scene)

        successful_strings = [_subgoal_to_string(atom) for atom in successful_atoms]
        stage1_prompt = build_stage1_prompt(
            config.open_goal,
            scene,
            successful_strings,
            failure_context=failure_context,
        )
        stage1_response = client.generate(stage1_prompt, image_path=str(image_path), stage="stage1")
        stage2_prompt = build_stage2_prompt(config.open_goal, scene, stage1_response, successful_strings)
        stage2_response = client.generate(stage2_prompt, image_path=str(image_path), stage="stage2")

        _write_text(vlm_dir / f"query_{query_idx:02d}_stage1_prompt.txt", stage1_prompt)
        _write_text(vlm_dir / f"query_{query_idx:02d}_stage1_response.txt", stage1_response)
        _write_text(vlm_dir / f"query_{query_idx:02d}_stage2_prompt.txt", stage2_prompt)
        _write_text(vlm_dir / f"query_{query_idx:02d}_stage2_response.txt", stage2_response)

        parsed_subgoals, parse_errors = parse_formal_subgoals(
            stage2_response,
            current_env,
            achieved_atoms=set(successful_atoms),
            hand_empty=hand_empty,
        )
        validated_strings = [subgoal.canonical for subgoal in parsed_subgoals]
        query_record = {
            "query_idx": query_idx,
            "image_path": str(image_path),
            "stage1_response": stage1_response,
            "stage2_response": stage2_response,
            "validated_subgoals": validated_strings,
            "parse_errors": parse_errors,
            "failure_context": failure_context,
        }
        query_trace.append(query_record)
        _write_json(vlm_dir / f"query_{query_idx:02d}_parsed.json", query_record)

        if not parsed_subgoals:
            last_failure_reason = "VLM formal subgoal parsing produced no valid remaining subgoals"
            failure_context = {
                "failed_subgoal": None,
                "blockers": [],
                "details": last_failure_reason + (f"; parse_errors={parse_errors}" if parse_errors else ""),
            }
            reprompt_count += 1
            query_idx += 1
            continue

        plan_failed = False
        for formal_subgoal in parsed_subgoals:
            goal_atom = formal_subgoal.atom
            subgoal_seq_idx = len(successful_atoms) + 1
            if check_subgoal_achieved(current_env, goal_atom, hand_empty=hand_empty):
                attempt_trace.append(
                    {
                        "subgoal": formal_subgoal.canonical,
                        "subgoal_idx": subgoal_seq_idx,
                        "status": "already_achieved",
                    }
                )
                successful_atoms.append(goal_atom)
                continue

            relevant_movables = _subgoal_movable_names(goal_atom)
            planner_attempts: list[tuple[str, list[str], list[str]]] = []
            if relevant_movables and config.vlm_enable_object_reduction:
                planner_attempts.append(("reduced", relevant_movables, []))
            planner_attempts.append(("full", [obj.name for obj in current_env.movables], []))

            successful_result = None
            blockers = []
            for attempt_idx, (mode, movable_names, promoted_blockers) in enumerate(planner_attempts, start=1):
                result = _run_subgoal_attempt(
                    current_env=current_env,
                    goal_atom=goal_atom,
                    config=config,
                    cost_reducer=cost_reducer,
                    constraint_checker=constraint_checker,
                    cutamp_root=cutamp_root,
                    subgoal_idx=subgoal_seq_idx,
                    attempt_idx=attempt_idx,
                    mode=mode,
                    movable_names=movable_names,
                    q_init=current_q_init,
                    planner_fn=planner_fn,
                )
                blockers = summarize_blockers(result.collision_summary, current_env, exclude=set(relevant_movables))
                attempt_record = {
                    "subgoal": formal_subgoal.canonical,
                    "subgoal_idx": subgoal_seq_idx,
                    "mode": mode,
                    "kept_movables": movable_names,
                    "promoted_blockers": promoted_blockers,
                    "blocker_candidates": blockers,
                    "found_solution": result.found_solution,
                    "collision_summary": result.collision_summary,
                    "overall_metrics": result.overall_metrics,
                }
                attempt_trace.append(attempt_record)
                _write_json(vlm_dir / "attempt_trace.json", {"attempts": attempt_trace})

                if result.found_solution:
                    successful_result = result
                    break

                if mode == "reduced" and blockers:
                    promoted = sorted(set(relevant_movables).union(blockers))
                    planner_attempts.insert(attempt_idx, ("reduced_with_blockers", promoted, blockers))

            if successful_result is None:
                last_failure_reason = f"Planning failed for subgoal {formal_subgoal.canonical}"
                failure_context = {
                    "failed_subgoal": formal_subgoal.canonical,
                    "blockers": blockers,
                    "details": last_failure_reason,
                }
                plan_failed = True
                break

            current_env = successful_result.final_env or current_env
            if successful_result.final_q_init is not None:
                current_q_init = successful_result.final_q_init.detach().cpu().tolist()
            hand_empty = True
            successful_atoms.append(goal_atom)
            last_best_particle = successful_result.best_particle
            _save_env_yaml(vlm_dir / f"state_after_{len(successful_atoms):02d}.yml", current_env)

        if not plan_failed:
            final_q_init = current_q_init
            if last_best_particle is not None:
                torch.save(last_best_particle, vlm_dir / "final_best_particle.pt")
            final_summary = {
                "found_solution": True,
                "subgoals_validated": [_subgoal_to_string(atom) for atom in successful_atoms],
                "num_queries": len(query_trace),
                "attempt_count": len(attempt_trace),
            }
            _write_json(vlm_dir / "final_summary.json", final_summary)
            _save_env_yaml(vlm_dir / "final_env.yml", current_env)
            return VLMTAMPResult(
                found_solution=True,
                subgoals_raw=[record["stage2_response"] for record in query_trace],
                subgoals_validated=[_subgoal_to_string(atom) for atom in successful_atoms],
                attempt_trace=attempt_trace,
                final_env=current_env,
                final_q_init=final_q_init,
                best_particle=last_best_particle,
                experiment_dir=str(experiment_dir),
                query_trace=query_trace,
            )

        reprompt_count += 1
        query_idx += 1

    final_summary = {
        "found_solution": False,
        "subgoals_validated": [_subgoal_to_string(atom) for atom in successful_atoms],
        "num_queries": len(query_trace),
        "attempt_count": len(attempt_trace),
        "failure_reason": last_failure_reason,
    }
    _write_json(vlm_dir / "final_summary.json", final_summary)
    _save_env_yaml(vlm_dir / "final_env.yml", current_env)
    return VLMTAMPResult(
        found_solution=False,
        subgoals_raw=[record["stage2_response"] for record in query_trace],
        subgoals_validated=[_subgoal_to_string(atom) for atom in successful_atoms],
        attempt_trace=attempt_trace,
        final_env=current_env,
        final_q_init=current_q_init,
        best_particle=last_best_particle,
        experiment_dir=str(experiment_dir),
        query_trace=query_trace,
        failure_reason=last_failure_reason,
    )
