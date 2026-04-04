# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Core cuTAMP algorithm implementation."""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Union, Optional, Tuple
from unittest.mock import Mock

import torch
from curobo.types.base import TensorDeviceType

from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import TAMPEnvironment, clone_tamp_environment, set_object_pose, set_openable_state
from cutamp.experiment_logger import ExperimentLogger
from cutamp.motion_solver import solve_curobo
from cutamp.optimize_plan import ParticleOptimizer
from cutamp.particle_initialization import ParticleInitializer
from cutamp.retrieval import get_elite_particles, save_retrieval_artifact
from cutamp.robots import get_q_home, load_robot_container
from cutamp.rollout import RolloutFunction
from cutamp.tamp_domain import get_tamp_operators_for_env
from cutamp.tamp_world import TAMPWorld, get_tamp_world_initial_collisions
from cutamp.task_planning import PlanSkeleton, task_plan_generator
from cutamp.utils.common import Particles, mat4x4_to_pose_list
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import RerunVisualizer, MockVisualizer

_log = logging.getLogger(__name__)


@dataclass
class CutampRunResult:
    curobo_plan: Any
    num_satisfying_final: int
    found_solution: bool
    best_particle: Optional[Particles]
    final_rollout: Optional[dict]
    final_env: Optional[TAMPEnvironment]
    final_q_init: Optional[torch.Tensor]
    final_openables_state: dict[str, bool] = field(default_factory=dict)
    final_plan_skeleton: Optional[list[str]] = None
    overall_metrics: dict = field(default_factory=dict)
    timer_summaries: dict = field(default_factory=dict)
    collision_summary: dict[str, float] = field(default_factory=dict)


def _sync_perf_counter() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _metric_total(timer_summaries: dict, name: str) -> float:
    metric = timer_summaries.get(name)
    if metric is None:
        return 0.0
    return float(metric["total"])


def _summarize_collision_costs(cost_dict: dict) -> dict[str, float]:
    collision_info = cost_dict.get("Collision")
    if collision_info is None:
        return {}

    values = collision_info.get("values", {})
    summary = {}
    for name, tensor in values.items():
        summary[name] = float(tensor.detach().mean().item())
    return summary


def _get_plan_collision_summary(plan_info: Optional[dict]) -> dict[str, float]:
    if plan_info is None:
        return {}

    particles = plan_info["particles"]
    rollout_fn = plan_info["rollout_fn"]
    cost_fn = plan_info["cost_fn"]
    with torch.no_grad():
        rollout = rollout_fn(particles)
        cost_dict = cost_fn(rollout)
    return _summarize_collision_costs(cost_dict)


def _extract_final_openables_state(world: TAMPWorld, plan_skeleton: PlanSkeleton) -> dict[str, bool]:
    openable_states = {
        name: bool(info["is_open"]) for name, info in world.env.metadata.get("openables", {}).items()
    }
    for ground_op in plan_skeleton:
        if ground_op.operator.name == "Open":
            openable_states[ground_op.values[0]] = True
    return openable_states


def _build_final_environment(
    world: TAMPWorld,
    final_rollout: dict,
    final_openables_state: Optional[dict[str, bool]] = None,
) -> TAMPEnvironment:
    final_env = clone_tamp_environment(world.env)
    for obj_name, poses in final_rollout["obj_to_pose"].items():
        last_pose = poses[0, -1].detach().cpu()
        set_object_pose(final_env, obj_name, mat4x4_to_pose_list(last_pose))
    if final_openables_state is not None:
        for openable, is_open in final_openables_state.items():
            set_openable_state(final_env, openable, is_open)
    return final_env


def _build_timing_breakdown(timer_summaries: dict, plan_timing_rows: list[dict], overall_metrics: dict) -> dict:
    return {
        "notes": [
            "top_level_sec contains the main wall-clock phases for the run.",
            "detail sections are nested breakdowns and should not be summed together with top-level phases.",
        ],
        "top_level_sec": {
            "load_tamp_world": _metric_total(timer_summaries, "load_tamp_world"),
            "warmup_ik_solver": _metric_total(timer_summaries, "warmup_ik_solver"),
            "warmup_motion_gen": _metric_total(timer_summaries, "curobo_motion_gen_warmup"),
            "get_plan_generator": _metric_total(timer_summaries, "get_plan_generator"),
            "sample_initial_plans": _metric_total(timer_summaries, "sample_initial_plans"),
            "sort_plans": _metric_total(timer_summaries, "sort_plans"),
            "start_optimization": _metric_total(timer_summaries, "start_optimization"),
            "curobo_planning": _metric_total(timer_summaries, "curobo_planning"),
        },
        "sampling_detail_sec": {
            "task_plan_next": _metric_total(timer_summaries, "task_plan_next"),
            "sample_plan_skeleton_total": _metric_total(timer_summaries, "sample_plan_skeleton_total"),
            "initialize_particles": _metric_total(timer_summaries, "initialize_particles"),
            "measure_heuristic": _metric_total(timer_summaries, "measure_heuristic"),
            "get_satisfying_mask": _metric_total(timer_summaries, "get_satisfying_mask"),
            "compute_best_cost": _metric_total(timer_summaries, "compute_best_cost"),
        },
        "optimization_detail_sec": {
            "optimize_plan_total": _metric_total(timer_summaries, "optimize_plan_total"),
            "setup_optimizer": _metric_total(timer_summaries, "setup_optimizer"),
            "optimization_step": _metric_total(timer_summaries, "optimization_step"),
            "resample_duration": _metric_total(timer_summaries, "resample_duration"),
            "resample_plan_info": _metric_total(timer_summaries, "resample_plan_info"),
            "visualize_opt_rollout": _metric_total(timer_summaries, "visualize_opt_rollout"),
            "visualize_rollout": _metric_total(timer_summaries, "visualize_rollout"),
        },
        "counts": {
            "sampled_plan_skeletons": len([row for row in plan_timing_rows if row.get("plan_skeleton") is not None]),
            "failed_subgraphs": len([row for row in plan_timing_rows if row.get("failed_subgraph")]),
            "optimized_plans": overall_metrics["num_optimized_plans"],
        },
    }


def heuristic_fn(
    plan_skeleton: PlanSkeleton, cost_dict: dict, constraint_checker: ConstraintChecker, verbose: bool = True
) -> float:
    """
    Get a single heuristic value for a cost dict corresponding to a rollout.

    We first compute the success rate of each constraint. If the constraint has zero success, we assign it a penalty
    of -num_particles. We then compute the mean success rate across all constraints, and use the failure rate as the
    heuristic (lower the better).
    """
    full_mask = constraint_checker.get_full_mask(cost_dict)
    successes = []
    num_particles = None
    for con_type, con_info in full_mask.items():
        for name, mask in con_info.items():
            if mask.ndim == 2:
                satisfying = mask.sum(0)
            else:
                satisfying = mask.sum()

            if num_particles is None:
                num_particles = mask.shape[0]
            else:
                assert num_particles == mask.shape[0]

            # replace zeros with -num_particles
            satisfying[satisfying == 0] = -num_particles
            successes.extend(satisfying.tolist())
            if verbose:
                _log.debug(f"{con_type} {name} {satisfying.tolist()}")
    success_mean = sum(successes) / len(successes)
    success_rate = success_mean / num_particles
    failure_rate = 1 - success_rate
    heuristic = 100 * failure_rate

    # We have a preference for shorter plans
    heuristic += len(plan_skeleton)
    return heuristic


def get_best_particle(
    plan_info: dict, config: TAMPConfiguration, constraint_checker: ConstraintChecker, cost_reducer: CostReducer
) -> dict:
    """Get the particle that satisfies the constraints and has the best soft cost."""
    particles, rollout_fn, cost_fn = plan_info["particles"], plan_info["rollout_fn"], plan_info["cost_fn"]
    with torch.no_grad():
        rollout = rollout_fn(particles)
        cost_dict = cost_fn(rollout)

    # Take the best particle that is satisfying and has the best soft cost
    satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    if not satisfying_mask.any():
        raise RuntimeError("No satisfying particles found")

    soft_costs = cost_reducer.soft_costs(cost_dict)
    satisfying_costs = soft_costs[satisfying_mask]
    best_satisfying_idx = satisfying_costs.argmin()
    indices = torch.arange(config.num_particles, device=satisfying_costs.device)
    best_idx = indices[satisfying_mask][best_satisfying_idx]
    best_particle = {k: v[best_idx].detach().clone() for k, v in particles.items()}
    return best_particle


def sample_plan_skeleton(
    plan_skeleton: PlanSkeleton,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    plan_count: int,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
    particle_initializer: ParticleInitializer,
    task_plan_next_sec: Optional[float] = None,
) -> Tuple[Union[dict, None], bool, dict]:
    """
    Try sampling particles for a plan skeleton and compute the heuristic.
    Returns the plan_info dict and whether any satisfying particles were found upon initialization.
    """
    plan_str = [op.name for op in plan_skeleton]
    _log.debug(f"[Plan {plan_count + 1}] Sampled plan {plan_str}")

    sample_start = _sync_perf_counter()

    # Sample particles
    timer.start("initialize_particles")
    plan_particles = particle_initializer(plan_skeleton)
    initialize_particles_sec = timer.stop("initialize_particles")
    if plan_particles is None:  # failed subgraph
        sample_timings = {
            "plan_idx": plan_count,
            "plan_skeleton": [str(op) for op in plan_skeleton],
            "task_plan_next_sec": task_plan_next_sec,
            "sample_plan_total_sec": _sync_perf_counter() - sample_start,
            "initialize_particles_sec": initialize_particles_sec,
            "measure_heuristic_sec": 0.0,
            "get_satisfying_mask_sec": 0.0,
            "compute_best_cost_sec": 0.0,
            "failed_subgraph": True,
            "retrieval": particle_initializer.latest_retrieval_info.copy(),
        }
        timer._metrics["sample_plan_skeleton_total"].append(sample_timings["sample_plan_total_sec"])
        return None, False, sample_timings

    # Rollout particles and compute costs
    rollout_fn = RolloutFunction(plan_skeleton, world, config)
    cost_fn = CostFunction(plan_skeleton, world, config)
    timer.start("measure_heuristic")
    with torch.no_grad():
        rollout = rollout_fn(plan_particles)
        cost_dict = cost_fn(rollout)
        heuristic = heuristic_fn(plan_skeleton, cost_dict, constraint_checker)
    measure_heuristic_sec = timer.stop("measure_heuristic")

    # Number of satisfying particles
    timer.start("get_satisfying_mask")
    satisfying_mask = constraint_checker.get_mask(cost_dict)
    get_satisfying_mask_sec = timer.stop("get_satisfying_mask")
    num_satisfying = satisfying_mask.sum().item()

    if config.stick_button_experiment and num_satisfying > 0:
        # Custom logic in stick button for breaking early for sampling baseline
        heuristic -= 100
        print(f"Found satisfying plan: {plan_str} heuristic -= 100")

    # Best cost initially
    timer.start("compute_best_cost")
    consider_types = {"constraint"}
    if config.optimize_soft_costs:
        consider_types.add("cost")
    costs = cost_reducer(cost_dict, consider_types=consider_types)
    if satisfying_mask.any():
        best_cost = costs[satisfying_mask].min().item()
        best_soft_cost = cost_reducer.soft_costs(cost_dict)[satisfying_mask].min().item()
    else:
        best_cost, best_soft_cost = float("inf"), float("inf")
    compute_best_cost_sec = timer.stop("compute_best_cost")
    sample_plan_total_sec = _sync_perf_counter() - sample_start
    timer._metrics["sample_plan_skeleton_total"].append(sample_plan_total_sec)

    plan_info = {
        "idx": plan_count,
        "plan_skeleton": plan_skeleton,
        "particles": plan_particles,
        "rollout_fn": rollout_fn,
        "cost_fn": cost_fn,
        "heuristic": heuristic,
        "num_satisfying": num_satisfying,
        "best_cost": best_cost,
        "best_soft_cost": best_soft_cost,
        "retrieval": particle_initializer.latest_retrieval_info.copy(),
    }
    sample_timings = {
        "plan_idx": plan_count,
        "plan_skeleton": [str(op) for op in plan_skeleton],
        "task_plan_next_sec": task_plan_next_sec,
        "sample_plan_total_sec": sample_plan_total_sec,
        "initialize_particles_sec": initialize_particles_sec,
        "measure_heuristic_sec": measure_heuristic_sec,
        "get_satisfying_mask_sec": get_satisfying_mask_sec,
        "compute_best_cost_sec": compute_best_cost_sec,
        "failed_subgraph": False,
        "heuristic": heuristic,
        "num_satisfying_initial": num_satisfying,
        "best_cost_initial": best_cost,
        "best_soft_cost_initial": best_soft_cost,
        "retrieval": particle_initializer.latest_retrieval_info.copy(),
        "optimized": False,
    }
    plan_info["timing_entry"] = sample_timings

    _log.debug(
        f"[Plan {plan_count + 1}] {plan_info['num_satisfying']}/{config.num_particles} satisfying, "
        f"heuristic = {plan_info['heuristic']}"
    )
    return plan_info, num_satisfying > 0, sample_timings


def resample_plan_info(
    plan_info: dict,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    particle_initializer: ParticleInitializer,
) -> int:
    """
    Sample particles again in-place for a plan info container with a plan skeleton. This can be used for rejection
    sampling strategy (for the sampling baseline), or for random restarts.

    Returns number of satisfying particles after re-sampling.
    """
    with timer.time("initialize_particles"), timer.time("resample_particles"):
        plan_particles = particle_initializer(plan_info["plan_skeleton"], verbose=False)

    # Rollout new particles and compute costs
    with timer.time("measure_heuristic"), torch.no_grad():
        rollout = plan_info["rollout_fn"](plan_particles)
        cost_dict = plan_info["cost_fn"](rollout)
        heuristic = heuristic_fn(plan_info["plan_skeleton"], cost_dict, constraint_checker, verbose=False)

    # Number of satisfying particles
    with timer.time("get_satisfying_mask"):
        satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    num_satisfying = satisfying_mask.sum().item()

    # Best cost
    with timer.time("compute_best_cost"):
        consider_types = {"constraint"}
        if config.optimize_soft_costs:
            consider_types.add("cost")
        costs = cost_reducer(cost_dict, consider_types=consider_types)
        if satisfying_mask.any():
            best_cost = costs[satisfying_mask].min().item()  # note: should consider satisfying mask?
            soft_costs = cost_reducer.soft_costs(cost_dict)
            best_soft_cost = soft_costs[satisfying_mask].min().item()
            indices = torch.arange(config.num_particles, device=soft_costs.device)
            best_idx = indices[satisfying_mask][costs[satisfying_mask].argmin()]
            best_soft_idx = indices[satisfying_mask][soft_costs[satisfying_mask].argmin()]
        else:
            best_cost, best_soft_cost = float("inf"), float("inf")
            best_idx = None
            best_soft_idx = None

    # Update plan info
    plan_info["particles"] = plan_particles
    plan_info["heuristic"] = heuristic
    plan_info["num_satisfying"] = num_satisfying
    plan_info["best_cost"] = best_cost
    plan_info["best_soft_cost"] = best_soft_cost
    plan_info["rollout"] = rollout
    plan_info["best_idx"] = best_idx
    plan_info["best_soft_idx"] = best_soft_idx
    return num_satisfying


def setup_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
):
    # Validate args and setup experiment logger
    validate_tamp_config(config)
    if experiment_id is None:
        experiment_id = datetime.now().isoformat().split(".")[0]

    exp_logger = ExperimentLogger(name=experiment_id, config=config) if config.enable_experiment_logging else Mock()
    exp_logger.save_env(env)

    # Loading robot can be done offline, so doesn't count towards timing
    tensor_args = TensorDeviceType()
    robot_container = load_robot_container(config.robot, tensor_args)
    if q_init is None:
        q_init = get_q_home(config.robot)
    q_init = tensor_args.to_device(q_init)

    # Load TAMP world and warmup IK solver
    timer = TorchTimer()
    with timer.time("load_tamp_world", log_callback=_log.info):
        world = TAMPWorld(
            env,
            tensor_args,
            robot=robot_container,
            q_init=q_init,
            collision_activation_distance=config.world_activation_distance,
            coll_n_spheres=config.coll_n_spheres,
            coll_sphere_radius=config.coll_sphere_radius,
        )

    if config.warmup_ik:
        with timer.time("warmup_ik_solver", log_callback=_log.info):
            world.warmup_ik_solver(config.num_particles)

    # Setup visualizer (doesn't count towards timing)
    if config.enable_visualizer:
        rr_recording_path = str(exp_logger.exp_dir / "rerun.rrd")
        visualizer = RerunVisualizer(
            config,
            q_init,
            application_id=env.name,
            recording_id=experiment_id,
            spawn=config.rr_spawn,
            save_path=rr_recording_path,
        )
        visualizer.log_tamp_world(world)
    else:
        visualizer = MockVisualizer()
    return exp_logger, visualizer, timer, world


def run_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
):
    """Overall cuTAMP algorithm implementation."""
    # Setup all the things and load the world
    exp_logger, visualizer, timer, world = setup_cutamp(env, config, q_init, experiment_id)

    initial_collisions = get_tamp_world_initial_collisions(world)
    if initial_collisions:
        sorted_collisions = dict(sorted(initial_collisions.items(), key=lambda item: item[1], reverse=True))
        _log.warning("Initial state in collision: %s", sorted_collisions)
        overall_metrics = {
            "num_optimized_plans": 0,
            "num_initial_plans": 0,
            "num_skipped_plans": 0,
            "num_satisfying_final": 0,
            "num_particles": config.num_particles,
            "best_cost": float("inf"),
            "best_soft_cost": float("inf"),
            "found_solution": False,
            "initial_collisions": sorted_collisions,
        }
        timer_summaries = timer.get_summaries()
        exp_logger.log_dict("overall_metrics", overall_metrics)
        exp_logger.log_dict("timer_metrics", timer_summaries)
        exp_logger.log_dict("plan_timing_metrics", {"plans": []})
        exp_logger.log_dict("timing_breakdown", _build_timing_breakdown(timer_summaries, [], overall_metrics))
        exp_logger.log_dict("multipliers", cost_reducer.cost_config)
        exp_logger.log_dict("tolerances", constraint_checker.constraint_config)
        return CutampRunResult(
            curobo_plan=None,
            num_satisfying_final=0,
            found_solution=False,
            best_particle=None,
            final_rollout=None,
            final_env=None,
            final_q_init=None,
            overall_metrics=overall_metrics,
            timer_summaries=timer_summaries,
            collision_summary=sorted_collisions,
        )

    particle_initializer = ParticleInitializer(world, config)

    # Task plan generator
    _log.info(f"Initial State: {world.initial_state}")
    _log.info(f"Goal State: {world.goal_state}")
    with timer.time("get_plan_generator", log_callback=_log.info):
        plan_gen = task_plan_generator(
            world.initial_state,
            world.goal_state,
            operators=get_tamp_operators_for_env(env.name),
            explored_state_check=config.explored_state_check,
            max_depth=config.task_plan_max_depth,
        )

    # Sample initial plans and particles
    found_solution_initially = False
    num_skipped_plans = 0
    plan_timing_rows = []
    with timer.time("sample_initial_plans", log_callback=_log.info):
        plan_queue: List[dict] = []
        plan_count = 0
        for idx in range(config.num_initial_plans):
            timer.start("task_plan_next")
            plan_skeleton = next(plan_gen, None)
            task_plan_next_sec = timer.stop("task_plan_next")
            if plan_skeleton is None:
                _log.info("Ran out of plans to sample")
                break

            plan_info, has_solution, sample_timings = sample_plan_skeleton(
                plan_skeleton,
                world,
                config,
                timer,
                idx,
                constraint_checker,
                cost_reducer,
                particle_initializer,
                task_plan_next_sec=task_plan_next_sec,
            )
            sample_timings["sample_order"] = idx
            plan_timing_rows.append(sample_timings)
            if plan_info is None:
                _log.debug("failed subgraph, skipping...")
                num_skipped_plans += 1
                continue

            plan_queue.append(plan_info)
            if has_solution:
                found_solution_initially = True
                break
            plan_count += 1

    # Sort plans by heuristic
    def sort_plans():
        with timer.time("sort_plans"):
            plan_queue.sort(
                key=lambda x: (
                    0 if x.get("retrieval", {}).get("exact_env_match") else 1,
                    0 if x.get("retrieval", {}).get("hit") else 1,
                    x.get("retrieval", {}).get("score", float("inf")),
                    x["heuristic"],
                )
            )

    sort_plans()
    _log.info(f"Num plans: {len(plan_queue)}, num skipped: {num_skipped_plans}")
    overall_metrics = {
        "num_optimized_plans": 0,
        "num_initial_plans": plan_count,
        "num_skipped_plans": num_skipped_plans,
        "num_satisfying_final": 0,
        "num_particles": config.num_particles,
        "best_cost": float("inf"),
        "best_soft_cost": float("inf"),
    }
    curobo_plan = None
    found_solution = False
    final_plan_info = None
    final_best_particle = None
    final_rollout = None
    final_env = None
    final_q_init = None
    final_openables_state = {}
    failure_plan_info = None
    particle_optimizer = ParticleOptimizer(config, cost_reducer, constraint_checker)
    timer.start("first_solution")
    if found_solution_initially:
        found_solution = True
        timer.stop("first_solution")

    # Optimization loop for each skeleton and its particles
    timer.start("start_optimization")
    for idx, plan_info in enumerate(plan_queue):
        opt_iter = idx + 1
        should_break = False
        plan_skeleton = plan_info["plan_skeleton"]
        _log.info(
            f"[Opt {opt_iter}] Optimizing plan {[op.name for op in plan_skeleton]}, plan idx = {plan_info['idx']}, "
            f"heuristic = {plan_info['heuristic']:.2f}"
        )
        best_particle = None
        timing_entry = plan_info.get("timing_entry")
        if timing_entry is not None:
            timing_entry["optimized"] = True
            timing_entry["optimization_order"] = opt_iter
            timing_entry["queue_order"] = idx

        plan_opt_start = _sync_perf_counter()
        timer.start("optimize_plan_total")

        if config.approach == "optimization":
            has_satisfying, metrics, time_exceeded = particle_optimizer(plan_info, timer, visualizer)
            metrics["retrieval"] = plan_info.get("retrieval", {})
            if metrics["best_cost"] is not None:
                overall_metrics["best_cost"] = min(overall_metrics["best_cost"], metrics["best_cost"])
            if metrics["best_soft_cost"] is not None:
                overall_metrics["best_soft_cost"] = min(overall_metrics["best_soft_cost"], metrics["best_soft_cost"])
            if time_exceeded:
                _log.info(f"Max loop duration reached, stopping optimization")
                should_break = True
            exp_logger.log_dict(f"optimization/opt_{opt_iter:04d}", metrics)
            if has_satisfying:
                best_particle = get_best_particle(plan_info, config, constraint_checker, cost_reducer)
        else:
            # This is the parallelized sampling baseline
            assert config.approach == "sampling"
            num_resample_attempts = 0
            resample_dur = 0.0
            has_satisfying = plan_info["num_satisfying"] > 0
            total_num_satisfying = plan_info["num_satisfying"]
            best_particle = None
            best_soft_costs = []
            elapsed = []

            if not has_satisfying or not config.break_on_satisfying:
                timer.start("resample_duration")
                for resample_idx in range(config.num_resampling_attempts):
                    if config.max_loop_dur is not None and timer.elapsed("start_optimization") >= config.max_loop_dur:
                        _log.info(f"Max loop duration reached, stopping resampling")
                        should_break = True
                        break
                    timer.start("resample_plan_info")
                    num_satisfying = resample_plan_info(
                        plan_info,
                        world,
                        config,
                        timer,
                        cost_reducer,
                        constraint_checker,
                        particle_initializer,
                    )
                    total_num_satisfying += num_satisfying
                    if plan_info["best_soft_cost"] < overall_metrics["best_soft_cost"]:
                        best_soft_idx = plan_info["best_soft_idx"]
                        best_particle = {
                            k: v[best_soft_idx].detach().clone() for k, v in plan_info["particles"].items()
                        }

                    overall_metrics["best_cost"] = min(overall_metrics["best_cost"], plan_info["best_cost"])
                    overall_metrics["best_soft_cost"] = min(
                        overall_metrics["best_soft_cost"], plan_info["best_soft_cost"]
                    )

                    # Keep track of the best soft cost since start of resampling
                    best_soft_costs.append(overall_metrics["best_soft_cost"])
                    elapsed.append(timer.elapsed("start_optimization"))

                    resample_plan_info_dur = timer.stop("resample_plan_info")
                    _log.debug(
                        f"[Plan {plan_info['idx'] + 1}] Resample attempt {resample_idx + 1}/{config.num_resampling_attempts}, "
                        f"{num_satisfying}/{config.num_particles} satisfying particles. Total satisfying {total_num_satisfying}. "
                        f"Took {resample_plan_info_dur:.2f}s"
                    )
                    has_satisfying = num_satisfying > 0
                    num_resample_attempts += 1

                    # Visualize best particle rollout state
                    rollout = plan_info["rollout"]
                    best_soft_idx = plan_info["best_soft_idx"]
                    if best_soft_idx is None:
                        best_soft_idx = 0
                    visualizer.set_time_sequence(f"samp", num_resample_attempts)
                    q_last = rollout["confs"][best_soft_idx, -1].tolist()
                    visualizer.set_joint_positions(q_last)
                    for obj in rollout["obj_to_pose"]:
                        mat4x4_last = rollout["obj_to_pose"][obj][best_soft_idx, -1]
                        visualizer.log_mat4x4(f"world/{obj}", mat4x4_last)

                    if has_satisfying:
                        if timer.has_timer("first_solution"):
                            time_to_first_sol = timer.stop("first_solution")
                            _log.info(f"Found first solution in {time_to_first_sol:.2f}s after sampling plans")
                        if config.break_on_satisfying:
                            should_break = True
                            break
                resample_dur = timer.stop("resample_duration")
                _log.info(f"Total resample duration: {resample_dur:.2f}s")
            else:
                _log.info("Already has satisfying particles, skipping resampling")
                overall_metrics["best_cost"] = min(overall_metrics["best_cost"], plan_info["best_cost"])
                overall_metrics["best_soft_cost"] = min(overall_metrics["best_soft_cost"], plan_info["best_soft_cost"])
                if config.break_on_satisfying:
                    should_break = True

            metrics = {
                "plan_skeleton": [str(op) for op in plan_skeleton],
                "num_particles": config.num_particles,
                "num_resample_attempts": num_resample_attempts,
                "resample_duration": resample_dur,
                "num_satisfying_final": total_num_satisfying,
                "total_num_particles": config.num_particles * (num_resample_attempts + 1),
                "best_cost": overall_metrics["best_cost"],
                "best_soft_cost": overall_metrics["best_soft_cost"],
                "best_soft_costs": best_soft_costs,
                "elapsed": elapsed,
                "retrieval": plan_info.get("retrieval", {}),
            }
            exp_logger.log_dict(f"sampling/samp_{opt_iter:04d}", metrics)
            has_satisfying = total_num_satisfying > 0
            overall_metrics["num_satisfying_final"] = total_num_satisfying

            # Log best particle as last
            if best_particle is not None:
                rollout = plan_info["rollout_fn"]({k: v[None] for k, v in best_particle.items()})
                visualizer.set_time_sequence(f"samp", num_resample_attempts)
                q_last = rollout["confs"][0, -1].tolist()
                visualizer.set_joint_positions(q_last)

                for obj in rollout["obj_to_pose"]:
                    mat4x4_last = rollout["obj_to_pose"][obj][0, -1]
                    visualizer.log_mat4x4(f"world/{obj}", mat4x4_last)

        optimize_plan_total_sec = timer.stop("optimize_plan_total")
        if timing_entry is not None:
            timing_entry["optimization_total_sec"] = optimize_plan_total_sec
            timing_entry["optimization_wall_sec"] = _sync_perf_counter() - plan_opt_start
            timing_entry["found_solution_after_optimization"] = has_satisfying
            timing_entry["num_satisfying_final"] = metrics["num_satisfying_final"]
            timing_entry["best_cost_final"] = metrics.get("best_cost")
            timing_entry["best_soft_cost_final"] = metrics.get("best_soft_cost")
            timing_entry["retrieval"] = plan_info.get("retrieval", {})
            if config.approach == "optimization":
                timing_entry["optimization_step_count"] = len(metrics.get("num_satisfying", []))
                timing_entry["opt_start_to_first_solution_sec"] = metrics.get("opt_start_to_first_sol")
            else:
                timing_entry["resample_attempts"] = metrics.get("num_resample_attempts")
                timing_entry["resample_duration_sec"] = metrics.get("resample_duration")

        # Now we've either optimized or resampled
        overall_metrics["num_optimized_plans"] += 1
        if has_satisfying:
            found_solution = True
            final_plan_info = plan_info
            if best_particle is None:
                best_particle = get_best_particle(plan_info, config, constraint_checker, cost_reducer)
            final_best_particle = best_particle
            final_rollout = plan_info["rollout_fn"]({k: v[None].detach().clone() for k, v in best_particle.items()})
            final_openables_state = _extract_final_openables_state(world, plan_skeleton)
            final_env = _build_final_environment(world, final_rollout, final_openables_state)
            final_q_init = final_rollout["confs"][0, -1].detach().clone()
            if config.curobo_plan:
                curobo_plan = solve_curobo(
                    plan_info,
                    best_particle,
                    world,
                    config,
                    timer,
                    visualizer,
                )
            overall_metrics["num_satisfying_final"] = metrics["num_satisfying_final"]
            overall_metrics["final_plan_skeleton"] = [str(op) for op in plan_skeleton]
            overall_metrics["retrieval"] = plan_info.get("retrieval", {})
            _log.debug(f"Total num satisfying {metrics['num_satisfying_final']}")
            if config.break_on_satisfying:
                should_break = True
        else:
            failure_plan_info = plan_info

        if should_break:
            break

        # TODO: complete version of our algorithm that adds additional skeletons to the queue, resorts, revisits
        #  skeletons, etc.
        # new_plan_info = sample_plan_skeleton()
        # if new_plan_info is not None:
        #     plan_queue.append(new_plan_info)
        #     sort_plans()

    opt_elapsed = timer.stop("start_optimization")
    _log.debug(f"Optimization loop took roughly {opt_elapsed:.2f}s")
    if not found_solution:
        _log.warning("No satisfying particles found after optimizing all plans")
    _log.debug(f"Best cost: {overall_metrics['best_cost']:.4f}, soft cost: {overall_metrics['best_soft_cost']:.4f}")

    # Dump metrics out
    overall_metrics["found_solution"] = found_solution
    timer_summaries = timer.get_summaries()
    exp_logger.log_dict("overall_metrics", overall_metrics)
    exp_logger.log_dict("timer_metrics", timer_summaries)
    exp_logger.log_dict("plan_timing_metrics", {"plans": plan_timing_rows})
    exp_logger.log_dict("timing_breakdown", _build_timing_breakdown(timer_summaries, plan_timing_rows, overall_metrics))

    # Save retrieval artifact for future warm starts
    if found_solution and config.enable_experiment_logging and config.save_retrieval_artifacts and final_plan_info is not None:
        elite_particles = get_elite_particles(final_plan_info, config, constraint_checker, cost_reducer)
        if elite_particles is not None:
            save_retrieval_artifact(exp_logger, world, config, final_plan_info["plan_skeleton"], elite_particles)

    # Log constraint and cost multipliers
    exp_logger.log_dict("multipliers", cost_reducer.cost_config)
    exp_logger.log_dict("tolerances", constraint_checker.constraint_config)
    collision_summary = _get_plan_collision_summary(final_plan_info if found_solution else failure_plan_info)
    return CutampRunResult(
        curobo_plan=curobo_plan,
        num_satisfying_final=overall_metrics["num_satisfying_final"],
        found_solution=found_solution,
        best_particle=final_best_particle,
        final_rollout=final_rollout,
        final_env=final_env,
        final_q_init=final_q_init,
        final_openables_state=final_openables_state,
        final_plan_skeleton=overall_metrics.get("final_plan_skeleton"),
        overall_metrics=overall_metrics,
        timer_summaries=timer_summaries,
        collision_summary=collision_summary,
    )
