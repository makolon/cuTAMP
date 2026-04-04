"""Benchmark end-to-end VLM-TAMP tasks on the fixed mini-kitchen task suite."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from cutamp.clients.vlm_client import create_vlm_client
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.mini_kitchen import load_mini_kitchen_env
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol, set_random_seed, setup_logging
from cutamp.vlm_tamp import VLMTAMPResult, run_vlm_tamp


def _task_suite() -> dict[str, str]:
    env = load_mini_kitchen_env()
    return dict(env.metadata["task_suite"])


def _attempt_total_sec(attempt: dict) -> float:
    timer_summaries = attempt.get("timer_summaries") or {}
    top_level_metrics = [
        "load_tamp_world",
        "warmup_ik_solver",
        "curobo_motion_gen_warmup",
        "get_plan_generator",
        "sample_initial_plans",
        "sort_plans",
        "start_optimization",
        "curobo_planning",
    ]
    total = 0.0
    for metric_name in top_level_metrics:
        metric = timer_summaries.get(metric_name)
        if metric is not None:
            total += float(metric["total"])
    return total


def _summarize_result(task_name: str, open_goal: str, wall_time_sec: float, result: VLMTAMPResult) -> dict:
    successful_attempts = [attempt for attempt in result.attempt_trace if attempt.get("found_solution")]
    total_primitive_actions = sum(int(attempt.get("primitive_action_count", 0)) for attempt in successful_attempts)
    successful_subgoal_time_sec = sum(_attempt_total_sec(attempt) for attempt in successful_attempts)
    num_subgoals = len(result.subgoals_validated)
    return {
        "task_name": task_name,
        "open_goal": open_goal,
        "found_solution": result.found_solution,
        "wall_time_sec": wall_time_sec,
        "num_vlm_queries": len(result.query_trace),
        "subgoal_success_count": num_subgoals,
        "subgoals_validated": result.subgoals_validated,
        "attempt_count": len(result.attempt_trace),
        "total_primitive_actions": total_primitive_actions,
        "successful_subgoal_time_sec": successful_subgoal_time_sec,
        "mean_subgoal_time_sec": successful_subgoal_time_sec / num_subgoals if num_subgoals > 0 else None,
        "experiment_dir": result.experiment_dir,
        "failure_reason": result.failure_reason,
    }


def entrypoint():
    parser = argparse.ArgumentParser(
        description="Run the fixed mini-kitchen VLM-TAMP task suite and summarize long-horizon performance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", choices=["mini_kitchen"], default="mini_kitchen")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment_root", type=str, default="logs/vlm-benchmark-experiments")
    parser.add_argument("--experiment_prefix", type=str, default="mini_kitchen_benchmark")
    parser.add_argument("--task_plan_max_depth", type=int, default=TAMPConfiguration.task_plan_max_depth)
    parser.add_argument("--num_particles", type=int, default=1024)
    parser.add_argument("--num_initial_plans", type=int, default=30)
    parser.add_argument("--num_opt_steps", type=int, default=1000)
    parser.add_argument("--max_duration", type=float, default=None)
    parser.add_argument("--robot", choices=["panda", "ur5"], default="panda")
    parser.add_argument("--grasp_dof", choices=[4, 6], type=int, default=4)
    parser.add_argument("--approach", choices=["optimization", "sampling"], default="optimization")
    parser.add_argument("--num_resampling_attempts", type=int, default=100)
    parser.add_argument("--cache_subgraphs", action="store_true")
    parser.add_argument("--motion_plan", action="store_true")
    parser.add_argument("--disable_visualizer", action="store_true")
    parser.add_argument("--disable_robot_mesh", action="store_true")
    parser.add_argument("--vlm_model_name", type=str, default=TAMPConfiguration.vlm_model_name)
    parser.add_argument("--vlm_device", type=str, default=TAMPConfiguration.vlm_device)
    parser.add_argument("--vlm_dtype", type=str, default=TAMPConfiguration.vlm_dtype)
    parser.add_argument("--vlm_device_map", type=str, default=TAMPConfiguration.vlm_device_map)
    parser.add_argument(
        "--vlm_attention_implementation",
        type=str,
        default=TAMPConfiguration.vlm_attention_implementation,
    )
    parser.add_argument(
        "--vlm_quantization",
        choices=["none", "4bit", "8bit"],
        default=TAMPConfiguration.vlm_quantization,
    )
    parser.add_argument("--vlm_max_new_tokens", type=int, default=TAMPConfiguration.vlm_max_new_tokens)
    parser.add_argument("--vlm_max_time_sec", type=float, default=TAMPConfiguration.vlm_max_time_sec)
    parser.add_argument("--vlm_temperature", type=float, default=TAMPConfiguration.vlm_temperature)
    parser.add_argument("--vlm_do_sample", action="store_true")
    parser.add_argument("--vlm_max_reprompts", type=int, default=TAMPConfiguration.vlm_max_reprompts)
    parser.add_argument("--vlm_cache_dir", type=str, default=None)
    parser.add_argument("--disable_object_reduction", action="store_true")
    args = parser.parse_args()

    setup_logging()
    set_random_seed(args.seed)

    root = Path(args.experiment_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().isoformat(timespec="seconds").replace(":", "-")

    base_config = TAMPConfiguration(
        seed=args.seed,
        num_particles=args.num_particles,
        robot=args.robot,
        grasp_dof=args.grasp_dof,
        approach=args.approach,
        num_resampling_attempts=args.num_resampling_attempts,
        num_opt_steps=args.num_opt_steps,
        max_loop_dur=args.max_duration,
        num_initial_plans=args.num_initial_plans,
        task_plan_max_depth=args.task_plan_max_depth,
        cache_subgraphs=args.cache_subgraphs,
        curobo_plan=args.motion_plan,
        enable_visualizer=not args.disable_visualizer,
        viz_robot_mesh=not args.disable_robot_mesh,
        experiment_root=args.experiment_root,
        enable_vlm_tamp=True,
        vlm_model_name=args.vlm_model_name,
        vlm_device=args.vlm_device,
        vlm_dtype=args.vlm_dtype,
        vlm_device_map=args.vlm_device_map,
        vlm_attention_implementation=args.vlm_attention_implementation,
        vlm_quantization=args.vlm_quantization,
        vlm_max_new_tokens=args.vlm_max_new_tokens,
        vlm_max_time_sec=args.vlm_max_time_sec,
        vlm_temperature=args.vlm_temperature,
        vlm_do_sample=args.vlm_do_sample,
        vlm_max_reprompts=args.vlm_max_reprompts,
        vlm_render_style="pybullet_rgb",
        vlm_enable_object_reduction=not args.disable_object_reduction,
        vlm_cache_dir=args.vlm_cache_dir,
    )
    client = create_vlm_client(base_config)
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    task_rows = []
    for task_name, open_goal in _task_suite().items():
        run_config = replace(base_config, open_goal=open_goal)
        validate_tamp_config(run_config)
        experiment_id = f"{args.experiment_prefix}_{run_id}_{task_name}"
        env = load_mini_kitchen_env()
        start = time.perf_counter()
        result = run_vlm_tamp(
            env,
            run_config,
            cost_reducer,
            constraint_checker,
            experiment_id=experiment_id,
            vlm_client=client,
        )
        wall_time_sec = time.perf_counter() - start
        task_rows.append(_summarize_result(task_name, open_goal, wall_time_sec, result))

    success_count = sum(1 for row in task_rows if row["found_solution"])
    summary = {
        "env": args.env,
        "seed": args.seed,
        "num_tasks": len(task_rows),
        "success_count": success_count,
        "success_rate": success_count / len(task_rows) if task_rows else 0.0,
        "mean_wall_time_sec": sum(row["wall_time_sec"] for row in task_rows) / len(task_rows) if task_rows else None,
        "mean_subgoal_time_sec": (
            sum(row["mean_subgoal_time_sec"] for row in task_rows if row["mean_subgoal_time_sec"] is not None)
            / len([row for row in task_rows if row["mean_subgoal_time_sec"] is not None])
            if any(row["mean_subgoal_time_sec"] is not None for row in task_rows)
            else None
        ),
        "tasks": task_rows,
    }

    summary_path = root / f"{args.experiment_prefix}_{run_id}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[vlm-benchmark] wrote summary to {summary_path}")


if __name__ == "__main__":
    entrypoint()
