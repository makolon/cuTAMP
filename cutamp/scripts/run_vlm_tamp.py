# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""CLI for VLM-guided cuTAMP experiments."""

from __future__ import annotations

import argparse

from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.book_shelf import load_book_shelf_env
from cutamp.envs.mini_kitchen import load_mini_kitchen_env
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol, set_random_seed, setup_logging
from cutamp.vlm_tamp import run_vlm_tamp


def entrypoint():
    parser = argparse.ArgumentParser(
        description="Run the VLM-guided cuTAMP pipeline on supported tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="book_shelf", choices=["book_shelf", "mini_kitchen"], help="Environment to run.")
    parser.add_argument("--open_goal", required=True, help="Natural language instruction given to the VLM.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    parser.add_argument("--include_obstacle", action="store_true", help="Include the optional obstacle book in shelf.")

    parser.add_argument("-n", "--num_particles", type=int, default=1024, help="Number of particles.")
    parser.add_argument("--num_initial_plans", type=int, default=30, help="Number of plan skeletons to sample.")
    parser.add_argument(
        "--task_plan_max_depth",
        type=int,
        default=TAMPConfiguration.task_plan_max_depth,
        help="Maximum symbolic search depth for each subgoal planning problem.",
    )
    parser.add_argument("--num_opt_steps", type=int, default=1000, help="Maximum optimization steps per skeleton.")
    parser.add_argument("--max_duration", type=float, default=None, help="Maximum wall-clock duration per subplan.")
    parser.add_argument("--robot", default="panda", choices=["panda", "ur5"], help="Robot embodiment.")
    parser.add_argument("--grasp_dof", type=int, default=6, choices=[4, 6], help="Grasp parameterization DOF.")
    parser.add_argument("--num_resampling_attempts", type=int, default=100, help="Sampling baseline retries.")
    parser.add_argument(
        "--approach",
        default="optimization",
        choices=["optimization", "sampling"],
        help="Planning backend used for each subgoal.",
    )
    parser.add_argument("--cache_subgraphs", action="store_true", help="Enable cuTAMP subgraph caching.")
    parser.add_argument("--motion_plan", action="store_true", help="Run cuRobo motion planning after successful TAMP.")

    parser.add_argument("--disable_visualizer", action="store_true", help="Disable Rerun visualization.")
    parser.add_argument("--viz_interval", type=int, default=10, help="Visualization interval in optimizer steps.")
    parser.add_argument("--disable_robot_mesh", action="store_true", help="Disable robot mesh visualization.")
    parser.add_argument(
        "--experiment_root",
        type=str,
        default="logs/vlm-tamp-experiments",
        help="Root directory used for top-level VLM-TAMP logs.",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Experiment identifier. Results are saved to <experiment_root>/<experiment_id>/.",
    )

    parser.add_argument("--vlm_model_name", type=str, default=TAMPConfiguration.vlm_model_name, help="HF model id.")
    parser.add_argument("--vlm_device", type=str, default=TAMPConfiguration.vlm_device, help="VLM device.")
    parser.add_argument("--vlm_dtype", type=str, default=TAMPConfiguration.vlm_dtype, help="VLM dtype.")
    parser.add_argument(
        "--vlm_device_map",
        type=str,
        default=TAMPConfiguration.vlm_device_map,
        help="Optional Transformers device_map for the VLM model.",
    )
    parser.add_argument(
        "--vlm_attention_implementation",
        type=str,
        default=TAMPConfiguration.vlm_attention_implementation,
        help="Optional attention implementation string forwarded to Transformers.",
    )
    parser.add_argument(
        "--vlm_quantization",
        choices=["none", "4bit", "8bit"],
        default=TAMPConfiguration.vlm_quantization,
        help="Optional quantization mode for loading the VLM.",
    )
    parser.add_argument(
        "--vlm_max_new_tokens",
        type=int,
        default=TAMPConfiguration.vlm_max_new_tokens,
        help="Maximum generated tokens per VLM call.",
    )
    parser.add_argument(
        "--vlm_max_time_sec",
        type=float,
        default=TAMPConfiguration.vlm_max_time_sec,
        help="Optional max_time value forwarded to model.generate.",
    )
    parser.add_argument(
        "--vlm_temperature",
        type=float,
        default=TAMPConfiguration.vlm_temperature,
        help="Sampling temperature for VLM generation.",
    )
    parser.add_argument(
        "--vlm_do_sample",
        action="store_true",
        help="Enable sampling for the VLM instead of deterministic decoding.",
    )
    parser.add_argument(
        "--vlm_max_reprompts",
        type=int,
        default=TAMPConfiguration.vlm_max_reprompts,
        help="Maximum number of reprompt rounds after the initial query.",
    )
    parser.add_argument(
        "--vlm_cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for VLM requests.",
    )
    parser.add_argument(
        "--vlm_render_style",
        choices=["pybullet_rgb", "sim_3d", "simple_annotated"],
        default=None,
        help="Image style used for VLM scene rendering. Defaults to pybullet_rgb for mini_kitchen.",
    )
    parser.add_argument(
        "--disable_object_reduction",
        action="store_true",
        help="Disable object reduction for sequential subgoal planning.",
    )

    args = parser.parse_args()

    setup_logging()
    set_random_seed(args.seed)

    default_render_style = "pybullet_rgb" if args.env == "mini_kitchen" else TAMPConfiguration.vlm_render_style
    render_style = args.vlm_render_style or default_render_style

    config = TAMPConfiguration(
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
        opt_viz_interval=args.viz_interval,
        viz_robot_mesh=not args.disable_robot_mesh,
        experiment_root=args.experiment_root,
        enable_vlm_tamp=True,
        open_goal=args.open_goal,
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
        vlm_render_style=render_style,
        vlm_enable_object_reduction=not args.disable_object_reduction,
        vlm_cache_dir=args.vlm_cache_dir,
    )
    validate_tamp_config(config)

    env = load_book_shelf_env(include_obstacle=args.include_obstacle) if args.env == "book_shelf" else load_mini_kitchen_env()
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
    run_vlm_tamp(
        env,
        config,
        cost_reducer,
        constraint_checker,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    entrypoint()
