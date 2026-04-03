# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import argparse
from typing import Optional

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs import TAMPEnvironment
from cutamp.envs.book_shelf import load_book_shelf_env
from cutamp.envs.stick_button import load_stick_button_env
from cutamp.envs.tetris import load_tetris_env
from cutamp.envs.utils import get_env_dir, load_env

from cutamp.scripts.utils import (
    default_constraint_to_mult,
    default_constraint_to_tol,
    setup_logging,
    get_tetris_tuned_constraint_to_mult,
    set_random_seed,
)


def load_demo_env(name: str, tetris_random_yaws: bool = False) -> TAMPEnvironment:
    if name.startswith("tetris_"):
        num_blocks = int(name.split("tetris_")[-1])
        env = load_tetris_env(num_blocks, buffer_multiplier=1.0, random_yaws=tetris_random_yaws)
    elif name == "book_shelf":
        env = load_book_shelf_env()
    elif name == "stick_button":
        env = load_stick_button_env()
    elif name == "blocks":
        env_path = os.path.join(get_env_dir(), "obstacle_blocks_large_region.yml")
        env = load_env(env_path)
    elif name == "unpack":
        env_path = os.path.join(get_env_dir(), "unpack_3.yml")
        env = load_env(env_path)
    else:
        raise ValueError(f"Unknown environment name: {name}")
    return env


def cutamp_demo(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    experiment_id: Optional[str] = None,
    use_tetris_tuned_weights: bool = False,
):
    setup_logging()
    set_random_seed(config.seed)

    constraint_to_mult = (
        get_tetris_tuned_constraint_to_mult() if use_tetris_tuned_weights else default_constraint_to_mult.copy()
    )
    cost_reducer = CostReducer(constraint_to_mult)
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    run_cutamp(env, config, cost_reducer, constraint_checker, experiment_id=experiment_id)


def entrypoint():
    parser = argparse.ArgumentParser(
        description="Run cuTAMP demo. We do not expose all the configs so check cutamp/config.py for additional configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--env",
        help="Environment name to run",
        default="tetris_3",
        choices=["tetris_1", "tetris_2", "tetris_3", "tetris_5", "book_shelf", "stick_button", "blocks", "unpack"],
    )
    parser.add_argument(
        "-n", "--num_particles", type=int, default=1024, help="Number of particles to use (i.e. batch size)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")

    # Soft costs
    parser.add_argument(
        "--optimize_soft_costs", action="store_true", help="Whether to optimize soft costs (default: False)"
    )
    parser.add_argument(
        "--soft_cost",
        choices=["dist_from_origin", "max_obj_dist", "min_obj_dist", "min_y", "max_y", "align_yaw"],
        help="Soft cost to optimize or minimize. If used without --optimize_soft_costs, will be used to score the particles.",
    )

    # Robot and grasp
    parser.add_argument("--robot", default="panda", choices=["panda", "ur5"], help="Robot to use")
    parser.add_argument(
        "--grasp_dof",
        type=int,
        default=4,
        choices=[4, 6],
        help="Grasp DOF to use. 6-DOF is really only supported for the book_shelf environment.",
    )
    parser.add_argument(
        "--tetris_random_yaws",
        action="store_true",
        help="Randomize initial yaw of Tetris blocks. Useful for retrieval data collection and benchmarking.",
    )

    # Approach
    parser.add_argument(
        "--approach",
        default="optimization",
        choices=["optimization", "sampling"],
        help="Approach to use. Optimization is cuTAMP (sampling + optimization), while sampling is just resampling."
        "If using sampling, you may need to modify the --num_resampling_attempts.",
    )
    parser.add_argument(
        "--num_resampling_attempts",
        type=int,
        default=100,
        help="Number of resampling attempts per skeleton if using sampling approach.",
    )
    parser.add_argument(
        "--num_opt_steps", type=int, default=1000, help="Number of optimization steps to run for each skeleton."
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=None,
        help="Maximum duration for optimization or sampling in seconds. Overrides --num_resampling_attempts and --num_opt_steps if set.",
    )
    parser.add_argument(
        "--num_initial_plans", type=int, default=30, help="Number of initial plans to sample with task planner."
    )
    parser.add_argument("--cache_subgraphs", action="store_true", help="Whether to cache subgraph samples for reuse.")
    parser.add_argument(
        "--motion_plan",
        action="store_true",
        help="Whether to plan for full motions after using cuTAMP. Not supported in stick_button domain yet.",
    )

    # Visualization and logging
    parser.add_argument(
        "--disable_visualizer",
        action="store_true",
        help="Disable the rerun visualizer. Note if you want accurate timing information, you should disable the visualizer.",
    )
    parser.add_argument(
        "--viz_interval", type=int, default=10, help="Interval for visualizing optimization state in steps."
    )
    parser.add_argument(
        "--disable_robot_mesh",
        action="store_true",
        help="Disable robot mesh visualization to save visualization bandwidth.",
    )
    parser.add_argument(
        "--experiment_root",
        type=str,
        default="logs/cutamp-experiments",
        help="Root directory for experiment logging.",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Experiment ID for logging. Results will be saved in <experiment_root>/<experiment_id>",
    )
    parser.add_argument(
        "--enable_retrieval",
        action="store_true",
        help="Warm start particle initialization from prior successful runs in the retrieval root.",
    )
    parser.add_argument(
        "--retrieval_root",
        type=str,
        default=None,
        help="Root directory to search for retrieval artifacts. Defaults to --experiment_root when omitted.",
    )
    parser.add_argument(
        "--retrieval_num_particles",
        type=int,
        default=TAMPConfiguration.retrieval_num_particles,
        help="Number of particles to warm start for approximate retrieval matches. Exact matches may expand to the full batch.",
    )
    parser.add_argument(
        "--retrieval_max_env_distance",
        type=float,
        default=TAMPConfiguration.retrieval_max_env_distance,
        help="Maximum environment distance allowed for approximate retrieval matches. Use a negative value to disable the limit.",
    )
    parser.add_argument(
        "--retrieval_noise_scale",
        type=float,
        default=TAMPConfiguration.retrieval_noise_scale,
        help="Gaussian noise scale applied to approximate retrieved particles.",
    )
    parser.add_argument(
        "--retrieval_exact_env_tol",
        type=float,
        default=TAMPConfiguration.retrieval_exact_env_tol,
        help="Environment pose-distance threshold used to treat a retrieved task as an exact match.",
    )
    parser.add_argument(
        "--retrieval_min_approx_saved_particles",
        type=int,
        default=TAMPConfiguration.retrieval_min_approx_saved_particles,
        help="Minimum number of saved particles required before using an approximate retrieval match.",
    )
    parser.add_argument(
        "--retrieval_approx_movable_yaw_weight",
        type=float,
        default=TAMPConfiguration.retrieval_approx_movable_yaw_weight,
        help="Yaw weight for movable objects when scoring approximate retrieval matches.",
    )
    parser.add_argument(
        "--retrieval_approx_static_yaw_weight",
        type=float,
        default=TAMPConfiguration.retrieval_approx_static_yaw_weight,
        help="Yaw weight for static objects when scoring approximate retrieval matches.",
    )
    parser.add_argument(
        "--retrieval_num_saved_particles",
        type=int,
        default=TAMPConfiguration.retrieval_num_saved_particles,
        help="Number of particles to persist in each retrieval artifact for future warm starts.",
    )

    # Tetris tuned weights
    parser.add_argument(
        "--tuned_tetris_weights", action="store_true", help="Use weights tuned on tetris_5 for constraint multipliers."
    )

    # We only expose a subset of the full TAMPConfiguration. Check config.py for the full configuration.
    args = parser.parse_args()
    retrieval_max_env_distance = args.retrieval_max_env_distance
    if retrieval_max_env_distance is not None and retrieval_max_env_distance < 0:
        retrieval_max_env_distance = None

    config = TAMPConfiguration(
        seed=args.seed,
        num_particles=args.num_particles,
        robot=args.robot,
        grasp_dof=args.grasp_dof,
        approach=args.approach,
        num_resampling_attempts=args.num_resampling_attempts,
        num_opt_steps=args.num_opt_steps,
        max_loop_dur=args.max_duration,
        optimize_soft_costs=args.optimize_soft_costs,
        soft_cost=args.soft_cost,
        num_initial_plans=args.num_initial_plans,
        cache_subgraphs=args.cache_subgraphs,
        curobo_plan=args.motion_plan,
        enable_visualizer=not args.disable_visualizer,
        opt_viz_interval=args.viz_interval,
        viz_robot_mesh=not args.disable_robot_mesh,
        experiment_root=args.experiment_root,
        enable_retrieval=args.enable_retrieval,
        retrieval_root=args.retrieval_root,
        retrieval_num_particles=args.retrieval_num_particles,
        retrieval_max_env_distance=retrieval_max_env_distance,
        retrieval_min_approx_saved_particles=args.retrieval_min_approx_saved_particles,
        retrieval_approx_movable_yaw_weight=args.retrieval_approx_movable_yaw_weight,
        retrieval_approx_static_yaw_weight=args.retrieval_approx_static_yaw_weight,
        retrieval_noise_scale=args.retrieval_noise_scale,
        retrieval_exact_env_tol=args.retrieval_exact_env_tol,
        retrieval_num_saved_particles=args.retrieval_num_saved_particles,
    )
    validate_tamp_config(config)

    # Load env and run demo
    env = load_demo_env(args.env, tetris_random_yaws=args.tetris_random_yaws)
    cutamp_demo(env, config, experiment_id=args.experiment_id, use_tetris_tuned_weights=args.tuned_tetris_weights)


if __name__ == "__main__":
    entrypoint()
