# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import argparse
import cProfile
import contextlib
import logging
import os
from typing import Optional

import torch

from curobo._src.util.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import DeviceCfg
from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs import TAMPEnvironment, get_env_dir, load_env
from cutamp.robots import RobotContainer
from .envs import load_book_shelf_env, load_stick_button_env, load_tetris_env
from .utils import (
    default_constraint_to_mult,
    default_constraint_to_tol,
    setup_logging,
    get_tetris_tuned_constraint_to_mult,
)

_log = logging.getLogger(__name__)


_ROBOT_TO_CUROBO_CONFIG = {
    "franka_panda": "franka.yml",
    "franka_robotiq_2f_85": "franka_robotiq_2f_85.yml",
    "franka_robotiq_2f_140": "franka_robotiq_2f_140.yml",
    "ur5_robotiq_2f_85": "ur5_robotiq_2f_85.yml",
    "ur5_robotiq_2f_140": "ur5_robotiq_2f_140.yml",
    "xarm7": "xarm7.yml",
}


def load_demo_env(name: str) -> TAMPEnvironment:
    if name.startswith("tetris_"):
        num_blocks = int(name.split("tetris_")[-1])
        env = load_tetris_env(num_blocks, buffer_multiplier=1.0)
    elif name == "book_shelf":
        env = load_book_shelf_env()
    elif name == "stick_button":
        env = load_stick_button_env()
    elif name == "blocks":
        env_path = os.path.join(get_env_dir(), "obstacle_blocks_large_region.yml")
        env = load_env(env_path)
    elif name == "blocks_rotate":
        env_path = os.path.join(get_env_dir(), "obstacle_blocks_rotated_region.yml")
        env = load_env(env_path)
    elif name == "blocks_tight":
        env_path = os.path.join(get_env_dir(), "obstacle_blocks_tight_region.yml")
        env = load_env(env_path)
    elif name == "unpack":
        env_path = os.path.join(get_env_dir(), "unpack_3.yml")
        env = load_env(env_path)
    elif name == "blocks_5":
        env_path = os.path.join(get_env_dir(), "blocks_5.yml")
        env = load_env(env_path)
    else:
        raise ValueError(f"Unknown environment name: {name}")
    return env


def load_demo_robot(robot_name: str) -> tuple[RobotContainer, torch.Tensor]:
    """Load the cuRobo robot model expected by the core cuTAMP API."""
    robot_config = _ROBOT_TO_CUROBO_CONFIG.get(robot_name, f"{robot_name}.yml")
    device_cfg = DeviceCfg()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_config))
    kinematics_cfg = KinematicsCfg.from_robot_yaml_file(robot_cfg, device_cfg=device_cfg)
    kinematics = Kinematics(kinematics_cfg)
    joint_limits = kinematics.get_joint_limits().position
    q_init = kinematics.default_joint_position.reshape(-1)

    tool_from_ee = torch.eye(4, device=device_cfg.device, dtype=device_cfg.dtype)
    if robot_name.startswith("franka"):
        # Match the upstream Panda hand frame used by the built-in cuTAMP samplers.
        tool_from_ee[:3, :3] = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            device=device_cfg.device,
            dtype=device_cfg.dtype,
        )
        tool_from_ee[:3, 3] = torch.tensor([0.0, 0.0, 0.105], device=device_cfg.device, dtype=device_cfg.dtype)
    robot = RobotContainer(
        name=robot_name,
        kinematics=kinematics,
        joint_names=tuple(kinematics.joint_names),
        tool_frame=kinematics.tool_frames[0],
        joint_limits=joint_limits,
        gripper_spheres=kinematics.robot_spheres,
        tool_from_ee=tool_from_ee,
        robot_cfg=robot_cfg,
    )
    return robot, q_init


def cutamp_demo(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    experiment_id: Optional[str] = None,
    use_tetris_tuned_weights: bool = False,
):
    setup_logging()

    constraint_to_mult = (
        get_tetris_tuned_constraint_to_mult() if use_tetris_tuned_weights else default_constraint_to_mult.copy()
    )
    cost_reducer = CostReducer(constraint_to_mult)
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
    robot, q_init = load_demo_robot(config.robot)

    curobo_plan, _, failure_reason = run_cutamp(
        env,
        config,
        robot,
        cost_reducer,
        constraint_checker,
        q_init=q_init,
        experiment_id=experiment_id,
    )
    if failure_reason is not None:
        _log.warning(f"No plan found: {failure_reason}")
    elif config.curobo_plan and curobo_plan is None:
        _log.warning("Satisfying particles found, but no cuRobo motion plan was returned")


def entrypoint():
    parser = argparse.ArgumentParser(
        description="Run cuTAMP demo. We do not expose all the configs so check cutamp/config.py for additional configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--env",
        help="Environment name to run",
        default="tetris_3",
        choices=[
            "tetris_1",
            "tetris_2",
            "tetris_3",
            "tetris_5",
            "book_shelf",
            "stick_button",
            "blocks",
            "blocks_rotate",
            "blocks_tight",
            "unpack",
            "blocks_5",
        ],
    )
    parser.add_argument(
        "-n", "--num_particles", type=int, default=2000, help="Number of particles to use (i.e. batch size)"
    )

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
    parser.add_argument(
        "--robot",
        default="franka_panda",
        choices=[
            "franka_panda",
            "franka_robotiq_2f_85",
            "franka_robotiq_2f_140",
            "ur5_robotiq_2f_85",
            "ur5_robotiq_2f_140",
            "xarm7",
        ],
        help="Robot to use",
    )
    parser.add_argument(
        "--grasp_dof",
        type=int,
        default=4,
        choices=[4, 6],
        help="Grasp DOF to use. 6-DOF is really only supported for the book_shelf environment.",
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
        "--lr",
        type=float,
        default=None,
        help="Optimizer learning rate for action parameters. Defaults to TAMPConfiguration.lr.",
    )
    parser.add_argument(
        "--conf_lr",
        type=float,
        default=None,
        help="Optimizer learning rate for robot configurations. Defaults to TAMPConfiguration.conf_lr.",
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
    parser.add_argument(
        "--max_motion_refine_attempts",
        type=int,
        default=None,
        help="Max satisfying particles to try motion refinement on per skeleton. None = try all.",
    )
    parser.add_argument(
        "--motion-refinement-mode",
        choices=("ee_strict", "joint"),
        default="joint",
        help="Optional override for cuTAMP motion refinement mode.",
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
        "--experiment_root", type=str, default="cutamp-experiments", help="Root directory for experiment logging."
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default=None,
        help="Experiment ID for logging. Results will be saved in <experiment_root>/<experiment_id>",
    )

    # Object collision spheres and placement
    parser.add_argument("--coll_n_spheres", type=int, default=50, help="Number of collision spheres per object.")
    parser.add_argument(
        "--placement_shrink_dist",
        type=float,
        default=0.0,
        help="Shrink distance for placement validity check (meters). Larger values enforce a bigger "
        "margin from surface edges (harder to satisfy but more robust). Tight-placement envs like "
        "blocks_tight require 0.0; tetris_3 yields more satisfying particles at 0.02.",
    )
    parser.add_argument(
        "--prop_satisfying_break",
        type=float,
        default=0.1,
        help="Break optimization when this proportion of particles satisfy constraints. Set to 0 to disable.",
    )

    # Tetris tuned weights
    parser.add_argument(
        "--tuned_tetris_weights", action="store_true", help="Use weights tuned on tetris_5 for constraint multipliers."
    )

    # Profiling
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling with cProfile. Output can be visualized with snakeviz.",
    )
    parser.add_argument(
        "--profile_output",
        type=str,
        default="cutamp_profile.prof",
        help="Output file for profiling data.",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Enable GPU profiling with torch.profiler. Outputs a Chrome trace JSON viewable in chrome://tracing.",
    )
    parser.add_argument(
        "--torch-profile-output",
        type=str,
        default="cutamp_torch_trace.json",
        help="Output file for torch profiler Chrome trace.",
    )

    # We only expose a subset of the full TAMPConfiguration. Check config.py for the full configuration.
    args = parser.parse_args()
    config_kwargs = dict(
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
        max_motion_refine_attempts=args.max_motion_refine_attempts,
        enable_visualizer=not args.disable_visualizer,
        opt_viz_interval=args.viz_interval,
        viz_robot_mesh=not args.disable_robot_mesh,
        experiment_root=args.experiment_root,
        coll_n_spheres=args.coll_n_spheres,
        # Note: these are new features with this fork of cuTAMP
        placement_check="obb",
        placement_shrink_dist=args.placement_shrink_dist,
        prop_satisfying_break=args.prop_satisfying_break if args.prop_satisfying_break > 0 else None,
    )
    if args.lr is not None:
        config_kwargs["lr"] = args.lr
    if args.conf_lr is not None:
        config_kwargs["conf_lr"] = args.conf_lr
    if args.motion_refinement_mode is not None:
        config_kwargs["motion_refinement_mode"] = args.motion_refinement_mode
    config = TAMPConfiguration(**config_kwargs)
    validate_tamp_config(config)

    # Load env and run demo
    env = load_demo_env(args.env)

    # Profiling setup — cProfile and torch profiler can run simultaneously
    cprofile = None
    if args.profile:
        print(f"cProfile enabled. Output will be saved to {args.profile_output}")
        cprofile = cProfile.Profile()

    torch_profiler = None
    if args.torch_profile:
        print(f"Torch profiling enabled. Trace will be saved to {args.torch_profile_output}")
        torch_profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    try:
        if cprofile is not None:
            cprofile.enable()
        with torch_profiler if torch_profiler is not None else contextlib.nullcontext():
            cutamp_demo(
                env,
                config,
                experiment_id=args.experiment_id,
                use_tetris_tuned_weights=args.tuned_tetris_weights,
            )
    finally:
        if cprofile is not None:
            cprofile.disable()
            cprofile.dump_stats(args.profile_output)
            print(f"cProfile results saved to {args.profile_output}")
        if torch_profiler is not None:
            torch_profiler.export_chrome_trace(args.torch_profile_output)
            print(f"Torch profile trace saved to {args.torch_profile_output}")
            print("\n" + torch_profiler.key_averages().table(sort_by="cuda_time_total", row_limit=30))


if __name__ == "__main__":
    entrypoint()
