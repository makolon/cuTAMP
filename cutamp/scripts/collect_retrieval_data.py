import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _build_demo_command(args: argparse.Namespace, seed: int, experiment_id: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "cutamp.scripts.run_cutamp",
        "--env",
        args.env,
        "--seed",
        str(seed),
        "--experiment_root",
        args.experiment_root,
        "--experiment_id",
        experiment_id,
        "--num_particles",
        str(args.num_particles),
        "--num_opt_steps",
        str(args.num_opt_steps),
        "--num_initial_plans",
        str(args.num_initial_plans),
        "--approach",
        args.approach,
        "--robot",
        args.robot,
        "--grasp_dof",
        str(args.grasp_dof),
    ]
    if args.max_duration is not None:
        cmd.extend(["--max_duration", str(args.max_duration)])
    if args.num_resampling_attempts is not None:
        cmd.extend(["--num_resampling_attempts", str(args.num_resampling_attempts)])
    if args.motion_plan:
        cmd.append("--motion_plan")
    if args.cache_subgraphs:
        cmd.append("--cache_subgraphs")
    if args.tuned_tetris_weights:
        cmd.append("--tuned_tetris_weights")
    if args.tetris_random_yaws:
        cmd.append("--tetris_random_yaws")
    if not args.enable_visualizer:
        cmd.append("--disable_visualizer")
    if args.disable_robot_mesh:
        cmd.append("--disable_robot_mesh")
    return cmd


def entrypoint():
    parser = argparse.ArgumentParser(
        description="Run multiple cuTAMP experiments to collect retrieval artifacts from successful solves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="tetris_3")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--experiment_root", type=str, default="logs/retrieval-data")
    parser.add_argument("--experiment_prefix", type=str, default="collect")
    parser.add_argument("--num_particles", type=int, default=1024)
    parser.add_argument("--num_opt_steps", type=int, default=1000)
    parser.add_argument("--num_initial_plans", type=int, default=30)
    parser.add_argument("--approach", choices=["optimization", "sampling"], default="optimization")
    parser.add_argument("--num_resampling_attempts", type=int, default=100)
    parser.add_argument("--max_duration", type=float, default=None)
    parser.add_argument("--robot", choices=["panda", "ur5"], default="panda")
    parser.add_argument("--grasp_dof", choices=[4, 6], type=int, default=4)
    parser.add_argument("--motion_plan", action="store_true")
    parser.add_argument("--cache_subgraphs", action="store_true")
    parser.add_argument("--tuned_tetris_weights", action="store_true")
    parser.add_argument("--tetris_random_yaws", action="store_true")
    parser.add_argument("--enable_visualizer", action="store_true")
    parser.add_argument("--disable_robot_mesh", action="store_true")
    args = parser.parse_args()

    root = Path(args.experiment_root)
    root.mkdir(parents=True, exist_ok=True)

    summary = {"runs": []}
    for offset in range(args.num_runs):
        seed = args.start_seed + offset
        experiment_id = f"{args.experiment_prefix}_{seed:04d}"
        cmd = _build_demo_command(args, seed=seed, experiment_id=experiment_id)
        start = time.perf_counter()
        subprocess.run(cmd, check=True)
        wall_time = time.perf_counter() - start

        exp_dir = root / experiment_id
        overall_path = exp_dir / "overall_metrics.json"
        artifact_path = exp_dir / "retrieval" / "artifact.json"
        with open(overall_path, "r") as f:
            overall_metrics = json.load(f)

        summary["runs"].append(
            {
                "seed": seed,
                "experiment_id": experiment_id,
                "wall_time_sec": wall_time,
                "found_solution": overall_metrics["found_solution"],
                "num_satisfying_final": overall_metrics["num_satisfying_final"],
                "artifact_exists": artifact_path.exists(),
                "artifact_path": str(artifact_path) if artifact_path.exists() else None,
            }
        )
        print(
            f"[collect] seed={seed} found_solution={overall_metrics['found_solution']} "
            f"artifact={artifact_path.exists()} wall_time={wall_time:.2f}s"
        )

    solved = [run for run in summary["runs"] if run["found_solution"]]
    summary["num_runs"] = len(summary["runs"])
    summary["num_solved"] = len(solved)
    summary["solve_rate"] = len(solved) / max(len(summary["runs"]), 1)

    summary_path = root / f"{args.experiment_prefix}_collection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[collect] wrote summary to {summary_path}")


if __name__ == "__main__":
    entrypoint()
