import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def _metric_total(timer_metrics: dict, name: str) -> Optional[float]:
    metric = timer_metrics.get(name)
    if metric is None:
        return None
    return float(metric["total"])


def _build_demo_command(args: argparse.Namespace, seed: int, experiment_id: str, enable_retrieval: bool) -> list[str]:
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
    if enable_retrieval:
        cmd.extend(
            [
                "--enable_retrieval",
                "--retrieval_root",
                args.retrieval_root,
                "--retrieval_num_particles",
                str(args.retrieval_num_particles or args.num_particles),
                "--retrieval_noise_scale",
                str(args.retrieval_noise_scale),
                "--retrieval_exact_env_tol",
                str(args.retrieval_exact_env_tol),
            ]
        )
    return cmd


def _run_case(args: argparse.Namespace, seed: int, experiment_id: str, enable_retrieval: bool) -> dict:
    cmd = _build_demo_command(args, seed=seed, experiment_id=experiment_id, enable_retrieval=enable_retrieval)
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    wall_time = time.perf_counter() - start

    exp_dir = Path(args.experiment_root) / experiment_id
    with open(exp_dir / "overall_metrics.json", "r") as f:
        overall_metrics = json.load(f)
    with open(exp_dir / "timer_metrics.json", "r") as f:
        timer_metrics = json.load(f)

    return {
        "experiment_id": experiment_id,
        "seed": seed,
        "retrieval_enabled": enable_retrieval,
        "wall_time_sec": wall_time,
        "found_solution": overall_metrics["found_solution"],
        "num_satisfying_final": overall_metrics["num_satisfying_final"],
        "retrieval_info": overall_metrics.get("retrieval"),
        "sample_initial_plans_sec": _metric_total(timer_metrics, "sample_initial_plans"),
        "start_optimization_sec": _metric_total(timer_metrics, "start_optimization"),
        "initialize_particles_sec": _metric_total(timer_metrics, "initialize_particles"),
    }


def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(statistics.mean(values))


def entrypoint():
    parser = argparse.ArgumentParser(
        description="Benchmark cuTAMP with and without retrieval warm starts on the same sequence of seeds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", default="tetris_3")
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--start_seed", type=int, default=1000)
    parser.add_argument("--experiment_root", type=str, default="logs/retrieval-benchmark")
    parser.add_argument("--retrieval_root", type=str, required=True)
    parser.add_argument("--experiment_prefix", type=str, default="benchmark")
    parser.add_argument("--num_particles", type=int, default=1024)
    parser.add_argument("--num_opt_steps", type=int, default=1000)
    parser.add_argument("--num_initial_plans", type=int, default=30)
    parser.add_argument("--approach", choices=["optimization", "sampling"], default="optimization")
    parser.add_argument("--num_resampling_attempts", type=int, default=100)
    parser.add_argument("--max_duration", type=float, default=None)
    parser.add_argument("--robot", choices=["panda", "ur5"], default="panda")
    parser.add_argument("--grasp_dof", choices=[4, 6], type=int, default=4)
    parser.add_argument("--retrieval_num_particles", type=int, default=None)
    parser.add_argument("--retrieval_noise_scale", type=float, default=0.0)
    parser.add_argument("--retrieval_exact_env_tol", type=float, default=1e-3)
    parser.add_argument("--motion_plan", action="store_true")
    parser.add_argument("--cache_subgraphs", action="store_true")
    parser.add_argument("--tuned_tetris_weights", action="store_true")
    parser.add_argument("--tetris_random_yaws", action="store_true")
    parser.add_argument("--enable_visualizer", action="store_true")
    parser.add_argument("--disable_robot_mesh", action="store_true")
    args = parser.parse_args()

    root = Path(args.experiment_root)
    root.mkdir(parents=True, exist_ok=True)

    trial_rows = []
    for offset in range(args.num_trials):
        seed = args.start_seed + offset
        baseline_id = f"{args.experiment_prefix}_baseline_{seed:04d}"
        retrieval_id = f"{args.experiment_prefix}_retrieval_{seed:04d}"

        baseline = _run_case(args, seed=seed, experiment_id=baseline_id, enable_retrieval=False)
        retrieval = _run_case(args, seed=seed, experiment_id=retrieval_id, enable_retrieval=True)
        row = {
            "seed": seed,
            "baseline": baseline,
            "retrieval": retrieval,
            "wall_time_speedup": baseline["wall_time_sec"] / retrieval["wall_time_sec"]
            if retrieval["wall_time_sec"] > 0
            else None,
        }
        trial_rows.append(row)
        print(
            f"[benchmark] seed={seed} baseline={baseline['wall_time_sec']:.2f}s "
            f"retrieval={retrieval['wall_time_sec']:.2f}s speedup={row['wall_time_speedup']:.3f}x"
        )

    baseline_times = [row["baseline"]["wall_time_sec"] for row in trial_rows]
    retrieval_times = [row["retrieval"]["wall_time_sec"] for row in trial_rows]
    speedups = [row["wall_time_speedup"] for row in trial_rows if row["wall_time_speedup"] is not None]

    summary = {
        "num_trials": len(trial_rows),
        "retrieval_root": args.retrieval_root,
        "baseline_wall_time_mean_sec": _mean(baseline_times),
        "retrieval_wall_time_mean_sec": _mean(retrieval_times),
        "mean_speedup_x": _mean(speedups),
        "baseline_solve_rate": sum(row["baseline"]["found_solution"] for row in trial_rows) / max(len(trial_rows), 1),
        "retrieval_solve_rate": sum(row["retrieval"]["found_solution"] for row in trial_rows) / max(len(trial_rows), 1),
        "trials": trial_rows,
    }

    summary_path = root / f"{args.experiment_prefix}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[benchmark] wrote summary to {summary_path}")


if __name__ == "__main__":
    entrypoint()
