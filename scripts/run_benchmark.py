"""Run the noise-mode benchmark sweep: modes x seeds, in parallel on CPU.

Each cell is a full SAC run (separate subprocess, capped threads so we can run
several at once). Writes a manifest the plotter consumes. Clean-eval returns are
logged by the SAC script itself to results/<run_name>.csv.

  uv run python scripts/run_benchmark.py --total-timesteps 60000 --jobs 4
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import tyro


@dataclass
class Args:
    env_id: str = "HalfCheetah-v5"
    modes: list[str] = field(default_factory=lambda: ["none", "uniform", "uniform-calibrated", "realistic"])
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    rho: float = 0.1
    noise_target: str = "obs"
    total_timesteps: int = 60_000
    learning_starts: int = 5_000
    eval_interval: int = 2_000
    eval_episodes: int = 5
    jobs: int = 4
    threads_per_job: int = 3
    outdir: str = "results"


def run_one(args: Args, mode: str, seed: int) -> dict:
    run_name = f"{args.env_id}__{mode}_{args.noise_target}_rho{args.rho}__{seed}"
    csv_path = os.path.join(args.outdir, f"{run_name}.csv")
    cmd = [
        sys.executable, "scripts/sac_continuous_action.py",
        "--env-id", args.env_id,
        "--noise-mode", mode,
        "--noise-target", args.noise_target,
        "--rho", str(args.rho),
        "--seed", str(seed),
        "--total-timesteps", str(args.total_timesteps),
        "--learning-starts", str(args.learning_starts),
        "--eval-interval", str(args.eval_interval),
        "--eval-episodes", str(args.eval_episodes),
        "--torch-threads", str(args.threads_per_job),
        "--csv-path", csv_path,
    ]
    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads_per_job),
           "MKL_NUM_THREADS": str(args.threads_per_job)}
    t0 = time.time()
    log_path = os.path.join(args.outdir, f"{run_name}.log")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED({proc.returncode})"
    print(f"[{status}] {mode} seed={seed}  {dt:.0f}s  -> {csv_path}")
    return {"mode": mode, "seed": seed, "csv": csv_path, "run_name": run_name,
            "returncode": proc.returncode, "seconds": round(dt)}


def main(args: Args):
    os.makedirs(args.outdir, exist_ok=True)
    jobs = [(m, s) for m in args.modes for s in args.seeds]
    print(f"Running {len(jobs)} SAC runs ({len(args.modes)} modes x {len(args.seeds)} seeds), "
          f"{args.total_timesteps} steps each, {args.jobs} in parallel.\n")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda js: run_one(args, *js), jobs))
    manifest = {"args": vars(args), "runs": results, "wall_seconds": round(time.time() - t0)}
    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    n_ok = sum(r["returncode"] == 0 for r in results)
    print(f"\nDone: {n_ok}/{len(results)} ok in {manifest['wall_seconds']}s. "
          f"Manifest -> {args.outdir}/manifest.json")


if __name__ == "__main__":
    main(tyro.cli(Args))
