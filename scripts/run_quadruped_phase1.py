"""Phase-1 quadruped-run: 5 conditions x 5 seeds x 1M steps, single rolling pool of K.

One pool across all 25 heterogeneous (condition, seed) runs so K stay in flight
continuously (no barriered waves). Crash-safe: each run checkpoints every 50k and
can be resumed by re-running with RELOAD=True.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ENV = "dm_control/quadruped-run-v0"
STEPS = 1_000_000
K = 4                       # shared box: ~6 GB used by others + swap full -> 4 x 1.4 GB keeps ~3-4 GB headroom
SEEDS = [1, 2, 3, 4, 5]
RELOAD = os.environ.get("RELOAD", "0") == "1"
CONDS = [  # (name, noise_mode, noise_target, rho)
    ("clean",      "none",               "obs",        0.0),
    ("obs_r005",   "uniform-calibrated", "obs",        0.05),
    ("obs_r010",   "uniform-calibrated", "obs",        0.10),
    ("trans_r005", "uniform-calibrated", "transition", 0.05),
    ("trans_r010", "uniform-calibrated", "transition", 0.10),
]


def run_one(cond, seed):
    name, mode, target, rho = cond
    outdir, ckdir = f"results/quadruped/{name}", f"checkpoints/quadruped/{name}"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(ckdir, exist_ok=True)
    csv, ckpt = f"{outdir}/seed{seed}.csv", f"{ckdir}/seed{seed}.pt"
    cmd = [sys.executable, "scripts/sac_continuous_action.py",
           "--env-id", ENV, "--noise-mode", mode, "--noise-target", target,
           "--rho", str(rho), "--seed", str(seed),
           "--total-timesteps", str(STEPS), "--learning-starts", "5000",
           "--eval-interval", "25000", "--eval-episodes", "3", "--torch-threads", "1",
           "--csv-path", csv, "--checkpoint-path", ckpt, "--checkpoint-interval", "50000"]
    if RELOAD and os.path.exists(ckpt):
        cmd.append("--reloading")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    t0 = time.time()
    with open(f"{outdir}/seed{seed}.log", "w") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"[{'ok' if rc == 0 else 'FAIL ' + str(rc)}] {name} seed{seed}  {time.time()-t0:.0f}s", flush=True)
    return rc


jobs = [(c, s) for c in CONDS for s in SEEDS]
print(f"quadruped phase-1: {len(jobs)} runs, K={K} in flight, {STEPS} steps each, reload={RELOAD}", flush=True)
t0 = time.time()
with ThreadPoolExecutor(max_workers=K) as pool:
    res = list(pool.map(lambda a: run_one(*a), jobs))
print(f"DONE {sum(r == 0 for r in res)}/{len(res)} ok in {time.time()-t0:.0f}s", flush=True)
