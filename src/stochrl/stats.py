from __future__ import annotations

import numpy as np


def iqm(x) -> float:
    """Interquartile mean (Agarwal et al. 2021); plain mean for n < 4."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    k = int(n * 0.25)
    core = x[k:n - k] if n - 2 * k > 0 else x
    return float(np.mean(core))


def estimator_name(n: int) -> str:
    """Figure label for what iqm() computes at n samples."""
    return "IQM" if int(n * 0.25) >= 1 else "mean"


def bootstrap_ci(x, agg=iqm, reps: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """(point, lo, hi) percentile bootstrap; anti-conservative at small n."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([agg(rng.choice(x, size=len(x), replace=True)) for _ in range(reps)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return agg(x), float(lo), float(hi)
