"""Aggregate statistics for RL evaluation (Agarwal et al., 2021, "rliable").

Point estimates use the interquartile mean (IQM) — the mean of the middle 50% of
runs — which is far less sensitive to outlier seeds than the mean and more
efficient than the median. Uncertainty is a percentile bootstrap confidence
interval over seeds. With few seeds the IQM degrades gracefully to the mean and
the CI is (honestly) wide.
"""

from __future__ import annotations

import numpy as np


def iqm(x) -> float:
    """Interquartile mean: mean of the middle 50% of values (trims 25% each tail)."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    k = int(n * 0.25)  # integer trim; for n < 4 this is 0 -> plain mean
    core = x[k:n - k] if n - 2 * k > 0 else x
    return float(np.mean(core))


def estimator_name(n: int) -> str:
    """What iqm() actually computes for n samples: 'IQM' once it trims (n>=4), else 'mean'.

    Lets callers label figures honestly instead of claiming IQM when it reduces to
    the plain mean at small n.
    """
    return "IQM" if int(n * 0.25) >= 1 else "mean"


def bootstrap_ci(x, agg=iqm, reps: int = 10_000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for an aggregate over samples (seeds).

    Returns (point_estimate, ci_low, ci_high) at the (1-alpha) level. Note: the
    percentile bootstrap is anti-conservative at small n (empirically ~75% coverage
    at n=3, ~90% at n=10 for a nominal 95% interval); treat CIs as indicative until
    seeds are plentiful (>=~10).
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boots = np.array([agg(rng.choice(x, size=len(x), replace=True)) for _ in range(reps)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return agg(x), float(lo), float(hi)
