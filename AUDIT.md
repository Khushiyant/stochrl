# Correctness audit

Before trusting any benchmark number, the SAC + noise code went through a
correctness review covering the SAC algorithm, the noise math and
implementation, and the benchmark methodology. Of 20 candidate issues, 4 were
confirmed and fixed; the rest were dismissed as style or non-issues.

| # | Severity | Bug | Fix |
|---|---|---|---|
| 1 | critical | `OrnsteinUhlenbeck` discretization inflated the stationary std by `√(2/(2−θ))`, so "realistic" noise was not at level `rho`, defeating the calibration premise. | Exact AR(1): `y ← (1−θ)y + std·√(1−(1−θ)²)·ε`. Verified: realized std = 0.1000 for the rho=0.1 target. |
| 2 | critical | Observation-noise, action-noise, calibration, env and policy RNGs were all derived from one seed, giving correlated noise streams. | Independent streams via `np.random.SeedSequence(seed).spawn(...)`. |
| 3 | major | `env.action_space` was never seeded, so the random-exploration phase differed run-to-run even with a fixed `--seed`. | `env.action_space.seed(args.seed)`. Verified: identical seeds give byte-identical runs. |
| 4 | major | Calibration used the training seed, so the per-channel noise scale silently varied across seeds and modes and performance gaps couldn't be attributed to the noise. | Fixed `calib_seed` (default 0), shared by all runs as a constant treatment. |

Two of the fixes matter most: without #1 the noise wasn't actually
calibrated, and without #4 the comparison between seeds and modes wasn't
controlled. Both are the kind of bug that produces plausible-but-wrong results.
