# Correctness audit

Before trusting any benchmark number, the SAC + noise code was put through an
adversarial multi-agent audit: five independent reviewers (SAC algorithm, SAC
engineering, noise math, noise implementation, benchmark methodology), each flagged
issue then re-checked by a separate skeptic that had to *refute or confirm* it.
Result: 20 findings reviewed, **5 confirmed and fixed**, 15 dismissed as style/non-issues.

| # | Severity | Bug | Fix |
|---|---|---|---|
| 1 | critical | `OrnsteinUhlenbeck` discretization inflated the stationary std by `√(2/(2−θ))`, so "realistic" noise was *not* at level `rho` — defeats the project's calibration premise. | Exact AR(1): `y ← (1−θ)y + std·√(1−(1−θ)²)·ε`. Verified: realized std = 0.1000 for the rho=0.1 target. |
| 2 | critical | Observation-noise, action-noise, calibration, env and policy RNGs all derived from one seed → correlated noise streams. | Independent streams via `np.random.SeedSequence(seed).spawn(...)`. |
| 3 | major | `env.action_space` was never seeded, so the random-exploration phase differed run-to-run even with a fixed `--seed`. | `env.action_space.seed(args.seed)`. Verified: identical seeds → byte-identical runs. |
| 4 | major | Calibration used the *training* seed, so the per-channel noise **scale** silently varied across seeds and modes — performance gaps couldn't be attributed to the noise. | Fixed `calib_seed` (default 0), shared by all runs as a constant treatment. |
| 5 | minor | (Duplicate of #1, found independently by a second reviewer — same root cause, same fix.) | — |

The two critical fixes matter most: without #1 the noise wasn't actually calibrated,
and without #4 the comparison between seeds/modes wasn't controlled. Both are exactly
the kind of bug that produces plausible-but-wrong results.
