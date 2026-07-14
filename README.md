# StochRL

A benchmark for reinforcement learning under realistic sensor noise, built on Soft Actor Critic and MuJoCo.

Most studies that test RL under noise add the same Gaussian jitter to every sensor. Real sensors do not behave that way. Some are noisier than others, some get worse in certain situations, and the errors are not always clean bell curves. StochRL adds noise the way real hardware does. It scales the noise to each sensor, lets it depend on what the robot is doing, and measures how much that changes what SAC learns.

The main experiments run on `HalfCheetah-v5`, with a smaller check on `Hopper-v5` and `Walker2d-v5`. Training always happens under noise, and every score is measured on a clean, noise free copy of the environment, so we see how much the noise damaged learning rather than how blind the agent is at test time.

## What we found

![Learning curves](assets/benchmark_curves.png)

Noise hurts, and the usual way of adding it understates the damage. A single absolute amount barely touches the fast moving sensors, the ones the controller leans on, so scaling the noise to each sensor is fairer and clearly harder. The choice of that single amount also matters far more than it should: pinned low on the spread of sensor scales, the noise is mild; pinned at the mean, learning nearly stops.

Where the noise lands matters as much as how much is added. The same total noise, aimed at the fast and busy moments, hurts far more than noise spread evenly over time. And the damage runs through the speed sensors: noise on the joint angles barely dents performance, while the same relative noise on the speed sensors is very damaging.

## All results in one place

Return is shown as a percentage of the no noise score, so higher is better.

| Experiment | Condition | Return vs no noise | Seeds |
|---|---|---|---|
| Noise style | no noise | 100% | 3 |
| | same absolute amount on every sensor | 63% | 3 |
| | scaled to each sensor, 10% of its spread | 45% | 3 |
| | realistic mix (drift, dropouts, rounding) | 49% | 3 |
| Fixed amount, by pinning point | 25th percentile of sensor scales | 85% | 8 |
| | median of sensor scales | 65% | 8 |
| | mean of sensor scales | 2% | 8 |
| | 75th percentile of sensor scales | -1% | 8 |
| | scaled to each sensor (8-seed rerun) | 53% | 8 |
| Velocity timing | steady over time | 72% | 8 |
| | concentrated at fast moments | 54% | 8 |
| Position timing | steady over time | 92% | 8 |
| | concentrated at extreme angles | 89% | 8 |
| Both sensors noisy | both steady | 69% | 8 |
| | velocity timing, position steady | 54% | 8 |
| | velocity steady, position timing | 61% | 8 |
| | both timing | 54% | 8 |
| Noise level, scaled | 5%, 10%, 20% of each sensor | 80%, 45%, 18% | 3 |
| Noise level, realistic | 5%, 10%, 20% of each sensor | 79%, 49%, 29% | 3 |
| Hopper-v5, noise style | absolute / scaled / realistic | 42%, 78%, 27% | 5 |
| Walker2d-v5, noise style | absolute / scaled / realistic | 93%, 104%, 108% | 5 |

## How the noise is added

Every sensor gets noise sized to its own normal range. For one sensor at one moment the rule is

```
noise size = rho * (that sensor's normal spread) * (a state factor)
```

`rho` is the single knob, set to 0.1, which means 10 percent. Flat noise keeps the state factor constant, so a sensor gets the same jitter whether the robot is calm or thrashing. State dependent noise keeps the total the same but concentrates it at the extreme moments: more noise at high speed for a speed sensor, more at large deflections for a joint angle. Sensors can also get richer patterns: positions get a little Gaussian noise plus rounding, like a real encoder, and velocities get slow drift plus the occasional dropped reading.

![Clean vs noisy signals](assets/02_trajectories.png)

Sizing per sensor matters because the sensors are on very different scales. On HalfCheetah they span about forty times, so a single absolute amount is huge for the small sensors and invisible for the large ones.

![Per sensor scale](assets/01_signal_scale.png)

## The experiments

### Noise style

Adding noise at `rho = 0.1` drops SAC to between 45 and 63 percent of its clean score (learning curves above), and the clean run is still improving at the end while the noisy ones have flattened. The absolute amount version looks least harmful only because it barely perturbs the fast sensors; scaled to each sensor, the noise bites harder. The realistic mix lands in between.

### Where the fixed amount is pinned

The sensor scales that a single absolute amount summarises are heavily skewed: on HalfCheetah the 25th percentile is 0.28, the median 0.73, the mean 2.6 and the 75th percentile 6.1, because a few fast velocity channels pull the upper end. Pinning the fixed amount to each of those points gives completely different verdicts, from mild to fatal, with the per sensor scaled version in between. A benchmark built on one fixed number is largely a benchmark of where that number was pinned.

![Fixed amount by pinning point](assets/fixed_pinning.png)

### Where the noise lands

Each condition here carries the same total noise and differs only in timing. On the velocity sensors the timing matters: concentrating the noise at the fast moments drops the score from 72 to 54 percent of clean, a gap well outside the spread across seeds. On the position sensors it barely does (92 versus 89 percent, within the seed spread). With both groups noisy at once the velocity timing does the damage: it drops 69 to 54 percent, position timing only reaches 61, and both together are no worse than velocity timing alone. State dependence matters where the sensor matters, and on HalfCheetah that is velocity.

![Velocity timing](assets/statedep_curves.png)

### How damage scales with the noise level

![Score vs noise level](assets/rho_sweep.png)

At 20 percent of each sensor's spread, SAC drops below a third of its clean score. The two styles look the same at low levels, and the realistic drift noise becomes somewhat less harmful as the level rises, probably because slow drift is more predictable than fresh random jitter. The spread across seeds is wide at 20 percent, so treat that gap as a trend.

### Other environments

The noise style study repeats on Hopper-v5 and Walker2d-v5 at 5 seeds. On Hopper the ordering flips: the absolute amount hurts more than the scaled version (42 versus 78 percent), most likely because Hopper's sensors sit on smaller scales, so the same absolute number is relatively larger there, and the realistic mix is the most damaging at 27 percent. Walker2d says less: even without noise SAC reaches only about 340 in 50,000 steps, versus roughly 4,700 on HalfCheetah, and at that level the three noise styles are indistinguishable.

## Parameters and constants

Everything the experiments and code depend on, in one place.

| Setting | Value |
|---|---|
| Environment | HalfCheetah-v5 for the main studies; Hopper-v5 and Walker2d-v5 for the environment check |
| Algorithm | Soft Actor Critic, CleanRL single file, run unchanged |
| Replay buffer | stable-baselines3 buffer, size 1,000,000 |
| Network | two hidden layers of 256 units, ReLU, twin critics, tanh Gaussian policy |
| Discount gamma | 0.99 |
| Target smoothing tau | 0.005 |
| Batch size | 256 |
| Learning starts after | 5,000 steps |
| Policy learning rate | 3e-4 |
| Critic learning rate | 1e-3 |
| Update schedule | policy every 2 steps, target networks every step |
| Entropy temperature | tuned automatically, target entropy equal to minus the action dimension |
| Log std range | negative 5 to positive 2 |
| Training length | 50,000 steps per run, a short prototype budget (real studies use 1,000,000) |
| Noise level rho | 0.1 by default, also 0.05 and 0.2 for the level sweep |
| Calibration | each sensor's scale measured from a 10,000 step random rollout, fixed seed 0, shared by all runs |
| State factor, state dependent | 0.5 plus the distance from typical, in units of the sensor's own spread |
| State factor, flat | a constant equal to the root mean square of the above, about 1.43, so both inject equal total noise |
| Position channels used | joint angle sensors only, the first two (torso height and pitch) are left clean because they drift between runs |
| Evaluation | 3 episodes on a clean environment every 2,500 steps, using the greedy mean action |
| Seeds | 3 for the noise style and level studies, 8 for the state dependence and fixed amount studies, 5 for Hopper and Walker2d |
| Aggregation | interquartile mean across seeds with a 95 percent bootstrap interval, falls back to the plain mean below 4 seeds |
| Compute | CPU, 1 thread per run, 12 runs in parallel, which is fastest for these small networks |

## Correctness and limits

The SAC and noise code went through a line-by-line correctness review that caught and fixed four real bugs, among them a miscalibrated drift process and a calibration seed that varied between runs. Swapping in the exact CleanRL SAC reproduced our numbers to the digit, which confirms the baseline is faithful. Full notes live in AUDIT.md.

The limits: the budget is a short 50,000 steps everywhere. The noise style and level studies use 3 seeds and are reported as plain means; the state dependence and fixed amount studies use 8 seeds with interquartile means and intervals; the Hopper and Walker2d checks use 5. Walker2d barely learns at this budget, so its numbers say little. The stochastic transition variant, where the world itself is jolted rather than the readings, and a comparison across learning algorithms are both still to do.

## Reproduce it

```bash
uv sync

# noise pattern figures
uv run python scripts/explore_noise.py --env HalfCheetah-v5 --outdir assets

# noise style study
uv run python scripts/run_benchmark.py --modes none uniform uniform-calibrated realistic \
    --seeds 1 2 3 --total-timesteps 50000 --jobs 12 --threads-per-job 1 --outdir results_modes
uv run python scripts/plot_results.py --outdir results_modes --figdir assets --prefix benchmark

# fixed amount by pinning point
uv run python scripts/run_benchmark.py \
    --modes none fixed-p25 fixed-median fixed-mean fixed-p75 uniform-calibrated \
    --seeds 1 2 3 4 5 6 7 8 --total-timesteps 50000 --jobs 12 --threads-per-job 1 --outdir results_fixed
uv run python scripts/plot_fixed.py --outdir results_fixed --figdir assets

# state dependence study, velocity and position and both
uv run python scripts/run_benchmark.py \
    --modes none vel-flat vel-statedep pos-flat pos-statedep both-ff both-sf both-fs both-ss \
    --seeds 1 2 3 4 5 6 7 8 --total-timesteps 50000 --jobs 12 --threads-per-job 1 --outdir results_sd8
uv run python scripts/plot_results.py --outdir results_sd8 --figdir assets --prefix statedep --modes none vel-flat vel-statedep

# noise level sweep
uv run python scripts/run_benchmark.py --modes uniform-calibrated realistic --seeds 1 2 3 \
    --rho 0.05 --outdir results_rho005 --total-timesteps 50000 --jobs 12 --threads-per-job 1
uv run python scripts/run_benchmark.py --modes uniform-calibrated realistic --seeds 1 2 3 \
    --rho 0.2 --outdir results_rho020 --total-timesteps 50000 --jobs 12 --threads-per-job 1
uv run python scripts/plot_rho.py --pairs 0.05:results_rho005 0.1:results_modes 0.2:results_rho020 \
    --clean-dir results_modes

# other environments (repeat with Walker2d-v5 / results_walker2d / --prefix walker2d)
uv run python scripts/run_benchmark.py --env-id Hopper-v5 --modes none uniform uniform-calibrated realistic \
    --seeds 1 2 3 4 5 --total-timesteps 50000 --jobs 12 --threads-per-job 1 --outdir results_hopper
uv run python scripts/plot_results.py --outdir results_hopper --figdir assets --prefix hopper
```

## Repo layout

```
src/stochrl/
  noise.py       the noise processes and the model that applies them
  calibrate.py   measures each sensor's normal spread
  presets.py     ready made noise setups (uniform, fixed, realistic, isolation, combined)
  wrappers.py    gymnasium wrappers that add observation or action noise
  plotting.py    shared black and white figure style
  stats.py       interquartile mean and bootstrap intervals
scripts/
  explore_noise.py           draw the noise pattern figures
  sac_continuous_action.py   the CleanRL SAC with noise switched in
  run_benchmark.py           run many seeds and modes in parallel
  plot_results.py            turn results into figures and tables
  plot_fixed.py              the fixed-amount pinning figure
  plot_rho.py                the noise level figure
```
