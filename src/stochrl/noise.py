"""Composable, signal-calibrated noise processes for control benchmarks.

This module is the conceptual core of the project. The paper we build on
("Towards Understanding the Impact of Plasticity Loss on RL in Stochastic
Environments") adds a *single, uniform* Gaussian noise to every observation /
action dimension. Its own Limitations section flags this as the main weakness:

    "...applying the same level of noise indiscriminate of the observation
     dimension."

and its Conclusion asks for

    "...observation noise that is well-calibrated with regard to the overall
     signal magnitude on the respective channels."

So we need to answer three questions for a *meaningful* benchmark:

    1. WHICH channel gets noise?        -> per-channel assignment (ChannelNoise)
    2. WHICH PATTERN of noise?          -> a NoiseProcess (Gaussian, OU, dropout,
                                            quantization, delay, bias, ...)
    3. WHAT DEGREE / how much?          -> calibrate to each channel's own signal
                                            magnitude (see `calibrate.py`). A single
                                            relative level rho then means the same
                                            thing across channels AND environments.

Optionally, the degree can also depend on the *state* itself (a velocity sensor
that gets noisier at high velocity), via a per-channel `gain_fn`.

Everything is driven by an explicit numpy Generator so runs are reproducible,
which is non-negotiable for a benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Noise processes: each operates on the sub-vector of channels it is assigned. #
# --------------------------------------------------------------------------- #
class NoiseProcess(ABC):
    """A noise pattern applied to a fixed-width slice of a signal vector.

    `scale` is the per-channel signal magnitude (set by calibration). A process
    expresses its strength *relative* to this scale, so `relative=0.1` always
    means "10% of this channel's natural variation" regardless of units.
    """

    def bind(self, scale: Array) -> "NoiseProcess":
        """Attach the calibrated per-channel signal scale for these channels."""
        self.scale = np.asarray(scale, dtype=np.float64)
        return self

    def reset(self, rng: np.random.Generator) -> None:
        """Reset any internal episode state (drift, bias, delay buffers)."""

    @abstractmethod
    def __call__(self, x: Array, rng: np.random.Generator, gain: Array) -> Array:
        """Return a perturbed copy of `x`. `gain` is an optional state multiplier."""


@dataclass
class Gaussian(NoiseProcess):
    """Additive white Gaussian noise: the baseline used by prior work.

    sigma = relative * signal_scale * gain(state).
    """

    relative: float = 0.05
    scale: Array = field(default=None, repr=False)

    def __call__(self, x, rng, gain):
        sigma = self.relative * self.scale * gain
        return x + rng.normal(0.0, 1.0, size=x.shape) * sigma


@dataclass
class OrnsteinUhlenbeck(NoiseProcess):
    """Temporally correlated (colored) noise: models sensor drift / 1-f noise.

    Real sensor error is rarely fresh white noise each step; it wanders. This is
    an AR(1) / discretized OU process with stationary std = relative*signal_scale.
    `theta` is the mean-reversion rate (small theta -> slowly drifting bias).
    """

    relative: float = 0.05
    theta: float = 0.15
    scale: Array = field(default=None, repr=False)
    _y: Array = field(default=None, repr=False)

    def reset(self, rng):
        self._y = None

    def __call__(self, x, rng, gain):
        std = self.relative * self.scale
        a = 1.0 - self.theta  # AR(1) coefficient; needs 0 < theta < 2 for stability
        if self._y is None:
            self._y = rng.normal(0.0, 1.0, size=x.shape) * std
        # Exact discrete AR(1): stationary Var = b^2 / (1 - a^2) = std^2, so the
        # process is calibrated to `std` AND stationary from step 0. (Using the
        # continuous-time sqrt(2*theta) coefficient with dt=1 would inflate it.)
        self._y = a * self._y + std * np.sqrt(1.0 - a * a) * rng.normal(0.0, 1.0, size=x.shape)
        return x + self._y * gain


@dataclass
class MultiplicativeGaussian(NoiseProcess):
    """Gain / scale error: x *= (1 + eps). Models miscalibrated sensors/actuators.

    Note this is unit-free already, so it ignores signal_scale by construction.
    """

    relative: float = 0.05

    def __call__(self, x, rng, gain):
        return x * (1.0 + rng.normal(0.0, 1.0, size=x.shape) * self.relative * gain)


@dataclass
class Bias(NoiseProcess):
    """Constant per-episode offset: a calibration error fixed at reset time."""

    relative: float = 0.05
    scale: Array = field(default=None, repr=False)
    _b: Array = field(default=None, repr=False)

    def reset(self, rng):
        self._b = None

    def __call__(self, x, rng, gain):
        if self._b is None:
            self._b = rng.normal(0.0, 1.0, size=x.shape) * self.relative * self.scale
        return x + self._b * gain


@dataclass
class Dropout(NoiseProcess):
    """Sensor dropout: each step a channel may go dead.

    mode='zero' returns 0 (lost packet); mode='hold' returns the last good value
    (a stuck sensor). A classic non-Gaussian failure the baseline can't express.
    """

    prob: float = 0.05
    mode: str = "hold"
    _last: Array = field(default=None, repr=False)

    def reset(self, rng):
        self._last = None

    def __call__(self, x, rng, gain):
        if self._last is None:
            self._last = x.copy()
        drop = rng.random(x.shape) < self.prob
        out = np.where(drop, 0.0 if self.mode == "zero" else self._last, x)
        self._last = out.copy()
        return out


@dataclass
class Quantization(NoiseProcess):
    """ADC / encoder resolution: round to a finite number of levels per channel."""

    levels: int = 64
    scale: Array = field(default=None, repr=False)

    def __call__(self, x, rng, gain):
        # Quantize over +/- 3 signal-scales, a generous sensor range.
        lo, hi = -3.0 * self.scale, 3.0 * self.scale
        step = (hi - lo) / max(self.levels - 1, 1)
        step = np.where(step == 0, 1.0, step)
        return lo + np.round((np.clip(x, lo, hi) - lo) / step) * step


@dataclass
class Saturation(NoiseProcess):
    """Sensor range limits: clip to +/- `limit` signal-scales."""

    limit: float = 3.0
    scale: Array = field(default=None, repr=False)

    def __call__(self, x, rng, gain):
        return np.clip(x, -self.limit * self.scale, self.limit * self.scale)


@dataclass
class Delay(NoiseProcess):
    """Observation latency: return the value from `steps` ago (FIFO buffer)."""

    steps: int = 1
    _buf: deque = field(default=None, repr=False)

    def reset(self, rng):
        self._buf = None

    def __call__(self, x, rng, gain):
        if self._buf is None:
            self._buf = deque([x.copy()] * (self.steps + 1), maxlen=self.steps + 1)
        self._buf.append(x.copy())
        return self._buf[0]


@dataclass
class Compose(NoiseProcess):
    """Chain several processes on the same channels (e.g. drift -> quantize)."""

    processes: list = field(default_factory=list)

    def bind(self, scale):
        self.scale = scale
        for p in self.processes:
            p.bind(scale)
        return self

    def reset(self, rng):
        for p in self.processes:
            p.reset(rng)

    def __call__(self, x, rng, gain):
        for p in self.processes:
            x = p(x, rng, gain)
        return x


# --------------------------------------------------------------------------- #
# Channel assignment + the full model that ties it together.                  #
# --------------------------------------------------------------------------- #
@dataclass
class ChannelNoise:
    """Assign a NoiseProcess to a set of channel indices, with optional state gain.

    `gain_fn(ref) -> float | array` lets the noise *degree depend on the state*
    (the realised, pre-noise observation). Returns 1.0 by default (homoscedastic).
    """

    indices: list
    process: NoiseProcess
    gain_fn: callable = None

    def gain(self, ref: Array) -> Array:
        if self.gain_fn is None:
            return np.ones(len(self.indices))
        g = np.asarray(self.gain_fn(ref), dtype=np.float64)
        return np.broadcast_to(g, (len(self.indices),))


class NoiseModel:
    """Applies per-channel noise to a vector, calibrated to per-channel scale.

    Parameters
    ----------
    dim   : width of the signal vector (obs dim or action dim).
    scale : per-channel signal magnitude, length `dim` (from `calibrate`). The
            single source of "what degree" — every process strength is relative
            to this. Pass ones(dim) to fall back to absolute, uncalibrated noise.
    specs : list of ChannelNoise. Channels not covered are passed through clean.
    record: keep per-step (clean, noisy) history for analysis / plotting.
    """

    def __init__(self, dim, scale, specs, record=False):
        self.dim = dim
        self.scale = np.asarray(scale, dtype=np.float64)
        self.specs = specs
        for s in specs:
            s.process.bind(self.scale[s.indices])
        self.record = record
        self.history = []

    def reset(self, rng):
        for s in self.specs:
            s.process.reset(rng)

    def __call__(self, x: Array, rng: np.random.Generator) -> Array:
        x = np.asarray(x, dtype=np.float64)
        out = x.copy()
        for s in self.specs:
            out[s.indices] = s.process(x[s.indices], rng, s.gain(x))
        if self.record:
            self.history.append((x.copy(), out.copy()))
        return out.astype(np.float32)
