"""Composable per-channel noise processes for control benchmarks.

A NoiseModel assigns noise processes (Gaussian, OU drift, dropout,
quantization, ...) to sets of channels. Each process expresses its strength
relative to the channel's signal magnitude (see calibrate.py), so a single
relative level rho is comparable across channels and environments. A
per-channel gain_fn can additionally make the strength depend on the current
state, e.g. a velocity sensor that gets noisier at speed. All randomness
comes from an explicit numpy Generator so runs are reproducible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


class NoiseProcess(ABC):
    """A noise pattern applied to a fixed-width slice of a signal vector.

    `scale` is the per-channel signal magnitude set by calibration. Strength
    is relative to it, so relative=0.1 means 10% of the channel's natural
    variation regardless of units.
    """

    def bind(self, scale: Array) -> "NoiseProcess":
        """Attach the calibrated signal scale for these channels."""
        self.scale = np.asarray(scale, dtype=np.float64)
        return self

    def reset(self, rng: np.random.Generator) -> None:
        """Reset per-episode state (drift, bias, delay buffers)."""

    @abstractmethod
    def __call__(self, x: Array, rng: np.random.Generator, gain: Array) -> Array:
        """Return a perturbed copy of `x`. `gain` is a per-channel state multiplier."""


@dataclass
class Gaussian(NoiseProcess):
    """Additive white noise: sigma = relative * signal_scale * gain(state)."""

    relative: float = 0.05
    scale: Array = field(default=None, repr=False)

    def __call__(self, x, rng, gain):
        sigma = self.relative * self.scale * gain
        return x + rng.normal(0.0, 1.0, size=x.shape) * sigma


@dataclass
class OrnsteinUhlenbeck(NoiseProcess):
    """Temporally correlated noise (sensor drift), as a discrete AR(1) process.

    Stationary std = relative * signal_scale. `theta` is the mean-reversion
    rate; small theta gives a slowly drifting bias.
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
        # Stationary Var = std^2 exactly, from step 0. Don't swap in the
        # continuous-time sqrt(2*theta) coefficient: at dt=1 it inflates the std.
        self._y = a * self._y + std * np.sqrt(1.0 - a * a) * rng.normal(0.0, 1.0, size=x.shape)
        return x + self._y * gain


@dataclass
class MultiplicativeGaussian(NoiseProcess):
    """Gain error: x *= (1 + eps). Unit-free, so it ignores signal_scale."""

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

    mode='zero' returns 0 (lost packet); mode='hold' repeats the last good
    value (stuck sensor).
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
        # quantize over a +/-3 signal-scale sensor range
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


@dataclass
class ChannelNoise:
    """Assign a NoiseProcess to a set of channel indices.

    `gain_fn(ref) -> float | array` makes the noise strength depend on the
    pre-noise observation. Defaults to a constant 1 (homoscedastic).
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
    """Applies per-channel noise to a vector.

    dim    : width of the signal vector (obs dim or action dim).
    scale  : per-channel signal magnitude, length `dim` (from `calibrate`).
             Pass ones(dim) for absolute, uncalibrated noise.
    specs  : list of ChannelNoise. Channels not covered pass through clean.
    record : keep per-step (clean, noisy) pairs in `self.history`.
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
