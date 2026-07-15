from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


class NoiseProcess(ABC):
    """Noise on a slice of a signal vector; strength is relative to the calibrated `scale`."""

    def bind(self, scale: Array) -> "NoiseProcess":
        self.scale = np.asarray(scale, dtype=np.float64)
        return self

    def reset(self, rng: np.random.Generator) -> None:
        """Reset per-episode state."""

    @abstractmethod
    def __call__(self, x: Array, rng: np.random.Generator, gain: Array) -> Array:
        ...


@dataclass
class Gaussian(NoiseProcess):
    """sigma = relative * scale * gain."""

    relative: float = 0.05
    scale: Array = field(default=None, repr=False)

    def __call__(self, x, rng, gain):
        sigma = self.relative * self.scale * gain
        return x + rng.normal(0.0, 1.0, size=x.shape) * sigma


@dataclass
class OrnsteinUhlenbeck(NoiseProcess):
    """Sensor drift as AR(1); stationary std = relative * scale, theta = mean reversion."""

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
        # exact AR(1): stationary var = std^2 (the continuous-time sqrt(2*theta) inflates it at dt=1)
        self._y = a * self._y + std * np.sqrt(1.0 - a * a) * rng.normal(0.0, 1.0, size=x.shape)
        return x + self._y * gain


@dataclass
class MultiplicativeGaussian(NoiseProcess):
    """x *= (1 + eps); unit-free, ignores scale."""

    relative: float = 0.05

    def __call__(self, x, rng, gain):
        return x * (1.0 + rng.normal(0.0, 1.0, size=x.shape) * self.relative * gain)


@dataclass
class Bias(NoiseProcess):
    """Constant per-episode offset."""

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
    """Per-step channel dropout; mode='zero' or 'hold' (repeat last good value)."""

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
    """Round to `levels` steps over a +/-3 signal-scale range."""

    levels: int = 64
    scale: Array = field(default=None, repr=False)

    def __call__(self, x, rng, gain):
        lo, hi = -3.0 * self.scale, 3.0 * self.scale
        step = (hi - lo) / max(self.levels - 1, 1)
        step = np.where(step == 0, 1.0, step)
        return lo + np.round((np.clip(x, lo, hi) - lo) / step) * step


@dataclass
class Saturation(NoiseProcess):
    """Clip to +/- `limit` signal-scales."""

    limit: float = 3.0
    scale: Array = field(default=None, repr=False)

    def __call__(self, x, rng, gain):
        return np.clip(x, -self.limit * self.scale, self.limit * self.scale)


@dataclass
class Delay(NoiseProcess):
    """Return the value from `steps` ago."""

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
    """Chain processes on the same channels."""

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
    """A NoiseProcess on a set of channel indices; gain_fn(pre-noise obs) scales it, default 1."""

    indices: list
    process: NoiseProcess
    gain_fn: callable = None

    def gain(self, ref: Array) -> Array:
        if self.gain_fn is None:
            return np.ones(len(self.indices))
        g = np.asarray(self.gain_fn(ref), dtype=np.float64)
        return np.broadcast_to(g, (len(self.indices),))


class NoiseModel:
    """Applies ChannelNoise specs to a vector; uncovered channels pass through clean.

    `scale` comes from calibrate (ones(dim) for absolute noise); `record`
    keeps per-step (clean, noisy) pairs in `self.history`.
    """

    def __init__(self, dim, scale, specs, record=False, out_dtype=np.float32):
        self.dim = dim
        self.out_dtype = out_dtype  # float64 for the transition-noise state path
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
        return out.astype(self.out_dtype)
