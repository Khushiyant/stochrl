"""stochrl: signal-calibrated, per-channel noise benchmarks for continuous control."""

from .calibrate import SignalStats, collect_signal_stats
from .noise import (
    Bias,
    ChannelNoise,
    Compose,
    Delay,
    Dropout,
    Gaussian,
    MultiplicativeGaussian,
    NoiseModel,
    NoiseProcess,
    OrnsteinUhlenbeck,
    Quantization,
    Saturation,
)
from .presets import actuator_noise, realistic_sensors, uniform_gaussian
from .wrappers import ActionNoise, ObservationNoise

__all__ = [
    "SignalStats",
    "collect_signal_stats",
    "NoiseModel",
    "NoiseProcess",
    "ChannelNoise",
    "Compose",
    "Gaussian",
    "OrnsteinUhlenbeck",
    "MultiplicativeGaussian",
    "Bias",
    "Dropout",
    "Quantization",
    "Saturation",
    "Delay",
    "uniform_gaussian",
    "realistic_sensors",
    "actuator_noise",
    "ObservationNoise",
    "ActionNoise",
]
