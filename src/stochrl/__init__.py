"""stochrl: signal-calibrated, per-channel noise benchmarks for continuous control."""

from .calibrate import SignalStats, collect_qvel_stats, collect_signal_stats
from .envs import make_flat
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
from .wrappers import ActionNoise, ObservationNoise, TransitionNoise

__all__ = [
    "SignalStats",
    "collect_qvel_stats",
    "collect_signal_stats",
    "make_flat",
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
    "TransitionNoise",
]
