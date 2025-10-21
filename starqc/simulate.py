"""Synthetic signal generators and artifact injectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

TWOPI = 2.0 * np.pi


@dataclass
class SimulationConfig:
    channels: int
    samples: int
    fs: float
    oscillation_hz: float | None = 12.0
    oscillation_amp: float = 20.0
    noise_scale: float = 5.0


def base_signal(config: SimulationConfig) -> np.ndarray:
    """Generate coloured noise with an optional oscillatory component."""

    rng = np.random.default_rng(1234)
    noise = rng.standard_normal((config.channels, config.samples)).astype(np.float32)
    # 1/f shaping in frequency domain via FFT scaling.
    freq = np.fft.rfftfreq(config.samples, 1.0 / config.fs)
    scale = np.sqrt(1.0 / np.maximum(freq, 1.0))
    shaped = np.fft.irfft(np.fft.rfft(noise, axis=1) * scale[None, :], n=config.samples, axis=1)
    signal = config.noise_scale * shaped.astype(np.float32)

    if config.oscillation_hz:
        t = np.arange(config.samples, dtype=np.float32) / config.fs
        osc = np.sin(TWOPI * config.oscillation_hz * t, dtype=np.float32)
        signal += config.oscillation_amp * osc

    return signal.astype(np.float32)


def inject_line_hum(x: np.ndarray, fs: float, amplitude: float = 25.0, freq: float = 60.0) -> np.ndarray:
    t = np.arange(x.shape[1], dtype=np.float32) / fs
    hum = amplitude * np.sin(TWOPI * freq * t, dtype=np.float32)
    return (x + hum).astype(np.float32)


def inject_slow_drift(x: np.ndarray, fs: float, amplitude: float = 100.0, freq: float = 0.2) -> np.ndarray:
    t = np.arange(x.shape[1], dtype=np.float32) / fs
    drift = amplitude * np.sin(TWOPI * freq * t, dtype=np.float32)
    return (x + drift).astype(np.float32)


def inject_stim(x: np.ndarray, fs: float, stim_times_s: Sequence[float], magnitude: float = 200.0, width_ms: float = 2.0) -> tuple[np.ndarray, List[int]]:
    samples = int(round(width_ms * fs / 1000.0))
    samples = max(samples, 1)
    stim_indices: List[int] = []
    augmented = x.copy()
    for stim in stim_times_s:
        idx = int(round(stim * fs))
        stim_indices.append(idx)
        lo = max(idx - samples // 2, 0)
        hi = min(idx + samples // 2 + 1, x.shape[1])
        augmented[:, lo:hi] += magnitude
    return augmented.astype(np.float32), stim_indices


def inject_clipping(x: np.ndarray, fraction: float = 0.02, rail: float = 500.0) -> np.ndarray:
    clipped = x.copy()
    total = x.shape[1]
    span = max(int(total * fraction), 1)
    start = total // 3
    clipped[:, start : start + span] = np.clip(clipped[:, start : start + span], -rail, rail)
    return clipped.astype(np.float32)


def inject_dropout(x: np.ndarray, fs: float, duration_ms: float = 80.0) -> np.ndarray:
    samples = int(round(duration_ms * fs / 1000.0))
    samples = max(samples, 1)
    start = x.shape[1] // 2
    dropped = x.copy()
    dropped[:, start : start + samples] = dropped[:, start, None]
    return dropped.astype(np.float32)
