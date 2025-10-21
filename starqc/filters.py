"""Filtering utilities for StarQC."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .config import DriftRemovalConfig, LineRemovalConfig

_TWO_PI = 2.0 * np.pi
_MEDIAN_STOP_RATIO = 0.3
_LINEAR_STOP_RATIO = 0.02


def common_median_reference(x: np.ndarray) -> np.ndarray:
    """Subtract the per-sample median across channels."""

    if x.shape[0] <= 1:
        return x.astype(np.float32, copy=True)
    median = np.median(x, axis=0, keepdims=True)
    return (x - median).astype(np.float32, copy=False)


def remove_line_hum(
    x: np.ndarray,
    fs: float,
    config: LineRemovalConfig,
    start_sample: int = 0,
) -> np.ndarray:
    """Project out power-line sinusoids using least-squares regression."""

    if config.notch_hz is None or config.notch_hz <= 0:
        return x.astype(np.float32, copy=True)

    cleaned = x.astype(np.float64, copy=True)
    harmonics = max(1, int(config.harmonics))
    samples = cleaned.shape[1]

    indices = start_sample + np.arange(samples, dtype=np.float64)

    for harmonic in range(1, harmonics + 1):
        freq = harmonic * float(config.notch_hz)
        if freq <= 0 or freq >= fs / 2:
            continue
        basis = _sinusoid_basis(freq, fs, indices)
        gram = basis @ basis.T
        inv_gram = np.linalg.pinv(gram, rcond=1e-12)
        coeff = cleaned @ basis.T
        weights = coeff @ inv_gram
        projection = weights @ basis
        cleaned -= projection

    return cleaned.astype(np.float32)


def _sinusoid_basis(freq: float, fs: float, indices: np.ndarray) -> np.ndarray:
    phase = _TWO_PI * freq * indices / fs
    sin_wave = np.sin(phase)
    cos_wave = np.cos(phase)
    return np.vstack([sin_wave, cos_wave])


def remove_slow_drift(
    x: np.ndarray,
    fs: float,
    config: DriftRemovalConfig,
) -> np.ndarray:
    """Remove baseline wander using the configured method."""

    method = config.detrend_method.lower()
    if method == "median":
        window_samples = max(int(round(fs * config.median_window_s)), 1)
        if window_samples % 2 == 0:
            window_samples += 1
        baseline = signal.medfilt(x, kernel_size=(1, window_samples))
        baseline_rms = np.sqrt(np.mean(baseline.astype(np.float64) ** 2))
        signal_rms = np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12
        if baseline_rms <= _MEDIAN_STOP_RATIO * signal_rms:
            return x.astype(np.float32, copy=True)
        return (x - baseline.astype(np.float32, copy=False)).astype(np.float32, copy=False)
    if method == "highpass":
        nyq = fs / 2.0
        cutoff = max(config.highpass_hz, 1e-3)
        norm = min(cutoff / nyq, 0.99)
        sos = signal.butter(2, norm, btype="highpass", output="sos")
        filtered = signal.sosfiltfilt(sos, x, axis=1).astype(np.float32, copy=False)
        diff_rms = np.sqrt(np.mean((filtered.astype(np.float64) - x.astype(np.float64)) ** 2))
        signal_rms = np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12
        if diff_rms <= _LINEAR_STOP_RATIO * signal_rms:
            return x.astype(np.float32, copy=True)
        return filtered
    raise ValueError(f"Unsupported detrend_method: {config.detrend_method}")
