"""Artifact detection utilities for StarQC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .config import DetectConfig


@dataclass(frozen=True)
class DetectionSummary:
    """Summary statistics for detection, useful for debugging and QC."""

    clip_counts: np.ndarray
    flatline_counts: np.ndarray
    stim_counts: np.ndarray


def validate_input(x: np.ndarray, fs: float) -> None:
    """Validate the core input array and sampling rate."""

    if not isinstance(x, np.ndarray):
        raise ValueError("x must be a numpy.ndarray")
    if x.dtype != np.float32:
        raise ValueError("x must have dtype float32")
    if x.ndim != 2:
        raise ValueError("x must have shape (channels, samples)")
    if fs is None or fs <= 0:
        raise ValueError("fs must be a positive float")


def detect_artifacts(
    x: np.ndarray,
    fs: float,
    config: DetectConfig,
    stim_times_s: Optional[Iterable[float]] = None,
    voltage_range: Optional[tuple[float, float]] = None,
) -> tuple[np.ndarray, DetectionSummary, dict[str, np.ndarray]]:
    """Return mask, summary, and component masks."""

    validate_input(x, fs)
    mask = np.zeros_like(x, dtype=bool)

    clip_mask, clip_counts = _detect_clipping(x, config, voltage_range)
    mask |= clip_mask

    flat_mask, flat_counts = _detect_flatlines(x, fs, config, voltage_range)
    mask |= flat_mask

    stim_mask, stim_counts = _stim_mask(x, fs, config, stim_times_s)
    mask |= stim_mask

    if config.min_mask_run_ms > 0:
        mask = _drop_short_runs(mask, fs, config.min_mask_run_ms)

    summary = DetectionSummary(
        clip_counts=clip_counts,
        flatline_counts=flat_counts,
        stim_counts=stim_counts,
    )

    components = {
        "clip": clip_mask.copy(),
        "flatline": flat_mask.copy(),
        "stim": stim_mask.copy(),
    }

    return mask, summary, components


def _detect_clipping(
    x: np.ndarray,
    config: DetectConfig,
    voltage_range: Optional[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Detect samples that are close to the rails."""

    threshold = config.clip_threshold if config.clip_threshold is not None else 0.98
    if threshold <= 0.5 or threshold >= 1.0:
        raise ValueError("clip_threshold must be in (0.5, 1.0)")

    if voltage_range is not None:
        lo, hi = float(voltage_range[0]), float(voltage_range[1])
        lo_arr = np.full(x.shape[0], lo, dtype=np.float32)
        hi_arr = np.full(x.shape[0], hi, dtype=np.float32)
    else:
        lo_arr = np.min(x, axis=1)
        hi_arr = np.max(x, axis=1)

    dynamic = np.maximum(hi_arr - lo_arr, np.finfo(np.float32).eps)
    lower = lo_arr + (1.0 - threshold) * dynamic
    upper = lo_arr + threshold * dynamic

    mask = (x <= lower[:, None]) | (x >= upper[:, None])
    counts = mask.sum(axis=1).astype(np.int64)
    return mask, counts


def _detect_flatlines(
    x: np.ndarray,
    fs: float,
    config: DetectConfig,
    voltage_range: Optional[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Detect low-variance segments that indicate flatlines or dropouts."""

    window_ms = max(config.flatline_ms, 1.0)
    window_samples = int(round(fs * window_ms / 1000.0))
    window_samples = max(window_samples, 3)

    if voltage_range is not None:
        span = abs(float(voltage_range[1]) - float(voltage_range[0]))
    else:
        span = float(np.max(x) - np.min(x))
    tol = max(config.flatline_epsilon * span, np.finfo(np.float32).eps)

    C, T = x.shape
    mask = np.zeros((C, T), dtype=bool)
    counts = np.zeros(C, dtype=np.int64)

    min_run = max(window_samples - 1, 1)
    padding = window_samples // 4

    for ch in range(C):
        diffs = np.abs(np.diff(x[ch]))
        if diffs.size == 0:
            continue
        flat_diff = diffs < tol
        idx = 0
        while idx < flat_diff.size:
            if not flat_diff[idx]:
                idx += 1
                continue
            start = idx
            while idx < flat_diff.size and flat_diff[idx]:
                idx += 1
            run_len = idx - start
            if run_len >= min_run:
                lo = max(start - padding, 0)
                hi = min(idx + padding + 1, T)
                mask[ch, lo:hi] = True
        counts[ch] = int(np.count_nonzero(mask[ch]))
    return mask, counts


def _stim_mask(
    x: np.ndarray,
    fs: float,
    config: DetectConfig,
    stim_times_s: Optional[Iterable[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a mask that covers stimulation windows if stims are provided."""

    C, T = x.shape
    mask = np.zeros((C, T), dtype=bool)
    counts = np.zeros(C, dtype=np.int64)

    if stim_times_s is None:
        return mask, counts

    pad_samples = int(round(config.pad_ms * fs / 1000.0))
    pad_samples = max(pad_samples, 0)

    for stim in stim_times_s:
        if stim is None:
            continue
        sample = int(round(float(stim) * fs))
        lo = max(sample - pad_samples, 0)
        hi = min(sample + pad_samples + 1, T)
        if lo >= hi:
            continue
        mask[:, lo:hi] = True
        counts += hi - lo

    return mask, counts


def _drop_short_runs(mask: np.ndarray, fs: float, min_run_ms: float) -> np.ndarray:
    """Suppress masked runs that are shorter than the configured duration."""

    min_samples = int(round(fs * min_run_ms / 1000.0))
    if min_samples <= 1:
        return mask

    cleaned = mask.copy()
    C, T = mask.shape
    for ch in range(C):
        start = 0
        while start < T:
            if not mask[ch, start]:
                start += 1
                continue
            end = start
            while end < T and mask[ch, end]:
                end += 1
            run_len = end - start
            if run_len < min_samples:
                cleaned[ch, start:end] = False
            start = end
    return cleaned
