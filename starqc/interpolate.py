"""Interpolation helpers for StarQC."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import interpolate

from .config import InterpolateConfig


def interpolate_short_gaps(
    x: np.ndarray,
    mask: np.ndarray,
    fs: float,
    config: InterpolateConfig,
    immutable: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill short masked gaps using the requested interpolation method."""

    if x.shape != mask.shape:
        raise ValueError("x and mask must share the same shape")

    if immutable is not None and immutable.shape != mask.shape:
        raise ValueError("immutable mask must match shape of x")

    max_gap = int(round(fs * config.interpolate_max_ms / 1000.0))
    max_gap = max(max_gap, 1)

    filled = x.copy()
    updated_mask = mask.copy()
    immutable_mask = immutable if immutable is not None else np.zeros_like(mask, dtype=bool)
    C, T = x.shape

    for ch in range(C):
        bad = mask[ch]
        if not np.any(bad):
            continue
        idx = 0
        while idx < T:
            if not bad[idx]:
                idx += 1
                continue
            start = idx
            while idx < T and bad[idx]:
                idx += 1
            end = idx
            run = end - start
            if immutable_mask[ch, start:end].any():
                continue
            if run <= max_gap:
                _interpolate_segment(
                    filled[ch],
                    start,
                    end,
                    method=config.method,
                )
                updated_mask[ch, start:end] = False
    updated_mask |= immutable_mask
    return filled.astype(np.float32, copy=False), updated_mask


def _interpolate_segment(trace: np.ndarray, start: int, end: int, method: str) -> None:
    """In-place interpolation for a segment [start, end)."""

    T = trace.shape[0]
    length = end - start
    if length <= 0:
        return

    left_idx = start - 1
    right_idx = end

    if left_idx < 0 and right_idx >= T:
        return

    left_val = trace[left_idx] if left_idx >= 0 else trace[right_idx]
    right_val = trace[right_idx] if right_idx < T else trace[left_idx]

    if method == "cubic" and left_idx >= 1 and right_idx <= T - 2:
        anchor_x = np.array([left_idx - 1, left_idx, right_idx, right_idx + 1])
        anchor_y = trace[anchor_x]
        spline = interpolate.CubicSpline(anchor_x, anchor_y, bc_type="natural")
        new_idx = np.arange(start, end)
        trace[start:end] = spline(new_idx).astype(np.float32)
        return

    new_vals = np.linspace(left_val, right_val, length + 2, dtype=np.float32)[1:-1]
    trace[start:end] = new_vals
