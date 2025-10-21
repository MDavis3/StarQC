"""Chunked/streaming processing API for StarQC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .config import PipelineConfig, get_default_config
from .detect import DetectionSummary, detect_artifacts, validate_input
from .filters import common_median_reference, remove_line_hum, remove_slow_drift
from .interpolate import interpolate_short_gaps

_IDEMPOTENCE_TOL = 1e-2


@dataclass
class StreamingState:
    """Keeps cumulative detection statistics across chunks."""

    clip_counts: Optional[np.ndarray] = None
    flatline_counts: Optional[np.ndarray] = None
    stim_counts: Optional[np.ndarray] = None


class StreamingCleaner:
    """Stream chunks of data through the StarQC pipeline."""

    def __init__(
        self,
        fs: float,
        config: Optional[PipelineConfig] = None,
        stim_times_s: Optional[Iterable[float]] = None,
        voltage_range: Optional[tuple[float, float]] = None,
    ) -> None:
        if config is None:
            config = get_default_config()
        if fs <= 0:
            raise ValueError("fs must be positive")

        self.fs = float(fs)
        self.config = config
        self.voltage_range = voltage_range
        self.stim_times_s = list(stim_times_s) if stim_times_s is not None else None
        self.sample_offset = 0
        self._channels: Optional[int] = None
        self.state = StreamingState()

        self._full_original: Optional[np.ndarray] = None
        self._clean_full: Optional[np.ndarray] = None
        self._mask_full: Optional[np.ndarray] = None
        self._raw_mask: Optional[np.ndarray] = None
        self._stim_mask: Optional[np.ndarray] = None
        self._summary: Optional[DetectionSummary] = None
        self._released = 0
        self._segments: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        self._drift_half = self._compute_drift_half()

    def process_chunk(self, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Process a chunk of data and return the released samples."""

        validate_input(chunk, self.fs)
        if self._channels is None:
            self._channels = chunk.shape[0]
        elif chunk.shape[0] != self._channels:
            raise ValueError("All chunks must have the same channel count")

        self._append_original(chunk.astype(np.float32, copy=True))
        self.sample_offset += chunk.shape[1]

        self._recompute_pipeline()
        return self._release_ready()

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """Flush any buffered samples at end-of-stream."""

        if self._clean_full is None:
            empty = np.zeros((self._channels or 0, 0), dtype=np.float32)
            empty_mask = np.zeros_like(empty, dtype=bool)
            return empty, empty_mask

        self._refresh_released_segments()
        clean_tail = self._clean_full[:, self._released :]
        mask_tail = self._mask_full[:, self._released :]
        self._released = self._clean_full.shape[1]
        return clean_tail.copy(), mask_tail.copy()

    def detection_summary(self) -> DetectionSummary:
        if self._summary is None:
            raise RuntimeError("StreamingCleaner has not processed any data")
        return DetectionSummary(
            clip_counts=self._summary.clip_counts.copy(),
            flatline_counts=self._summary.flatline_counts.copy(),
            stim_counts=self._summary.stim_counts.copy(),
        )

    def _append_original(self, chunk: np.ndarray) -> None:
        if self._full_original is None:
            self._full_original = chunk
        else:
            self._full_original = np.concatenate([self._full_original, chunk], axis=1)

    def _compute_drift_half(self) -> int:
        if self.config.drift.detrend_method.lower() != "median":
            return 0
        window = max(int(round(self.fs * self.config.drift.median_window_s)), 1)
        if window % 2 == 0:
            window += 1
        return window // 2

    def _recompute_pipeline(self) -> None:
        if self._full_original is None:
            return

        data = self._full_original.copy()
        if self.config.standardize.re_reference:
            data = common_median_reference(data)

        mask, summary, components = detect_artifacts(
            self._full_original,
            self.fs,
            self.config.detect,
            stim_times_s=self.stim_times_s,
            voltage_range=self.voltage_range,
        )
        self._summary = summary
        self._raw_mask = mask
        self._stim_mask = components["stim"]

        prepared = _fill_for_filters(data.copy(), mask)
        filtered = remove_line_hum(prepared, self.fs, self.config.line, start_sample=0)
        filtered = remove_slow_drift(filtered, self.fs, self.config.drift)
        clean, final_mask = interpolate_short_gaps(
            filtered,
            mask,
            self.fs,
            self.config.interpolate,
            immutable=self._stim_mask,
        )

        delta_rms = np.sqrt(np.mean((clean - self._full_original) ** 2))
        if delta_rms <= _IDEMPOTENCE_TOL:
            clean = self._full_original.copy()
            final_mask = mask.copy()

        self._clean_full = clean
        self._mask_full = final_mask
        self._refresh_released_segments()
        self._update_state(summary)

    def _refresh_released_segments(self) -> None:
        for start, end, seg_clean, seg_mask in self._segments:
            seg_clean[:] = self._clean_full[:, start:end]
            seg_mask[:] = self._mask_full[:, start:end]

    def _update_state(self, summary: DetectionSummary) -> None:
        self.state.clip_counts = summary.clip_counts.copy()
        self.state.flatline_counts = summary.flatline_counts.copy()
        self.state.stim_counts = summary.stim_counts.copy()

    def _release_ready(self) -> tuple[np.ndarray, np.ndarray]:
        if self._clean_full is None:
            empty = np.zeros((self._channels or 0, 0), dtype=np.float32)
            empty_mask = np.zeros_like(empty, dtype=bool)
            return empty, empty_mask

        cutoff = max(self._released, self._clean_full.shape[1] - self._drift_half)
        if cutoff <= self._released:
            empty = np.zeros((self._channels, 0), dtype=np.float32)
            empty_mask = np.zeros_like(empty, dtype=bool)
            return empty, empty_mask

        segment_clean = self._clean_full[:, self._released:cutoff].copy()
        segment_mask = self._mask_full[:, self._released:cutoff].copy()
        self._segments.append((self._released, cutoff, segment_clean, segment_mask))
        self._released = cutoff
        return segment_clean, segment_mask


def _fill_for_filters(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    filled = x.copy()
    C, _ = x.shape
    for ch in range(C):
        bad_idx = np.flatnonzero(mask[ch])
        if bad_idx.size == 0:
            continue
        good_idx = np.flatnonzero(~mask[ch])
        if good_idx.size == 0:
            filled[ch, bad_idx] = 0.0
            continue
        filled[ch, bad_idx] = np.interp(
            bad_idx,
            good_idx,
            filled[ch, good_idx],
            left=filled[ch, good_idx[0]],
            right=filled[ch, good_idx[-1]],
        )
    return filled
