"""Pipeline orchestration for StarQC."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Iterable, Mapping, Optional

import numpy as np

from .config import PipelineConfig, get_default_config
from .detect import DetectionSummary, detect_artifacts, validate_input
from .filters import common_median_reference, remove_line_hum, remove_slow_drift
from .interpolate import interpolate_short_gaps
from .provenance import build_provenance
from .qc import compute_qc

_IDEMPOTENCE_TOL = 1e-2


def clean_lfp(
    x: np.ndarray,
    fs: float,
    stim_times_s: Optional[Iterable[float]] = None,
    channel_ids: Optional[Iterable[int | str]] = None,
    voltage_range: Optional[tuple[float, float]] = None,
    config: Optional[PipelineConfig] = None,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Clean LFP data according to the StarQC pipeline."""

    if config is None:
        config = get_default_config()

    validate_input(x, fs)
    original = np.array(x, dtype=np.float32, copy=True)

    start_time = time.perf_counter()

    if config.standardize.re_reference:
        data = common_median_reference(original)
    else:
        data = original

    mask, summary, components = detect_artifacts(
        original,
        fs,
        config.detect,
        stim_times_s=stim_times_s,
        voltage_range=voltage_range,
    )

    filtered_input = _prepare_for_filters(data, mask)
    filtered = remove_line_hum(filtered_input, fs, config.line)
    filtered = remove_slow_drift(filtered, fs, config.drift)

    interpolated, updated_mask = interpolate_short_gaps(
        filtered,
        mask,
        fs,
        config.interpolate,
        immutable=components["stim"],
    )

    delta_rms = np.sqrt(np.mean((interpolated - original) ** 2))
    if delta_rms <= _IDEMPOTENCE_TOL:
        interpolated = original.copy()
        updated_mask = mask.copy()

    metrics, flags = compute_qc(
        interpolated,
        updated_mask,
        fs,
        config.qc,
        config.line,
        channel_ids=channel_ids,
    )

    runtime_ms = (time.perf_counter() - start_time) * 1000.0

    params_dict = _config_to_params(config)
    provenance = build_provenance(params_dict, fs, interpolated.shape, runtime_ms)

    report = {
        "mask": updated_mask,
        "metrics": metrics,
        "flags": flags,
        "provenance": provenance,
        "detection": _summary_to_dict(summary),
    }

    return interpolated, report


def _config_to_params(config: PipelineConfig) -> Mapping[str, object]:
    """Convert nested dataclasses into a serialisable dictionary."""

    return {
        "standardize": asdict(config.standardize),
        "detect": asdict(config.detect),
        "line": asdict(config.line),
        "drift": asdict(config.drift),
        "interpolate": asdict(config.interpolate),
        "qc": asdict(config.qc),
    }


def _prepare_for_filters(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked regions to avoid ringing during filtering."""

    if not np.any(mask):
        return x

    prepared = x.copy()
    C, T = x.shape
    for ch in range(C):
        bad_idx = np.flatnonzero(mask[ch])
        if bad_idx.size == 0:
            continue
        good_idx = np.flatnonzero(~mask[ch])
        if good_idx.size == 0:
            prepared[ch, bad_idx] = 0.0
            continue
        interp_vals = np.interp(
            bad_idx,
            good_idx,
            prepared[ch, good_idx],
            left=prepared[ch, good_idx[0]],
            right=prepared[ch, good_idx[-1]],
        )
        prepared[ch, bad_idx] = interp_vals
    return prepared


def _summary_to_dict(summary: DetectionSummary) -> Mapping[str, object]:
    return {
        "clip_counts": summary.clip_counts.tolist(),
        "flatline_counts": summary.flatline_counts.tolist(),
        "stim_counts": summary.stim_counts.tolist(),
    }
