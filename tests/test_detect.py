"""Tests for artifact detection."""

from dataclasses import replace

import numpy as np
import pytest

from starqc import config
from starqc.detect import detect_artifacts


def test_detects_clipping_and_stim_window():
    cfg = config.get_default_config()
    fs = 1000.0
    samples = 2000
    channels = 2
    rng = np.random.default_rng(0)
    x = rng.standard_normal((channels, samples)).astype(np.float32)

    # Inject clipping on channel 0.
    x[0, 500:520] = 490.0

    stim_time = 0.8
    mask, summary, _ = detect_artifacts(
        x,
        fs,
        cfg.detect,
        stim_times_s=[stim_time],
        voltage_range=(-500.0, 500.0),
    )

    assert mask[0, 500:520].all()
    stim_idx = int(round(stim_time * fs))
    pad = int(round(cfg.detect.pad_ms * fs / 1000.0))
    assert mask[:, stim_idx - pad : stim_idx + pad + 1].all()
    assert summary.clip_counts[0] >= 20


def test_detects_flatline_segments():
    cfg = config.get_default_config()
    fs = 1000.0
    samples = 1500
    x = np.zeros((1, samples), dtype=np.float32)
    x[0, :500] = np.linspace(-10, 10, 500, dtype=np.float32)
    x[0, 500:600] = 5.0  # flatline
    x[0, 600:] = np.linspace(5, -5, samples - 600, dtype=np.float32)

    mask, summary, _ = detect_artifacts(
        x,
        fs,
        cfg.detect,
        stim_times_s=None,
        voltage_range=(-20.0, 20.0),
    )

    assert mask[0, 500:600].any()
    assert summary.flatline_counts[0] >= 50


def test_drop_short_runs_removes_spurious_mask():
    cfg = config.get_default_config()
    detect_cfg = replace(cfg.detect, min_mask_run_ms=10.0)
    fs = 1000.0
    rng = np.random.default_rng(42)
    x = rng.normal(scale=0.1, size=(1, 200)).astype(np.float32)
    x[0, 50:52] = 500.0  # 2 ms clip

    mask, _, _ = detect_artifacts(
        x,
        fs,
        detect_cfg,
        voltage_range=(-500.0, 500.0),
    )

    assert not mask.any()


def test_flatline_epsilon_controls_sensitivity():
    cfg = config.get_default_config()
    detect_cfg = replace(cfg.detect, flatline_epsilon=0.05)
    fs = 1000.0
    samples = 500
    baseline = np.linspace(0, 1, samples, dtype=np.float32)
    x = (baseline + 1e-3).astype(np.float32)
    data = x[None, :]

    mask, _, _ = detect_artifacts(
        data,
        fs,
        detect_cfg,
        voltage_range=(0.0, 1.0),
    )

    assert mask.any()


def test_clip_threshold_guard():
    cfg = config.get_default_config()
    bad_cfg = replace(cfg.detect, clip_threshold=0.4)
    fs = 1000.0
    x = np.zeros((1, 10), dtype=np.float32)

    with pytest.raises(ValueError):
        detect_artifacts(x, fs, bad_cfg)


def test_validate_input_errors():
    cfg = config.get_default_config()
    fs = 1000.0
    with pytest.raises(ValueError):
        detect_artifacts([0.0, 1.0], fs, cfg.detect)  # not ndarray
    with pytest.raises(ValueError):
        detect_artifacts(np.zeros((1, 10), dtype=np.float64), fs, cfg.detect)
    with pytest.raises(ValueError):
        detect_artifacts(np.zeros(10, dtype=np.float32), fs, cfg.detect)
    with pytest.raises(ValueError):
        detect_artifacts(np.zeros((1, 10), dtype=np.float32), -fs, cfg.detect)
