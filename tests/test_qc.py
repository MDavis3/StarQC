"""QC metric tests."""

from dataclasses import replace

import numpy as np
import pytest

from starqc import config
from starqc.clean import clean_lfp
from starqc.qc import compute_qc
from starqc.simulate import (
    SimulationConfig,
    base_signal,
    inject_line_hum,
    inject_slow_drift,
)


def test_qc_metrics_improve_after_cleaning():
    cfg = config.get_default_config()
    fs = 1000.0
    base = base_signal(SimulationConfig(channels=1, samples=6000, fs=fs))
    contaminated = inject_line_hum(base, fs, amplitude=60.0)
    contaminated = inject_slow_drift(contaminated, fs, amplitude=150.0)
    mask = np.zeros_like(contaminated, dtype=bool)

    raw_metrics, _ = compute_qc(contaminated, mask, fs, cfg.qc, cfg.line)

    cleaned, report = clean_lfp(
        contaminated,
        fs=fs,
        stim_times_s=None,
        voltage_range=(-500.0, 500.0),
        config=cfg,
    )

    metrics_after = report["metrics"][0]

    assert metrics_after["line_ratio"] < raw_metrics[0]["line_ratio"]
    assert metrics_after["drift_index"] < raw_metrics[0]["drift_index"]
    assert metrics_after["snr_proxy"] > raw_metrics[0]["snr_proxy"]


def test_qc_flags_thresholds_trigger():
    cfg = config.get_default_config()
    fs = 1000.0
    samples = 4000
    t = np.arange(samples, dtype=np.float32) / fs
    signal_in = 50.0 * np.sin(2 * np.pi * 60.0 * t) + 200.0 * np.sin(2 * np.pi * 0.2 * t)
    data = signal_in[None, :].astype(np.float32)
    mask = np.zeros_like(data, dtype=bool)

    metrics, flags = compute_qc(data, mask, fs, cfg.qc, cfg.line)
    reasons = flags[0]["reasons"]
    assert any(reason.startswith("line_ratio") for reason in reasons)
    assert any(reason.startswith("drift_index") for reason in reasons)


def test_qc_handles_channel_ids_and_full_mask():
    cfg = config.get_default_config()
    fs = 1000.0
    samples = 512
    data = np.zeros((1, samples), dtype=np.float32)
    mask = np.ones_like(data, dtype=bool)

    metrics, flags = compute_qc(data, mask, fs, cfg.qc, cfg.line, channel_ids=["ch0"])
    assert "ch0" in metrics
    assert not flags["ch0"]["pass"] and any(
        "masked_frac" in reason for reason in flags["ch0"]["reasons"]
    )


def test_qc_shape_and_channel_validation():
    cfg = config.get_default_config()
    fs = 1000.0
    data = np.zeros((1, 10), dtype=np.float32)
    mask = np.zeros((1, 9), dtype=bool)
    with pytest.raises(ValueError):
        compute_qc(data, mask, fs, cfg.qc, cfg.line)
    with pytest.raises(ValueError):
        compute_qc(data, np.zeros_like(data, dtype=bool), fs, cfg.qc, cfg.line, channel_ids=[0, 1])


def test_line_ratio_zero_when_disabled():
    cfg = config.get_default_config()
    fs = 1000.0
    data = np.zeros((1, 1024), dtype=np.float32)
    mask = np.zeros_like(data, dtype=bool)
    line_cfg = replace(cfg.line, notch_hz=0.0)
    metrics, _ = compute_qc(data, mask, fs, cfg.qc, line_cfg)
    assert metrics[0]["line_ratio"] == 0.0
