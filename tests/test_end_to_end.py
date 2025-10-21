"""End-to-end pipeline tests."""

import numpy as np

from starqc import config
from starqc.clean import clean_lfp
from starqc.qc import compute_qc
from starqc.simulate import (
    SimulationConfig,
    base_signal,
    inject_clipping,
    inject_dropout,
    inject_line_hum,
    inject_slow_drift,
    inject_stim,
)


def test_pipeline_meets_acceptance_targets():
    cfg = config.get_default_config()
    fs = 1000.0
    samples = 10000

    base = base_signal(SimulationConfig(channels=2, samples=samples, fs=fs))
    data = inject_line_hum(base, fs, amplitude=40.0)
    data = inject_slow_drift(data, fs, amplitude=500.0)
    data, stim_indices = inject_stim(data, fs, [2.0, 4.5], magnitude=300.0)

    # Short dropout (80 ms) and long dropout (200 ms)
    short_start = int(1.5 * fs)
    short_end = short_start + int(0.08 * fs)
    data[:, short_start:short_end] = data[:, short_start][:, None]

    long_start = 6000
    long_end = long_start + int(0.2 * fs)
    data[:, long_start:long_end] = data[:, long_start][:, None]

    # Clipping burst
    data[:, 1000:1010] = 480.0

    raw_mask = np.zeros_like(data, dtype=bool)
    raw_metrics, _ = compute_qc(data, raw_mask, fs, cfg.qc, cfg.line)

    clean, report = clean_lfp(
        data,
        fs=fs,
        stim_times_s=[2.0, 4.5],
        voltage_range=(-500.0, 500.0),
        config=cfg,
    )

    metrics = report["metrics"]
    ch0 = metrics[0]

    assert ch0["line_ratio"] <= raw_metrics[0]["line_ratio"] * 0.2
    assert ch0["drift_index"] <= raw_metrics[0]["drift_index"] * 0.5
    assert ch0["snr_proxy"] >= raw_metrics[0]["snr_proxy"] * 1.3

    final_mask = report["mask"]
    assert not final_mask[:, short_start:short_end].any()
    assert final_mask[:, long_start:long_end].all()

    pad = int(round(cfg.detect.pad_ms * fs / 1000.0))
    for stim in stim_indices:
        lo = max(stim - pad, 0)
        hi = min(stim + pad + 1, clean.shape[1])
        assert final_mask[:, lo:hi].all()

    clean2, _ = clean_lfp(
        clean,
        fs=fs,
        stim_times_s=[2.0, 4.5],
        voltage_range=(-500.0, 500.0),
        config=cfg,
    )

    diff = clean2 - clean
    rms = np.sqrt(np.mean(diff[~final_mask] ** 2))
    assert rms < 1e-6
