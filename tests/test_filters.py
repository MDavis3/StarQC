"""Tests for filtering utilities."""

from dataclasses import replace

import numpy as np
from scipy import signal

from starqc import config
from starqc.filters import remove_line_hum, remove_slow_drift
from starqc.simulate import SimulationConfig, base_signal, inject_line_hum, inject_slow_drift


def _welch_power(trace: np.ndarray, fs: float, target: float, window: float = 1.0) -> float:
    freqs, psd = signal.welch(trace, fs=fs, nperseg=1024)
    mask = (freqs >= target - window) & (freqs <= target + window)
    return float(np.trapz(psd[mask], freqs[mask]))


def test_remove_line_hum_reduces_60hz_power():
    cfg = config.get_default_config()
    fs = 1000.0
    base = base_signal(SimulationConfig(channels=1, samples=5000, fs=fs))
    contaminated = inject_line_hum(base, fs, amplitude=50.0, freq=60.0)

    before = _welch_power(contaminated[0], fs, 60.0)
    cleaned = remove_line_hum(contaminated, fs, cfg.line)
    after = _welch_power(cleaned[0], fs, 60.0)

    assert after <= before * 0.2


def test_remove_line_hum_is_idempotent():
    cfg = config.get_default_config()
    fs = 1000.0
    base = base_signal(SimulationConfig(channels=1, samples=4000, fs=fs))
    contaminated = inject_line_hum(base, fs, amplitude=40.0)

    once = remove_line_hum(contaminated, fs, cfg.line)
    twice = remove_line_hum(once, fs, cfg.line)

    delta = np.sqrt(np.mean((twice - once) ** 2))
    assert delta < 1e-7


def test_remove_line_hum_harmonics():
    cfg = config.get_default_config()
    cfg = config.PipelineConfig(
        standardize=cfg.standardize,
        detect=cfg.detect,
        line=replace(cfg.line, harmonics=2),
        drift=cfg.drift,
        interpolate=cfg.interpolate,
        qc=cfg.qc,
    )
    fs = 1000.0
    base = base_signal(SimulationConfig(channels=1, samples=6000, fs=fs))
    contaminated = inject_line_hum(base, fs, amplitude=40.0, freq=60.0)
    contaminated = inject_line_hum(contaminated, fs, amplitude=30.0, freq=120.0)

    before_60 = _welch_power(contaminated[0], fs, 60.0)
    before_120 = _welch_power(contaminated[0], fs, 120.0)

    cleaned = remove_line_hum(contaminated, fs, cfg.line)

    after_60 = _welch_power(cleaned[0], fs, 60.0)
    after_120 = _welch_power(cleaned[0], fs, 120.0)

    assert after_60 <= before_60 * 0.2
    assert after_120 <= before_120 * 0.2


def test_remove_slow_drift_halves_low_frequency_power():
    cfg = config.get_default_config()
    fs = 1000.0
    base = base_signal(SimulationConfig(channels=1, samples=4000, fs=fs))
    contaminated = inject_slow_drift(base, fs, amplitude=200.0, freq=0.2)

    freqs, psd = signal.welch(contaminated[0], fs=fs, nperseg=2048)
    low_power_before = float(np.trapz(psd[freqs <= 1.0], freqs[freqs <= 1.0]))

    cleaned = remove_slow_drift(contaminated, fs, cfg.drift)

    freqs_c, psd_c = signal.welch(cleaned[0], fs=fs, nperseg=2048)
    low_power_after = float(np.trapz(psd_c[freqs_c <= 1.0], freqs_c[freqs_c <= 1.0]))

    assert low_power_after <= low_power_before * 0.5


def test_highpass_drift_path():
    cfg = config.get_default_config()
    drift_cfg = replace(cfg.drift, detrend_method="highpass", highpass_hz=1.0)
    fs = 1000.0
    base = base_signal(SimulationConfig(channels=1, samples=3000, fs=fs))
    contaminated = inject_slow_drift(base, fs, amplitude=100.0, freq=0.3)

    freqs_b, psd_b = signal.welch(contaminated[0], fs=fs, nperseg=1024)
    low_before = float(np.trapz(psd_b[freqs_b <= 1.0], freqs_b[freqs_b <= 1.0]))

    cleaned = remove_slow_drift(contaminated, fs, drift_cfg)
    freqs_a, psd_a = signal.welch(cleaned[0], fs=fs, nperseg=1024)
    low_after = float(np.trapz(psd_a[freqs_a <= 1.0], freqs_a[freqs_a <= 1.0]))

    assert low_after < low_before * 0.5


def test_drift_skip_when_clean():
    cfg = config.get_default_config()
    fs = 1000.0
    zeros = np.zeros((2, 500), dtype=np.float32)

    cleaned = remove_slow_drift(zeros, fs, cfg.drift)
    assert np.shares_memory(cleaned, zeros) or np.allclose(cleaned, zeros)


def test_highpass_skip_when_clean():
    cfg = config.get_default_config()
    drift_cfg = replace(cfg.drift, detrend_method="highpass", highpass_hz=1.0)
    fs = 1000.0
    signal_in = np.zeros((1, 800), dtype=np.float32)

    cleaned = remove_slow_drift(signal_in, fs, drift_cfg)
    assert np.allclose(cleaned, signal_in)
