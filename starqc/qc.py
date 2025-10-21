"""Quality-control metrics for StarQC."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from .config import LineRemovalConfig, QCConfig


def compute_qc(
    cleaned: np.ndarray,
    mask: np.ndarray,
    fs: float,
    qc_config: QCConfig,
    line_config: LineRemovalConfig,
    channel_ids: Iterable[int | str] | None = None,
) -> Tuple[Dict[int | str, Dict[str, float]], Dict[int | str, Dict[str, object]]]:
    """Compute metrics and pass/fail flags per channel."""

    if cleaned.shape != mask.shape:
        raise ValueError("cleaned and mask must have the same shape")

    ids: List[int | str]
    if channel_ids is None:
        ids = list(range(cleaned.shape[0]))
    else:
        ids = [int(idx) if isinstance(idx, (np.integer,)) else idx for idx in channel_ids]
        if len(ids) != cleaned.shape[0]:
            raise ValueError("channel_ids must match number of channels")

    metrics: Dict[int | str, Dict[str, float]] = {}
    flags: Dict[int | str, Dict[str, object]] = {}

    thresholds = qc_config.thresholds

    for ch, channel_id in enumerate(ids):
        channel = cleaned[ch]
        channel_mask = mask[ch]
        masked_fraction = float(np.mean(channel_mask))
        psd_freqs, psd = _welch_with_mask(channel, channel_mask, fs)

        line_ratio = _line_ratio(psd_freqs, psd, qc_config, line_config)
        drift_index = _band_power_ratio(psd_freqs, psd, 0.0, qc_config.low_freq_band)
        broadband_power = _band_power(psd_freqs, psd, *qc_config.broadband_band)
        snr_band_power = _band_power(psd_freqs, psd, *qc_config.snr_band)
        snr_proxy = snr_band_power / max(broadband_power, 1e-12)
        stationarity = _stationarity_proxy(channel, channel_mask, fs, qc_config.stationarity_window_s)

        metrics[channel_id] = {
            "line_ratio": float(line_ratio),
            "drift_index": float(drift_index),
            "masked_frac": float(masked_fraction),
            "snr_proxy": float(snr_proxy),
            "stationarity_proxy": float(stationarity),
        }

        reasons: List[str] = []
        if line_ratio > thresholds.max_line_ratio:
            reasons.append(f"line_ratio>{thresholds.max_line_ratio}")
        if drift_index > thresholds.max_drift_index:
            reasons.append(f"drift_index>{thresholds.max_drift_index}")
        if masked_fraction > thresholds.max_masked_fraction:
            reasons.append(f"masked_frac>{thresholds.max_masked_fraction}")
        if snr_proxy < thresholds.min_snr_proxy:
            reasons.append(f"snr_proxy<{thresholds.min_snr_proxy}")
        if stationarity > thresholds.max_stationarity_cv:
            reasons.append(f"stationarity>{thresholds.max_stationarity_cv}")

        flags[channel_id] = {
            "pass": len(reasons) == 0,
            "reasons": reasons,
        }

    return metrics, flags


def _welch_with_mask(
    channel: np.ndarray,
    mask: np.ndarray,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a PSD after interpolating masked samples."""

    if not np.any(mask):
        freqs, psd = signal.welch(channel, fs=fs, nperseg=min(2048, len(channel)))
        return freqs, psd

    good_idx = np.flatnonzero(~mask)
    if good_idx.size == 0:
        nperseg = min(2048, len(channel))
        if nperseg <= 0:
            return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        psd = np.zeros_like(freqs, dtype=np.float64)
        return freqs, psd

    filled = channel.copy()
    bad_idx = np.flatnonzero(mask)
    filled[bad_idx] = np.interp(
        bad_idx,
        good_idx,
        channel[good_idx],
        left=channel[good_idx[0]],
        right=channel[good_idx[-1]],
    )
    freqs, psd = signal.welch(filled, fs=fs, nperseg=min(2048, len(channel)))
    return freqs, psd


def _line_ratio(
    freqs: np.ndarray,
    psd: np.ndarray,
    qc_config: QCConfig,
    line_config: LineRemovalConfig,
) -> float:
    notch_freq = line_config.notch_hz if line_config.notch_hz else 60.0
    window = qc_config.line_window_hz
    if notch_freq <= 0:
        return 0.0

    band_mask = (freqs >= max(notch_freq - window, 0.0)) & (freqs <= notch_freq + window)
    if not np.any(band_mask):
        return 0.0
    band_freqs = freqs[band_mask]
    band_power = psd[band_mask]
    notch_idx = np.argmin(np.abs(band_freqs - notch_freq))
    notch_power = float(band_power[notch_idx])
    neighbor_power = float(np.mean(np.delete(band_power, notch_idx))) if band_power.size > 1 else notch_power
    return notch_power / max(neighbor_power, 1e-12)


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo_hz: float, hi_hz: float) -> float:
    mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _band_power_ratio(
    freqs: np.ndarray,
    psd: np.ndarray,
    lo_hz: float,
    hi_hz: float,
) -> float:
    numerator = _band_power(freqs, psd, lo_hz, hi_hz)
    denominator = _band_power(freqs, psd, hi_hz, freqs[-1])
    return numerator / max(denominator, 1e-12)


def _stationarity_proxy(
    channel: np.ndarray,
    mask: np.ndarray,
    fs: float,
    window_s: float,
) -> float:
    window = max(int(round(window_s * fs)), 4)
    if window >= len(channel):
        return 0.0

    good = ~mask
    clean = np.copy(channel)
    if np.any(~good):
        good_idx = np.flatnonzero(good)
        if good_idx.size == 0:
            return 0.0
        interp_vals = np.interp(
            np.flatnonzero(~good),
            good_idx,
            channel[good_idx],
            left=channel[good_idx[0]],
            right=channel[good_idx[-1]],
        )
        clean[~good] = interp_vals

    windows = sliding_window_view(clean, window_shape=window)
    variances = np.var(windows, axis=-1)
    mean_var = float(np.mean(variances))
    if mean_var <= 0:
        return 0.0
    return float(np.std(variances) / (mean_var + 1e-12))
