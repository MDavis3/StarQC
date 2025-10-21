"""Streaming parity tests."""

import numpy as np

from starqc import config
from starqc.clean import clean_lfp
from starqc.simulate import SimulationConfig, base_signal, inject_line_hum, inject_slow_drift
from starqc.stream import StreamingCleaner


_DEF_FS = 1000.0


def _expected_tail_samples(cfg: config.PipelineConfig, fs: float) -> int:
    window = max(int(round(fs * cfg.drift.median_window_s)), 1)
    if window % 2 == 0:
        window += 1
    return window // 2


def test_streaming_matches_batch_processing():
    cfg = config.get_default_config()
    fs = _DEF_FS
    samples = 8000
    base = base_signal(SimulationConfig(channels=2, samples=samples, fs=fs))
    contaminated = inject_line_hum(base, fs, amplitude=40.0)
    contaminated = inject_slow_drift(contaminated, fs, amplitude=120.0)

    cleaner = StreamingCleaner(
        fs=fs,
        config=cfg,
        stim_times_s=[2.5],
        voltage_range=(-500.0, 500.0),
    )

    chunk = int(fs * 0.25)
    stream_clean = []
    stream_mask = []
    tail_size = _expected_tail_samples(cfg, fs)

    for start in range(0, samples, chunk):
        stop = min(start + chunk, samples)
        clean_chunk, mask_chunk = cleaner.process_chunk(contaminated[:, start:stop])
        if clean_chunk.size:
            stream_clean.append(clean_chunk)
            stream_mask.append(mask_chunk)

    final_clean, final_mask = cleaner.finalize()
    assert final_clean.shape[1] == tail_size
    if final_clean.size:
        stream_clean.append(final_clean)
        stream_mask.append(final_mask)

    clean_stream = np.concatenate(stream_clean, axis=1)
    mask_stream = np.concatenate(stream_mask, axis=1)

    clean_batch, report = clean_lfp(
        contaminated,
        fs=fs,
        stim_times_s=[2.5],
        voltage_range=(-500.0, 500.0),
        config=cfg,
    )

    mask_batch = report["mask"]
    assert clean_stream.shape == clean_batch.shape
    diff = clean_stream - clean_batch
    rms = np.sqrt(np.mean(diff[~mask_batch] ** 2))

    assert rms < 1e-6
    assert np.all(mask_stream == mask_batch)


def test_streaming_alignment_with_initial_offset():
    cfg = config.get_default_config()
    fs = _DEF_FS
    samples = 6000
    base = base_signal(SimulationConfig(channels=2, samples=samples, fs=fs))
    contaminated = inject_line_hum(base, fs, amplitude=35.0)
    contaminated = inject_slow_drift(contaminated, fs, amplitude=90.0)

    # Prepend a short block of zeros to force a non-zero start_sample for later chunks.
    pad = np.zeros((2, 200), dtype=np.float32)
    padded = np.concatenate([pad, contaminated], axis=1)

    cleaner = StreamingCleaner(fs=fs, config=cfg, voltage_range=(-500.0, 500.0))
    chunk = 300
    stream_clean = []
    stream_mask = []
    for start in range(0, padded.shape[1], chunk):
        stop = min(start + chunk, padded.shape[1])
        clean_chunk, mask_chunk = cleaner.process_chunk(padded[:, start:stop])
        if clean_chunk.size:
            stream_clean.append(clean_chunk)
            stream_mask.append(mask_chunk)
    tail_clean, tail_mask = cleaner.finalize()
    if tail_clean.size:
        stream_clean.append(tail_clean)
        stream_mask.append(tail_mask)

    clean_stream = np.concatenate(stream_clean, axis=1)
    mask_stream = np.concatenate(stream_mask, axis=1)

    clean_batch, report = clean_lfp(
        padded,
        fs=fs,
        stim_times_s=None,
        voltage_range=(-500.0, 500.0),
        config=cfg,
    )

    mask_batch = report["mask"]
    diff = clean_stream - clean_batch
    rms = np.sqrt(np.mean(diff[~mask_batch] ** 2))

    assert rms < 1e-6
    assert np.all(mask_stream == mask_batch)
