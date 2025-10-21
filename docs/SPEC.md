# StarQC Specification

## Mission
Turn raw, multi-channel neural recordings into clean, scored, reproducible data using the simplest working methods. Emphasise via negativa: remove obvious junk first, avoid fragile heuristics, and guarantee determinism.

## Core Constraints
- Inputs: `x: float32[C, T]`, `fs: float`, optional `stim_times_s`, `channel_ids`, `voltage_range`.
- Output: `(clean: float32[C, T], report)` with `mask`, per-channel metrics, pass/fail flags, provenance record.
- Deterministic and idempotent (`<1e-6` RMS on re-run ignoring masked samples).
- Streaming parity: chunk and batch outputs match within `1e-6` RMS and metrics within 1%.
- Performance budgets: ≥20× realtime for 32 ch @1 kHz, chunk latency <50 ms, memory ≤4× input buffer.

## Pipeline Modules
1. **Standardise**: validate shapes, optional common-median re-reference.
2. **Detect & Mask**: clipping, flatlines/dropouts, stimulation pads. Returns boolean mask.
3. **Remove Line Hum**: regression-based sinusoid subtraction at 60 Hz (+ harmonics on demand).
4. **Remove Slow Drift**: median-window subtraction (default) or 0.5 Hz high-pass.
5. **Interpolate Short Gaps**: linear/cubic fills for spans ≤100 ms (default); longer gaps remain masked.
6. **QC Metrics**: per-channel line ratio, drift index, masked fraction, SNR proxy, stationarity proxy; pass/fail using `config.QCThresholds`.
7. **Report & Provenance**: include mask, metrics, flags, runtime, params, git hash, versions.

## Public APIs

```python
from starqc.clean import clean_lfp

clean, report = clean_lfp(
    x: np.ndarray,
    fs: float,
    stim_times_s: Optional[Iterable[float]] = None,
    channel_ids: Optional[Iterable[int | str]] = None,
    voltage_range: Optional[tuple[float, float]] = None,
    config: Optional[PipelineConfig] = None,
)
```

```python
from starqc.stream import StreamingCleaner

cleaner = StreamingCleaner(fs, config=None, stim_times_s=None, voltage_range=None)
chunk_clean, chunk_mask = cleaner.process_chunk(chunk: np.ndarray)
summary = cleaner.detection_summary()
```

```bash
starqc clean <input.npy> --fs <Hz> --stim stim.csv --out clean.npy --report report.json
```

## Configuration Defaults
- Standardise: re-reference = True.
- Detect: `pad_ms=3`, dynamic clipping threshold 98% of range, `flatline_ms=20`, epsilon `1e-6`, `min_mask_run_ms=0`.
- Line removal: `notch_hz=60`, `harmonics=1`.
- Drift: `detrend_method='median'`, `median_window_s=1`, `highpass_hz=0.5`.
- Interpolation: `interpolate_max_ms=100`, `method='linear'`.
- QC thresholds: `line_ratio≤0.2`, `drift_index≤0.15`, `masked_frac≤0.1`, `snr_proxy≥2.0`, `stationarity≤0.35`.

## Reports
```json
{
  "mask": [[bool, ...]],
  "metrics": {"0": {...}},
  "flags": {"0": {"pass": true, "reasons": []}},
  "detection": {"clip_counts": [...], ...},
  "provenance": {"package_version": "0.1.0", "git_commit": "...", ...}
}
```

## Acceptance Criteria
- Line-hum injected at 60 Hz reduces line-ratio ≥80%.
- Slow drift power halves after detrending.
- SNR proxy improves by ≥30% on oscillatory fixtures.
- Gaps ≤ threshold interpolated; longer gaps remain masked.
- Streaming parity and idempotence tolerances as above.

## Backlog (not implemented yet)
- Auto-detect higher harmonics.
- Adaptive notch tracking.
- Cross-channel QC metrics.
- Visualisation helpers in `viz/`.
