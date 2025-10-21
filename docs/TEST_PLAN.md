# Test Plan

## Scope
Covers unit, property, streaming, and performance smoke tests for the StarQC pipeline.

## Fixtures
- Synthetic 1/f noise with optional oscillation (`simulate.base_signal`).
- Additive mains hum at 60 Hz (`inject_line_hum`).
- Slow drift (<0.2 Hz) via sine (`inject_slow_drift`).
- Stimulation pulses with ring-down (`inject_stim`).
- Clipping bursts (`inject_clipping`).
- Dropout/flatline spans (`inject_dropout`).

## Unit Tests
- `tests/test_detect.py`: clipping, flatline, and stim padding detection.
- `tests/test_filters.py`: Welch power ratios for line hum; drift power reduction.
- `tests/test_interpolate.py`: short-gap interpolation clears mask; long gaps persist.
- `tests/test_qc.py`: QC metrics monotonic improvements after cleaning.

## Integration & Streaming
- `tests/test_streaming.py`: chunked pipeline parity with batch output (RMS <1e-6, identical mask).
- `tests/test_end_to_end.py`: acceptance criteria cross-check (line ratio drop, drift reduction, SNR gain, interpolation honesty, idempotence, stim masking).

## Property Checks
- Idempotence validated in end-to-end test.
- Streaming parity ensures determinism.
- Line ratio proportionality validated by comparing before/after PSD.

## Performance Smoke
- CI includes a smoke benchmark (pytest marker `perf` placeholder) ensuring 32×1e6 sample case completes under configured budgets; measured via simple timing harness (see `tests/test_end_to_end.py` docstring for manual trigger).

## Coverage Target
Coverage is enforced at ≥90% for the critical modules (`detect`, `filters`, `qc`) via pytest-cov and coverage configuration.
