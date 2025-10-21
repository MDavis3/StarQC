# Architecture

## Module Overview
- `starqc.clean`: orchestrates the pipeline (standardise → detect → filters → interpolate → QC → report).
- `starqc.detect`: clipping, flatline, and stimulation masks.
- `starqc.filters`: common-median reference, sinusoid regression notch, drift removal.
- `starqc.interpolate`: short-gap interpolation helpers.
- `starqc.qc`: PSD-based metrics and pass/fail logic.
- `starqc.stream`: streaming/chunked processing wrapper.
- `starqc.simulate`: synthetic fixture generators for tests.
- `starqc.provenance`: runtime+version reporting utilities.
- `starqc.config`: structured defaults for every stage.
- `starqc.cli`: console entry-point.

## Data Flow
```
          +-------------+
          |  Input x    |
          +------+------+
                 |
                 v
       +---------+----------+
       |  Standardise       |
       +---------+----------+
                 |
                 v
       +---------+----------+
       | Detect & Mask      |
       +----+----------+----+
            |          |
            | mask     v
            |   +------+------+
            |   | Interpolate |
            |   +------+------+
            |          |
            v          v
    +-------+------+ +--------+-------+
    | Filters (hum) | | Filters (drift)|
    +-------+------+ +--------+-------+
                 |         (cleaned data)
                 +------------------+
                                    v
                             +------+------+
                             | QC Metrics  |
                             +------+------+
                                    |
                                    v
                             +------+------+
                             | Report +    |
                             | Provenance  |
                             +-------------+
```

## Streaming Strategy
- Maintains sample offset to keep notch regression phase-aligned.
- Converts stimulation times into per-chunk relative offsets with pad spillover.
- Fills masked samples before filtering to avoid ringing; interpolation clears short gaps after filtering.
- Accumulates detection counts to match batch reporting.

## Provenance
`build_provenance` serialises configuration dictionaries, sampling rate, shape, runtime, package version, git commit, Python/NumPy versions, and UTC timestamp.

## Extensibility
- Add advanced detectors by extending `DetectConfig` and composing in `detect_artifacts`.
- Additional QC metrics plug into `qc.compute_qc` without touching the pipeline orchestrator.
- Future viz helpers live in a dedicated `viz/` package to keep core dependency-light.
