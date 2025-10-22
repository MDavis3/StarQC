# StarQC

StarQC is the Starfish Artifact Removal & QC Library. It turns raw, multi-channel neural recordings into clean, reproducible data products using a simple, transparent pipeline that prioritises via negativa: remove obvious junk first, stay deterministic, and keep dependencies light.

## Why StarQC?
- **Deterministic**: chunked and batch results stay within 1e-6 RMS.
- **Lean**: pure NumPy/SciPy with optional numba acceleration hooks.
- **Trustworthy**: every run emits a provenance record you can audit.
- **Portable**: designed for laptop-class CPUs and low-power lab rigs.

## Quickstart

Install StarQC in a fresh environment:

```bash
pip install -e .[dev]
```

Clean a NumPy voltage file and generate a report:

```bash
starqc clean data.npy --fs 1000 --out clean.npy --report report.json
```

Use the Python API for in-memory arrays:

```python
import numpy as np
from starqc.clean import clean_lfp

x = np.load("data.npy").astype(np.float32)
clean, report = clean_lfp(x, fs=1000.0, stim_times_s=[1.23, 2.5])
good_channels = [cid for cid, flag in report["flags"].items() if flag["pass"]]
```

See `docs/QUICKSTART.md` for end-to-end walkthroughs and fixture recipes.

## Configuration tips

Tune mains removal and QC thresholds from Python:

```python
from dataclasses import replace
from starqc.config import get_default_config

cfg = get_default_config()
# Example: remove 60/120/180 Hz in labs with strong mains
cfg = replace(cfg, line=replace(cfg.line, harmonics=3))
# Example: keep default QC thresholds (line_ratio≤0.2, etc.)
```

CLI with known voltage rails and stim times:

```bash
starqc clean data.npy \
  --fs 1000 \
  --stim stim.csv \
  --out clean.npy \
  --report report.json \
  --voltage-range -500 500
```

Note on demos: synthetic fixtures with modest oscillations can show low SNR proxy even when cleaning is correct. Focus first on line_ratio, drift_index, and masked_frac; adjust harmonics as needed.

## MVP notes (for reviewers)

- Purpose: deterministic, streaming-safe artifact removal and simple QC for early LFP data.
- What to look at first: line_ratio, drift_index, masked_frac; then SNR proxy and stationarity.
- Demo config: harmonics=3 for 60/120/180 Hz; default QC thresholds kept.
- Caveat: synthetic demo has modest oscillation, so SNR proxy may read low despite correct cleaning.
- Why now: reproducible cleaning + audit-ready reports de-risk early experiments and speed up closed-loop tests.
