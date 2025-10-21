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
