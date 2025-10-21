# Quickstart

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## 2. Prepare Data

Store your channels × samples float32 array in NumPy format:

```python
import numpy as np
np.save("lfp.npy", lfp.astype(np.float32))
```

Optional stimulus times (seconds) go in a CSV:

```text
1.23, 2.50, 4.75
```

## 3. Run the CLI

```bash
starqc clean lfp.npy --fs 1000 --stim stim.csv --out clean.npy --report report.json
```

The command writes the filtered array to `clean.npy` and a JSON report (mask, metrics, flags, provenance).

## 4. Use the Python API

```python
import numpy as np
from starqc.clean import clean_lfp

lfp = np.load("lfp.npy").astype(np.float32)
clean, report = clean_lfp(lfp, fs=1000.0, stim_times_s=[1.23, 2.50])
print(report["flags"])
```

## 5. Streaming Mode

```python
from starqc.stream import StreamingCleaner

cleaner = StreamingCleaner(fs=1000.0, stim_times_s=[1.23])
chunks = []
for start in range(0, lfp.shape[1], 250):
    chunk, mask = cleaner.process_chunk(lfp[:, start:start+250])
    chunks.append(chunk)
clean_stream = np.concatenate(chunks, axis=1)
```

## 6. Run Tests and Coverage

```bash
pytest -q
```

## 7. Provenance & Reports

`report.json` contains cleaning parameters, git commit, runtime, mask, and per-channel metrics. Append it to lab notebooks for reproducibility.
