import json
import numpy as np
from starqc.config import get_default_config
from starqc.clean import clean_lfp
from starqc.simulate import SimulationConfig, base_signal, inject_line_hum, inject_slow_drift, inject_stim

fs = 1000.0
samples = 5000
cfg = get_default_config()
base = base_signal(SimulationConfig(channels=2, samples=samples, fs=fs))
data = inject_line_hum(base, fs, amplitude=40.0)
data = inject_slow_drift(data, fs, amplitude=200.0)
data, stim_idx = inject_stim(data, fs, [2.0, 4.0], magnitude=300.0)

clean, report = clean_lfp(data.astype(np.float32), fs=fs, stim_times_s=[2.0, 4.0], voltage_range=(-500.0, 500.0), config=cfg)

np.save('demo_clean.npy', clean)
report_json = {
  'mask': report['mask'].tolist(),
  'metrics': {str(k): v for k, v in report['metrics'].items()},
  'flags': {str(k): v for k, v in report['flags'].items()},
  'provenance': report['provenance']
}
with open('demo_report.json', 'w', encoding='utf-8') as f:
    json.dump(report_json, f, indent=2)

print('Clean shape:', clean.shape)
print('Channel 0 metrics:', report['metrics'][0])
print('Channel 1 metrics:', report['metrics'][1])
print('Flags:', report['flags'])
print('Runtime ms:', report['provenance']['runtime_ms'])
print('Artifacts saved: demo_clean.npy, demo_report.json')
