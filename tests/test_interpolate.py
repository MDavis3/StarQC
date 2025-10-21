"""Tests for interpolation helpers."""

import numpy as np

from starqc import config
from starqc.interpolate import interpolate_short_gaps


def test_interpolates_short_gaps_and_leaves_long_ones():
    cfg = config.get_default_config()
    fs = 1000.0
    data = np.linspace(0, 1, 200, dtype=np.float32)[None, :]
    mask = np.zeros_like(data, dtype=bool)

    # Short gap (25 samples -> 25 ms)
    start_short, end_short = 50, 75
    mask[0, start_short:end_short] = True

    # Long gap (200 samples -> 200 ms)
    start_long, end_long = 120, 320
    mask = np.pad(mask, ((0, 0), (0, 200)), mode="constant")
    data = np.pad(data, ((0, 0), (0, 200)), mode="edge")
    mask[0, start_long:end_long] = True

    filled, updated_mask = interpolate_short_gaps(data, mask, fs, cfg.interpolate)

    assert not updated_mask[0, start_short:end_short].any()
    assert updated_mask[0, start_long:end_long].all()
    assert not np.isclose(filled[0, start_short], data[0, start_short - 1])
