"""Configuration objects and defaults for StarQC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class StandardizeConfig:
    """Configuration for the standardization stage."""

    re_reference: bool = True


@dataclass(frozen=True)
class DetectConfig:
    """Configuration for artifact detection and masking."""

    pad_ms: float = 3.0
    clip_threshold: Optional[float] = None
    flatline_ms: float = 20.0
    flatline_epsilon: float = 1e-6
    min_mask_run_ms: float = 0.0


@dataclass(frozen=True)
class LineRemovalConfig:
    """Configuration for line-hum removal."""

    notch_hz: Optional[float] = 60.0
    harmonics: int = 1
    adaptive_harmonics: bool = False
    q_factor: float = 40.0


@dataclass(frozen=True)
class DriftRemovalConfig:
    """Configuration for baseline drift removal."""

    detrend_method: str = "median"
    median_window_s: float = 1.0
    highpass_hz: float = 0.5


@dataclass(frozen=True)
class InterpolateConfig:
    """Configuration for short-gap interpolation."""

    interpolate_max_ms: float = 100.0
    method: str = "linear"


@dataclass(frozen=True)
class QCThresholds:
    """Thresholds that drive pass/fail decisions."""

    max_line_ratio: float = 0.2
    max_drift_index: float = 0.15
    max_masked_fraction: float = 0.1
    min_snr_proxy: float = 2.0
    max_stationarity_cv: float = 0.35


@dataclass(frozen=True)
class QCConfig:
    """Configuration for the QC metrics stage."""

    thresholds: QCThresholds = field(default_factory=QCThresholds)
    line_window_hz: float = 2.0
    snr_band: tuple[float, float] = (8.0, 30.0)
    broadband_band: tuple[float, float] = (1.0, 100.0)
    low_freq_band: float = 1.0
    stationarity_window_s: float = 1.0


@dataclass(frozen=True)
class PipelineConfig:
    """Aggregate configuration for the full cleaning pipeline."""

    standardize: StandardizeConfig = field(default_factory=StandardizeConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    line: LineRemovalConfig = field(default_factory=LineRemovalConfig)
    drift: DriftRemovalConfig = field(default_factory=DriftRemovalConfig)
    interpolate: InterpolateConfig = field(default_factory=InterpolateConfig)
    qc: QCConfig = field(default_factory=QCConfig)


PERFORMANCE_BUDGETS: Dict[str, float] = {
    "realtime_factor": 20.0,
    "chunk_latency_ms": 50.0,
    "memory_multiplier": 4.0,
}


def get_default_config() -> PipelineConfig:
    """Return a new copy of the default pipeline configuration."""

    return PipelineConfig()
