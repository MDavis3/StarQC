"""Provenance utilities for StarQC runs."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .version import __version__


def _safe_git_commit() -> str:
    """Return the current git commit hash if available."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "unknown"

    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def build_provenance(
    params: Mapping[str, Any],
    fs: float,
    shape: tuple[int, int],
    runtime_ms: float,
) -> Dict[str, Any]:
    """Construct a provenance record for a cleaning run."""

    return {
        "package_version": __version__,
        "git_commit": _safe_git_commit(),
        "params": dict(params),
        "fs": float(fs),
        "shape": tuple(int(v) for v in shape),
        "runtime_ms": float(runtime_ms),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def dump_report(report_path: str | Path, report: Mapping[str, Any]) -> None:
    """Persist the report dictionary as JSON."""

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
