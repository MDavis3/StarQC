"""StarQC public API.

Deterministic artifact removal and QC for multi-site, low-power neural implants.
"""

from .clean import clean_lfp
from .version import __version__, get_version

# Optionally expose streaming for embedded/low-latency workflows
try:  # guarded import to avoid hard failures in constrained environments
    from .stream import StreamingCleaner  # type: ignore  # noqa: F401
    _EXPOSE_STREAMING = True
except Exception:
    _EXPOSE_STREAMING = False

__all__ = ["clean_lfp", "__version__", "get_version"]
if _EXPOSE_STREAMING:
    __all__.append("StreamingCleaner")
