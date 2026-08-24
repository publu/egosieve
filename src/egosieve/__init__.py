"""EgoSieve: compact video-readiness modeling tools.

Model imports are lazy so metadata, sampling, and compiler utilities stay
usable in lightweight environments that do not load PyTorch at startup.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_MODEL_EXPORTS = {
    "BOUNDARY_LABELS",
    "EgoSieveConfig",
    "EgoSieveModel",
    "EgoSieveModelOutput",
    "EgoSievePreTrainedModel",
    "ISSUE_LABELS",
    "READINESS_LABELS",
}

__all__ = sorted(_MODEL_EXPORTS | {"EgoSieveProcessor"})


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        return getattr(import_module(".modeling", __name__), name)
    if name == "EgoSieveProcessor":
        return getattr(import_module(".processing_egosieve", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
