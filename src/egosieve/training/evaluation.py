"""Compatibility import surface for NumPy evaluation metrics."""

from . import metrics as _metrics
from .metrics import *  # noqa: F403

__all__ = _metrics.__all__
