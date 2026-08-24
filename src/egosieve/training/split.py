"""Compatibility import surface for grouped split utilities."""

from . import splits as _splits
from .splits import *  # noqa: F403

__all__ = _splits.__all__
