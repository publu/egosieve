"""Compatibility import surface for the training-data schema utilities."""

from . import data as _data
from .data import *  # noqa: F403

__all__ = _data.__all__
