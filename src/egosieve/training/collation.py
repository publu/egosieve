"""Compatibility import surface for sampled-target collation utilities."""

from . import targets as _targets
from .targets import *  # noqa: F403

__all__ = _targets.__all__
