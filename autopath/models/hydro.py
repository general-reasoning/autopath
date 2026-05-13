"""Backward-compatibility re-export shim.

All model classes have been moved to :mod:`autopath.stills`.
This module re-exports everything so that existing ``from
autopath.models.hydro import …`` statements continue to work.
"""
# ruff: noqa: F401, F403
from autopath.stills.hydro import *
