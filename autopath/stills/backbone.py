"""Backward-compatibility shim — all definitions live in bitpath.py.

Serialised ``dbx.quote`` specs embed the fully-qualified class name
(e.g. ``autopath.stills.backbone.BitConvDeepBackboneEvaluatorFactory``).
This module must continue to exist and re-export those names so that
old specs remain resolvable and their hashes are unchanged.

Do **not** define new classes here; add them to
:mod:`autopath.stills.bitpath` instead.
"""

from autopath.stills.bitpath import (   # noqa: F401
    bitconv_tile_transform,
    bitpath_tile_transform,
    BitConvDeepBackboneEvaluator,
    BitConvDeepBackboneEvaluatorFactory,
    BitPathConvDeepBackboneEvaluator,
    BitPathConvDeepBackboneEvaluatorFactory,
)

__all__ = [
    'bitconv_tile_transform',
    'bitpath_tile_transform',
    'BitConvDeepBackboneEvaluator',
    'BitConvDeepBackboneEvaluatorFactory',
    'BitPathConvDeepBackboneEvaluator',
    'BitPathConvDeepBackboneEvaluatorFactory',
]
