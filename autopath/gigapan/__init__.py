"""Deep feature hierarchy — multi-layer activation capture and analysis.

This package provides the deep feature infrastructure for configurable
multi-layer activation extraction from the GigaPath ViT backbone.

Modules
-------
features
    `DeepFeatureBag`, `DeepFeatureClip`, and their bipolar encoding variants.
probes
    `DeepFeatureAffineLogisticProbe`, `DeepFeatureStatsProbe`, and their bipolar variants.
pipelines
    Pipeline entrypoints for GigaPath deep feature extraction.
"""

from . import features, probes

