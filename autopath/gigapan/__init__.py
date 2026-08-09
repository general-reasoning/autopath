"""Deep feature hierarchy — multi-layer activation capture and analysis.

This package provides the deep feature infrastructure for configurable
multi-layer activation extraction from the GigaPath ViT backbone.

Modules
-------
tabfeatures
    `DeepFeatureBag`, `DeepFeatureClip`, and their bipolar encoding variants.
tabprobes
    `DeepFeatureAffineLogisticProbe`, `DeepFeatureStatsProbe`, and their bipolar variants.
features
    `DeepFeatureBag`, `DeepFeatureClip`, and their spherical/corner encoding variants.
probes
    `DeepFeatureAffineLogisticProbe`, `DeepFeatureStatsProbe`, `DeepFeatureSpectralProber`,
    `DeepBackboneSpectralEvaluator`, and `DeepFeatureSpectralProbe`.
pipelines
    Pipeline entrypoints for GigaPath deep feature extraction.
"""

from . import tabfeatures, tabprobes

