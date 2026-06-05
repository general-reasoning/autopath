"""Deep feature hierarchy — multi-layer activation capture and analysis.

This package provides the deep feature infrastructure for configurable
multi-layer activation extraction from the GigaPath ViT backbone.

Modules
-------
features
    :class:`DeepFeatureBag`, :class:`DeepFeatureClip`, and their
    spherical/corner encoding variants.
probes
    :class:`DeepFeatureAffineLogisticProbe`,
    :class:`DeepFeatureStatsProbe`, :class:`DeepFeatureSpectralProber`,
    :class:`DeepBackboneSpectralEvaluator`, and
    :class:`DeepFeatureSpectralProbe`.
pipelines
    Pipeline entrypoints for GigaPath deep feature extraction:
    :func:`gigapath_deep_backbone_evaluator`,
    :func:`gigapath_deep_feature_bag`,
    :func:`gigapath_deep_feature_clip`,
    :func:`gigapath_deep_feature_clip_dataloader_samples`.
"""
