"""
    Examples:
    ## GigaqStill: BASELINE_CPTAC_8020: train:
    $ dbx "autopath.gigapath.dinov2.pipelines.gigaq_still('BASELINE_CPTAC_8020_TRAIN').build()"
	# or 
	> import autopath.gigapath.dinov2.pipelines; autopath.gigapath.dinov2.pipelines.gigaq_still('BASELINE_CPTAC_8020_TRAIN',).build()



"""

from functools import partial
from typing import Any, TypeVar, Optional, Callable, List


import torch
from torchvision import transforms

from .still import ARCH, GigaqStill
from . import backbone

from .augmentations import DataAugmentationDINO, dino_tile_transform

from ...pancan.tiles import PancanTileSet
from ...pancan.pipelines import pancan_tile_fold


def dino_augmentations(name="DINO_DEFAULT"):
    if name == "DINO_DEFAULT":
        return DataAugmentationDINO(
            global_crops_scale=[0.32, 1.0],
            local_crops_scale=[0.05, 0.32],
            local_crops_number=8,
            global_crops_size=224,
            local_crops_size=96,
        )
    else:
        raise ValueError(f"Unknown augmentation: {name}")


def dino_pancan_tile_dataset(name):
    if name == "CPTAC_8020_TRAIN":
        tileset = PancanTileSet(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_fold(name='CPTAC_8020_TRAIN')", 
                                       transform=f"@autopath.gigapath.dinov2.pipelines.dino_augmentations('DINO_DEFAULT')"))
    elif name == "CPTAC_8020_TEST":
        tileset = PancanTileSet(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_fold(name='CPTAC_8020_TEST')", 
                                       transform=f"@autopath.gigapath.dinov2.pipelines.dino_augmentations('DINO_DEFAULT')"))
    elif name == "CPTAC_9802_TRAIN":
        tileset = PancanTileSet(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_fold(name='CPTAC_9802_TRAIN')", 
                                       transform=f"@autopath.gigapath.dinov2.pipelines.dino_augmentations('DINO_DEFAULT')"))
    elif name == "CPTAC_9802_TEST":
        tileset =PancanTileSet(spec=dict(tileshards=f"@autopath.pancan.pipelines.pancan_tile_fold(name='CPTAC_9802_TEST')", 
                                       transform=f"@autopath.gigapath.dinov2.pipelines.dino_augmentations('DINO_DEFAULT')"))
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return tileset.dataset


def gigapath_tile_backbone(name):
    if name == "GIGAPATH_BASELINE_BACKBONE":
        return backbone.gigapath_tile_backbone()
    else:
        raise ValueError(f"Unknown backbone: {name}")


def gigapath_tile_backbone_evaluator(name):
    if name == "GIGAPATH_BASELINE_BACKBONE":
        return backbone.BackboneEvaluator(gigapath_tile_backbone(name), transform=dino_augmentations())
    else:
        raise ValueError(f"Unknown backbone evaluator: {name}")


def gigaq_arch(name):
    if name == 'BASELINE':
        return ARCH()
    else:
        raise ValueError(f"Unknown gigaq arch: {name}")


def gigaq_still(name):
    if name == 'BASELINE_CPTAC_9802_TRAIN':
        return GigaqStill(spec=dict(dataset="@autopath.gigapath.dinov2.pipelines.dino_pancan_tile_dataset('CPTAC_9802_TRAIN')",
                               arch="@autopath.gigapath.dinov2.pipelines.gigaq_arch('BASELINE')",))
    if name == 'BASELINE_CPTAC_8020_TRAIN':
        return GigaqStill(spec=dict(dataset="@autopath.gigapath.dinov2.pipelines.dino_pancan_tile_dataset('CPTAC_8020_TRAIN')",
                               arch="@autopath.gigapath.dinov2.pipelines.gigaq_arch('BASELINE')",))
    
    else:
        raise ValueError(f"Unknown gigaq: {name}")



