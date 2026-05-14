from typing import Optional

#TODO: REMOVE?
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import dbx

from autopath.pancan.clips import PancanTileBag, PancanTileClip, PancanTilePartition, PancanTileFold, TileClipDatasetBuilder, pancan_tile_clip_dataset_builder
from autopath.env import PANCAN_CPTAC_ROOT, PANCAN_CPTAC_SAMPLE, PANCAN_CPTAC_RESOLUTION


# git commit -am "gigaq: PancanTileBag: READ" >/dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').valid()"
# git commit -am "gigaq: PancanTileBag: READ" >/dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').read('tiles')"
# git commit -am "gigaq: PancanTileBag: READ" >/dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').read('labels')"
def pancan_tile_bag(name=None) -> PancanTileBag:
    if name is None:
        return PancanTileBag
    elif name == "CPTAC_SAMPLE":     
        return PancanTileBag(spec=dict(source=PANCAN_CPTAC_SAMPLE))
    else:
        raise ValueError(f"Unknown tile_bag: {name}")

# git commit -am "gigaq: PancanTileClip: BUILD" > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_bag_clip('CPTAC').build()"
def pancan_tile_bag_clip(name=None) -> PancanTileClip:
    if name is None:
        return PancanTileClip
    elif name == "CPTAC":     
        return PancanTileClip(spec=dict(
                                source=PANCAN_CPTAC_ROOT,
                                resolution=PANCAN_CPTAC_RESOLUTION,
        ))
    else:
        raise ValueError(f"Unknown tile clip: {name}")



# git commit -am 'gigaq: PancanTilePartition: BUILD'; dbx.print 'autopath.pancan.pipelines.pancan_tile_partition("CPTAC_8020").build_tree()'
# git commit -am 'gigaq: PancanTilePartition: BUILD'; dbx.print 'autopath.pancan.pipelines.pancan_tile_partition("CPTAC_9802").build_tree()'
# git commit -am 'gigaq: PancanTilePartition: BUILD'; dbx.print 'autopath.pancan.pipelines.pancan_tile_partition("CPTAC_404020").build_tree()'
# git commit -am 'gigaq: PancanTilePartition: BUILD'; dbx.print 'autopath.pancan.pipelines.pancan_tile_partition("CPTAC_206020").build_tree()'
# git commit -am 'gigaq: PancanTilePartition: BUILD'; dbx.print 'autopath.pancan.pipelines.pancan_tile_partition("CPTAC_400159").build_tree()'
# git commit -am 'gigaq: PancanTilePartition: BUILD'; dbx.print 'autopath.pancan.pipelines.pancan_tile_partition("CPTAC_200179").build_tree()'
def pancan_tile_partition(name=None, fold_fractions: Optional[list[float]] = None) -> PancanTilePartition:
    if name is None:
        return PancanTilePartition
    elif name == "CPTAC":
        assert fold_fractions is not None, "fold_fractions must be specified"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_bag_clip, 'CPTAC'), fold_fractions=fold_fractions))   
    elif name == "CPTAC_8020":
        assert fold_fractions is None or fold_fractions == [0.8, 0.2], "fold_fractions must be [0.8, 0.2]"   
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_bag_clip, 'CPTAC'), fold_fractions=[0.8, 0.2]))
    elif name == "CPTAC_9802": 
        assert fold_fractions is None or fold_fractions == [0.98, 0.02], "fold_fractions must be [0.98, 0.02]"  
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_bag_clip, 'CPTAC'), fold_fractions=[0.98, 0.02])
        )
    elif name == "CPTAC_404020":
        assert fold_fractions is None or fold_fractions == [0.4, 0.4, 0.2], "fold_fractions must be [0.4, 0.4, 0.2]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_bag_clip, 'CPTAC'), fold_fractions=[0.4, 0.4, 0.2]))
    elif name == "CPTAC_206020":
        assert fold_fractions is None or fold_fractions == [0.2, 0.6, 0.2], "fold_fractions must be [0.2, 0.6, 0.2]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_bag_clip, 'CPTAC'), fold_fractions=[0.2, 0.6, 0.2]))
    elif name == "CPTAC_400159":
        assert fold_fractions is None or fold_fractions == [0.4, 0.01, 0.59], "fold_fractions must be [0.4, 0.01, 0.59]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_bag_clip, 'CPTAC'), fold_fractions=[0.4, 0.01, 0.59]))
    elif name == "CPTAC_200179":
        assert fold_fractions is None or fold_fractions == [0.2, 0.01, 0.79], "fold_fractions must be [0.2, 0.01, 0.79]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_bag_clip, 'CPTAC'), fold_fractions=[0.2, 0.01, 0.79]))
    else:
        raise ValueError(f"Unknown tile_partition: {name}")

# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TRAIN').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TEST').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TRAIN').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST').build()"
#
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_404020_CALIBRATE').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_404020_TRAIN').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_404020_TEST').build()"
#
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_305020_CALIBRATE').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_305020_TRAIN').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_305020_TEST').build()"
#
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_206020_CALIBRATE').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_206020_TRAIN').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_206020_TEST').build()"
#
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_400159_CALIBRATE').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_400159_TRAIN').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_400159_TEST').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_400159_CALIBRATE').shard(0)"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_400159_CALIBRATE').shard_lens"
#
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_200179_CALIBRATE').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_200179_TRAIN').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_200179_TEST').build()"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_200179_CALIBRATE').shard(0)"
# git commit -am 'gigaq: PancanTileFold: BUILD'; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_200179_CALIBRATE').shard_lens"

def pancan_tile_fold(name=None) -> PancanTileFold:
    if name is None:
        return PancanTileFold
    elif name == "CPTAC_8020_TEST":   
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_8020'), fold='test'))
    elif name == "CPTAC_8020_TRAIN":   
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_8020'), fold='train'))
    elif name == "CPTAC_9802_TEST":   
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_9802'), fold='test'))
    elif name == "CPTAC_9802_TRAIN":   
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_9802'), fold='train'))
    elif name == "CPTAC_404020_CALIBRATE":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_404020'), fold="0"))
    elif name == "CPTAC_404020_TRAIN":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_404020'), fold="1"))
    elif name == "CPTAC_404020_TEST":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_404020'), fold="2"))
    elif name == "CPTAC_206020_CALIBRATE":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_206020'), fold="0"))
    elif name == "CPTAC_206020_TRAIN":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_206020'), fold="1"))
    elif name == "CPTAC_206020_TEST":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_206020'), fold="2"))
    elif name == "CPTAC_400159_CALIBRATE":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_400159'), fold="0"))
    elif name == "CPTAC_400159_TRAIN":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_400159'), fold="1"))
    elif name == "CPTAC_400159_TEST":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_400159'), fold="2"))
    elif name == "CPTAC_200179_CALIBRATE":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_200179'), fold="0"))
    elif name == "CPTAC_200179_TRAIN":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_200179'), fold="1"))
    elif name == "CPTAC_200179_TEST":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_200179'), fold="2"))
    else:
        raise ValueError(f"Unknown tile_fold: {name}")

# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_8020_TRAIN')[0]"
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_8020_TEST')[0]"
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_9802_TRAIN')[0]"
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_9802_TEST')[0]"
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_404020_CALIBRATE')[0]"
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_404020_TRAIN')[0]"
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_404020_TEST')[0]"
#
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_400159_CALIBRATE')[0]"
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_400159_TRAIN')[0]"
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_400159_TEST')[0]"
#
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_200179_CALIBRATE')[0]"
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_200179_TRAIN')[0]"
# git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE'; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_200179_TEST')[0]"

def pancan_tilebag_dataset(name=None) -> TileClipDatasetBuilder:
    if name == "CPTAC_8020_TEST":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TEST'),)
    elif name == "CPTAC_8020_TRAIN":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TRAIN'),)
    elif name == "CPTAC_9802_TEST":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_9802_TEST'),)
    elif name == "CPTAC_9802_TRAIN":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_9802_TRAIN'),)
    elif name == "CPTAC_404020_CALIBRATE":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_404020_CALIBRATE'),)
    elif name == "CPTAC_404020_TRAIN":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_404020_TRAIN'),)
    elif name == "CPTAC_404020_TEST":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_404020_TEST'),)
    elif name == "CPTAC_400159_CALIBRATE":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_400159_CALIBRATE'),)
    elif name == "CPTAC_400159_TRAIN":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_400159_TRAIN'),)
    elif name == "CPTAC_400159_TEST":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_400159_TEST'),)
    elif name == "CPTAC_200179_CALIBRATE":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_200179_CALIBRATE'),)
    elif name == "CPTAC_200179_TRAIN":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_200179_TRAIN'),)
    elif name == "CPTAC_200179_TEST":
        builder = pancan_tile_clip_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_200179_TEST'),)   
    else:
        raise ValueError(f"Unknown pancan_tilebag_dataset: {name}")
    return builder.dataset()

