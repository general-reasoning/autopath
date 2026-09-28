from typing import Optional
import tqdm
import torch
import torchvision

import dbx
from dbx.datablocks import Datablock, DIR, DIRTOPIC, DATAFILE
from dbx.datapoints import SLICETOPIC, DatapointTable, DatapointFold

from autopath.pancan.tabclips import PancanTileBag, PancanTileClip, PancanTilePartition, PancanTileFold
from autopath.autobits import sanitize_collate
from autopath.env import PANCAN_CPTAC_ROOT, PANCAN_CPTAC_SAMPLE, PANCAN_CPTAC_RESOLUTION

PANCAN_TILE_BAG_SPECIALIZATIONS = [
    Datablock.Specialization(
        spec={},
        topics={
            'tiles': SLICETOPIC,
            'annotations': SLICETOPIC,
            'bag_name': SLICETOPIC,
            'tile_index': SLICETOPIC,
        },
        version=1,
        note='Sentinel-era topic declaration',
    ),
    Datablock.Specialization(
        spec={},
        topics={
            'tiles': SLICETOPIC,
            'annotations': SLICETOPIC,
            'bag_name': SLICETOPIC,
            'tile_index': SLICETOPIC,
        },
        redirect_topics=['tiles', 'bag_name', 'tile_index'],
        version=1,
        note='Sentinel-era topic declaration (tiles only, owe annotations)',
    ),
]

PANCAN_TILE_CLIP_SPECIALIZATIONS = [
    DatapointTable.Specialization(
        spec={},
        topics={'tabs': DIR, 'done': DATAFILE('done'), 'bag_lens': DATAFILE('bag_lens.npz')},
        TAB=None,
        note='Pre-BLOCK tabular build',
    ),
    DatapointTable.Specialization(
        spec={},
        topics={'tabs': DIRTOPIC, 'done': 'done', 'bag_lens': 'bag_lens.npz'},
        version=1,
        TAB=None,
        note='Sentinel-era clip declaration',
    ),
]

PANCAN_TILE_FOLD_SPECIALIZATIONS = [
    DatapointFold.Specialization(
        spec={},
        topics={'tabs': DIR, 'done': DATAFILE('done'), 'bag_lens': DATAFILE('bag_lens.npz')},
        TAB=None,
        note='Pre-BLOCK tabular fold build',
    ),
]

"""
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_bag('CPTAC_SAMPLE').build().valid()"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_bag('CPTAC_SAMPLE').build().paths()"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_bag('CPTAC_SAMPLE').data('tiles', concat=True)['tile'].shape"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_bag('CPTAC_SAMPLE').data('annotations', concat=True)['annotations']['case_id']"
"""
def pancan_tile_bag(name=None) -> PancanTileBag:
    if name is None:
        return PancanTileBag
    elif name == "CPTAC_SAMPLE":     
        return PancanTileBag(
            spec=dict(source=PANCAN_CPTAC_SAMPLE),
            SPECIALIZATIONS=PANCAN_TILE_BAG_SPECIALIZATIONS,
        )
    else:
        raise ValueError(f"Unknown tile_bag: {name}")

"""
dbx.print "autopath.pancan.tabpipelines.pancan_tile_clip('CPTAC').build().valid()"

dbx.print "autopath.pancan.tabpipelines.pancan_tile_clip('CPTAC', n_workers=32, parallelization='multithreading').build().valid()"
"""
def pancan_tile_clip(name=None, *, n_workers: int = 1, parallelization: str | None = None) -> PancanTileClip:
    if parallelization is None and n_workers > 1:
        parallelization = "multiprocessing"
    if name is None:
        return PancanTileClip
    elif name == "CPTAC":     
        return PancanTileClip(
            spec=dict(
                source=PANCAN_CPTAC_ROOT,
                resolution=PANCAN_CPTAC_RESOLUTION,
            ),
            n_workers=n_workers,
            parallelization=parallelization,
            SPECIALIZATIONS=PANCAN_TILE_CLIP_SPECIALIZATIONS,
            TAB_SPECIALIZATIONS=PANCAN_TILE_BAG_SPECIALIZATIONS,
        )
    else:
        raise ValueError(f"Unknown tile clip: {name}")


"""
dbx.print "autopath.pancan.tabpipelines.pancan_tile_partition('CPTAC_8020').build_tree()"
#
dbx.print "autopath.pancan.tabpipelines.pancan_tile_partition('CPTAC_9802').build_tree()"
#
dbx.print "autopath.pancan.tabpipelines.pancan_tile_partition('CPTAC_404020').build_tree()"
#
dbx.print "autopath.pancan.tabpipelines.pancan_tile_partition('CPTAC_206020').build_tree()"
#
dbx.print "autopath.pancan.tabpipelines.pancan_tile_partition('CPTAC_602020').build_tree()"
#
dbx.print "autopath.pancan.tabpipelines.pancan_tile_partition('CPTAC_400159').build_tree()"
#
dbx.print "autopath.pancan.tabpipelines.pancan_tile_partition('CPTAC_200179').build_tree()"
#
dbx.print "autopath.pancan.tabpipelines.pancan_tile_partition('CPTAC_305020').build_tree()"
"""
def pancan_tile_partition(name=None, fractions: Optional[list[float]] = None, fold_fractions: Optional[list[float]] = None) -> PancanTilePartition:
    if fractions is None and fold_fractions is not None:
        fractions = fold_fractions
    if name is None:
        return PancanTilePartition
    elif name == "CPTAC":
        assert fractions is not None, "fractions must be specified"
        return PancanTilePartition(spec=dict(datapoint_table=dbx.quote(pancan_tile_clip, 'CPTAC'), fractions=fractions, partition_slice='tiles'))   
    elif name == "CPTAC_8020":
        assert fractions is None or fractions == [0.8, 0.2], "fractions must be [0.8, 0.2]"   
        return PancanTilePartition(spec=dict(datapoint_table=dbx.quote(pancan_tile_clip, 'CPTAC'), fractions=[0.8, 0.2], partition_slice='tiles'))
    elif name == "CPTAC_9802": 
        assert fractions is None or fractions == [0.98, 0.02], "fractions must be [0.98, 0.02]"  
        return PancanTilePartition(spec=dict(datapoint_table=dbx.quote(pancan_tile_clip, 'CPTAC'), fractions=[0.98, 0.02], partition_slice='tiles'))
    elif name == "CPTAC_404020":
        assert fractions is None or fractions == [0.4, 0.4, 0.2], "fractions must be [0.4, 0.4, 0.2]"
        return PancanTilePartition(spec=dict(datapoint_table=dbx.quote(pancan_tile_clip, 'CPTAC'), fractions=[0.4, 0.4, 0.2], partition_slice='tiles'))
    elif name == "CPTAC_305020":
        assert fractions is None or fractions == [0.3, 0.5, 0.2], "fractions must be [0.3, 0.5, 0.2]"
        return PancanTilePartition(spec=dict(datapoint_table=dbx.quote(pancan_tile_clip, 'CPTAC'), fractions=[0.3, 0.5, 0.2], partition_slice='tiles'))
    elif name == "CPTAC_206020":
        assert fractions is None or fractions == [0.2, 0.6, 0.2], "fractions must be [0.2, 0.6, 0.2]"
        return PancanTilePartition(spec=dict(datapoint_table=dbx.quote(pancan_tile_clip, 'CPTAC'), fractions=[0.2, 0.6, 0.2], partition_slice='tiles'))
    elif name == "CPTAC_602020":
        assert fractions is None or fractions == [0.6, 0.2, 0.2], "fractions must be [0.6, 0.2, 0.2]"
        return PancanTilePartition(spec=dict(datapoint_table=dbx.quote(pancan_tile_clip, 'CPTAC'), fractions=[0.6, 0.2, 0.2], partition_slice='tiles'))
    elif name == "CPTAC_400159":
        assert fractions is None or fractions == [0.4, 0.01, 0.59], "fractions must be [0.4, 0.01, 0.59]"
        return PancanTilePartition(spec=dict(datapoint_table=dbx.quote(pancan_tile_clip, 'CPTAC'), fractions=[0.4, 0.01, 0.59], partition_slice='tiles'))
    elif name == "CPTAC_200179":
        assert fractions is None or fractions == [0.2, 0.01, 0.79], "fractions must be [0.2, 0.01, 0.79]"
        return PancanTilePartition(spec=dict(datapoint_table=dbx.quote(pancan_tile_clip, 'CPTAC'), fractions=[0.2, 0.01, 0.79], partition_slice='tiles'))
    else:
        raise ValueError(f"Unknown tile_partition: {name}")


"""
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_8020_TRAIN').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_8020_TEST').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_9802_TRAIN').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_9802_TEST').build().valid()"

dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_404020_CALIBRATE').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_404020_TRAIN').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_404020_TEST').build().valid()"

dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_305020_CALIBRATE').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_305020_TRAIN').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_305020_TEST').build().valid()"

dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_206020_CALIBRATE').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_206020_TRAIN').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_206020_TEST').build().valid()"

dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_602020_CALIBRATE').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_602020_TRAIN').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_602020_TEST').build().valid()"

dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_400159_CALIBRATE').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_400159_TRAIN').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_400159_TEST').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_400159_CALIBRATE').block(0)"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_400159_CALIBRATE').block_lens"

dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_200179_CALIBRATE').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_200179_TRAIN').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_200179_TEST').build().valid()"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_200179_CALIBRATE').block(0)"
dbx.print "autopath.pancan.tabpipelines.pancan_tile_fold('CPTAC_200179_CALIBRATE').block_lens"
"""
def pancan_tile_fold(name=None) -> PancanTileFold:
    if name is None:
        return PancanTileFold
    fold_map = {
        "CPTAC_8020_TEST": ('CPTAC_8020', -1),
        "CPTAC_8020_TRAIN": ('CPTAC_8020', 0),
        "CPTAC_9802_TEST": ('CPTAC_9802', -1),
        "CPTAC_9802_TRAIN": ('CPTAC_9802', 0),
        "CPTAC_404020_TRAIN": ('CPTAC_404020', 0),
        "CPTAC_404020_CALIBRATE": ('CPTAC_404020', 1),
        "CPTAC_404020_TEST": ('CPTAC_404020', -1),
        "CPTAC_305020_TRAIN": ('CPTAC_305020', 0),
        "CPTAC_305020_CALIBRATE": ('CPTAC_305020', 1),
        "CPTAC_305020_TEST": ('CPTAC_305020', -1),
        "CPTAC_206020_TRAIN": ('CPTAC_206020', 0),
        "CPTAC_206020_CALIBRATE": ('CPTAC_206020', 1),
        "CPTAC_206020_TEST": ('CPTAC_206020', -1),
        "CPTAC_602020_TRAIN": ('CPTAC_602020', 0),
        "CPTAC_602020_CALIBRATE": ('CPTAC_602020', 1),
        "CPTAC_602020_TEST": ('CPTAC_602020', -1),
        "CPTAC_400159_TRAIN": ('CPTAC_400159', 0),
        "CPTAC_400159_CALIBRATE": ('CPTAC_400159', 1),
        "CPTAC_400159_TEST": ('CPTAC_400159', -1),
        "CPTAC_200179_TRAIN": ('CPTAC_200179', 0),
        "CPTAC_200179_CALIBRATE": ('CPTAC_200179', 1),
        "CPTAC_200179_TEST": ('CPTAC_200179', -1),
    }
    if name in fold_map:
        part, fold = fold_map[name]
        return PancanTileFold(
            spec=dict(partition=dbx.quote(pancan_tile_partition, part), fold=fold),
            SPECIALIZATIONS=PANCAN_TILE_FOLD_SPECIALIZATIONS,
            TAB_SPECIALIZATIONS=PANCAN_TILE_BAG_SPECIALIZATIONS,
        )
    raise ValueError(f"Unknown tile_fold: {name}")


"""
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC', n=8, batch_size=1, mode='iter')"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_8020_TRAIN', 'tiles', 'annotations', n=8, batch_size=4, mode='iter')"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_8020_TEST', n=8, batch_size=4, mode='map')"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_9802_TRAIN', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_9802_TEST', n=8, batch_size=4)"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_404020_TRAIN', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_404020_CALIBRATE', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_404020_TEST', n=8, batch_size=4)"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_305020_TRAIN', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_305020_CALIBRATE', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_305020_TEST', n=8, batch_size=4)"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_206020_TRAIN', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_206020_CALIBRATE', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_206020_TEST', n=8, batch_size=4)"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_602020_TRAIN', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_602020_CALIBRATE', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_602020_TEST', n=8, batch_size=4)"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_400159_TRAIN', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_400159_CALIBRATE', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_400159_TEST', n=8, batch_size=4)"

dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_200179_TRAIN', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_200179_CALIBRATE', n=8, batch_size=4)"
dbx.pprint "autopath.pancan.tabpipelines.pancan_tile_dataset_samples('CPTAC_200179_TEST', n=8, batch_size=4)"
"""
def pancan_tile_dataset_samples(
    name=None,
    *slices,
    n: int = 5,
    batch_size: int = 4,
    mode: str = 'map',
    return_last: bool = True,
    **dataset_kwargs,
):
    """Iterate *n* samples from the MDS-backed tile dataset.

    Useful for smoke-testing the pipeline and inspecting sample dicts.

    Parameters
    ----------
    name : str
        A clip name (e.g. ``'CPTAC'``) or fold name
        (e.g. ``'CPTAC_404020_TRAIN'``).
    *slices : tuple[str, ...]
        Names of slices to include.
    n : int
        Number of samples to iterate.
    batch_size : int
        DataLoader batch size.
    mode : str
        'map' or 'iter' for the returned dataset.
    return_last : bool
        If ``True``, return the last batch.
    **dataset_kwargs
        Extra keyword arguments forwarded to the ``dataset()`` call
        (e.g., shuffle, etc.).
    """
    def _resolve_tile_clip(name):
        try:
            return pancan_tile_fold(name)
        except ValueError:
            return pancan_tile_clip(name)
            
    clip = _resolve_tile_clip(name)
    ds = clip.dataset(*slices, mode=mode, batch_size=batch_size, **dataset_kwargs)
    
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=sanitize_collate)
    progress = tqdm.tqdm(total=n)
    last = None
    for i, batch in enumerate(loader):
        progress.update(batch_size)
        last = batch
        if (i + 1) * batch_size >= n:
            break
    if return_last:
        return last
