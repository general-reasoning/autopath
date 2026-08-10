from typing import Optional
import tqdm
import torch
import torchvision

import dbx

from autopath.pancan.tabclips import PancanTileBag, PancanTileClip, PancanTilePartition, PancanTileFold
from autopath.autobits import sanitize_collate
from autopath.env import PANCAN_CPTAC_ROOT, PANCAN_CPTAC_SAMPLE, PANCAN_CPTAC_RESOLUTION

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
        return PancanTileBag(spec=dict(source=PANCAN_CPTAC_SAMPLE))
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
def pancan_tile_partition(name=None, fold_fractions: Optional[list[float]] = None) -> PancanTilePartition:
    if name is None:
        return PancanTilePartition
    elif name == "CPTAC":
        assert fold_fractions is not None, "fold_fractions must be specified"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=fold_fractions))   
    elif name == "CPTAC_8020":
        assert fold_fractions is None or fold_fractions == [0.8, 0.2], "fold_fractions must be [0.8, 0.2]"   
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.8, 0.2]))
    elif name == "CPTAC_9802": 
        assert fold_fractions is None or fold_fractions == [0.98, 0.02], "fold_fractions must be [0.98, 0.02]"  
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.98, 0.02]))
    elif name == "CPTAC_404020":
        assert fold_fractions is None or fold_fractions == [0.4, 0.4, 0.2], "fold_fractions must be [0.4, 0.4, 0.2]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.4, 0.4, 0.2]))
    elif name == "CPTAC_305020":
        assert fold_fractions is None or fold_fractions == [0.3, 0.5, 0.2], "fold_fractions must be [0.3, 0.5, 0.2]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.3, 0.5, 0.2]))
    elif name == "CPTAC_206020":
        assert fold_fractions is None or fold_fractions == [0.2, 0.6, 0.2], "fold_fractions must be [0.2, 0.6, 0.2]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.2, 0.6, 0.2]))
    elif name == "CPTAC_602020":
        assert fold_fractions is None or fold_fractions == [0.6, 0.2, 0.2], "fold_fractions must be [0.6, 0.2, 0.2]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.6, 0.2, 0.2]))
    elif name == "CPTAC_400159":
        assert fold_fractions is None or fold_fractions == [0.4, 0.01, 0.59], "fold_fractions must be [0.4, 0.01, 0.59]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.4, 0.01, 0.59]))
    elif name == "CPTAC_200179":
        assert fold_fractions is None or fold_fractions == [0.2, 0.01, 0.79], "fold_fractions must be [0.2, 0.01, 0.79]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.2, 0.01, 0.79]))
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
    elif name == "CPTAC_8020_TEST":   
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_8020'), fold='-1'))
    elif name == "CPTAC_8020_TRAIN":   
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_8020'), fold='0'))
    elif name == "CPTAC_9802_TEST":   
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_9802'), fold='-1'))
    elif name == "CPTAC_9802_TRAIN":   
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_9802'), fold='0'))
    elif name == "CPTAC_404020_TRAIN":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_404020'), fold='0'))
    elif name == "CPTAC_404020_CALIBRATE":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_404020'), fold='1'))
    elif name == "CPTAC_404020_TEST":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_404020'), fold='-1'))
    elif name == "CPTAC_305020_TRAIN":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_305020'), fold='0'))
    elif name == "CPTAC_305020_CALIBRATE":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_305020'), fold='1'))
    elif name == "CPTAC_305020_TEST":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_305020'), fold='-1'))
    elif name == "CPTAC_206020_TRAIN":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_206020'), fold='0'))
    elif name == "CPTAC_206020_CALIBRATE":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_206020'), fold='1'))
    elif name == "CPTAC_206020_TEST":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_206020'), fold='-1'))
    elif name == "CPTAC_602020_TRAIN":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_602020'), fold='0'))
    elif name == "CPTAC_602020_CALIBRATE":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_602020'), fold='1'))
    elif name == "CPTAC_602020_TEST":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_602020'), fold='-1'))
    elif name == "CPTAC_400159_TRAIN":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_400159'), fold='0'))
    elif name == "CPTAC_400159_CALIBRATE":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_400159'), fold='1'))
    elif name == "CPTAC_400159_TEST":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_400159'), fold='-1'))
    elif name == "CPTAC_200179_TRAIN":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_200179'), fold='0'))
    elif name == "CPTAC_200179_CALIBRATE":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_200179'), fold='1'))
    elif name == "CPTAC_200179_TEST":
        return PancanTileFold(spec=dict(partition=dbx.quote(pancan_tile_partition, 'CPTAC_200179'), fold='-1'))
    else:
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
