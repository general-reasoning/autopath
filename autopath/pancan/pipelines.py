from typing import Optional
import tqdm
import torch
import torchvision

#TODO: REMOVE?
# import torch.multiprocessing as mp
# mp.set_start_method('spawn', force=True)

import dbx

from autopath.pancan.clips import PancanTileBag, PancanTileClip, PancanTilePartition, PancanTileFold, TileClipDatasetBuilder
from autopath.env import PANCAN_CPTAC_ROOT, PANCAN_CPTAC_SAMPLE, PANCAN_CPTAC_RESOLUTION

"""
git commit -am "gigaq: PancanTileBag: BUILD" >/dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').build().valid()"
git commit -am "gigaq: PancanTileBag: BUILD" >/dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').build().paths()"
git commit -am "gigaq: PancanTileBag: READ SHARDS" >/dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').read('shards')"
git commit -am "gigaq: PancanTileBag: DATASET" >/dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_bag('CPTAC_SAMPLE').dataset()[0]"
"""
def pancan_tile_bag(name=None) -> PancanTileBag:
    if name is None:
        return PancanTileBag
    elif name == "CPTAC_SAMPLE":     
        return PancanTileBag(spec=dict(source=PANCAN_CPTAC_SAMPLE), keyby='tag_version_hash')
    else:
        raise ValueError(f"Unknown tile_bag: {name}")

"""
git commit -am "gigaq: PancanTileClip: BUILD" > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_clip('CPTAC').build().valid()"
git commit -am "gigaq: PancanTileClip: BUILD" > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_clip('CPTAC', n_workers=8).build().valid()"
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
            keyby='tag_version_hash',
            n_workers=n_workers,
            parallelization=parallelization,
        )
    else:
        raise ValueError(f"Unknown tile clip: {name}")



"""
git commit -am 'gigaq: PancanTilePartition: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_partition('CPTAC_8020').build_tree()"
git commit -am 'gigaq: PancanTilePartition: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_partition('CPTAC_9802').build_tree()"
git commit -am 'gigaq: PancanTilePartition: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_partition('CPTAC_404020').build_tree()"
git commit -am 'gigaq: PancanTilePartition: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_partition('CPTAC_206020').build_tree()"
git commit -am 'gigaq: PancanTilePartition: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_partition('CPTAC_400159').build_tree()"
git commit -am 'gigaq: PancanTilePartition: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_partition('CPTAC_200179').build_tree()"
git commit -am 'gigaq: PancanTilePartition: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_partition('CPTAC_305020').build_tree()"
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
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.98, 0.02])
        )
    elif name == "CPTAC_404020":
        assert fold_fractions is None or fold_fractions == [0.4, 0.4, 0.2], "fold_fractions must be [0.4, 0.4, 0.2]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.4, 0.4, 0.2]))
    elif name == "CPTAC_305020":
        assert fold_fractions is None or fold_fractions == [0.3, 0.5, 0.2], "fold_fractions must be [0.3, 0.5, 0.2]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.3, 0.5, 0.2]))
    elif name == "CPTAC_206020":
        assert fold_fractions is None or fold_fractions == [0.2, 0.6, 0.2], "fold_fractions must be [0.2, 0.6, 0.2]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.2, 0.6, 0.2]))
    elif name == "CPTAC_400159":
        assert fold_fractions is None or fold_fractions == [0.4, 0.01, 0.59], "fold_fractions must be [0.4, 0.01, 0.59]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.4, 0.01, 0.59]))
    elif name == "CPTAC_200179":
        assert fold_fractions is None or fold_fractions == [0.2, 0.01, 0.79], "fold_fractions must be [0.2, 0.01, 0.79]"
        return PancanTilePartition(spec=dict(clip=dbx.quote(pancan_tile_clip, 'CPTAC'), fold_fractions=[0.2, 0.01, 0.79]))
    else:
        raise ValueError(f"Unknown tile_partition: {name}")


"""
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TRAIN').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_8020_TEST').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TRAIN').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_9802_TEST').build()"

git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_404020_CALIBRATE').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_404020_TRAIN').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_404020_TEST').build()"

git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_305020_CALIBRATE').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_305020_TRAIN').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_305020_TEST').build()"

git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_206020_CALIBRATE').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_206020_TRAIN').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_206020_TEST').build()"

git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_400159_CALIBRATE').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_400159_TRAIN').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_400159_TEST').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_400159_CALIBRATE').shard(0)"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_400159_CALIBRATE').shard_lens"

git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_200179_CALIBRATE').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_200179_TRAIN').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_200179_TEST').build()"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_200179_CALIBRATE').shard(0)"
git commit -am 'gigaq: PancanTileFold: BUILD' > /dev/null || true; dbx.print "autopath.pancan.pipelines.pancan_tile_fold('CPTAC_200179_CALIBRATE').shard_lens"
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


# git commit -am "gigaq: pancan_tile_dataset: CPTAC" > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset('CPTAC')[0]"
# git commit -am "gigaq: pancan_tile_dataset: fold" > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset('CPTAC_404020_TRAIN')[0]"
def pancan_tile_dataset(name=None, *, shuffle: bool = False, skip_invalid_bags: bool = False, batch_size: int = 1):
    """Return an MDS-backed :class:`StreamingDataset` over tile shards.

    Uses the MDS multi-stream interface — each bag's shards are
    combined into a single streaming dataset.

    Parameters
    ----------
    name : str
        A clip name (e.g. ``'CPTAC'``) or fold name
        (e.g. ``'CPTAC_404020_TRAIN'``).
    shuffle : bool
        Whether to shuffle within the streaming dataset.
    skip_invalid_bags : bool
        If ``True``, silently skip bags whose MDS shards have not
        been built yet.
    batch_size : int | None
        Per-device batch size passed to :class:`StreamingDataset`.
        Required when the dataset will be consumed via a
        :class:`DataLoader`.
    """
    def _resolve_tile_clip(name):
        """Resolve *name* to a :class:`PancanTileClip` or :class:`PancanTileFold`.

        Tries ``pancan_tile_fold(name)`` first; falls back to
        ``pancan_tile_clip(name)`` for bare clip names like ``'CPTAC'``.
        """
        try:
            return pancan_tile_fold(name)
        except ValueError:
            return pancan_tile_clip(name)
    clip = _resolve_tile_clip(name)
    return clip.dataset(shuffle=shuffle, skip_invalid_bags=skip_invalid_bags, batch_size=batch_size)


"""
git commit -am 'gigaq: pancan_tile_dataset_samples: CPTAC' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC', n=8, batch_size=1)"

git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_8020_TRAIN', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_8020_TEST', n=8, batch_size=4)"

git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_9802_TRAIN', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_9802_TEST', n=8, batch_size=4)"

git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_404020_TRAIN', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_404020_CALIBRATE', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_404020_TEST', n=8, batch_size=4)"

git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_305020_TRAIN', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_305020_CALIBRATE', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_305020_TEST', n=8, batch_size=4)"

git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_206020_TRAIN', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_206020_CALIBRATE', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_206020_TEST', n=8, batch_size=4)"

git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_400159_TRAIN', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_400159_CALIBRATE', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_400159_TEST', n=8, batch_size=4)"

git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_200179_TRAIN', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_200179_CALIBRATE', n=8, batch_size=4)"
git commit -am 'gigaq: pancan_tile_dataset_samples: SAMPLE' > /dev/null || true; dbx.pprint "autopath.pancan.pipelines.pancan_tile_dataset_samples('CPTAC_200179_TEST', n=8, batch_size=4)"
"""
def pancan_tile_dataset_samples(
    name=None,
    n: int = 5,
    *,
    batch_size: int = 4,
    shuffle: bool = False,
    skip_invalid_bags: bool = False,
    return_last: bool = True,
    **dataloader_kwargs,
):
    """Iterate *n* samples from the MDS-backed tile dataset.

    Useful for smoke-testing the pipeline and inspecting sample dicts.

    Parameters
    ----------
    name : str
        A clip name (e.g. ``'CPTAC'``) or fold name
        (e.g. ``'CPTAC_404020_TRAIN'``).
    n : int
        Number of samples to iterate.
    batch_size : int
        DataLoader batch size.
    shuffle, skip_invalid_bags
        Forwarded to :func:`pancan_tile_dataset`.
    return_last : bool
        If ``True``, return the last batch.
    **dataloader_kwargs
        Extra keyword arguments forwarded to
        :class:`torch.utils.data.DataLoader`.
    """
    ds = pancan_tile_dataset(name, shuffle=shuffle, skip_invalid_bags=skip_invalid_bags, batch_size=batch_size)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, **dataloader_kwargs)
    progress = tqdm.tqdm(total=n)
    last = None
    for i, batch in enumerate(loader):
        progress.update(batch_size)
        last = batch
        if (i + 1) * batch_size >= n:
            break
    if return_last:
        return last


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_8020_TRAIN')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_8020_TEST')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_9802_TRAIN')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_9802_TEST')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_404020_CALIBRATE')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_404020_TRAIN')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_404020_TEST')[0]"

git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_400159_CALIBRATE')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_400159_TRAIN')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_400159_TEST')[0]"

git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_200179_CALIBRATE')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_200179_TRAIN')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_200179_TEST')[0]"

git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_305020_CALIBRATE')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_305020_TRAIN')[0]"
git commit -am 'gigaq: pancan_tile_bag_dataset: SAMPLE' > /dev/null || true; dbx "autopath.pancan.pipelines.pancan_tilebag_dataset('CPTAC_305020_TEST')[0]"
"""
def pancan_tilebag_dataset_builder(
	*,
	clip: PancanTileClip,
	transform: Optional[torchvision.transforms.Compose] = None,
	debug: bool = False,
	verbose: bool = False,
	log = None,
):
		clip = dbx.eval(clip)
		transform = dbx.eval(transform)
		kwargs = dict(
			debug=debug, 
			verbose=verbose,
		)
		if log is not None:
			kwargs['log'] = log

		return TileClipDatasetBuilder(spec=dict(clip=clip, transform=transform,), **kwargs)


def pancan_tilebag_dataset(name=None) -> TileClipDatasetBuilder:
    if name == "CPTAC_8020_TEST":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TEST'),)
    elif name == "CPTAC_8020_TRAIN":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TRAIN'),)
    elif name == "CPTAC_9802_TEST":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_9802_TEST'),)
    elif name == "CPTAC_9802_TRAIN":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_9802_TRAIN'),)
    elif name == "CPTAC_404020_CALIBRATE":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_404020_CALIBRATE'),)
    elif name == "CPTAC_404020_TRAIN":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_404020_TRAIN'),)
    elif name == "CPTAC_404020_TEST":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_404020_TEST'),)
    elif name == "CPTAC_305020_CALIBRATE":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_305020_CALIBRATE'),)
    elif name == "CPTAC_305020_TRAIN":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_305020_TRAIN'),)
    elif name == "CPTAC_305020_TEST":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_305020_TEST'),)
    elif name == "CPTAC_400159_CALIBRATE":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_400159_CALIBRATE'),)
    elif name == "CPTAC_400159_TRAIN":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_400159_TRAIN'),)
    elif name == "CPTAC_400159_TEST":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_400159_TEST'),)
    elif name == "CPTAC_200179_CALIBRATE":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_200179_CALIBRATE'),)
    elif name == "CPTAC_200179_TRAIN":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_200179_TRAIN'),)
    elif name == "CPTAC_200179_TEST":
        builder = pancan_tilebag_dataset_builder(clip=dbx.quote(pancan_tile_fold, 'CPTAC_200179_TEST'),)   
    else:
        raise ValueError(f"Unknown pancan_tilebag_dataset: {name}")
    return builder.dataset()

