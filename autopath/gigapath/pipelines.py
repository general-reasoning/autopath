import os
from typing import Optional, List, Literal

from tqdm import tqdm

import torch
import torch.multiprocessing as mp

import dbx
from dbx import tagged

from autopath.pancan.clips import TileClipDatasetBuilder, TileClipDataLoaderBuilder

from autopath.pancan.pipelines import (
    pancan_tile_bag,
    pancan_tile_clip,
    pancan_tile_fold,
)

from autopath.features import (
    FeatureBag, 
    FeatureBagClip,
    featurebag_dataset,
    FeaturesToFloat,
    FeaturesLabelTileToFloat,
    BipolarFeatureBagClip,
    BipolarSingleFeatureBagClip,
    SphericalFeatureBagClip,
    SpectralFeatureBag,
    SpectralFeatureBagClip,
)

from autopath.probes import (
    LogisticFeatureBagProbe, 
    AffineLogisticFeatureBagProbe,
    FeatureBagMedianProbe,
    BipolarFeatureBagProbe,
    #
    FeaturePairwiseDistances,
    FeatureSortedDistances,
    Feature2NNDistances,
    Feature2NNDim,
)





from autopath.stills.hydro import (
    Hydro,
    HydroLightning,
    HydroStill,
)

from autopath.gigapath.dinov2.backbone import (
    BackboneEvaluator,
    SidebandBackboneEvaluator,
    SpectralBackboneEvaluator,
    GIGAPATH_BACKBONE_DEPTH,
)


mp.set_start_method("spawn", force=True)

def quote_extractor(name, sideband: bool = False, capture_blocks: Optional[List[int]] = None):
        if sideband: 
            return dbx.quote(gigapath_backbone_evaluator, name, sideband=True, capture_blocks=capture_blocks)
        else:
            return dbx.quote(gigapath_backbone_evaluator, name)


def gigapath_backbone_evaluator(name, *, device: str = 'cuda',):
    def select_capture_blocks(n_blocks: int = 1):
        if n_blocks <= 0 or n_blocks > GIGAPATH_BACKBONE_DEPTH:
            return None
        inc = GIGAPATH_BACKBONE_DEPTH // (n_blocks - 1)
        return list(range(0, GIGAPATH_BACKBONE_DEPTH, inc))
    
    if name == "GIGAPATH_BASELINE_BACKBONE_EVALUATOR":
        return BackboneEvaluator(device=device)
    elif name == "GIGAPATH_BASELINE_BACKBONE_5BLOCK_EVALUATOR":
        capture_blocks = select_capture_blocks(n_blocks=5)
        return SidebandBackboneEvaluator(device=device, capture_blocks=capture_blocks)
    elif name.startswith("GIGAPATH_SPECTRAL_BACKBONE_EVALUATOR_"):
        import re
        suffix = name[len("GIGAPATH_SPECTRAL_BACKBONE_EVALUATOR_"):]
        m = re.fullmatch(r'BLOCKS:(\d+)(CAPPED)?_SPEC:(CLS|FULL|BOTH)(\d*)', suffix)
        if m is None:
            raise ValueError(
                f"Cannot parse spectral evaluator descriptor {suffix!r} from {name!r}. "
                f"Expected e.g. _BLOCKS:5_SPEC:CLS, _BLOCKS:5CAPPED_SPEC:FULL10, _BLOCKS:3_SPEC:BOTH20"
            )
        n_blocks = int(m.group(1))
        capped = m.group(2) is not None
        spectral_mode = m.group(3).lower()       # 'cls', 'full', or 'both'
        spectral_k = int(m.group(4)) if m.group(4) else 10   # default k=10
        probe_blocks = select_capture_blocks(n_blocks=n_blocks)
        if capped and probe_blocks and probe_blocks[-1] != GIGAPATH_BACKBONE_DEPTH - 1:
            probe_blocks.append(GIGAPATH_BACKBONE_DEPTH - 1)
        return SpectralBackboneEvaluator(
            device=device,
            spectral_probe_blocks=probe_blocks,
            spectral_mode=spectral_mode,
            spectral_k=spectral_k,
        )
    else:
        raise ValueError(f"Unknown backbone evaluator: {name}")

# git commit -am "gigaq: FeatureBag: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag('GIGAPATH_BASELINE_CPTAC_SAMPLE').set(device='cuda', gpu_batch_size=1024).build()"
def gigapath_feature_bag(name: str = None, *, root: str = None) -> FeatureBag:
    if name is None:
        return FeatureBag
    elif name == "GIGAPATH_BASELINE_CPTAC_SAMPLE":
        return FeatureBag(
            root=root,
            spec=dict(
                tilebag=dbx.quote(pancan_tile_bag, 'CPTAC_SAMPLE'),
                extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR'),
        ))
    else:
        raise ValueError(f"Unknown feature shard: {name}")

# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC', n_devices=2, n_workers=16).set(gpu_batch_size=1024).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_9802_TEST', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_9802_TRAIN', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', n_workers=16).build()"
#
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_206020_CALIBRATE', n_workers=16).build_tree()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_206020_TRAIN', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_206020_TEST', n_workers=16).build()"
#
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159_CALIBRATE', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159_TRAIN', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159_TEST', n_workers=16).build()"
#
# git commit -am "gigaq: FeatureBagClip: BUILD" > /dev/null || true; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_200179_CALIBRATE', n_workers=16, cpu_parallelization='ray').build()"
# git commit -am "gigaq: FeatureBagClip: BUILD" > /dev/null || true; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_200179_TRAIN', n_workers=16, cpu_parallelization='ray').build()"
# git commit -am "gigaq: FeatureBagClip: BUILD" > /dev/null || true; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_200179_TEST', n_workers=16, cpu_parallelization='ray').build()"

@tagged
def gigapath_feature_bag_clip(name:str = None, *, tag: str | None = None, root:str = None, n_workers: int = 1, n_devices: int = 1, gpu_batch_size: int = 1024, cpu_batch_size: int = None,
                                 gpu_parallelization: Literal['Multiprocessing', 'Multithreading'] = 'multithreading', 
                                 cpu_parallelization: Literal['Ray', 'Multiprocessing', 'Multithreading', 'Inline'] = 'ray',
                                 bag_n_workers: int = 1,
                                 bag_cpu_batch_size: int|None = None,
                                 bag_cpu_parallelization: str|None = None,
    ) -> FeatureBagClip:
    devices = [f'cuda:{i}' for i in range(n_devices)]
    if name is None:
        return FeatureBagClip
    extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
    if name == "GIGAPATH_BASELINE_CPTAC":
        tilebagclip=dbx.quote(pancan_tile_clip, 'CPTAC')
    elif name == "GIGAPATH_BASELINE_CPTAC_9802_TEST":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_9802_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_9802_TRAIN":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_9802_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TRAIN":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_8020_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_404020_CALIBRATE":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_404020_CALIBRATE')
    elif name == "GIGAPATH_BASELINE_CPTAC_404020_TRAIN":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_404020_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_404020_TEST":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_404020_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_206020_CALIBRATE":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_206020_CALIBRATE')
    elif name == "GIGAPATH_BASELINE_CPTAC_206020_TRAIN":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_206020_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_206020_TEST":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_206020_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_400159_CALIBRATE":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_400159_CALIBRATE')
    elif name == "GIGAPATH_BASELINE_CPTAC_400159_TRAIN":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_400159_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_400159_TEST":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_400159_TEST')
    elif name == "GIGAPATH_BASELINE_5B_CPTAC_8020_TEST":
        tilebagclip = dbx.quote(pancan_tile_fold, 'CPTAC_8020_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_200179_CALIBRATE":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_200179_CALIBRATE')
    elif name == "GIGAPATH_BASELINE_CPTAC_200179_TRAIN":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_200179_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_200179_TEST":
        tilebagclip=dbx.quote(pancan_tile_fold, 'CPTAC_200179_TEST')
    else:
        raise ValueError(f"Unknown gigapath_feature_clip: {repr(name)}")
    return FeatureBagClip(
        root=root, 
        spec=dict(
            extractor=extractor, 
            tilebagclip=tilebagclip
        ), 
        n_workers=n_workers,
        devices=devices, 
        gpu_batch_size=gpu_batch_size,
        cpu_batch_size=cpu_batch_size,
        gpu_parallelization=gpu_parallelization,
        cpu_parallelization=cpu_parallelization,
        bag_n_workers=bag_n_workers,
        bag_cpu_batch_size=bag_cpu_batch_size,
        bag_cpu_parallelization=bag_cpu_parallelization,
        tag=tag,
    )


# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC')[0]"
#
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_8020_TEST')[0]"
#
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_9802_TEST')[0]"
#
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159_CALIBRATE')[0]"
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159_TRAIN')[0]"
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159_TEST')[0]"
#
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_200179_CALIBRATE')[0]"
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_200179_TRAIN')[0]"
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_200179_TEST')[0]"
#
def gigapath_featurebag_dataset(name, *, root: str = None, shuffle_bags_seed: int = None, 
                              n_workers: int = 1, n_devices: int = 1, gpu_batch_size: int = 1024, cpu_batch_size: int = None,
                              cpu_parallelization: str = 'Inline', gpu_parallelization: str = 'Multithreading'
    ) -> torch.utils.data.Dataset:
    featureclip = gigapath_feature_bag_clip(name, root=root, n_workers=n_workers, n_devices=n_devices, gpu_batch_size=gpu_batch_size, cpu_batch_size=cpu_batch_size, cpu_parallelization=cpu_parallelization, gpu_parallelization=gpu_parallelization)
    dbx.Logger().debug(f"===================> {featureclip=}\nquoted featureclip={dbx.quote(featureclip)}")
    return featurebag_dataset(dbx.quote(featureclip), bags_shuffle_seed=shuffle_bags_seed)


def gigapath_featurebag_dataloader_builder(name, root: str = None, shuffle_bags_seed: int = None, 
                                          n_workers: int = 1, n_devices: int = 1, gpu_batch_size: int = 1024, cpu_batch_size: int = None,
                                          cpu_parallelization: str = 'Inline', gpu_parallelization: str = 'Multithreading',
                                          shuffle_bags: bool = False,
                                          **dataloader_kwargs):
    featureclip = gigapath_feature_bag_clip(name, root=root, n_workers=n_workers, n_devices=n_devices, gpu_batch_size=gpu_batch_size, cpu_batch_size=cpu_batch_size, cpu_parallelization=cpu_parallelization, gpu_parallelization=gpu_parallelization)
    clip_dataset_builder = TileClipDatasetBuilder(spec=dict(clip=dbx.quote(featureclip), shuffle_seed=shuffle_bags_seed))
    return TileClipDataLoaderBuilder(spec=dict(
                            clip_dataset_builder=clip_dataset_builder,
                            batch_size=dataloader_kwargs.get('batch_size', None),
                            shuffle=shuffle_bags or dataloader_kwargs.get('shuffle', False),
                          ),  
                          dataloader_kwargs=dataloader_kwargs,
    )


# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TEST', 10, batch_size=1, num_workers=1)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 10, batch_size=1, num_workers=1)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 20, batch_size=1, num_workers=2)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=1)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=2)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=1, prefetch_factor=1)" 2.71s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=4, prefetch_factor=1)" 2.61s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=2, prefetch_factor=None)" 2.58s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=2, prefetch_factor=1)" 2.47s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=8, num_workers=1, prefetch_factor=2)" 2.47s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=8, num_workers=1, prefetch_factor=1,)" 2.38s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=4, prefetch_factor=1,)" 2.28s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=1, prefetch_factor=1,)" 2.25s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=4, prefetch_factor=None)" 1.39s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=8, prefetch_factor=1)" ~=
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=2, prefetch_factor=2)" ~=
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=2, prefetch_factor=1)" ~=
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=8, num_workers=2, prefetch_factor=1)" ~=
#
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159_CALIBRATE', 100, batch_size=8, num_workers=2, prefetch_factor=1)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159_TRAIN', 100, batch_size=8, num_workers=2, prefetch_factor=1)"
#
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_206020_CALIBRATE', 100, batch_size=8, num_workers=2, prefetch_factor=1)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_206020_TRAIN', 100, batch_size=8, num_workers=2, prefetch_factor=1)"
#
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_200179_CALIBRATE', 100, batch_size=8, num_workers=2, prefetch_factor=1)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigapath.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_200179_TRAIN', 100, batch_size=8, num_workers=2, prefetch_factor=1)"
#
def gigapath_featurebag_dataloader_samples(name, n, root: str = None, shuffle_bags: bool = False, return_last: bool = True, **dataloader_kwargs):
    batch_size = dataloader_kwargs.get('batch_size', None)
    dataloader_builder = gigapath_featurebag_dataloader_builder(name, root=root, shuffle_bags=shuffle_bags, **dataloader_kwargs)
    progress = tqdm(total=n)
    for i, _ in enumerate(dataloader_builder.dataloader()):
        progress.update(batch_size if batch_size is not None else 1)
        if i*batch_size >= n-1:
            break
    if return_last:
        return _



# git commit -am "gigaq: FeatureBagMedianProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bags_median_probe('GIGAPATH_BASELINE_CPTAC_8020_TRAIN').build()"
# git commit -am "gigaq: FeatureBagMedianProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bags_median_probe('GIGAPATH_BASELINE_CPTAC_206020_CALIBRATE', n_workers=4, cpu_parallelization='multithreading').build()"
# git commit -am "gigaq: FeatureBagMedianProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bags_median_probe('GIGAPATH_BASELINE_CPTAC_400159_CALIBRATE').build()"
# git commit -am "gigaq: FeatureBagMedianProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_bags_median_probe('GIGAPATH_BASELINE_CPTAC_200179_CALIBRATE', n_workers=4, cpu_parallelization='ray').build()"
@tagged
def gigapath_feature_bags_median_probe(name, *, tag: str | None = None, n_devices: int = 1, gpu_batch_size: int = 16, n_workers: int = 1, cpu_batch_size: int = None,
                                    cpu_parallelization: str = 'Inline', gpu_parallelization: str = 'Multithreading'
    ) -> FeatureBagMedianProbe:
    """Compute the element-wise median (and min/max) of GigaPath tile features across all bags in a clip.

    Constructs a ``FeatureBagMedianProbe(Datablock)`` that, when built, iterates over every
    ``FeatureBag`` in the clip identified by ``name``, concatenates their
    tile-level feature vectors, and computes the per-dimension median, min,
    and max.  The resulting median vector is used downstream by
    ``gigapath_bipolar_feature_bags_probe`` to binarise continuous features
    into bipolar (+1 / -1) representations.

    Args:
        name: Named configuration selecting the ``FeatureBagClip`` to probe
              (e.g. ``'GIGAPATH_BASELINE_CPTAC_200179_CALIBRATE'``).
        tag: Optional human-readable pipeline tag propagated to the clip.
        n_devices: Number of GPU devices used for feature extraction.
        gpu_batch_size: Batch size per GPU device.
        n_workers: Number of CPU workers for parallel bag loading.
        cpu_batch_size: Optional batch size for CPU-side parallel execution.
        cpu_parallelization: CPU parallelization strategy (``'Inline'``,
            ``'ray'``, ``'multiprocessing'``, ``'multithreading'``).
        gpu_parallelization: GPU parallelization strategy.

    Returns:
        A ``FeatureBagMedianProbe`` datablock whose ``median``, ``min``, and
        ``max`` properties expose the computed statistics after building.
    """
    return FeatureBagMedianProbe(spec=dict(featurebagclip=gigapath_feature_bag_clip(name, tag=tag, n_devices=n_devices, gpu_batch_size=gpu_batch_size, n_workers=n_workers, cpu_batch_size=cpu_batch_size, cpu_parallelization=cpu_parallelization, gpu_parallelization=gpu_parallelization),))


# git commit -am "gigaq: BipolarFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_400159').build()"
# git commit -am "gigaq: BipolarFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_400159', n_devices=1).build_tree()"
#
# git commit -am "gigaq: BipolarFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_200179', n_devices=1).build_tree()"
@tagged
def gigapath_bipolar_feature_bags_probe(name, 
                                        *, 
                                        tag: str | None = None,
                                        cpu_parallelization: str = 'ray',
                                        cpu_batch_size: int = None,
                                        n_workers: int = 2,
                                        bag_cpu_parallelization: str = 'multiprocessing',
                                        bag_cpu_batch_size: int = 16,
                                        bag_n_workers: int = 1,
                                        gpu_parallelization: str = 'multithreading',    
                                        n_devices: int = 1, 
                                        gpu_batch_size: int = 16,
    ) -> BipolarFeatureBagProbe:
    """Bipolarise GigaPath tile features and compute comprehensive clip-level statistics.

    Constructs a ``BipolarFeatureBagProbe(Datablock)`` that operates at the
    **clip level**: it processes *all* bags in the clip together and saves
    the results as single clip-wide files — nothing is saved per bag at
    this stage.  When built, the probe:

    1. Loads all tile-level features from the ``_TRAIN`` split of ``name``
       by iterating over every ``FeatureBag`` in the ``FeatureBagClip``.
    2. Obtains the per-dimension median from the ``_CALIBRATE`` split via
       ``gigapath_feature_bags_median_probe``.
    3. Thresholds every tile feature at the median to produce a single
       clip-wide bipolar (+1 / -1) tile-feature matrix
       (``tile_bipolar_features.npz``).
    4. Aggregates bipolar tile features into bag-level bipolar features
       (majority vote per dimension) and continuous bag-level means,
       stored as clip-wide arrays (``bag_features.npz``,
       ``bag_bipolar_features.npz``).
    5. Runs logistic-regression evaluation (via ``LogisticFeatureBagProber``)
       on both continuous and bipolar representations at the bag and tile
       levels, saving the classification reports as
       ``bag_logistic_evaluation_reports.pkl`` and
       ``tile_logistic_evaluation_reports.pkl``.
    6. Computes additional clip-level statistics: distinctness ratios at
       tile, bag, and label granularity; pairwise Hamming distances
       between bags; and bag-level cosine similarities.

    The per-bag materialisation of bipolar features happens downstream in
    ``gigapath_bipolar_feature_bag_clip``, which slices the probe's
    clip-wide ``tile_bipolar_features`` into individual
    ``BipolarFeatureBag`` datablocks.

    ``name`` should be the *base* split identifier without ``_TRAIN`` /
    ``_CALIBRATE`` suffixes (e.g. ``'GIGAPATH_BASELINE_CPTAC_400159'``);
    the function appends them automatically.

    Args:
        name: Base split name — ``_TRAIN`` is used for feature extraction,
              ``_CALIBRATE`` for median computation.
        tag: Optional human-readable pipeline tag.
        cpu_parallelization: Strategy for CPU-bound parallel work.
        cpu_batch_size: Batch size for CPU-side executors.
        n_workers: CPU worker count for the feature-bag clip.
        bag_cpu_parallelization: Parallelization for bag formation.
        bag_cpu_batch_size: Batch size for bag formation.
        bag_n_workers: Worker count for bag formation.
        gpu_parallelization: GPU parallelization strategy.
        n_devices: Number of GPU devices.
        gpu_batch_size: Batch size per GPU.

    Returns:
        A ``BipolarFeatureBagProbe`` datablock exposing clip-wide tile- and
        bag-level bipolar features, labels, statistics, logistic-regression
        evaluation reports, and similarity matrices.
    """
    return BipolarFeatureBagProbe(
        spec=dict(featurebagclip=gigapath_feature_bag_clip(f"{name}_TRAIN", 
                                                            tag=tag,
                                                            gpu_parallelization=gpu_parallelization,
                                                            n_devices=n_devices, 
                                                            gpu_batch_size=gpu_batch_size, 
                                                            cpu_parallelization=cpu_parallelization, 
                                                            n_workers=n_workers, 
                                                            cpu_batch_size=cpu_batch_size, 
                                                            bag_cpu_parallelization=bag_cpu_parallelization,
                                                            bag_cpu_batch_size=bag_cpu_batch_size,
                                                            bag_n_workers=bag_n_workers,
                                                        ), 
                  medianprobe=gigapath_feature_bags_median_probe(f"{name}_CALIBRATE", 
                                                                 tag=tag,
                                                                 n_devices=n_devices, gpu_batch_size=gpu_batch_size, n_workers=n_workers, cpu_batch_size=cpu_batch_size, cpu_parallelization=cpu_parallelization, gpu_parallelization=gpu_parallelization),
        ),
        devices=[f'cuda:{i}' for i in range(n_devices)],
    )
 
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_8020', n_workers=4).build()"
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_8020').build()"
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159', cpu_parallelization='multithreading', n_workers=4).build_tree()"
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_206020').build()"
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_200179', cpu_parallelization='multiprocessing', n_workers=4).build_tree()"
#
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159', single=0).build()"
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159', single=1).build()"
@tagged
def gigapath_bipolar_feature_bag_clip(name, 
                                     *, 
                                     tag: str | None = None,
                                     root: str = None, 
                                     build_missing_only: bool = False, 
                                     single: int|None = None, 
                                     cpu_parallelization: str|None = None,
                                     cpu_batch_size: int = None,
                                     n_workers: int = 2,
                                     bag_cpu_batch_size: int = 0,
                                     bag_cpu_parallelization: str = None,
                                     bag_n_workers: int = 0,
                                     gpu_parallelization: str = 'multithreading',
                                     n_devices: int = 1,
                                     gpu_batch_size: int = 16,
    ) -> BipolarFeatureBagClip:
    """Materialise a clip of per-bag bipolar features for a GigaPath split.

    This is the **per-bag materialisation** step, complementing the
    clip-level analysis performed by ``gigapath_bipolar_feature_bags_probe``.
    Wraps the probe inside a ``BipolarFeatureBagClip(Clip)`` (or
    ``BipolarSingleFeatureBagClip(Clip)`` when ``single`` is provided).

    During ``build()``, each ``BipolarFeatureBag(Bag)`` reads the probe's
    clip-wide ``tile_bipolar_features`` array, slices out its segment using
    ``probe.bag_bounds[i]:probe.bag_bounds[i+1]``, and writes a single
    ``bipolar_features.npz`` to its own directory.  This converts the
    probe's monolithic clip-level arrays into independent per-bag
    datablocks, enabling efficient random access and downstream dataset
    construction via ``gigapath_bipolar_featurebag_dataset``.

    In contrast to the probe (which saves clip-wide statistics, logistic
    evaluation reports, and similarity matrices as single files), this clip
    saves only the per-bag feature slices and a ``bag_lens`` index.

    Args:
        name: Base split name (e.g. ``'GIGAPATH_BASELINE_CPTAC_400159'``).
            ``_TRAIN`` and ``_CALIBRATE`` suffixes are appended internally
            by the probe.
        tag: Optional human-readable pipeline tag.
        root: Alternative storage root for the clip artefacts.
        build_missing_only: If ``True``, only build bags that are not
            already materialised on disk.
        single: If set, return a ``BipolarSingleFeatureBagClip`` containing
            only the bag at this index — useful for fast debugging or
            single-slide experiments.
        cpu_parallelization: Strategy for CPU-bound parallel work.
        cpu_batch_size: Batch size for CPU-side executors.
        n_workers: CPU worker count for building bags.
        bag_cpu_batch_size: Batch size for bag formation.
        bag_cpu_parallelization: Parallelization for bag formation.
        bag_n_workers: Worker count for bag formation.
        gpu_parallelization: GPU parallelization strategy.
        n_devices: Number of GPU devices.
        gpu_batch_size: Batch size per GPU.

    Returns:
        A ``BipolarFeatureBagClip`` (or ``BipolarSingleFeatureBagClip``)
        whose ``.bags`` property yields ``BipolarFeatureBag`` instances,
        each storing its own ``bipolar_features.npz``.
    """
    devices = [f'cuda:{i}' for i in range(n_devices)]
    probe = dbx.quote(
        gigapath_bipolar_feature_bags_probe, 
        name=name,
        gpu_parallelization=gpu_parallelization, 
        n_devices=n_devices, 
        gpu_batch_size=gpu_batch_size, 
        cpu_parallelization=cpu_parallelization,
        n_workers=n_workers, 
        cpu_batch_size=cpu_batch_size,
        bag_cpu_parallelization=bag_cpu_parallelization,
        bag_n_workers=bag_n_workers,
        bag_cpu_batch_size=bag_cpu_batch_size,
    )
    if single is not None:
        dbx.Logger(name='gigapath_bipolar_feature_bag_clip').debug(f"====================> single: {repr(single)}\nname: {repr(name)}")
        clip = BipolarSingleFeatureBagClip(
            root=root, 
            spec=dict(probe=probe, idx=single), 
            cpu_parallelization=cpu_parallelization,
            n_workers=n_workers, 
            cpu_batch_size=cpu_batch_size, 
            gpu_parallelization=gpu_parallelization,
            tag=tag,
            )
    else:
        clip = BipolarFeatureBagClip(
            root=root, 
            spec=dict(probe=probe), 
            build_missing_only=build_missing_only, 
            cpu_parallelization=cpu_parallelization,
            cpu_batch_size=cpu_batch_size,
            n_workers=n_workers,
            bag_cpu_parallelization=bag_cpu_parallelization,
            bag_n_workers=bag_n_workers,
            bag_cpu_batch_size=bag_cpu_batch_size,
            gpu_parallelization=gpu_parallelization,
            gpu_batch_size=gpu_batch_size,
            devices=devices,
            tag=tag,
        )
    return clip

# git commit -am "gigaq: BipolarFeaturebagDataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_8020')[0]"
# git commit -am "gigaq: BipolarFeaturebagDataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159')[0]"
# git commit -am "gigaq: BipolarFeaturebagDataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_206020')[0]"
#
# git commit -am "gigaq: BipolarFeaturebagDataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159_SINGLE_0')[0]"
# git commit -am "gigaq: BipolarFeaturebagDataset: TEST"; dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159_SINGLE_1')[0]"
def gigapath_bipolar_featurebag_dataset(name, *, root: str = None, shuffle_bags_seed: int = None,
                                      n_workers: int = 0, n_devices: int = 1, gpu_batch_size: int = 16,
                                      cpu_parallelization: str = 'Inline', gpu_parallelization: str = 'Multithreading'
    ) -> torch.utils.data.Dataset:
    if 'SINGLE' in name:
        single = int(name.split('_')[-1])
        name = name.split('_SINGLE')[0]
        dbx.Logger(name='gigapath_bipolar_featurebag_dataset').debug(f"===================> single: {repr(single)}\nname: {repr(name)}")
        featureclip = gigapath_bipolar_feature_bag_clip(name, root=root, single=single, n_workers=n_workers, n_devices=n_devices, gpu_batch_size=gpu_batch_size, cpu_parallelization=cpu_parallelization, gpu_parallelization=gpu_parallelization)
    else:
        featureclip = gigapath_bipolar_feature_bag_clip(name, root=root, n_workers=n_workers, n_devices=n_devices, gpu_batch_size=gpu_batch_size, cpu_parallelization=cpu_parallelization, gpu_parallelization=gpu_parallelization)
    dbx.Logger(name='gigapath_bipolar_featurebag_dataset').debug(f"===================> {featureclip=}\nquoted featureclip {dbx.quote(featureclip)}")
    transform = dbx.quote(FeaturesToFloat, dtype='float32')
    target_transform = dbx.quote(FeaturesLabelTileToFloat, dtype='float32')
    return featurebag_dataset(dbx.quote(featureclip), transform=transform, target_transform=target_transform, bags_shuffle_seed=shuffle_bags_seed)


def gigapath_bipolar_featurebag_dataloader_builder(name, root: str = None, shuffle_bags_seed: int = None, 
                                                  n_workers: int = 0, n_devices: int = 1, gpu_batch_size: int = 16,
                                                  cpu_parallelization: str = 'Inline', gpu_parallelization: str = 'Multithreading',
                                                  shuffle_bags: bool = False,
                                                  **dataloader_kwargs):
    if 'SINGLE' in name:
        single = int(name.split('_')[-1])
        name = name.split('_SINGLE')[0]
        featureclip = gigapath_bipolar_feature_bag_clip(name, root=root, single=single, n_workers=n_workers, n_devices=n_devices, gpu_batch_size=gpu_batch_size, cpu_parallelization=cpu_parallelization, gpu_parallelization=gpu_parallelization)
    else:
        featureclip = gigapath_bipolar_feature_bag_clip(name, root=root, n_workers=n_workers, n_devices=n_devices, gpu_batch_size=gpu_batch_size, cpu_parallelization=cpu_parallelization, gpu_parallelization=gpu_parallelization)
    transform = dbx.quote(FeaturesToFloat, dtype='float32')
    target_transform = dbx.quote(FeaturesLabelTileToFloat, dtype='float32')
    clip_dataset_builder = TileClipDatasetBuilder(spec=dict(clip=dbx.quote(featureclip), transform=transform, target_transform=target_transform, shuffle_seed=shuffle_bags_seed))
    return TileClipDataLoaderBuilder(spec=dict(
                            clip_dataset_builder=clip_dataset_builder,
                            batch_size=dataloader_kwargs.get('batch_size', None),
                            shuffle=shuffle_bags or dataloader_kwargs.get('shuffle', False),
                          ),  
                          dataloader_kwargs=dataloader_kwargs,
    )
 

# git commit -am "gigaq: BipolarFeaturebagDataloader: SAMPLES"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020', 1000)"
# git commit -am "gigaq: BipolarFeaturebagDataloader: SAMPLES"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159', 1000)"
#
# git commit -am "gigaq: BipolarFeaturebagDataloader: SAMPLES"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159_SINGLE_0', 1000)"
# git commit -am "gigaq: BipolarFeaturebagDataloader: SAMPLES"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159_SINGLE_1', 1000)"
#
# git commit -am "gigaq: BipolarFeaturebagDataloader: SAMPLES"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159_SINGLE_1', 1000, shuffle=True)"
def gigapath_bipolar_featurebag_dataloader_samples(name, n, root: str = None, shuffle_bags: bool = False, return_last: bool = True, **dataloader_kwargs):
    batch_size = dataloader_kwargs.get('batch_size', 1)
    dataloader_builder = gigapath_bipolar_featurebag_dataloader_builder(name, root=root, shuffle_bags=shuffle_bags, **dataloader_kwargs)
    progress = tqdm(total=n, desc=f"READING SAMPLES")
    for i, _ in enumerate(dataloader_builder.dataloader()):
        progress.update(batch_size if batch_size is not None else 1)
        if i*batch_size >= n-1:
            break
    if return_last:
        return _



# git commit -am "gigaq: HYDRO"; dbx.pprint "autopath.gigapath.pipelines.gigapath_hydro('GIGAPATH_BIPOLAR_HYDRO')"
def gigapath_hydro(name, *, model: Literal['cnn', 'vit'] = 'cnn', **kwargs):
    """Create a Hydro configuration by name.
    
    Args:
        name: Configuration name (e.g., 'GIGAPATH_BIPOLAR_HYDRO')
        model: Model architecture identifier (e.g., 'cnn' or 'vit')
        **kwargs: Override default parameters
        
    Returns:
        Hydro
    """
    if name == "GIGAPATH_BIPOLAR_HYDRO":
        latent_dim = kwargs.get('latent_dim', 1536)
        image_size = kwargs.get('image_size', 256)
        cnn_initial_size = kwargs.get('cnn_initial_size', 8)
        cnn_hidden_channels = kwargs.get('cnn_hidden_channels', 256)
    elif name == "GIGAPATH_BIPOLAR_HYDRO_SMALL":
        latent_dim = kwargs.get('latent_dim', 1536)
        image_size = kwargs.get('image_size', 256)
        cnn_initial_size = kwargs.get('cnn_initial_size', 8)
        cnn_hidden_channels = kwargs.get('cnn_hidden_channels', 128)
    elif name == "GIGAPATH_BIPOLAR_HYDRO_LARGE":
        latent_dim = kwargs.get('latent_dim', 1536)
        image_size = kwargs.get('image_size', 256)
        cnn_initial_size = kwargs.get('cnn_initial_size', 8)
        cnn_hidden_channels = kwargs.get('cnn_hidden_channels', 512)
    else:
        raise ValueError(f"Unknown gigapath_hydro: {name}")
    
    hydro = Hydro(spec=dict(
        model=model,
        latent_dim=latent_dim,
        image_size=image_size,
        cnn_initial_size=cnn_initial_size,
        cnn_hidden_channels=cnn_hidden_channels,
        cnn_use_batch_norm=kwargs.get('cnn_use_batch_norm', True),
        cnn_use_bilinear_upsampling=kwargs.get('cnn_use_bilinear_upsampling', True),
        cnn_use_pixel_shuffle=kwargs.get('cnn_use_pixel_shuffle', False),
        cnn_use_residual=kwargs.get('cnn_use_residual', False),
        cnn_use_residual_upsampling=kwargs.get('cnn_use_residual_upsampling', False),

        cnn_use_spatial_attention_gates=kwargs.get('cnn_use_spatial_attention_gates', False),
        cnn_use_feature_modulation=kwargs.get('cnn_use_feature_modulation', False),
        vit_patch_size=kwargs.get('vit_patch_size', 16),
        vit_hidden_channels=kwargs.get('vit_hidden_channels', 256),
        vit_n_layers=kwargs.get('vit_n_layers', 12),
        vit_n_heads=kwargs.get('vit_n_heads', 8),
        vit_dim_feedforward=kwargs.get('vit_dim_feedforward', 1024),
        loss_type=kwargs.get('loss_type', 'mse'),
        ssim_companion_weight=kwargs.get('ssim_companion_weight', 0.8),
        ssim_companion_loss=kwargs.get('ssim_companion_loss', 'lpips'),
    ))
    return hydro


"""
### CNN
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_MEDIUM_CPTAC_400159', \\
    log_images=True, n_devices=1, batch_size=6, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_MEDIUM_CPTAC_400159_SINGLE_0', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=64, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_MEDIUM_CPTAC_400159_SINGLE_1', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=64, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_MEDIUM_CPTAC_400159_SINGLE_0', \\
    loss_type='l1', log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_MEDIUM_CPTAC_400159_SINGLE_0', \\
    loss_type='l1', cnn_use_residual_upsampling=True, cnn_use_spatial_attention_gates=False, cnn_use_feature_modulation=False, suffix='cnn_resup_nosag_nofmod_l1', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True \
    dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_HIGHEST_CPTAC_400159_SINGLE_0', \\
    loss_type='l1', cnn_use_residual_upsampling=True, cnn_use_spatial_attention_gates=True, cnn_use_feature_modulation=True, suffix='cnn_resup_sag_fmod_l1', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True \
    dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_HIGHEST_CPTAC_400159_SINGLE_0', \\
    loss_type='lpips', cnn_use_residual_upsampling=True, cnn_use_spatial_attention_gates=True, cnn_use_feature_modulation=True, suffix='cnn_resup_sag_fmod_lpips', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True \
    dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_HIGHEST_CPTAC_400159_SINGLE_0', \\
    loss_type='lpips', cnn_use_residual_upsampling=True, cnn_use_bilinear_upsampling=True, cnn_use_spatial_attention_gates=True, cnn_use_feature_modulation=True, \\
    model='cnn', suffix='cnn_resup_bilup_sag_fmod_lpips', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
### ViT
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True \
    dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_HIGHEST_CPTAC_400159_SINGLE_0', \\
    model='vit', loss_type='lpips',  suffix='vit_lpips', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True \
    dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_HIGHEST_CPTAC_400159_SINGLE_0', \\
    model='vit', loss_type='ssim',  suffix='vit_ssim', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True \
    dbx.pprint "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_HIGHEST_CPTAC_400159_SINGLE_0', \\
    model='vit', loss_type='ssim',  ssim_companion_loss='lpips', ssim_companion_weight=0.1, suffix='vit_ssim_lpips_0.8',   \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
###: slurm
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True \
    dbx.slurm.pprint \
    "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_HIGHEST_CPTAC_400159_SINGLE_0', \\
    cnn_use_residual_upsampling=True, cnn_use_spatial_attention_gates=True, cnn_use_feature_modulation=True, \\
    loss_type='ssim', ssim_companion_loss='lpips', ssim_companion_weight=0.0, suffix='cnn_resup_sag_fmod_ssim', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()" \
    nodelist=radish mem=4G gpus=1 cpus=2
### interactive
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True \
    dbx.pprint \
    "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_HIGHEST_CPTAC_206020', \\
    cnn_use_residual_upsampling=True, cnn_use_spatial_attention_gates=True, cnn_use_feature_modulation=True, \\
    loss_type='ssim', ssim_companion_weight=0.0, suffix='cnn_resup_sag_fmod_ssim', \\
    log_images=True, log_images_interval=1, max_epochs=16, n_devices=1, batch_size=46, num_workers=2, prefetch_factor=1, pin_memory=True).build()"
### training set warmup
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True \
    dbx.pprint \
    "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_HIGH_CPTAC_206020', \\
    cnn_use_residual_upsampling=True, cnn_use_spatial_attention_gates=True, cnn_use_feature_modulation=True, \\
    loss_type='ssim', ssim_companion_weight=0.0, \\
    initial_ckpt='/home/t-9dkarp/datalake/autopath.models.hydro.HydroStill/8407f891562e15019daa1cee2a565820602a4492eb48415a1a6ba1fb80bb2e78/ckpts/epoch=2-step=171600.ckpt', \\
    max_epochs=16, max_steps='10%', reset_optimizer_state=True, \\
    suffix='cnn_resup_sag_fmod_ssim_10pct', \\
    log_images=True, log_images_interval=1, \\
    n_devices=1, batch_size=46, num_workers=2, prefetch_factor=1, pin_memory=True).build()"
### training set warmup2: TOO SLOW: BATCH SIZE THAT FITS INTO GRAM IS TOO LOW
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBX_USE_WORK_REPO=True \
    dbx.pprint \
    "autopath.gigapath.pipelines.gigapath_bipolar_hydro_still('GIGAPATH_BIPOLAR_HYDRO_HIGH_CPTAC_206020', \\
    cnn_use_residual_upsampling=True, cnn_use_spatial_attention_gates=True, cnn_use_feature_modulation=True, \\
    loss_type='ssim', ssim_companion_weight=0.0, \\
    initial_ckpt='/home/t-9dkarp/datalake/autopath.models.hydro.HydroStill/1f67c4f05eb5ddfa858e4e4ab703fd4aed68c309165d0fb479a94ba428c653db/ckpts/epoch=2-step=14800.ckpt', \\
    max_epochs=16, max_steps='10%', \\
    suffix='cnn_resup_sag_fmod_ssim_10pct_2', \\
    log_images=True, log_images_interval=1, \\
    n_devices=3, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).build()"
"""
def gigapath_bipolar_hydro_still(
    hydro_dataset_name = None, 
    *,  
    n_devices: int = 1,
    logsroot: str = None,
    log_images: bool = False,
    log_images_interval: int = 100,
    from_scratch: bool = False,
    reset_optimizer_state: bool = False,
    initial_ckpt: str = None,
    max_epochs: int = 1,
    max_steps: int = None,
    ckpt_every_n_steps: int = 100,
    use_bags: bool = True,
    shuffle_bags_seed: int = 42,
    learning_rate: float = 1e-4,
    scheduler: str = 'cosine',
    gradient_clip_algorithm: str = 'norm',
    gradient_clip_val: float = 10.0,
    loss_type: str = 'mse',
    cnn_use_batch_norm: bool = True,
    cnn_use_bilinear_upsampling: bool = False,
    cnn_use_pixel_shuffle: bool = False,
    cnn_use_residual: bool = False,
    cnn_use_residual_upsampling: bool = False,
    cnn_use_spatial_attention_gates: bool = False,
    cnn_use_feature_modulation: bool = False,
    model: Literal['cnn', 'vit'] = 'cnn',
    ssim_companion_weight: float = 0.8,
    ssim_companion_loss: str = 'lpips',
    suffix: str|None = None,
    **dataloader_kwargs,
):
    """Create a HydroStill training pipeline.
    
    Args:
        hydro_dataset_name: Name in format 'GIGAPATH_BIPOLAR_HYDRO_<precision>_CPTAC_<split>'
        n_devices: Number of GPU devices
        logsroot: Root directory for tensorboard logs
        log_images: Whether to log images during training
        log_images_interval: Interval (in steps) to log images
        from_scratch: Whether to restart training from scratch
        reset_optimizer_state: Whether to reset optimizer state when resuming from checkpoint
        initial_ckpt: Optional path to an initial checkpoint to load
        model: Model architecture identifier (e.g., 'cnn' or 'vit')
        **dataloader_kwargs: DataLoader parameters (batch_size, num_workers, etc.)
        
    Returns:
        HydroStill instance
    """
    if hydro_dataset_name is None:
        still = HydroStill
    else:
        hydroname_precision, _splitname = hydro_dataset_name.split('_CPTAC_')
        splitname = "GIGAPATH_BASELINE_CPTAC_" + _splitname
        bits = hydroname_precision.split('_')
        # GIGAPATH_BIPOLAR_HYDRO_MEDIUM -> bits = ['GIGAPATH', 'BIPOLAR', 'HYDRO', 'MEDIUM']
        # GIGAPATH_BIPOLAR_HYDRO -> bits = ['GIGAPATH', 'BIPOLAR', 'HYDRO']
        if bits[-1] in ['HIGH','MEDIUM', 'HIGHEST']:
            precision = bits[-1].lower()
            hydroname = '_'.join(bits[:-1])
        else:
            precision = 'highest'
            hydroname = hydroname_precision

        hydro = gigapath_hydro(hydroname, 
                             model=model,
                             loss_type=loss_type, 
                             cnn_use_batch_norm=cnn_use_batch_norm, 
                             cnn_use_bilinear_upsampling=cnn_use_bilinear_upsampling,
                             cnn_use_pixel_shuffle=cnn_use_pixel_shuffle,
                             cnn_use_residual=cnn_use_residual,
                             cnn_use_residual_upsampling=cnn_use_residual_upsampling,
                             cnn_use_spatial_attention_gates=cnn_use_spatial_attention_gates,
                             cnn_use_feature_modulation=cnn_use_feature_modulation,
                             ssim_companion_weight=ssim_companion_weight,
                             ssim_companion_loss=ssim_companion_loss)

        tag = f"{hydro_dataset_name}/{suffix}" if suffix is not None else f"{hydro_dataset_name}"

        logsroot = logsroot or f'{os.environ["HOME"]}/autopath/tensorboard/hydro'
        
        if use_bags:
            featureloader_builder = dbx.quote(gigapath_bipolar_featurebag_dataloader_builder, splitname, shuffle_bags_seed=shuffle_bags_seed, **dataloader_kwargs)
        else:
            raise NotImplementedError(f"Shard dataloader_builder")
        
        lightning = HydroLightning(
            spec=dict(hydro=hydro, 
                     learning_rate=learning_rate, 
                     scheduler=scheduler, 
                     log_images=log_images, log_images_interval=log_images_interval,
                     cnn_use_pixel_shuffle=cnn_use_pixel_shuffle,
                     cnn_use_residual=cnn_use_residual,
                     cnn_use_residual_upsampling=cnn_use_residual_upsampling,
                     cnn_use_spatial_attention_gates=cnn_use_spatial_attention_gates,
                     cnn_use_feature_modulation=cnn_use_feature_modulation,
                     ssim_companion_weight=ssim_companion_weight,
                     ssim_companion_loss=ssim_companion_loss)
        )

        still = HydroStill(spec=dict(
                    lightning=lightning, 
                    dataloader=featureloader_builder,
                    max_epochs=max_epochs,
                    max_steps=max_steps,
                    ckpt_every_n_steps=ckpt_every_n_steps,
                    gradient_clip_val=gradient_clip_val,
                    gradient_clip_algorithm=gradient_clip_algorithm,
                    precision=precision,
                    from_scratch=from_scratch,
                    reset_optimizer_state=reset_optimizer_state,
                    initial_ckpt=initial_ckpt,
                    ssim_companion_weight=ssim_companion_weight,
                    ssim_companion_loss=ssim_companion_loss,
                    cnn_use_pixel_shuffle=cnn_use_pixel_shuffle,
                    cnn_use_residual=cnn_use_residual,
                    cnn_use_residual_upsampling=cnn_use_residual_upsampling,
                    cnn_use_spatial_attention_gates=cnn_use_spatial_attention_gates,
                    cnn_use_feature_modulation=cnn_use_feature_modulation,
                ),
            n_devices=n_devices,
            logsroot=logsroot,
            tag=tag,
            **dataloader_kwargs,
        )
    return still
    

# git commit -am "gigaq: LogisticFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_bins=2).build()"
# git commit -am "gigaq: LogisticFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_bins=2).read('evaluation_reports')"
#
# git commit -am "gigaq: LogisticFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_9802_TRAIN', n_bins=2, polarize=True).build()"
#
# git commit -am "gigaq: LogisticFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_9802_TRAIN', n_bins=2, aggregation='cdf').build()"

def gigapath_logistic_feature_bags_probe(name, n_bins: int = 2, polarize: bool = False, aggregation: str = 'mean',
                                         n_workers: int = 1, n_devices: int = 1, gpu_batch_size: int = 1024,
                                         cpu_parallelization: str = 'Inline', gpu_parallelization: str = 'Multithreading'
    ) -> LogisticFeatureBagProbe:
    return LogisticFeatureBagProbe(spec=dict(featurebagclip=gigapath_feature_bag_clip(name, n_workers=n_workers, n_devices=n_devices, gpu_batch_size=gpu_batch_size, cpu_parallelization=cpu_parallelization, gpu_parallelization=gpu_parallelization), n_bins=n_bins, polarize=polarize, aggregation=aggregation))
    

# ---- SphericalFeatureBagClip ------------------------------------------------
# git commit -am "gigaq: SphericalFeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_spherical_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159').build()"
# git commit -am "gigaq: SphericalFeatureBagClip: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_spherical_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159', cpu_parallelization='multithreading', n_workers=4).build_tree()"
@tagged
def gigapath_spherical_feature_bag_clip(name,
                                       *,
                                       tag: str | None = None,
                                       build_missing_only: bool = False,
                                       cpu_parallelization: str | None = None,
                                       cpu_batch_size: int | None = None,
                                       n_workers: int = 1,
                                       bag_cpu_parallelization: str | None = None,
                                       bag_cpu_batch_size: int | None = None,
                                       bag_n_workers: int = 1,
                                       gpu_parallelization: str = 'multithreading',
                                       n_devices: int = 1,
                                       gpu_batch_size: int = 16,
    ) -> SphericalFeatureBagClip:
    """Materialise a clip of L2-normalised (spherical) feature bags.

    Wraps ``gigapath_feature_bag_clip`` inside a ``SphericalFeatureBagClip``.
    During ``build()``, each ``SphericalFeatureBag`` reads the raw features
    from its corresponding ``FeatureBag``, L2-normalises every tile feature
    vector, and persists the result.

    Args:
        name: Named configuration selecting the ``FeatureBagClip`` to
              normalise (e.g. ``'GIGAPATH_BASELINE_CPTAC_400159'``).
        tag: Optional human-readable pipeline tag.
        build_missing_only: If ``True``, only build bags not already on disk.
        cpu_parallelization: Strategy for CPU-bound parallel work.
        cpu_batch_size: Batch size for CPU-side executors.
        n_workers: CPU worker count for building bags.
        bag_cpu_parallelization: Parallelization for bag formation.
        bag_cpu_batch_size: Batch size for bag formation.
        bag_n_workers: Worker count for bag formation.
        gpu_parallelization: GPU parallelization strategy (passed to
            the underlying ``FeatureBagClip``).
        n_devices: Number of GPU devices.
        gpu_batch_size: Batch size per GPU.

    Returns:
        A ``SphericalFeatureBagClip`` whose ``.bags`` yields
        ``SphericalFeatureBag`` instances with unit-norm features.
    """
    return SphericalFeatureBagClip(
        spec=dict(
            featurebagclip=gigapath_feature_bag_clip(
                name,
                tag=tag,
                n_devices=n_devices,
                gpu_batch_size=gpu_batch_size,
                n_workers=n_workers,
                cpu_batch_size=cpu_batch_size,
                cpu_parallelization=cpu_parallelization,
                gpu_parallelization=gpu_parallelization,
            ),
        ),
        tag=tag,
        build_missing_only=build_missing_only,
        cpu_parallelization=cpu_parallelization,
        cpu_batch_size=cpu_batch_size,
        n_workers=n_workers,
        bag_cpu_parallelization=bag_cpu_parallelization,
        bag_cpu_batch_size=bag_cpu_batch_size,
        bag_n_workers=bag_n_workers,
    )


# ---- AffineLogisticFeatureBagProbe ------------------------------------------
# git commit -am "gigaq: AffineLogisticFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_affine_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_400159').build()"
# git commit -am "gigaq: AffineLogisticFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_affine_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_400159', fit_intercept=False).build()"
# git commit -am "gigaq: AffineLogisticFeatureBagProbe: READ";  dbx.pprint "autopath.gigapath.pipelines.gigapath_affine_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_400159').read('intercept')"
#
# Run on spherical features:
# git commit -am "gigaq: AffineLogisticFeatureBagProbe(spherical): BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_affine_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_400159', spherical=True).build()"
# git commit -am "gigaq: AffineLogisticFeatureBagProbe(spherical): BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_affine_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_400159', spherical=True, fit_intercept=False).build()"
# git commit -am "gigaq: AffineLogisticFeatureBagProbe(spherical): READ";  dbx.pprint "autopath.gigapath.pipelines.gigapath_affine_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_400159', spherical=True).read('intercept')"
@tagged
def gigapath_affine_logistic_feature_bags_probe(name,
                                                *,
                                                tag: str | None = None,
                                                spherical: bool = False,
                                                fit_intercept: bool = True,
                                                evaluation_fraction: float = 0.8,
                                                aggregation: str = 'mean',
                                                n_workers: int = 1,
                                                n_devices: int = 1,
                                                gpu_batch_size: int = 16,
                                                cpu_parallelization: str = 'Inline',
                                                gpu_parallelization: str = 'Multithreading',
    ) -> AffineLogisticFeatureBagProbe:
    """Fit a linear (logistic) classifier on GigaPath bag features and persist the model.

    Instantiates an ``AffineLogisticFeatureBagProbe`` that trains a
    ``LogisticRegression`` on bag-level mean features and persists the
    fitted ``coef_``, ``intercept_``, and ``classification_report``.

    When ``spherical=True``, the probe operates on a
    ``SphericalFeatureBagClip`` (L2-normalised features) instead of the
    raw ``FeatureBagClip``.

    Args:
        name: Named configuration selecting the feature clip.
        tag: Optional human-readable pipeline tag.
        spherical: If ``True``, run on L2-normalised features via
            ``gigapath_spherical_feature_bag_clip``.
        fit_intercept: Whether the classifier learns a bias term.
            Set to ``False`` to force separating planes through the origin.
        evaluation_fraction: Train/test split fraction.
        aggregation: Bag-level feature aggregation (``"mean"``).
        n_workers: CPU worker count.
        n_devices: Number of GPU devices.
        gpu_batch_size: Batch size per GPU.
        cpu_parallelization: CPU parallelization strategy.
        gpu_parallelization: GPU parallelization strategy.

    Returns:
        An ``AffineLogisticFeatureBagProbe`` with persisted ``coef``,
        ``intercept``, ``classes``, and ``evaluation_report`` topics.
    """
    if spherical:
        clip = gigapath_spherical_feature_bag_clip(
            name, tag=tag,
            n_workers=n_workers, n_devices=n_devices,
            gpu_batch_size=gpu_batch_size,
            cpu_parallelization=cpu_parallelization,
            gpu_parallelization=gpu_parallelization,
        )
    else:
        clip = gigapath_feature_bag_clip(
            name, tag=tag,
            n_workers=n_workers, n_devices=n_devices,
            gpu_batch_size=gpu_batch_size,
            cpu_parallelization=cpu_parallelization,
            gpu_parallelization=gpu_parallelization,
        )
    return AffineLogisticFeatureBagProbe(
        spec=dict(
            featurebagclip=clip,
            fit_intercept=fit_intercept,
            evaluation_fraction=evaluation_fraction,
            aggregation=aggregation,
        ),
    )

# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000').set(n_devices=3).build()"
# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000').set(n_devices=3).build()"
# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_20_4000_20000').set(n_devices=3).build()"
def gigapath_feature_pairwise_distances(name) -> FeaturePairwiseDistances:
    if name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=10,
                              row_chunk_size=4000,
                              col_chunk_size=4000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=10,
                              row_chunk_size=4000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_20_4000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=20,
                              row_chunk_size=4000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_4000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=4000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_4000_4000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=4000,
                              col_chunk_size=4000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_20000_4000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=20000,
                              col_chunk_size=4000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_5000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=5000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_10000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=10000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_20000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=20000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_50000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=50000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_100000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=100000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_200000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=200000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_400000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=400000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_50_100_800000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=50,
                              row_chunk_size=100,
                              col_chunk_size=800000,
                    ), 
        )
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST_5_1000_150000":
        return FeaturePairwiseDistances(
                    spec=dict(featurebags=dbx.quote(gigapath_feature_bag_clip, 'GIGAPATH_BASELINE_CPTAC_8020_TEST'),
                              n_chunks=5,
                              row_chunk_size=1000,
                              col_chunk_size=150000,
                    ), 
        )
    else:
        raise ValueError(f"Unknown feature bags pairwise distances datablock: {name}")


# git commit -am "gigaq: FeatureSortedDistances: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_sorted_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000',).set(n_workers=3,).build()"
# git commit -am "gigaq: FeatureSortedDistances: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_sorted_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000',).set(n_workers=3,).build()"
def gigapath_feature_sorted_distances(name) -> FeatureSortedDistances:
    return FeatureSortedDistances(
                    spec=dict(features_pairwise_distances=dbx.quote(gigapath_feature_pairwise_distances, name),)
    )
    

# git commit -am "gigaq: Features=2NNDistances: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_2nn_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000',).set(n_workers=3).build()"
# git commit -am "gigaq: Feature2NNDistances: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_2nn_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000',).set(n_workers=3).build()"
def gigapath_feature_2nn_distances(name) -> Feature2NNDistances:
    return Feature2NNDistances(
                    spec=dict(features_sorted_distances=dbx.quote(gigapath_feature_sorted_distances, name),),
        )
    

# git commit -am "gigaq: Feature2NNDim: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_2nn_dim('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000',).build_tree().dim"
# git commit -am "gigaq: Feature2NNDim: BUILD"; dbx.pprint "autopath.gigapath.pipelines.gigapath_feature_2nn_dim('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000',).build_tree().dim"
def gigapath_feature_2nn_dim(name) -> Feature2NNDim:
    return Feature2NNDim(
                    spec=dict(features_2nn_distances=dbx.quote(gigapath_feature_2nn_distances, name),),
        )


# =============================================================================
#  Spectral probing pipelines
# =============================================================================
def quote_spectral_extractor(name='GIGAPATH_SPECTRAL_BACKBONE_EVALUATOR_BLOCKS:5_SPEC:CLS'):
        return dbx.quote(gigapath_backbone_evaluator, name)
        

# git commit -am "gigaq: SpectralFeatureBag: BUILD" > /dev/null || true; dbx.pprint "autopath.gigapath.pipelines.gigapath_spectral_feature_bag('GIGAPATH_CPTAC_SAMPLE_SPECTRAL_BLOCKS:5_SPEC:CLS').set(device='cuda', gpu_batch_size=1024).build()"
# git commit -am "gigaq: SpectralFeatureBag: BUILD" > /dev/null || true; dbx.pprint "autopath.gigapath.pipelines.gigapath_spectral_feature_bag('GIGAPATH_CPTAC_SAMPLE_SPECTRAL_BLOCKS:5CAPPED_SPEC:CLS').set(device='cuda', gpu_batch_size=1024).build()"
# git commit -am "gigaq: SpectralFeatureBag: BUILD" > /dev/null || true; dbx.pprint "autopath.gigapath.pipelines.gigapath_spectral_feature_bag('GIGAPATH_CPTAC_SAMPLE_SPECTRAL_BLOCKS:40CAPPED_SPEC:CLS').set(device='cuda', gpu_batch_size=1024).build()"
# git commit -am "gigaq: SpectralFeatureBag: BUILD" > /dev/null || true; dbx.pprint "autopath.gigapath.pipelines.gigapath_spectral_feature_bag('GIGAPATH_CPTAC_SAMPLE_SPECTRAL_BLOCKS:40CAPPED_SPEC:BOTH10').set(device='cuda', gpu_batch_size=64).build()"

def gigapath_spectral_feature_bag(name: str = None, *, root: str = None) -> SpectralFeatureBag:
    if name is None:
        return SpectralFeatureBag
    elif name.startswith("GIGAPATH_CPTAC_SAMPLE_SPECTRAL_"):
        evaluator = name.removeprefix("GIGAPATH_CPTAC_SAMPLE_SPECTRAL_")
        return SpectralFeatureBag(
            root=root,
            spec=dict(
                tilebag=dbx.quote(pancan_tile_bag, 'CPTAC_SAMPLE'),
                extractor=quote_spectral_extractor(f'GIGAPATH_SPECTRAL_BACKBONE_EVALUATOR_{evaluator}'),
        ))
    else:
        raise ValueError(f"Unknown spectral feature bag: {name}")


# git commit -am "gigaq: SpectralFeatureBagClip: BUILD" > /dev/null || true; dbx.pprint "autopath.gigapath.pipelines.gigapath_spectral_feature_bag_clip('GIGAPATH_SPECTRAL_CPTAC_200179_TEST', n_workers=16).build()"
@tagged
def gigapath_spectral_feature_bag_clip(name: str = None, *, tag: str | None = None, root: str = None,
                                       n_workers: int = 1, n_devices: int = 1,
                                       gpu_batch_size: int = 1,
                                       cpu_batch_size: int | None = None,
                                       gpu_parallelization: Literal['Multiprocessing', 'Multithreading'] = 'multithreading',
                                       cpu_parallelization: Literal['Ray', 'Multiprocessing', 'Multithreading', 'Inline'] = 'ray',
                                       bag_n_workers: int = 1,
                                       bag_cpu_batch_size: int | None = None,
                                       bag_cpu_parallelization: str | None = None,
    ) -> SpectralFeatureBagClip:
    devices = [f'cuda:{i}' for i in range(n_devices)]
    if name is None:
        return SpectralFeatureBagClip
    extractor = quote_spectral_extractor('GIGAPATH_SPECTRAL_BACKBONE_EVALUATOR_BLOCKS:5_SPEC:CLS')
    if name == "GIGAPATH_SPECTRAL_CPTAC_200179_TEST":
        tilebagclip = dbx.quote(pancan_tile_fold, 'CPTAC_200179_TEST')
    elif name == "GIGAPATH_SPECTRAL_CPTAC_200179_TRAIN":
        tilebagclip = dbx.quote(pancan_tile_fold, 'CPTAC_200179_TRAIN')
    elif name == "GIGAPATH_SPECTRAL_CPTAC_200179_CALIBRATE":
        tilebagclip = dbx.quote(pancan_tile_fold, 'CPTAC_200179_CALIBRATE')
    else:
        raise ValueError(f"Unknown spectral feature clip: {repr(name)}")
    return SpectralFeatureBagClip(
        root=root,
        spec=dict(
            extractor=extractor,
            tilebagclip=tilebagclip,
        ),
        n_workers=n_workers,
        devices=devices,
        gpu_batch_size=gpu_batch_size,
        cpu_batch_size=cpu_batch_size,
        gpu_parallelization=gpu_parallelization,
        cpu_parallelization=cpu_parallelization,
        bag_n_workers=bag_n_workers,
        bag_cpu_batch_size=bag_cpu_batch_size,
        bag_cpu_parallelization=bag_cpu_parallelization,
        tag=tag,
    )

