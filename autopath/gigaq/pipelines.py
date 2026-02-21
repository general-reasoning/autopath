import os
from typing import Optional, List

from tqdm import tqdm

import torch
import torch.multiprocessing as mp

import dbx

from autopath.databits import ClipDatasetBuilder, ClipDataLoaderBuilder

from autopath.pancan.pipelines import (
    pancan_tile_bag,
    pancan_tile_bag_clip,
    pancan_tile_bag_fold,
)

from autopath.features import (
    FeatureBag, 
    FeatureBagClip,
    featurebag_dataset,
    FeaturesToFloat,
    FeaturesLabelTileToFloat,
    BipolarFeatureBagClip,
    BipolarSingleFeatureBagClip,
)

from autopath.pancan.probes import (
    LogisticFeatureBagProbe, 
    FeatureBagMedianProbe,
    BipolarFeatureBagProbe,
    #
    FeaturePairwiseDistances,
    FeatureSortedDistances,
    Feature2NNDistances,
    Feature2NNDim,
)

from autopath.gigaq.dinov2.backbone import (
    BackboneEvaluator, 
    SidebandBackboneEvaluator, 
    GIGAPATH_BACKBONE_DEPTH,
)



from autopath.models.hydro import (
    Hydro,
    HydroLightning,
    HydroStill,
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
    else:
        raise ValueError(f"Unknown backbone evaluator: {name}")

# git commit -am "gigaq: FeatureBag: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag('GIGAPATH_BASELINE_CPTAC_SAMPLE').set(device='cuda', gpu_batch_size=1024).build()"
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

# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC', n_devices=2, n_workers=16).set(gpu_batch_size=1024).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_9802_TEST', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_9802_TRAIN', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_404020_CALIBRATE', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_404020_TRAIN', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_404020_TEST', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159_CALIBRATE', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159_TRAIN', n_workers=16).build()"
# git commit -am "gigaq: FeatureBagClip: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159_TEST', n_workers=16).build()"

def gigapath_feature_bag_clip(name:str = None, *, root:str = None, n_workers: int = 1, n_devices: int = 1, gpu_batch_size: int = 1024) -> FeatureBagClip:
    devices = [f'cuda:{i}' for i in range(n_devices)]
    if name is None:
        return FeatureBagClip
    if name == "GIGAPATH_BASELINE_CPTAC":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_clip, 'CPTAC')
    elif name == "GIGAPATH_BASELINE_CPTAC_9802_TEST":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_9802_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_9802_TRAIN":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_9802_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TEST":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_8020_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_8020_TRAIN":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_8020_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_404020_CALIBRATE":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_404020_CALIBRATE')
    elif name == "GIGAPATH_BASELINE_CPTAC_404020_TRAIN":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_404020_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_404020_TEST":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_404020_TEST')
    elif name == "GIGAPATH_BASELINE_CPTAC_400159_CALIBRATE":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_400159_CALIBRATE')
    elif name == "GIGAPATH_BASELINE_CPTAC_400159_TRAIN":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_400159_TRAIN')
    elif name == "GIGAPATH_BASELINE_CPTAC_400159_TEST":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_EVALUATOR')
        tilebagclip=dbx.quote(pancan_tile_bag_fold, 'CPTAC_400159_TEST')
    elif name == "GIGAPATH_BASELINE_5B_CPTAC_8020_TEST":
        extractor=quote_extractor('GIGAPATH_BASELINE_BACKBONE_5B_EVALUATOR')
        tilebagclip = dbx.quote(pancan_tile_bag_fold, 'CPTAC_8020_TEST')
    else:
        raise ValueError(f"Unknown gigapath_feature_clip: {repr(name)}")
    return FeatureBagClip(
        root=root, 
        spec=dict(extractor=extractor, tilebagclip=tilebagclip), 
        n_workers=n_workers,
        devices=devices, 
        gpu_batch_size=gpu_batch_size,
    )


# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC')[0]"
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_8020_TEST')[0]"
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_9802_TEST')[0]"
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159_CALIBRATE')[0]"
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159_TRAIN')[0]"
# git commit -am "gigaq: featurebag_dataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159_TEST')[0]"
def gigapath_featurebag_dataset(name, *, root: str = None, shuffle_bags_seed: int = None) -> torch.utils.data.Dataset:
    featureclip = gigapath_feature_bag_clip(name, root=root)
    dbx.Logger().debug(f"===================> {featureclip=}\nquoted featureclip={dbx.quote(featureclip)}")
    return featurebag_dataset(dbx.quote(featureclip), bags_shuffle_seed=shuffle_bags_seed)


def gigapath_featurebag_dataloader_builder(name, root: str = None, shuffle_bags_seed: int = None, **dataloader_kwargs):
    featureclip = gigapath_feature_bag_clip(name, root=root)
    clip_dataset_builder = ClipDatasetBuilder(spec=dict(clip=dbx.quote(featureclip), shuffle_seed=shuffle_bags_seed))
    return ClipDataLoaderBuilder(spec=dict(
                            clip_dataset_builder=clip_dataset_builder,
                            batch_size=dataloader_kwargs.get('batch_size', None),
                            shuffle=dataloader_kwargs.get('shuffle', False),
                          ),  
                          dataloader_kwargs=dataloader_kwargs,
    )


# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TEST', 10, batch_size=1, num_workers=1)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 10, batch_size=1, num_workers=1)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 20, batch_size=1, num_workers=2)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=1)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=2)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=1, prefetch_factor=1)" 2.71s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=4, prefetch_factor=1)" 2.61s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=2, prefetch_factor=None)" 2.58s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 100, batch_size=1, num_workers=2, prefetch_factor=1)" 2.47s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=8, num_workers=1, prefetch_factor=2)" 2.47s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=8, num_workers=1, prefetch_factor=1,)" 2.38s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=4, prefetch_factor=1,)" 2.28s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=1, prefetch_factor=1,)" 2.25s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=4, prefetch_factor=None)" 1.39s/it
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=8, prefetch_factor=1)" ~=
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=2, prefetch_factor=2)" ~=
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=4, num_workers=2, prefetch_factor=1)" ~=
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020_TRAIN', 400, batch_size=8, num_workers=2, prefetch_factor=1)" ~=
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159_CALIBRATE', 100, batch_size=8, num_workers=2, prefetch_factor=1)"
# git commit -am "gigaq: FeaturebagDataloader: SAMPLES"; dbx.pprint "autopath.gigaq.pipelines.gigapath_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159_TRAIN', 100, batch_size=8, num_workers=2, prefetch_factor=1)"
def gigapath_featurebag_dataloader_samples(name, n, root: str = None, shuffle_bags: bool = False, return_last: bool = True, **dataloader_kwargs):
    batch_size = dataloader_kwargs.get('batch_size', None)
    dataloader_builder = gigapath_featurebag_dataloader_builder(name, root=root, shuffle=shuffle_bags, **dataloader_kwargs)
    progress = tqdm(total=n)
    for i, _ in enumerate(dataloader_builder.dataloader()):
        progress.update(batch_size if batch_size is not None else 1)
        if i*batch_size >= n-1:
            break
    if return_last:
        return _



# git commit -am "gigaq: FeatureBagMedianProbe: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bags_median_probe('GIGAPATH_BASELINE_CPTAC_8020_TRAIN').build()"
# git commit -am "gigaq: FeatureBagMedianProbe: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_bags_median_probe('GIGAPATH_BASELINE_CPTAC_400159_CALIBRATE').build()"
def gigapath_feature_bags_median_probe(name, n_devices: int = 1, gpu_batch_size: int = 16) -> FeatureBagMedianProbe:
    return FeatureBagMedianProbe(spec=dict(featurebagclip=gigapath_feature_bag_clip(name, n_devices=n_devices, gpu_batch_size=gpu_batch_size),))


# git commit -am "gigaq: BipolarFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_400159').build()"
# git commit -am "gigaq: BipolarFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_400159', n_devices=1).build_tree()"
def gigapath_bipolar_feature_bags_probe(name, *, n_devices: int = 1, gpu_batch_size: int = 16) -> BipolarFeatureBagProbe:
    return BipolarFeatureBagProbe(
        spec=dict(featurebagclip=gigapath_feature_bag_clip(f"{name}_TRAIN", n_devices=n_devices, gpu_batch_size=gpu_batch_size), 
                  medianprobe=gigapath_feature_bags_median_probe(f"{name}_CALIBRATE", n_devices=n_devices, gpu_batch_size=gpu_batch_size),
        ),
        devices=[f'cuda:{i}' for i in range(n_devices)],
    )
 
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_8020', n_workers=4).build()"
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_8020').build()"
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159').build()"
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159', single=0).build()"
# git commit -am "gigaq: BipolarFeatureBagClip: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_feature_bag_clip('GIGAPATH_BASELINE_CPTAC_400159', single=1).build()"
def gigapath_bipolar_feature_bag_clip(name, *, root: str = None, n_workers: int = 0, build_missing_only: bool = False, n_devices: int = 1, gpu_batch_size: int = 16, single: int|None = None) -> BipolarFeatureBagClip:
    probe = gigapath_bipolar_feature_bags_probe(name, n_devices=n_devices, gpu_batch_size=gpu_batch_size)
    if single is not None:
        dbx.Logger(name='gigapath_bipolar_feature_bag_clip').debug(f"===================> single: {repr(single)}\nname: {repr(name)}")
        clip = BipolarSingleFeatureBagClip(root=root, spec=dict(probe=dbx.quote(probe), idx=single))
    else:
        clip = BipolarFeatureBagClip(root=root, spec=dict(probe=dbx.quote(probe)), n_workers=n_workers, build_missing_only=build_missing_only)
    return clip


# git commit -am "gigaq: BipolarFeaturebagDataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_8020')[0]"
# git commit -am "gigaq: BipolarFeaturebagDataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159')[0]"
# git commit -am "gigaq: BipolarFeaturebagDataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159_SINGLE_0')[0]"
# git commit -am "gigaq: BipolarFeaturebagDataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_featurebag_dataset('GIGAPATH_BASELINE_CPTAC_400159_SINGLE_1')[0]"
def gigapath_bipolar_featurebag_dataset(name, *, root: str = None, shuffle_bags_seed: int = None) -> torch.utils.data.Dataset:
    if 'SINGLE' in name:
        single = int(name.split('_')[-1])
        name = name.split('_SINGLE')[0]
        dbx.Logger(name='gigapath_bipolar_featurebag_dataset').debug(f"===================> single: {repr(single)}\nname: {repr(name)}")
        featureclip = gigapath_bipolar_feature_bag_clip(name, root=root, single=single)
    else:
        featureclip = gigapath_bipolar_feature_bag_clip(name, root=root)
    dbx.Logger(name='gigapath_bipolar_featurebag_dataset').debug(f"===================> {featureclip=}\nquoted featureclip {dbx.quote(featureclip)}")
    transform = dbx.quote(FeaturesToFloat, dtype='float32')
    target_transform = dbx.quote(FeaturesLabelTileToFloat, dtype='float32')
    return featurebag_dataset(dbx.quote(featureclip), transform=transform, target_transform=target_transform, bags_shuffle_seed=shuffle_bags_seed)


def gigapath_bipolar_featurebag_dataloader_builder(name, root: str = None, shuffle_bags_seed: int = None, **dataloader_kwargs):
    if 'SINGLE' in name:
        single = int(name.split('_')[-1])
        name = name.split('_SINGLE')[0]
        featureclip = gigapath_bipolar_feature_bag_clip(name, root=root, single=single)
    else:
        featureclip = gigapath_bipolar_feature_bag_clip(name, root=root)
    transform = dbx.quote(FeaturesToFloat, dtype='float32')
    target_transform = dbx.quote(FeaturesLabelTileToFloat, dtype='float32')
    clip_dataset_builder = ClipDatasetBuilder(spec=dict(clip=dbx.quote(featureclip), transform=transform, target_transform=target_transform, shuffle_seed=shuffle_bags_seed))
    return ClipDataLoaderBuilder(spec=dict(
                            clip_dataset_builder=clip_dataset_builder,
                            batch_size=dataloader_kwargs.get('batch_size', None),
                            shuffle=dataloader_kwargs.get('shuffle', False),
                          ),  
                          dataloader_kwargs=dataloader_kwargs,
    )
 

# git commit -am "gigaq: BipolarFeaturebagDataloader: SAMPLES"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_8020', 1000)"
# git commit -am "gigaq: BipolarFeaturebagDataloader: SAMPLES"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159', 1000)"
# git commit -am "gigaq: BipolarFeaturebagDataloader: SAMPLES"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159_SINGLE_0', 1000)"
# git commit -am "gigaq: BipolarFeaturebagDataloader: SAMPLES"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_featurebag_dataloader_samples('GIGAPATH_BASELINE_CPTAC_400159_SINGLE_1', 1000)"
def gigapath_bipolar_featurebag_dataloader_samples(name, n, root: str = None, shuffle_bags: bool = False, return_last: bool = True, **dataloader_kwargs):
    batch_size = dataloader_kwargs.get('batch_size', 1)
    dataloader_builder = gigapath_bipolar_featurebag_dataloader_builder(name, root=root, shuffle=shuffle_bags, **dataloader_kwargs)
    progress = tqdm(total=n, desc=f"READING SAMPLES")
    for i, _ in enumerate(dataloader_builder.dataloader()):
        progress.update(batch_size if batch_size is not None else 1)
        if i*batch_size >= n-1:
            break
    if return_last:
        return _


# git commit -am "gigaq: HYDRO"; dbx.pprint "autopath.gigaq.pipelines.gigapath_hydro('GIGAPATH_HYDRO_DEFAULT')"
def gigapath_hydro(name, *, model: int = 0, **kwargs):
    """Create a Hydro configuration by name.
    
    Args:
        name: Configuration name (e.g., 'GIGAPATH_HYDRO_DEFAULT')
        model: Model architecture identifier (e.g., 0 selects Model_0)
        **kwargs: Override default parameters
        
    Returns:
        Hydro
    """
    if name == "GIGAPATH_HYDRO_DEFAULT":
        latent_dim = kwargs.get('latent_dim', 1536)
        image_size = kwargs.get('image_size', 256)
        initial_size = kwargs.get('initial_size', 8)
        hidden_channels = kwargs.get('hidden_channels', 256)
    elif name == "GIGAPATH_HYDRO_SMALL":
        latent_dim = kwargs.get('latent_dim', 1536)
        image_size = kwargs.get('image_size', 256)
        initial_size = kwargs.get('initial_size', 8)
        hidden_channels = kwargs.get('hidden_channels', 128)
    elif name == "GIGAPATH_HYDRO_LARGE":
        latent_dim = kwargs.get('latent_dim', 1536)
        image_size = kwargs.get('image_size', 256)
        initial_size = kwargs.get('initial_size', 8)
        hidden_channels = kwargs.get('hidden_channels', 512)
    else:
        raise ValueError(f"Unknown gigapath_hydro: {name}")
    
    hydro = Hydro(spec=dict(
        model=model,
        latent_dim=latent_dim,
        image_size=image_size,
        initial_size=initial_size,
        hidden_channels=hidden_channels,
        use_batch_norm=kwargs.get('use_batch_norm', True),
        use_bilinear_upsampling=kwargs.get('use_bilinear_upsampling', True),
        use_residual_upsampling=kwargs.get('use_residual_upsampling', False),
        use_spatial_attention_gates=kwargs.get('use_spatial_attention_gates', False),
        use_feature_modulation=kwargs.get('use_feature_modulation', False),
        loss_type=kwargs.get('loss_type', 'mse'),
    ))
    return hydro


"""
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_hydro_still('GIGAPATH_HYDRO_DEFAULT_MEDIUM_BASELINE_CPTAC_400159', \\
    log_images=True, n_devices=1, batch_size=6, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_hydro_still('GIGAPATH_HYDRO_DEFAULT_MEDIUM_BASELINE_CPTAC_400159_SINGLE_0', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=64, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_hydro_still('GIGAPATH_HYDRO_DEFAULT_MEDIUM_BASELINE_CPTAC_400159_SINGLE_1', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=64, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_hydro_still('GIGAPATH_HYDRO_DEFAULT_HIGHEST_BASELINE_CPTAC_400159_SINGLE_0', \\
    loss_type='l1', log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_hydro_still('GIGAPATH_HYDRO_DEFAULT_HIGHEST_BASELINE_CPTAC_400159_SINGLE_0', \\
    loss_type='l1', use_residual_upsampling=True, use_spatial_attention_gates=False, use_feature_modulation=False, suffix='resup_nosag_nofmod_l1', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
#
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBXWRKREPO=True \\
    dbx.pprint "autopath.gigaq.pipelines.gigapath_hydro_still('GIGAPATH_HYDRO_DEFAULT_HIGHEST_BASELINE_CPTAC_400159_SINGLE_0', \\
    loss_type='l1', use_residual_upsampling=True, use_spatial_attention_gates=True, use_feature_modulation=True, suffix='resup_sag_fmod_l1', \\
    log_images=True, log_images_interval=1, max_epochs=100, n_devices=1, batch_size=16, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"

"""
def gigapath_hydro_still(hydro_dataset_name = None, 
                        *,  
                        n_devices: int = 1,
                        logsroot: str = None,
                        log_images: bool = False,
                        log_images_interval: int = 100,
                        from_scratch: bool = False,
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
                        use_batch_norm: bool = True,
                        use_bilinear_upsampling: bool = False,
                        use_residual_upsampling: bool = False,
                        use_spatial_attention_gates: bool = False,
                        use_feature_modulation: bool = False,
                        model: int = 0,
                        suffix: str|None = None,
                        **dataloader_kwargs,
    ):
    """Create a HydroStill training pipeline.
    
    Args:
        hydro_dataset_name: Name in format 'GIGAPATH_HYDRO_<config>_<precision>_BASELINE_CPTAC_<split>'
        dataroot: Root directory for data
        n_devices: Number of GPU devices
        logsroot: Root directory for tensorboard logs
        log_images: Whether to log images during training
        log_images_interval: Interval (in steps) to log images
        from_scratch: Whether to restart training from scratch
        model: Model architecture identifier (e.g., 0 selects Model_0)
        **dataloader_kwargs: DataLoader parameters (batch_size, num_workers, etc.)
        
    Returns:
        HydroStill instance
    """
    if hydro_dataset_name is None:
        still = HydroStill
    else:
        hydroname_precision, _splitname = hydro_dataset_name.split('_BASELINE_CPTAC_')
        splitname = "GIGAPATH_BASELINE_CPTAC_" + _splitname
        bits = hydroname_precision.split('_')
        precision = bits[-1].lower()
        hydroname = '_'.join(bits[:-1])
        hydro = gigapath_hydro(hydroname, 
                             model=model,
                             loss_type=loss_type, 
                             use_batch_norm=use_batch_norm, 
                             use_bilinear_upsampling=use_bilinear_upsampling,
                             use_residual_upsampling=use_residual_upsampling,
                             use_spatial_attention_gates=use_spatial_attention_gates,
                             use_feature_modulation=use_feature_modulation)
        tag = f"{hydro_dataset_name}/{suffix}" if suffix is not None else f"{hydro_dataset_name}"

        logsroot = logsroot or f'{os.environ["HOME"]}/autopath/tensorboard/hydro'
        
        if use_bags:
            featureloader_builder = dbx.quote(gigapath_bipolar_featurebag_dataloader_builder, splitname, shuffle_bags_seed=shuffle_bags_seed, **dataloader_kwargs)
        else:
            raise NotImplementedError(f"Shard dataloader_builder")
        
        lightning = HydroLightning(
            spec=dict(hydro=hydro, learning_rate=learning_rate, scheduler=scheduler, 
                     log_images=log_images, log_images_interval=log_images_interval,
                     use_residual_upsampling=use_residual_upsampling,
                     use_spatial_attention_gates=use_spatial_attention_gates,
                     use_feature_modulation=use_feature_modulation)
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
                    use_residual_upsampling=use_residual_upsampling,
                    use_spatial_attention_gates=use_spatial_attention_gates,
                    use_feature_modulation=use_feature_modulation,
                ),
            n_devices=n_devices,
            logsroot=logsroot,
            tag=tag,
        )
    return still
    

# git commit -am "gigaq: LogisticFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_bins=2).build()"
# git commit -am "gigaq: LogisticFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_8020_TEST', n_bins=2).read('evaluation_reports')"
#
# git commit -am "gigaq: LogisticFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_9802_TRAIN', n_bins=2, polarize=True).build()"
#
# git commit -am "gigaq: LogisticFeatureBagProbe: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_logistic_feature_bags_probe('GIGAPATH_BASELINE_CPTAC_9802_TRAIN', n_bins=2, aggregation='cdf').build()"

def gigapath_logistic_feature_bags_probe(name, n_bins: int = 2, polarize: bool = False, aggregation: str = 'mean') -> LogisticFeatureBagProbe:
    return LogisticFeatureBagProbe(spec=dict(featurebagclip=gigapath_feature_bag_clip(name), n_bins=n_bins, polarize=polarize, aggregation=aggregation))
    

# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000').set(n_devices=3).build()"
# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000').set(n_devices=3).build()"
# git commit -am "gigaq: FeaturePairwiseDistances: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_pairwise_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_20_4000_20000').set(n_devices=3).build()"
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


# git commit -am "gigaq: FeatureSortedDistances: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_sorted_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000',).set(n_workers=3,).build()"
# git commit -am "gigaq: FeatureSortedDistances: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_sorted_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000',).set(n_workers=3,).build()"
def gigapath_feature_sorted_distances(name) -> FeatureSortedDistances:
    return FeatureSortedDistances(
                    spec=dict(features_pairwise_distances=dbx.quote(gigapath_feature_pairwise_distances, name),)
    )
    

# git commit -am "gigaq: Features=2NNDistances: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_2nn_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000',).set(n_workers=3).build()"
# git commit -am "gigaq: Feature2NNDistances: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_2nn_distances('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000',).set(n_workers=3).build()"
def gigapath_feature_2nn_distances(name) -> Feature2NNDistances:
    return Feature2NNDistances(
                    spec=dict(features_sorted_distances=dbx.quote(gigapath_feature_sorted_distances, name),),
        )
    

# git commit -am "gigaq: Feature2NNDim: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_2nn_dim('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_4000',).build_tree().dim"
# git commit -am "gigaq: Feature2NNDim: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_feature_2nn_dim('GIGAPATH_BASELINE_CPTAC_8020_TEST_10_4000_20000',).build_tree().dim"
def gigapath_feature_2nn_dim(name) -> Feature2NNDim:
    return Feature2NNDim(
                    spec=dict(features_2nn_distances=dbx.quote(gigapath_feature_2nn_distances, name),),
        )

