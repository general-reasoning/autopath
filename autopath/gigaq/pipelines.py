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


from autopath.models.vred import (
    VariationalReEncoderDecoder,
    VariationalReEncoderDecoderEvaluator,
    VariationalReEncoderDecoderLightning,
    VariationalReEncoderDecoderStill,
)

from autopath.models.hydro import (
    HydroDecoder,
    HydroDecoderLightning,
    HydroDecoderStill,
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

# git commit -am "gigaq: VRED"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred('GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG')"
# git commit -am "gigaq: VRED"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred('GIGAPATH_VRED_10HDN_1CLS_5CHN_LOG')"
def gigapath_vred(name, **kwargs):
    if name == "GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG":
        n_hidden_layers = 2
        n_classes = 100
        latent_gaussian_n_channels = 5
        latent_var_max = 1.0
        decoder_var_max = 1.0
        log_mixture_distributions = True
        log_latent_mixture_distributions = True
        tag = None
    elif name == "GIGAPATH_VRED_10HDN_1CLS_5CHN_LOG":
        n_hidden_layers = 10
        n_classes = 1
        latent_gaussian_n_channels = 5
        latent_var_max = kwargs.get('latent_var_max', 1.0)
        decoder_var_max = kwargs.get('decoder_var_max', 1.0)
        log_mixture_distributions = True
        log_latent_mixture_distributions = True
        tag = f"LVAR{latent_var_max}_VAR{decoder_var_max}"
    elif name == "GIGAPATH_VRED_10HDN_2CLS_5CHN_LOG":
        n_hidden_layers = 10
        n_classes = 2
        latent_gaussian_n_channels = 5
        latent_var_max = kwargs.get('latent_var_max', 1.0)
        decoder_var_max = kwargs.get('decoder_var_max', 1.0)
        log_mixture_distributions = True
        log_latent_mixture_distributions = True
        tag = f"LVAR{latent_var_max}_VAR{decoder_var_max}"
    else:
        raise ValueError(f"Unknown gigapath_vred: {name}")
    
    input_dim = 1536
    vred = VariationalReEncoderDecoder(spec=dict(
        classifier_input_dim=input_dim,
        classifier_n_hidden_layers=n_hidden_layers,
        classifier_n_classes=n_classes,
        latent_gaussian_n_classes=n_classes,
        latent_gaussian_n_hidden_layers=n_hidden_layers,
        latent_gaussian_n_channels=latent_gaussian_n_channels,
        latent_gaussian_var_max=latent_var_max,
        decoder_var_max=decoder_var_max,
        log_mixture_distributions=log_mixture_distributions,
        log_latent_mixture_distributions=log_latent_mixture_distributions,
        ),
        force_gc=kwargs.get('force_gc', False),
    )
    return vred, tag
    
# git commit -am "gigaq: VRED EVAL"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG_BASELINE_CPTAC_8020_TEST', batch_size=1).to('cuda').samples(1)"
# git commit -am "gigaq: VRED EVAL"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG_BASELINE_CPTAC_8020_TEST', batch_size=1).to('cuda').samples(2)"
# git commit -am "gigaq: VRED EVAL"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG_BASELINE_CPTAC_8020_TEST', batch_size=2).to('cuda').samples(1)"
# git commit -am "gigaq: VRED EVAL"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG_BASELINE_CPTAC_8020_TEST', batch_size=2).to('cuda').samples(2)"
#
# git commit -am "gigaq: VRED EVAL"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG_BASELINE_CPTAC_8020_TEST', batch_size=1).to('cuda').losses(1)"
# git commit -am "gigaq: VRED EVAL"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG_BASELINE_CPTAC_8020_TEST', batch_size=1).to('cuda').losses(2)"
# git commit -am "gigaq: VRED EVAL"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG_BASELINE_CPTAC_8020_TEST', batch_size=2).to('cuda').losses(1)"
# git commit -am "gigaq: VRED EVAL"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_evaluator('GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG_BASELINE_CPTAC_8020_TEST', batch_size=2).to('cuda').losses(2)"
def gigapath_vred_evaluator(vred_dataset_name, *, use_bags: bool = True, shuffle_bags_seed: int = 42, log_mixture_distributions: bool = False, **dataloader_kwargs):
    vredname, _clipname = vred_dataset_name.split('_BASELINE_CPTAC_')
    clipname = "GIGAPATH_BASELINE_CPTAC_" + _clipname
    vred, suffix = gigapath_vred(vredname, log_mixture_distributions=log_mixture_distributions)
    if use_bags:
        featureloader = gigapath_featurebag_dataloader_builder(clipname, shuffle_bags_seed=shuffle_bags_seed, **dataloader_kwargs)
    else:
        raise NotImplementedError(f"Shard dataloader_builder")
    vred_evaluator = VariationalReEncoderDecoderEvaluator(spec=dict(vred=vred, dataloader=featureloader))
    return vred_evaluator


"""
git commit -am "gigaq: VRED: STILL: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_still('GIGAPATH_VRED_2HDN_100CLS_5CHN_LVAR1_0_VAR1_0_LOG_MEDIUM_BASELINE_CPTAC_9802_TEST', dataroot='/tmp/dmitry/datalake', \
    n_devices=1, batch_size=6, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"

git commit -am "gigaq: VRED: STILL: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_still('GIGAPATH_VRED_10HDN_1CLS_5CHN_LOG_HIGHEST_BASELINE_CPTAC_9802_TEST', dataroot='/tmp/dmitry/datalake', \
    n_devices=1, batch_size=6, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"

git commit -am "gigaq: VRED: STILL: BUILD"; dbx.pprint "autopath.gigaq.pipelines.gigapath_vred_still('GIGAPATH_VRED_10HDN_2CLS_5CHN_LOG_HIGHEST_BASELINE_CPTAC_9802_TEST', dataroot='/tmp/dmitry/datalake', \
    n_devices=1, batch_size=6, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"

"""
def gigapath_vred_still(vred_dataset_name = None, 
                        *, 
                        dataroot: str = None, 
                        n_devices: int = 1,
                        force_gc: bool = False,
                        logs: str = None,
                        latent_var_max: float = 1.0,
                        decoder_var_max: float = 1.0,
                        **dataloader_kwargs,
    ):
    if vred_dataset_name is None:
        still = VariationalReEncoderDecoderStill
    else:
        vredname_precision, _clipname = vred_dataset_name.split('_BASELINE_CPTAC_')
        clipname = "GIGAPATH_BASELINE_CPTAC_" + _clipname
        bits = vredname_precision.split('_')
        precision = bits[-1].lower()
        vredname = '_'.join(bits[:-1])
        vred, suffix = gigapath_vred(vredname, force_gc=force_gc, latent_var_max=latent_var_max, decoder_var_max=decoder_var_max)
        tag = vred_dataset_name if suffix is None else f"{vred_dataset_name}_{suffix}"

        max_epochs=1
        max_steps=None
        ckpt_every_n_steps=100
        use_bags=True
        shuffle_bags_seed=42
        logs='/home/t-9dkarp/autopath/tensorboard/vred'
        learning_rate=1e-4
        scheduler='cosine'
        gradient_clip_algorithm='norm'
        gradient_clip_val=10.0
        skip_invalid_gradients: bool = True
        log_weights: bool = False
        log_gradients: bool = False
        ckpt: str = None
        
        if use_bags:
            featureloader_builder = dbx.quote(gigapath_featurebag_dataloader_builder, clipname, root=dataroot, shuffle_bags_seed=shuffle_bags_seed, **dataloader_kwargs)
        else:
            raise NotImplementedError(f"Shard dataloader_builder")
        lightning = VariationalReEncoderDecoderLightning( 
                        spec=dict(vred=vred, learning_rate=learning_rate, scheduler=scheduler)
        )

        still = VariationalReEncoderDecoderStill(spec=dict(
                    lightning=lightning, 
                    dataloader=featureloader_builder,
                    max_epochs=max_epochs,
                    max_steps=max_steps,
                    ckpt_every_n_steps=ckpt_every_n_steps,
                    skip_invalid_gradients=skip_invalid_gradients,
                    log_weights=log_weights,
                    log_gradients=log_gradients,
                    init_ckpt_path_or_anchor=ckpt,
                    gradient_clip_val=gradient_clip_val,
                    gradient_clip_algorithm=gradient_clip_algorithm,
                    precision=precision,
                ),
            n_devices=n_devices,
            logs=logs,
            tag=tag,
        )
    return still


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
def gigapath_bipolar_feature_bag_clip(name, *, root: str = None, n_workers: int = 0, build_missing_only: bool = False, n_devices: int = 1, gpu_batch_size: int = 16) -> BipolarFeatureBagClip:
    probe = gigapath_bipolar_feature_bags_probe(name, n_devices=n_devices, gpu_batch_size=gpu_batch_size)
    return BipolarFeatureBagClip(root=root, spec=dict(probe=dbx.quote(probe)), n_workers=n_workers, build_missing_only=build_missing_only)


# git commit -am "gigaq: BipolarFeaturebagDataset: TEST"; dbx.pprint "autopath.gigaq.pipelines.gigapath_bipolar_featurebag('GIGAPATH_BASELINE_CPTAC_8020')[0]"
def gigapath_bipolar_featurebag_dataset(name, *, root: str = None, shuffle_bags_seed: int = None) -> torch.utils.data.Dataset:
    featureclip = gigapath_bipolar_feature_bag_clip(name, root=root)
    dbx.Logger().debug(f"===================> {featureclip=}\nquoted featureclip {dbx.quote(featureclip)}")
    transform = dbx.quote(FeaturesToFloat, dtype='float32')
    target_transform = dbx.quote(FeaturesLabelTileToFloat, dtype='float32')
    return featurebag_dataset(dbx.quote(featureclip), transform=transform, target_transform=target_transform, bags_shuffle_seed=shuffle_bags_seed)


def gigapath_bipolar_featurebag_dataloader_builder(name, root: str = None, shuffle_bags_seed: int = None, **dataloader_kwargs):
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
def gigapath_hydro(name, **kwargs):
    """Create a HydroDecoder configuration by name.
    
    Args:
        name: Configuration name (e.g., 'GIGAPATH_HYDRO_DEFAULT')
        **kwargs: Override default parameters
        
    Returns:
        HydroDecoder
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
    
    hydro = HydroDecoder(spec=dict(
        latent_dim=latent_dim,
        image_size=image_size,
        initial_size=initial_size,
        hidden_channels=hidden_channels,
        use_batch_norm=kwargs.get('use_batch_norm', True),
        loss_type=kwargs.get('loss_type', 'mse'),
    ))
    return hydro


"""
git commit -am "gigaq: HYDRO: STILL: BUILD"; DBXWRKREPO=True dbx.pprint "autopath.gigaq.pipelines.gigapath_hydro_still('GIGAPATH_HYDRO_DEFAULT_MEDIUM_BASELINE_CPTAC_400159', \\
    log_images=True, n_devices=1, batch_size=6, num_workers=2, prefetch_factor=1, pin_memory=True).set(capture_output=True).build()"
"""
def gigapath_hydro_still(hydro_dataset_name = None, 
                        *,  
                        n_devices: int = 1,
                        logsroot: str = None,
                        log_images: bool = False,
                        log_image_interval: int = 100,
                        from_scratch: bool = False,
                        **dataloader_kwargs,
    ):
    """Create a HydroDecoderStill training pipeline.
    
    Args:
        hydro_dataset_name: Name in format 'GIGAPATH_HYDRO_<config>_<precision>_BASELINE_CPTAC_<split>'
        dataroot: Root directory for data
        n_devices: Number of GPU devices
        logsroot: Root directory for tensorboard logs
        log_images: Whether to log images during training
        log_image_interval: Interval (in steps) to log images
        from_scratch: Whether to restart training from scratch
        **dataloader_kwargs: DataLoader parameters (batch_size, num_workers, etc.)
        
    Returns:
        HydroDecoderStill instance
    """
    if hydro_dataset_name is None:
        still = HydroDecoderStill
    else:
        hydroname_precision, _splitname = hydro_dataset_name.split('_BASELINE_CPTAC_')
        splitname = "GIGAPATH_BASELINE_CPTAC_" + _splitname
        bits = hydroname_precision.split('_')
        precision = bits[-1].lower()
        hydroname = '_'.join(bits[:-1])
        hydro = gigapath_hydro(hydroname)
        tag = hydro_dataset_name
        logsroot = logsroot or '/home/t-9dkarp/autopath/tensorboard/hydro'

        max_epochs = 1
        max_steps = None
        ckpt_every_n_steps = 100
        use_bags = True
        shuffle_bags_seed = 42
        learning_rate = 1e-4
        scheduler = 'cosine'
        gradient_clip_algorithm = 'norm'
        gradient_clip_val = 10.0
        
        if use_bags:
            featureloader_builder = dbx.quote(gigapath_bipolar_featurebag_dataloader_builder, splitname, shuffle_bags_seed=shuffle_bags_seed, **dataloader_kwargs)
        else:
            raise NotImplementedError(f"Shard dataloader_builder")
        
        lightning = HydroDecoderLightning(
            spec=dict(hydro=hydro, learning_rate=learning_rate, scheduler=scheduler, 
                     log_images=log_images, log_image_interval=log_image_interval)
        )

        still = HydroDecoderStill(spec=dict(
                    lightning=lightning, 
                    dataloader=featureloader_builder,
                    max_epochs=max_epochs,
                    max_steps=max_steps,
                    ckpt_every_n_steps=ckpt_every_n_steps,
                    gradient_clip_val=gradient_clip_val,
                    gradient_clip_algorithm=gradient_clip_algorithm,
                    precision=precision,
                    from_scratch=from_scratch,
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

