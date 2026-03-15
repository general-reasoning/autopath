
from dataclasses import dataclass
import datetime
import functools
import gc
import itertools
import math
from typing import Optional, Union, Tuple

import tqdm


import numpy as np
import pandas as pd
import torch


from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression

import dbx
from dbx import (
	Logger,
	Datablock,
    write_tensor, 
    read_tensor,
    write_npz,
    read_npz,
    write_pickle,
    read_pickle,
    TorchMultithreadingDatablocksBuilder,
    TorchMultiprocessingDatablocksBuilder,
    MultithreadingCallableExecutor,
    MultiprocessingDatablocksBuilder,
    RayDatablocksBuilder,
)

from autopath.databits import Bag, Clip

from autopath import tools
from autopath.features import FeatureBag, FeatureBagClip


class FeatureBagProber:
    def __init__(self, log = Logger()):
        self.log = log

    @staticmethod
    def ndarray(X: Union[np.ndarray, list, torch.Tensor, pd.DataFrame]):
        if isinstance(X, list):
            X = np.array(X)
        elif isinstance(X, torch.Tensor):
            X = X.numpy()
        elif isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise ValueError(f"Unknown ndarray input type: {type(X)}")
        return X

    @staticmethod
    def discretize_features(X: Union[np.ndarray, list, torch.Tensor, pd.DataFrame], d:int = 4) -> np.ndarray:
        features = FeatureBagProber.ndarray(X)
        return pd.DataFrame(features).apply(lambda c: pd.qcut(c, d, labels=False, duplicates='drop')).fillna(0.0).values
    
    @staticmethod
    def polarize_features(X: Union[np.ndarray, list, torch.Tensor, pd.DataFrame]) -> np.ndarray:
        features = FeatureBagProber.ndarray(X)
        return (2.0*pd.DataFrame(features).apply(lambda c: pd.qcut(c, 2, labels=False, duplicates='drop')) - 1.0).fillna(0.0).values

    @staticmethod
    def split_train_test(X: Union[np.ndarray, list, torch.Tensor, pd.DataFrame], 
                         y: Optional[Union[np.ndarray, list, torch.Tensor, pd.DataFrame]] = None, 
                         train_fraction:float=0.8
    ):
        #TODO: split_slides_labels_train_test() -> split_features_labels_train_test()
        X = FeatureBagProber.ndarray(X)
        if y is not None:
            y = FeatureBagProber.ndarray(y)
        N = y.shape[0]
        permutation = permutation = np.random.permutation(range(N))
        n = int(math.floor(N*train_fraction))
        train = list(permutation[:n])
        test = list(permutation[n:])
        if y is not None:
            return (X[train, :], y[train]), (X[test, :], y[test])
        else:
            return X[train, :], X[test, :]
        
    @staticmethod
    def plot_features_umap(Xy,
                           *, 
                           fraction = 0.01, 
                           title="", 
                           label="",
                           use_umap_plot=False,
                           output_path=None,
                           log: Logger = Logger(),
    ):
        features, labels = Xy
        import umap
        import umap.plot
        from matplotlib import pyplot as plt
        # UMAP visualization of original features (1% subsample)
        um = umap.UMAP(n_components=2, random_state=42)
        umap_features = um.fit_transform(features)
        ulabels = set(labels)
        ilablesmap = {l: i for i, l in enumerate(ulabels)}
        ilabels = [ilablesmap[l] for l in labels]
        plt.scatter(umap_features[:, 0], umap_features[:, 1], c=ilabels, label=labels)

        if output_path is not None:
            plt.savefig(output_path)
            log.verbose(f"Wrote figure to '{output_path}'")
        else:
            plt.show()
        return umap_features

    @dataclass
    class HammingDistanceStats:
        means: np.ndarray
        mins: np.ndarray
        maxs: np.ndarray
        stds: np.ndarray
        distances: np.ndarray
        unique_labels: np.ndarray

    def pairwise_hamming_distance_stats(self, features: np.ndarray, labels: np.ndarray):
        self.log.verbose(f"COMPUTING pairwise Hamming distances: BEGIN")
        features = torch.tensor(features).to(torch.float32)
        distances = 0.5*torch.cdist(features, features, p=1).numpy()
        self.log.debug(f"distances.shape: {distances.shape}")
        self.log.verbose(f"COMPUTING pairwise Hamming distances: END")
        unique_labels = np.unique(labels)
        n = len(unique_labels)
        means = np.zeros((n, n))
        mins = np.zeros((n, n))
        maxs = np.zeros((n, n))
        stds = np.zeros((n, n))
        ulabels2 = list(itertools.product(enumerate(unique_labels), enumerate(unique_labels)))
        self.log.verbose(f"COMPUTING statistics of label distances: BEGIN")
        if self.log.ist('verbose'):
            ulabels2 = tqdm.tqdm(ulabels2)
        for il1, il2 in ulabels2:
            i1, l1 = il1
            i2, l2 = il2
            if i1 > i2:
                continue
            sel1 = (labels == l1)
            sel2 = (labels == l2)
            n1 = sel1.sum()
            n2 = sel2.sum()
            if n1 == 0 or n2 == 0:
                continue
            dist12 = distances[sel1, :][:, sel2]
            if i1 == i2:  
                means[i1, i2] = (dist12.sum())/(n1*(n1-1))
                means[i2, i1] = means[i1, i2]
            else: 
                mean12 = dist12.mean()
                means[i1, i2] = mean12
                means[i2, i1] = mean12
            mins[i1, i2] = dist12.min()
            mins[i2, i1] = mins[i1, i2]
            maxs[i1, i2] = dist12.max()
            maxs[i2, i1] = maxs[i1, i2]
            stds[i1, i2] = dist12.std()
            stds[i2, i1] = stds[i1, i2]
        self.log.verbose(f"COMPUTING statistics of label distances: BEGIN")          

        return self.HammingDistanceStats(
            means=means,
            mins=mins,
            maxs=maxs,
            stds=stds,
            distances=distances,
            unique_labels=unique_labels,
        )



class LogisticFeatureBagProber(FeatureBagProber):
    @staticmethod
    def evaluate_features(Xy: Tuple[Union[np.ndarray, list, torch.Tensor, pd.DataFrame], 
                                    Union[np.ndarray, list, torch.Tensor, pd.DataFrame],], 
                          *, 
                          fraction=0.8
    ):
        features, labels = Xy
        features = LogisticFeatureBagProber.ndarray(features)
        labels = LogisticFeatureBagProber.ndarray(labels)
        N = len(labels)
        ntrain = int(N*fraction)
        randomidx = np.random.permutation(list(range(N)))
        trainidx = randomidx[:ntrain]
        testidx = randomidx[ntrain:]
       
        X_train = features[trainidx, :]
        y_train = labels[trainidx]

        X_test = features[testidx, :]
        y_test = labels[testidx]

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report

    @staticmethod
    def evaluate_features2(Xy1,
                           Xy2,
                           *, 
                           fraction=0.8,
                           tags=("(1)","(2)"),
                           log: Logger = Logger(),
    ):
        label1, label2 = tags
        log.verbose(f"EVALUATING features: {label1}: started at {datetime.datetime.now()}")
        report1 = LogisticFeatureBagProber.evaluate_features(Xy1, fraction=fraction)
        log.verbose(f"EVALUATING features: {label1}: finished at {datetime.datetime.now()}")
        log.verbose(f"EVALUATING features: {label2}: started at {datetime.datetime.now()}")
        report2 = LogisticFeatureBagProber.evaluate_features(Xy2, fraction=fraction)
        log.verbose(f"EVALUATING features: {label2}: finished at {datetime.datetime.now()}")

        rstr = f"---------- {label1} ------------\n{report1}\n---------- {label2} ------------\n{report2}"
        log.verbose(rstr)
        return report1, report2


class LogisticFeatureBagProbe(Datablock, LogisticFeatureBagProber):
    TOPICFILES = {
        'bag_labels': 'bag_labels.npz',
        'bag_features': 'bag_features.npy',
        'discretized_bag_features': 'discretized_bag_features.npy',
        'evaluation_reports': 'evaluation_reports.pkl',
    }
    @dataclass
    class CONFIG:
        featurebagclip: FeatureBagClip
        n_bins: int = 2
        polarize: bool = False
        evaluation_fraction: float = 0.8
        aggregation: str = "mean"

    def __post_init__(self):
        assert self.cfg.aggregation in ["mean",], f"Unknown aggregation: {self.cfg.aggregation}"
        return self

    def __build__(self):
        bag_labels = []
        bag_feature_list = []
        self.log.verbose(f"READING featurebags and bag names")
        if self.verbose:
            bagitor = tqdm.tqdm(self.cfg.featurebagclip.bags)
        else:
            bagitor = self.cfg.featurebagclip.bags
        for featurebag in bagitor:
            bag_labels.append(featurebag.cfg.tilebag.label)
            if self.cfg.aggregation == "mean":
                _bag_features = torch.mean(featurebag.features, dim=0)
            bag_feature_list.append(_bag_features)
        bag_features = torch.stack(bag_feature_list)
        assert len(bag_labels) == len(bag_features), f"len(bag_labels) != len(bag_features): {len(bag_labels)} != {len(bag_features)}"
        write_npz(self.path('bag_labels', ensure_dirpath=True), bag_labels=bag_labels)
        write_tensor(bag_features, self.path('bag_features', ensure_dirpath=True))

        # discretized_features
        prober = FeatureBagProber()
        if self.cfg.polarize:
            assert self.cfg.n_bins == 2, f"Polarizing features with n_bins != 2: {self.cfg.n_bins}"
            self.log.verbose(f"POLARIZING features")
            discretized_bag_features = torch.tensor(prober.polarize_features(bag_features.numpy()))
        else:
            discretized_bag_features = torch.Tensor(prober.discretize_features(bag_features.numpy(), self.cfg.n_bins))
        write_tensor(discretized_bag_features, self.path('discretized_bag_features', ensure_dirpath=True))
      
        continuous, discretized = self.evaluate_features2(
            (bag_features, bag_labels), 
            (discretized_bag_features, bag_labels),
            fraction=self.cfg.evaluation_fraction,
            tags=('(continuous)', '(discretized)',),
            log=self.log,
        )
        evaluation_reports = {'continuous': continuous, 'discretized': discretized}
        write_pickle(evaluation_reports, self.path('evaluation_reports', ensure_dirpath=True))
        return self

    def __read__(self, topic):
        if topic == 'bag_labels':
            result = read_npz(self.path('bag_labels'), 'labels')
        elif topic == 'bag_features':
            result = read_tensor(self.path('bag_features'))
        elif topic == 'discretized_bag_features':
            result = read_tensor(self.path('discretized_bag_features'))
        elif topic == 'evaluation_reports':
            result = read_pickle(self.path('evaluation_reports'))
        else:
            raise ValueError(f"Unknown topic: {topic}")
        return result
    

class FeatureBagMedianProbe(Datablock):
    TOPICFILES = {
        'median': 'median.npz',
        'min': 'min.npz',
        'max': 'max.npz',
    }
    @dataclass
    class CONFIG:
        featurebagclip: FeatureBagClip

    def __build__(self):
        bag_feature_list = []
        self.log.verbose(f"READING featurebags: BEGIN")
        if self.verbose:
            bagitor = tqdm.tqdm(self.cfg.featurebagclip.bags)
        else:
            bagitor = self.cfg.featurebagclip.bags
        for featurebag in bagitor:
            bag_feature_list.append(featurebag.features)
        self.log.verbose(f"READING featurebags: END")
        self.log.verbose(f"CONCATENATING bag features: BEGIN")
        features = torch.cat(bag_feature_list, dim=0).numpy()
        del bag_feature_list
        gc.collect()
        self.log.verbose(f"CONCATENATING bag features: END")
        self.log.verbose(f"COMPUTING features median, min, max: BEGIN")
        median  = np.median(features, axis=0)
        min = np.min(features, axis=0)
        max = np.max(features, axis=0)
        self.log.verbose(f"COMPUTING features median, min, max: END")
        write_npz(self.path('min', ensure_dirpath=True), min=min)
        write_npz(self.path('max', ensure_dirpath=True), max=max)
        write_npz(self.path('median', ensure_dirpath=True), median=median)
        return self

    def __read__(self, topic):
        result = read_npz(self.path(topic), topic)[topic]
        return result
    
    @functools.cached_property
    def median(self):
        return self.read('median')
    
    @functools.cached_property
    def min(self):
        return self.read('min')
    
    @functools.cached_property
    def max(self):
        return self.read('max')
    
    
class BipolarFeatureBagProbe(Datablock):
    VERSION = 1
    TOPICFILES = {
        'labels': 'labels.npz',
        'bags': 'bags.npz',
        'tile_labels': 'tile_labels.npz',
        'tile_bags': 'tile_bags.npz',
        'bag_labels': 'bag_labels.npz',
        'tile_label_indices': 'tile_label_indices.npz',
        'tile_bag_indices': 'tile_bag_indices.npz',
        'bag_label_indices': 'bag_label_indices.npz',
        'tile_bipolar_features': 'tile_bipolar_features.npz',
        'bag_features': 'bag_features.npz',
        'bag_bipolar_features': 'bag_bipolar_features.npz',
        'bag_lens': 'bag_lens.npz',
        'bag_bounds': 'bag_bounds.npz',
        #
        'bag_uq': 'bag_uq.npz',
        'bag_bipolar_uq': 'bag_bipolar_uq.npz',
        #
        'bag_logistic_evaluation_reports': 'bag_logistic_evaluation_reports.pkl',
        'tile_logistic_evaluation_reports': 'tile_logistic_evaluation_reports.pkl',
        #
        'stats_tile_features': 'stats_tile_features.npz',
        'stats_distinct_tile_features': 'stats_bag_distinct_tile_features.npz',
        'stats_distinct_tile_bipolar_features': 'stats_distinct_tile_bipolar_features.npz',
        'stats_distinct_tile_features_ratio': 'stats_distinct_tile_features_ratio.npz',
        'stats_distinct_tile_bipolar_features_ratio': 'stats_distinct_tile_bipolar_features_ratio.npz',
        #
        'stats_label_tile_features': 'stats_label_tile_features.npz',
        'stats_label_distinct_tile_features': 'stats_label_distinct_tile_features.npz',
        'stats_label_distinct_tile_bipolar_features': 'stats_label_distinct_tile_bipolar_features.npz',
        'stats_label_distinct_tile_features_ratio': 'stats_label_distinct_tile_features_ratio.npz',
        'stats_label_distinct_tile_bipolar_features_ratio': 'stats_label_distinct_tile_bipolar_features_ratio.npz',
        #
        'stats_label_bag_features': 'stats_label_bag_features.npz',
        'stats_label_distinct_bag_features': 'stats_label_distinct_bag_features.npz',
        'stats_label_distinct_bag_bipolar_features': 'stats_label_distinct_bag_bipolar_features.npz',
        'stats_label_distinct_bag_features_ratio': 'stats_label_distinct_bag_features_ratio.npz',
        'stats_label_distinct_bag_bipolar_features_ratio': 'stats_label_distinct_bag_bipolar_features_ratio.npz',
        #
        'stats_bag_tile_features': 'stats_bag_tile_features.npz',
        'stats_bag_distinct_tile_features': 'stats_distinct_tile_features.npz',
        'stats_bag_distinct_tile_bipolar_features': 'stats_bag_distinct_tile_bipolar_features.npz',
        'stats_bag_distinct_tile_features_ratio': 'stats_bag_distinct_tile_features_ratio.npz',
        'stats_bag_distinct_tile_bipolar_features_ratio': 'stats_bag_distinct_tile_bipolar_features_ratio.npz',
        #
        'stats_bag_bipolar_feature_nonzeros': 'stats_bag_bipolar_feature_nonzeros.npz',
        'stats_label_bag_bipolar_feature_nonzeros': 'stats_label_bag_bipolar_feature_nonzeros.npz',
        'stats_hamming_distances': 'stats_hamming_distances.pkl',
        #
        'bag_similarities': 'bag_similarities.npz',
        'label_similarities': 'label_similarities.npz',
    }
    @dataclass
    class CONFIG:
        featurebagclip: FeatureBagClip
        medianprobe: FeatureBagMedianProbe

    class BipolarFeaturesSimilarityShardComputer:
        def __init__(self, rows, *, device: str = 'cuda', gpu_batch_size: int = 1024, log: Logger = Logger()):
            self.rows = rows
            self.device = device
            self.gpu_batch_size = gpu_batch_size
            self.log = log

        def __call__(self, probe, features, groups, similarities):
            self.log.detailed(f"Executing BipolarFeaturesSimilarityShardComputer with {len(self.rows)} rows on device {self.device} using batch size {self.gpu_batch_size}: BEGIN")
            row_blocks = torch.split(torch.tensor(self.rows), self.gpu_batch_size)
            col_blocks = torch.split(torch.arange(features.shape[0]), self.gpu_batch_size)
            for row_block_idx, col_block_idx in itertools.product(range(len(row_blocks)), range(len(col_blocks))):
                self.log.verbose(f"COMPUTING similarities between row block {row_block_idx} and col block {col_block_idx}: BEGIN")
                row_indices = row_blocks[row_block_idx]
                col_indices = col_blocks[col_block_idx]
                f_row = torch.as_tensor(features[row_indices]).to(self.device).float()
                f_col = torch.as_tensor(features[col_indices]).to(self.device).float()
                _similarities = (f_row @ f_col.T).cpu()
                row_groups = groups[row_indices].to('cpu')
                col_groups = groups[col_indices].to('cpu')
                target_rows = row_groups.unsqueeze(1).expand(-1, len(col_groups)).reshape(-1)
                target_cols = col_groups.unsqueeze(0).expand(len(row_groups), -1).reshape(-1)
                similarities.index_put_((target_rows, target_cols), _similarities.reshape(-1), accumulate=True)
                self.log.verbose(f"COMPUTING similarities between row block {row_block_idx} and col block {col_block_idx}: END")
            self.log.detailed(f"Executing BipolarFeaturesSimilarityShardComputer with {len(self.rows)} rows on device {self.device} using batch size {self.gpu_batch_size}: END")

    @dataclass
    class Stats:
        tile_features: np.array
        distinct_tile_features: np.array
        distinct_tile_bipolar_features: np.array
        distinct_tile_features_ratio: np.array
        distinct_tile_bipolar_features_ratio: np.array
        #
        label_tile_features: np.array
        distinct_label_tile_features: np.array
        distinct_label_distinct_tile_bipolar_features: np.array
        distinct_label_tile_features_ratio: np.array
        distinct_label_distinct_tile_bipolar_features_ratio: np.array
        #
        label_bag_features: np.array
        distinct_label_bag_features: np.array
        distinct_label_bag_bipolar_features: np.array
        distinct_label_bag_features_ratio: np.array
        distinct_label_bag_bipolar_features_ratio: np.array
        #
        bag_tile_features: np.array
        distinct_bag_tile_features: np.array
        distinct_bag_bipolar_tile_features: np.array
        distinct_bag_tile_features_ratio: np.array
        distinct_bag_bipolar_tile_features_ratio: np.array
        #
        bag_bipolar_feature_nonzeros: np.array
        label_bag_bipolar_feature_nonzeros: np.array

    @dataclass
    class LogisticEvaluationReports:
        bag: dict
        tile: dict

    @dataclass
    class Similarities:
        bag: np.array
        label: np.array

    def __init__(self, *args, devices=['cuda:0'], gpu_batch_size: int = 1024, **kwargs):
        super().__init__(*args, devices=devices, gpu_batch_size=gpu_batch_size, **kwargs)

    def __build__(self):
        if not self.validtopics([
            'labels',
            'bags',
            'tile_labels',
            'tile_bags',
            'bag_labels',
            'tile_label_indices',
            'tile_bag_indices',
            'bag_label_indices',
            'tile_bipolar_features',  
            'bag_features', 
            'bag_bipolar_features', 
            'bag_lens', 
            'bag_bounds',
            'bag_uq', 
            'bag_bipolar_uq',
        ], reduce=True):
            def strings_to_indices(string_list, string_set):
                string_set = list(string_set)
                return np.array([string_set.index(s) for s in string_list])
                            
            tile_labels_list = []
            tile_bags_list = []
            bag_labels_list = []
            bag_lens_list = []
            label_list = []
            bag_list = []
            tile_feature_list = []
            self.log.verbose(f"READING tile features and labels from bags: BEGIN")
            if self.verbose:
                bagitor = tqdm.tqdm(self.cfg.featurebagclip.bags)
            else:
                bagitor = self.cfg.featurebagclip.bags
            for featurebag in bagitor:
                tile_feature_list.extend(featurebag.features)
                bag_labels_list.append(featurebag.cfg.tilebag.label)
                bag_lens_list.append(len(featurebag.features))
                bag_list.append(featurebag.cfg.tilebag.name)
                label_list.append(featurebag.cfg.tilebag.label)
                tile_labels_list.extend(featurebag.cfg.tilebag.labels)
                tile_bags_list.extend([featurebag.cfg.tilebag.name]*len(featurebag.cfg.tilebag.labels))
            labels = np.array(list(set(label_list)))
            bags = np.array(bag_list)
            tile_labels = np.array(tile_labels_list)
            tile_bags = np.array(tile_bags_list)
            bag_labels = np.array(bag_labels_list)
            bag_lens = np.array(bag_lens_list)
            tile_label_indices = strings_to_indices(tile_labels, labels)
            tile_bag_indices = strings_to_indices(tile_bags, bags)
            bag_label_indices = strings_to_indices(bag_labels, labels)
            write_npz(self.path('labels', ensure_dirpath=True), labels=labels)
            write_npz(self.path('bags', ensure_dirpath=True), bags=bags)
            bag_lens0 = np.concatenate([np.array([0]), bag_lens])
            bag_bounds = np.cumsum(bag_lens0)
            write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
            write_npz(self.path('bag_bounds', ensure_dirpath=True), bag_bounds=bag_bounds)
            write_npz(self.path('tile_labels', ensure_dirpath=True), tile_labels=tile_labels)
            write_npz(self.path('tile_label_indices', ensure_dirpath=True), tile_label_indices=tile_label_indices)
            write_npz(self.path('tile_bag_indices', ensure_dirpath=True), tile_bag_indices=tile_bag_indices)
            write_npz(self.path('bag_label_indices', ensure_dirpath=True), bag_label_indices=bag_label_indices)
            write_npz(self.path('tile_bags', ensure_dirpath=True), tile_bags=tile_bags)
            write_npz(self.path('bag_labels', ensure_dirpath=True), bag_labels=bag_labels)
            self.log.verbose(f"READING tile features and labels from bags: END")
            #
            tile_features = torch.stack(tile_feature_list, dim=0).numpy()
            self.log.verbose(f"BIPOLARIZING tile features: BEGIN")
            median = self.cfg.medianprobe.median
            tile_binary_features = (tile_features >= median).astype(int)
            tile_bipolar_features = (2.0*tile_binary_features - 1.0).astype(int)
            del tile_binary_features
            gc.collect()
            self.log.verbose(f"BIPOLARIZING tile features: END")   
            write_npz(self.path('tile_bipolar_features', ensure_dirpath=True), tile_bipolar_features=tile_bipolar_features)
            #
            self.log.verbose(f"AGGREGATING bag features: BEGIN")
            bag_feature_lists = []
            bag_bipolar_feature_lists = []
            for i in range(len(bag_bounds)-1):
                bag_bounds_diff = bag_bounds[i+1] - bag_bounds[i]
                assert bag_bounds_diff > 0, f"Nonpositive bag_bounds_diff: {i}: {bag_bounds_diff}"
                assert bag_bounds_diff == bag_lens[i], f"bag_bounds_diff != bag_lens[{i}]: {bag_bounds_diff} != {bag_lens[i]}"
                self.log.detailed(f"COMPUTING feature means of bag {i} with feature bounds {bag_bounds[i]} to {bag_bounds[i+1]} out of {len(tile_bipolar_features)}")
                _bag_features = tile_features[bag_bounds[i]:bag_bounds[i+1], :].mean(axis=0)
                bag_feature_lists.append(_bag_features)
                _bag_bipolar_features = tile_bipolar_features[bag_bounds[i]:bag_bounds[i+1], :].mean(axis=0)
                _bag_bipolar_features_int = np.round(_bag_bipolar_features).astype(int)
                bag_bipolar_feature_lists.append(_bag_bipolar_features_int)
                m, M = np.min(_bag_bipolar_features_int), np.max(_bag_bipolar_features_int)
                if m < -1 or M > 1:
                    self.log.warning(f"bag {i}: _bag_bipolar_features out of bounds: float: {_bag_bipolar_features}")
                    self.log.warning(f"bag {i}: _bag_bipolar_features out of bounds: int: {_bag_bipolar_features_int}")
            bag_features = np.stack(bag_feature_lists, axis=0)
            bag_bipolar_features = np.stack(bag_bipolar_feature_lists, axis=0)
            del bag_feature_lists
            del bag_bipolar_feature_lists
            gc.collect()
            assert bag_features.shape == (len(bag_bounds)-1, tile_features.shape[1]), \
                f"bag_features.shape != (len(bag_bounds)-1, tile_features.shape[1]), "\
                    f"{bag_features.shape} != {(len(bag_bounds)-1, tile_features.shape[1])}"
            assert bag_bipolar_features.shape == (len(bag_bounds)-1, tile_features.shape[1]), \
                f"bag_bipolar_features.shape != (len(bag_bounds)-1, tile_features.shape[1]), "\
                    f"{bag_bipolar_features.shape} != {(len(bag_bounds)-1, tile_features.shape[1])}"
            write_npz(self.path('bag_features', ensure_dirpath=True), bag_features=bag_features)
            write_npz(self.path('bag_bipolar_features', ensure_dirpath=True), bag_bipolar_features=bag_bipolar_features)
            self.log.verbose(f"AGGREGATING bag features: END")
            #
            self.log.verbose(f"COMPUTING bag features UQs: BEGIN")
            bag_uq_list = []
            bag_bipolar_uq_list = []
            for i in range(len(bag_bounds)-1):
                bag_uq_list.append(tile_features[bag_bounds[i]:bag_bounds[i+1], :].std(axis=0))
                bag_bipolar_uq_list.append(tile_bipolar_features[bag_bounds[i]:bag_bounds[i+1], :].std(axis=0))
            bag_uq = np.stack(bag_uq_list, axis=0)
            bag_bipolar_uq = np.stack(bag_bipolar_uq_list, axis=0)
            self.log.verbose(f"COMPUTING bag features UQs: END")     
            write_npz(self.path('bag_uq', ensure_dirpath=True), bag_uq=bag_uq)
            write_npz(self.path('bag_bipolar_uq', ensure_dirpath=True), bag_bipolar_uq=bag_bipolar_uq)
        else:
            labels = self.labels
            bags = self.bags
            tile_labels = self.tile_labels
            tile_bags = self.tile_bags
            bag_labels = self.bag_labels
            tile_features = self.tile_features
            tile_bipolar_features = self.tile_bipolar_features
            bag_features = self.bag_features
            bag_bipolar_features = self.bag_bipolar_features
            bag_lens = self.bag_lens
            bag_bounds = self.bag_bounds
            #
            bag_uq = self.bag_uq
            bag_bipolar_uq = self.bag_bipolar_uq

        # logistic eval
        if not self.validtopic('bag_logistic_evaluation_reports'):
            self.log.verbose(f"EVALUATING CONTINUOUS and BIPOLAR BAG features: BEGIN")
            prober = LogisticFeatureBagProber()
            bag_logistic_evaluation_reports = prober.evaluate_features2(
                (bag_features, bag_labels), 
                (bag_bipolar_features, bag_labels),
                tags=('bag_features', 'bag_bipolar_features',),
            )
            write_pickle(bag_logistic_evaluation_reports, self.path('bag_logistic_evaluation_reports', ensure_dirpath=True))
            self.log.verbose(f"EVALUATING CONTINUOUS and BIPOLAR BAG features: END")
        
        if not self.validtopic('tile_logistic_evaluation_reports'):
            self.log.verbose(f"EVALUATING CONTINUOUS and BIPOLAR TILE features: BEGIN")
            prober = LogisticFeatureBagProber()
            tile_logistic_evaluation_reports = prober.evaluate_features2(
                (tile_features, tile_labels), 
                (tile_bipolar_features, tile_labels),
                tags=('tile_features', 'tile_bipolar_features',),
            )
            write_pickle(tile_logistic_evaluation_reports, self.path('tile_logistic_evaluation_reports', ensure_dirpath=True))
            self.log.verbose(f"EVALUATING CONTINUOUS and BIPOLAR TILE features: END")

        # stats
        self.log.verbose(f"COMPUTING stats: BEGIN")
        if not self.validtopics([
            'stats_tile_features',
            'stats_distinct_tile_features',
            'stats_distinct_tile_bipolar_features',
            'stats_distinct_tile_features_ratio',
            'stats_distinct_tile_bipolar_features_ratio',
            
        ], reduce=True):
            self.log.verbose(f"COMPUTING tile features stats: BEGIN")
            stats_tile_features = np.array(len(tile_features))
            stats_distinct_tile_features = np.array(np.unique(tile_features, axis=0).shape[0])
            stats_distinct_tile_bipolar_features = np.array(np.unique(tile_bipolar_features, axis=0).shape[0])
            stats_distinct_tile_features_ratio = np.array([stats_distinct_tile_features/len(tile_features)])
            stats_distinct_tile_bipolar_features_ratio = np.array([stats_distinct_tile_bipolar_features/len(tile_features)])
            write_npz(self.path('stats_tile_features', ensure_dirpath=True), stats_tile_features=stats_tile_features)
            write_npz(self.path('stats_distinct_tile_features', ensure_dirpath=True), stats_distinct_tile_features=stats_distinct_tile_features)
            write_npz(self.path('stats_distinct_tile_bipolar_features', ensure_dirpath=True), stats_distinct_tile_bipolar_features=stats_distinct_tile_bipolar_features)
            write_npz(self.path('stats_distinct_tile_features_ratio', ensure_dirpath=True), stats_distinct_tile_features_ratio=stats_distinct_tile_features_ratio)
            write_npz(self.path('stats_distinct_tile_bipolar_features_ratio', ensure_dirpath=True), stats_distinct_tile_bipolar_features_ratio=stats_distinct_tile_bipolar_features_ratio)
            self.log.verbose(f"COMPUTING tile features stats: END")
        #
        if not self.validtopics([
            'stats_label_tile_features',
            'stats_label_distinct_tile_features',
            'stats_label_distinct_tile_bipolar_features',
            'stats_label_distinct_tile_features_ratio',
            'stats_label_distinct_tile_bipolar_features_ratio',
        ], reduce=True):
            self.log.verbose(f"COMPUTING label tile features stats: BEGIN")
            stats_label_tile_features = np.array([(tile_labels == label).sum() for label in labels])
            stats_label_distinct_tile_features = np.array([np.unique(tile_features[tile_labels == label, :], axis=0).shape[0] for label in labels])
            stats_label_distinct_tile_bipolar_features = np.array([np.unique(tile_bipolar_features[tile_labels == label, :], axis=0).shape[0] for label in labels])
            stats_label_distinct_tile_features_ratio = stats_label_distinct_tile_features/stats_label_tile_features
            stats_label_distinct_tile_bipolar_features_ratio = stats_label_distinct_tile_bipolar_features/stats_label_tile_features
            write_npz(self.path('stats_label_tile_features', ensure_dirpath=True), stats_label_tile_features=stats_label_tile_features)
            write_npz(self.path('stats_label_distinct_tile_features', ensure_dirpath=True), stats_label_distinct_tile_features=stats_label_distinct_tile_features)
            write_npz(self.path('stats_label_distinct_tile_bipolar_features', ensure_dirpath=True), stats_label_distinct_tile_bipolar_features=stats_label_distinct_tile_bipolar_features)
            write_npz(self.path('stats_label_distinct_tile_features_ratio', ensure_dirpath=True), stats_label_distinct_tile_features_ratio=stats_label_distinct_tile_features_ratio)    
            write_npz(self.path('stats_label_distinct_tile_bipolar_features_ratio', ensure_dirpath=True), stats_label_distinct_tile_bipolar_features_ratio=stats_label_distinct_tile_bipolar_features_ratio)
            self.log.verbose(f"COMPUTING label tile features stats: END")
        #
        if not self.validtopics([
            'stats_label_bag_features',
            'stats_label_distinct_bag_features',
            'stats_label_distinct_bag_bipolar_features',
            'stats_label_distinct_bag_features_ratio',
            'stats_label_distinct_bag_bipolar_features_ratio',
        ], reduce=True):
            self.log.verbose(f"COMPUTING label bag features stats: BEGIN")
            stats_label_bag_features = np.array([(bag_labels == label).sum() for label in labels])
            stats_label_distinct_bag_features = np.array([np.unique(bag_features[bag_labels == label, :], axis=0).shape[0] for label in labels])
            stats_label_distinct_bag_bipolar_features = np.array([np.unique(bag_bipolar_features[bag_labels == label, :], axis=0).shape[0] for label in labels])
            stats_label_distinct_bag_features_ratio = stats_label_distinct_bag_features/stats_label_bag_features
            stats_label_distinct_bag_bipolar_features_ratio = stats_label_distinct_bag_bipolar_features/stats_label_bag_features
            write_npz(self.path('stats_label_bag_features', ensure_dirpath=True), stats_label_bag_features=stats_label_bag_features)
            write_npz(self.path('stats_label_distinct_bag_features', ensure_dirpath=True), stats_label_distinct_bag_features=stats_label_distinct_bag_features)
            write_npz(self.path('stats_label_distinct_bag_bipolar_features', ensure_dirpath=True), stats_label_distinct_bag_bipolar_features=stats_label_distinct_bag_bipolar_features)
            write_npz(self.path('stats_label_distinct_bag_features_ratio', ensure_dirpath=True), stats_label_distinct_bag_features_ratio=stats_label_distinct_bag_features_ratio)    
            write_npz(self.path('stats_label_distinct_bag_bipolar_features_ratio', ensure_dirpath=True), stats_label_distinct_bag_bipolar_features_ratio=stats_label_distinct_bag_bipolar_features_ratio)
            self.log.verbose(f"COMPUTING label bag features stats: END")
        #
        if not self.validtopics([
            'stats_bag_tile_features',
            'stats_bag_distinct_tile_features',
            'stats_bag_distinct_tile_bipolar_features',
            'stats_bag_distinct_tile_features_ratio',
            'stats_bag_distinct_tile_bipolar_features_ratio',
        ], reduce=True):
            self.log.verbose(f"COMPUTING bag tile features stats: BEGIN")
            stats_bag_tile_features = bag_lens
            stats_bag_distinct_tile_features = np.array([np.unique(tile_features[bag_bounds[i]:bag_bounds[i+1], :], axis=0).shape[0] for i in range(len(bags))])
            stats_bag_distinct_tile_bipolar_features = np.array([np.unique(tile_bipolar_features[bag_bounds[i]:bag_bounds[i+1], :], axis=0).shape[0] for i in range(len(bags))])
            stats_bag_distinct_tile_features_ratio = np.array([stats_bag_distinct_tile_features/stats_bag_tile_features])
            stats_bag_distinct_tile_bipolar_features_ratio = np.array([stats_bag_distinct_tile_bipolar_features/stats_bag_tile_features])
            write_npz(self.path('stats_bag_tile_features', ensure_dirpath=True), stats_bag_tile_features=stats_bag_tile_features)
            write_npz(self.path('stats_bag_distinct_tile_features', ensure_dirpath=True), stats_bag_distinct_tile_features=stats_bag_distinct_tile_features)
            write_npz(self.path('stats_bag_distinct_tile_bipolar_features', ensure_dirpath=True), stats_bag_distinct_tile_bipolar_features=stats_bag_distinct_tile_bipolar_features)
            write_npz(self.path('stats_bag_distinct_tile_features_ratio', ensure_dirpath=True), stats_bag_distinct_tile_features_ratio=stats_bag_distinct_tile_features_ratio)
            write_npz(self.path('stats_bag_distinct_tile_bipolar_features_ratio', ensure_dirpath=True), stats_bag_distinct_tile_bipolar_features_ratio=stats_bag_distinct_tile_bipolar_features_ratio)
            self.log.verbose(f"COMPUTING bag tile features stats: END")
        #
        if not self.validtopic('stats_bag_bipolar_feature_nonzeros'):
            self.log.verbose(f"COMPUTING bag bipolar feature nonzeros: BEGIN")
            stats_bag_bipolar_feature_nonzeros = (bag_bipolar_features != 0).sum(axis=1)
            write_npz(self.path('stats_bag_bipolar_feature_nonzeros', ensure_dirpath=True), stats_bag_bipolar_feature_nonzeros=stats_bag_bipolar_feature_nonzeros)
            self.log.verbose(f"COMPUTING bag bipolar feature nonzeros: END")  
            self.log.verbose(f"COMPUTING label bag bipolar feature nonzeros: BEGIN")
            stats_label_bag_bipolar_feature_nonzeros = np.array([
                (bag_bipolar_features[bag_labels == label, :] != 0).sum()/(bag_bipolar_features[bag_labels == label, :] != 0).shape[0]
                for label in labels
            ])
            write_npz(self.path('stats_label_bag_bipolar_feature_nonzeros', ensure_dirpath=True), stats_label_bag_bipolar_feature_nonzeros=stats_label_bag_bipolar_feature_nonzeros)
            self.log.verbose(f"COMPUTING label bag bipolar feature nonzeros: END")
        #                             
        self.log.verbose(f"COMPUTING stats: END")
        #
        if not self.validtopic('stats_hamming_distances'):
            prober = FeatureBagProber()
            self.log.verbose(f"COMPUTING Hamming distance stats: BEGIN")
            hamming_stats = prober.pairwise_hamming_distance_stats(bag_bipolar_features, bag_labels)
            write_pickle(hamming_stats, self.path('stats_hamming_distances', ensure_dirpath=True))
            self.log.verbose(f"COMPUTING Hamming distance stats: END")
        #
        self.__build_similarities__(tile_bipolar_features=tile_bipolar_features)
        return self
    
    def __read__(self, topic):
        if topic in [
            'bag_logistic_evaluation_reports',
            'tile_logistic_evaluation_reports',
            'stats_hamming_distances',
        ]:
            result = read_pickle(self.path(topic))
        else:
            result = read_npz(self.path(topic), topic)[topic]
        return result
    
    @functools.cached_property
    def labels(self):
        return self.read('labels')
    
    @property
    def n_distinct_labels(self):
        return len(set(self.labels))
    
    @functools.cached_property
    def bags(self):
        return self.read('bags')
    
    @property
    def n_bags(self):
        return len(self.bag_lens)
    
    @functools.cached_property
    def bag_lens(self):
        return self.read('bag_lens')
    
    @functools.cached_property
    def bag_bounds(self):
        return self.read('bag_bounds')
    
    @functools.cached_property
    def tile_bags(self):
        return torch.tensor(list(itertools.chain.from_iterable([i]*self.bag_lens[i] for i in range(self.n_bags))))
            
    @functools.cached_property
    def tile_labels(self):
        return self.read('tile_labels')
    
    @functools.cached_property
    def tile_bags(self):
        return self.read('tile_bags')
    
    @functools.cached_property
    def tile_label_indices(self):
        return self.read('tile_label_indices')
    
    @functools.cached_property
    def tile_bag_indices(self):
        return self.read('tile_bag_indices')
    
    @functools.cached_property
    def bag_label_indices(self):
        return self.read('bag_label_indices')
    
    @functools.cached_property
    def label_tiles(self):
        return self.read('label_tiles')
    
    @functools.cached_property
    def bag_labels(self):
        return self.read('bag_labels')
    
    @functools.cached_property
    def label_bags(self):
        return self.read('label_bags')

    @functools.cached_property
    def tile_features(self):
        tile_feature_list = []
        self.log.verbose(f"READING tile features from bags: BEGIN")
        if self.verbose:
            bagitor = tqdm.tqdm(self.cfg.featurebagclip.bags)
        else:
            bagitor = self.cfg.featurebagclip.bags
        for featurebag in bagitor:
            tile_feature_list.extend(featurebag.features)
        self.log.verbose(f"READING tile features from bags: END")
        tile_features = torch.stack(tile_feature_list, dim=0).numpy()
        del tile_feature_list
        gc.collect()
        return tile_features
    
    @functools.cached_property
    def tile_bipolar_features(self):
        return self.read('tile_bipolar_features')
    
    @functools.cached_property
    def bag_features(self):
        return self.read('bag_features')
    
    @functools.cached_property
    def bag_bipolar_features(self):
        return self.read('bag_bipolar_features')
    
    @functools.cached_property
    def bag_uq(self):
        return self.read('bag_uq')
    
    @functools.cached_property
    def bag_bipolar_uq(self):
        return self.read('bag_bipolar_uq')
    
    @functools.cached_property
    def bag_logistic_evaluation_reports(self):
        return self.read('bag_logistic_evaluation_reports')
    
    @functools.cached_property
    def stats(self):
        return self.Stats(
           tile_features=self.read('stats_tile_features').item(),
           distinct_tile_features=self.read('stats_distinct_tile_features').item(),
           distinct_tile_bipolar_features=self.read('stats_distinct_tile_bipolar_features').item(),
           distinct_tile_features_ratio=self.read('stats_distinct_tile_features_ratio').item(),
           distinct_tile_bipolar_features_ratio=self.read('stats_distinct_tile_bipolar_features_ratio').item(),
           #
           label_tile_features=self.read('stats_label_tile_features'),
           distinct_label_tile_features=self.read('stats_label_distinct_tile_features'),
           distinct_label_distinct_tile_bipolar_features=self.read('stats_label_distinct_tile_bipolar_features'),
           distinct_label_tile_features_ratio=self.read('stats_label_distinct_tile_features_ratio'),
           distinct_label_distinct_tile_bipolar_features_ratio=self.read('stats_label_distinct_tile_bipolar_features_ratio'),
           #
           label_bag_features=self.read('stats_label_bag_features'),
           distinct_label_bag_features=self.read('stats_label_distinct_bag_features'),
           distinct_label_bag_bipolar_features=self.read('stats_label_distinct_bag_bipolar_features'),
           distinct_label_bag_features_ratio=self.read('stats_label_distinct_bag_features_ratio'),
           distinct_label_bag_bipolar_features_ratio=self.read('stats_label_distinct_bag_bipolar_features_ratio'),
           #
           bag_tile_features=self.read('stats_bag_tile_features'),
           distinct_bag_tile_features=self.read('stats_bag_distinct_tile_features'),
           distinct_bag_bipolar_tile_features=self.read('stats_bag_distinct_tile_bipolar_features'),
           distinct_bag_tile_features_ratio=self.read('stats_bag_distinct_tile_features_ratio'),
           distinct_bag_bipolar_tile_features_ratio=self.read('stats_bag_distinct_tile_bipolar_features_ratio'),
           #
           bag_bipolar_feature_nonzeros=self.read('stats_bag_bipolar_feature_nonzeros'),
           label_bag_bipolar_feature_nonzeros=self.read('stats_label_bag_bipolar_feature_nonzeros'),
        )
    @functools.cached_property
    def hamming_distance_stats(self):
        return self.read('stats_hamming_distances')
    
    @functools.cached_property
    def similarities(self):
        return self.Similarities(
            bag=self.bag_similarities, 
            label=self.label_similarities
        )
    
    @functools.cached_property
    def bag_similarities(self):
        return self.read('bag_similarities')
    
    @functools.cached_property
    def label_similarities(self):
        return self.read('label_similarities')
    
    def __build_similarities__(self, tile_bipolar_features):
        if not self.validtopics([
            'bag_similarities',
            'label_similarities',
        ], reduce=True):
            if not self.validtopic('bag_similarities'):
                bag_similarities = torch.zeros((self.n_bags, self.n_bags))
                bag_indices = torch.tensor_split(torch.arange(self.n_bags), len(self.devices))
                bag_similarities_computers = [
                    self.BipolarFeaturesSimilarityShardComputer(bag_indices[i], device=self.devices[i], gpu_batch_size=self.gpu_batch_size, log=self.log)
                    for i in range(len(self.devices))
                ]
                self.log.verbose(f"COMPUTING bag similarities using {len(self.devices)} devices with gpu_batch_size {self.gpu_batch_size}: BEGIN")
                bag_executor = MultithreadingCallableExecutor(n_threads=len(self.devices), log=self.log)
                bag_executor.execute(bag_similarities_computers, self, tile_bipolar_features, torch.tensor(self.tile_bag_indices), bag_similarities)
                write_npz(self.path('bag_similarities', ensure_dirpath=True), bag_similarities=bag_similarities)
                self.log.verbose(f"COMPUTING bag similarities using {len(self.devices)} devices with gpu_batch_size {self.gpu_batch_size}: END")

            if not self.validtopic('label_similarities'):  
                label_similarities = torch.zeros((self.n_distinct_labels, self.n_distinct_labels))
                label_indices = torch.tensor_split(torch.arange(self.n_distinct_labels), len(self.devices))
                label_similarities_computers = [
                    self.BipolarFeaturesSimilarityShardComputer(label_indices[i], device=self.devices[i], gpu_batch_size=self.gpu_batch_size, log=self.log)
                    for i in range(len(self.devices))
                ]
                self.log.verbose(f"COMPUTING label similarities using {len(self.devices)} devices with gpu_batch_size {self.gpu_batch_size}: BEGIN")
                label_executor = MultithreadingCallableExecutor(n_threads=len(self.devices), log=self.log)
                label_executor.execute(label_similarities_computers, self, tile_bipolar_features, torch.tensor(self.tile_label_indices), label_similarities)
                write_npz(self.path('label_similarities', ensure_dirpath=True), label_similarities=label_similarities)
                self.log.verbose(f"COMPUTING label similarities using {len(self.devices)} devices with gpu_batch_size {self.gpu_batch_size}: END")   
        


class FeaturePairwiseDistancesShard(Datablock):
    VERSION = 5
    TOPICFILES = {
        'rows': 'rows.npz',
        'cols': 'cols.npz',
        'distances': "distances.pt"
    }

    @dataclass
    class CONFIG:
        features: FeatureBagClip
        shard_idx: int
        row_shard_size: int
        col_shard_size: int
        seed: int = 42

    def __build__(self, features):
        rng = np.random.default_rng(self.cfg.seed)
        M = features.shape[0]
        rows = rng.permutation(M)
        _rows = rows[self.cfg.shard_idx*self.cfg.row_shard_size:min((self.cfg.shard_idx+1)*self.cfg.row_shard_size, M)]
        del rows
        cols = rng.permutation(M)
        col_offset = rng.integers(M-self.cfg.col_shard_size)
        _cols = cols[col_offset:min(col_offset+self.cfg.col_shard_size, M)]
        del cols
        self.log.detailed(f"Building FeaturePairwiseDistancesShard at index {self.cfg.shard_idx} with rows {_rows} and {len(_cols)} columns on device {features.device}")
        pairwise_distances = torch.cdist(features[_rows], features[_cols]).to('cpu')
        self.log.debug(f"Built FeaturePairwiseDistancesShard at index {self.cfg.shard_idx} with shape {pairwise_distances.shape} on device {features.device}")
        write_npz(self.path('rows', ensure_dirpath=True), rows=_rows)
        write_npz(self.path('cols', ensure_dirpath=True), cols=_cols)
        write_tensor(pairwise_distances, self.path('distances', ensure_dirpath=True))
        del _rows
        del _cols
        del pairwise_distances
        return self
    
    def __read__(self, topic):
        if topic == 'rows':
            result = read_npz(self.path('rows'), 'rows')[0]
        elif topic == 'cols':
            result = read_npz(self.path('cols'), 'cols')[0]
        elif topic == 'distances':
            result = read_tensor(self.path('distances'))
        else:
            raise ValueError(f"Unknown topic: {topic}") 
        return result
    
    @functools.cached_property
    def rows(self):
        return self.read('rows')
    
    @functools.cached_property
    def cols(self):
        return self.read('cols')

    @functools.cached_property
    def distances(self):
        return self.read('distances')
    
    @functools.cached_property
    def tensor(self):
        return self.distances
    

class FeaturePairwiseDistances(Datablock):
    VERSION = 5 
    TOPICFILES = {
        "features_shape": "features_shape.npy",
    }
    @dataclass
    class CONFIG:
        featurebagclip: FeatureBagClip
        n_shards: int
        row_shard_size: int
        col_shard_size: int
        layer: str = None
        seed: int = 42

    def __init__(self, *args, n_devices: int = 1, gpu_batch_size: int = 1024, **kwargs):
        super().__init__(*args, n_devices=n_devices, gpu_batch_size=gpu_batch_size, **kwargs)

    def __post_init__(self):
        self.devices = [f"cuda:{i}" for i in range(self.n_devices)]
        self.cfg.featurebagclip = self.cfg.featurebagclip.set(devices=self.devices, gpu_batch_size=self.gpu_batch_size)
        return self
    
    def build_tree(self):
        self.cfg.featurebagclip.set(devices=self.devices).build()
        return self.build()
        
    def features(self):
        self.log.debug(f"Reading features from layer {self.cfg.layer} of {len(self.cfg.featurebagclip.shards)} feature shards")
        feature_list = []
        for featureshard in self.cfg.featurebagclip.shards:
            feature_list.extend(featureshard.layer(self.cfg.layer))
        features = torch.stack(feature_list)
        self.log.debug(f"Combined features: shape: {features.shape}")
        return features

    @functools.cached_property
    def shards(self):
        return self._shards(self.features_shape)

    def _shards(self, shape):
        assert self.cfg.n_shards*self.cfg.row_shard_size <= shape[0], f"Too many shards or shards too big for n_features: {shape[0]}"
        assert self.cfg.col_shard_size <= shape[0], f"col_shard_size {self.cfg.col_shard_size=} too large for n_features {shape[0]}"
        return [FeaturePairwiseDistancesShard(spec=dict(
                    features=self.spec['features'],
                    shard_idx=i,
                    row_shard_size=self.cfg.row_shard_size,
                    col_shard_size=self.cfg.col_shard_size,
                    seed=self.cfg.seed,
                    )) 
                for i in range(self.cfg.n_shards)
            ]
    
    @property
    def n_shards(self):
        return self.cfg.n_shards
    
    @functools.cached_property
    def features_shape(self):
        if self.valid():
            self.log.debug(f"Reading features_shape from {self.path('features_shape')}")
            return read_tensor(self.path('features_shape'))
        else:
            self.log.debug(f"Calculating features_shape")
            return self.features().shape

    def __build__(self):
        self.log.debug("Retrieving features")
        features = self.features()
        shards = self._shards(features.shape)
        self.log.debug(f"Forming {self.cfg.n_shards} FeaturePairwiseDistancesShards from features of shape {features.shape}, "
                       f"row_shard_size {self.cfg.row_shard_size}, col_shard_size: {self.cfg.col_shard_size}"
        )
        self.log.debug(f"Formed {len(shards)} FeaturePairwiseDistancesShards.  Looking for missing shards")
        missing_shards = [shard for shard in shards if not shard.valid()]
        self.log.debug(f"Found {len(missing_shards)} missing shards")
        self.log.debug(f"Building all missing pairwise feature distance shards")
        built_shards = TorchMultithreadingDatablocksBuilder(devices=self.devices, log=self.log).build_blocks(missing_shards, features)
        self.log.verbose(f"Built all missing pairwise feature distance shards: {len(built_shards)}")
        write_tensor(torch.tensor(features.shape), self.path('features_shape', ensure_dirpath=True))
        return self
    
    def __read__(self, topic):
        if topic == 'features_shape':
            result = read_tensor(self.path('features_shape'))
        return result


class FeatureSortedDistancesShard(Datablock):
    VERSION = 5
    TOPICFILES = {
        "sorted_distances": "sorted_distances.npy",
        "original_order_indices": "original_order_indices.npy",
    }
    
    @dataclass
    class CONFIG:
        distshard: FeaturePairwiseDistancesShard

    def __build__(self):
        shard = self.cfg.distshard.distances.to(self.device)
        self.log.debug(f"Sorting pairwise distances in FeaturePairwiseDistancesShard {self.cfg.distshard.hashpath()} of shape {shard.shape} on device {self.device}")
        sorted_shard, original_order_indices = torch.sort(shard, dim=-1, descending=False)
        write_tensor(sorted_shard.to('cpu'), self.path('sorted_distances', ensure_dirpath=True))
        write_tensor(original_order_indices.to('cpu'), self.path('original_order_indices', ensure_dirpath=True))
        del shard
        del sorted_shard
        del original_order_indices
        gc.collect()
        self.log.debug(f"Built FeatureSortedDistancesShard {self.hashpath()} on device {self.device}")
        return self
    
    def __read__(self, topic):
        if topic == 'sorted_distances':
            result = read_tensor(self.path(topic))
        elif topic == 'original_order_indices':
            result = read_tensor(self.path(topic))
        else:
            raise ValueError(f"Unknown topic: {topic}")
        return result
    
    @functools.cached_property
    def original_order_indices(self):
        return self.read('original_order_indices')
    
    @property
    def order(self):
        return self.original_order_indices
    
    @functools.cached_property
    def sorted_distances(self):
        return self.read('sorted_distances')

    @functools.cached_property
    def tensor(self):
        tensor = self.read('sorted_distances')
        return tensor
        

class FeatureSortedDistances(Datablock):
    VERSION = 5
    TOPICFILE = "breadcrumbs"

    @dataclass
    class CONFIG:
        features_pairwise_distances: FeaturePairwiseDistances

    def __init__(self, *args, n_workers: int = 1, use_gpus: bool = True, **kwargs):
        super().__init__(*args, n_workers=n_workers, use_gpus=use_gpus, **kwargs)

    def __post_init__(self):
        self.devices = [f"cuda:{i}" if self.use_gpus else f"cpu" for i in range(self.n_workers)]
        return self

    def __build__(self):
        sorted_distshards = self.shards
        self.log.debug(f"Formed {len(sorted_distshards)} FeatureSortedDistancesShards")
        missing_sorted_distshards = [shard for shard in sorted_distshards if not shard.valid()]
        self.log.debug(f"Found among them {len(missing_sorted_distshards)} missing FeatureSortedDistancesShards")
        self.log.debug(f"Building {len(missing_sorted_distshards)} FeatureSortedDistancesShards")
        built_sorted_distshards = TorchMultiprocessingDatablocksBuilder(devices=self.devices, log=self.log).build_blocks(missing_sorted_distshards)
        self.log.verbose(f"Built all missing FeatureSortedDistancesShards: {len(built_sorted_distshards)}")
        return self
    
    def UNSAFE_clear_shards(self):
        for shard in self.shards:
            shard.UNSAFE_clear()
        return self
    
    @functools.cached_property
    def shards(self):
        distshards = self.cfg.features_pairwise_distances.shards
        sorted_distshards = [FeatureSortedDistancesShard(spec=dict(distshard=distshard,),) 
                            for distshard in distshards
        ]
        return sorted_distshards
    

class Feature2NNDistancesShard(Datablock):
    TOPICFILE = "twonn_distances.pt"
    @dataclass
    class CONFIG:
        sorted_distshard: FeatureSortedDistancesShard

    def __build__(self):
        sorted_distshard = self.cfg.sorted_distshard
        sorted_distshard_tensor = sorted_distshard.tensor
        self.log.debug(f"Locating smallest nonzero pairwise distances in FeatureSortedDistancesShard {sorted_distshard.hashpath()} of shape {sorted_distshard_tensor.shape}")
        firstidx_j = (sorted_distshard_tensor[:, 0] == 0.0).to(int)
        firstidx_i = torch.arange(sorted_distshard_tensor.shape[0])
        firstdist = sorted_distshard_tensor[firstidx_i, firstidx_j]
        if self.debug:
            firstdist_min, firstdist_max = firstdist.min(), firstdist.max()
            self.log.debug(f"firstdist_min: {firstdist_min}, firstdist_max: {firstdist_max}")
        self.log.debug(f"Locating second smallest pairwise distances in FeatureSortedDistancesShard {sorted_distshard.hashpath()} of shape {sorted_distshard_tensor.shape}")
        seconddist = sorted_distshard_tensor[firstidx_i, firstidx_j+1]
        if self.debug:
            seconddist_min, seconddist_max = seconddist.min(), seconddist.max()
            self.log.debug(f"seconddist_min: {seconddist_min}, seconddist_max: {seconddist_max}")
        del sorted_distshard_tensor
        twonn_distances = torch.stack([firstdist, seconddist], dim=-1)
        self.log.debug(f"Built Feature2NNDistancesShard {self.hashpath()} with shape {twonn_distances.shape}")
        write_tensor(twonn_distances, self.path(ensure_dirpath=True))
        return self
    
    def __read__(self):
        result = read_tensor(self.path())
        return result
    
    @functools.cached_property
    def tensor(self):
        return self.read()
    

class Feature2NNDistances(Datablock):
    TOPICFILES = {
        "twonn_distances": "twonn_distances.pt", 
    }
    @dataclass
    class CONFIG:
        feature_sorted_distances: FeatureSortedDistances

    def __init__(self, *args, n_workers: int = 1, use_gpus: bool = False, **kwargs):
        super().__init__(*args, n_workers=n_workers, use_gpus=use_gpus, **kwargs)

    def __post_init__(self):
        self.devices = [f"cuda:{i}" if self.use_gpus else f"cpu" for i in range(self.n_workers)]
        return self
    
    def shards(self):
        sorted_distshards = self.cfg.feature_sorted_distances.shards
        twonndist_shards = [Feature2NNDistancesShard(spec=dict(sorted_distshard=sorted_distshard)) 
                            for sorted_distshard in sorted_distshards
        ]
        return twonndist_shards
    
    def UNSAFE_clear_shards(self):
        for shard in self.shards():
            shard.UNSAFE_clear()
        return self

    def __build__(self):
        twonndist_shards = self.shards()
        self.log.debug(f"Formed {len(twonndist_shards)} Feature2NNDistancesShards")
        missing_twonndist_shards = [shard for shard in twonndist_shards if not shard.valid()]
        self.log.debug(f"Found {len(missing_twonndist_shards)} missing Feature2NNDistancesShards")
        self.log.debug(f"Building {len(missing_twonndist_shards)} Feature2NNDistancesShards")
        built_twonndist_shards = TorchMultiprocessingDatablocksBuilder(devices=self.devices, log=self.log).build_blocks(missing_twonndist_shards)
        self.log.verbose(f"Built all missing Feature2NNDistancesShards: {len(built_twonndist_shards)}")
        twonndists = torch.cat([shard.tensor for shard in twonndist_shards], dim=0)
        write_tensor(twonndists, self.path('twonn_distances', ensure_dirpath=True))
        return self
    
    def __read__(self, topic):
        if topic == 'twonn_distances':
            result = read_tensor(self.path('twonn_distances'))
        return result
    
    @functools.cached_property
    def tensor(self):
        return self.read('twonn_distances')
                       

class Feature2NNDim(Datablock):
    TOPICFILES = {
        "dimension": "dimension.pt",
        "model": "model.pkl",
    }
    @dataclass
    class CONFIG:
        feature_2nn_distances: Feature2NNDistances

    def __build__(self):
        twonndists = self.cfg.feature_2nn_distances.tensor
        mus = twonndists[:, 1]/twonndists[:, 0]
        sortedmus, _ = torch.sort(mus, descending=False)
        del mus
        logsortedmus = torch.log(sortedmus)
        cumprob = torch.arange(0, len(logsortedmus))/len(logsortedmus)
        x = logsortedmus
        y = -torch.log(1.0 - cumprob)
        noninf = torch.isfinite(x) & torch.isfinite(y)
        x = x[noninf]
        y = y[noninf]
        X = x.reshape(-1, 1)
        Y = y.reshape(-1, 1)
        model = LinearRegression()
        self.log.debug(f"Fitting linear model on {len(X)} mu points")
        model.fit(X, Y)
        self.log.debug(f"Built linear model of dimension {model.coef_[0]}")
        write_tensor(torch.tensor([model.coef_[0]]), self.path('dimension', ensure_dirpath=True))
        write_pickle(model, self.path('model', ensure_dirpath=True))
        return self

    def __read__(self, topic):
        if topic == 'dimension':
            result = read_tensor(self.path('dimension'))
        elif topic == 'model':
            result = read_pickle(self.path('model'))
        return result
    
    @functools.cached_property
    def dimension(self):
        return self.read('dimension').item()
    
    @property
    def dim(self):
        return self.dimension
    
    @functools.cached_property
    def model(self):
        return self.read('model')
