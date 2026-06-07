"""Deep feature probes — logistic regression, statistics, and spectral analysis.

Provides :class:`DeepFeatureAffineLogisticProbe` (logistic probe with model
persistence), :class:`DeepFeatureStatsProbe` (tile/bag-level statistics),
:class:`DeepFeatureSpectralProber` (model-agnostic Jacobian SVD engine),
:class:`DeepBackboneSpectralEvaluator` (runtime evaluator with spectral probing),
and :class:`DeepFeatureSpectralProbe` (persistent spectral probe Datablock)
for deep feature clips and multi-layer activations.
"""

import functools
import gc
from dataclasses import dataclass

import tqdm
import numpy as np
import torch

from scipy.sparse.linalg import svds as scipy_svds
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

from typing import Dict, List, Optional, Tuple

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

import dbx
from dbx import (
    Datablock,
    write_tensor,
    read_tensor,
    write_npz,
    read_npz,
    write_pickle,
    read_pickle,
)

from autopath.autobits import DeepBackboneEvaluator


# ═══════════════════════════════════════════════════════════════════════
#  Annotation-based label extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_bag_label(bag, annotation_key: str | None) -> str:
    """Extract a label string for a :class:`DeepFeatureBag`.

    Parameters
    ----------
    bag : DeepFeatureBag
        Must expose ``.tilebag`` (a :class:`PancanTileBag`).
    annotation_key : str | None
        If ``None``, returns ``bag.tilebag.label`` (the cohort-level
        cancer-type string).  Otherwise, reads the first sample from
        the tilebag's MDS dataset and navigates the dotted key path
        into the ``annotations`` dict.  For example,
        ``"gdc_clinical.primary_diagnosis"`` fetches
        ``sample['annotations']['gdc_clinical']['primary_diagnosis']``.

    Returns
    -------
    str
        The extracted label.
    """
    if annotation_key is None:
        return bag.tilebag.label
    annotations = bag.annotations()
    if annotations is None:
        return bag.tilebag.label
    val = annotations
    for part in annotation_key.split('.'):
        if isinstance(val, dict):
            val = val.get(part)
        else:
            return bag.tilebag.label
    return str(val) if val is not None else bag.tilebag.label


def _extract_bag_annotations(bag) -> dict | None:
    """Read the bag-level annotation dict from the first MDS tile sample.

    Delegates to :meth:`DeepFeatureBag.annotations`, which handles
    dataset lifecycle (shared-memory cleanup).

    Returns ``None`` when the tilebag has no ``annotations`` column or
    the value is ``None``.
    """
    return bag.annotations()




_NORMALIZE_MODES = {None, 'l2', 'corner-l1', 'corner-l2', 'corner-linfty'}


def normalize_features(features: torch.Tensor, mode: str | None) -> torch.Tensor:
    """Apply optional normalization to a feature tensor.

    Parameters
    ----------
    features : Tensor
        Shape ``(N, D)`` or ``(D,)``.
    mode : str | None
        - ``None`` — no-op (raw features).
        - ``'l2'`` — L2-normalise (project onto unit sphere).
        - ``'corner-l1'`` — snap to nearest hypercube vertex in L1
          norm: ``sign(x)``.  Each coordinate is independently mapped
          to ±1.
        - ``'corner-l2'`` — snap to nearest hypercube vertex in L2
          norm: ``sign(x)`` (same vertex as L1, since the closest
          vertex in L2 is also ``sign(x)``).
        - ``'corner-linfty'`` — snap to nearest hypercube vertex in
          L∞ norm: set the coordinate with largest absolute value to
          ±1 and all others to 0.  This selects the axis-aligned
          vertex closest in L∞.

    Returns
    -------
    Tensor
        Normalised features (same shape).
    """
    if mode is None:
        return features
    elif mode == 'l2':
        return torch.nn.functional.normalize(features.float(), p=2, dim=-1)
    elif mode in ('corner-l1', 'corner-l2'):
        # For both L1 and L2 norms, the nearest vertex of {-1,+1}^d
        # to a point x is sign(x).
        return torch.sign(features)
    elif mode == 'corner-linfty':
        # Nearest vertex in L∞: pick the coordinate with the largest
        # absolute value, set it to ±1, and zero the rest.
        abs_f = features.abs()
        idx = abs_f.argmax(dim=-1, keepdim=True)
        result = torch.zeros_like(features)
        result.scatter_(-1, idx, features.gather(-1, idx).sign())
        return result
    else:
        raise ValueError(
            f"Unknown normalization mode {mode!r}; "
            f"expected one of {sorted(m for m in _NORMALIZE_MODES if m is not None)}"
        )



# ═══════════════════════════════════════════════════════════════════════
#  Deep Feature Affine Logistic Prober
# ═══════════════════════════════════════════════════════════════════════

class DeepFeatureAffineLogisticProber:
    """Standalone logistic regression evaluator for deep features.

    Provides static ``evaluate_features`` and ``evaluate_features2``
    helpers that train/test a :class:`~sklearn.linear_model.LogisticRegression`
    and return :func:`~sklearn.metrics.classification_report` strings.

    Unlike the legacy :class:`LogisticFeatureBagProber`, this class is
    intended to be used as an **attribute** (or called directly), not
    as a superclass of a :class:`Datablock`.
    """

    def __init__(self, log=dbx.Logger()):
        self.log = log

    @staticmethod
    def ndarray(X):
        """Coerce *X* to a numpy ndarray."""
        if isinstance(X, list):
            return np.array(X)
        elif isinstance(X, torch.Tensor):
            return X.numpy()
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError(f"Unknown input type: {type(X)}")

    @staticmethod
    def evaluate_features(Xy, *, fraction=0.8, fit_intercept=True):
        """Train/test a LogisticRegression and return the classification report.

        Parameters
        ----------
        Xy : tuple[array-like, array-like]
            ``(features, labels)``.
        fraction : float
            Fraction used for training.
        fit_intercept : bool
            Forwarded to :class:`LogisticRegression`.

        Returns
        -------
        str
            ``classification_report`` text.
        """
        features, labels = Xy
        features = DeepFeatureAffineLogisticProber.ndarray(features)
        labels = DeepFeatureAffineLogisticProber.ndarray(labels)
        N = len(labels)
        ntrain = int(N * fraction)
        perm = np.random.permutation(N)
        X_train, y_train = features[perm[:ntrain]], labels[perm[:ntrain]]
        X_test, y_test = features[perm[ntrain:]], labels[perm[ntrain:]]

        clf = LogisticRegression(fit_intercept=fit_intercept)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return classification_report(y_test, y_pred)

    @staticmethod
    def evaluate_features2(Xy1, Xy2, *, fraction=0.8, fit_intercept=True,
                           tags=("(1)", "(2)"), log=dbx.Logger()):
        """Evaluate two feature sets side-by-side.

        Returns
        -------
        tuple[str, str]
            ``(report1, report2)`` — classification report strings.
        """
        import datetime
        label1, label2 = tags
        log.verbose(f"EVALUATING features: {label1}: started at {datetime.datetime.now()}")
        report1 = DeepFeatureAffineLogisticProber.evaluate_features(
            Xy1, fraction=fraction, fit_intercept=fit_intercept)
        log.verbose(f"EVALUATING features: {label1}: finished at {datetime.datetime.now()}")
        log.verbose(f"EVALUATING features: {label2}: started at {datetime.datetime.now()}")
        report2 = DeepFeatureAffineLogisticProber.evaluate_features(
            Xy2, fraction=fraction, fit_intercept=fit_intercept)
        log.verbose(f"EVALUATING features: {label2}: finished at {datetime.datetime.now()}")

        rstr = (f"---------- {label1} ------------\n{report1}\n"
                f"---------- {label2} ------------\n{report2}")
        log.verbose(rstr)
        return report1, report2


# ═══════════════════════════════════════════════════════════════════════
#  Deep Feature Affine Logistic Feature-Bag Probe
# ═══════════════════════════════════════════════════════════════════════

class DeepFeatureAffineLogisticProbe(Datablock):
    """Fit a logistic classifier on bag-level deep features for a single layer.

    Adapted from :class:`~autopath.probes.AffineLogisticFeatureBagProbe`
    for the deep feature clip architecture.  Persists the fitted
    classifier's ``coef_``, ``intercept_``, and ``classes_`` arrays so
    that the separating hyperplane can be inspected after building.

    Labels are extracted via annotations when ``annotation_key`` is set
    (e.g. ``"cohort"`` or ``"gdc_clinical.primary_diagnosis"``), falling
    back to ``tilebag.label`` when ``annotation_key is None``.

    An optional ``normalize`` config controls feature pre-processing
    before fitting:

    - ``None`` — use raw features
    - ``'l2'`` — L2-normalise (project onto unit sphere)
    - ``'corner-l1'`` / ``'corner-l2'`` — snap to nearest hypercube vertex (``sign(x)``)
    - ``'corner-linfty'`` — snap to axis-aligned vertex (largest-magnitude coordinate)

    The :class:`DeepFeatureAffineLogisticProber` is used as an
    **attribute** rather than a superclass.

    Persisted topics
    ----------------
    - ``bag_labels``          — per-bag label array
    - ``bag_annotations``     — per-bag annotation dicts
    - ``bag_features``        — per-bag mean feature vectors
    - ``evaluation_report``   — sklearn ``classification_report`` string
    - ``coef``                — weight matrix  ``(n_classes, n_features)``
    - ``intercept``           — intercept vector ``(n_classes,)``
    - ``classes``             — ordered class labels from the fitted model
    """

    TOPICFILES = {
        'bag_labels':        'bag_labels.npz',
        'bag_annotations':   'bag_annotations.pkl',
        'bag_features':      'bag_features.npy',
        'evaluation_report': 'evaluation_report.pkl',
        'coef':              'coef.npy',
        'intercept':         'intercept.npy',
        'classes':           'classes.npz',
    }

    @dataclass
    class CONFIG:
        clip: object        # DeepFeatureClip
        layer: str          # which capture key to probe (e.g. "B_0_norm1")
        annotation_key: str | None = None  # dotted path into annotations dict, or None for tilebag.label
        fit_intercept: bool = True
        evaluation_fraction: float = 0.8
        aggregation: str = "mean"
        normalize: str | None = None   # None, 'l2', 'corner-l1', 'corner-l2', 'corner-linfty'

    def __post_init__(self):
        assert self.cfg.aggregation in ["mean"], \
            f"Unknown aggregation: {self.cfg.aggregation}"
        assert self.cfg.normalize in _NORMALIZE_MODES, \
            f"Unknown normalize mode: {self.cfg.normalize!r}"
        self._prober = DeepFeatureAffineLogisticProber(log=self.log)
        return self

    def __build__(self):
        # ---- collect bag features, labels, and annotations ----------------
        bag_labels = []
        bag_annotations = []
        bag_feature_list = []
        self.log.verbose(f"READING deep feature bags for layer '{self.cfg.layer}'")
        clip = self.cfg.clip
        if self.verbose:
            bagitor = tqdm.tqdm(clip.bags, desc="Reading bag features")
        else:
            bagitor = clip.bags
        for bag in bagitor:
            bag_labels.append(_extract_bag_label(bag, self.cfg.annotation_key))
            bag_annotations.append(_extract_bag_annotations(bag))
            layer_features = bag.layer_features(self.cfg.layer).float()
            layer_features = normalize_features(layer_features, self.cfg.normalize)
            if self.cfg.aggregation == "mean":
                _bag_features = torch.mean(layer_features, dim=0)
            bag_feature_list.append(_bag_features)
        bag_features = torch.stack(bag_feature_list)
        assert len(bag_labels) == len(bag_features), \
            f"len(bag_labels) != len(bag_features): {len(bag_labels)} != {len(bag_features)}"
        write_npz(self.path('bag_labels', ensure_dirpath=True),
                  bag_labels=bag_labels)
        write_pickle(bag_annotations,
                     self.path('bag_annotations', ensure_dirpath=True))
        write_tensor(bag_features,
                     self.path('bag_features', ensure_dirpath=True))

        # ---- train / test split -------------------------------------------
        X = bag_features.numpy()
        y = np.array(bag_labels)
        N = len(y)
        ntrain = int(N * self.cfg.evaluation_fraction)
        perm = np.random.permutation(N)
        X_train, y_train = X[perm[:ntrain]], y[perm[:ntrain]]
        X_test,  y_test  = X[perm[ntrain:]], y[perm[ntrain:]]

        # ---- fit classifier -----------------------------------------------
        self.log.verbose(f"FITTING LogisticRegression "
                         f"(fit_intercept={self.cfg.fit_intercept}, "
                         f"layer='{self.cfg.layer}', "
                         f"annotation_key={self.cfg.annotation_key!r}, "
                         f"normalize={self.cfg.normalize!r})")
        clf = LogisticRegression(fit_intercept=self.cfg.fit_intercept)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        report = classification_report(y_test, y_pred)
        self.log.verbose(f"Classification report:\n{report}")

        # ---- persist results ----------------------------------------------
        write_pickle(report,
                     self.path('evaluation_report', ensure_dirpath=True))
        write_tensor(torch.from_numpy(clf.coef_),
                     self.path('coef', ensure_dirpath=True))
        write_tensor(torch.from_numpy(clf.intercept_),
                     self.path('intercept', ensure_dirpath=True))
        write_npz(self.path('classes', ensure_dirpath=True),
                  classes=clf.classes_)

        self.log.verbose(
            f"intercept norm = {np.linalg.norm(clf.intercept_):.6f}  "
            f"(fit_intercept={self.cfg.fit_intercept})"
        )
        return self

    def __read__(self, topic):
        if topic == 'bag_labels':
            result = read_npz(self.path('bag_labels'), 'bag_labels')['bag_labels']
        elif topic == 'bag_annotations':
            result = read_pickle(self.path('bag_annotations'))
        elif topic == 'bag_features':
            result = read_tensor(self.path('bag_features'))
        elif topic == 'evaluation_report':
            result = read_pickle(self.path('evaluation_report'))
        elif topic == 'coef':
            result = read_tensor(self.path('coef'))
        elif topic == 'intercept':
            result = read_tensor(self.path('intercept'))
        elif topic == 'classes':
            result = read_npz(self.path('classes'), 'classes')['classes']
        else:
            raise ValueError(f"Unknown topic: {topic}")
        return result

    def asphericity(self):
        """Per-class ratio ``‖intercept‖ / ‖coef‖``.

        A small ratio means the linear decision boundary nearly passes
        through the origin, indicating the feature cloud is roughly
        centred — consistent with an isotropic latent space.

        Returns
        -------
        dict
            ``{class_label: ratio}`` for each class.
        """
        coef = self.read('coef')            # (n_classes, n_features)
        intercept = self.read('intercept')  # (n_classes,)
        classes = self.read('classes')      # (n_classes,)
        coef_norms = coef.norm(dim=1)       # (n_classes,)
        ratios = intercept.abs() / coef_norms
        return {str(c): float(r) for c, r in zip(classes, ratios)}

# ═══════════════════════════════════════════════════════════════════════
#  Deep Feature Stats Probe
# ═══════════════════════════════════════════════════════════════════════

class DeepFeatureStatsProbe(Datablock):
    """Per-layer tile/bag statistics for a :class:`DeepFeatureClip`.

    Computes tile-level and bag-level statistics — mean, std, median,
    min, max, L2 norms, and distinct-tile counts — for a single
    capture layer across all bags in a clip.

    An optional ``normalize`` config controls feature pre-processing:

    - ``None`` — use raw features
    - ``'l2'`` — L2-normalise (project onto unit sphere)
    - ``'corner-l1'`` / ``'corner-l2'`` — snap to nearest hypercube vertex (``sign(x)``)
    - ``'corner-linfty'`` — snap to axis-aligned vertex (largest-magnitude coordinate)

    Subsumes the functionality of
    :class:`~autopath.probes.FeatureBagMedianProbe`.
    """

    TOPICFILES = {
        'tile_count':           'tile_count.npz',
        'tile_feature_mean':    'tile_feature_mean.npz',
        'tile_feature_std':     'tile_feature_std.npz',
        'tile_feature_median':  'tile_feature_median.npz',
        'tile_feature_min':     'tile_feature_min.npz',
        'tile_feature_max':     'tile_feature_max.npz',
        'tile_feature_norms':   'tile_feature_norms.npz',
        'bag_feature_mean':     'bag_feature_mean.npz',
        'bag_feature_std':      'bag_feature_std.npz',
        'distinct_tile_count':  'distinct_tile_count.npz',
    }

    @dataclass
    class CONFIG:
        clip: object   # DeepFeatureClip
        layer: str     # which capture key to compute stats for
        normalize: str | None = None   # None, 'l2', 'corner-l1', 'corner-l2', 'corner-linfty'

    def __post_init__(self):
        assert self.cfg.normalize in _NORMALIZE_MODES, \
            f"Unknown normalize mode: {self.cfg.normalize!r}"
        return self

    def __build__(self):
        self.log.verbose(f"COMPUTING stats for layer '{self.cfg.layer}' "
                         f"(normalize={self.cfg.normalize!r}): BEGIN")
        clip = self.cfg.clip

        tile_feature_list = []
        bag_feature_list = []

        if self.verbose:
            bagitor = tqdm.tqdm(clip.bags)
        else:
            bagitor = clip.bags

        for bag in bagitor:
            layer_features = bag.layer_features(self.cfg.layer).float()
            layer_features = normalize_features(layer_features, self.cfg.normalize)
            tile_feature_list.append(layer_features)
            bag_feature_list.append(layer_features.mean(dim=0))

        # Concatenate all tiles
        all_tiles = torch.cat(tile_feature_list, dim=0).numpy()
        del tile_feature_list
        gc.collect()

        # Tile-level stats
        tile_count = np.array(len(all_tiles))
        tile_feature_mean = np.mean(all_tiles, axis=0)
        tile_feature_std = np.std(all_tiles, axis=0)
        tile_feature_median = np.median(all_tiles, axis=0)
        tile_feature_min = np.min(all_tiles, axis=0)
        tile_feature_max = np.max(all_tiles, axis=0)
        tile_feature_norms = np.linalg.norm(all_tiles, axis=1)

        write_npz(self.path('tile_count', ensure_dirpath=True),
                  tile_count=tile_count)
        write_npz(self.path('tile_feature_mean', ensure_dirpath=True),
                  tile_feature_mean=tile_feature_mean)
        write_npz(self.path('tile_feature_std', ensure_dirpath=True),
                  tile_feature_std=tile_feature_std)
        write_npz(self.path('tile_feature_median', ensure_dirpath=True),
                  tile_feature_median=tile_feature_median)
        write_npz(self.path('tile_feature_min', ensure_dirpath=True),
                  tile_feature_min=tile_feature_min)
        write_npz(self.path('tile_feature_max', ensure_dirpath=True),
                  tile_feature_max=tile_feature_max)
        write_npz(self.path('tile_feature_norms', ensure_dirpath=True),
                  tile_feature_norms=tile_feature_norms)

        # Distinct tile count
        distinct_tile_count = np.array(np.unique(all_tiles, axis=0).shape[0])
        write_npz(self.path('distinct_tile_count', ensure_dirpath=True),
                  distinct_tile_count=distinct_tile_count)

        del all_tiles
        gc.collect()

        # Bag-level stats
        bag_features = torch.stack(bag_feature_list).numpy()
        bag_feature_mean = np.mean(bag_features, axis=0)
        bag_feature_std = np.std(bag_features, axis=0)
        write_npz(self.path('bag_feature_mean', ensure_dirpath=True),
                  bag_feature_mean=bag_feature_mean)
        write_npz(self.path('bag_feature_std', ensure_dirpath=True),
                  bag_feature_std=bag_feature_std)

        self.log.verbose(
            f"COMPUTING stats for layer '{self.cfg.layer}': END "
            f"({tile_count} tiles, {distinct_tile_count} distinct, "
            f"{len(bag_features)} bags)"
        )
        return self

    def __read__(self, topic):
        return read_npz(self.path(topic), topic)[topic]

    @functools.cached_property
    def tile_count(self):
        return self.read('tile_count')

    @functools.cached_property
    def tile_feature_mean(self):
        return self.read('tile_feature_mean')

    @functools.cached_property
    def tile_feature_std(self):
        return self.read('tile_feature_std')

    @functools.cached_property
    def tile_feature_median(self):
        return self.read('tile_feature_median')

    @functools.cached_property
    def tile_feature_min(self):
        return self.read('tile_feature_min')

    @functools.cached_property
    def tile_feature_max(self):
        return self.read('tile_feature_max')

    @functools.cached_property
    def tile_feature_norms(self):
        return self.read('tile_feature_norms')


# ═══════════════════════════════════════════════════════════════════════
#  Deep Feature Spectral Prober — model-agnostic Jacobian SVD analysis
# ═══════════════════════════════════════════════════════════════════════

class DeepFeatureSpectralProber:
    """Estimates singular values of layer-to-layer Jacobians dh_{l+1}/dh_l.

    Treats each transformer block as a map F_l : h_l -> h_{l+1} and estimates
    the singular value spectrum of its Jacobian.  Two modes are available:

        * **cls** — materialises the d×d Jacobian restricted to the CLS token
          and computes a full dense SVD.  Fast and gives the complete spectrum.

        * **full** — uses JVP/VJP to define a LinearOperator for the full
          (Nd)×(Nd) Jacobian and calls scipy.sparse.linalg.svds for the top-k
          and bottom-k singular values.  Memory-efficient but slower.

    This class is **model-agnostic**: it accepts blocks as a plain list of
    ``nn.Module`` objects.  Model-specific block unwrapping (e.g. GigaPath's
    chunked ``nn.ModuleList``) is the caller's responsibility.

    Args:
        blocks:         List of ``nn.Module`` transformer blocks, in order.
        probe_blocks:   Which block indices (into *blocks*) to probe.
        k:              Number of extreme singular values to estimate in full mode.
        device:         Torch device string.
    """

    def __init__(
        self,
        blocks: List[torch.nn.Module],
        probe_blocks: List[int],
        k: int = 10,
        device: str = 'cuda',
        log: dbx.Logger = dbx.Logger(name='SpectralProbe', stack_depth=3),
    ):
        self._blocks = list(blocks)
        self.probe_blocks = probe_blocks
        self.k = k
        self.device = device
        self.log = log

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _block_fn(self, block_idx: int):
        """Return a pure function h -> block(h) for a single block."""
        block = self._blocks[block_idx]
        def fn(h):
            return block(h)
        return fn

    # ------------------------------------------------------------------
    #  Mode 1: CLS-token-only  (d × d, fully materialisable)
    # ------------------------------------------------------------------

    def _probe_cls(self, block_idx: int, h: torch.Tensor) -> np.ndarray:
        """Materialise the d×d CLS-token Jacobian and return all singular values.

        Args:
            block_idx:  Index of the block.
            h:          Activation tensor, shape (B, N, d):
                - B is batch size
                - N is number of tokens
                - d is embedding dimension
                Only the CLS token (index 0) of sample 0 from the batch is used.

        Returns:
            Singular values as a 1-D numpy array in decreasing order, length d.
        """
        block = self._blocks[block_idx]
        d = h.shape[-1]
        h0 = h[0:1].detach().clone().requires_grad_(True)          # (1, N, d)

        def cls_fn(cls_vec):
            """Replace the CLS token, run the block, return the output CLS token."""
            h_in = h0.clone()
            h_in[0, 0, :] = cls_vec
            h_out = block(h_in)
            return h_out[0, 0, :]                                  # (d,)

        J = torch.autograd.functional.jacobian(cls_fn, h0[0, 0, :].detach())   # (d, d)
        sv = torch.linalg.svdvals(J.float()).cpu().numpy()
        return sv

    # ------------------------------------------------------------------
    #  Mode 1b: CLS-token composed Jacobian (first → last probed block)
    # ------------------------------------------------------------------

    def _probe_composed_cls(self, first_block: int, last_block: int,
                            h_input: torch.Tensor) -> np.ndarray:
        """Materialise the d×d CLS-token Jacobian of the composed map
        blocks[first] ∘ … ∘ blocks[last] and return all singular values.

        This chains every block from *first_block* to *last_block*
        (inclusive), so it captures the end-to-end transformation over
        the probed depth range.

        Args:
            first_block:  Index of the first block in the chain.
            last_block:   Index of the last block in the chain.
            h_input:      Activation tensor at the **input** of
                          ``first_block``, shape ``(B, N, d)``.
                          Only sample 0 is used.

        Returns:
            Singular values as a 1-D numpy array in decreasing order,
            length *d*.
        """
        d = h_input.shape[-1]
        h0 = h_input[0:1].detach().clone().requires_grad_(True)   # (1, N, d)

        def composed_cls_fn(cls_vec):
            h = h0.clone()
            h[0, 0, :] = cls_vec
            for b_idx in range(first_block, last_block + 1):
                h = self._blocks[b_idx](h)
            return h[0, 0, :]                                     # (d,)

        self.log.verbose(
            f"Computing composed CLS Jacobian for blocks {first_block}→{last_block}"
        )
        J = torch.autograd.functional.jacobian(
            composed_cls_fn, h0[0, 0, :].detach(),
        )                                                         # (d, d)
        sv = torch.linalg.svdvals(J.float()).cpu().numpy()
        return sv

    # ------------------------------------------------------------------
    #  Mode 2: Full-sequence matrix-free  (Nd × Nd, via JVP / VJP)
    # ------------------------------------------------------------------

    def _probe_full(self, block_idx: int, h: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate top-k and bottom-k singular values of the full Jacobian.

        Uses torch.func.jvp / vjp wrapped in a scipy LinearOperator fed to
        scipy.sparse.linalg.svds.

        Args:
            block_idx:  Index of the block.
            h:          Activation tensor, shape (B, N, d).  Only sample 0 is used.

        Returns:
            (s_top, s_bot) — each a 1-D numpy array of length k.
        """
        block = self._blocks[block_idx]
        B, N, d = h.shape
        flat_dim = N * d
        h0 = h[0:1].detach()                                      # (1, N, d)
        dtype = h.dtype
        k = min(self.k, flat_dim - 1)                              # svds requires k < min(m,n)

        def flat_fn(flat_h):
            return block(flat_h.view(1, N, d)).view(flat_dim)

        def matvec(v):
            """J @ v  via JVP."""
            v_t = torch.tensor(v, device=self.device, dtype=dtype).view(flat_dim)
            _, jvp_out = torch.func.jvp(flat_fn, (h0.view(flat_dim),), (v_t,))
            return jvp_out.detach().cpu().numpy().astype(np.float64)

        def rmatvec(u):
            """J^T @ u  via VJP."""
            u_t = torch.tensor(u, device=self.device, dtype=dtype).view(flat_dim)
            _, vjp_fn = torch.func.vjp(flat_fn, h0.view(flat_dim))
            v = vjp_fn(u_t)[0]
            return v.detach().cpu().numpy().astype(np.float64)

        J_op = ScipyLinearOperator(
            shape=(flat_dim, flat_dim),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )

        # Top-k singular values
        try:
            _, s_top, _ = scipy_svds(J_op, k=k, which='LM')
            s_top = np.sort(s_top)[::-1]
        except Exception as e:
            self.log.warning(f"scipy_svds (LM) failed for block {block_idx}: {e}")
            s_top = np.full(k, np.nan)

        # Bottom-k singular values
        try:
            _, s_bot, _ = scipy_svds(J_op, k=k, which='SM')
            s_bot = np.sort(s_bot)
        except Exception as e:
            self.log.warning(f"svds (SM) failed for block {block_idx}: {e}")
            s_bot = np.full(k, np.nan)

        return s_top, s_bot

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def probe(
        self,
        activations: Dict[int, torch.Tensor],
        mode: str = 'cls',
    ) -> Dict[int, dict]:
        """Run the spectral probe on pre-captured activations.

        Args:
            activations:  Mapping from block index to activation tensor (B, N, d),
                          as captured by DeepBackboneEvaluator hooks.
            mode:         ``'cls'`` for CLS-token-only, ``'full'`` for matrix-free,
                          or ``'both'`` to run both.

        Returns:
            Dict mapping block_idx -> result dict with keys depending on mode:
                cls:  {'singular_values': ndarray of shape (d,)}
                full: {'top_singular_values': ndarray, 'bottom_singular_values': ndarray,
                       'condition_number': float}
        """
        results = {}
        for block_idx in self.probe_blocks:
            if block_idx not in activations:
                self.log.warning(f"No activation captured for block {block_idx}, skipping")
                continue

            h = activations[block_idx]
            self.log.debug(f"Probing block {block_idx}, h.shape={tuple(h.shape)}, mode={mode}")
            entry = {}

            if mode in ('cls', 'both'):
                sv = self._probe_cls(block_idx, h)
                entry['singular_values'] = sv
                entry['log_singular_values'] = np.log(np.clip(sv, 1e-12, None))
                entry['condition_number_cls'] = float(sv[0] / (sv[-1] + 1e-12))

            if mode in ('full', 'both'):
                s_top, s_bot = self._probe_full(block_idx, h)
                entry['top_singular_values'] = s_top
                entry['bottom_singular_values'] = s_bot
                entry['condition_number_full'] = float(s_top[0] / (s_bot[0] + 1e-12))

            results[block_idx] = entry
        return results


# Backward-compat alias — dinov2/backbone.py imports this name.
SpectralProbe = DeepFeatureSpectralProber


# ═══════════════════════════════════════════════════════════════════════
#  Deep Backbone Spectral Evaluator — runtime evaluator with probing
# ═══════════════════════════════════════════════════════════════════════

class DeepBackboneSpectralEvaluator(DeepBackboneEvaluator):
    """Abstract :class:`DeepBackboneEvaluator` with spectral probing.

    Adds Jacobian singular-value analysis on top of the multi-layer
    activation capture provided by :class:`DeepBackboneEvaluator`.
    After each ``__call__``, a :class:`DeepFeatureSpectralProber`
    is run on the captured activations.

    Subclasses **must** implement:

    * :attr:`backbone_blocks` — property returning a ``list[nn.Module]``
      of transformer blocks in forward-pass order.  The
      :class:`DeepFeatureSpectralProber` uses these to compute Jacobians
      via autograd.
    * All abstract methods inherited from :class:`DeepBackboneEvaluator`.

    Parameters
    ----------
    spectral_probe_blocks : list[int] | None
        Block indices to probe.  If ``None``, defaults to the
        evaluator's entire ``layer_names`` list filtered for integer
        block keys.
    spectral_mode : str
        ``'cls'``, ``'full'``, or ``'both'``.
    spectral_k : int
        Number of extreme singular values for full mode.
    device : str
        Target device.
    """

    def __init__(
        self,
        *,
        spectral_probe_blocks=None,
        spectral_mode: str = 'cls',
        spectral_k: int = 10,
        device: str = "cuda",
        log: dbx.Logger = dbx.Logger(stack_depth=3),
    ):
        super().__init__(device=device, log=log)
        self.spectral_probe_blocks = spectral_probe_blocks or []
        self.spectral_mode = spectral_mode
        self.spectral_k = spectral_k
        self._spectral_probe = None
        self._spectral_results = None

    @property
    def backbone_blocks(self):
        """Return the ordered list of ``nn.Module`` transformer blocks.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def spectral_probe(self):
        """Lazily create the :class:`DeepFeatureSpectralProber`."""
        if self._spectral_probe is None and self.spectral_probe_blocks:
            self._spectral_probe = DeepFeatureSpectralProber(
                blocks=self.backbone_blocks,
                probe_blocks=self.spectral_probe_blocks,
                k=self.spectral_k,
                device=self.device,
            )
        return self._spectral_probe

    def _run_spectral_probe(self, layer_features):
        """Run spectral probing on captured activations.

        Called automatically after ``__call__``.  Populates
        ``spectral_results``.
        """
        if self.spectral_probe is None:
            return
        # Build activations dict from layer_features: keys like "block.0"
        activations = {}
        for b in self.spectral_probe_blocks:
            key = f"block.{b}"
            if key in layer_features and layer_features[key] is not None:
                activations[b] = layer_features[key].to(self.device)
        if activations:
            self._spectral_results = self.spectral_probe.probe(
                activations, mode=self.spectral_mode,
            )
            # Composed Jacobian (first → last probed block, CLS-only)
            if len(self.spectral_probe_blocks) >= 2:
                first_b = min(self.spectral_probe_blocks)
                last_b = max(self.spectral_probe_blocks)
                input_key = f"block.{first_b}_input"
                if input_key in layer_features and layer_features[input_key] is not None:
                    h_input = layer_features[input_key].to(self.device)
                    sv = self.spectral_probe._probe_composed_cls(
                        first_b, last_b, h_input,
                    )
                    self._spectral_results['composed'] = {
                        'singular_values': sv,
                        'log_singular_values': np.log(np.clip(sv, 1e-12, None)),
                        'condition_number_cls': float(sv[0] / (sv[-1] + 1e-12)),
                    }

    @property
    def spectral_results(self):
        """Spectral probe results from the last forward pass, keyed by block index."""
        return self._spectral_results

    def clear_spectral_results(self):
        self._spectral_results = None
        return self

# Backward-compat alias for autobits.
SpectralDeepBackboneEvaluator = DeepBackboneSpectralEvaluator

# ═══════════════════════════════════════════════════════════════════════
#  Deep Feature Spectral Probe — persistent Datablock wrapper
# ═══════════════════════════════════════════════════════════════════════

class DeepFeatureSpectralProbe(Datablock):
    """Persistent spectral probe for a :class:`DeepFeatureClip`.

    Iterates over bags in a deep feature clip, loads tiles from the
    source :class:`PancanTileBag`, runs a live forward pass through
    the backbone evaluator (with ``requires_grad`` enabled for Jacobian
    computation), and runs the :class:`DeepFeatureSpectralProber` engine
    on each bag's activations.

    Unlike the other deep feature probes (which read persisted features
    from MDS shards), spectral probing **requires live nn.Module blocks**
    for autograd-based Jacobian estimation.  Hence this probe takes an
    ``evaluator_factory`` and loads the model at build time.

    Labels and annotations are extracted for each bag using the same
    mechanism as :class:`DeepFeatureAffineLogisticProbe`.

    Persisted topics
    ----------------
    - ``bag_labels``       — per-bag label array
    - ``bag_annotations``  — per-bag annotation dicts
    - ``spectral_results`` — pickled dict: ``{bag_idx: {block_idx: result_dict}}``
    - ``summary``          — pickled per-block aggregate summary
    """

    TOPICFILES = {
        'bag_labels':       'bag_labels.npz',
        'bag_annotations':  'bag_annotations.pkl',
        'spectral_results': 'spectral_results.pkl',
        'summary':          'summary.pkl',
    }

    @dataclass
    class CONFIG:
        clip: object                   # DeepFeatureClip
        evaluator_factory: object      # DeepBackboneEvaluatorFactory
        probe_blocks: list             # list[int] — transformer block indices to probe
        annotation_key: str | None = None
        mode: str = 'cls'              # 'cls', 'full', or 'both'
        k: int = 10                    # top/bottom-k for full mode
        gpu_batch_size: int = 64       # tiles per forward pass
        n_sample_bags: int | None = None  # probe only first N bags (None = all)

    def __init__(self, *args, device: str = "cuda", **kwargs):
        Datablock.__init__(self, *args, device=device, **kwargs)

    def __post_init__(self):
        assert self.cfg.mode in ('cls', 'full', 'both'), \
            f"Unknown spectral mode: {self.cfg.mode!r}"
        return self

    def __build__(self):
        import math
        clip = self.cfg.clip
        factory = self.cfg.evaluator_factory

        # ── Create evaluator and spectral probe engine ────────────
        # We need a *non-cls-token-only* evaluator for spectral probing
        # because the SpectralProbe needs full (B, N, d) activations.
        evaluator = factory.evaluator(device=self.device, log=self.log)

        # Get the backbone blocks for Jacobian computation.
        # Works for GigaPath evaluators which expose backbone_blocks.
        if hasattr(evaluator, 'backbone_blocks'):
            blocks = evaluator.backbone_blocks
        else:
            from autopath.gigapath.dinov2.backbone import backbone_blocks
            blocks = list(backbone_blocks(evaluator.backbone))

        spectral_engine = DeepFeatureSpectralProber(
            blocks=blocks,
            probe_blocks=self.cfg.probe_blocks,
            k=self.cfg.k,
            device=self.device,
            log=self.log,
        )

        # ── Iterate bags ──────────────────────────────────────────
        bag_labels = []
        bag_annotations_list = []
        all_spectral_results = {}

        n_bags = clip.n_bags
        if self.cfg.n_sample_bags is not None:
            n_bags = min(n_bags, self.cfg.n_sample_bags)

        if self.verbose:
            bagitor = tqdm.tqdm(range(n_bags), desc='spectral-probe')
        else:
            bagitor = range(n_bags)

        for bag_idx in bagitor:
            bag = clip.bag(bag_idx)
            bag_labels.append(_extract_bag_label(bag, self.cfg.annotation_key))
            bag_annotations_list.append(_extract_bag_annotations(bag))

            # Load tiles from the source tilebag and run the evaluator
            # **with grad** so the SpectralProbe can compute Jacobians.
            tiles_tensor = bag.tilebag.tiles
            n_tiles = len(tiles_tensor)

            # Run forward pass in batches, collecting block activations.
            # For spectral probing we only need one representative batch
            # (the Jacobian at a single point), so use the first batch.
            batch_size = min(self.cfg.gpu_batch_size, n_tiles)
            batch = tiles_tensor[:batch_size].to(self.device)

            # Run the evaluator (which registers hooks internally).
            with torch.enable_grad():
                result = evaluator(batch)

            # Build activations dict for the spectral engine.
            activations = {}
            for b in self.cfg.probe_blocks:
                key = f"block.{b}"
                if key in result and result[key] is not None:
                    act = result[key].to(self.device)
                    # SpectralProbe needs 3D (B, N, d); if cls_token_only
                    # reduced to 2D, we can't compute the full Jacobian.
                    if act.dim() == 2:
                        self.log.warning(
                            f"Block {b} activation is 2D (cls_token_only?); "
                            f"spectral probing requires full-sequence 3D tensors. "
                            f"Skipping this block."
                        )
                        continue
                    activations[b] = act

            if activations:
                bag_results = spectral_engine.probe(activations, mode=self.cfg.mode)

                # Composed Jacobian (first → last probed block, CLS-only)
                active_blocks = sorted(activations.keys())
                if len(active_blocks) >= 2:
                    first_b = active_blocks[0]
                    last_b = active_blocks[-1]
                    input_key = f"block.{first_b}_input"
                    if input_key in result and result[input_key] is not None:
                        h_input = result[input_key].to(self.device)
                        sv = spectral_engine._probe_composed_cls(
                            first_b, last_b, h_input,
                        )
                        bag_results['composed'] = {
                            'singular_values': sv,
                            'log_singular_values': np.log(np.clip(sv, 1e-12, None)),
                            'condition_number_cls': float(sv[0] / (sv[-1] + 1e-12)),
                        }

                all_spectral_results[bag_idx] = bag_results

            # Cleanup
            evaluator.clear()
            del batch, result, tiles_tensor
            gc.collect()
            torch.cuda.empty_cache()

        # ── Persist labels and annotations ────────────────────────
        write_npz(self.path('bag_labels', ensure_dirpath=True),
                  bag_labels=bag_labels)
        write_pickle(bag_annotations_list,
                     self.path('bag_annotations', ensure_dirpath=True))

        # ── Persist spectral results ──────────────────────────────
        write_pickle(all_spectral_results,
                     self.path('spectral_results', ensure_dirpath=True))

        # ── Compute and persist summary ───────────────────────────
        summary = self._compute_summary(all_spectral_results)
        write_pickle(summary,
                     self.path('summary', ensure_dirpath=True))

        self.log.verbose(
            f"DeepFeatureSpectralProbe complete: "
            f"{len(all_spectral_results)}/{n_bags} bags probed, "
            f"mode={self.cfg.mode!r}, "
            f"probe_blocks={self.cfg.probe_blocks}"
        )
        return self

    @staticmethod
    def _compute_summary(all_results: dict) -> dict:
        """Aggregate per-bag spectral results into a per-block summary.

        Returns
        -------
        dict
            ``{block_idx: {'mean_sv': ndarray, 'std_sv': ndarray,
            'mean_condition_number': float, ...}}``
        """
        from collections import defaultdict
        block_svs = defaultdict(list)
        block_conds = defaultdict(list)

        for bag_idx, bag_result in all_results.items():
            for block_idx, entry in bag_result.items():
                if block_idx == 'composed':
                    continue
                if 'singular_values' in entry:
                    block_svs[block_idx].append(entry['singular_values'])
                if 'condition_number_cls' in entry:
                    block_conds[block_idx].append(entry['condition_number_cls'])

        summary = {}
        for block_idx in sorted(block_svs.keys()):
            svs = np.array(block_svs[block_idx])
            summary[block_idx] = {
                'mean_sv': np.mean(svs, axis=0),
                'std_sv': np.std(svs, axis=0),
                'median_sv': np.median(svs, axis=0),
                'n_bags': len(svs),
            }
            if block_conds[block_idx]:
                conds = np.array(block_conds[block_idx])
                summary[block_idx]['mean_condition_number'] = float(np.mean(conds))
                summary[block_idx]['std_condition_number'] = float(np.std(conds))
        return summary

    def __read__(self, topic):
        if topic == 'bag_labels':
            return read_npz(self.path('bag_labels'), 'bag_labels')['bag_labels']
        elif topic == 'bag_annotations':
            return read_pickle(self.path('bag_annotations'))
        elif topic == 'spectral_results':
            return read_pickle(self.path('spectral_results'))
        elif topic == 'summary':
            return read_pickle(self.path('summary'))
        else:
            raise ValueError(f"Unknown topic: {topic!r}")

    @functools.cached_property
    def spectral_results(self):
        """Per-bag spectral results: ``{bag_idx: {block_idx: result_dict}}``."""
        return self.read('spectral_results')

    @functools.cached_property
    def summary(self):
        """Per-block aggregate summary: ``{block_idx: summary_dict}``."""
        return self.read('summary')

    @functools.cached_property
    def bag_labels(self):
        return self.read('bag_labels')['bag_labels']
