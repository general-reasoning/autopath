"""Deep feature probes — logistic regression, statistics, and spectral analysis.

Provides :class:`LinearDeepFeatureClipProbe` (per-layer logistic regression),
:class:`StatsDeepFeatureClipProbe` (tile/bag-level statistics), and
:class:`SpectralProbe` (model-agnostic Jacobian SVD analysis) for
deep feature clips and multi-layer activations.
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

from autopath.probes import LogisticFeatureBagProber


class LinearDeepFeatureClipProbe(Datablock, LogisticFeatureBagProber):
    """Logistic regression probe on a single layer of a DeepFeatureClip.

    Collects per-bag mean features for the specified ``layer``, fits a
    :class:`~sklearn.linear_model.LogisticRegression`, and persists the
    model parameters (``coef_``, ``intercept_``, ``classes_``) alongside
    the classification report.

    This is the deep-feature analogue of
    :class:`AffineLogisticFeatureBagProbe`.
    """

    TOPICFILES = {
        'bag_labels':        'bag_labels.npz',
        'bag_features':      'bag_features.npy',
        'evaluation_report': 'evaluation_report.pkl',
        'coef':              'coef.npy',
        'intercept':         'intercept.npy',
        'classes':           'classes.npz',
    }

    @dataclass
    class CONFIG:
        clip: object        # DeepFeatureClip (or SphericalDeepFeatureClip / CornerDeepFeatureClip)
        layer: str          # which capture key to probe (e.g. "B_0_norm1")
        fit_intercept: bool = True
        evaluation_fraction: float = 0.8
        aggregation: str = "mean"

    def __post_init__(self):
        assert self.cfg.aggregation in ["mean"], \
            f"Unknown aggregation: {self.cfg.aggregation}"
        return self

    def __build__(self):
        # ---- collect bag features & labels --------------------------------
        bag_labels = []
        bag_feature_list = []
        self.log.verbose(f"READING deep feature bags for layer '{self.cfg.layer}'")
        clip = self.cfg.clip
        if self.verbose:
            bagitor = tqdm.tqdm(clip.bags)
        else:
            bagitor = clip.bags
        for bag in bagitor:
            bag_labels.append(bag.tilebag.label)
            layer_features = bag.layer_features(self.cfg.layer)
            if self.cfg.aggregation == "mean":
                _bag_features = torch.mean(layer_features.float(), dim=0)
            bag_feature_list.append(_bag_features)
        bag_features = torch.stack(bag_feature_list)
        assert len(bag_labels) == len(bag_features), \
            f"len(bag_labels) != len(bag_features): {len(bag_labels)} != {len(bag_features)}"
        write_npz(self.path('bag_labels', ensure_dirpath=True),
                  bag_labels=bag_labels)
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
                         f"layer='{self.cfg.layer}')")
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
            result = read_npz(self.path('bag_labels'), 'labels')
        elif topic == 'bag_features':
            result = read_tensor(self.path('bag_features'))
        elif topic == 'evaluation_report':
            result = read_pickle(self.path('evaluation_report'))
        elif topic == 'coef':
            result = read_tensor(self.path('coef'))
        elif topic == 'intercept':
            result = read_tensor(self.path('intercept'))
        elif topic == 'classes':
            result = read_npz(self.path('classes'), 'classes')
        else:
            raise ValueError(f"Unknown topic: {topic}")
        return result


class StatsDeepFeatureClipProbe(Datablock):
    """Per-layer tile/bag statistics for a :class:`DeepFeatureClip`.

    Computes lightweight statistics — mean, std, median, L2 norms,
    and distinct-tile counts — for a single capture layer across all
    bags in a clip.  This is a streamlined analogue of the statistics
    section in :class:`BipolarFeatureBagProbe`.
    """

    TOPICFILES = {
        'tile_count':           'tile_count.npz',
        'tile_feature_mean':    'tile_feature_mean.npz',
        'tile_feature_std':     'tile_feature_std.npz',
        'tile_feature_median':  'tile_feature_median.npz',
        'tile_feature_norms':   'tile_feature_norms.npz',
        'bag_feature_mean':     'bag_feature_mean.npz',
        'bag_feature_std':      'bag_feature_std.npz',
        'distinct_tile_count':  'distinct_tile_count.npz',
    }

    @dataclass
    class CONFIG:
        clip: object   # DeepFeatureClip
        layer: str     # which capture key to compute stats for

    def __build__(self):
        self.log.verbose(f"COMPUTING stats for layer '{self.cfg.layer}': BEGIN")
        clip = self.cfg.clip

        tile_feature_list = []
        bag_feature_list = []

        if self.verbose:
            bagitor = tqdm.tqdm(clip.bags)
        else:
            bagitor = clip.bags

        for bag in bagitor:
            layer_features = bag.layer_features(self.cfg.layer).float()
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
        tile_feature_norms = np.linalg.norm(all_tiles, axis=1)

        write_npz(self.path('tile_count', ensure_dirpath=True),
                  tile_count=tile_count)
        write_npz(self.path('tile_feature_mean', ensure_dirpath=True),
                  tile_feature_mean=tile_feature_mean)
        write_npz(self.path('tile_feature_std', ensure_dirpath=True),
                  tile_feature_std=tile_feature_std)
        write_npz(self.path('tile_feature_median', ensure_dirpath=True),
                  tile_feature_median=tile_feature_median)
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
        return read_npz(self.path(topic), topic)

    @functools.cached_property
    def tile_count(self):
        return self.read('tile_count')['tile_count']

    @functools.cached_property
    def tile_feature_median(self):
        return self.read('tile_feature_median')['tile_feature_median']

    @functools.cached_property
    def tile_feature_norms(self):
        return self.read('tile_feature_norms')['tile_feature_norms']


# ═══════════════════════════════════════════════════════════════════════
#  Spectral Probe — model-agnostic Jacobian SVD analysis
# ═══════════════════════════════════════════════════════════════════════

class SpectralProbe:
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
