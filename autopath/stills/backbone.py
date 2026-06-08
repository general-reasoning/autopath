"""BitPathConv-specific deep backbone evaluation infrastructure.

Provides :class:`BitPathConvDeepBackboneEvaluator` — which runs the
trained :class:`~autopath.stills.bitpath.BitPathConvNet` and converts
per-dimension logits to ternary predictions ``{-1, 0, +1}`` — and
:class:`BitPathConvDeepBackboneEvaluatorFactory`, a concrete
:class:`~autopath.autobits.DeepBackboneEvaluatorFactory` that references
a trained :class:`~autopath.stills.bitpath.BitPathConvStill` and
participates in the usual spec-hash lineage tracking.

The evaluator is a drop-in replacement for
:class:`~autopath.gigapath.backbone.GigapathDeepBackboneEvaluator`
wherever the consumer only requires ``features_output`` and the
continuous-valued nature of the Gigapath backbone is not central (e.g.
when plugging into :class:`~autopath.deep.features.DeepFeatureClip` or
probes that operate on the ternary feature space).

Tile format expected by the evaluator
--------------------------------------
The same format as the rest of the pipeline:
``(B, H, W, 3)`` uint8 or float, *or* already channel-first
``(B, 3, H, W)``.  The internal :func:`bitpath_tile_transform` handles
both variants by detecting the channel dimension position and casting to
float.  No ImageNet normalisation is applied — the :class:`BitPathConvNet`
was trained directly on raw tile values.
"""

from dataclasses import dataclass
import gc
from typing import Dict, List

import torch

import dbx
from dbx import Logger

from autopath.autobits import (
    DeepBackboneEvaluator,
    DeepBackboneEvaluatorFactory,
)

from autopath.stills.bitpath import BitPathConvNet


# ─── Tile preprocessing ────────────────────────────────────────────────

def bitpath_tile_transform(x: torch.Tensor) -> torch.Tensor:
    """Minimal preprocessing: permute HWC → CHW and cast to float.

    The :class:`BitPathConvNet` expects ``(B, 3, H, W)`` float tensors
    in the raw pixel value range (no ImageNet normalisation).

    Parameters
    ----------
    x : Tensor
        Tile batch of shape ``(B, H, W, 3)`` (channel-last, uint8 or
        float) *or* already ``(B, 3, H, W)`` (channel-first).

    Returns
    -------
    Tensor
        Float tensor of shape ``(B, 3, H, W)``.
    """
    if x.ndim == 4 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2).contiguous()
    return x.float()


# ─── Evaluator ────────────────────────────────────────────────────────

class BitPathConvDeepBackboneEvaluator(DeepBackboneEvaluator):
    """Evaluator that wraps a trained :class:`BitPathConvNet`.

    On each forward call, runs the network and converts the per-class
    logits to ternary predictions via::

        preds = logits.argmax(dim=1) - 1   # ∈ {-1, 0, +1}, shape (B, D)

    The result is returned as float32 for compatibility with the MDS
    writer used by :class:`~autopath.deep.features.DeepFeatureBag`.

    Parameters
    ----------
    model : BitPathConvNet
        Network with weights already loaded (or random for testing).
    device : str
        Target compute device (``"cuda"`` or ``"cpu"``).
    log : Logger
        Logger instance.
    """

    def __init__(
        self,
        model: BitPathConvNet,
        *,
        device: str = 'cuda',
        log: Logger = Logger(),
    ):
        super().__init__(device=device, log=log)
        self._model = model.eval().to(device)
        self._last_features: Dict[str, torch.Tensor] = {}

    # ── Interface ──────────────────────────────────────────────────────

    @property
    def layer_names(self) -> List[str]:
        """Returns ``['output']`` — the single ternary feature layer."""
        return ['output']

    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run a forward pass and return ternary predictions.

        Parameters
        ----------
        x : Tensor
            Tile batch, shape ``(B, H, W, 3)`` or ``(B, 3, H, W)``.

        Returns
        -------
        dict[str, Tensor]
            ``{'output': Tensor}`` where the tensor has shape
            ``(B, output_dim)`` and dtype float32 with values in
            ``{-1.0, 0.0, +1.0}``.
        """
        x = bitpath_tile_transform(x.to(self.device))
        with torch.no_grad():
            logits = self._model(x)          # (B, n_classes, D)
            preds = (logits.argmax(dim=1) - 1).float()  # (B, D)
        self._last_features = {'output': preds.cpu()}
        return dict(self._last_features)

    @property
    def layer_features(self) -> Dict[str, torch.Tensor]:
        """Most recently computed ternary features (read-only snapshot)."""
        return dict(self._last_features)

    def clear(self):
        """Release cached tensors and free GPU memory."""
        self._last_features.clear()
        gc.collect()
        torch.cuda.empty_cache()
        return self


# ─── Factory ──────────────────────────────────────────────────────────

class BitPathConvDeepBackboneEvaluatorFactory(DeepBackboneEvaluatorFactory):
    """Factory that loads a trained :class:`BitPathConvNet` from a still.

    Participates in spec-based hash / lineage tracking in exactly the
    same way as
    :class:`~autopath.gigapath.backbone.GigapathDeepBackboneEvaluatorFactory`.
    The factory's hash is derived from the still's hash, so any change
    to the training configuration or checkpoint is automatically
    reflected in the hashes of downstream
    :class:`~autopath.deep.features.DeepFeatureBag` and
    :class:`~autopath.deep.features.DeepFeatureClip` instances.

    Usage
    -----
    ::

        from autopath.stills.backbone import BitPathConvDeepBackboneEvaluatorFactory

        factory = BitPathConvDeepBackboneEvaluatorFactory(
            still=dbx.quote(my_bitpath_still),
        )
        clip = deep_feature_clip(
            tilebagclip=tile_clip,
            evaluator_factory=dbx.quote(factory),
            ...
        )

    Parameters
    ----------
    still : BitPathConvStill
        Quoted reference to a trained still.  Its latest local checkpoint
        is loaded when :meth:`evaluator` is first called.
    """

    @dataclass
    class CONFIG:
        still: object  # BitPathConvStill (quoted)

    # ── Public API ─────────────────────────────────────────────────────

    def evaluator(
        self,
        *,
        device: str = 'cuda',
        log: Logger = None,
    ) -> BitPathConvDeepBackboneEvaluator:
        """Return a live :class:`BitPathConvDeepBackboneEvaluator`.

        The model is built from the still's architecture config and
        weights are loaded from its latest local checkpoint on the first
        call.  The result is cached per device so that repeated calls
        with the same *device* return the same instance.

        Parameters
        ----------
        device : str
            Target compute device.
        log : Logger | None
            Logger; falls back to ``self.log`` when ``None``.
        """
        if not hasattr(self, '_evaluators'):
            self._evaluators = {}
        if device not in self._evaluators:
            log = log or self.log
            self._evaluators[device] = self._build_evaluator(device=device, log=log)
        return self._evaluators[device]

    # ── Private helpers ────────────────────────────────────────────────

    def _build_evaluator(
        self,
        device: str,
        log: Logger,
    ) -> BitPathConvDeepBackboneEvaluator:
        """Instantiate model, load checkpoint, return wrapped evaluator."""
        still = self.cfg.still
        lm_cfg = still.cfg.lightning.cfg

        # Build a fresh BitPathConvNet with the correct architecture.
        model = BitPathConvNet(
            n_blocks=lm_cfg.n_blocks,
            hidden_channels=lm_cfg.hidden_channels,
            output_dim=lm_cfg.output_dim,
            n_classes=lm_cfg.n_classes,
            activation_bits=lm_cfg.activation_bits,
        )

        # Load weights from the latest available checkpoint.
        ckpt_path = still.ckpt()
        if ckpt_path is not None:
            log.info(
                f'BitPathConvDeepBackboneEvaluatorFactory: '
                f'loading checkpoint {ckpt_path}'
            )
            state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Lightning saves the full LightningModule state_dict with a
            # 'model.' prefix corresponding to the `self.model` attribute
            # of BitPathConvLightning.Lightning.
            model_state = {
                k[len('model.'):]: v
                for k, v in state['state_dict'].items()
                if k.startswith('model.')
            }
            model.load_state_dict(model_state)
            log.info('BitPathConvDeepBackboneEvaluatorFactory: checkpoint loaded')
        else:
            log.warning(
                'BitPathConvDeepBackboneEvaluatorFactory: no checkpoint found '
                'for the provided still — evaluator will use random weights'
            )

        return BitPathConvDeepBackboneEvaluator(model=model, device=device, log=log)
