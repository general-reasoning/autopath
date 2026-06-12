"""BitConv-specific deep backbone evaluation infrastructure.

Provides :class:`BitConvDeepBackboneEvaluator` — which runs the
trained :class:`~autopath.stills.bitpath.BitPathConvNet` and converts
per-dimension logits to ternary predictions ``{-1, 0, +1}`` — and
:class:`BitConvDeepBackboneEvaluatorFactory`, a concrete
:class:`~autopath.autobits.DeepBackboneEvaluatorFactory` that references
a trained :class:`~autopath.stills.bitpath.BitPathConvStill` and
participates in the usual spec-hash lineage tracking.

The evaluator is a drop-in replacement for
:class:`~autopath.gigapath.backbone.GigapathDeepBackboneEvaluator`
wherever the consumer only requires ``features_output`` and the
continuous-valued nature of the Gigapath backbone is not central (e.g.
when plugging into :class:`~autopath.deep.features.DeepFeatureClip` or
probes that operate on the ternary feature space).

Architecture notes
------------------
:class:`~autopath.stills.bitpath.BitPathConvNet` uses **simulated
BitNet-1.58b quantization**: weights are stored in full float32 (shadow
weights) and quantized to ``{-1, 0, +1} × α`` on every forward pass via
absmean quantization with STE gradients.  Activations are similarly
quantized to 8-bit range before each BitConv2d / BitLinear layer.
All arithmetic is therefore float32 (no hardware int8/int4 intrinsics).
Because the network is small relative to the GigaPath ViT and does not
benefit from GPU integer acceleration, the default compute device is
``"cpu"``.

Tile format expected by the evaluator
--------------------------------------
``(B, H, W, 3)`` uint8 or float (channel-last), *or* already
``(B, 3, H, W)`` (channel-first).  The internal
:func:`bitconv_tile_transform` handles both variants by detecting the
channel dimension position and casting to float.  No ImageNet
normalisation is applied — the model was trained directly on raw tile
values in [0, 255].
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

from autopath.stills.bitpath import (
    BitPathConvNet,
    _weight_quant_absmean,
)


# ─── Tile preprocessing ────────────────────────────────────────────────

def bitconv_tile_transform(x: torch.Tensor) -> torch.Tensor:
    """Minimal preprocessing: permute HWC → CHW and cast to float.

    The :class:`~autopath.stills.bitpath.BitPathConvNet` expects
    ``(B, 3, H, W)`` float tensors in the raw pixel-value range
    (no ImageNet normalisation).

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


# keep old name as alias for any code that already imported it
bitpath_tile_transform = bitconv_tile_transform


# ─── Evaluator ────────────────────────────────────────────────────────

class BitConvDeepBackboneEvaluator(DeepBackboneEvaluator):
    """Evaluator that wraps a trained :class:`~autopath.stills.bitpath.BitPathConvNet`.

    On each forward call, runs the network and converts the per-class
    logits to ternary predictions via::

        preds = logits.argmax(dim=1) - 1   # ∈ {-1, 0, +1}, shape (B, D)

    The result is returned as float32 for compatibility with the MDS
    writer used by :class:`~autopath.deep.features.DeepFeatureBag`.

    Default device is ``"cpu"`` because the network uses simulated
    float32 quantization (not hardware integer ops) and is small enough
    that CPU throughput is acceptable.

    Parameters
    ----------
    model : BitPathConvNet
        Network with weights already loaded (or random for testing).
    device : str
        Target compute device.  Defaults to ``"cpu"``.
    log : Logger
        Logger instance.
    """

    def __init__(
        self,
        model: BitPathConvNet,
        *,
        device: str = 'cpu',
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
        x = bitconv_tile_transform(x.to(self.device))
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
        """Release cached tensors."""
        self._last_features.clear()
        gc.collect()
        return self


# ─── Factory ──────────────────────────────────────────────────────────

class BitConvDeepBackboneEvaluatorFactory(DeepBackboneEvaluatorFactory):
    """Factory that loads a trained :class:`~autopath.stills.bitpath.BitPathConvNet` from a still.

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

        from autopath.stills.backbone import BitConvDeepBackboneEvaluatorFactory
        from autopath.stills.pipelines import bitconv_deep_feature_clip

        # Low-level
        factory = BitConvDeepBackboneEvaluatorFactory(
            still=dbx.quote(my_bitpath_still),
        )
        clip = deep_feature_clip(
            tilebagclip=...,
            evaluator_factory=dbx.quote(factory),
            ...,
        )

        # Via pipeline entrypoint (recommended)
        clip = bitconv_deep_feature_clip('BITCONV_DEEP_CPTAC_602020_TRAIN', still=my_still)

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

    @property
    def layer_names(self) -> list:
        """Layer names produced by this evaluator factory.

        Returns ``['output']`` without loading the model, allowing
        :meth:`~autopath.deep.features.DeepFeatureBag.__post_init__`
        to determine the MDS column schema cheaply.
        """
        return ['output']

    def evaluator(
        self,
        *,
        device: str = 'cpu',
        log: Logger = None,
    ) -> BitConvDeepBackboneEvaluator:
        """Return a live :class:`BitConvDeepBackboneEvaluator`.

        The model is built from the still's architecture config and
        weights are loaded from its latest local checkpoint on the first
        call.  The result is cached per device so that repeated calls
        with the same *device* return the same instance.

        Parameters
        ----------
        device : str
            Target compute device.  Defaults to ``"cpu"``.
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
    ) -> BitConvDeepBackboneEvaluator:
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
                f'BitConvDeepBackboneEvaluatorFactory: '
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
            log.info('BitConvDeepBackboneEvaluatorFactory: checkpoint loaded')
        else:
            raise FileNotFoundError(
                f'BitConvDeepBackboneEvaluatorFactory: no checkpoint found '
                f'for still {still!r} '
                f'(looked in {still._local_ckpts_dir!r}). '
                f"Run the still's build() before evaluating."
            )

        return BitConvDeepBackboneEvaluator(model=model, device=device, log=log)


# ── Backward-compatibility aliases ────────────────────────────────────
# These were the names used before the rename; kept so that any
# serialised dbx.quote references still resolve.
BitPathConvDeepBackboneEvaluator = BitConvDeepBackboneEvaluator
BitPathConvDeepBackboneEvaluatorFactory = BitConvDeepBackboneEvaluatorFactory
