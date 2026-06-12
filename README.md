# BitPath: BitNet1.58b Distillation of Gigapath

**Goal**: Train a lightweight ternary convolutional network (BitNet1.58b-style)
that predicts bipolar features `{−1, 0, +1}` directly from raw pathology tiles,
bypassing the heavyweight Gigapath backbone at inference time.

---

## 1  High-Level Pipeline

```mermaid
flowchart LR
    subgraph Training Data
        BC["BipolarDeepFeatureClip<br/>.dataset(include_tiles=True)"]
    end

    subgraph BitPathNet
        T["Raw Tile<br/>256×256×3"] --> CONV["BitConv2d158<br/>blocks ×N"]
        CONV --> GAP["Global Avg Pool"]
        GAP --> FC["BitLinear158<br/>→ 3×1536 logits"]
    end

    subgraph Loss
        FC --> CE["Cross-Entropy Loss<br/>per feature dimension"]
        BC --> |"tile_bipolar_features<br/>{−1,+1} → class idx"| CE
    end
```

Each training sample from `BipolarDeepFeatureClip.dataset(include_tiles=True)`
contains:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `tile` | `uint8` ndarray | `(256, 256, 3)` | Raw RGB tile from the slide |
| `tile_bipolar_features` | `int8` ndarray | `(1536,)` | Bipolar features: values in `{−1, +1}` |
| `tile_index` | `int32` | scalar | Tile index within the bag |
| `bag_index` | `int32` | scalar | Which bag this tile belongs to |

The network outputs `(B, 3, 1536)` logits — **3 classes per feature dimension** —
and the loss is per-dimension cross-entropy against the target class index
derived from the bipolar value.

### Target Encoding

```
Bipolar value    Class index
─────────────    ───────────
     −1      →       0
      0      →       1       (bag-level only; tiles are always ±1)
     +1      →       2
```

At tile level the targets are always `{−1, +1}` → classes `{0, 2}`.
The 3-class setup future-proofs for bag-level supervision if desired.

---

## 2  BitNet 1.58b: Ternary Weight Quantization

BitNet 1.58b ([Ma et al., 2024](https://arxiv.org/abs/2402.17764)) constrains
every weight to `{−1, 0, +1}`, reducing multiply-accumulate operations to
additions and subtractions. The name "1.58 bits" comes from
`log₂(3) ≈ 1.585` — each weight carries ~1.58 bits of information.

### 2.1  Absmean Quantization

Given a real-valued weight tensor `W`, the ternary approximation `W̃` is:

```
α = mean(|W|)                          # per-tensor scale factor
W̃ = round(clip(W / α, −1, +1))        # quantize to {−1, 0, +1}
```

```mermaid
flowchart LR
    W["W<br/>(full precision)"] --> ABS["α = mean │W│"]
    W --> SCALE["W / α"]
    SCALE --> CLIP["clip(·, −1, +1)"]
    CLIP --> ROUND["round(·)"]
    ROUND --> WQ["W̃ ∈ {−1, 0, +1}"]
    ABS -.->|"scale factor"| SCALE
```

**Why absmean?**  It's parameter-free, cheap to compute, and adapts to the
weight distribution automatically.  Weights near zero get snapped to `0`
(pruned), while large-magnitude weights keep their sign.

### 2.2  Straight-Through Estimator (STE)

Quantization (`round`, `clip`) is non-differentiable.  During backpropagation
we use the **Straight-Through Estimator**: pretend the quantization function
is the identity for gradient purposes.

```mermaid
flowchart TB
    subgraph Forward Pass
        W1["W (latent)"] --> Q["quantize(W)"] --> WQ1["W̃ (ternary)"]
        WQ1 --> MM["matmul(x, W̃ᵀ)"]
    end

    subgraph Backward Pass
        dL["∂L/∂W̃"] --> |"STE: treat quantize<br/>as identity"| dW["∂L/∂W ≈ ∂L/∂W̃"]
    end
```

In PyTorch this is implemented as:

```python
# Forward: use quantized weights for computation
# Backward: gradients flow through to the latent (full-precision) weights
W_quant = quantize(W)
W_ste = W + (W_quant - W).detach()   # forward = W_quant, backward ∂/∂W = I
output = F.linear(x, W_ste, bias)
```

The optimizer updates `W` (full precision).  At the start of each forward pass,
`W` is re-quantized on the fly.  The latent weights slowly drift under gradient
pressure, and the quantized snapshot changes discretely.

### 2.3  Activation Quantization (Optional)

BitNet 1.58b also quantizes activations to 8-bit before each linear/conv layer
using **absmax** quantization:

```
γ = max(|x|)
Q_b = 2^(b−1)                        # e.g. 128 for 8-bit
x̃ = clip(round(x × Q_b / γ), −Q_b + 1, Q_b − 1)
```

This is applied *after* layer normalisation (RMSNorm in the original paper),
so the activation distribution is well-behaved.

---

## 3  BitConv2d158 — Ternary Convolution Layer

A drop-in replacement for `nn.Conv2d` with BitNet 1.58b quantization.

```mermaid
flowchart TB
    subgraph BitConv2d158
        X["Input x<br/>(B, C_in, H, W)"] --> RMS["RMSNorm<br/>(per-channel)"]
        RMS --> AQ["Activation Quant<br/>(absmax, 8-bit)"]
        AQ --> CONV["F.conv2d(x̃, W̃, ...)"]

        W["Weight W<br/>(C_out, C_in, kH, kW)"] --> WQ["Absmean Quant<br/>→ {−1,0,+1}"]
        WQ --> CONV

        CONV --> SCALE2["× α<br/>(rescale output)"]
        SCALE2 --> OUT["Output<br/>(B, C_out, H', W')"]
    end
```

### Forward pass pseudocode

```python
def forward(self, x):
    # 1. Normalise activations
    x = self.rms_norm(x)

    # 2. Quantize activations (absmax, 8-bit)
    gamma = x.abs().max()
    Qb = 2 ** (self.activation_bits - 1)
    x_quant = (x * Qb / gamma).round().clamp(-Qb + 1, Qb - 1)

    # 3. Quantize weights (absmean → ternary)
    alpha = self.weight.abs().mean()
    w_scaled = self.weight / (alpha + eps)
    w_quant = w_scaled.clamp(-1, 1).round()
    w_ste = self.weight + (w_quant * alpha - self.weight).detach()

    # 4. Convolution with quantized operands
    #    (in practice x_quant × w_ste, so matmuls become adds/subs)
    return F.conv2d(x_quant, w_ste, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)
```

### Key properties

| Property | Value |
|----------|-------|
| Weight values | `{−1, 0, +1} × α` |
| Activation precision | 8-bit symmetric |
| Normalization | RMSNorm (per-channel, before quant) |
| Gradient flow | STE through both weight and activation quantization |
| Memory savings | ~16× vs FP32 weights (1.58 bits per weight) |
| Compute savings | Multiply → add/subtract (ternary matmul) |

---

## 4  BitPathNet — Full Network Architecture

```mermaid
flowchart TB
    INPUT["Input Tile<br/>(B, 3, 256, 256)"] --> STEM

    subgraph STEM ["Stem (full-precision)"]
        S1["Conv2d 3→C, 7×7, stride 2<br/>(B, C, 128, 128)"]
        S1 --> BN0["BatchNorm + ReLU"]
    end

    BN0 --> B1

    subgraph B1 ["BitBlock 1"]
        BC1["BitConv2d158 C→C, 3×3"]
        BC1 --> BN1["BatchNorm + ReLU"]
        BN1 --> BC2["BitConv2d158 C→C, 3×3"]
        BC2 --> BN2["BatchNorm + ReLU + residual"]
    end

    B1 --> |"stride-2 downsample"| B2

    subgraph B2 ["BitBlock 2"]
        BC3["BitConv2d158 C→2C, 3×3"]
        BC3 --> BN3["BatchNorm + ReLU"]
        BN3 --> BC4["BitConv2d158 2C→2C, 3×3"]
        BC4 --> BN4["BatchNorm + ReLU + residual"]
    end

    B2 --> |"stride-2 downsample"| MORE["... BitBlocks 3–N<br/>(channel doubling, spatial halving)"]

    MORE --> GAP["Global Average Pool<br/>(B, C_final)"]
    GAP --> FC["BitLinear158<br/>C_final → 3×1536"]
    FC --> RESHAPE["Reshape → (B, 3, 1536)"]
```

### Design choices

- **Stem is full-precision**: The first conv uses standard `nn.Conv2d` because
  the input (RGB pixels) has very low entropy and quantizing 3-channel inputs
  to ternary weights loses too much information.

- **Residual connections**: Each BitBlock uses a skip connection around the two
  `BitConv2d158` layers.  When channel dimensions change, a 1×1 projection
  adapts the shortcut.

- **No pooling layers**: Spatial reduction is done via stride-2 convolutions
  (integrated into the first conv of each block), preserving information
  better than max/avg pooling.

- **Global average pool → linear head**: After the conv blocks, spatial
  dimensions are collapsed and a single `BitLinear158` layer produces the
  3×1536 output logits.

### Default configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_blocks` | 6 | Number of residual BitBlocks |
| `hidden_channels` | 128 | Base channel width (doubles each block) |
| `output_dim` | 1536 | Gigapath feature dimension |
| `n_classes` | 3 | `{−1, 0, +1}` |
| `activation_bits` | 8 | Activation quantization precision |
| `stem_channels` | `hidden_channels` | Stem output channels |

### Parameter count estimate

With `hidden_channels=128` and `n_blocks=6`:
- Stem: ~18K params
- BitBlocks: channels go 128→128→256→256→512→512→... ≈ 4–8M params
- Head: `C_final × 3×1536` ≈ 4.7M (if C_final=1024)
- **Total**: ~10–15M parameters (all ternary except stem)

---

## 5  Training Process

```mermaid
sequenceDiagram
    participant DL as DataLoader
    participant Net as BitPathNet
    participant Loss as CrossEntropy
    participant Opt as Adam

    loop Each batch
        DL->>Net: tiles (B, 3, 256, 256)
        Note over Net: Forward pass with<br/>STE-quantized weights
        Net->>Loss: logits (B, 3, 1536)
        DL->>Loss: bipolar targets (B, 1536)<br/>mapped to class indices
        Loss->>Opt: dL/dW (via STE)
        Opt->>Net: Update latent weights W<br/>(full precision)
    end
```

### Loss function

Per-dimension cross-entropy, averaged over all 1536 feature dimensions:

```python
# logits: (B, 3, 1536)  — 3-class logits per feature dim
# targets: (B, 1536)    — class indices in {0, 1, 2}
loss = F.cross_entropy(
    logits.permute(0, 2, 1).reshape(-1, 3),  # (B*1536, 3)
    targets.reshape(-1),                      # (B*1536,)
)
```

The target mapping is:
- `tile_bipolar_features == -1` → class `0`
- `tile_bipolar_features ==  0` → class `1`
- `tile_bipolar_features == +1` → class `2`

### Optimizer & scheduler

- **Optimizer**: AdamW with `weight_decay` (default 0.01)
- **LR scheduler**: Cosine annealing with warmup
- **Gradient clipping**: By norm (default 1.0)
- **Precision**: `bf16-mixed` for speed on A100/H100

### Multi-GPU training

Lightning `Trainer` with `strategy='ddp'` when `n_devices > 1`.
The `BitPathStill` constructor accepts `n_devices` or `devices` list.

---

## 6  Datablock Hierarchy

```mermaid
classDiagram
    class BitPathStill {
        CONFIG
        +lightning: BitPathLightning
        +dataloader: BitPathDataloaderBuilder
        +max_epochs: int
        +max_steps: int
        +gradient_clip_val: float
        +ckpt_every_n_steps: int
        __build__()
        ckpt()
        valid()
    }

    class BitPathLightning {
        CONFIG
        +n_blocks: int
        +hidden_channels: int
        +output_dim: int
        +n_classes: int
        +activation_bits: int
        +learning_rate: float
        +scheduler: str
        +weight_decay: float
        lightning_module
    }

    class BitPathDataloaderBuilder {
        CONFIG
        +clip: BipolarDeepFeatureClip
        +batch_size: int
        +shuffle: bool
        +seed: int?
        +num_workers: int
        dataloader()
    }

    class BipolarDeepFeatureClip {
        dataset(include_tiles=True)
    }

    BitPathStill --> BitPathLightning : cfg.lightning
    BitPathStill --> BitPathDataloaderBuilder : cfg.dataloader
    BitPathDataloaderBuilder --> BipolarDeepFeatureClip : cfg.clip
```

### CONFIG boundaries

Only parameters that affect the **output** of training go in CONFIG
(so the Datablock hash changes when results would change):

| Datablock | CONFIG params | Non-CONFIG params |
|-----------|--------------|-------------------|
| `BitPathDataloaderBuilder` | `clip`, `batch_size`, `shuffle`, `seed`, `num_workers` | — |
| `BitPathLightning` | `n_blocks`, `hidden_channels`, `output_dim`, `n_classes`, `activation_bits`, `learning_rate`, `scheduler`, `weight_decay` | — |
| `BitPathStill` | `lightning`, `dataloader`, `max_epochs`, `max_steps`, `gradient_clip_val`, `ckpt_every_n_steps`, `precision` | `n_devices`, `devices`, `logsroot` |

---

## 7  Pipeline: `bitpath_still()`

```python
from autopath.deep.pipelines import bitpath_still

still = bitpath_still(
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
    cfg_layer='output',
    cfg_cls_token_only=True,
    cfg_shard_size=64,
    cfg_batch_size=64,
    cfg_n_blocks=6,
    cfg_hidden_channels=128,
    cfg_learning_rate=1e-3,
    max_steps=10000,
    n_devices=4,
)
still.build_tree()
```

The pipeline:
1. Constructs a `BipolarDeepFeatureClip` (via `bipolar_deep_feature_clip()`)
2. Wraps it in a `BitPathDataloaderBuilder`
3. Creates a `BitPathLightning` with architecture params
4. Assembles `BitPathStill` with the lightning + dataloader + training params
5. Returns the still, ready for `.build_tree()`

---

## 8  Inference (Post-Training)

After training, the model's ternary weights can be extracted and stored
compactly.  Inference on a new tile:

```python
still = bitpath_still('GIGAPATH_DEEP_CPTAC_602020_TRAIN', ...)
model = still.cfg.lightning.lightning_module
model.eval()

tile = torch.randn(1, 3, 256, 256)  # preprocessed tile
logits = model(tile)                  # (1, 3, 1536)
predicted_bipolar = logits.argmax(dim=1) - 1  # class {0,1,2} → {-1,0,+1}
```

The predicted bipolar features should approximate the Gigapath-derived
bipolar features but computed ~100–1000× faster (no ViT backbone, only
ternary convolutions).

---

## Results

### June 12, 2026 — AffineLogisticProbe on CPTAC 60/20/20

All probes use `cfg_layer='output'`, `cfg_cls_token_only=True`, `cfg_shard_size=64` on the CPTAC 60/20/20 split.  Evaluation is on the 20% test fold (n=444 slides, 10 cancer types).

**Asphericity** measures how elongated per-class feature clusters are.  Values near 0 = spherical; near 1 = highly elongated.

---

#### Gigapath real-valued features (`gigapath_deep_feature_affine_logistic_probe`)

| Normalization | Accuracy | Macro F1 | Weighted F1 | Mean Asphericity |
|---|---|---|---|---|
| none | **0.97** | 0.93 | 0.97 | 0.020 |
| `corner-l1` | **0.97** | **0.96** | **0.97** | **0.015** |
| `l2` | 0.90 | 0.88 | 0.89 | 0.087 |
| `corner-linfty` | 0.56 | 0.36 | 0.53 | 0.140 |

<details>
<summary>Per-class — gigapath, no normalization (accuracy 0.97)</summary>

BRCA–OV rows not captured (output truncated at collection time).
Known: PDA P=0.98/R=0.98/F1=0.98 (n=49), UCEC P=1.00/R=0.97/F1=0.99 (n=68). Macro F1=0.93, weighted F1=0.97.

Asphericity: BRCA=0.021, CCRCC=0.018, COAD=0.028, GBM=0.025, HNSCC=0.026, LSCC=0.019, LUAD=0.012, OV=0.021, PDA=0.005, UCEC=0.022.
</details>

<details>
<summary>Per-class — gigapath, corner-l1 (accuracy 0.97) ← best normalization</summary>

| Cancer type | Precision | Recall | F1 | n |
|---|---|---|---|---|
| BRCA | 1.00 | 0.85 | 0.92 | 13 |
| CCRCC | 1.00 | 1.00 | 1.00 | 60 |
| COAD | 1.00 | 1.00 | 1.00 | 21 |
| GBM | 0.98 | 1.00 | 0.99 | 57 |
| HNSCC | 0.79 | 0.92 | 0.85 | 12 |
| LSCC | 0.95 | 0.95 | 0.95 | 76 |
| LUAD | 0.97 | 0.96 | 0.97 | 72 |
| OV | 0.83 | 1.00 | 0.91 | 5 |
| PDA | 0.98 | 1.00 | 0.99 | 51 |
| UCEC | 1.00 | 0.97 | 0.99 | 77 |

Asphericity: BRCA=0.015, CCRCC=0.018, COAD=0.019, GBM=0.021, HNSCC=0.023, LSCC=0.016, LUAD=0.005, OV=0.015, PDA=0.005, UCEC=0.014.
</details>

<details>
<summary>Per-class — gigapath, l2 (accuracy 0.90)</summary>

| Cancer type | Precision | Recall | F1 | n |
|---|---|---|---|---|
| BRCA | 1.00 | 1.00 | 1.00 | 13 |
| CCRCC | 0.95 | 0.94 | 0.94 | 63 |
| COAD | 0.90 | 1.00 | 0.95 | 9 |
| GBM | 1.00 | 1.00 | 1.00 | 60 |
| **HNSCC** | 1.00 | **0.27** | **0.42** | 15 |
| LSCC | 0.75 | 0.89 | 0.81 | 74 |
| LUAD | 0.87 | 0.88 | 0.88 | 78 |
| OV | 1.00 | 0.83 | 0.91 | 6 |
| PDA | 0.94 | 0.94 | 0.94 | 47 |
| UCEC | 0.92 | 0.89 | 0.90 | 79 |

Asphericity: BRCA=0.111, CCRCC=0.078, COAD=0.127, GBM=0.094, HNSCC=0.109, LSCC=0.070, LUAD=0.011, OV=0.202, PDA=0.002, UCEC=0.066.
L2 collapses HNSCC recall to 0.27 and raises mean asphericity from 0.020 to 0.087.
</details>

<details>
<summary>Per-class — gigapath, corner-linfty (accuracy 0.56) ← degenerate</summary>

| Cancer type | Precision | Recall | F1 | n |
|---|---|---|---|---|
| **BRCA** | **0.00** | **0.00** | **0.00** | 14 |
| CCRCC | 0.93 | 0.71 | 0.81 | 56 |
| **COAD** | **0.00** | **0.00** | **0.00** | 16 |
| GBM | 1.00 | 0.48 | 0.65 | 58 |
| **HNSCC** | **0.00** | **0.00** | **0.00** | 13 |
| LSCC | 0.38 | 0.76 | 0.51 | 79 |
| LUAD | 0.50 | 0.71 | 0.58 | 80 |
| **OV** | **0.00** | **0.00** | **0.00** | 9 |
| PDA | 0.71 | 0.32 | 0.44 | 47 |
| UCEC | 0.61 | 0.67 | 0.64 | 72 |

Asphericity: BRCA=0.180, CCRCC=0.052, COAD=0.137, GBM=0.083, HNSCC=0.260, LSCC=0.105, LUAD=0.075, OV=0.371, PDA=0.034, UCEC=0.099.
4 of 10 cancer types collapse to zero. `corner-linfty` is degenerate on this feature space.
</details>

---

#### Gigapath bipolar features (`gigapath_bipolar_deep_feature_affine_logistic_probe`)

| Variant | Accuracy | Macro F1 | Weighted F1 | Mean Asphericity |
|---|---|---|---|---|
| none | **0.97** | **0.96** | **0.97** | 0.436 |
| `ternarize_tiles=True` | **0.97** | 0.94 | **0.97** | 0.390 |
| `normalize='l2'` | 0.89 | 0.77 | 0.88 | 0.095 |

<details>
<summary>Per-class — bipolar, no normalization (accuracy 0.97)</summary>

| Cancer type | Precision | Recall | F1 | n |
|---|---|---|---|---|
| BRCA | 0.94 | 0.88 | 0.91 | 17 |
| CCRCC | 0.96 | 1.00 | 0.98 | 44 |
| COAD | 0.95 | 1.00 | 0.98 | 21 |
| GBM | 1.00 | 1.00 | 1.00 | 60 |
| HNSCC | 0.91 | 0.91 | 0.91 | 11 |
| LSCC | 0.95 | 0.98 | 0.96 | 91 |
| LUAD | 0.99 | 0.95 | 0.97 | 87 |
| OV | 1.00 | 0.80 | 0.89 | 5 |
| PDA | 1.00 | 1.00 | 1.00 | 38 |
| UCEC | 0.99 | 0.97 | 0.98 | 70 |

Asphericity: BRCA=0.554, CCRCC=0.243, COAD=0.582, GBM=0.216, HNSCC=0.510, LSCC=0.689, LUAD=0.665, OV=0.594, PDA=0.261, UCEC=0.045.
High asphericity is expected for ternary/bipolar features (hypercube corners). 0.97 accuracy confirms the discrete structure is highly linearly separable.
</details>

<details>
<summary>Per-class — bipolar, ternarize_tiles=True (accuracy 0.97)</summary>

| Cancer type | Precision | Recall | F1 | n |
|---|---|---|---|---|
| BRCA | 0.95 | 0.95 | 0.95 | 19 |
| CCRCC | 1.00 | 1.00 | 1.00 | 59 |
| COAD | 1.00 | 1.00 | 1.00 | 19 |
| GBM | 1.00 | 1.00 | 1.00 | 49 |
| HNSCC | 0.78 | 0.64 | 0.70 | 11 |
| LSCC | 0.93 | 0.94 | 0.93 | 82 |
| LUAD | 0.99 | 0.98 | 0.98 | 87 |
| OV | 0.89 | 0.89 | 0.89 | 9 |
| PDA | 1.00 | 1.00 | 1.00 | 43 |
| UCEC | 0.96 | 0.98 | 0.97 | 66 |

Asphericity: BRCA=0.512, CCRCC=0.235, COAD=0.506, GBM=0.212, HNSCC=0.451, LSCC=0.618, LUAD=0.548, OV=0.612, PDA=0.166, UCEC=0.039.
Ternarizing input tiles at probe evaluation time preserves 0.97 accuracy and slightly reduces asphericity (0.436 -> 0.390), previewing what to expect from the BitPath distillation pipeline.
</details>

<details>
<summary>Per-class — bipolar, normalize=l2 (accuracy 0.89)</summary>

| Cancer type | Precision | Recall | F1 | n |
|---|---|---|---|---|
| BRCA | 0.67 | 0.93 | 0.78 | 15 |
| CCRCC | 0.97 | 0.87 | 0.91 | 68 |
| COAD | 0.86 | 1.00 | 0.93 | 19 |
| GBM | 1.00 | 1.00 | 1.00 | 53 |
| HNSCC | 1.00 | 0.33 | 0.50 | 9 |
| LSCC | 0.80 | 0.92 | 0.86 | 71 |
| LUAD | 0.87 | 0.90 | 0.88 | 77 |
| **OV** | **0.00** | **0.00** | **0.00** | 12 |
| PDA | 0.90 | 0.98 | 0.94 | 48 |
| UCEC | 0.94 | 0.94 | 0.94 | 72 |

Asphericity: BRCA=0.148, CCRCC=0.044, COAD=0.135, GBM=0.017, HNSCC=0.064, LSCC=0.096, LUAD=0.102, OV=0.273, PDA=0.022, UCEC=0.047.
L2 projection onto the unit sphere destroys the hypercube-corner geometry of bipolar features: OV collapses to zero, macro F1 drops 0.96 -> 0.77.
</details>

---

#### Key takeaways

- **Best normalization for Gigapath real features**: `corner-l1` — same accuracy as no-norm (0.97) but higher macro F1 (0.96 vs 0.93) and lowest asphericity (0.015).
- **Avoid `corner-linfty`**: Completely degenerate (0.56 accuracy); 4/10 cancer types collapse.
- **Avoid `l2` for both feature types**: Degrades performance in both real (0.97 -> 0.90) and bipolar (0.97 -> 0.89) settings.
- **Bipolar features match real features**: 0.97 accuracy at no normalization despite extreme discretization.
- **Ternarizing tiles at probe time is safe**: `ternarize_tiles=True` preserves 0.97 accuracy.
