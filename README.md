# BitPath (autopath)

**BitPath** is a framework for distilling heavyweight pathology foundation models (such as **Prov-GIGAPATH**) into lightweight, 1.58-bit ternary neural networks (**BitNet 1.58b** style). By learning to predict discrete **bipolar feature vectors** ($\{-1, 0, +1\}^{1536}$) directly from raw 256×256 RGB pathology tiles, BitPath enables fast, CPU-native slide processing, feature extraction, and downstream cancer classification without requiring GPU backbones at inference time.

> 📘 **Full Logs & Results**: For detailed benchmark results, training curves, connectivity analysis, and research plans, see [LOG.md](file:///home/t-9dkarp/autopath/LOG.md).

---

## 🚀 Key Features & Architectural Highlights

1. **Bipolar Feature Distillation**:
   - Quantizes high-dimensional Gigapath representations into 1.58-bit ternary space while maintaining high linear separability.
   - Evaluates minimal accuracy loss (e.g. 0.97 $\to$ 0.92 cancer-of-origin probe accuracy) on CPTAC and TCGA datasets.

2. **BitNet 1.58b Quantization (`autopath.autobits`)**:
   - **Ternary Weight Quantization**: Absmean scaling and Straight-Through Estimator (STE) gradient training (`{-1, 0, +1}`).
   - **8-bit Activation Quantization**: Absmax scaling with RMSNorm for parameter-free, low-latency execution.
   - Custom ternary convolutions (`BitConv2d158`) and linear layers (`BitLinear158`).

3. **Modular Datablock & Pipeline Engine (`autopath.gigapan`)**:
   - Functional datablocks, datastacks, and pipelines (`bitpath_still`, `bitconv_still`) for reproducible training and cached evaluations.
   - Support for DDP multi-GPU training via PyTorch Lightning.

4. **Linear & Affine Probing Suite (`autopath.gigapan.probes`)**:
   - Built-in probing metrics including Accuracy, Macro/Weighted F1, and **Asphericity** (classifier intercept-to-coefficient ratio).

---

## 📁 Repository Structure

```
autopath/
├── autopath/
│   ├── autobits.py         # BitNet 1.58b layers (BitConv2d158, BitLinear158, RMSNorm)
│   ├── gigapath/           # Gigapath & DINOv2 backbone adapters
│   ├── gigapan/            # Datablocks, datastacks, & distillation pipelines
│   ├── pancan/             # CPTAC & TCGA dataset schemas & loaders
│   └── stills/             # Lightning modules & training stills
├── LOG.md                  # Comprehensive log, training curves, results & research plan
├── README.md               # Codebase overview and quickstart
└── setup.py                # Package setup & installation configuration
```

---

## 💡 Quickstart

```python
from autopath.gigapan.pipelines import bitpath_still

# Assemble and launch BitPath convolutional distillation pipeline
still = bitpath_still(
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
    cfg_layer='final',
    cfg_cls_token_only=True,
    cfg_shard_size=64,
    cfg_batch_size=64,
    cfg_n_blocks=6,
    cfg_hidden_channels=128,
    cfg_learning_rate=1e-3,
    max_steps=10000,
    n_devices=4,
)

# Build training graph
still.build_tree()
```

---

## 📖 Further Documentation

- **Full Project Log & Benchmark Results**: [LOG.md](file:///home/t-9dkarp/autopath/LOG.md)
- **Environment & Setup**: [SETUP.md](file:///home/t-9dkarp/autopath/SETUP.md)
- **SLURM Cluster Training**: [SLURM.md](file:///home/t-9dkarp/autopath/SLURM.md)
