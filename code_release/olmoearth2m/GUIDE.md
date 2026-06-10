# OLMoEarth — Complete Technical Guide

> A deep-dive into the architecture, training pipeline, and evaluation framework
> for the OLMoEarth family of Earth Observation foundation models.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Repository Structure](#3-repository-structure)
4. [Data & Modalities](#4-data--modalities)
5. [Model Architecture](#5-model-architecture)
6. [Pretraining Pipeline](#6-pretraining-pipeline)
7. [Launching Training](#7-launching-training)
8. [Evaluation](#8-evaluation)
9. [Inference Quickstart](#9-inference-quickstart)
10. [Testing](#10-testing)

---

## 1. Project Overview

OLMoEarth is a **multi-modal, spatio-temporal Vision Transformer** trained on satellite imagery via **Latent Masked Image Modeling (Latent MIM)**. It is developed by the Allen Institute for AI (AI2).

**What makes it special:**
- Handles **multiple satellite modalities** simultaneously (optical, SAR, DEM, land-cover maps)
- Uses **FlexiViT**: flexible patch sizes, flexible resolution inputs
- Learns via **self-supervised pretraining** — no labels needed
- Transfers to downstream tasks via KNN, linear probing, or fine-tuning

**Model family (four sizes):**

| Model | Encoder Params | Decoder Params | HuggingFace |
|-------|---------------|----------------|-------------|
| Nano  | 1.4 M         | 0.8 M          | `allenai/OlmoEarth-v1-Nano` |
| Tiny  | 6.2 M         | 1.9 M          | `allenai/OlmoEarth-v1-Tiny` |
| Base  | 89 M          | 30 M           | `allenai/OlmoEarth-v1-Base` |
| Large | 308 M         | 53 M           | `allenai/OlmoEarth-v1-Large` |

---

## 2. Environment Setup

The project uses `uv` for dependency management. The recommended way to set up a conda environment is:

```bash
# Step 1: Create conda environment with Python 3.12
conda create -n olmoearth python=3.12 -y
conda activate olmoearth

# Step 2: Install uv into the conda env
pip install uv

# Step 3: Install all dependencies into the conda env
cd /path/to/olmoearth_pretrain
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --locked --extra all-no-flash
```

> **Note on `all-no-flash`**: This installs everything except `flash-attn`, which requires
> compiling CUDA extensions. If you have a compatible CUDA environment and want the full speed,
> additionally run: `UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync --locked --all-extras`

**Verify installation:**
```bash
python -c "import torch; print(torch.__version__); import olmoearth_pretrain; print('OK')"
# Expected: 2.7.1+cu128 / OK
```

**Optional dependency groups** (all included in `all-no-flash`):

| Group | Contains |
|-------|---------|
| `training` | olmo-core, rasterio, wandb, albumentations, cartopy |
| `eval` | pytorch-lightning, scikit-learn, terratorch, geobench, timm |
| `dev` | pytest, pytest-xdist, ruff, pre-commit |
| `beaker` | AI2-internal cluster submission tools |
| `dataset-creation` | Tools to create H5PY datasets from raw GeoTIFFs |

---

## 3. Repository Structure

```
olmoearth_pretrain/
├── olmoearth_pretrain/          # Main Python package
│   ├── config.py                # Base config system (from_dict / build)
│   ├── datatypes.py             # OlmoEarthSample, MaskedOlmoEarthSample, MaskValue
│   ├── model_loader.py          # Load pretrained models from HuggingFace
│   ├── data/                    # Dataset loading, normalization, constants
│   │   ├── constants.py         # Modality specs, band definitions, resolutions
│   │   ├── dataset.py           # OlmoEarthDataset (reads H5PY files)
│   │   ├── dataloader.py        # OlmoEarthDataLoader (batching, masking)
│   │   └── normalize.py         # Per-modality normalization statistics
│   ├── nn/                      # Neural network modules
│   │   ├── flexi_vit.py         # Encoder, Predictor, CompositeEncodings, MultiModalPatchEmbeddings
│   │   ├── flexi_patch_embed.py # FlexiPatchEmbed (the patch projection layer)
│   │   ├── attention.py         # Block, Attention, Mlp, LayerScale, DropPath
│   │   ├── encodings.py         # Sinusoidal positional encoding functions
│   │   ├── tokenization.py      # TokenizationConfig (band grouping)
│   │   ├── latent_mim.py        # LatentMIM (main pretraining model)
│   │   └── galileo.py           # Galileo variant (two-decoder contrastive)
│   ├── train/                   # Training infrastructure
│   │   ├── train_module/        # OlmoEarthTrainModule (the training loop)
│   │   ├── loss.py              # PatchDiscriminationLoss, InfoNCELoss, KoLeo, MAELoss
│   │   ├── masking.py           # Masking strategies
│   │   └── callbacks/           # Evaluation callbacks (downstream_evaluator)
│   ├── evals/                   # Evaluation framework
│   │   ├── eval_wrapper.py      # Unified model interface for evaluation
│   │   ├── knn.py               # K-Nearest Neighbor evaluation
│   │   ├── linear_probe.py      # Linear probe training & evaluation
│   │   ├── finetune/            # Fine-tuning evaluation
│   │   ├── datasets/            # Downstream dataset loaders & configs
│   │   └── models/              # Baseline model wrappers (Galileo, Clay, Satlas…)
│   └── internal/                # Experiment launchers & sweep scripts
│       ├── experiment.py        # Main entry point (subcommands: train, launch…)
│       ├── full_eval_sweep.py   # KNN / linear probe sweep launcher
│       ├── full_eval_sweep_finetune.py
│       └── all_evals.py         # Registry of all evaluation tasks
├── scripts/
│   └── official/                # Training scripts per model size
│       ├── nano.py / tiny.py / base.py / large.py
│       └── script.py            # Shared config builder utilities
├── tests/                       # Test suite (unit + integration)
├── tests_minimal_deps/          # Model loading tests (no olmo-core)
├── docs/                        # Documentation
│   ├── Pretraining.md
│   ├── Evaluation.md
│   └── Inference-Quickstart.md
└── helios/                      # Backward-compat shim (old package name)
```

---

## 4. Data & Modalities

### 4.1 Pretraining Dataset

- **285,288 samples** globally distributed across diverse land cover types
- Each sample covers a **2560×2560 m grid cell** with a **360-day time window**
- Data is broken into **12 monthly 30-day mosaics** per modality
- Download from HuggingFace: `allenai/olmoearth_pretrain_dataset`

### 4.2 Supported Modalities

| Modality | Bands | Resolution | Temporal | Notes |
|----------|-------|-----------|----------|-------|
| `sentinel2_l2a` | 12 (3 band-sets: 10/20/40 m) | 10 m base | Yes | Primary optical sensor |
| `sentinel1` | 2 (VV, VH) | 10 m | Yes | SAR radar imagery |
| `landsat` | 11 (2 band-sets: 10/20 m) | 15 m base | Yes | Historical optical (Landsat 8/9) |
| `worldcover` | 1 | 10 m | No | Land cover classification |
| `srtm` | 1 | ~30 m | No | Digital elevation model |
| `openstreetmap_raster` | 30 | 10 m | No | OSM features rasterized |
| `wri_canopy_height_map` | 1 | ~25 m | No | Forest canopy height |
| `cdl` | 1 | ~30 m | No | US Cropland Data Layer |
| `worldcereal` | 8 | 10 m | No | Crop type classifications |
| `era5_10` | 6 | Non-spatial | Yes | Weather variables (temperature, precip…) |
| `naip` | 4 (RGB+IR) | 1 m | No | US high-res aerial imagery |

**Multi-resolution band sets** (e.g., Sentinel-2):
- Band-set 0 (10 m): B02, B03, B04, B08
- Band-set 1 (20 m): B05, B06, B07, B8A, B11, B12
- Band-set 2 (60 m): B01, B09

Each band-set becomes its own token (or group of tokens) in the model.

### 4.3 H5PY File Format

Training data is stored in HDF5 files, one file per sample:

```
sample_XXXXX.h5
├── latlon                    # [2] — [latitude, longitude]
├── timestamps                # [T, 3] — [day, month, year] for T timesteps
├── sentinel2_l2a             # [H, W, T, C] — image data
├── sentinel1                 # [H, W, T, C]
├── landsat                   # [H, W, T, C]
├── worldcover                # [H, W, C] — static
├── srtm                      # [H, W, C] — static
└── missing_timesteps_masks/
    ├── sentinel2_l2a         # [T] boolean — which timesteps have real data
    └── sentinel1             # [T] boolean
```

---

## 5. Model Architecture

### 5.1 High-Level Overview

OLMoEarth uses the **Latent MIM** (Masked Image Modeling) framework:

```
Input satellite data (multiple modalities)
        │
        ▼
 ┌─────────────┐         ┌──────────────────┐
 │   Masking   │         │  Target Encoder  │ ◄── EMA copy of Encoder (frozen)
 └─────────────┘         └──────────────────┘
        │                         │
        ▼                         ▼
 ┌─────────────┐         ┌──────────────────┐
 │   Encoder   │         │  Target Tokens   │ (ground truth for loss)
 │  (online)   │         └──────────────────┘
 └─────────────┘                  │
        │                         │
        ▼                         │
 ┌─────────────┐                  │
 │   Decoder   │──────────────────┤
 │ (predictor) │                  │
 └─────────────┘                  │
        │                         │
        └──────── MSE Loss ───────┘
                (masked tokens only)
```

The key insight: the decoder is trained to **predict what the target encoder would output** for the masked tokens — this is "latent" MIM, as opposed to pixel-level reconstruction.

### 5.2 Data Structures (`datatypes.py`)

**`OlmoEarthSample`** — raw input:
```python
@dataclass
class OlmoEarthSample:
    sentinel2_l2a: Tensor | None    # [B, H, W, T, C]  — all spatial modalities follow this shape
    sentinel1: Tensor | None
    landsat: Tensor | None
    worldcover: Tensor | None
    srtm: Tensor | None
    # ... more modalities ...
    latlon: Tensor | None           # [B, 2]
    timestamps: Tensor | None       # [B, T, 3]  —  [day, month, year]
```

**`MaskedOlmoEarthSample`** — input with masks:
```python
@dataclass
class MaskedOlmoEarthSample(OlmoEarthSample):
    sentinel2_l2a_mask: Tensor | None   # same spatial shape as sentinel2_l2a
    sentinel1_mask: Tensor | None
    # ... one mask per modality ...
```

**`MaskValue`** — four states for each token:
```python
class MaskValue(IntEnum):
    ONLINE_ENCODER = 0        # Token is seen by the online encoder
    TARGET_ENCODER_ONLY = 1   # Token is only seen by the target encoder
    DECODER = 2               # Token is only seen by the decoder (masked from encoder)
    MISSING = 3               # Data is missing (no satellite coverage)
```

### 5.3 Patch Embedding (`flexi_patch_embed.py`)

**`FlexiPatchEmbed`** converts raw satellite bands into token vectors:

```
Input: [B, H, W, T, C]   (batch, height, width, time, channels)
    │
    ├── For temporal modalities: flatten T into B → [B*T, H, W, C]
    │
    ├── Reshape into patches of size P×P → [B*T, h, w, P*P*C]
    │
    └── Linear projection → [B*T, h, w, D]   (D = embedding_size)
```

Key feature: **flexible patch size**. The same weights can be used with different patch sizes via weight interpolation — useful for multi-resolution inference.

**`MultiModalPatchEmbeddings`** applies `FlexiPatchEmbed` to every modality and every band-set independently:

```
MaskedOlmoEarthSample
    │
    ├── sentinel2_l2a → FlexiPatchEmbed (band-set 0) → tokens [B, h, w, T, D]
    │                 → FlexiPatchEmbed (band-set 1) → tokens
    │                 → FlexiPatchEmbed (band-set 2) → tokens
    │
    ├── sentinel1 → FlexiPatchEmbed → tokens
    │
    └── ... → tokens
    
    Combined: dict[modality_name → Tensor[B, h, w, T, num_bandsets, D]]
```

**Band dropout** (during training): randomly zeros out spectral bands before embedding, forcing the model to learn cross-band representations. Controlled by `band_dropout_rate`.

### 5.4 Positional Encodings (`flexi_vit.py`, `encodings.py`)

**`CompositeEncodings`** adds four encoding types, each occupying D/4 of the embedding dimension:

| Component | Dimension | What it encodes |
|-----------|-----------|-----------------|
| Channel embedding | D/4 | Which band-set this token belongs to (learnable) |
| Temporal encoding | D/4 | Timestep index (sinusoidal 1D) |
| Month encoding | D/4 | Calendar month (cyclic sinusoidal, 12-month) |
| Spatial encoding | D/4 | Patch (row, col) position (sinusoidal 2D, resolution-aware) |

The spatial encoding is **resolution-aware**: tokens from a 20 m modality get a different scale than 10 m tokens, allowing the model to understand absolute geographic position regardless of sensor resolution.

### 5.5 Encoder Architecture (`flexi_vit.py`)

```
MultiModalPatchEmbeddings   →  dict of tokens per modality
        │
        ▼
CompositeEncodings          →  add positional/temporal/channel info
        │
        ▼
[Optional] Register Tokens  →  prepend learnable stability tokens
        │
        ▼
Flatten all modalities      →  [B, N_total_tokens, D]
        │
        ▼
Transformer Blocks × depth  →  [B, N_total_tokens, D]
   (LayerNorm → Attention → Add → LayerNorm → MLP → Add)
        │
        ▼
[Optional] Project & Pool   →  [B, D_out]  (for contrastive loss)
        │
        ▼
Reshape to spatial          →  dict of tensors per modality
```

**Encoder config parameters (base model as example):**

```python
EncoderConfig(
    supported_modality_names=["sentinel2_l2a", "sentinel1", "landsat", ...],
    embedding_size=768,          # Token dimension D
    depth=12,                    # Number of transformer layers
    num_heads=12,                # Attention heads
    mlp_ratio=4.0,               # MLP hidden dim = mlp_ratio × D
    max_patch_size=8,            # Maximum patch size (in pixels at 10m)
    min_patch_size=1,
    drop_path=0.1,               # Stochastic depth rate
    max_sequence_length=12,      # Max timesteps
    num_register_tokens=0,       # Optional stability tokens
    learnable_channel_embeddings=True,
    use_flash_attn=False,        # Toggle Flash Attention
    qk_norm=False,               # QK normalization
    output_embedding_size=None,  # If set, adds a final projection
    band_dropout_rate=0.0,       # Spectral band dropout
)
```

### 5.6 Decoder / Predictor Architecture

The decoder (called `Predictor` in the code) has a symmetric structure to the encoder but is typically shallower.

```
Encoder outputs (visible tokens only)
        │
        ▼
Decode mask tokens        →  replace masked positions with learned [MASK] embedding
        │
        ▼
Add positional encodings  →  same CompositeEncodings as encoder
        │
        ▼
Transformer Blocks × depth_decoder
        │
        ▼
Linear projection         →  [B, N_masked, D]  (predictions for masked tokens)
```

The decoder predicts the **full embedding** that the target encoder would produce — not pixels.

### 5.7 Target Encoder (EMA)

The target encoder is an **Exponential Moving Average** (EMA) copy of the online encoder:

```python
# After each optimizer step:
target_param = ema_momentum * target_param + (1 - ema_momentum) * online_param
```

- The target encoder receives **all tokens** (no masking), producing stable target representations
- It is **never directly optimized** — only updated via EMA
- This prevents representation collapse (the model cannot trivially "cheat")

Default: `ema_momentum = 1.0` (target is effectively frozen at initialization, EMA rate can be annealed)

### 5.8 Loss Functions (`train/loss.py`)

**Primary: Patch Discrimination Loss**

For each pair of (decoder prediction, target encoder output) at masked positions:

```
1. L2-normalize both predictions and targets
2. Compute cosine similarity matrix (within each sample)
3. Apply cross-entropy with temperature τ = 0.1
   — Diagonal entries are positives (same spatial location)
   — Off-diagonal entries are negatives (different locations)
```

This is a **per-sample** contrastive loss (memory-efficient vs. full-batch contrastive).

**Optional: InfoNCE Contrastive Loss**

When training with two different masked views (A and B) of the same sample:

```
1. Pool encoder outputs for view A and view B → [B, D] each
2. L2-normalize
3. Compute full batch similarity matrix
4. Cross-entropy: same sample = positive, different samples = negative
```

**Optional: KoLeo Regularizer**

Penalizes embedding collapse by encouraging uniform span:
```
L_koleo = -log(min_distance_to_neighbor)
```

**Combined loss:**
```
L_total = L_patch_discrimination + λ_contrastive × L_contrastive + λ_koleo × L_koleo
```

### 5.9 Masking Strategies (`train/masking.py`)

The masking strategy controls which tokens each component sees:

**`modality_cross_random`** (default, most powerful):
- Each modality is masked independently
- 50% of tokens → online encoder (ONLINE_ENCODER)
- 50% of tokens → decoder target (DECODER)
- Some modalities are **decode-only** (never encoded): `worldcover`, `srtm`, `openstreetmap_raster`, etc.
  - These act as reconstruction targets — the model learns to predict map features from imagery

**Other strategies:**
- `random`: Simple random masking across all modalities uniformly
- `time`: Mask entire timesteps coherently (test temporal understanding)
- `space`: Mask spatial patches coherently

### 5.10 Galileo Variant (`nn/galileo.py`)

Galileo uses **two independent decoders** (decoder_a and decoder_b — a deep copy of each other) and adds a multi-crop contrastive objective between their outputs. This improves representation quality at the cost of more memory.

### 5.11 Tokenization Configuration

`TokenizationConfig` controls how bands within a modality are grouped into tokens:

```python
# Default: use the modality's natural band-set structure
# Sentinel-2 → 3 band-sets (10m, 20m, 60m bands)

# Custom: merge all bands into one token
TokenizationConfig(overrides={
    "sentinel2_l2a": ModalityTokenization(
        band_groups=[["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]]
    )
})
```

---

## 6. Pretraining Pipeline

### 6.1 Training Loop (`train/train_module/train_module.py`)

Each training step:

```
1. Load batch from dataloader (H5PY → MaskedOlmoEarthSample)
   - Workers apply masking on CPU

2. Split into microbatches (for gradient accumulation)

3. For each microbatch:
   a. Online encoder forward (masked input) → encoder tokens + pooled features
   b. Decoder forward (encoder tokens) → predicted masked tokens
   c. Target encoder forward (no_grad, full input) → target tokens
   d. Compute patch discrimination loss (decoder predictions vs. target tokens)
   e. If contrastive: compute InfoNCE loss on pooled features
   f. loss.backward() — accumulate gradients

4. Optimizer step:
   - Clip gradient norm (max=1.0)
   - Update learning rate (cosine schedule)
   - AdamW step

5. EMA update: target_encoder ← ema * target + (1-ema) * online

6. Every N steps: checkpoint, log metrics, run downstream eval
```

### 6.2 Data Pipeline

```
H5PY files on disk
        │
        ▼
OlmoEarthDataset (one file per sample)
  - Load all modality arrays from HDF5
  - Apply spatial crop (random 1280×1280 m sub-tile)
  - Apply temporal subset (up to max_sequence_length=12 timesteps)
  - Enforce token budget (max tokens = 2250 per sample)
  - Apply normalization (per-modality mean/std)
        │
        ▼
DataLoader workers (CPU)
  - Apply flip / rotation augmentations
  - Apply masking strategy → MaskedOlmoEarthSample
  - Create 1 or 2 masked views (for contrastive training)
        │
        ▼
Collation → (patch_size, MaskedOlmoEarthSample)
        │
        ▼
GPU training
```

**Token budget enforcement**: Because different spatial crops lead to different numbers of patches, the dataloader dynamically adjusts patch size to keep the sequence length within the budget. `sampled_hw_p_list = [1, 2, ..., 12]` controls possible numbers of spatial patches per side.

### 6.3 Optimizer & Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 (base/large), 1e-3 (nano) |
| Weight decay | 0.02 |
| β₁, β₂ | 0.9, 0.999 |
| Gradient clip | 1.0 |
| Schedule | Cosine with linear warmup |
| Warmup steps | 8000 |
| Batch size | 512 (global) |

### 6.4 Checkpointing

- **Permanent** checkpoint every **5000 steps** (kept forever)
- **Ephemeral** checkpoint every **250 steps** (overwritten by next)
- Training automatically resumes from the latest checkpoint

---

## 7. Launching Training

### 7.1 Local Multi-GPU Training

First, prepare your dataset in H5PY format (see `docs/Pretraining-Dataset.md`), then:

**Nano** (4× GPU, 16 GB+ each):
```bash
torchrun --nproc_per_node=4 scripts/official/nano.py train my_nano_run local \
  --dataset.h5py_dir=/path/to/h5data/1138828
```

**Tiny** (4–8× GPU, 24 GB+ each):
```bash
torchrun --nproc_per_node=4 scripts/official/tiny.py train my_tiny_run local \
  --dataset.h5py_dir=/path/to/h5data/1138828
```

**Base** (8× GPU, 40 GB+ each):
```bash
torchrun --nproc_per_node=8 scripts/official/base.py train my_base_run local \
  --dataset.h5py_dir=/path/to/h5data/1138828
```

**Large** (8× H100 80 GB):
```bash
torchrun --nproc_per_node=8 scripts/official/large.py train my_large_run local \
  --dataset.h5py_dir=/path/to/h5data/1138828
```

### 7.2 Validating Your Config (Dry Run)

Before committing to a full run, validate the configuration:
```bash
python scripts/official/base.py dry_run config_test local
```

### 7.3 Common Overrides

```bash
torchrun --nproc_per_node=8 scripts/official/base.py train my_run local \
  --dataset.h5py_dir=/path/to/data \
  --common.save_folder=/path/to/checkpoints \
  --data_loader.global_batch_size=256 \
  --data_loader.num_workers=8 \
  --train_module.rank_microbatch_size=8 \
  --train_module.optim_config.lr=0.0002 \
  --train_module.optim_config.weight_decay=0.02 \
  --train_module.scheduler.warmup_steps=5000 \
  --trainer.max_duration='{"unit": "epochs", "value": 100}'
```

### 7.4 Hardware Guide

| Model | Min GPUs | GPU RAM | Notes |
|-------|----------|---------|-------|
| Nano | 1–4 | 16 GB | Good for debugging / ablations |
| Tiny | 4–8 | 24 GB | Small-scale experiments |
| Base | 8 | 40 GB | Main production model |
| Large | 8 | 80 GB | H100s strongly recommended |

**Adapting to limited hardware:**
- Reduce `global_batch_size` (e.g., 256 instead of 512)
- Reduce `rank_microbatch_size` (e.g., 8 instead of 32)
- Increase gradient accumulation automatically
- Fewer GPUs → scale down `nproc_per_node` and `global_batch_size` proportionally

### 7.5 Distributed Training Details

The model uses OLMo-core's distributed infrastructure:

- **FSDP** (Fully Sharded Data Parallel): shards weights + gradients across GPUs
  - `param_dtype=bfloat16`, `reduce_dtype=float32` (mixed precision)
- **Gradient accumulation**: controlled by `global_batch_size / (rank_microbatch_size × n_gpus)`
- **Flash Attention**: optional, requires `flash-attn` extra

---

## 8. Evaluation

### 8.1 Evaluation Modes

Three modes are supported:

| Mode | What it tests | When to use |
|------|--------------|-------------|
| **KNN** | Nearest-neighbor classification using frozen embeddings | Fast probe of representation quality |
| **Linear Probe** | Train a linear layer on frozen embeddings | Standard SSL benchmark |
| **Fine-tuning** | Unfreeze backbone + train a head end-to-end | Maximum downstream performance |

### 8.2 Downstream Datasets (25+)

**Classification datasets:**

| Dataset | Classes | Modality | Metric |
|---------|---------|---------|--------|
| m-eurosat | 10 | Sentinel-2 | Accuracy |
| m-bigearthnet | 43 (multilabel) | Sentinel-2 | Macro F1 |
| m-so2sat | 17 | Sentinel-2 | Accuracy |
| m-brick-kiln | 2 | Sentinel-2 | F1 |
| breizhcrops | 9 (timeseries) | Sentinel-2 | Accuracy |
| m-forestnet | 12 | Landsat | Accuracy |

**Segmentation datasets:**

| Dataset | Classes | Resolution | Metric |
|---------|---------|-----------|--------|
| mados | 15 | 80×80 | mIoU |
| sen1floods11 | 2 | 64×64 | mIoU |
| pastis | 19 (timeseries) | 64×64 | mIoU |
| m-sa-crop-type | 10 | 256×256 | mIoU |
| m-cashew-plant | 7 | 256×256 | mIoU |

### 8.3 Running Evaluation Sweeps

**Quick dry-run (print commands without executing):**
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=local \
  --checkpoint_path=/path/to/OlmoEarth-v1-Base \
  --module_path=scripts/official/base.py \
  --defaults_only \
  --dry_run
```

**Full KNN + linear probe sweep (default hyperparameters):**
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=local \
  --checkpoint_path=/path/to/OlmoEarth-v1-Base \
  --module_path=scripts/official/base.py \
  --project_name=my_evals \
  --defaults_only
```

**Single task with test-set evaluation:**
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=local \
  --checkpoint_path=/path/to/OlmoEarth-v1-Base \
  --module_path=scripts/official/base.py \
  --project_name=my_evals \
  --select_best_val \
  --trainer.callbacks.downstream_evaluator.run_on_test=True \
  --trainer.callbacks.downstream_evaluator.tasks_to_run=\[m_eurosat\] \
  --defaults_only
```

**Fine-tuning evaluation:**
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep_finetune \
  --cluster=local \
  --checkpoint_path=/path/to/OlmoEarth-v1-Base \
  --module_path=scripts/official/base.py \
  --project_name=my_evals \
  --defaults_only
```

**Evaluating a baseline model (e.g., Galileo):**
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=local \
  --model=galileo \
  --all_sizes \
  --project_name=baseline_evals \
  --defaults_only
```

### 8.4 How KNN Evaluation Works

1. Run the encoder on all training samples → extract embeddings (L2-normalized)
2. For each test sample, find the k=20 nearest neighbors by cosine similarity
3. Aggregate neighbor labels via temperature-weighted voting (τ=0.07)
4. Bootstrap resampling (n=1000) for confidence intervals

For **multilabel datasets** (e.g., BigEarthNet), a separate KNN is run per class.

### 8.5 How Linear Probe Works

For **classification:**
1. Extract frozen embeddings for all train/val/test samples
2. Train `Linear(D → num_classes)` on top
3. Sweep learning rates: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
4. Pick best LR on validation set, report test set

For **segmentation:**
1. Extract patch-level embeddings (spatial structure preserved)
2. Train `AttnPoolLinearProbe`: multi-head attention pooling + linear head
3. Bilinear interpolation back to full resolution
4. Loss: weighted Dice (for class imbalance)

### 8.6 Evaluation During Training

The training loop automatically runs downstream evaluation every N steps using `DownstreamEvaluatorCallback`. Default tasks: m-eurosat (every 4000 steps), mados (every 4000 steps), m-so2sat (every 20000 steps), pastis (every 20000 steps).

To disable:
```bash
--trainer.callbacks.downstream_evaluator.enabled=False
```

---

## 9. Inference Quickstart

### 9.1 Load a Pretrained Model

```python
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

# Load from HuggingFace (downloads ~350 MB for Base)
model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
model.eval()

# Or load from local path
from olmoearth_pretrain.model_loader import load_model_from_path
model = load_model_from_path("/path/to/checkpoint/")
```

Available `ModelID` values:
- `OLMOEARTH_V1_NANO`
- `OLMOEARTH_V1_TINY`
- `OLMOEARTH_V1_BASE`
- `OLMOEARTH_V1_LARGE`

### 9.2 Extract Embeddings (Synthetic Data)

```python
import torch
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

# Create synthetic Sentinel-2 input
# Shape: [batch, height, width, time, channels]
B, H, W, T, C = 1, 64, 64, 3, 12
image = torch.randn(B, H, W, T, C)

# All tokens visible to encoder (no masking)
# Sentinel-2 has 3 band-sets → mask shape last dim = 3
mask = torch.full((B, H, W, T, 3), MaskValue.ONLINE_ENCODER.value, dtype=torch.float32)

# Timestamps: [day, month (0-indexed), year]
timestamps = torch.tensor([[[15, 3, 2024], [20, 6, 2024], [10, 9, 2024]]])  # 3 timesteps

sample = MaskedOlmoEarthSample(
    sentinel2_l2a=image,
    sentinel2_l2a_mask=mask,
    timestamps=timestamps,
)

with torch.no_grad():
    output = model.encoder(sample, patch_size=4)

# Per-token embeddings: [B, h, w, T, num_bandsets, D]
tokens = output["tokens_and_masks"].sentinel2_l2a

# Pooled global embedding: [B, D]  (if model has projection head)
# pooled = output.get("projected_and_pooled")

# Simple mean pooling across space, time, and band-sets
embedding = tokens.mean(dim=[1, 2, 3, 4])  # → [B, D]
print(embedding.shape)  # torch.Size([1, 768])
```

### 9.3 Real Sentinel-2 Imagery

```python
import glob
import numpy as np
import rasterio
import torch
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy

# 1. Load bands in the correct order for the model
modality = Modality.SENTINEL2_L2A
fnames = []
for band_name in modality.band_order:
    fname = glob.glob(f"*.SAFE/GRANULE/*/IMG_DATA/*/*_{band_name}_*.jp2")[0]
    fnames.append(fname)

# 2. Read and resample to consistent size
with rasterio.open(fnames[0]) as src:
    crs, transform = src.crs, src.transform

H, W = 512, 512
image = np.zeros((len(fnames), H, W), dtype=np.int32)
for i, fname in enumerate(fnames):
    with rasterio.open(fname) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=W, height=H,
                       resampling=Resampling.bilinear) as vrt:
            image[i] = vrt.read(1)

# 3. Rearrange to [B, H, W, T, C]
image = image.transpose(1, 2, 0)[None, :, :, None, :]  # → [1, 512, 512, 1, 12]

# 4. Normalize (mean/std computed from pretraining dataset)
normalizer = Normalizer(Strategy.COMPUTED)
image = normalizer.normalize(Modality.SENTINEL2_L2A, image)

# 5. Run model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE).to(device)

sample = MaskedOlmoEarthSample(
    sentinel2_l2a=torch.tensor(image[:, :64, :64, :, :], dtype=torch.float32, device=device),
    sentinel2_l2a_mask=torch.full((1, 64, 64, 1, 3), MaskValue.ONLINE_ENCODER.value,
                                  dtype=torch.float32, device=device),
    timestamps=torch.tensor([[[22, 6, 2024]]], device=device),  # day, month(0-11), year
)

with torch.no_grad():
    output = model.encoder(sample, patch_size=4)
    tokens = output["tokens_and_masks"].sentinel2_l2a  # [1, 16, 16, 1, 3, 768]
    embedding = tokens.mean(dim=[1, 2, 3, 4])          # [1, 768]
```

### 9.4 Multi-Modality Input

```python
sample = MaskedOlmoEarthSample(
    # Optical: [B, H, W, T, 12 bands]
    sentinel2_l2a=s2_image,
    sentinel2_l2a_mask=s2_mask,  # shape [B, H, W, T, 3]  (3 band-sets)

    # SAR: [B, H, W, T, 2 bands]
    sentinel1=s1_image,
    sentinel1_mask=s1_mask,  # shape [B, H, W, T, 1]  (1 band-set)

    # DEM: [B, H, W, 1 band] — no time dimension for static modalities
    srtm=dem_image,
    srtm_mask=dem_mask,  # shape [B, H, W, 1]

    timestamps=timestamps,  # [B, T, 3]
)
```

---

## 10. Testing

### 10.1 Test Suites

```
tests/
├── unit/                    # Fast, no-GPU, test individual functions
│   ├── nn/                  # Encoder, tokenization, patch embedding
│   ├── data/                # Dataset, normalizer, dataloader
│   ├── eval/                # Metrics, embedding transforms
│   └── train/               # Loss functions, masking, utils
└── integration/             # End-to-end tests with real forward passes
    ├── nn/                  # Full model forward passes (FlexiViT, MAE, Latent MIM)
    ├── train/               # Full training step (latent MIM, contrastive)
    └── eval/                # Linear probe, fine-tuning, GeoBench dataset
```

### 10.2 Running Tests

```bash
# All unit tests (fast, parallelized)
pytest tests/unit/ -n auto -v

# Integration: evaluation only
pytest tests/integration/eval/ -v

# Integration: pretraining / training only
pytest tests/integration/train/ -v

# Integration: model architecture
pytest tests/integration/nn/ -v

# All tests
pytest tests/ -n auto -v

# Model loading tests (minimal dependencies)
pytest tests_minimal_deps/ -v

# Single test file
pytest tests/unit/train/test_loss.py -v

# Single test function
pytest tests/integration/eval/test_probe.py::test_probe_cls -v
```

### 10.3 Test Configuration

Pytest is configured in `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["--import-mode=importlib"]
```

All tests use `seed=42` via the `set_random_seeds()` autouse fixture in `tests/conftest.py`, ensuring reproducibility.

---

## Key Concepts Summary

| Concept                    | What it means in OLMoEarth                                                   |
| -------------------------- | ---------------------------------------------------------------------------- |
| **FlexiViT**               | ViT with flexible patch size — same weights work for different resolutions   |
| **Band-set**               | A group of spectral bands at the same resolution; each becomes its own token |
| **Latent MIM**             | Predict target encoder outputs (not pixels) at masked positions              |
| **Target encoder**         | EMA copy of encoder; provides stable prediction targets                      |
| **Composite encodings**    | 4-part positional signal: spatial + temporal + month + channel               |
| **Modality cross masking** | Each modality is masked independently; some are decode-only                  |
| **Band dropout**           | Randomly zero spectral bands during training to force cross-band learning    |
| **Token budget**           | Cap on total tokens per sample to fit in GPU memory                          |
| **Register tokens**        | Learnable tokens prepended to sequence for training stability                |
| **KoLeo regularizer**      | Loss term encouraging embeddings to uniformly span the space                 |
