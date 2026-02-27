# DiffSegLung: Diffusion Radiomic Distillation for Unsupervised Lung Pathology Segmentation

> **MICCAI 2026** | Anonymized submission

---

## Overview

**DiffSegLung** is an unsupervised framework for simultaneous segmentation of four lung pathologies (emphysema, fibrosis, GGO, consolidation) from unlabelled 3D CT volumes.

The key idea: handcrafted radiomic descriptors (GLCM, LBP, Gabor, HU statistics) serve as a **physics-grounded teacher** to shape the bottleneck of a 3D diffusion U-Net via InfoNCE contrastive distillation — no annotations required.

![Pipeline](figures/pipeline_distilation.pdf)

---

## Method Summary

**Training**
- 3D pixel-space DDPM trained on HU-preserved CT patches
- Radiomic descriptors (34-dim) act as non-differentiable teacher
- InfoNCE loss aligns student bottleneck with teacher embeddings
- Warmup schedule prevents bottleneck collapse

**Inference**
- DPM-Solver at 250 steps (8× faster than full DDPM)
- Multi-timestep bottleneck features aggregated → 384-dim descriptor
- GMM clustering (K=5) with HU-guided label assignment
- Sobel-Diffusion Fusion for boundary refinement

---

## Repository Structure

```
DiffSegLung/
│
├── train.py                              # Stage 1: base DDPM training
├── train_distill.py                      # Stage 2: distillation training
│
├── distillation.py                       # Core: projection heads, InfoNCE, warmup
├── gpu_radiomics.py                      # GPU radiomic feature extraction (fast)
├── radiomic_features.py                  # CPU radiomic features (34-dim, full)
│
├── lung_dataset.py                       # Dataset from NPZ files (mmap)
├── lung_dataset_precomputed.py           # Dataset from precomputed .npy patches
├── precompute_patches.py                 # Precompute patches to .npy (recommended)
├── patch_utils.py                        # Shared patch extraction utilities
│
├── inference_volume_distill.py           # Run inference on a full CT volume
├── segment_volume_voxel_level_ditil.py   # Segmentation with distillation
├── segment_volume_voxel_level_abalation.py # Ablation segmentation variants
├── gpu_gmm.py                            # GPU-accelerated GMM clustering
│
├── sample.py                             # Generate CT samples from trained model
├── visualize_clusters.py                 # Visualize radiomic clustering on a volume
├── sanity_check.py                       # Verify all components before training
│
├── requirements.txt
├── checkpoints/                          # ← put your checkpoints here (see below)
│   ├── ddpm_base.pt                      #   Stage 1 checkpoint (base DDPM)
│   └── ddpm_distill.pt                   #   Stage 2 checkpoint (with distillation)
│
└── figures/
    ├── pipeline_distilation.pdf
    ├── qualitative_generation_difseglung.drawio.pdf
    └── qualitative_segmentatio_reuslts_miccai.pdf
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/DiffSegLung.git
cd DiffSegLung
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA 11.8+

---

## Data Preparation

Your CT volumes should be in NPZ format with keys `ct` (float32, HU normalized to [-1,1]) and `mask` (binary lung mask).

Normalization formula used: `norm = (HU + 1000) / 700 - 1`
- HU = -1000 (air) → -1.0
- HU = -300 (normal lung) → 0.0  
- HU = +400 (bone) → +1.0

**Precompute patches** (strongly recommended for fast training):

```bash
python precompute_patches.py \
    --npz_dir /path/to/your/NPZ_DATA \
    --out_dir ./patches_precomputed \
    --patches_per_patient 150 \
    --num_workers 24
```

This saves `ct.npy`, `mask.npy`, `patient_ids.npy` — loaded instantly via mmap during training.

---

## Training

### Step 1 — Verify everything works

```bash
python sanity_check.py
```

Expected output: `ALL 5 TESTS PASSED`

### Step 2 — Base DDPM training (Stage 1)

```bash
# Single GPU
python train.py --npz_dir /path/to/NPZ_DATA --results_dir ./results

# 2 GPUs
torchrun --nproc_per_node=2 train.py \
    --npz_dir /path/to/NPZ_DATA \
    --results_dir ./results \
    --batch_size 8 \
    --lr 1e-4 \
    --epochs 100000 \
    --sample_every 500
```

### Step 3 — Distillation training (Stage 2)

Start from a Stage 1 checkpoint (recommended: after ~8000 steps when the model generates reasonable CT patches).

```bash
torchrun --nproc_per_node=2 train_distill.py \
    --patches_dir ./patches_precomputed \
    --results_dir ./results_distill \
    --resume ./checkpoints/ddpm_base.pt \
    --start_step 4000 \
    --warmup_start 4000 \
    --warmup_end 6000 \
    --batch_size 16 \
    --lr 1e-4 \
    --lambda_max 0.5
```

### Checkpoint placement

Put your checkpoints in the `checkpoints/` folder:

```
checkpoints/
├── ddpm_base.pt        # Stage 1 — base DDPM (before distillation)
└── ddpm_distill.pt     # Stage 2 — with distillation (use this for inference)
```

> **Note:** Checkpoints are not included in this repository due to file size.  
> Download links will be provided upon paper acceptance.

---

## Inference

Run segmentation on a single CT volume:

```bash
python inference_volume_distill.py \
    --npz    /path/to/patient.npz \
    --model  ./checkpoints/ddpm_distill.pt \
    --out    ./results/segmentation
```

Visualize radiomic clustering on a volume:

```bash
python visualize_clusters.py \
    --npz  /path/to/patient.npz \
    --out  ./cluster_preview \
    --k    5
```

---

## Results

### Segmentation (190 expert-annotated axial slices)

| Method | Healthy DSC ↑ | GGO DSC ↑ | Fibrosis DSC ↑ | Emph. DSC ↑ | HD95 ↓ |
|---|---|---|---|---|---|
| K-Means on radiomics | 71.2 | 66.8 | 68.4 | 58.3 | 18.7 |
| Diffusion features only | 78.4 | 71.3 | 73.1 | 62.7 | 15.4 |
| DAAM | 81.6 | 75.2 | 77.8 | 66.4 | 13.2 |
| **DiffSegLung (ours)** | **89.3** | **84.1** | **86.4** | **76.2** | **8.6** |

### Generation Quality

| Method | FID ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---|
| MedVAE | 48.3 | 0.721 | 24.6 |
| LungDDPM+ | 31.7 | 0.803 | 27.2 |
| **DiffSegLung (ours)** | **18.4** | **0.891** | **31.8** |

---

## Citation

```bibtex
@inproceedings{diffseglung2026,
  title     = {DiffSegLung: Diffusion Radiomic Distillation for Unsupervised Lung Pathology Segmentation},
  author    = {Anonymized Authors},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

---

## License

This project is released for research purposes. See `LICENSE` for details.
