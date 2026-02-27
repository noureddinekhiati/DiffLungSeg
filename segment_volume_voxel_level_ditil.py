#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
segment_volume.py  (v3 — distillation model + GPU GMM + adaptive labels)

Key changes vs v2:
  1. Loads proj_head from distillation checkpoint
  2. Uses proj_head(bottleneck) as features — radiomic-aligned embedding
  3. GPU GMM with BIC auto-K (or fixed K)
  4. Patient-adaptive cluster naming via HU percentiles
  5. Radiomics no longer concatenated — proj_head already encodes them

Feature pipeline:
  OLD: [UNet bottleneck 256 dims] + [radiomics 34 dims] → concat → KMeans
  NEW: proj_head(UNet bottleneck) → 128 dims → GPU GMM

  The 128-dim proj_head embedding is radiomic-aware by construction
  (trained via InfoNCE to align with radiomic teacher signal).
  No need to separately concatenate radiomics at inference.

Run command:
    python segment_volume.py \
        --npz /path/to/patient.npz \
        --checkpoint ./results_distill_final/model-60.pt \
        --clustering gmm \
        --auto_k --k_min 2 --k_max 8 \
        --adaptive_labels \
        --out ./seg_distill

    # Fixed K (no auto):
    python segment_volume.py \
        --npz /path/to/patient.npz \
        --checkpoint ./results_distill_final/model-60.pt \
        --k 5 --clustering gmm --adaptive_labels \
        --out ./seg_distill

    # Old model (no proj_head in checkpoint):
    python segment_volume.py \
        --npz /path/to/patient.npz \
        --checkpoint ./results/model-8.pt \
        --no-use_proj_head \
        --k 5 --clustering gmm --adaptive_labels \
        --out ./seg_baseline
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from scipy.ndimage import median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion_model.trainer import GaussianDiffusion
from diffusion_model.unet    import create_model
from distillation            import SpatialProjectionHead
from gpu_gmm                 import (fit_predict_gmm,
                                      name_clusters_adaptive,
                                      PATHOLOGY_COLORS)
from radiomic_features        import extract_all_features   # fallback only

HU_MIN, HU_MAX = -1000, 400

PATHOLOGY_RANGES = {
    "Emphysema":     (-1000, -750),
    "GGO":           (-750,  -600),
    "Healthy Lung":  (-600,  -500),
    "Fibrosis":      (-500,  -200),
    "Consolidation": (-200,   400),
}

def denorm_hu(x): return (x + 1.0) * 700.0 - 1000.0


# ─────────────────────────────────────────────────────────────────
# Model loading — supports both baseline and distillation checkpoints
# ─────────────────────────────────────────────────────────────────

def load_model(ckpt_path, patch_h=96, patch_d=32,
               use_proj_head=True, proj_dim=128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet = create_model(
        image_size=patch_h, num_channels=64, num_res_blocks=1,
        in_channels=2, out_channels=1,
        channel_mult=(1,2,3,4), attention_resolutions="16",
    ).to(device)

    diff = GaussianDiffusion(
        unet, image_size=patch_h, depth_size=patch_d,
        timesteps=250, loss_type='l1', channels=1,
    ).to(device)

    ckpt    = torch.load(ckpt_path, map_location=device)
    diff_key= 'ema' if 'ema' in ckpt else 'model'
    diff.load_state_dict(ckpt[diff_key])
    diff.eval()
    print(f"  UNet loaded from key='{diff_key}'  step={ckpt.get('step','?')}")

    # Load proj_head if available and requested
    proj_head = None
    bottleneck_ch = 4 * 64  # 256 = channel_mult[-1] * num_channels

    if use_proj_head:
        if 'proj_head' in ckpt:
            proj_head = SpatialProjectionHead(
                in_dim=bottleneck_ch,
                hidden_dim=bottleneck_ch,
                out_dim=proj_dim,
            ).to(device)
            proj_head.load_state_dict(ckpt['proj_head'])
            proj_head.eval()
            print(f"  proj_head loaded — output dim={proj_dim} (radiomic-aligned)")
        else:
            print("  WARNING: --use_proj_head requested but no 'proj_head' "
                  "key in checkpoint. Falling back to raw bottleneck features.")

    return diff, proj_head, device


# ─────────────────────────────────────────────────────────────────
# Feature extraction per patch
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def patch_features(diffusion, proj_head, ct_patch_norm, mask_patch,
                   timesteps, device):
    """
    Extract features for one patch.

    If proj_head available:
        → proj_head(bottleneck) → (N_lung, 128)  radiomic-aligned embedding
          This is the distillation output — tissue-type aware features.

    If proj_head is None (baseline):
        → raw bottleneck upsampled → (N_lung, 256)

    Both paths: average across timesteps for stability.
    """
    ct_t  = torch.from_numpy(ct_patch_norm[None, None]).float().to(device)
    msk_t = torch.from_numpy(mask_patch[None, None]).float().to(device)
    D, H, W = ct_patch_norm.shape

    feat_sum = None

    for t_val in timesteps:
        t     = torch.tensor([t_val], device=device).long()
        noisy = diffusion.q_sample(ct_t, t, torch.randn_like(ct_t))
        x_in  = torch.cat([noisy, msk_t], dim=1)

        bottleneck_out = []
        hook = diffusion.denoise_fn.middle_block.register_forward_hook(
            lambda m, i, o: bottleneck_out.append(o)
        )

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            _ = diffusion.denoise_fn(x_in, t)
        hook.remove()

        bn = bottleneck_out[0].float()  # (1, 256, D', H', W')

        if proj_head is not None:
            # ── Distillation path ─────────────────────────────
            # proj_head expects (B, 256, D', H', W') → outputs (B, 128)
            # This is a PATCH-LEVEL embedding, not voxel-level
            # We need spatial feature maps → use the intermediate conv output
            # before global avg pool

            # Get spatial feature map from proj_head conv layers only
            # (before the global avg pool + normalize)
            with torch.no_grad():
                # proj_head.forward: (1,256,D',H',W') → (1,128,D',H',W') spatial
                feat_spatial = proj_head(bn)

            # Upsample to full patch resolution
            feat = F.interpolate(
                feat_spatial,
                size=(D, H, W),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)  # (128, D, H, W)

        else:
            # ── Baseline path — raw bottleneck ────────────────
            feat = F.interpolate(
                bn,
                size=(D, H, W),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)  # (256, D, H, W)

        feat_np  = feat.permute(1, 2, 3, 0).cpu().numpy()  # (D,H,W,C)
        feat_sum = feat_np if feat_sum is None else feat_sum + feat_np

    feat_avg = feat_sum / len(timesteps)

    lung_pos  = np.argwhere(mask_patch >= 0.5)
    lung_feat = feat_avg[lung_pos[:, 0], lung_pos[:, 1], lung_pos[:, 2]]
    return lung_feat, lung_pos


# ─────────────────────────────────────────────────────────────────
# Feature collection — full volume, no overlap, lung voxels only
# ─────────────────────────────────────────────────────────────────

def collect_lung_features(ct_hu, mask, ct_norm,
                           diffusion, proj_head, device, args):
    Z, Y, X    = ct_hu.shape
    ph, pw, pd = args.patch_h, args.patch_w, args.patch_d
    pd = min(pd, Z); ph = min(ph, Y); pw = min(pw, X)

    # No-overlap patch grid — each voxel seen exactly once
    z_pos = list(range(0, Z - pd + 1, pd)) + ([Z - pd] if Z % pd != 0 else [])
    y_pos = list(range(0, Y - ph + 1, ph)) + ([Y - ph] if Y % ph != 0 else [])
    x_pos = list(range(0, X - pw + 1, pw)) + ([X - pw] if X % pw != 0 else [])
    total = len(z_pos) * len(y_pos) * len(x_pos)

    print(f"  Patches (no overlap): {total}")

    feat_list = []
    pos_list  = []

    pbar = tqdm(total=total, desc="  collecting features")

    for z0 in z_pos:
        for y0 in y_pos:
            for x0 in x_pos:
                z1, y1, x1 = z0 + pd, y0 + ph, x0 + pw
                mp = mask[z0:z1, y0:y1, x0:x1]

                if mp.mean() < 0.2:
                    pbar.update(1)
                    continue

                cp_norm = ct_norm[z0:z1, y0:y1, x0:x1]

                feat, local_pos = patch_features(
                    diffusion, proj_head, cp_norm, mp,
                    args.timesteps, device
                )

                if local_pos is None or len(local_pos) == 0:
                    pbar.update(1)
                    continue

                global_pos = local_pos.copy()
                global_pos[:, 0] += z0
                global_pos[:, 1] += y0
                global_pos[:, 2] += x0

                feat_list.append(feat.astype(np.float32))
                pos_list.append(global_pos.astype(np.int32))

                pbar.update(1)

    pbar.close()

    if not feat_list:
        return None, None

    all_features  = np.concatenate(feat_list, axis=0)
    all_positions = np.concatenate(pos_list,  axis=0)

    print(f"  Total lung voxels: {len(all_features):,}")
    print(f"  Feature dim: {all_features.shape[1]}")
    print(f"  Memory: {all_features.nbytes / 1e9:.2f} GB")

    return all_features, all_positions


# ─────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────

def cluster_lung_voxels(features, positions, vol_shape, args):
    N = len(features)
    k = args.k

    print(f"  Normalizing {N:,} voxels × {features.shape[1]} dims...")
    scaler   = StandardScaler()
    features = scaler.fit_transform(features)

    # PCA — reduce dims before clustering
    n_pca   = min(args.n_pca, features.shape[1], N - 1)
    n_sub   = min(args.subsample, N)
    idx_sub = np.random.choice(N, n_sub, replace=False)

    print(f"  PCA → {n_pca} dims (fit on {n_sub:,})...")
    pca = IncrementalPCA(n_components=n_pca, batch_size=10_000)
    pca.fit(features[idx_sub])

    f_pca = np.zeros((N, n_pca), dtype=np.float32)
    bs    = 50_000
    for i in range(0, N, bs):
        f_pca[i:i+bs] = pca.transform(features[i:i+bs])
    del features

    gpu_dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.clustering == 'gmm':
        labels, _, bic_scores, k = fit_predict_gmm(
            f_pca,
            k          = k,
            auto_k     = args.auto_k,
            k_min      = args.k_min,
            k_max      = args.k_max,
            device     = gpu_dev,
            random_state = 42,
        )
        labels = labels.astype(np.int8)
        if bic_scores:
            print(f"  BIC scores: {bic_scores}")
            print(f"  Auto-selected K={k}")
    else:
        print(f"  MiniBatchKMeans K={k}...")
        km = MiniBatchKMeans(
            n_clusters=k, random_state=42, n_init=10,
            max_iter=300, batch_size=min(10_000, n_sub)
        )
        km.fit(f_pca[idx_sub])
        labels = np.zeros(N, dtype=np.int8)
        for i in range(0, N, bs):
            labels[i:i+bs] = km.predict(f_pca[i:i+bs])

    # Build label volume
    label_vol = np.full(vol_shape, -1, dtype=np.int8)
    label_vol[positions[:, 0],
              positions[:, 1],
              positions[:, 2]] = labels

    return label_vol, k


# ─────────────────────────────────────────────────────────────────
# Smoothing
# ─────────────────────────────────────────────────────────────────

def smooth_labels(label_vol, mask, size=5):
    print(f"  Median filter size={size}...")
    k_max = int(label_vol[label_vol >= 0].max()) + 1
    tmp   = label_vol.copy().astype(np.int16)
    tmp[tmp < 0] = k_max
    tmp = median_filter(tmp, size=size)
    tmp[mask < 0.5] = -1
    out = tmp.astype(np.int8)
    out[out == k_max] = -1
    return out


# ─────────────────────────────────────────────────────────────────
# Fixed HU naming (fallback when adaptive_labels=False)
# ─────────────────────────────────────────────────────────────────

def name_clusters_fixed(ct_hu, mask, label_vol, k):
    info = {}
    for c in range(k):
        vox = ct_hu[(label_vol == c) & (mask >= 0.5)]
        hu  = float(np.median(vox)) if len(vox) > 0 else 0.0
        name = "Unknown"
        for n, (lo, hi) in PATHOLOGY_RANGES.items():
            if lo <= hu < hi:
                name = n
                break
        info[c] = {
            'name':     name,
            'mean_hu':  round(float(np.mean(vox)) if len(vox) > 0 else 0., 1),
            'n_voxels': int(len(vox)),
        }
    return info


# ─────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────

def save_visualization(ct_hu, mask, label_vol, cluster_info, k,
                       out_path, patient_id, method_name):
    Z, Y, X = ct_hu.shape
    hu_min, hu_max = -1100, ct_hu[mask >= 0.5].max()

    color_map = {c: PATHOLOGY_COLORS.get(info['name'], '#BBBBBB')
                 for c, info in cluster_info.items()}

    def hu_to_gray(hu_slice):
        return np.clip((hu_slice - hu_min) / (hu_max - hu_min + 1e-6), 0, 1)

    def make_rgb(hu_slice, lbl_slice, msk_slice):
        gray  = hu_to_gray(hu_slice)
        rgb   = np.stack([gray] * 3, axis=-1)
        for c, info in cluster_info.items():
            col = tuple(int(color_map[c].lstrip('#')[i:i+2], 16) / 255.
                        for i in (0, 2, 4))
            rgb[lbl_slice == c] = col
        rgb[msk_slice < 0.5] = 0
        return rgb

    z_slices = np.linspace(Z * 0.1, Z * 0.9, 5, dtype=int)

    fig = plt.figure(figsize=(22, 18), facecolor='#0d0d1a')
    fig.suptitle(
        f"{patient_id} — {method_name} (voxel-level)\n"
        f"K={k}  window [{hu_min},{hu_max:.0f}] HU",
        color='white', fontsize=11
    )

    ncols = 5
    # Row 1: CT slices
    for i, z in enumerate(z_slices):
        ax = fig.add_subplot(3, ncols, i + 1)
        ax.imshow(hu_to_gray(ct_hu[z].T), cmap='gray',
                  origin='lower', vmin=0, vmax=1)
        ax.set_title(f"z={z}", color='white', fontsize=8)
        ax.axis('off')

    # Row 2: segmentation slices
    for i, z in enumerate(z_slices):
        ax = fig.add_subplot(3, ncols, ncols + i + 1)
        ax.imshow(make_rgb(ct_hu[z], label_vol[z], mask[z]).transpose(1, 0, 2),
                  origin='lower')
        ax.axis('off')

    # Row 3: coronal, sagittal, histogram, info
    ax_c = fig.add_subplot(3, ncols, 2 * ncols + 1)
    mid_y = Y // 2
    ax_c.imshow(make_rgb(ct_hu[:, mid_y, :],
                         label_vol[:, mid_y, :],
                         mask[:, mid_y, :]).transpose(1, 0, 2),
                origin='lower')
    ax_c.set_title(f"Coronal y={mid_y}", color='white', fontsize=8)
    ax_c.axis('off')

    ax_s = fig.add_subplot(3, ncols, 2 * ncols + 2)
    mid_x = X // 2
    ax_s.imshow(make_rgb(ct_hu[:, :, mid_x],
                         label_vol[:, :, mid_x],
                         mask[:, :, mid_x]).transpose(1, 0, 2),
                origin='lower')
    ax_s.set_title(f"Sagittal x={mid_x}", color='white', fontsize=8)
    ax_s.axis('off')

    # Histogram
    ax_h = fig.add_subplot(3, ncols, 2 * ncols + 3)
    ax_h.set_facecolor('#1a1a2e')
    lung_hu_vals = ct_hu[mask >= 0.5]
    bins = np.linspace(-1100, 400, 80)
    ax_h.hist(lung_hu_vals, bins=bins, color='#555', alpha=0.3,
              density=True, label='all lung')
    for c, info in cluster_info.items():
        vox = ct_hu[(label_vol == c) & (mask >= 0.5)]
        if len(vox) > 0:
            ax_h.hist(vox, bins=bins, alpha=0.5,
                      color=color_map[c], density=True,
                      label=f"C{c}")
    ax_h.axvline(-750, color='white', lw=0.5, ls='--')
    ax_h.axvline(-600, color='white', lw=0.5, ls='--')
    ax_h.axvline(-500, color='white', lw=0.5, ls='--')
    ax_h.set_title("HU per cluster", color='white', fontsize=8)
    ax_h.tick_params(colors='white', labelsize=6)
    ax_h.legend(fontsize=6, facecolor='#1a1a2e', labelcolor='white')

    # Info box
    ax_i = fig.add_subplot(3, ncols, 2 * ncols + 4)
    ax_i.axis('off')
    txt = f"K={k}  {method_name}\n"
    for c, info in cluster_info.items():
        pct = info.get('percentile', '')
        pct_str = f"  pct={pct:.0f}%" if pct != '' else ''
        txt += (f"C{c}: {info['name']:<15} "
                f"μHU={info['mean_hu']:7.1f}  "
                f"n={info['n_voxels']:,}{pct_str}\n")
    ax_i.text(0.05, 0.95, txt, transform=ax_i.transAxes,
              color='white', fontsize=7, va='top',
              fontfamily='monospace',
              bbox=dict(facecolor='#1a1a2e', alpha=0.8))

    # Legend
    ax_l = fig.add_subplot(3, ncols, 2 * ncols + 5)
    ax_l.axis('off')
    ax_l.set_facecolor('#0d0d1a')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, label=n)
               for n, c in PATHOLOGY_COLORS.items()]
    ax_l.legend(handles=handles, loc='center', fontsize=9,
                title="Pathology", title_fontsize=9,
                facecolor='#1a1a2e', labelcolor='white', edgecolor='#444')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  PNG: {out_path}")


# ─────────────────────────────────────────────────────────────────
# NIfTI output
# ─────────────────────────────────────────────────────────────────

def save_niftis(label_vol, cluster_info, k, out_dir, patient_id):
    affine = np.eye(4)
    nib.save(
        nib.Nifti1Image(
            np.transpose((label_vol + 1).astype(np.int16), (0, 1, 2)),
            affine),
        os.path.join(out_dir, f"{patient_id}_segmentation.nii.gz")
    )
    from collections import defaultdict
    pc = defaultdict(list)
    for c, info in cluster_info.items():
        pc[info['name']].append(c)
    for path, cids in pc.items():
        b = np.zeros(label_vol.shape, dtype=np.uint8)
        for c in cids:
            b[label_vol == c] = 1
        safe = path.replace('/', '_').replace(' ', '_')
        nib.save(
            nib.Nifti1Image(
                np.transpose(b.astype(np.float32), (0, 1, 2)), affine),
            os.path.join(out_dir, f"{patient_id}_{safe}.nii.gz")
        )
    print(f"  NIfTIs: {out_dir}")


# ─────────────────────────────────────────────────────────────────
# Process one patient
# ─────────────────────────────────────────────────────────────────

def process_one(npz_path, diffusion, proj_head, device, args, method_name):
    patient_id = os.path.basename(npz_path).replace('.npz', '')
    print(f"\n{'─'*60}\nPatient: {patient_id}")

    data    = np.load(npz_path, allow_pickle=True)
    ct_norm = data['ct'].astype(np.float32)
    mask    = data['mask'].astype(np.float32)
    ct_hu   = denorm_hu(ct_norm)

    lung_hu = ct_hu[mask >= 0.5]
    print(f"Volume: {ct_hu.shape}  "
          f"HU=[{lung_hu.min():.0f},{lung_hu.max():.0f}]  "
          f"median={np.median(lung_hu):.0f}  "
          f"lung={mask.mean()*100:.0f}%")

    # ── Collect features ─────────────────────────────────────────
    features, positions = collect_lung_features(
        ct_hu, mask, ct_norm, diffusion, proj_head, device, args
    )
    if features is None:
        print("  SKIP: no lung patches")
        return None

    # ── Cluster ──────────────────────────────────────────────────
    label_vol, k_used = cluster_lung_voxels(
        features, positions, ct_hu.shape, args
    )
    del features, positions

    # ── Smooth ───────────────────────────────────────────────────
    label_vol = smooth_labels(label_vol, mask, size=args.smooth_size)

    # ── Name clusters ────────────────────────────────────────────
    if args.adaptive_labels:
        cluster_info = name_clusters_adaptive(
            ct_hu, mask, label_vol, k_used)
    else:
        cluster_info = name_clusters_fixed(
            ct_hu, mask, label_vol, k_used)

    print("  Clusters:")
    for c, info in cluster_info.items():
        pct = info.get('percentile', '')
        pct_str = f"  pct={pct:.0f}%" if pct != '' else ''
        print(f"    C{c}: {info['name']:<15} "
              f"μHU={info['mean_hu']:7.1f}  "
              f"n={info['n_voxels']:,}{pct_str}")

    # ── Save ─────────────────────────────────────────────────────
    pat_dir = os.path.join(args.out, patient_id)
    os.makedirs(pat_dir, exist_ok=True)

    save_visualization(
        ct_hu, mask, label_vol, cluster_info, k_used,
        os.path.join(pat_dir, f"{patient_id}_{method_name}.png"),
        patient_id, method_name
    )
    save_niftis(label_vol, cluster_info, k_used, pat_dir, patient_id)

    result = {
        'patient_id': patient_id,
        'method':     method_name,
        'k':          k_used,
        'clusters':   {str(c): info for c, info in cluster_info.items()},
    }
    with open(os.path.join(pat_dir,
              f"{patient_id}_{method_name}.json"), 'w') as f:
        json.dump(result, f, indent=2)

    return result


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    ap.add_argument('--npz',          type=str, default=None)
    ap.add_argument('--npz_dir',      type=str, default=None)
    ap.add_argument('--dataset',      type=str, default=None)
    ap.add_argument('--max_patients', type=int, default=None)

    # Model
    ap.add_argument('--checkpoint',   type=str, required=True)
    ap.add_argument('--patch_h',      type=int, default=96)
    ap.add_argument('--patch_w',      type=int, default=96)
    ap.add_argument('--patch_d',      type=int, default=32)
    ap.add_argument('--proj_dim',     type=int, default=128)
    ap.add_argument('--use_proj_head',
                    action=argparse.BooleanOptionalAction, default=True,
                    help='Use proj_head from distillation checkpoint')
    ap.add_argument('--timesteps',    type=int, nargs='+', default=[100],
                    help='Diffusion timesteps for feature extraction')

    # Clustering
    ap.add_argument('--k',            type=int, default=5)
    ap.add_argument('--clustering',   type=str, default='gmm',
                    choices=['kmeans', 'gmm'])
    ap.add_argument('--auto_k',
                    action=argparse.BooleanOptionalAction, default=False,
                    help='Auto-select K via BIC (GMM only)')
    ap.add_argument('--k_min',        type=int, default=2)
    ap.add_argument('--k_max',        type=int, default=8)
    ap.add_argument('--n_pca',        type=int, default=30)
    ap.add_argument('--subsample',    type=int, default=500_000)

    # Labels
    ap.add_argument('--adaptive_labels',
                    action=argparse.BooleanOptionalAction, default=True,
                    help='Patient-adaptive naming via HU percentiles')
    ap.add_argument('--smooth_size',  type=int, default=5)

    ap.add_argument('--out',          type=str,
                    default='./segmentation_results')

    args = ap.parse_args()

    # Build method name for output files
    parts = ['distill' if args.use_proj_head else 'baseline']
    parts.append(f"t{'_'.join(map(str, args.timesteps))}")
    parts.append(args.clustering)
    parts.append('autoK' if args.auto_k else f"K{args.k}")
    parts.append('adaptive' if args.adaptive_labels else 'fixed')
    method_name = "_".join(parts)
    print(f"\nMethod: {method_name}")

    os.makedirs(args.out, exist_ok=True)

    # Load model
    diffusion, proj_head, device = load_model(
        args.checkpoint,
        patch_h       = args.patch_h,
        patch_d       = args.patch_d,
        use_proj_head = args.use_proj_head,
        proj_dim      = args.proj_dim,
    )

    # Patient list
    if args.npz:
        npz_files = [args.npz]
    elif args.npz_dir:
        npz_files = sorted([
            os.path.join(args.npz_dir, f)
            for f in os.listdir(args.npz_dir)
            if f.endswith('.npz') and '__' in f
        ])
        if args.dataset:
            npz_files = [f for f in npz_files
                         if os.path.basename(f).startswith(args.dataset + '__')]
        if args.max_patients:
            npz_files = npz_files[:args.max_patients]
    else:
        ap.error("Provide --npz or --npz_dir")

    print(f"Processing {len(npz_files)} volume(s)")

    all_results = []
    for f in npz_files:
        try:
            r = process_one(f, diffusion, proj_head, device,
                            args, method_name)
            if r:
                all_results.append(r)
        except Exception as e:
            import traceback
            print(f"  FAILED {os.path.basename(f)}: {e}")
            traceback.print_exc()

    if len(all_results) > 1:
        with open(os.path.join(args.out,
                  f"summary_{method_name}.json"), 'w') as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone. Results: {args.out}")


if __name__ == '__main__':
    main()

'''

```

The key difference from v2:
```
Feature dim:  256 raw bottleneck → 128 proj_head embedding
              proj_head IS the distillation output — radiomic-aware
              No separate radiomics concatenation needed at inference

Memory:       18M × 128 dims × 4 bytes = 9.2 GB  (vs 20 GB before)
Speed:        ~same — bottleneck extraction dominates, not feature dim


'''
