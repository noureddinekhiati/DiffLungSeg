#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_volume.py
───────────────────
Run the trained diffusion model on WHOLE volumes (not just patches).

Strategy:
  1. Slide a 96x96x32 patch window over the full lung volume
  2. For each patch: generate a denoised reconstruction using DPM-Solver
  3. Compute reconstruction error (MSE) between input and reconstruction
     → high error = region the model struggles to reconstruct
     → these are anomalous/pathological regions
  4. Stitch all patch errors back to full volume
  5. Save as NIfTI + PNG slices for visual inspection

Why reconstruction error?
  A diffusion model trained on healthy+pathological lung learns the
  "normal" distribution. Regions that deviate from what it learned
  produce higher reconstruction error — this is the unsupervised
  anomaly signal we use before adding distillation.

Usage:
    
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Import model components ────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_model.trainer import GaussianDiffusion
from diffusion_model.unet import create_model

HU_MIN, HU_MAX = -1000, 400


# ─────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str,
               patch_h: int = 96,
               patch_w: int = 96,
               patch_d: int = 32,
               device: torch.device = None) -> GaussianDiffusion:
    """Load trained GaussianDiffusion model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model(
        image_size        = patch_h,
        num_channels      = 64,
        num_res_blocks    = 1,
        in_channels       = 2,       # CT + lung mask
        out_channels      = 1,
        channel_mult      = (1, 2, 3, 4),
        attention_resolutions = "16",
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size = patch_h,
        depth_size = patch_d,
        timesteps  = 250,
        loss_type  = 'l1',
        channels   = 1,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Try EMA weights first (better quality), fallback to model weights
    if 'ema' in ckpt:
        diffusion.load_state_dict(ckpt['ema'])
        print(f"  Loaded EMA weights from step {ckpt.get('step', '?')}")
    elif 'model' in ckpt:
        diffusion.load_state_dict(ckpt['model'])
        print(f"  Loaded model weights from step {ckpt.get('step', '?')}")
    else:
        diffusion.load_state_dict(ckpt)
        print(f"  Loaded weights directly")

    diffusion.eval()
    return diffusion, device


# ─────────────────────────────────────────────────────────────────
# Patch-based inference on full volume
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def reconstruct_volume(diffusion:   GaussianDiffusion,
                        ct_norm:    np.ndarray,
                        mask:       np.ndarray,
                        patch_h:    int = 96,
                        patch_w:    int = 96,
                        patch_d:    int = 32,
                        stride_xy:  int = 48,   # 50% overlap in XY
                        stride_z:   int = 16,   # 50% overlap in Z
                        dpm_steps:  int = 20,
                        device:     torch.device = None,
                        min_lung:   float = 0.3) -> dict:
    """
    Slide patch window over full volume, reconstruct each patch,
    compute reconstruction error, stitch back to full volume.

    Returns dict with:
        recon_vol   : (Z,Y,X) reconstructed CT (normalized [-1,1])
        error_vol   : (Z,Y,X) per-voxel reconstruction error (MSE)
        count_vol   : (Z,Y,X) number of patches contributing to each voxel
        anomaly_map : (Z,Y,X) normalized anomaly score [0,1]
    """
    Z, Y, X = ct_norm.shape

    recon_vol = np.zeros((Z, Y, X), dtype=np.float32)
    error_vol = np.zeros((Z, Y, X), dtype=np.float32)
    count_vol = np.zeros((Z, Y, X), dtype=np.float32)

    # Clamp patch sizes to volume
    pd = min(patch_d, Z)
    ph = min(patch_h, Y)
    pw = min(patch_w, X)

    # Generate all patch positions
    z_starts = list(range(0, max(1, Z - pd + 1), stride_z))
    y_starts = list(range(0, max(1, Y - ph + 1), stride_xy))
    x_starts = list(range(0, max(1, X - pw + 1), stride_xy))

    # Make sure we cover the last patch
    if z_starts[-1] + pd < Z: z_starts.append(Z - pd)
    if y_starts[-1] + ph < Y: y_starts.append(Y - ph)
    if x_starts[-1] + pw < X: x_starts.append(X - pw)

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    lung_patches  = 0
    skipped       = 0

    pbar = tqdm(total=total_patches, desc="  Reconstructing patches", leave=False)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                z1, y1, x1 = z0+pd, y0+ph, x0+pw

                # Check lung coverage
                mp = mask[z0:z1, y0:y1, x0:x1]
                if mp.mean() < min_lung:
                    pbar.update(1)
                    skipped += 1
                    continue

                lung_patches += 1

                # Prepare tensors
                ct_patch   = ct_norm[z0:z1, y0:y1, x0:x1]
                mask_patch = mp

                ct_t   = torch.from_numpy(ct_patch[np.newaxis, np.newaxis]).float().to(device)
                mask_t = torch.from_numpy(mask_patch[np.newaxis, np.newaxis]).float().to(device)

                # Add noise at a fixed mid timestep then reconstruct
                # t=100 gives a good balance: enough noise to be interesting,
                # but not so much that reconstruction is random
                t_val = torch.tensor([100], device=device).long()
                noise = torch.randn_like(ct_t)

                # Noisy version of input patch
                ct_noisy = diffusion.q_sample(ct_t, t_val, noise)

                # Denoise: predict what the model thinks this should look like
                with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
                    x_input    = torch.cat([ct_noisy, mask_t], dim=1)
                    noise_pred = diffusion.denoise_fn(x_input, t_val)

                    # Reconstruct x0 from predicted noise
                    ct_recon = diffusion.predict_start_from_noise(
                        ct_noisy, t_val, noise_pred
                    ).clamp(-1, 1)

                # Reconstruction error per voxel
                error = (ct_t - ct_recon).abs().squeeze().cpu().numpy()
                recon = ct_recon.squeeze().cpu().numpy()

                # Accumulate (average over overlapping patches)
                recon_vol[z0:z1, y0:y1, x0:x1] += recon
                error_vol[z0:z1, y0:y1, x0:x1] += error
                count_vol[z0:z1, y0:y1, x0:x1] += 1.0

                pbar.update(1)

    pbar.close()

    print(f"  Lung patches: {lung_patches}/{total_patches} "
          f"({skipped} skipped, outside lung)")

    # Average over overlapping patches
    valid = count_vol > 0
    recon_vol[valid] /= count_vol[valid]
    error_vol[valid] /= count_vol[valid]

    # Mask to lung only
    recon_vol[mask < 0.5] = ct_norm[mask < 0.5]   # keep original outside lung
    error_vol[mask < 0.5] = 0.0

    # Normalize anomaly map to [0,1] within lung
    lung_errors = error_vol[mask >= 0.5]
    if len(lung_errors) > 0:
        p5  = np.percentile(lung_errors, 5)
        p95 = np.percentile(lung_errors, 95)
        anomaly_map = np.clip((error_vol - p5) / (p95 - p5 + 1e-8), 0, 1)
        anomaly_map[mask < 0.5] = 0.0
    else:
        anomaly_map = error_vol.copy()

    return {
        'recon_vol':   recon_vol,
        'error_vol':   error_vol,
        'count_vol':   count_vol,
        'anomaly_map': anomaly_map,
    }


# ─────────────────────────────────────────────────────────────────
# Denormalization
# ─────────────────────────────────────────────────────────────────

def denorm_hu(ct_norm: np.ndarray) -> np.ndarray:
    """
    Inverse of: norm = (HU + 1000) / 700 - 1
    So:         HU   = (norm + 1) * 700 - 1000
    """
    return (ct_norm + 1.0) * 700.0 - 1000.0


# ─────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────

def save_visualization(ct_hu:       np.ndarray,
                        mask:        np.ndarray,
                        recon_hu:    np.ndarray,
                        error_vol:   np.ndarray,
                        anomaly_map: np.ndarray,
                        out_path:    str,
                        patient_id:  str):
    """
    Multi-panel PNG showing:
      Row 0: Original CT (axial slices)
      Row 1: Reconstructed CT
      Row 2: Reconstruction error (absolute difference)
      Row 3: Anomaly map (normalized, thresholded)
    """
    # Pick 5 evenly spaced lung slices
    lung_z    = np.where(mask.any(axis=(1, 2)))[0]
    n_slices  = 5
    slice_ids = lung_z[np.linspace(0, len(lung_z)-1, n_slices, dtype=int)] \
                if len(lung_z) >= n_slices \
                else np.linspace(0, ct_hu.shape[0]-1, n_slices, dtype=int)

    fig, axes = plt.subplots(4, n_slices, figsize=(n_slices*4, 4*4))
    fig.suptitle(
        f"Diffusion Reconstruction — {patient_id}\n"
        f"(t=100 noisy → denoise → error map)",
        fontsize=13, fontweight='bold'
    )

    for col, sl in enumerate(slice_ids):
        # Row 0: original CT
        ax = axes[0, col]
        im = ax.imshow(ct_hu[sl], cmap='gray', vmin=-1000, vmax=400)
        ax.set_title(f"Original  z={sl}", fontsize=9)
        ax.axis('off')
        if col == n_slices - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, label='HU')

        # Row 1: reconstruction
        ax = axes[1, col]
        im = ax.imshow(recon_hu[sl], cmap='gray', vmin=-1000, vmax=400)
        ax.set_title(f"Recon  z={sl}", fontsize=9)
        ax.axis('off')
        if col == n_slices - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, label='HU')

        # Row 2: absolute error (HU)
        ax = axes[2, col]
        im = ax.imshow(error_vol[sl], cmap='hot',
                       vmin=0, vmax=np.percentile(error_vol[mask>=0.5], 95))
        ax.set_title(f"|Error|  z={sl}", fontsize=9)
        ax.axis('off')
        if col == n_slices - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, label='|Δ| norm')

        # Row 3: anomaly map overlaid on CT
        ax = axes[3, col]
        ax.imshow(ct_hu[sl], cmap='gray', vmin=-1000, vmax=400)
        anom_sl = anomaly_map[sl]
        # Only show high anomaly regions (top 30%)
        anom_overlay = np.ma.masked_where(anom_sl < 0.7, anom_sl)
        im = ax.imshow(anom_overlay, cmap='YlOrRd', alpha=0.7,
                       vmin=0.7, vmax=1.0)
        ax.set_title(f"Anomaly>0.7  z={sl}", fontsize=9)
        ax.axis('off')
        if col == n_slices - 1:
            plt.colorbar(im, ax=ax, fraction=0.046, label='anomaly')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  PNG saved: {out_path}")


def save_niftis(ct_hu:       np.ndarray,
                recon_hu:    np.ndarray,
                error_vol:   np.ndarray,
                anomaly_map: np.ndarray,
                out_dir:     str,
                patient_id:  str):
    """Save full volume NIfTIs — load in ITK-SNAP or 3D Slicer."""
    affine = np.eye(4)
    files  = {
        'ct_original':   ct_hu,
        'ct_recon':      recon_hu,
        'error_map':     error_vol,
        'anomaly_map':   anomaly_map,
    }
    for name, vol in files.items():
        # Transpose to (X,Y,Z) for NIfTI convention
        vol_nii = np.transpose(vol.astype(np.float32), (0, 1, 2))
        path    = os.path.join(out_dir, f"{patient_id}_{name}.nii.gz")
        nib.save(nib.Nifti1Image(vol_nii, affine), path)
    print(f"  NIfTIs saved in: {out_dir}")


# ─────────────────────────────────────────────────────────────────
# Per-patient stats
# ─────────────────────────────────────────────────────────────────

def compute_stats(ct_hu, recon_hu, error_vol, anomaly_map, mask):
    """Summary statistics for one patient."""
    lung_mask = mask >= 0.5

    # HU ranges for known pathologies
    emphysema_mask    = lung_mask & (ct_hu < -700)
    consolidation_mask= lung_mask & (ct_hu > -300)
    healthy_mask      = lung_mask & (ct_hu >= -700) & (ct_hu <= -300)

    def mean_anom(m):
        return float(anomaly_map[m].mean()) if m.sum() > 0 else 0.0

    return {
        'mean_recon_error_lung':     float(error_vol[lung_mask].mean()),
        'mean_anomaly_lung':         float(anomaly_map[lung_mask].mean()),
        'mean_anomaly_emphysema':    mean_anom(emphysema_mask),
        'mean_anomaly_healthy':      mean_anom(healthy_mask),
        'mean_anomaly_consolidation':mean_anom(consolidation_mask),
        'pct_high_anomaly':          float((anomaly_map[lung_mask] > 0.7).mean() * 100),
        'lung_voxels':               int(lung_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def process_one(npz_path, diffusion, device, args):
    patient_id = os.path.basename(npz_path).replace('.npz', '')
    print(f"\n{'─'*60}")
    print(f"Patient: {patient_id}")

    # Load
    data    = np.load(npz_path, allow_pickle=True)
    ct_norm = data['ct'].astype(np.float32)
    mask    = data['mask'].astype(np.float32)
    ct_hu   = denorm_hu(ct_norm)

    print(f"Volume : {ct_hu.shape}  "
          f"HU=[{ct_hu[mask>=0.5].min():.0f}, {ct_hu[mask>=0.5].max():.0f}]  "
          f"Lung={mask.mean()*100:.1f}%")

    # Reconstruct
    results  = reconstruct_volume(
        diffusion, ct_norm, mask,
        patch_h   = args.patch_h,
        patch_w   = args.patch_w,
        patch_d   = args.patch_d,
        stride_xy = args.patch_h // 2,
        stride_z  = args.patch_d // 2,
        dpm_steps = args.steps,
        device    = device,
    )

    recon_hu    = denorm_hu(results['recon_vol'])
    error_vol   = results['error_vol']
    anomaly_map = results['anomaly_map']

    # Save outputs
    pat_dir = os.path.join(args.out, patient_id)
    os.makedirs(pat_dir, exist_ok=True)

    # PNG visualization
    png_path = os.path.join(pat_dir, f"{patient_id}_inference.png")
    save_visualization(ct_hu, mask, recon_hu, error_vol,
                       anomaly_map, png_path, patient_id)

    # NIfTIs
    save_niftis(ct_hu, recon_hu, error_vol, anomaly_map,
                pat_dir, patient_id)

    # Stats JSON
    stats    = compute_stats(ct_hu, recon_hu, error_vol, anomaly_map, mask)
    json_path = os.path.join(pat_dir, f"{patient_id}_stats.json")
    with open(json_path, 'w') as f:
        json.dump({'patient_id': patient_id, **stats}, f, indent=2)

    print(f"  Mean recon error (lung): {stats['mean_recon_error_lung']:.4f}")
    print(f"  High anomaly voxels    : {stats['pct_high_anomaly']:.1f}%")
    print(f"  Anomaly emphysema zone : {stats['mean_anomaly_emphysema']:.4f}")
    print(f"  Anomaly consolidation  : {stats['mean_anomaly_consolidation']:.4f}")

    return stats


def main():
    ap = argparse.ArgumentParser()

    # Input — one NPZ or whole folder
    ap.add_argument('--npz',         type=str, default=None,
                    help='Single NPZ file to process')
    ap.add_argument('--npz_dir',     type=str, default=None,
                    help='Folder of NPZ files (process all)')
    ap.add_argument('--dataset',     type=str, default=None,
                    help='Filter by dataset prefix e.g. bpco')
    ap.add_argument('--max_patients',type=int, default=None,
                    help='Max patients to process (for quick test)')

    # Model
    ap.add_argument('--checkpoint',  type=str, required=True,
                    help='Path to model-N.pt checkpoint')
    ap.add_argument('--patch_h',     type=int, default=96)
    ap.add_argument('--patch_w',     type=int, default=96)
    ap.add_argument('--patch_d',     type=int, default=32)
    ap.add_argument('--steps',       type=int, default=20,
                    help='DPM-Solver steps (20 is fast, 50 is higher quality)')

    # Output
    ap.add_argument('--out',         type=str, default='./inference_results',
                    help='Output directory')

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    diffusion, device = load_model(
        args.checkpoint,
        patch_h = args.patch_h,
        patch_w = args.patch_w,
        patch_d = args.patch_d,
    )
    print(f"Device: {device}")

    # Collect NPZ files
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
        print("ERROR: provide --npz or --npz_dir")
        return

    print(f"Processing {len(npz_files)} volume(s)")

    # Run inference
    all_stats = []
    for npz_path in npz_files:
        try:
            stats = process_one(npz_path, diffusion, device, args)
            all_stats.append(stats)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # Global summary
    if len(all_stats) > 1:
        print(f"\n{'='*60}")
        print(f"GLOBAL SUMMARY ({len(all_stats)} patients)")
        for key in ['mean_recon_error_lung', 'mean_anomaly_emphysema',
                    'mean_anomaly_consolidation', 'pct_high_anomaly']:
            vals = [s[key] for s in all_stats]
            print(f"  {key:<35}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        summary_path = os.path.join(args.out, 'global_stats.json')
        with open(summary_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nFull stats: {summary_path}")


if __name__ == '__main__':
    main()

"""

"""
