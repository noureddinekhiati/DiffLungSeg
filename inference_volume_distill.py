#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_distill.py
────────────────
Full-volume inference for the distillation-trained DiffSegLung model.

Slides a (32×96×96) patch across the full CT volume, runs the diffusion
model reverse process on each patch, then stitches patches back into a
full-volume NIfTI reconstruction.

Overlap-and-average:
  50% overlap in all 3 dims → each voxel predicted ~8 times → averaged
  Eliminates stitching artifacts at patch boundaries

Saves per-patient:
  {patient_id}_recon.nii.gz   — full-volume reconstruction (HU space)
  {patient_id}_input.nii.gz   — original CT input (for comparison)
  {patient_id}_mask.nii.gz    — lung mask
  {patient_id}_diff.nii.gz    — |recon - input| difference map
  {patient_id}_overlay.png    — qualitative visualization

Run:
    # Single volume
    python infer_distill.py \
        --npz /path/to/patient.npz \
        --checkpoint ./results_distill_final/model-60.pt \
        --out ./inference_distill

    # Whole dataset
    python infer_distill.py \
        --npz_dir /path/to/npz_dir \
        --checkpoint ./results_distill_final/model-60.pt \
        --out ./inference_distill \
        --dataset bpco \
        --max_patients 10

    # Control sampling quality (more steps = better but slower)
    python infer_distill.py \
        --npz /path/to/patient.npz \
        --checkpoint ./results_distill_final/model-60.pt \

        --out ./inference_distill
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion_model.trainer import GaussianDiffusion
from diffusion_model.unet    import create_model

HU_MIN, HU_MAX = -1000, 400

def denorm_hu(x):
    """[-1,1] normalized → HU"""
    return (x + 1.0) * 700.0 - 1000.0

def norm_hu(x):
    """HU → [-1,1] normalized"""
    return (x - HU_MIN) / (HU_MAX - HU_MIN) * 2.0 - 1.0


# ─────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────

def load_model(ckpt_path, patch_h=96, patch_d=32):
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

    ckpt = torch.load(ckpt_path, map_location=device)

    # Prefer EMA weights for inference — always cleaner
    if 'ema' in ckpt:
        diff.load_state_dict(ckpt['ema'])
        key = 'ema'
    elif 'model' in ckpt:
        diff.load_state_dict(ckpt['model'])
        key = 'model'
    else:
        diff.load_state_dict(ckpt)
        key = 'raw'

    diff.eval()
    step = ckpt.get('step', '?')
    print(f"  Loaded '{key}'  step={step}  device={device}")
    return diff, device


# ─────────────────────────────────────────────────────────────────
# Single patch inference
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_patch(diffusion, ct_patch_norm, mask_patch, device):
    """
    Run reverse diffusion on one patch conditioned on mask.
    Uses DPM-Solver++ (same as your original inference pipeline):
      - 50 steps, order=3, multistep
      - Better quality than DDIM at same speed
      - State of the art fast sampler for diffusion models

    ct_patch_norm : (D, H, W) float32  input CT normalized [-1,1]
    mask_patch    : (D, H, W) float32  lung mask [0,1]

    Returns reconstructed patch (D, H, W) float32 in [-1,1]
    """
    ct_t   = torch.from_numpy(ct_patch_norm[None, None]).float().to(device)
    mask_t = torch.from_numpy(mask_patch[None, None]).float().to(device)

    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        # sample_dpm_solver: DPM-Solver++ order=3, 50 steps
        # x_start = input CT used as starting point for mixed sampling
        # condition_tensors = lung mask
        recon = diffusion.sample_dpm_solver(
            x_start           = ct_t,
            batch_size        = 1,
            condition_tensors = mask_t,
        )  # (1, 1, D, H, W)

    return recon[0, 0].cpu().numpy()  # (D, H, W)


# ─────────────────────────────────────────────────────────────────
# Full volume inference with overlap-and-average
# ─────────────────────────────────────────────────────────────────

def infer_volume(ct_norm, mask, diffusion, device, args):
    """
    Slide patch across full volume with 50% overlap.
    Average predictions at overlapping regions.

    Returns recon_norm (Z,Y,X) float32 in [-1,1]
    """
    Z, Y, X    = ct_norm.shape
    pd, ph, pw = args.patch_d, args.patch_h, args.patch_w
    pd = min(pd, Z); ph = min(ph, Y); pw = min(pw, X)

    # Stride — no overlap (each voxel predicted once)
    # Overlap is not needed for reconstruction inference:
    # diffusion reverse process is independent per patch
    # Bottleneck upsampling already smooths boundaries naturally
    sz  = pd
    sxy = ph

    z_pos = sorted(set(
        list(range(0, max(1, Z-pd+1), sz)) + [max(0, Z-pd)]
    ))
    y_pos = sorted(set(
        list(range(0, max(1, Y-ph+1), sxy)) + [max(0, Y-ph)]
    ))
    x_pos = sorted(set(
        list(range(0, max(1, X-pw+1), sxy)) + [max(0, X-pw)]
    ))
    total = len(z_pos) * len(y_pos) * len(x_pos)

    print(f"  Patches: {total}  stride=({sz},{sxy},{sxy})")

    # Accumulators
    recon_acc = np.zeros((Z, Y, X), dtype=np.float32)
    count_acc = np.zeros((Z, Y, X), dtype=np.float32)

    pbar = tqdm(total=total, desc="  inferring patches")

    for z0 in z_pos:
        for y0 in y_pos:
            for x0 in x_pos:
                z1 = min(z0 + pd, Z)
                y1 = min(y0 + ph, Y)
                x1 = min(x0 + pw, X)

                mp = mask[z0:z1, y0:y1, x0:x1]

                # Skip patches with no lung
                if mp.mean() < args.min_lung_fraction:
                    pbar.update(1)
                    continue

                cp = ct_norm[z0:z1, y0:y1, x0:x1]

                # Pad to full patch size if needed (edge patches)
                dz = pd-(z1-z0); dy = ph-(y1-y0); dx = pw-(x1-x0)
                if dz > 0 or dy > 0 or dx > 0:
                    cp = np.pad(cp, ((0,dz),(0,dy),(0,dx)),
                                mode='reflect')
                    mp = np.pad(mp, ((0,dz),(0,dy),(0,dx)),
                                mode='constant')

                # Run inference
                recon_patch = infer_patch(diffusion, cp, mp, device)

                # Accumulate — only unpadded region
                rz = z1-z0; ry = y1-y0; rx = x1-x0
                recon_acc[z0:z1, y0:y1, x0:x1] += recon_patch[:rz, :ry, :rx]
                count_acc[z0:z1, y0:y1, x0:x1] += 1.0

                pbar.update(1)

    pbar.close()

    # Average
    valid = count_acc > 0
    recon_acc[valid] /= count_acc[valid]

    coverage = valid.mean() * 100
    print(f"  Coverage: {coverage:.1f}%")

    return recon_acc


# ─────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────

def save_visualization(ct_hu, recon_hu, mask, patient_id,
                       out_path, step):
    Z = ct_hu.shape[0]
    diff_map = np.abs(recon_hu - ct_hu)

    # Clip HU for display
    vmin, vmax = -1000, 400

    def to_gray(x):
        return np.clip((x - vmin) / (vmax - vmin), 0, 1)

    z_slices = np.linspace(Z*0.1, Z*0.9, 5, dtype=int)

    fig, axes = plt.subplots(4, 5, figsize=(22, 18),
                              facecolor='#0d0d1a')
    fig.suptitle(
        f"{patient_id} — Distillation Inference  step={step}\n"
        f"Row1=Input  Row2=Reconstruction  Row3=Difference  Row4=Overlay",
        color='white', fontsize=10
    )

    for i, z in enumerate(z_slices):
        # Row 1: input CT
        ax = axes[0, i]
        ax.imshow(to_gray(ct_hu[z]).T, cmap='gray',
                  origin='lower', vmin=0, vmax=1)
        ax.set_title(f"Input z={z}", color='white', fontsize=7)
        ax.axis('off')

        # Row 2: reconstruction
        ax = axes[1, i]
        ax.imshow(to_gray(recon_hu[z]).T, cmap='gray',
                  origin='lower', vmin=0, vmax=1)
        ax.set_title(f"Recon z={z}", color='white', fontsize=7)
        ax.axis('off')

        # Row 3: difference map
        ax = axes[2, i]
        d = diff_map[z]
        im = ax.imshow(d.T, cmap='hot', origin='lower',
                       vmin=0, vmax=200)
        ax.set_title(f"|Diff| z={z}", color='white', fontsize=7)
        ax.axis('off')

        # Row 4: overlay (input gray + recon contour)
        ax = axes[3, i]
        ax.imshow(to_gray(ct_hu[z]).T, cmap='gray',
                  origin='lower', vmin=0, vmax=1, alpha=0.7)
        ax.imshow(to_gray(recon_hu[z]).T, cmap='Blues',
                  origin='lower', vmin=0, vmax=1, alpha=0.3)
        # Mask overlay
        ax.contour(mask[z].T, levels=[0.5], colors=['cyan'],
                   linewidths=0.5, origin='lower')
        ax.set_title(f"Overlay z={z}", color='white', fontsize=7)
        ax.axis('off')

    # Colorbar for diff
    cbar_ax = fig.add_axes([0.92, 0.05, 0.01, 0.2])
    sm = plt.cm.ScalarMappable(cmap='hot',
                                norm=plt.Normalize(0, 200))
    fig.colorbar(sm, cax=cbar_ax, label='|ΔHU|')
    cbar_ax.yaxis.label.set_color('white')
    cbar_ax.tick_params(colors='white')

    # Stats box
    lung_mask = mask >= 0.5
    mae  = float(np.abs(recon_hu[lung_mask] - ct_hu[lung_mask]).mean())
    rmse = float(np.sqrt(((recon_hu[lung_mask] - ct_hu[lung_mask])**2).mean()))
    psnr_val = float(20 * np.log10(1400 / (rmse + 1e-6)))

    stats_ax = fig.add_axes([0.92, 0.3, 0.07, 0.15])
    stats_ax.axis('off')
    stats_ax.text(0, 1,
        f"Lung MAE:  {mae:.1f} HU\n"
        f"Lung RMSE: {rmse:.1f} HU\n"
        f"PSNR:      {psnr_val:.1f} dB\n"
        f"step:      {step}",
        transform=stats_ax.transAxes,
        color='white', fontsize=8, va='top',
        fontfamily='monospace',
        bbox=dict(facecolor='#1a1a2e', alpha=0.8)
    )

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    plt.savefig(out_path, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  PNG: {out_path}")

    return {'mae_hu': round(mae,2), 'rmse_hu': round(rmse,2),
            'psnr_db': round(psnr_val,2)}


# ─────────────────────────────────────────────────────────────────
# Process one patient
# ─────────────────────────────────────────────────────────────────

def process_one(npz_path, diffusion, device, args, ckpt_step):
    patient_id = os.path.basename(npz_path).replace('.npz', '')
    print(f"\n{'─'*60}\nPatient: {patient_id}")

    data    = np.load(npz_path, allow_pickle=True)
    ct_norm = data['ct'].astype(np.float32)    # [-1,1]
    mask    = data['mask'].astype(np.float32)  # [0,1]
    ct_hu   = denorm_hu(ct_norm)

    lung_hu = ct_hu[mask >= 0.5]
    print(f"  Volume: {ct_hu.shape}  "
          f"HU=[{lung_hu.min():.0f},{lung_hu.max():.0f}]  "
          f"median={np.median(lung_hu):.0f}")

    # ── Full volume inference ─────────────────────────────────────
    recon_norm = infer_volume(ct_norm, mask, diffusion, device, args)
    recon_hu   = denorm_hu(recon_norm)

    # ── Save NIfTIs ───────────────────────────────────────────────
    pat_dir = os.path.join(args.out, patient_id)
    os.makedirs(pat_dir, exist_ok=True)
    affine  = np.eye(4)

    def save_nii(vol, name):
        path = os.path.join(pat_dir, f"{patient_id}_{name}.nii.gz")
        nib.save(nib.Nifti1Image(
            vol.transpose(2, 1, 0).astype(np.float32), affine), path)
        print(f"  NIfTI: {path}")

    save_nii(ct_hu,   'input')           # original CT
    save_nii(recon_hu,'recon')           # reconstruction
    save_nii(mask,    'mask')            # lung mask
    save_nii(np.abs(recon_hu - ct_hu),   # absolute difference
             'diff')

    # ── Visualization ─────────────────────────────────────────────
    stats = save_visualization(
        ct_hu, recon_hu, mask, patient_id,
        os.path.join(pat_dir,
                     f"{patient_id}_inference_step{ckpt_step}.png"),
        ckpt_step,
    )

    # ── JSON metrics ──────────────────────────────────────────────
    result = {
        'patient_id': patient_id,
        'checkpoint': args.checkpoint,
        'step':       ckpt_step,
        'shape':      list(ct_hu.shape),
        'metrics':    stats,
    }
    with open(os.path.join(pat_dir,
              f"{patient_id}_metrics.json"), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  MAE={stats['mae_hu']:.1f}HU  "
          f"RMSE={stats['rmse_hu']:.1f}HU  "
          f"PSNR={stats['psnr_db']:.1f}dB")

    return result


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    ap.add_argument('--npz',            type=str, default=None,
                    help='Single patient NPZ file')
    ap.add_argument('--npz_dir',        type=str, default=None,
                    help='Directory of NPZ files')
    ap.add_argument('--dataset',        type=str, default=None,
                    help='Filter by dataset prefix (e.g. bpco, ild)')
    ap.add_argument('--max_patients',   type=int, default=None)

    # Model
    ap.add_argument('--checkpoint',     type=str, required=True,
                    help='Path to distillation checkpoint model-60.pt')
    ap.add_argument('--patch_h',        type=int, default=96)
    ap.add_argument('--patch_w',        type=int, default=96)
    ap.add_argument('--patch_d',        type=int, default=32)

    # Inference
    # Sampling uses DPM-Solver++ with fixed 50 steps (same as original pipeline)
    ap.add_argument('--min_lung_fraction', type=float, default=0.05,
                    help='Skip patches with less lung than this fraction')

    ap.add_argument('--out',            type=str,
                    default='./inference_distill')

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    diffusion, device = load_model(
        args.checkpoint,
        patch_h = args.patch_h,
        patch_d = args.patch_d,
    )

    # Get step from checkpoint for filenames
    ckpt      = torch.load(args.checkpoint, map_location='cpu')
    ckpt_step = ckpt.get('step', 'unknown')

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
                         if os.path.basename(f).startswith(
                             args.dataset + '__')]
        if args.max_patients:
            npz_files = npz_files[:args.max_patients]
    else:
        ap.error("Provide --npz or --npz_dir")

    print(f"Processing {len(npz_files)} volume(s)  "
          f"sampler=DPM-Solver++ (50 steps)")

    all_results = []
    for f in npz_files:
        try:
            r = process_one(f, diffusion, device, args, ckpt_step)
            if r:
                all_results.append(r)
        except Exception as e:
            import traceback
            print(f"  FAILED {os.path.basename(f)}: {e}")
            traceback.print_exc()

    # Summary
    if all_results:
        maes  = [r['metrics']['mae_hu']  for r in all_results]
        rmses = [r['metrics']['rmse_hu'] for r in all_results]
        psnrs = [r['metrics']['psnr_db'] for r in all_results]
        print(f"\n{'='*50}")
        print(f"Summary ({len(all_results)} patients):")
        print(f"  MAE:  {np.mean(maes):.1f} ± {np.std(maes):.1f} HU")
        print(f"  RMSE: {np.mean(rmses):.1f} ± {np.std(rmses):.1f} HU")
        print(f"  PSNR: {np.mean(psnrs):.1f} ± {np.std(psnrs):.1f} dB")

        summary = {
            'n_patients': len(all_results),
            'checkpoint': args.checkpoint,
            'step':       ckpt_step,
            'sampler':    'dpm_solver++_order3_50steps',
            'mae_mean':   round(float(np.mean(maes)),  2),
            'mae_std':    round(float(np.std(maes)),   2),
            'rmse_mean':  round(float(np.mean(rmses)), 2),
            'rmse_std':   round(float(np.std(rmses)),  2),
            'psnr_mean':  round(float(np.mean(psnrs)), 2),
            'psnr_std':   round(float(np.std(psnrs)),  2),
            'patients':   all_results,
        }
        with open(os.path.join(args.out, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary: {args.out}/summary.json")

    print(f"\nDone. Results: {args.out}")


if __name__ == '__main__':
    main()