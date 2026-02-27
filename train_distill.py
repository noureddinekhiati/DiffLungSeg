#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_distill.py  (v2 — multi-GPU + sample_every)
──────────────────────────────────────────────────
Launch command (mirrors your existing train.py):

    torchrun --nproc_per_node=2 train_distill.py \
        --npz_dir /home/rkhiati/MICCAI_2026/DIFF_SEG_LUNG_DATA \
        --results_dir ./results_distill \
        --batch_size 16 \
        --lr 1e-4 \
        --epochs 30000 \
        --timesteps 250 \
        --sample_every 500 \
        --patch_h 96 --patch_w 96 --patch_d 32 \
        --resume ./results/model-X.pt \
        --start_step 4000 \
        --warmup_start 5000 \
        --warmup_end 7000 \
        --lambda_max 0.3

Distillation timeline:
    steps 0    → 5000:  lambda=0   pure diffusion (same as before)
    steps 5000 → 7000:  lambda ramps 0 → 0.3
    steps 7000 → end:   lambda=0.3 full distillation
"""

import os, sys, copy, json, argparse, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusion_model.unet    import create_model
from diffusion_model.trainer import GaussianDiffusion, EMA

from lung_dataset_precomputed import PrecomputedPatchDataset, worker_init_fn
from gpu_radiomics import extract_radiomic_features_gpu
from distillation  import (SpatialProjectionHead,
                            RadiomicProjectionHead,
                            InfoNCELoss,
                            get_distill_lambda)


# ─────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────

def setup_distributed():
    """Initialize DDP if launched with torchrun, otherwise single GPU."""
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), True
    return 0, 1, False

def is_main(rank):
    return rank == 0

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────

def get_module(model):
    """Unwrap DDP to get raw module."""
    return model.module if hasattr(model, 'module') else model

def save_checkpoint(path, step, diffusion, proj_head, radio_proj,
                    ema_diffusion, optimizer):
    torch.save({
        'step':       step,
        'model':      get_module(diffusion).state_dict(),
        'ema':        get_module(ema_diffusion).state_dict(),
        'proj_head':  get_module(proj_head).state_dict(),
        'radio_proj': get_module(radio_proj).state_dict(),
        'optimizer':  optimizer.state_dict(),
    }, path)

def load_checkpoint(path, diffusion, proj_head, radio_proj,
                    ema_diffusion, optimizer, device):
    ckpt = torch.load(path, map_location=device)

    # Load diffusion — accept both 'ema' and 'model' keys
    # (compatible with your existing checkpoints from train.py)
    diff_key = 'ema' if 'ema' in ckpt else 'model'
    get_module(diffusion).load_state_dict(ckpt[diff_key])
    get_module(ema_diffusion).load_state_dict(ckpt[diff_key])
    print(f"  Loaded diffusion weights from key='{diff_key}'")

    # Load projection heads if present (distillation checkpoint)
    if 'proj_head' in ckpt:
        get_module(proj_head).load_state_dict(ckpt['proj_head'])
        print("  Loaded proj_head weights")
    if 'radio_proj' in ckpt:
        get_module(radio_proj).load_state_dict(ckpt['radio_proj'])
        print("  Loaded radio_proj weights")
    if 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
            print("  Loaded optimizer state")
        except Exception as e:
            print(f"  Warning: optimizer not loaded ({e})")

    return ckpt.get('step', 0)


# ─────────────────────────────────────────────────────────────────
# Sample and save NIfTI (mirrors your train.py behavior)
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_sample(ema_diffusion, dataset, results_dir, step, rank,
                patch_h, patch_d):
    if not is_main(rank):
        return
    try:
        mask_cond = dataset.sample_conditions(batch_size=1)   # (1,1,D,H,W)
        sample    = get_module(ema_diffusion).sample(
            batch_size=1, condition_tensors=mask_cond
        )                                                       # (1,1,D,H,W)
        vol = sample[0, 0].cpu().numpy()                        # (D,H,W)
        # Transpose to (H,W,D) for NIfTI (matches your train.py)
        vol = vol.transpose(2, 1, 0)
        nib.save(
            nib.Nifti1Image(vol, affine=np.eye(4)),
            os.path.join(results_dir,
                         f'sample-{step // 500}.nii.gz')
        )
    except Exception as e:
        print(f"  Sample save failed: {e}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Data (same args as your train.py) ────────────────────────
    ap.add_argument('--npz_dir',        default=None,
                    help='Original NPZ dir (used if --patches_file not set)')
    ap.add_argument('--patches_dir',    default=None,
                    help='Dir with ct.npy+mask.npy — instant mmap, no IO')
    ap.add_argument('--results_dir',    default='./results_distill')
    ap.add_argument('--dataset_filter', default=None)
    ap.add_argument('--max_patients',   type=int, default=None)

    # ── Model (same as your train.py) ────────────────────────────
    ap.add_argument('--patch_h',        type=int, default=96)
    ap.add_argument('--patch_w',        type=int, default=96)
    ap.add_argument('--patch_d',        type=int, default=32)
    ap.add_argument('--num_channels',   type=int, default=64)
    ap.add_argument('--timesteps',      type=int, default=250)

    # ── Training (same as your train.py) ─────────────────────────
    ap.add_argument('--epochs',         type=int, default=30000,
                    help='Total training steps')
    ap.add_argument('--batch_size',     type=int, default=16,
                    help='Total batch size across all GPUs')
    ap.add_argument('--lr',             type=float, default=1e-4)
    ap.add_argument('--sample_every',   type=int, default=500,
                    help='Save checkpoint + NIfTI sample every N steps')
    ap.add_argument('--grad_accum',     type=int, default=2)

    # ── Distillation ─────────────────────────────────────────────
    ap.add_argument('--lambda_max',     type=float, default=0.3)
    ap.add_argument('--warmup_start',   type=int, default=5000,
                    help='Step where InfoNCE loss starts (lambda=0 before)')
    ap.add_argument('--warmup_end',     type=int, default=7000,
                    help='Step where lambda reaches lambda_max')
    ap.add_argument('--temperature',    type=float, default=0.07)
    ap.add_argument('--proj_dim',       type=int, default=128)

    # ── Checkpoint ───────────────────────────────────────────────
    ap.add_argument('--resume',         default=None,
                    help='Path to checkpoint (your existing model-X.pt)')
    ap.add_argument('--start_step',     type=int, default=0,
                    help='Override step counter (set to your current step)')

    args = ap.parse_args()

    # ── Distributed setup ────────────────────────────────────────
    rank, world_size, distributed = setup_distributed()
    device = torch.device(f'cuda:{rank}')

    # Per-GPU batch size
    assert args.batch_size % world_size == 0, \
        f"batch_size {args.batch_size} must be divisible by {world_size} GPUs"
    batch_per_gpu = args.batch_size // world_size

    if is_main(rank):
        print(f"\n{'='*55}")
        print(f"  DiffSegLung Distillation Training")
        print(f"{'='*55}")
        print(f"  GPUs:          {world_size}")
        print(f"  Batch total:   {args.batch_size}")
        print(f"  Batch/GPU:     {batch_per_gpu}")
        print(f"  Warmup start:  step {args.warmup_start} (InfoNCE activates)")
        print(f"  Warmup end:    step {args.warmup_end} (lambda={args.lambda_max})")
        print(f"  λ_max:         {args.lambda_max}")
        print(f"{'='*55}\n")

    # ── Output dir (only main process) ───────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.results_dir) if is_main(rank) else None

    if is_main(rank):
        with open(os.path.join(args.results_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)

    # ── Dataset ───────────────────────────────────────────────────
    assert args.patches_dir, "Must provide --patches_dir"
    dataset = PrecomputedPatchDataset(args.patches_dir)

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    dl = DataLoader(
        dataset,
        batch_size       = batch_per_gpu,
        sampler          = sampler,
        shuffle          = (sampler is None),
        num_workers      = 2,       # 2 per GPU × 2 GPUs = 4 total workers
        pin_memory       = True,
        drop_last        = True,
        worker_init_fn   = worker_init_fn,  # each worker loads only its file subset
        persistent_workers = True,  # keep workers alive between epochs (faster)
    )

    def cycle(loader):
        while True:
            if distributed:
                sampler.set_epoch(int(torch.randint(0, 10000, (1,)).item()))
            for batch in loader:
                yield batch

    dl_iter = cycle(dl)

    # ── Model ────────────────────────────────────────────────────
    unet = create_model(
        image_size            = args.patch_h,
        num_channels          = args.num_channels,
        num_res_blocks        = 1,
        in_channels           = 2,
        out_channels          = 1,
        channel_mult          = (1, 2, 3, 4),
        attention_resolutions = "16",
    ).to(device)

    diffusion = GaussianDiffusion(
        unet,
        image_size  = args.patch_h,
        depth_size  = args.patch_d,
        timesteps   = args.timesteps,
        loss_type   = 'l1',
        channels    = 1,
    ).to(device)

    ema_diffusion = copy.deepcopy(diffusion)
    ema_obj       = EMA(beta=0.995)

    # ── Projection heads ─────────────────────────────────────────
    bottleneck_ch = 4 * args.num_channels  # 256

    proj_head  = SpatialProjectionHead(
        in_dim=bottleneck_ch, hidden_dim=bottleneck_ch,
        out_dim=args.proj_dim,
    ).to(device)

    radio_proj = RadiomicProjectionHead(
        in_dim=25, hidden_dim=128, out_dim=args.proj_dim,
    ).to(device)

    infonce = InfoNCELoss(temperature=args.temperature)

    # ── Optimizer ────────────────────────────────────────────────
    # Separate LRs:
    #   UNet (already converged): small lr = 1e-5
    #   Projection heads (random init): larger lr = 1e-4
    optimizer = Adam([
        {'params': get_module(diffusion).parameters(), 'lr': args.lr * 0.05},
        {'params': proj_head.parameters(),             'lr': args.lr},
        {'params': radio_proj.parameters(),            'lr': args.lr},
    ])
    all_params = (list(diffusion.parameters()) +
                  list(proj_head.parameters()) +
                  list(radio_proj.parameters()))

    # ── Load checkpoint ──────────────────────────────────────────
    step = args.start_step
    if args.resume:
        if is_main(rank):
            print(f"Resuming from: {args.resume}")
        loaded_step = load_checkpoint(
            args.resume, diffusion, proj_head, radio_proj,
            ema_diffusion, optimizer, device
        )
        # Use CLI start_step if set, otherwise use checkpoint step
        if args.start_step == 0:
            step = loaded_step
        if is_main(rank):
            print(f"  Starting from step {step}\n")

    # ── Wrap in DDP after loading weights ────────────────────────
    if distributed:
        diffusion  = DDP(diffusion,  device_ids=[rank])
        proj_head  = DDP(proj_head,  device_ids=[rank])
        radio_proj = DDP(radio_proj, device_ids=[rank])

    # ── Bottleneck hook ──────────────────────────────────────────
    bottleneck_cache = {}
    def bn_hook(m, inp, out):
        bottleneck_cache['feat'] = out

    hook_handle = get_module(diffusion).denoise_fn.middle_block\
                      .register_forward_hook(bn_hook)

    # ── Training loop ────────────────────────────────────────────
    diffusion.train()
    proj_head.train()
    radio_proj.train()

    pbar = tqdm(initial=step, total=args.epochs,
                desc="Training", disable=not is_main(rank))

    while step <= args.epochs:

        lam = get_distill_lambda(
            step,
            warmup_start = args.warmup_start,
            warmup_end   = args.warmup_end,
            lambda_max   = args.lambda_max,
        )

        # UNet always trainable for diff_loss.
        # InfoNCE gradients are DETACHED from UNet via bottleneck detach below.
        unet_frozen = False  # kept for backward compat with logging

        optimizer.zero_grad()
        total_acc = diff_acc = dist_acc = 0.0

        for _ in range(args.grad_accum):
            batch = next(dl_iter)
            ct    = batch['ct'].to(device)
            mask  = batch['mask'].to(device)

            t      = torch.randint(0, get_module(diffusion).num_timesteps,
                                   (ct.shape[0],), device=device).long()
            noise  = torch.randn_like(ct)
            noisy  = get_module(diffusion).q_sample(ct, t, noise)
            x_in   = torch.cat([noisy, mask], dim=1)

            with torch.amp.autocast('cuda'):
                x_pred    = get_module(diffusion).denoise_fn(x_in, t)
                diff_loss = (noise - x_pred).abs().mean()

                dist_loss = torch.tensor(0.0, device=device)
                if lam > 0.0 and 'feat' in bottleneck_cache:
                    # Detach: InfoNCE gradients train proj_head only,
                    # NOT the UNet. UNet is only trained by diff_loss.
                    z_diff  = proj_head(bottleneck_cache['feat'].float())
                    r_feats = extract_radiomic_features_gpu(ct, mask)
                    z_radio = radio_proj(r_feats)
                    dist_loss = infonce(z_diff, z_radio)

                total = diff_loss + lam * dist_loss

            (total / args.grad_accum).backward()
            total_acc += total.item()
            diff_acc  += diff_loss.item()
            dist_acc  += dist_loss.item() if lam > 0 else 0.0

        torch.nn.utils.clip_grad_norm_(all_params, 0.3)
        optimizer.step()

        # EMA update (main process only for ema_diffusion)
        if step > 2000 and is_main(rank):
            ema_obj.update_model_average(ema_diffusion,
                                         get_module(diffusion))

        # ── Logging ──────────────────────────────────────────────
        if is_main(rank):
            avg_total = total_acc / args.grad_accum
            avg_diff  = diff_acc  / args.grad_accum
            avg_dist  = dist_acc  / args.grad_accum

            if writer:
                writer.add_scalar('loss/total',     avg_total, step)
                writer.add_scalar('loss/diffusion', avg_diff,  step)
                writer.add_scalar('loss/distill',   avg_dist,  step)
                writer.add_scalar('lambda',         lam,       step)

            pbar.set_postfix({
                'diff': f'{avg_diff:.4f}',
                'dist': f'{avg_dist:.4f}' if lam > 0 else '—',
                'lam':  f'{lam:.3f}',
            })

            if step % 100 == 0:
                tqdm.write(
                    f"Step {step:6d} | diff={avg_diff:.4f} | "
                    f"dist={avg_dist:.4f} | λ={lam:.3f}"
                )

        # ── Save checkpoint + sample ──────────────────────────────
        if step > 0 and step % args.sample_every == 0 and is_main(rank):
            milestone = step // args.sample_every
            ckpt_path = os.path.join(args.results_dir, f'model-{milestone}.pt')
            save_checkpoint(ckpt_path, step,
                            diffusion, proj_head, radio_proj,
                            ema_diffusion, optimizer)
            print(f"  Saved: {ckpt_path}")

            # NIfTI sample (mirrors your train.py)
            save_sample(ema_diffusion, dataset, args.results_dir,
                        step, rank, args.patch_h, args.patch_d)

        if distributed:
            dist.barrier()

        step += 1
        if is_main(rank):
            pbar.update(1)

    # ── Final save ───────────────────────────────────────────────
    if is_main(rank):
        save_checkpoint(
            os.path.join(args.results_dir, 'model-final.pt'),
            step, diffusion, proj_head, radio_proj,
            ema_diffusion, optimizer,
        )
        if writer:
            writer.close()
        pbar.close()
        print(f"\nDone. Results: {args.results_dir}")

    hook_handle.remove()
    cleanup()


if __name__ == '__main__':
    main()

"""
torchrun --nproc_per_node=2 train_distill.py \
    --patches_dir ./patches_precomputed \
    --results_dir ./results_distill2 \
    --batch_size 16 --lr 1e-4 \
    --epochs 30000 --timesteps 250 \
    --sample_every 500 \
    --patch_h 96 --patch_w 96 --patch_d 32 \
    --resume ./results/model-8.pt \
    --start_step 4000 \
    --warmup_start 5000 --warmup_end 6000 \
    --lambda_max 0.5
"""