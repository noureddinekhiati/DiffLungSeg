# -*- coding: utf-8 -*-
"""
train.py — DiffSegLung training entry point.

Changes vs original:
  - Loads from NPZ dataset (LungPatchDataset) instead of LIDC-IDRI
  - 2-GPU DDP via torchrun
  - fp16 via torch.cuda.amp (replaces broken apex)
  - Lung mask as condition channel (1 channel in, 1 channel out)
  - Patch size 96x96x32
  - DPM-Solver sampling during periodic qualitative check

Launch:
    # Single GPU (debug)
    python train.py --npz_dir /path/to/NPZ_DATASET

    # 2 GPUs (full training)
    torchrun --nproc_per_node=2 train.py --npz_dir /path/to/NPZ_DATASET
"""

import argparse
import os
import torch
import torch.distributed as dist

from diffusion_model.trainer import GaussianDiffusion, Trainer
from diffusion_model.unet import create_model
from dataset import LungPatchDataset


def parse_args():
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument('--npz_dir',     type=str, required=True,
                   help='Path to unified NPZ dataset folder')
    p.add_argument('--results_dir', type=str, default='./results',
                   help='Where to save checkpoints and samples')

    # Training
    p.add_argument('--batch_size',  type=int,   default=8,
                   help='Per-GPU batch size (total = batch_size * n_gpus)')
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=100000)
    p.add_argument('--timesteps',   type=int,   default=250)
    p.add_argument('--sample_every',type=int,   default=500,
                   help='Save checkpoint + qualitative sample every N steps')

    # Resume
    p.add_argument('--resume',      type=str,   default='',
                   help='Path to checkpoint .pt file to resume from')
    p.add_argument('--start_steps', type=int,   default=0)

    # Model
    p.add_argument('--num_channels',    type=int, default=64)
    p.add_argument('--num_res_blocks',  type=int, default=1)

    # Patch
    p.add_argument('--patch_h', type=int, default=96)
    p.add_argument('--patch_w', type=int, default=96)
    p.add_argument('--patch_d', type=int, default=32)

    return p.parse_args()


def setup_ddp():
    """Initialize DDP if launched with torchrun, else single GPU."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size()
    else:
        return 0, 1


def is_main(local_rank):
    return local_rank == 0


def main():
    args       = parse_args()
    local_rank, world_size = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")

    if is_main(local_rank):
        print(f"Training on {world_size} GPU(s)")
        print(f"NPZ dir : {args.npz_dir}")
        print(f"Batch   : {args.batch_size} per GPU x {world_size} = "
              f"{args.batch_size * world_size} total")

    # ── Dataset ───────────────────────────────────────────────────
    dataset = LungPatchDataset(
        npz_dir    = args.npz_dir,
        patch_size = (args.patch_h, args.patch_w, args.patch_d),
        augment    = True,
        oversample_tb = 4,
    )

    # ── Model ─────────────────────────────────────────────────────
    # in_channels = 2: noisy CT (1) + lung mask condition (1)
    # out_channels = 1: predicted noise
    model = create_model(
        image_size          = args.patch_h,
        num_channels        = args.num_channels,
        num_res_blocks      = args.num_res_blocks,
        in_channels         = 2,        # CT + mask
        out_channels        = 1,
        channel_mult        = (1, 2, 3, 4),
        attention_resolutions = "16",
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size  = args.patch_h,
        depth_size  = args.patch_d,
        timesteps   = args.timesteps,
        loss_type   = 'l1',
        channels    = 1,
    ).to(device)

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        diffusion.load_state_dict(ckpt['ema'])
        if is_main(local_rank):
            print(f"Resumed from {args.resume}")

    # ── Trainer ───────────────────────────────────────────────────
    trainer = Trainer(
        diffusion_model       = diffusion,
        dataset               = dataset,
        local_rank            = local_rank,
        world_size            = world_size,
        results_folder        = args.results_dir,
        start_steps           = args.start_steps,
        train_batch_size      = args.batch_size,
        train_lr              = args.lr,
        train_num_steps       = args.epochs,
        gradient_accumulate_every = 2,
        ema_decay             = 0.995,
        save_and_sample_every = args.sample_every,
        depth_size            = args.patch_d,
        fp16                  = True,
    )

    trainer.train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

'''
# test command 



'''
