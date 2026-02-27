#!/usr/bin/env python3
"""
precompute_patches.py  (npy version — fast save)
─────────────────────────────────────────────────
Saves as separate .npy files — no compression, pure disk write.
Save time: ~30s vs ~20min for npz compressed.

Output:
    patches_precomputed/ct.npy          28.9 GB  float16
    patches_precomputed/mask.npy        14.5 GB  uint8
    patches_precomputed/patient_ids.npy  0.1 GB  int16
    Total: ~43 GB

Usage:
    python precompute_patches.py \
        --npz_dir /home/rkhiati/MICCAI_2026/DIFF_SEG_LUNG_DATA \
        --out_dir ./patches_precomputed \
        --patches_per_patient 150 \
        --num_workers 32
"""

import os, argparse, time
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def sample_patch(ct, mask, ph, pw, pd, min_lung=0.4, max_tries=50):
    Z, Y, X = ct.shape
    ph = min(ph, Y); pw = min(pw, X); pd = min(pd, Z)
    for _ in range(max_tries):
        z0 = np.random.randint(0, max(1, Z-pd))
        y0 = np.random.randint(0, max(1, Y-ph))
        x0 = np.random.randint(0, max(1, X-pw))
        mp = mask[z0:z0+pd, y0:y0+ph, x0:x0+pw]
        if mp.mean() >= min_lung:
            return ct[z0:z0+pd, y0:y0+ph, x0:x0+pw], mp
    lung_z = np.where(mask.any(axis=(1,2)))[0]
    cz = int(np.median(lung_z)) if len(lung_z) > 0 else Z//2
    z0 = max(0, min(Z-pd, cz-pd//2))
    y0 = (Y-ph)//2; x0 = (X-pw)//2
    return ct[z0:z0+pd, y0:y0+ph, x0:x0+pw], mask[z0:z0+pd, y0:y0+ph, x0:x0+pw]


def process_patient(args):
    path, pid, n_patches, ph, pw, pd, min_lung, seed = args
    np.random.seed(seed)
    try:
        data = np.load(path, allow_pickle=True)
        ct   = data['ct'].astype(np.float32)
        mask = data['mask'].astype(np.float32)
        if mask.mean() < 0.03:
            return None
        ct_out   = np.zeros((n_patches, 1, pd, ph, pw), dtype=np.float16)
        mask_out = np.zeros((n_patches, 1, pd, ph, pw), dtype=np.uint8)
        for i in range(n_patches):
            cp, mp = sample_patch(ct, mask, ph, pw, pd, min_lung)
            ct_out[i, 0]   = cp.astype(np.float16)
            mask_out[i, 0] = (mp > 0.5).astype(np.uint8)
        return (pid, ct_out, mask_out)
    except Exception as e:
        print(f"\n  Worker failed {os.path.basename(path)}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz_dir',             required=True)
    ap.add_argument('--out_dir',             required=True)
    ap.add_argument('--patches_per_patient', type=int,   default=150)
    ap.add_argument('--patch_h',             type=int,   default=96)
    ap.add_argument('--patch_w',             type=int,   default=96)
    ap.add_argument('--patch_d',             type=int,   default=32)
    ap.add_argument('--min_lung',            type=float, default=0.4)
    ap.add_argument('--dataset_filter',      default=None)
    ap.add_argument('--num_workers',         type=int,
                    default=min(32, cpu_count()))
    ap.add_argument('--seed',                type=int,   default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted([
        os.path.join(args.npz_dir, f)
        for f in os.listdir(args.npz_dir)
        if f.endswith('.npz') and '__' in f
    ])
    if args.dataset_filter:
        files = [f for f in files
                 if os.path.basename(f).startswith(args.dataset_filter+'__')]

    n_patients = len(files)
    n_patches  = args.patches_per_patient
    PD, PH, PW = args.patch_d, args.patch_h, args.patch_w

    print(f"{'='*55}")
    print(f"  Patch Precomputation (npy — no compression)")
    print(f"{'='*55}")
    print(f"  Patients:         {n_patients}")
    print(f"  Patches/patient:  {n_patches}")
    print(f"  Total patches:    {n_patients * n_patches:,}")
    print(f"  CPU workers:      {args.num_workers} / {cpu_count()} cores")
    ct_gb   = (n_patients * n_patches * PD * PH * PW * 2) / 1e9
    mask_gb = (n_patients * n_patches * PD * PH * PW * 1) / 1e9
    print(f"  ct.npy:           {ct_gb:.1f} GB")
    print(f"  mask.npy:         {mask_gb:.1f} GB")
    print(f"  Total:            {ct_gb+mask_gb:.1f} GB")
    print(f"{'='*55}\n")

    worker_args = [
        (path, pid, n_patches, PH, PW, PD, args.min_lung, args.seed + pid)
        for pid, path in enumerate(files)
    ]

    # ── Extract ───────────────────────────────────────────────────
    t0 = time.time()
    N_max = n_patients * n_patches

    ct_patches   = np.zeros((N_max, 1, PD, PH, PW), dtype=np.float16)
    mask_patches = np.zeros((N_max, 1, PD, PH, PW), dtype=np.uint8)
    patient_ids  = np.zeros(N_max, dtype=np.int16)

    idx = 0; skipped = 0

    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_patient, worker_args),
            total=n_patients,
            desc=f"Extracting [{args.num_workers} workers]"
        ):
            if result is None:
                skipped += 1
                continue
            pid, ct_out, mask_out = result
            end = idx + n_patches
            ct_patches[idx:end]   = ct_out
            mask_patches[idx:end] = mask_out
            patient_ids[idx:end]  = pid
            idx = end

    extract_time = time.time() - t0
    print(f"\nExtraction: {extract_time/60:.1f} min")

    # ── Shuffle ───────────────────────────────────────────────────
    ct_patches   = ct_patches[:idx]
    mask_patches = mask_patches[:idx]
    patient_ids  = patient_ids[:idx]

    print(f"Shuffling {idx:,} patches...")
    perm = np.random.default_rng(args.seed).permutation(idx)
    ct_patches   = ct_patches[perm]
    mask_patches = mask_patches[perm]
    patient_ids  = patient_ids[perm]

    # ── Save as .npy — fast, no compression ───────────────────────
    ct_path   = os.path.join(args.out_dir, 'ct.npy')
    mask_path = os.path.join(args.out_dir, 'mask.npy')
    ids_path  = os.path.join(args.out_dir, 'patient_ids.npy')

    print(f"Saving ct.npy   ({ct_patches.nbytes/1e9:.1f} GB)...")
    t1 = time.time()
    np.save(ct_path, ct_patches)
    print(f"Saving mask.npy ({mask_patches.nbytes/1e9:.1f} GB)...")
    np.save(mask_path, mask_patches)
    print(f"Saving patient_ids.npy...")
    np.save(ids_path, patient_ids)
    save_time = time.time() - t1

    total_gb = sum(os.path.getsize(p)/1e9
                   for p in [ct_path, mask_path, ids_path])

    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Patches:      {idx:,}")
    print(f"  Skipped:      {skipped} patients")
    print(f"  Total size:   {total_gb:.1f} GB")
    print(f"  Extract time: {extract_time/60:.1f} min")
    print(f"  Save time:    {save_time:.0f}s  ← fast, no compression")
    print(f"{'='*55}")
    print(f"""
Launch training:

    torchrun --nproc_per_node=2 train_distill.py \\
        --patches_dir {args.out_dir} \\
        --results_dir ./results_distill \\
        --resume ./results/model-8.pt \\
        --start_step 4000 \\
        --warmup_start 4000 --warmup_end 6000 \\
        --batch_size 16 --lr 1e-4 \\
        --epochs 30000 --sample_every 500 \\
        --lambda_max 0.5
""")


if __name__ == '__main__':
    main()

'''
python precompute_patches.py \
    --npz_dir /home/rkhiati/MICCAI_2026/DIFF_SEG_LUNG_DATA \
    --out_dir ./patches_precomputed \
    --patches_per_patient 150 \
    --num_workers 24
'''