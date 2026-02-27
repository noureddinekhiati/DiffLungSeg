# -*- coding: utf-8 -*-
"""
Dataset for DiffSegLung — loads from unified NPZ folder.

Each NPZ file contains:
    ct   : float16 (Z, Y, X), already normalized to [-1, 1]
    mask : uint8   (Z, Y, X), binary lung mask
    meta : 0-d string array (JSON with dataset name, spacing, etc.)

__getitem__ returns:
    'ct'   : float32 tensor (1, D, H, W)  — patch from CT
    'mask' : float32 tensor (1, D, H, W)  — same patch from lung mask

Patch sampling:
    - Random 3D patch (patch_h x patch_w x patch_d)
    - Rejection sampling: >= min_lung_ratio voxels inside lung
    - Fallback to lung centroid if rejection fails
    - Tuberculosis oversampled by oversample_tb factor
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter


class LungPatchDataset(Dataset):

    def __init__(
        self,
        npz_dir:        str,
        patch_size:     tuple = (96, 96, 32),  # (H, W, D)
        min_lung_ratio: float = 0.5,
        max_tries:      int   = 20,
        oversample_tb:  int   = 4,
        augment:        bool  = True,
        dataset_filter: list  = None,
    ):
        self.npz_dir        = npz_dir
        self.patch_h        = patch_size[0]
        self.patch_w        = patch_size[1]
        self.patch_d        = patch_size[2]
        self.min_lung_ratio = min_lung_ratio
        self.max_tries      = max_tries
        self.augment        = augment

        # Discover NPZ files
        all_files = sorted([
            f for f in os.listdir(npz_dir)
            if f.endswith(".npz") and "__" in f
        ])

        if dataset_filter is not None:
            all_files = [
                f for f in all_files
                if any(f.startswith(ds + "__") for ds in dataset_filter)
            ]

        # Oversample tuberculosis
        self.file_list = []
        for f in all_files:
            dataset_name = f.split("__")[0]
            n = oversample_tb if dataset_name == "tuberculosis" else 1
            self.file_list.extend([f] * n)

        random.shuffle(self.file_list)

        counts = Counter(f.split("__")[0] for f in self.file_list)
        print(f"[LungPatchDataset] {len(all_files)} unique files -> "
              f"{len(self.file_list)} entries after oversampling")
        for ds, n in sorted(counts.items()):
            print(f"  {ds:<20}: {n}")

    def _sample_patch(self, ct, mask):
        Z, Y, X = ct.shape
        pd = min(self.patch_d, Z)
        ph = min(self.patch_h, Y)
        pw = min(self.patch_w, X)

        z_range = max(1, Z - pd)
        y_range = max(1, Y - ph)
        x_range = max(1, X - pw)

        best_ct, best_mask, best_ratio = None, None, 0.0

        for _ in range(self.max_tries):
            z0 = random.randint(0, z_range)
            y0 = random.randint(0, y_range)
            x0 = random.randint(0, x_range)
            pm = mask[z0:z0+pd, y0:y0+ph, x0:x0+pw]
            ratio = float(pm.mean())
            if ratio >= self.min_lung_ratio:
                return ct[z0:z0+pd, y0:y0+ph, x0:x0+pw], pm
            if ratio > best_ratio:
                best_ratio = ratio
                best_ct   = ct[z0:z0+pd, y0:y0+ph, x0:x0+pw]
                best_mask = pm

        # Fallback: center on lung centroid
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            cz, cy, cx = coords.mean(axis=0).astype(int)
            z0 = int(np.clip(cz - pd // 2, 0, z_range))
            y0 = int(np.clip(cy - ph // 2, 0, y_range))
            x0 = int(np.clip(cx - pw // 2, 0, x_range))
            return (ct[z0:z0+pd, y0:y0+ph, x0:x0+pw],
                    mask[z0:z0+pd, y0:y0+ph, x0:x0+pw])

        return best_ct, best_mask

    def _augment(self, ct, mask):
        if random.random() > 0.5:
            ct = ct[:, ::-1, :].copy(); mask = mask[:, ::-1, :].copy()
        if random.random() > 0.5:
            ct = ct[:, :, ::-1].copy(); mask = mask[:, :, ::-1].copy()
        if random.random() > 0.5:
            ct = ct[::-1, :, :].copy(); mask = mask[::-1, :, :].copy()
        return ct, mask

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        fpath = os.path.join(self.npz_dir, self.file_list[index])
        data  = np.load(fpath, mmap_mode='r', allow_pickle=True)
        ct    = data['ct'].astype(np.float32)
        mask  = data['mask'].astype(np.float32)

        ct_patch, mask_patch = self._sample_patch(ct, mask)
        if self.augment:
            ct_patch, mask_patch = self._augment(ct_patch, mask_patch)

        ct_t   = torch.from_numpy(ct_patch[np.newaxis]).float()
        mask_t = torch.from_numpy(mask_patch[np.newaxis]).float()
        return {'ct': ct_t, 'mask': mask_t}

    def sample_conditions(self, batch_size: int):
        """Sample random masks for inference — called by trainer."""
        indices = random.sample(range(len(self)), min(batch_size, len(self)))
        masks = [self[i]['mask'].unsqueeze(0) for i in indices]
        return torch.cat(masks, dim=0).cuda()