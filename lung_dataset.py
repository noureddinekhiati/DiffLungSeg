# lung_dataset.py
# ───────────────
# Clean simple dataset using mmap_mode='r'.
# No lazy loading complexity needed — mmap handles everything.
# OS loads only the patch bytes touched, not the full 150MB file.

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class LungNPZDataset(Dataset):

    def __init__(self,
                 npz_dir,
                 patch_h=96,
                 patch_w=96,
                 patch_d=32,
                 min_lung=0.4,
                 dataset_filter=None,
                 max_patients=None):

        self.patch_h  = patch_h
        self.patch_w  = patch_w
        self.patch_d  = patch_d
        self.min_lung = min_lung

        all_files = sorted([
            os.path.join(npz_dir, f)
            for f in os.listdir(npz_dir)
            if f.endswith('.npz') and '__' in f
        ])
        if dataset_filter:
            all_files = [f for f in all_files
                         if os.path.basename(f).startswith(
                             dataset_filter + '__')]
        if max_patients:
            all_files = all_files[:max_patients]

        self.files = all_files
        self._cache = {}  # stores mmap objects, not data — negligible RAM

        print(f"Dataset: {len(self.files)} NPZ files (mmap)")

    def _load(self, path):
        """Memory-map the NPZ file. Only touched pages loaded by OS."""
        if path not in self._cache:
            data = np.load(path, allow_pickle=True, mmap_mode='r')
            self._cache[path] = (data['ct'], data['mask'])
        return self._cache[path]

    def _sample_patch(self, ct, mask):
        Z, Y, X = ct.shape
        ph = min(self.patch_h, Y)
        pw = min(self.patch_w, X)
        pd = min(self.patch_d, Z)

        for _ in range(50):
            z0 = np.random.randint(0, max(1, Z-pd))
            y0 = np.random.randint(0, max(1, Y-ph))
            x0 = np.random.randint(0, max(1, X-pw))
            mp = mask[z0:z0+pd, y0:y0+ph, x0:x0+pw]
            if mp.mean() >= self.min_lung:
                cp = ct[z0:z0+pd, y0:y0+ph, x0:x0+pw]
                return np.array(cp, dtype=np.float32), \
                       np.array(mp, dtype=np.float32)

        # fallback: lung center
        lung_z = np.where(np.asarray(mask).any(axis=(1, 2)))[0]
        cz = int(np.median(lung_z)) if len(lung_z) > 0 else Z // 2
        z0 = max(0, min(Z-pd, cz-pd//2))
        y0 = (Y-ph) // 2
        x0 = (X-pw) // 2
        return (np.array(ct[z0:z0+pd, y0:y0+ph, x0:x0+pw], dtype=np.float32),
                np.array(mask[z0:z0+pd, y0:y0+ph, x0:x0+pw], dtype=np.float32))

    def sample_conditions(self, batch_size):
        """For NIfTI sampling during training."""
        masks = []
        for _ in range(batch_size):
            path = self.files[np.random.randint(len(self.files))]
            ct, mask = self._load(path)
            _, mp = self._sample_patch(ct, mask)
            masks.append(torch.tensor(mp[None], dtype=torch.float32))
        return torch.stack(masks).cuda()

    def __len__(self):
        return len(self.files) * 300

    def __getitem__(self, idx):
        path = self.files[idx % len(self.files)]
        ct, mask = self._load(path)
        cp, mp = self._sample_patch(ct, mask)
        return {
            'ct':   torch.tensor(cp[None],  dtype=torch.float32),
            'mask': torch.tensor(mp[None],  dtype=torch.float32),
        }


def worker_init_fn(worker_id):
    np.random.seed(worker_id)