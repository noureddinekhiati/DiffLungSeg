# lung_dataset_precomputed.py  (npy version)
# ───────────────────────────────────────────
# Loads from ct.npy + mask.npy — instant startup via mmap.
# No decompression. Pure RAM access during training.

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PrecomputedPatchDataset(Dataset):

    def __init__(self, patches_dir):
        ct_path   = os.path.join(patches_dir, 'ct.npy')
        mask_path = os.path.join(patches_dir, 'mask.npy')
        ids_path  = os.path.join(patches_dir, 'patient_ids.npy')

        print(f"Mmapping precomputed patches from {patches_dir}...")

        # mmap_mode='r' → instant, OS loads pages on demand
        self.ct_patches   = np.load(ct_path,   mmap_mode='r')
        self.mask_patches = np.load(mask_path, mmap_mode='r')
        self.patient_ids  = np.load(ids_path,  mmap_mode='r')

        N = len(self.ct_patches)
        total_gb = (self.ct_patches.nbytes + self.mask_patches.nbytes) / 1e9
        print(f"Ready — {N:,} patches, {total_gb:.1f} GB "
              f"(mmap, no RAM used until accessed)")

    def sample_conditions(self, batch_size):
        idxs  = np.random.randint(len(self), size=batch_size)
        masks = [torch.tensor(
                     np.array(self.mask_patches[i], dtype=np.float32))
                 for i in idxs]
        return torch.stack(masks).cuda()

    def __len__(self):
        return len(self.ct_patches)

    def __getitem__(self, idx):
        # np.array() copies just this patch from mmap into contiguous RAM
        return {
            'ct':   torch.tensor(
                        np.array(self.ct_patches[idx],   dtype=np.float32)),
            'mask': torch.tensor(
                        np.array(self.mask_patches[idx], dtype=np.float32)),
        }


# backward compat
LungNPZDataset = PrecomputedPatchDataset

def worker_init_fn(worker_id):
    np.random.seed(worker_id)