#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_utils.py
──────────────
Shared patch extraction used by both visualize and precompute scripts.
Returns (ct_patch, mask_patch) pairs so features are always
computed on lung voxels only.
"""

import numpy as np


def extract_patches(ct_hu:      np.ndarray,
                    mask:       np.ndarray,
                    patch_size: int   = 32,
                    stride:     int   = 16,
                    min_lung:   float = 0.5):
    """
    Extract overlapping 3D patches strictly inside the lung mask.

    Args:
        ct_hu      : (Z, Y, X) float32 HU values
        mask       : (Z, Y, X) float32 binary lung mask
        patch_size : cubic patch size in voxels
        stride     : step between patches (stride=patch_size/2 = 50% overlap)
        min_lung   : minimum fraction of lung voxels required to keep patch
                     0.5 = at least 50% of patch must be inside lung
                     raised from 0.4 to 0.5 for cleaner features

    Returns:
        ct_patches   : list of (patch_size, patch_size, patch_size) arrays
        mask_patches : list of (patch_size, patch_size, patch_size) arrays
        centers      : list of (cz, cy, cx) center coordinates
    """
    Z, Y, X = ct_hu.shape
    ct_patches, mask_patches, centers = [], [], []

    for z in range(0, Z - patch_size + 1, stride):
        for y in range(0, Y - patch_size + 1, stride):
            for x in range(0, X - patch_size + 1, stride):

                mp = mask[z:z+patch_size, y:y+patch_size, x:x+patch_size]

                # Reject patches with too little lung
                if mp.mean() < min_lung:
                    continue

                cp = ct_hu[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                ct_patches.append(cp)
                mask_patches.append(mp)
                centers.append((
                    z + patch_size // 2,
                    y + patch_size // 2,
                    x + patch_size // 2,
                ))

    return ct_patches, mask_patches, centers


def build_label_volume(shape:      tuple,
                       centers:    list,
                       labels:     np.ndarray,
                       patch_size: int,
                       mask:       np.ndarray = None) -> np.ndarray:
    """
    Paint each patch region with its cluster label.
    If mask provided, only lung voxels get a label (others stay -1).

    Returns int8 volume, -1 = no label / outside lung.
    """
    label_vol = np.full(shape, -1, dtype=np.int8)
    half      = patch_size // 2
    Z, Y, X   = shape

    for (cz, cy, cx), lab in zip(centers, labels):
        z0, z1 = max(0, cz-half), min(Z, cz+half)
        y0, y1 = max(0, cy-half), min(Y, cy+half)
        x0, x1 = max(0, cx-half), min(X, cx+half)
        label_vol[z0:z1, y0:y1, x0:x1] = lab

    # Final masking — remove labels outside lung
    if mask is not None:
        label_vol[mask < 0.5] = -1

    return label_vol
