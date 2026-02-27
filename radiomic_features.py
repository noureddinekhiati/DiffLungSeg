#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radiomic_features.py
────────────────────
Full 34-dim radiomic descriptor per patch, computed ONLY on lung voxels.

  HU statistics  : 5  features
  GLCM           : 6  features
  LBP            : 10 features
  Gabor          : 13 features
  ─────────────────
  Total          : 34 features

IMPORTANT: all functions accept an optional `mask_patch` argument.
When provided, non-lung voxels are excluded from feature computation.
This prevents background air / bone / body tissue from corrupting features.
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.ndimage import convolve
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

HU_MIN, HU_MAX = -1000, 400


# ─────────────────────────────────────────────────────────────────
# Masking helper
# ─────────────────────────────────────────────────────────────────

def apply_lung_mask(patch_hu: np.ndarray,
                    mask_patch: np.ndarray) -> np.ndarray:
    """
    Zero out non-lung voxels in patch.
    Non-lung voxels are set to HU_MIN (-1000 = air) so they don't
    contribute to texture statistics but don't create outliers either.
    """
    masked = patch_hu.copy()
    masked[mask_patch < 0.5] = HU_MIN
    return masked


def get_lung_voxels(patch_hu: np.ndarray,
                    mask_patch: np.ndarray) -> np.ndarray:
    """Return only the HU values of lung voxels as 1D array."""
    return patch_hu[mask_patch >= 0.5].flatten()


# ─────────────────────────────────────────────────────────────────
# 1. HU STATISTICS (5 features) — density / physics layer
# ─────────────────────────────────────────────────────────────────

def extract_hu_stats(patch_hu: np.ndarray,
                     mask_patch: np.ndarray = None) -> np.ndarray:
    """
    Computed ONLY on lung voxels when mask provided.

    mean HU   → tissue type:  emphysema <-700, healthy -700/-300,
                fibrosis -300/-100, consolidation >-100
    std       → heterogeneity: fibrosis=high, emphysema=low
    skewness  → distribution shape
    kurtosis  → peakedness: cavity = very peaked (air + dense wall)
    entropy   → information content
    """
    if mask_patch is not None:
        v = get_lung_voxels(patch_hu, mask_patch)
        if len(v) < 10:
            return np.zeros(5, dtype=np.float32)
    else:
        v = patch_hu.flatten()

    hist, _ = np.histogram(v, bins=50, range=(HU_MIN, HU_MAX), density=True)
    hist    = hist + 1e-10
    ent     = -np.sum(hist * np.log(hist))

    return np.array([
        v.mean(),
        v.std(),
        float(skew(v)),
        float(kurtosis(v)),
        ent,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# 2. GLCM (6 features) — spatial texture layer
# ─────────────────────────────────────────────────────────────────

def extract_glcm_features(patch_hu: np.ndarray,
                           mask_patch: np.ndarray = None) -> np.ndarray:
    """
    GLCM on middle axial slice, lung voxels only.
    Non-lung voxels are replaced with the lung mean HU
    so they don't create artificial edges in the co-occurrence matrix.

    contrast      → large = big jumps (fibrosis reticular)
    dissimilarity → robust contrast
    homogeneity   → large = smooth (emphysema uniform air)
    energy/ASM    → texture uniformity
    correlation   → linear dependency between neighbors
    """
    mid = patch_hu.shape[0] // 2
    sl  = patch_hu[mid].copy()

    if mask_patch is not None:
        sl_mask = mask_patch[mid]
        if sl_mask.sum() < 10:
            return np.zeros(6, dtype=np.float32)
        # Replace non-lung with lung mean → no artificial edges
        lung_mean = float(sl[sl_mask >= 0.5].mean())
        sl[sl_mask < 0.5] = lung_mean

    # Quantize to 0-63
    sl_q = np.clip(sl, HU_MIN, HU_MAX)
    sl_q = ((sl_q - HU_MIN) / (HU_MAX - HU_MIN) * 63).astype(np.uint8)

    try:
        glcm = graycomatrix(sl_q, distances=[1],
                            angles=[0, np.pi/2],
                            levels=64, symmetric=True, normed=True)
        return np.array([
            graycoprops(glcm, p).mean()
            for p in ['contrast', 'dissimilarity', 'homogeneity',
                      'energy', 'correlation', 'ASM']
        ], dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────
# 3. LBP (10 features) — local microstructure layer
# ─────────────────────────────────────────────────────────────────

def extract_lbp_features(patch_hu: np.ndarray,
                          mask_patch: np.ndarray = None,
                          n_points: int = 8,
                          radius:   int = 1,
                          n_bins:   int = 10) -> np.ndarray:
    """
    LBP on 3 orthogonal mid-slices.
    Non-lung voxels replaced with lung mean before LBP computation,
    then LBP histogram computed ONLY on lung pixel positions.

    Why this matters: without masking, the lung boundary (lung/air interface)
    creates very strong LBP edge responses that dominate the histogram
    and are not related to pathology texture.
    """
    Z, Y, X = patch_hu.shape

    def _lbp_on_slice(sl_hu, sl_mask=None):
        sl = sl_hu.copy()
        if sl_mask is not None and sl_mask.sum() > 10:
            lung_mean = float(sl[sl_mask >= 0.5].mean())
            sl[sl_mask < 0.5] = lung_mean
        # Quantize to uint8
        sl_q = np.clip(sl, HU_MIN, HU_MAX)
        sl_q = ((sl_q - HU_MIN) / (HU_MAX - HU_MIN) * 255).astype(np.uint8)
        lbp  = local_binary_pattern(sl_q, n_points, radius, method='uniform')
        # Histogram only on lung pixels
        if sl_mask is not None and sl_mask.sum() > 10:
            lbp_lung = lbp[sl_mask >= 0.5]
        else:
            lbp_lung = lbp.flatten()
        hist, _ = np.histogram(lbp_lung, bins=n_bins,
                               range=(0, n_points + 2), density=True)
        return hist

    slices_hu   = [patch_hu[Z//2, :, :],
                   patch_hu[:, Y//2, :],
                   patch_hu[:, :, X//2]]
    slices_mask = [None, None, None]
    if mask_patch is not None:
        slices_mask = [mask_patch[Z//2, :, :],
                       mask_patch[:, Y//2, :],
                       mask_patch[:, :, X//2]]

    hists = [_lbp_on_slice(sl_hu, sl_mask)
             for sl_hu, sl_mask in zip(slices_hu, slices_mask)]

    return np.mean(hists, axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# 4. GABOR WAVELETS (13 features) — oriented edges / frequency layer
# ─────────────────────────────────────────────────────────────────

def _gabor_kernel_2d(frequency: float,
                      theta:     float,
                      sigma:     float = 3.0,
                      size:      int   = 15) -> np.ndarray:
    half = size // 2
    x, y = np.meshgrid(np.arange(-half, half+1),
                        np.arange(-half, half+1))
    x_theta =  x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    envelope = np.exp(-0.5 * (x_theta**2 + y_theta**2) / sigma**2)
    carrier  = np.cos(2 * np.pi * frequency * x_theta)
    kernel   = envelope * carrier
    kernel  -= kernel.mean()
    return kernel.astype(np.float32)


def extract_gabor_features(patch_hu: np.ndarray,
                            mask_patch: np.ndarray = None) -> np.ndarray:
    """
    Gabor at 3 scales × 4 orientations on middle axial slice.
    Energy computed ONLY on lung pixels.

    Without masking: the lung boundary dominates Gabor response
    (it's the strongest edge in the image) → all patches look the same.
    With masking: only internal texture edges contribute.

    fine scale   (freq=0.40) → fibrosis thin strands
    medium scale (freq=0.25) → GGO haze, vessel patterns
    coarse scale (freq=0.10) → consolidation large blobs
    """
    mid  = patch_hu.shape[0] // 2
    sl   = patch_hu[mid].copy()

    sl_mask = None
    if mask_patch is not None:
        sl_mask = mask_patch[mid]
        if sl_mask.sum() < 10:
            return np.zeros(13, dtype=np.float32)
        # Replace non-lung with lung mean → eliminates boundary edge response
        lung_mean = float(sl[sl_mask >= 0.5].mean())
        sl[sl_mask < 0.5] = lung_mean

    # Normalize to [0,1]
    sl_n = (np.clip(sl, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)

    frequencies = [0.10, 0.25, 0.40]
    thetas      = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    energies = []
    for freq in frequencies:
        for theta in thetas:
            kernel   = _gabor_kernel_2d(freq, theta, sigma=3.0, size=15)
            response = convolve(sl_n, kernel, mode='reflect')
            # Energy only on lung pixels
            if sl_mask is not None:
                energy = float(np.mean(response[sl_mask >= 0.5] ** 2))
            else:
                energy = float(np.mean(response ** 2))
            energies.append(energy)

    energies.append(float(np.mean(energies)))  # overall mean
    return np.array(energies, dtype=np.float32)  # (13,)


# ─────────────────────────────────────────────────────────────────
# Full 34-dim feature vector
# ─────────────────────────────────────────────────────────────────

def extract_all_features(patch_hu:   np.ndarray,
                          mask_patch: np.ndarray = None) -> np.ndarray:
    """
    Full radiomic feature extraction — lung voxels only.

    Args:
        patch_hu   : (Z, Y, X) float32 in HU
        mask_patch : (Z, Y, X) float32 binary lung mask — STRONGLY recommended

    Returns: (34,) float32
        [0:5]   HU stats     — density / physics
        [5:11]  GLCM         — spatial texture
        [11:21] LBP          — local microstructure
        [21:34] Gabor        — oriented edges, multi-scale
    """
    return np.concatenate([
        extract_hu_stats(patch_hu,    mask_patch),   # 5
        extract_glcm_features(patch_hu, mask_patch), # 6
        extract_lbp_features(patch_hu,  mask_patch), # 10
        extract_gabor_features(patch_hu, mask_patch),# 13
    ])


# ─────────────────────────────────────────────────────────────────
# Feature names
# ─────────────────────────────────────────────────────────────────

FEATURE_NAMES = (
    ['hu_mean', 'hu_std', 'hu_skew', 'hu_kurt', 'hu_entropy'] +
    ['glcm_contrast', 'glcm_dissim', 'glcm_homog',
     'glcm_energy', 'glcm_corr', 'glcm_asm'] +
    [f'lbp_{i}' for i in range(10)] +
    [f'gabor_f{f}_t{t}' for f in range(3) for t in range(4)] +
    ['gabor_mean']
)
assert len(FEATURE_NAMES) == 34
