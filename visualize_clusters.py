#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_clusters.py
─────────────────────
Visualize radiomic clustering on ONE NPZ volume.
All features computed ONLY on lung voxels (mask applied inside each descriptor).

Usage:
    python visualize_clusters.py \
        --npz /home/nkhiati/data_disk/CT_DATA/DIFF_SEG_LUNG_DATA/bpco__AKT_HIK_POUMON.npz \
        --out ./cluster_preview \
        --k 5

Try k=4,5,6 and pick the one with cleanest anatomical separation.
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from radiomic_features import extract_all_features, FEATURE_NAMES, HU_MIN, HU_MAX
from patch_utils import extract_patches, build_label_volume

# ─────────────────────────────────────────────────────────────────
PATHOLOGY_RANGES = {
    "Emphysema":    (-1000, -700),
    "Healthy Lung": (-700,  -300),
    "Fibrosis/GGO": (-300,  -100),
    "Consolidation":(-100,   400),
}
PATHOLOGY_COLORS = {
    "Emphysema":     "#4477AA",
    "Healthy Lung":  "#228833",
    "Fibrosis/GGO":  "#FF8800",
    "Consolidation": "#CC3311",
    "Unknown":       "#BBBBBB",
}

def denorm_hu(ct_norm):
    """
    Inverse of convert_to_npz.py normalization:
      Forward: norm = (HU + 1000) / 700 - 1
      Inverse: HU   = (norm + 1) * 700 - 1000

    Checks:
      norm=-1.0  -> HU=-1000  (air)
      norm= 0.0  -> HU= -300  (normal lung boundary)
      norm=+1.0  -> HU= +400  (bone)
    """
    return (ct_norm + 1.0) * 700.0 - 1000.0

def assign_pathology(mean_hu):
    for name, (lo, hi) in PATHOLOGY_RANGES.items():
        if lo <= mean_hu < hi:
            return name
    return "Unknown"

def cluster_patches(features, k):
    scaler = StandardScaler()
    f_norm = scaler.fit_transform(features)
    n_pca  = min(15, features.shape[1], len(features) - 1)
    f_pca  = PCA(n_components=n_pca, random_state=42).fit_transform(f_norm)
    labels = KMeans(n_clusters=k, random_state=42,
                    n_init=20, max_iter=300).fit_predict(f_pca)
    return labels

def name_clusters(features, labels, k):
    result = {}
    for c in range(k):
        mask_c  = labels == c
        mean_hu = float(features[mask_c, 0].mean()) if mask_c.sum() > 0 else 0.
        result[c] = (assign_pathology(mean_hu), mean_hu)
    return result

def visualize(ct_hu, mask, label_vol, features, labels,
              cluster_names, k, out_path, patient_id):

    Z, Y, X = ct_hu.shape
    colors  = [mcolors.to_rgb(
                PATHOLOGY_COLORS.get(cluster_names[c][0], '#BBBBBB'))
               for c in range(k)]

    # Pick 4 axial slices inside lung
    lung_z    = np.where(mask.any(axis=(1, 2)))[0]
    slice_ids = lung_z[np.linspace(0, len(lung_z)-1, 4, dtype=int)] \
                if len(lung_z) >= 4 else np.linspace(0, Z-1, 4, dtype=int)

    # RGBA overlay — only lung voxels colored
    rgba_vol = np.zeros((*label_vol.shape, 4), dtype=np.float32)
    for c in range(k):
        where = label_vol == c
        rgba_vol[where, :3] = colors[c]
        rgba_vol[where,  3] = 0.65
    # label_vol == -1 stays transparent (outside lung)

    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle(
        f"Radiomic Clustering — lung voxels only\n"
        f"{patient_id}   K={k}   "
        f"34-dim: HU(5)+GLCM(6)+LBP(10)+Gabor(13)",
        fontsize=12, fontweight='bold'
    )

    # ── Row 0: raw CT ─────────────────────────────────────────────
    for col, sl in enumerate(slice_ids):
        ax = axes[0, col]
        ax.imshow(ct_hu[sl], cmap='gray', vmin=-1000, vmax=400)
        ax.set_title(f"CT  z={sl}", fontsize=9)
        ax.axis('off')

    # ── Row 1: cluster overlay ────────────────────────────────────
    for col, sl in enumerate(slice_ids):
        ax = axes[1, col]
        ax.imshow(ct_hu[sl], cmap='gray', vmin=-1000, vmax=400)
        ax.imshow(rgba_vol[sl])
        ax.set_title(f"Clusters  z={sl}", fontsize=9)
        ax.axis('off')

    # ── Row 2: coronal + per-cluster HU histograms ────────────────
    mid_y = Y // 2
    ax_cor = axes[2, 0]
    ax_cor.imshow(ct_hu[:, mid_y, :], cmap='gray',
                  vmin=-1000, vmax=400, aspect='auto')
    ax_cor.imshow(rgba_vol[:, mid_y, :], aspect='auto')
    ax_cor.set_title(f"Coronal  y={mid_y}\n(lung mask applied)", fontsize=9)
    ax_cor.axis('off')

    for col in range(1, 4):
        c  = col - 1
        ax = axes[2, col]
        if c < k:
            mask_c  = labels == c
            name, mean_hu = cluster_names[c]
            hu_vals = features[mask_c, 0]
            ax.hist(hu_vals, bins=30, color=colors[c],
                    edgecolor='k', linewidth=0.5)
            ax.axvline(mean_hu, color='k', ls='--', lw=1.5)
            ax.set_title(f"C{c}: {name}\n"
                         f"μHU={mean_hu:.0f}  n={mask_c.sum()}", fontsize=8)
            ax.set_xlabel("Mean HU (lung voxels only)", fontsize=7)
        else:
            ax.axis('off')

    # ── Row 3: PCA scatter + feature importance + Gabor + LBP ────
    # PCA 2D
    ax_pca = axes[3, 0]
    f2d    = PCA(n_components=2, random_state=42).fit_transform(
                 StandardScaler().fit_transform(features))
    for c in range(k):
        mask_c = labels == c
        name, mean_hu = cluster_names[c]
        ax_pca.scatter(f2d[mask_c, 0], f2d[mask_c, 1],
                       c=[colors[c]], s=6, alpha=0.5,
                       label=f"C{c} {name}({mean_hu:.0f})")
    ax_pca.set_title("PCA 2D — all patches", fontsize=9)
    ax_pca.legend(fontsize=5)
    ax_pca.set_xlabel("PC1"); ax_pca.set_ylabel("PC2")

    # Feature importance (variance across cluster means)
    ax_imp = axes[3, 1]
    cluster_means = np.array([features[labels == c].mean(axis=0)
                               for c in range(k)])
    importance = cluster_means.std(axis=0)
    top10 = np.argsort(importance)[-10:][::-1]
    ax_imp.barh([FEATURE_NAMES[i] for i in top10],
                 importance[top10], color='steelblue')
    ax_imp.set_title("Top 10 discriminative features\n"
                     "(variance across cluster means)", fontsize=8)
    ax_imp.tick_params(labelsize=7)

    # Gabor heatmap per cluster
    ax_gab = axes[3, 2]
    gabor_means = np.array([features[labels == c, 21:34].mean(axis=0)
                             for c in range(k)])
    im = ax_gab.imshow(gabor_means, cmap='hot', aspect='auto')
    ax_gab.set_title("Gabor energy per cluster\n"
                     "(rows=clusters, cols=scale×orient)", fontsize=8)
    ax_gab.set_yticks(range(k))
    ax_gab.set_yticklabels([f"C{c} {cluster_names[c][0][:8]}"
                             for c in range(k)], fontsize=7)
    plt.colorbar(im, ax=ax_gab, fraction=0.046)

    # LBP profiles per cluster
    ax_lbp = axes[3, 3]
    for c in range(k):
        lbp_mean = features[labels == c, 11:21].mean(axis=0)
        ax_lbp.plot(lbp_mean, color=colors[c],
                    label=f"C{c}", linewidth=1.5)
    ax_lbp.set_title("LBP histogram per cluster\n"
                     "(computed on lung voxels)", fontsize=8)
    ax_lbp.set_xlabel("LBP bin", fontsize=7)
    ax_lbp.legend(fontsize=6)

    # ── Row 4: summary ────────────────────────────────────────────
    for col in range(4):
        axes[4, col].axis('off')

    summary = "\n".join([
        f"C{c}: {cluster_names[c][0]:<15} "
        f"μHU={cluster_names[c][1]:7.1f}  "
        f"n_patches={int((labels==c).sum())}"
        for c in range(k)
    ])
    axes[4, 0].text(
        0.02, 0.95,
        f"Cluster Summary (K={k})\n\n{summary}\n\n"
        f"All features computed on lung voxels only.\n"
        f"Non-lung voxels replaced with lung mean before GLCM/LBP/Gabor.",
        transform=axes[4, 0].transAxes,
        fontsize=8, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
    legend_handles = [
        plt.Rectangle((0,0),1,1, fc=c, label=n)
        for n, c in PATHOLOGY_COLORS.items()
    ]
    axes[4, 1].legend(handles=legend_handles, loc='center',
                      fontsize=9, title="Pathology colors")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz',        required=True,
                    help='Path to one NPZ file')
    ap.add_argument('--out',        default='./cluster_preview',
                    help='Output directory')
    ap.add_argument('--k',          type=int, default=5,
                    help='Number of clusters (try 4, 5, 6)')
    ap.add_argument('--patch_size', type=int, default=32)
    ap.add_argument('--stride',     type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    patient_id = os.path.basename(args.npz).replace('.npz', '')

    print(f"\nPatient : {patient_id}  K={args.k}")

    # Load
    data    = np.load(args.npz, allow_pickle=True)
    ct_norm = data['ct'].astype(np.float32)
    mask    = data['mask'].astype(np.float32)
    ct_hu   = denorm_hu(ct_norm)

    lung_pct = mask.mean() * 100
    print(f"Volume  : {ct_hu.shape}  "
          f"HU=[{ct_hu.min():.0f}, {ct_hu.max():.0f}]  "
          f"Lung={lung_pct:.1f}%")

    # Extract patches — returns ct + mask patches together
    print("Extracting patches (min 50% lung per patch)...")
    ct_patches, mask_patches, centers = extract_patches(
        ct_hu, mask, args.patch_size, args.stride, min_lung=0.5
    )
    print(f"  {len(ct_patches)} patches inside lung")

    if len(ct_patches) < args.k:
        print(f"ERROR: only {len(ct_patches)} patches, need at least {args.k}")
        return

    # Features — mask passed to every descriptor
    print("Extracting 34-dim features (lung voxels only)...")
    features = np.array([
        extract_all_features(cp, mp)
        for cp, mp in tqdm(zip(ct_patches, mask_patches),
                           total=len(ct_patches), desc="  features")
    ], dtype=np.float32)
    print(f"  Feature matrix: {features.shape}")

    # Cluster
    print(f"Clustering K={args.k}...")
    labels        = cluster_patches(features, args.k)
    cluster_names = name_clusters(features, labels, args.k)

    print("\nCluster summary:")
    for c, (name, hu) in cluster_names.items():
        print(f"  C{c}: {name:<20} μHU={hu:7.1f}  "
              f"n={int((labels==c).sum())}")

    # Build label volume (only lung voxels labeled)
    label_vol = build_label_volume(
        ct_hu.shape, centers, labels, args.patch_size, mask=mask
    )

    # Visualize
    out_png = os.path.join(args.out, f"{patient_id}_k{args.k}.png")
    visualize(ct_hu, mask, label_vol, features, labels,
              cluster_names, args.k, out_png, patient_id)

    # Save JSON
    json_path = out_png.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump({
            "patient_id": patient_id, "k": args.k,
            "n_patches": len(ct_patches),
            "lung_pct": round(lung_pct, 2),
            "clusters": {
                str(c): {"name": n, "mean_hu": round(float(h), 1),
                         "n_patches": int((labels==c).sum())}
                for c, (n, h) in cluster_names.items()
            }
        }, f, indent=2)
    print(f"  JSON : {json_path}")
    print(f"\nDone. Open: {out_png}")


if __name__ == '__main__':
    main()
