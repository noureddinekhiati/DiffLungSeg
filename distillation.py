# distillation.py
# ────────────────
# All distillation components in one file:
#   1. SpatialProjectionHead  — maps bottleneck → 128-dim per voxel
#   2. RadiomicProjectionHead — maps radiomic features → 128-dim per patch
#   3. InfoNCE loss           — pulls matched pairs together
#   4. lambda_schedule        — warmup from step 5000

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 1. SpatialProjectionHead
# ─────────────────────────────────────────────────────────────────
# Used BOTH during training (InfoNCE) AND at inference (segmentation).
# Maps bottleneck (B, 256, D', H', W') → (B, 128, D', H', W')
# via 1×1×1 convolutions — keeps spatial resolution.
# At inference: upsample to full patch size → voxel-level features.

class SpatialProjectionHead(nn.Module):
    """
    1x1x1 conv projection: 256 → 128 channels.
    Keeps spatial dimensions intact.
    Output is L2-normalized along channel dim (unit sphere per voxel).

    Architecture:
        Conv3d(256, 256, 1) → BN → ReLU → Conv3d(256, 128, 1) → L2 norm
    """
    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, out_dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        # x: (B, 256, D', H', W')
        out = self.net(x)               # (B, 128, D', H', W')
        # L2 normalize along channel dim → unit sphere per voxel
        return F.normalize(out, dim=1)  # (B, 128, D', H', W')


# ─────────────────────────────────────────────────────────────────
# 2. RadiomicProjectionHead
# ─────────────────────────────────────────────────────────────────
# Teacher network. Maps radiomic features (B, 25) → (B, 128).
# Only used during training. Discarded at inference.
# Small MLP: 25 → 128 → 128, L2 normalized.

class RadiomicProjectionHead(nn.Module):
    """
    MLP projection for radiomic teacher features.
    Input:  (B, 25)  GPU radiomic features
    Output: (B, 128) normalized embedding on unit sphere
    """
    def __init__(self, in_dim=25, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        # x: (B, 25)
        out = self.net(x)               # (B, 128)
        return F.normalize(out, dim=1)  # unit sphere


# ─────────────────────────────────────────────────────────────────
# 3. InfoNCE Loss
# ─────────────────────────────────────────────────────────────────
# Pulls z_diff[i] toward z_radio[i] (same patch, positive pair)
# Pushes z_diff[i] away from z_radio[j≠i] (different patches, negatives)
#
# Implementation detail:
# z_diff is spatial (B, 128, D', H', W') → we average over spatial dims
# to get one vector per patch (B, 128) for the loss.
# This is correct because the radiomic teacher gives one vector per patch.

class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) contrastive loss.

    Given:
      z_diff  : (B, 128, D', H', W') — spatial student embeddings
      z_radio : (B, 128)             — radiomic teacher embeddings

    Steps:
      1. Pool z_diff spatially → (B, 128)
      2. Compute similarity matrix (B × B)
      3. Diagonal = positive pairs, off-diagonal = negatives
      4. Cross-entropy loss: maximize similarity for positive pairs

    Temperature τ=0.07 (standard for medical imaging contrastive learning)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_diff, z_radio):
        """
        z_diff:  (B, 128, D', H', W')
        z_radio: (B, 128)
        Returns: scalar loss
        """
        B = z_diff.shape[0]

        # Pool spatial dims → (B, 128)
        z_student = z_diff.mean(dim=(2, 3, 4))        # (B, 128)
        z_student = F.normalize(z_student, dim=1)      # re-normalize after pooling
        z_teacher = z_radio                             # already normalized (B, 128)

        # Similarity matrix: (B, B)
        # sim[i,j] = cosine similarity between student[i] and teacher[j]
        sim = torch.mm(z_student, z_teacher.t()) / self.temperature  # (B, B)

        # Labels: diagonal is the positive pair (i matches i)
        labels = torch.arange(B, device=z_diff.device)

        # Cross-entropy: for each student[i], identify teacher[i] among all teachers
        loss = F.cross_entropy(sim, labels)

        return loss


# ─────────────────────────────────────────────────────────────────
# 4. Lambda schedule — warmup
# ─────────────────────────────────────────────────────────────────

def get_distill_lambda(step,
                        warmup_start=5000,
                        warmup_end=7000,
                        lambda_max=0.5):
    """
    Lambda schedule for distillation loss weight.

    Timeline:
      steps 0 → warmup_start:     lambda = 0.0
        (pure diffusion, UNet learns to reconstruct well)
      steps warmup_start → warmup_end: lambda ramps 0 → lambda_max
        (gradually introduce distillation)
      steps warmup_end → end:     lambda = lambda_max
        (full distillation)

    Default lambda_max=0.5 means:
      total_loss = diff_loss + 0.5 * infonce_loss
    """
    if step < warmup_start:
        return 0.0
    elif step < warmup_end:
        progress = (step - warmup_start) / (warmup_end - warmup_start)
        return lambda_max * progress
    else:
        return lambda_max