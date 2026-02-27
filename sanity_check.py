#!/usr/bin/env python3
"""
sanity_check.py
───────────────
Run this BEFORE starting distillation training to verify
all components work correctly.

Usage:
    python sanity_check.py

Expected output: ALL 5 TESTS PASSED
If any test fails, fix it before running train_distill.py
"""
import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"

def check(condition, msg=""):
    if not condition:
        print(f"  {FAIL} {msg}")
        sys.exit(1)
    return True

# ─────────────────────────────────────────────────────────────────
print("=" * 55)
print("  DiffSegLung Distillation — Sanity Check")
print("=" * 55)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if device.type == 'cuda':
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

B  = 4   # batch size
D  = 32  # depth
H  = 96  # height
W  = 96  # width

# ─────────────────────────────────────────────────────────────────
print("\n[1/5] GPU Radiomics")
print("      Input:  (B,1,D,H,W) CT patch in [-1,1]")
print("      Output: (B,25) feature vector")

from gpu_radiomics import GPURadiomics

ct_patch   = torch.randn(B, 1, D, H, W).to(device) * 0.3 - 0.5
mask_patch = torch.ones(B, 1, D, H, W).to(device)

extractor  = GPURadiomics().to(device)
with torch.no_grad():
    radio_feats = extractor(ct_patch, mask_patch)

check(radio_feats.shape == (B, 25),
      f"Shape wrong: {radio_feats.shape} (expected ({B},25))")
check(not torch.isnan(radio_feats).any(),
      "NaN values in radiomic features")
check((radio_feats.abs() < 1000).all(),
      "Suspiciously large values in radiomic features")

print(f"      Shape:  {radio_feats.shape}  {PASS}")
print(f"      Range:  [{radio_feats.min():.2f}, {radio_feats.max():.2f}]")
print(f"      NaNs:   {torch.isnan(radio_feats).sum().item()}")

# ─────────────────────────────────────────────────────────────────
print("\n[2/5] Projection Heads")

from distillation import SpatialProjectionHead, RadiomicProjectionHead

# Bottleneck shape: (B, 256, D/8, H/8, W/8)
Dp, Hp, Wp = D//8, H//8, W//8
print(f"      Bottleneck shape: ({B}, 256, {Dp}, {Hp}, {Wp})")

proj_head  = SpatialProjectionHead(in_dim=256, hidden_dim=256, out_dim=128).to(device)
radio_proj = RadiomicProjectionHead(in_dim=25,  hidden_dim=128, out_dim=128).to(device)

fake_bottleneck = torch.randn(B, 256, Dp, Hp, Wp).to(device)

z_diff  = proj_head(fake_bottleneck)     # (B, 128, Dp, Hp, Wp)
z_radio = radio_proj(radio_feats)        # (B, 128)

check(z_diff.shape  == (B, 128, Dp, Hp, Wp),
      f"z_diff shape wrong: {z_diff.shape}")
check(z_radio.shape == (B, 128),
      f"z_radio shape wrong: {z_radio.shape}")
check(not torch.isnan(z_diff).any(),  "NaN in z_diff")
check(not torch.isnan(z_radio).any(), "NaN in z_radio")

# Check L2 normalized (norm should be ~1 per voxel/patch)
norms_diff  = z_diff.norm(dim=1)   # (B, Dp, Hp, Wp)
norms_radio = z_radio.norm(dim=1)  # (B,)

check((norms_diff  - 1.0).abs().max() < 0.01,
      f"z_diff not normalized: norms range [{norms_diff.min():.3f},{norms_diff.max():.3f}]")
check((norms_radio - 1.0).abs().max() < 0.01,
      f"z_radio not normalized: norms range [{norms_radio.min():.3f},{norms_radio.max():.3f}]")

print(f"      z_diff  shape: {z_diff.shape}  {PASS}")
print(f"      z_radio shape: {z_radio.shape}  {PASS}")
print(f"      z_diff  norm:  {norms_diff.mean():.4f} (expect 1.0)")
print(f"      z_radio norm:  {norms_radio.mean():.4f} (expect 1.0)")

# ─────────────────────────────────────────────────────────────────
print("\n[3/5] InfoNCE Loss")

from distillation import InfoNCELoss

infonce = InfoNCELoss(temperature=0.07).to(device)
loss    = infonce(z_diff, z_radio)

import math
random_baseline = math.log(B)   # loss for random embeddings = log(batch_size)

check(not torch.isnan(loss),  "NaN loss")
check(loss.item() > 0,        "Loss should be positive")
check(loss.item() < 20.0,     f"Loss too large: {loss.item():.4f}")

print(f"      Loss value:      {loss.item():.4f}")
print(f"      Random baseline: {random_baseline:.4f}  (log({B}))")
print(f"      Status: {'close to random (expected at init)' if abs(loss.item()-random_baseline)<1.0 else 'unusual'}")
print(f"      {PASS}")

# ─────────────────────────────────────────────────────────────────
print("\n[4/5] Lambda Schedule")

from distillation import get_distill_lambda

tests = [
    (0,     5000, 7000, 0.5,  0.00),
    (4999,  5000, 7000, 0.5,  0.00),
    (5000,  5000, 7000, 0.5,  0.00),
    (5500,  5000, 7000, 0.5,  0.125),
    (6000,  5000, 7000, 0.5,  0.25),
    (7000,  5000, 7000, 0.5,  0.50),
    (10000, 5000, 7000, 0.5,  0.50),
]

all_ok = True
for step, ws, we, lm, expected in tests:
    lam = get_distill_lambda(step, ws, we, lm)
    ok  = abs(lam - expected) < 0.01
    print(f"      step={step:6d}  λ={lam:.4f}  "
          f"(expect {expected:.4f})  {'✓' if ok else '✗ FAIL'}")
    if not ok: all_ok = False

check(all_ok, "Lambda schedule incorrect")
print(f"      {PASS}")

# ─────────────────────────────────────────────────────────────────
print("\n[5/5] Full Backward Pass (gradients flow through everything)")

proj_head2  = SpatialProjectionHead(256, 256, 128).to(device)
radio_proj2 = RadiomicProjectionHead(25, 128, 128).to(device)
infonce2    = InfoNCELoss(0.07).to(device)
extractor2  = GPURadiomics().to(device)

# Simulate what happens in train_distill.py
fake_bottleneck2 = torch.randn(B, 256, Dp, Hp, Wp,
                                device=device, requires_grad=True)
ct2   = torch.randn(B, 1, D, H, W, device=device)
mask2 = torch.ones(B, 1, D, H, W, device=device)

# Forward
with torch.no_grad():
    radio_f2 = extractor2(ct2, mask2)   # (B,25) — no grad (teacher)

z_s = proj_head2(fake_bottleneck2)      # (B,128,Dp,Hp,Wp)
z_t = radio_proj2(radio_f2)             # (B,128)

loss2 = infonce2(z_s, z_t)

# Also add a fake diffusion loss
fake_noise = torch.randn_like(ct2)
fake_pred  = torch.randn_like(ct2)
diff_loss2 = (fake_noise - fake_pred).abs().mean()

total = diff_loss2 + 0.5 * loss2
total.backward()

grad = fake_bottleneck2.grad
check(grad is not None,             "No gradient on bottleneck")
check(grad.norm().item() > 0,       "Zero gradient (dead network)")
check(not torch.isnan(grad).any(),  "NaN gradient")

# Check proj_head params got gradients
proj_grads = [p.grad for p in proj_head2.parameters() if p.grad is not None]
check(len(proj_grads) > 0, "No gradients on proj_head params")

print(f"      Bottleneck grad norm:   {grad.norm():.6f}")
print(f"      diff_loss:              {diff_loss2.item():.4f}")
print(f"      infonce_loss:           {loss2.item():.4f}")
print(f"      total_loss:             {total.item():.4f}")
print(f"      proj_head params with grad: {len(proj_grads)}")
print(f"      {PASS}")

# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  ALL 5 TESTS PASSED")
print("  You can now run train_distill.py")
print("=" * 55)

# Print recommended launch command
print("""
Recommended launch command:

    python train_distill.py \\
        --npz_dir  \\
        --resume   \\
        --start_step 3000 \\
        --out ./runs/distill \\
        --steps 30000 \\
        --batch_size 2 \\
        --lambda_max 0.3 \\
        --warmup_start 5000 \\
        --warmup_end 7000

Watch TensorBoard:
    tensorboard --logdir ./runs/distill
""")
