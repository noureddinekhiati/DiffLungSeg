# gpu_radiomics.py  (fixed)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GPURadiomics(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('gabor_filters', self._make_gabor_filters())
        self.register_buffer('lbp_filters',   self._make_lbp_filters())

    def _make_gabor_filters(self):
        """Returns (8, 1, 3, 3, 3)"""
        filters = []
        coords  = torch.linspace(-1, 1, 3)
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
        # zz, yy, xx each (3,3,3)

        for freq in [2.0, 4.0]:
            for angle in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
                gauss   = torch.exp(-(xx**2 + yy**2 + zz**2) / 0.5)
                carrier = torch.cos(freq * (xx*np.cos(angle) + yy*np.sin(angle)))
                f = gauss * carrier          # (3, 3, 3)
                f = f - f.mean()
                if f.std() > 1e-8:
                    f = f / f.std()
                filters.append(f)            # append (3,3,3) — NOT unsqueezed

        # stack: list of 8 × (3,3,3) → (8,3,3,3) → unsqueeze(1) → (8,1,3,3,3)
        return torch.stack(filters).unsqueeze(1)

    def _make_lbp_filters(self):
        """Returns (6, 1, 3, 3, 3)"""
        filters = []
        for axis in range(3):
            for sign in [1, -1]:
                f = torch.zeros(3, 3, 3)    # (3,3,3)
                f[1, 1, 1] = 1.0
                idx = [1, 1, 1]
                idx[axis] += sign
                f[idx[0], idx[1], idx[2]] = -1.0
                filters.append(f)           # append (3,3,3) — NOT unsqueezed

        # stack: list of 6 × (3,3,3) → (6,3,3,3) → unsqueeze(1) → (6,1,3,3,3)
        return torch.stack(filters).unsqueeze(1)

    def _hu_stats(self, x, mask):
        """Returns (B, 5): mean, std, skew, kurtosis, entropy"""
        B = x.shape[0]
        feats = []
        for b in range(B):
            v = x[b, 0][mask[b, 0] > 0.5]
            if len(v) < 10:
                feats.append(torch.zeros(5, device=x.device))
                continue
            mu    = v.mean()
            sigma = v.std() + 1e-8
            z     = (v - mu) / sigma
            skew  = (z**3).mean()
            kurt  = (z**4).mean() - 3.0
            hist  = torch.histc(v, bins=50, min=-1.0, max=1.0)
            hist  = hist / (hist.sum() + 1e-8)
            entropy = -(hist * (hist + 1e-8).log()).sum()
            feats.append(torch.stack([mu, sigma, skew, kurt, entropy]))
        return torch.stack(feats)  # (B, 5)

    def _glcm_proxy(self, x, mask):
        """Returns (B, 6): contrast+homogeneity for 3 axes"""
        B = x.shape[0]
        results = []
        for b in range(B):
            vol = x[b, 0]
            mk  = mask[b, 0] > 0.5
            row = []
            for axis in range(3):
                sl_o = [slice(None)] * 3
                sl_n = [slice(None)] * 3
                sl_o[axis] = slice(None, -1)
                sl_n[axis] = slice(1, None)
                v1 = vol[sl_o[0], sl_o[1], sl_o[2]]
                v2 = vol[sl_n[0], sl_n[1], sl_n[2]]
                mk_p = mk[sl_o[0], sl_o[1], sl_o[2]] & \
                       mk[sl_n[0], sl_n[1], sl_n[2]]
                if mk_p.sum() < 4:
                    row.extend([0., 0.])
                    continue
                d = (v1[mk_p] - v2[mk_p])
                row.append(d.pow(2).mean())
                row.append((1.0 / (1.0 + d.pow(2))).mean())
            results.append(torch.tensor(row, device=x.device, dtype=torch.float32))
        return torch.stack(results)  # (B, 6)

    def forward(self, x, mask):
        """
        x:    (B, 1, D, H, W) in [-1, 1]
        mask: (B, 1, D, H, W) binary
        Returns: (B, 25)
        """
        # 1. HU stats (B, 5)
        hu = self._hu_stats(x, mask)

        # 2. Gabor responses (B, 8)
        # gabor_filters: (8, 1, 3, 3, 3) ✓
        gab = F.conv3d(x, self.gabor_filters, padding=1)  # (B, 8, D, H, W)
        gab = gab * mask
        lung_vol = mask.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1)
        gab = gab.abs().sum(dim=(2, 3, 4)) / lung_vol.squeeze(-1).squeeze(-1).squeeze(-1)
        # gab: (B, 8)

        # 3. LBP responses (B, 6)
        # lbp_filters: (6, 1, 3, 3, 3) ✓
        lbp = F.conv3d(x, self.lbp_filters, padding=1)   # (B, 6, D, H, W)
        lbp = lbp * mask
        lbp_feat = (lbp > 0).float() * mask
        lbp_feat = lbp_feat.sum(dim=(2, 3, 4)) / lung_vol.squeeze(-1).squeeze(-1).squeeze(-1)
        # lbp_feat: (B, 6)

        # 4. GLCM proxy (B, 6)
        glcm = self._glcm_proxy(x, mask)

        # 5. Concatenate (B, 25)
        return torch.cat([hu, gab, lbp_feat, glcm], dim=1)


_gpu_radiomics = None

def get_gpu_radiomics(device):
    global _gpu_radiomics
    if _gpu_radiomics is None:
        _gpu_radiomics = GPURadiomics().to(device)
        _gpu_radiomics.eval()
    return _gpu_radiomics

def extract_radiomic_features_gpu(ct_patch, mask_patch, device=None):
    if device is None:
        device = ct_patch.device
    extractor = get_gpu_radiomics(device)
    with torch.no_grad():
        return extractor(ct_patch, mask_patch)