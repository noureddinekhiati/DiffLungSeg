# gpu_gmm.py  (v2 — BIC auto-K + adaptive labeling)
# ──────────────────────────────────────────────────
# GPU GMM with:
#   1. Automatic K selection via BIC (K=2..8)
#   2. Patient-adaptive cluster naming via percentiles
#   3. Soft probability output for uncertainty maps
#   4. Batch prediction for 18M+ voxels without OOM

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────
# GPU GMM
# ─────────────────────────────────────────────────────────────────

class GaussianMixtureGPU:
    """
    Full-covariance GMM via EM on GPU (pure PyTorch).

    Parameters
    ----------
    n_components : int    number of tissue clusters
    n_iter       : int    max EM iterations
    tol          : float  convergence threshold on log-likelihood delta
    reg_covar    : float  diagonal regularization (prevents singular matrices)
    random_state : int    seed for reproducibility
    device       : str    'cuda' or 'cpu'
    """

    def __init__(self, n_components=5, n_iter=100, tol=1e-4,
                 reg_covar=1e-4, random_state=42, device='cuda'):
        self.K           = n_components
        self.n_iter      = n_iter
        self.tol         = tol
        self.reg_covar   = reg_covar
        self.device      = torch.device(device)
        self.random_state= random_state

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.pi_      = None   # (K,)       mixing weights
        self.mu_      = None   # (K, D)     means
        self.sigma_   = None   # (K, D, D)  covariances
        self.log_det_ = None   # (K,)       cached log-determinants
        self.inv_sig_ = None   # (K, D, D)  cached inverses
        self.final_ll_= None   # final log-likelihood (for BIC)

    # ── Initialization ────────────────────────────────────────────

    def _init_params(self, X):
        N, D = X.shape
        torch.manual_seed(self.random_state)

        # K-means++ seeding
        idx   = torch.randint(N, (1,), device=self.device).item()
        means = [X[idx]]
        for _ in range(self.K - 1):
            dists = torch.stack([
                ((X - m) ** 2).sum(dim=1) for m in means
            ], dim=1).min(dim=1).values
            probs = (dists / dists.sum()).clamp(min=0)
            idx   = torch.multinomial(probs, 1).item()
            means.append(X[idx])

        self.mu_ = torch.stack(means)  # (K, D)

        var = X.var(dim=0).mean().clamp(min=1e-6)
        self.sigma_ = torch.stack([
            torch.eye(D, device=self.device) * var
            for _ in range(self.K)
        ])
        self.pi_ = torch.ones(self.K, device=self.device) / self.K
        self._cache_inverses()

    def _cache_inverses(self):
        D   = self.mu_.shape[1]
        reg = self.reg_covar * torch.eye(D, device=self.device)
        inv_list, det_list = [], []

        for k in range(self.K):
            S = self.sigma_[k] + reg
            try:
                L       = torch.linalg.cholesky(S)
                inv_L   = torch.linalg.inv(L)
                inv_S   = inv_L.T @ inv_L
                log_det = 2 * L.diagonal().log().sum()
            except Exception:
                diag    = S.diagonal().clamp(min=1e-6)
                inv_S   = torch.diag(1.0 / diag)
                log_det = diag.log().sum()
            inv_list.append(inv_S)
            det_list.append(log_det)

        self.inv_sig_ = torch.stack(inv_list)
        self.log_det_ = torch.stack(det_list)

    # ── E / M steps ───────────────────────────────────────────────

    def _log_prob(self, X):
        """Returns (N, K) log-likelihoods."""
        D    = X.shape[1]
        diff = X.unsqueeze(1) - self.mu_.unsqueeze(0)          # (N,K,D)
        tmp  = torch.einsum('nkd,kde->nke', diff, self.inv_sig_)
        maha = (tmp * diff).sum(dim=-1)                         # (N,K)
        return -0.5 * (D * np.log(2*np.pi)
                       + self.log_det_.unsqueeze(0) + maha)

    def _e_step(self, X):
        log_p    = self._log_prob(X)
        log_resp = log_p + self.pi_.log().unsqueeze(0)
        log_sum  = torch.logsumexp(log_resp, dim=1, keepdim=True)
        resp     = (log_resp - log_sum).exp()
        ll       = log_sum.mean().item()
        return resp, ll

    def _m_step(self, X, resp):
        N, D = X.shape
        Nk   = resp.sum(dim=0).clamp(min=1e-6)
        self.pi_ = Nk / N
        self.mu_ = (resp.T @ X) / Nk.unsqueeze(1)
        sigma_new = []
        for k in range(self.K):
            diff = X - self.mu_[k]
            w    = resp[:, k].unsqueeze(1)
            sigma_new.append((w * diff).T @ diff / Nk[k])
        self.sigma_ = torch.stack(sigma_new)
        self._cache_inverses()

    # ── Fit ───────────────────────────────────────────────────────

    def fit(self, X_np, verbose=True):
        X = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        self._init_params(X)
        prev_ll = -np.inf

        for i in range(self.n_iter):
            resp, ll = self._e_step(X)
            self._m_step(X, resp)
            delta = ll - prev_ll
            if verbose and i % 10 == 0:
                print(f"      iter {i:3d}: ll={ll:.4f}  Δ={delta:.6f}")
            if i > 5 and abs(delta) < self.tol:
                if verbose:
                    print(f"      Converged at iter {i}  ll={ll:.4f}")
                break
            prev_ll = ll

        self.final_ll_ = prev_ll
        return self

    def bic(self, X_np):
        """
        BIC = -2 * log-likelihood * N + n_params * log(N)
        Lower is better.
        n_params = K*(1 + D + D*(D+1)/2) - 1
        """
        X   = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        N, D= X.shape
        _, ll = self._e_step(X)
        # Total log-likelihood (not mean)
        total_ll = ll * N
        # Number of free parameters
        n_params = self.K * (1 + D + D*(D+1)//2) - 1
        return -2 * total_ll + n_params * np.log(N)

    # ── Predict ───────────────────────────────────────────────────

    def predict(self, X_np, batch_size=50_000):
        """Hard labels (N,) int8."""
        N      = len(X_np)
        labels = np.zeros(N, dtype=np.int8)
        for s in range(0, N, batch_size):
            e     = min(s + batch_size, N)
            X_b   = torch.tensor(X_np[s:e], dtype=torch.float32,
                                 device=self.device)
            log_p = self._log_prob(X_b) + self.pi_.log().unsqueeze(0)
            labels[s:e] = log_p.argmax(dim=1).cpu().numpy()
        return labels

    def predict_proba(self, X_np, batch_size=50_000):
        """Soft probabilities (N, K) float32."""
        N     = len(X_np)
        probs = np.zeros((N, self.K), dtype=np.float32)
        for s in range(0, N, batch_size):
            e        = min(s + batch_size, N)
            X_b      = torch.tensor(X_np[s:e], dtype=torch.float32,
                                    device=self.device)
            resp, _  = self._e_step(X_b)
            probs[s:e] = resp.cpu().numpy()
        return probs


# ─────────────────────────────────────────────────────────────────
# Auto-K via BIC
# ─────────────────────────────────────────────────────────────────

def select_k_bic(features_pca, k_min=2, k_max=8,
                 n_bic_sub=100_000, device='cuda', random_state=42):
    """
    Fit GMM for K = k_min..k_max on a small subsample.
    Return optimal K (min BIC) and BIC scores dict.

    Uses small subsample (100k) for speed — BIC selection
    does not need the full dataset, just enough to estimate
    the number of modes.
    """
    rng     = np.random.RandomState(random_state)
    N       = len(features_pca)
    n_sub   = min(n_bic_sub, N)
    idx_sub = rng.choice(N, n_sub, replace=False)
    X_sub   = features_pca[idx_sub]

    print(f"  Auto-K BIC: testing K={k_min}..{k_max} "
          f"on {n_sub:,} subsample...")

    bic_scores = {}
    best_k, best_bic = k_min, np.inf

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixtureGPU(
            n_components=k, n_iter=50, tol=1e-3,
            reg_covar=1e-4, random_state=random_state,
            device=device,
        )
        gmm.fit(X_sub, verbose=False)
        bic = gmm.bic(X_sub)
        bic_scores[k] = round(bic, 1)

        marker = " ← best" if bic < best_bic else ""
        print(f"    K={k}: BIC={bic:.1f}{marker}")

        if bic < best_bic:
            best_bic = bic
            best_k   = k

    print(f"  → Optimal K={best_k}  (BIC={best_bic:.1f})")
    return best_k, bic_scores


# ─────────────────────────────────────────────────────────────────
# Main entry point — drop-in for MiniBatchKMeans
# ─────────────────────────────────────────────────────────────────

def fit_predict_gmm(features_pca, k,
                    auto_k=False, k_min=2, k_max=8,
                    device='cuda', random_state=42,
                    n_fit_sub=500_000):
    """
    Fit GMM and predict labels for all voxels.

    Parameters
    ----------
    features_pca : (N, D) float32 numpy  PCA-reduced features
    k            : int    number of clusters (ignored if auto_k=True)
    auto_k       : bool   if True, select K automatically via BIC
    k_min/k_max  : int    search range for auto-K
    device       : str    'cuda' or 'cpu'
    random_state : int    seed
    n_fit_sub    : int    subsample size for final fit

    Returns
    -------
    labels : (N,) int8
    gmm    : fitted GaussianMixtureGPU object
    bic_scores : dict {k: bic} or None
    optimal_k  : int  (same as k if auto_k=False)
    """
    rng = np.random.RandomState(random_state)
    N   = len(features_pca)

    bic_scores = None

    # ── Step 1: Auto-K selection ──────────────────────────────────
    if auto_k:
        k, bic_scores = select_k_bic(
            features_pca, k_min=k_min, k_max=k_max,
            device=device, random_state=random_state,
        )

    # ── Step 2: Fit final GMM on larger subsample ─────────────────
    n_sub   = min(n_fit_sub, N)
    idx_sub = rng.choice(N, n_sub, replace=False)

    print(f"  Fitting final GMM K={k} on {n_sub:,} voxels...")
    gmm = GaussianMixtureGPU(
        n_components = k,
        n_iter       = 100,
        tol          = 1e-4,
        reg_covar    = 1e-4,
        random_state = random_state,
        device       = device,
    )
    gmm.fit(features_pca[idx_sub], verbose=True)

    # ── Step 3: Predict all voxels in batches ─────────────────────
    print(f"  Predicting labels for {N:,} voxels...")
    labels = gmm.predict(features_pca, batch_size=50_000)

    return labels.astype(np.int8), gmm, bic_scores, k


# ─────────────────────────────────────────────────────────────────
# Patient-adaptive cluster naming
# ─────────────────────────────────────────────────────────────────

# Absolute anchors — always true regardless of patient
_ABS_EMPHYSEMA_MAX  = -800   # below this = definite emphysema
_ABS_CONSOLIDATION_MIN = -200  # above this = definite consolidation

PATHOLOGY_COLORS = {
    "Emphysema":     "#4477AA",
    "GGO":           "#66CCEE",
    "Healthy Lung":  "#228833",
    "Fibrosis":      "#FF8800",
    "Consolidation": "#CC3311",
    "Unknown":       "#BBBBBB",
}

def name_clusters_adaptive(ct_hu, mask, label_vol, k):
    """
    Assign pathology names to clusters using patient-adaptive
    percentile-based HU boundaries instead of fixed thresholds.

    Strategy:
      1. Compute lung HU distribution percentiles for THIS patient
      2. Absolute anchors handle extreme cases
         (very low HU = always emphysema, very high = always consolidation)
      3. Middle range uses relative percentiles:
         - below p33 of lung HU → emphysema-like
         - p33 to p66           → GGO / transition
         - above p66            → healthier / fibrosis / consolidation
      4. Final disambiguation uses texture rank within cluster
         (diffusion features encode texture — higher rank = more fibrosis-like)

    Returns dict: {cluster_id: {'name', 'mean_hu', 'n_voxels', 'percentile'}}
    """
    lung_hu = ct_hu[(mask >= 0.5) & (label_vol >= 0)]

    # Patient-specific statistics
    p10 = float(np.percentile(lung_hu, 10))
    p25 = float(np.percentile(lung_hu, 25))
    p50 = float(np.percentile(lung_hu, 50))   # median
    p75 = float(np.percentile(lung_hu, 75))
    p90 = float(np.percentile(lung_hu, 90))

    print(f"  Patient HU stats: "
          f"p10={p10:.0f} p25={p25:.0f} p50={p50:.0f} "
          f"p75={p75:.0f} p90={p90:.0f}")

    # Collect per-cluster stats
    cluster_stats = {}
    for c in range(k):
        vox = ct_hu[(label_vol == c) & (mask >= 0.5)]
        if len(vox) == 0:
            cluster_stats[c] = {
                'median_hu': p50, 'mean_hu': p50,
                'n_voxels': 0, 'percentile': 50.0
            }
            continue
        median_hu  = float(np.median(vox))
        mean_hu    = float(np.mean(vox))
        # Where does this cluster sit in the patient's HU distribution?
        percentile = float(np.mean(lung_hu <= median_hu) * 100)

        cluster_stats[c] = {
            'median_hu':  median_hu,
            'mean_hu':    mean_hu,
            'n_voxels':   int(len(vox)),
            'percentile': percentile,
        }

    # ── Assign names using adaptive + absolute rules ──────────────
    cluster_info = {}
    for c, stats in cluster_stats.items():
        hu   = stats['median_hu']
        pct  = stats['percentile']  # percentile within THIS patient

        # Absolute anchors first
        if hu < _ABS_EMPHYSEMA_MAX:        # below -800 → always emphysema
            name = "Emphysema"
        elif hu >= _ABS_CONSOLIDATION_MIN: # above -200 → always consolidation
            name = "Consolidation"

        # Middle range — adaptive percentile-based
        elif pct <= 25:
            # Bottom 25% of this patient's HU range
            # For BPCO: still relatively emphysematous
            # For ILD:  could be GGO
            if p50 < -700:          # BPCO-like patient (median very low)
                name = "Emphysema"
            else:
                name = "GGO"

        elif pct <= 50:
            # Between p25 and median
            if p50 < -700:
                name = "GGO"        # for BPCO: GGO transition zone
            else:
                name = "GGO"        # for ILD: GGO

        elif pct <= 75:
            # Between median and p75 — relatively healthier tissue
            if p75 > -500:          # ILD patient — high HU tissue
                name = "Fibrosis"
            else:
                name = "Healthy Lung"

        else:
            # Top 25% — highest HU in this patient
            if hu > -300:
                name = "Consolidation"
            elif hu > -500:
                name = "Fibrosis"
            else:
                name = "Healthy Lung"

        cluster_info[c] = {
            'name':       name,
            'mean_hu':    round(stats['mean_hu'], 1),
            'median_hu':  round(stats['median_hu'], 1),
            'n_voxels':   stats['n_voxels'],
            'percentile': round(stats['percentile'], 1),
        }

    return cluster_info