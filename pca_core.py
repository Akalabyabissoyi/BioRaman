"""
pca_core.py — scalable, publication-grade PCA engine for spectroscopy.

A pure (no-GUI) analysis core used by PCA Studio (pca_studio.py) and usable
directly from scripts / notebooks / the BioRaman headless pipeline.

Design goals
------------
* **Scales either way.** Exact randomized-SVD PCA for modest N; out-of-core
  `IncrementalPCA` (batched `partial_fit`) above a size threshold so maps with
  10^5–10^6 spectra never need to sit in memory as one float64 block.
* **Modern chemometric diagnostics.** Hotelling's T^2 *and* Q-residual (SPE)
  with proper confidence limits (T^2 via the F-distribution; Q via the
  Jackson–Mudholkar approximation). These two together are the field standard
  for spotting model outliers vs. moderate-but-unusual samples.
* **Robust & sparse variants.** Iteratively-reweighted robust PCA (outlier
  resistant centring/scaling) and MiniBatch Sparse PCA (zeroed loadings →
  interpretable components).
* **Memory discipline.** float32 option, single `vstack`, batched transform /
  reconstruction.

References (method provenance)
------------------------------
* Jackson & Mudholkar (1979) Q-statistic control limit.
* Halko, Martinsson & Tropp (2011) randomized SVD.
* Zou, Hastie & Tibshirani (2006) sparse PCA.
* Bro & Smilde (2014) "Principal component analysis" (scaling/centring guidance).

MIT License — part of the BioRaman project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    from sklearn.decomposition import PCA, IncrementalPCA, MiniBatchSparsePCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:                                              # pragma: no cover
    HAS_SKLEARN = False

try:
    from scipy.stats import f as _f_dist, norm as _norm
    HAS_SCIPY = True
except Exception:                                              # pragma: no cover
    HAS_SCIPY = False


# ───────────────────────────── scaling / centring ──────────────────────────
SCALINGS = ("none", "autoscale", "pareto", "snv", "vector")


def _row_snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate — per-spectrum centre & unit-variance.
    Removes multiplicative scatter (a classic vibrational-spectroscopy step)."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _row_vector(X: np.ndarray) -> np.ndarray:
    """L2 (unit-vector) normalisation per spectrum."""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm


def apply_scaling(X: np.ndarray, scaling: str):
    """Return (X_scaled, center, scale) for column scalings, or row-transformed
    X for SNV/vector (center/scale returned as None). PCA centring is applied
    on top of this by the fitter."""
    if scaling == "snv":
        return _row_snv(X), None, None
    if scaling == "vector":
        return _row_vector(X), None, None
    if scaling == "autoscale":
        ctr = X.mean(axis=0)
        scl = X.std(axis=0); scl[scl == 0] = 1.0
        return (X - ctr) / scl, ctr, scl
    if scaling == "pareto":
        ctr = X.mean(axis=0)
        scl = np.sqrt(X.std(axis=0)); scl[scl == 0] = 1.0
        return (X - ctr) / scl, ctr, scl
    # "none" — PCA will mean-centre internally
    return X.astype(X.dtype, copy=False), None, None


# ───────────────────────────────── result ──────────────────────────────────
@dataclass
class PCAResult:
    scores: np.ndarray                     # (n, A)
    loadings: np.ndarray                   # (A, n_features)
    explained_variance_ratio: np.ndarray   # (A,)
    explained_variance: np.ndarray         # (A,) eigenvalues retained
    mean: np.ndarray                       # (n_features,) centre used by PCA
    n_components: int
    method: str                            # 'exact' | 'incremental' | 'robust' | 'sparse'
    scaling: str
    t2: np.ndarray = field(default=None)            # (n,) Hotelling T^2
    q: np.ndarray = field(default=None)             # (n,) Q-residual (SPE)
    t2_limit: float = field(default=None)           # 95% control limit
    q_limit: float = field(default=None)
    eig_tail: np.ndarray = field(default=None)      # eigenvalues used for Q limit
    n_samples: int = field(default=0)
    weights: np.ndarray = field(default=None)       # robust sample weights (or None)

    @property
    def cumulative_variance(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)


# ─────────────────────────────── solver choice ─────────────────────────────
def choose_method(n_samples: int, n_features: int,
                  user: str = "auto",
                  max_in_memory_elems: int = 60_000_000) -> str:
    """Pick 'exact' (randomized SVD) vs 'incremental' (out-of-core)."""
    if user in ("exact", "incremental"):
        return user
    big = (n_samples * n_features > max_in_memory_elems) or (n_samples > 200_000)
    return "incremental" if big else "exact"


# ───────────────────────────────── core fit ────────────────────────────────
def fit_pca(X: np.ndarray,
            n_components: int = 5,
            scaling: str = "none",
            method: str = "auto",
            *,
            dtype=np.float32,
            batch_size: int = 4096,
            diagnostics: bool = True,
            tail_components: int = 50) -> PCAResult:
    """Fit PCA with diagnostics.

    Parameters
    ----------
    X : (n_samples, n_features) array of spectra (rows = spectra).
    scaling : one of SCALINGS.
    method : 'auto' | 'exact' | 'incremental'.
    dtype : float32 (default) halves memory vs float64 with negligible accuracy
            loss for scores/loadings.
    tail_components : how many extra eigenvalues to estimate for the Q-limit.
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required: pip install scikit-learn")
    X = np.ascontiguousarray(X, dtype=dtype)
    # sanitise non-finite samples (cannot poison the decomposition)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    n, p = X.shape
    Xs, _, _ = apply_scaling(X, scaling)
    Xs = np.ascontiguousarray(Xs, dtype=dtype)

    A = int(max(1, min(n_components, n - 1, p)))
    # estimate a few extra components for the Q-residual tail
    k_tail = int(min(max(A, tail_components), n - 1, p))
    chosen = choose_method(n, p, method)

    if chosen == "incremental":
        ipca = IncrementalPCA(n_components=k_tail, batch_size=max(batch_size, k_tail + 1))
        for i in range(0, n, batch_size):
            ipca.partial_fit(Xs[i:i + batch_size])
        scores_full = np.empty((n, k_tail), dtype=dtype)
        for i in range(0, n, batch_size):
            scores_full[i:i + batch_size] = ipca.transform(Xs[i:i + batch_size])
        comps_full = ipca.components_
        evr_full   = ipca.explained_variance_ratio_
        ev_full    = ipca.explained_variance_
        mean       = ipca.mean_
    else:
        pca = PCA(n_components=k_tail, svd_solver="randomized", random_state=42)
        scores_full = pca.fit_transform(Xs).astype(dtype)
        comps_full  = pca.components_
        evr_full    = pca.explained_variance_ratio_
        ev_full     = pca.explained_variance_
        mean        = pca.mean_

    res = PCAResult(
        scores=scores_full[:, :A].copy(),
        loadings=comps_full[:A].copy(),
        explained_variance_ratio=evr_full[:A].copy(),
        explained_variance=ev_full[:A].copy(),
        mean=mean, n_components=A, method=chosen, scaling=scaling,
        n_samples=n,
    )

    if diagnostics:
        _attach_diagnostics(res, Xs, scores_full, comps_full, ev_full, mean,
                            A, batch_size)
    return res


def _attach_diagnostics(res, Xs, scores_full, comps_full, ev_full, mean,
                        A, batch_size):
    """Hotelling T^2 and Q-residual (SPE) with 95% control limits."""
    n = Xs.shape[0]
    ev_A = ev_full[:A].copy()
    ev_A[ev_A <= 0] = np.finfo(float).eps
    sc_A = scores_full[:, :A]

    # Hotelling T^2 = sum_a score_a^2 / lambda_a
    t2 = np.sum(sc_A ** 2 / ev_A, axis=1)

    # Q-residual: ||x - x_hat||^2 with x_hat reconstructed from A comps.
    q = np.empty(n, dtype=float)
    PA = comps_full[:A]                                  # (A, p)
    for i in range(0, n, batch_size):
        xb = Xs[i:i + batch_size] - mean
        recon = sc_A[i:i + batch_size] @ PA
        q[i:i + batch_size] = np.sum((xb - recon) ** 2, axis=1)

    res.t2, res.q = t2, q
    res.eig_tail = ev_full[A:]
    if HAS_SCIPY:
        res.t2_limit = _t2_limit(A, n, 0.95)
        res.q_limit  = _q_limit(ev_full[A:], 0.95)


def _t2_limit(A: int, n: int, conf: float = 0.95) -> float:
    """Hotelling T^2 control limit via the F-distribution."""
    if n - A <= 0:
        return float("nan")
    Fcrit = _f_dist.ppf(conf, A, n - A)
    return A * (n - 1) / (n - A) * Fcrit


def _q_limit(tail_eigs: np.ndarray, conf: float = 0.95) -> float:
    """Q-residual (SPE) limit — Jackson & Mudholkar (1979)."""
    tail = np.asarray(tail_eigs, dtype=float)
    tail = tail[tail > 0]
    if tail.size == 0 or not HAS_SCIPY:
        return float("nan")
    th1 = tail.sum()
    th2 = np.sum(tail ** 2)
    th3 = np.sum(tail ** 3)
    if th2 <= 0:
        return float("nan")
    h0 = 1.0 - (2.0 * th1 * th3) / (3.0 * th2 ** 2)
    if abs(h0) < 1e-6:
        h0 = 1e-6
    ca = _norm.ppf(conf)
    term = (ca * np.sqrt(2.0 * th2 * h0 ** 2) / th1
            + 1.0
            + th2 * h0 * (h0 - 1.0) / th1 ** 2)
    return float(th1 * term ** (1.0 / h0))


# ─────────────────────────────── robust PCA ────────────────────────────────
def fit_robust_pca(X: np.ndarray, n_components: int = 5, scaling: str = "none",
                   *, dtype=np.float32, n_iter: int = 5, c_huber: float = 2.5,
                   batch_size: int = 4096, **kw) -> PCAResult:
    """Iteratively-reweighted robust PCA.

    Down-weights samples with large Hotelling T^2 (Huber weights) and refits on
    a weighted, median-centred matrix. Outlier-resistant centre/scale → a
    decomposition that is not dragged toward a handful of anomalous spectra.
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required")
    X = np.ascontiguousarray(X, dtype=dtype)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X)
    Xs, _, _ = apply_scaling(X, scaling)
    n, p = Xs.shape
    A = int(max(1, min(n_components, n - 1, p)))

    w = np.ones(n, dtype=float)
    res = None
    for _ in range(max(1, n_iter)):
        wsum = w.sum()
        mean = (w[:, None] * Xs).sum(axis=0) / wsum
        Xc = (Xs - mean) * np.sqrt(w)[:, None]
        # weighted SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:A]
        ev = (S[:A] ** 2) / max(wsum - 1.0, 1.0)
        sc = (Xs - mean) @ comps.T
        ev_safe = ev.copy(); ev_safe[ev_safe <= 0] = np.finfo(float).eps
        t2 = np.sum(sc ** 2 / ev_safe, axis=1)
        # Huber weights from the robust scale of T^2
        med = np.median(t2)
        mad = np.median(np.abs(t2 - med)) * 1.4826 + 1e-12
        z = np.abs(t2 - med) / mad
        w = np.where(z <= c_huber, 1.0, c_huber / np.maximum(z, 1e-12))

    # final full eigenvalue tail (for Q limit) from the weighted decomposition
    full = np.linalg.svd((Xs - mean), full_matrices=False, compute_uv=False)
    ev_full = (full ** 2) / max(n - 1.0, 1.0)
    res = PCAResult(
        scores=sc[:, :A].copy(), loadings=comps.copy(),
        explained_variance_ratio=(ev / ev_full.sum())[:A],
        explained_variance=ev[:A].copy(), mean=mean, n_components=A,
        method="robust", scaling=scaling, n_samples=n, weights=w,
    )
    # diagnostics using the robust model
    scores_full = (Xs - mean) @ Vt.T
    _attach_diagnostics(res, Xs, scores_full, Vt, ev_full, mean, A, batch_size)
    return res


# ─────────────────────────────── sparse PCA ────────────────────────────────
def fit_sparse_pca(X: np.ndarray, n_components: int = 5, scaling: str = "none",
                   *, alpha: float = 1.0, dtype=np.float32, batch_size: int = 4096,
                   **kw) -> PCAResult:
    """MiniBatch Sparse PCA — loadings driven to exact zeros for interpretability.
    Larger `alpha` → sparser (and more biased) loadings."""
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required")
    X = np.ascontiguousarray(X, dtype=dtype)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X)
    Xs, _, _ = apply_scaling(X, scaling)
    mean = Xs.mean(axis=0)
    A = int(max(1, min(n_components, Xs.shape[0] - 1, Xs.shape[1])))
    spca = MiniBatchSparsePCA(n_components=A, alpha=alpha, random_state=42,
                              batch_size=max(batch_size, A + 1))
    scores = spca.fit_transform(Xs - mean)
    comps = spca.components_
    # pseudo explained-variance for sparse comps (not orthogonal): variance of scores
    var = scores.var(axis=0)
    total = (Xs - mean).var(axis=0).sum()
    res = PCAResult(
        scores=scores.astype(dtype), loadings=comps,
        explained_variance_ratio=var / (total + 1e-12),
        explained_variance=var, mean=mean, n_components=A,
        method="sparse", scaling=scaling, n_samples=Xs.shape[0],
    )
    return res


# ──────────────────── high-level dispatcher ─────────────────────────────────
def run_pca(X, n_components=5, scaling="none", variant="standard",
            method="auto", alpha=1.0, **kw) -> PCAResult:
    """One entry point. variant ∈ {'standard','robust','sparse'}."""
    if variant == "robust":
        return fit_robust_pca(X, n_components, scaling, **kw)
    if variant == "sparse":
        return fit_sparse_pca(X, n_components, scaling, alpha=alpha, **kw)
    return fit_pca(X, n_components, scaling, method=method, **kw)


# ───────────────────── subset / residual PCA (Schuppert Fig. 4) ─────────────
def subset_pca(X_subset, n_components=2, scaling="none",
               deflate_loadings=None, deflate_mean=None, n_deflate=0,
               dtype=np.float32):
    """PCA on a chosen subset of spectra, optionally on the *residual* after
    projecting out the first `n_deflate` global components.

    Reproduces the "residual subset PCA" inset of Schuppert (2016): removing the
    dominant global axes from a subset can expose sub-structure the global model
    hides. With `n_deflate == 0` it is a plain subset PCA.
    """
    if not HAS_SKLEARN:
        raise RuntimeError("scikit-learn required")
    X = np.ascontiguousarray(X_subset, dtype=dtype)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X)
    Xs, _, _ = apply_scaling(X, scaling)

    if n_deflate and deflate_loadings is not None and deflate_mean is not None:
        P = np.asarray(deflate_loadings[:n_deflate], dtype=dtype)      # (k, p)
        Xc = Xs - deflate_mean
        Xs = Xc - (Xc @ P.T) @ P                  # residual (orthogonal complement)
    A = int(max(1, min(n_components, Xs.shape[0] - 1, Xs.shape[1])))
    pca = PCA(n_components=A, svd_solver="randomized", random_state=42)
    scores = pca.fit_transform(Xs).astype(dtype)
    return PCAResult(
        scores=scores, loadings=pca.components_[:A],
        explained_variance_ratio=pca.explained_variance_ratio_[:A],
        explained_variance=pca.explained_variance_[:A], mean=pca.mean_,
        n_components=A, method=("residual-subset" if n_deflate else "subset"),
        scaling=scaling, n_samples=Xs.shape[0])


# ─────────────────── scree / component-number guidance ──────────────────────
def suggest_n_components(evr_full: np.ndarray, eig_full: np.ndarray,
                         scaling: str) -> dict:
    """Heuristic guidance: 95% cumulative variance, Kaiser (eig>mean) for
    autoscaled data, and the elbow of the scree curve."""
    cum = np.cumsum(evr_full)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    kaiser = int(np.sum(eig_full > eig_full.mean())) if scaling == "autoscale" else None
    # elbow via max distance to the chord of the scree curve
    y = evr_full
    x = np.arange(1, len(y) + 1)
    if len(y) >= 3:
        x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
        d = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) \
            / np.hypot(y2 - y1, x2 - x1)
        elbow = int(x[np.argmax(d)])
    else:
        elbow = len(y)
    return {"var95": n95, "kaiser": kaiser, "elbow": elbow}


# ───────────────────── supervised follow-on (PLS-DA / LDA) ──────────────────
def classify_scores(scores: np.ndarray, labels: Sequence,
                    method: str = "lda", groups: Optional[Sequence] = None,
                    n_splits: int = 5) -> dict:
    """Cross-validated classification on PCA scores.

    method : 'lda' or 'plsda'. Scaler is fit *inside* each fold (no leakage).
    groups : optional sample-group IDs → GroupKFold (avoids leaking replicate
             pixels from the same map across train/test).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import accuracy_score, confusion_matrix

    y = np.asarray(labels)
    le = LabelEncoder(); yi = le.fit_transform(y)
    classes = le.classes_
    X = np.asarray(scores)

    if groups is not None and len(set(groups)) >= 2:
        splitter = GroupKFold(n_splits=min(n_splits, len(set(groups))))
        split_iter = splitter.split(X, yi, groups)
    else:
        splitter = StratifiedKFold(n_splits=min(n_splits, np.bincount(yi).min()),
                                   shuffle=True, random_state=42)
        split_iter = splitter.split(X, yi)

    y_true_all, y_pred_all = [], []
    for tr, te in split_iter:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if method == "plsda":
            ncomp = min(X.shape[1], len(classes) if len(classes) > 2 else 2,
                        max(1, len(tr) - 1))
            from sklearn.preprocessing import label_binarize
            Ytr = label_binarize(yi[tr], classes=np.arange(len(classes)))
            if Ytr.shape[1] == 1:
                Ytr = np.hstack([1 - Ytr, Ytr])
            pls = PLSRegression(n_components=ncomp).fit(Xtr, Ytr)
            pred = pls.predict(Xte)
            yhat = pred.argmax(axis=1)
        else:
            lda = LinearDiscriminantAnalysis().fit(Xtr, yi[tr])
            yhat = lda.predict(Xte)
        y_true_all.extend(yi[te]); y_pred_all.extend(yhat)

    acc = accuracy_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all)
    return {"accuracy": acc, "confusion": cm, "classes": classes,
            "method": method, "grouped": groups is not None}


# ───────────────────────── clustering on scores ────────────────────────────
def cluster_scores(scores: np.ndarray, method: str = "kmeans",
                   k: int = 3, **kw) -> dict:
    """Cluster in PC space. method ∈ {'kmeans','agglomerative','hdbscan'}."""
    X = np.asarray(scores)
    if method == "hdbscan":
        try:
            import hdbscan
            cl = hdbscan.HDBSCAN(min_cluster_size=kw.get("min_cluster_size", 50))
            lab = cl.fit_predict(X)
        except Exception as e:
            raise RuntimeError(f"hdbscan not available ({e}); pip install hdbscan")
    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        lab = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    else:
        from sklearn.cluster import MiniBatchKMeans
        lab = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

    sil = None
    try:
        from sklearn.metrics import silhouette_score
        uniq = set(lab) - {-1}
        if len(uniq) >= 2:
            m = lab != -1
            sil = float(silhouette_score(X[m][:, :min(5, X.shape[1])], lab[m]))
    except Exception:
        pass
    return {"labels": lab, "silhouette": sil, "method": method}
