"""
BioRaman core unit tests.

Run with:   pytest -q
These cover the pure, GUI-independent functions: file-cube reconstruction,
preprocessing recipes, processed-data export, Otsu thresholding, component
analysis (DCLS/NNLS), particle statistics and library-style correlation.
"""
import os
import sys
import importlib.util

import numpy as np
import pytest

# ── import bioraman.py as a module without launching the GUI ────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def br():
    spec = importlib.util.spec_from_file_location(
        "bioraman", os.path.join(ROOT, "bioraman.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # GUI only starts under __main__
    return mod


# ── cube reconstruction ─────────────────────────────────────────────────────
def test_to_cube_shapes(br):
    W = 50
    assert br._to_cube(np.zeros(W), n_points=W).shape == (1, 1, W)
    assert br._to_cube(np.zeros((30, W)), n_points=W).shape == (1, 30, W)
    # flattened N*W with a known map shape
    assert br._to_cube(np.zeros(12 * W), map_shape=(3, 4), n_points=W).shape \
        == (3, 4, W)
    assert br._to_cube(np.zeros((5, 5, W))).shape == (5, 5, W)


def test_cube_from_positions_averages_duplicates(br):
    # 4x4 grid, 3 repeats each -> averaged to a single (4,4,W) cube
    W = 8
    xs = np.arange(4); ys = np.arange(4)
    X, Y = [], []
    for _ in range(3):
        for y in ys:
            for x in xs:
                X.append(x); Y.append(y)
    X = np.array(X); Y = np.array(Y)
    S = np.random.rand(X.size, W)
    cube = br._cube_from_positions(S, X, Y)
    assert cube.shape == (4, 4, W)


def test_volume_from_positions(br):
    W = 6; n = 5
    xs = np.arange(n); ys = np.arange(n); zs = np.arange(n)
    X, Y, Z = [], [], []
    for z in zs:
        for y in ys:
            for x in xs:
                X.append(x); Y.append(y); Z.append(z)
    S = np.random.rand(len(X), W)
    vol, zv, yv, xv = br._volume_from_positions(
        S, np.array(X), np.array(Y), np.array(Z))
    assert vol.shape == (n, n, n, W)
    assert zv.size == n


# ── recipes & export ────────────────────────────────────────────────────────
def test_recipe_round_trip(br, tmp_path):
    p = br.PreprocessParams(asls_lam=3e4, normalisation="snv", sg_window=15)
    f = tmp_path / "r.json"
    br.save_recipe_file(str(f), p)
    q = br.load_recipe_file(str(f))
    assert q.asls_lam == 3e4 and q.normalisation == "snv" and q.sg_window == 15


def test_write_cube_npz_roundtrip(br, tmp_path):
    cube = np.random.rand(4, 5, 20); x = np.linspace(400, 1800, 20)
    f = tmp_path / "c.npz"
    br.write_cube(str(f), cube, x, {"a": 1}, source="t")
    z = np.load(f, allow_pickle=True)
    assert np.allclose(z["spectra"], cube) and np.allclose(z["xdata"], x)


def test_write_cube_long_format(br, tmp_path):
    cube = np.random.rand(3, 3, 10); x = np.linspace(500, 1500, 10)
    f = tmp_path / "c.csv"
    br.write_cube(str(f), cube, x, {}, source="t")
    rows = [ln.split(",") for ln in f.read_text().splitlines()[1:]]
    arr = np.array(rows, dtype=float)
    assert set(arr[:, 0].astype(int)) == {0, 1, 2}      # X column
    assert arr.shape[0] == 3 * 3 * 10


# ── Otsu ────────────────────────────────────────────────────────────────────
def test_otsu_separates_bimodal(br):
    rng = np.random.default_rng(0)          # deterministic
    img = np.concatenate([rng.normal(0.2, 0.03, 1000),
                          rng.normal(0.8, 0.03, 1000)]).reshape(40, 50)
    thr = br._otsu_threshold(img)
    assert 0.3 < thr < 0.7


# ── component analysis (DCLS / NNLS) ────────────────────────────────────────
def _synthetic(refs, weights):
    """Build spectra = weights @ refs with a little noise."""
    return weights @ refs + 0.001 * np.random.rand(weights.shape[0], refs.shape[1])


def test_component_fit_dcls_recovers(br):
    x = np.linspace(400, 3000, 200)

    def pk(c, w):
        return np.exp(-((x - c) ** 2) / (2 * w ** 2))
    refs = np.stack([pk(1000, 25), pk(1600, 20), pk(2900, 40)])
    W_true = np.random.rand(120, 3)
    flat = _synthetic(refs, W_true)
    out = br.component_fit(flat, refs, method="DCLS", preprocess="Spectrum",
                           normalise="None", background_order=0)
    # raw coefficients should match the known weights closely
    assert np.allclose(out["raw"], W_true, atol=1e-2)
    assert abs(out["overall"].sum() - 100.0) < 1e-6


def test_component_fit_nnls_nonnegative(br):
    x = np.linspace(400, 3000, 150)
    refs = np.stack([np.exp(-((x - 1000) ** 2) / 800),
                     np.exp(-((x - 2900) ** 2) / 1500)])
    flat = _synthetic(refs, np.random.rand(60, 2))
    out = br.component_fit(flat, refs, method="NNLS")
    assert (out["conc"] >= 0).all()
    assert abs(out["overall"].sum() - 100.0) < 1e-6


def test_component_fit_dominant_matches_truth(br):
    x = np.linspace(400, 3000, 180)
    refs = np.stack([np.exp(-((x - 1000) ** 2) / 800),
                     np.exp(-((x - 2900) ** 2) / 1500)])
    # left half pure comp0, right half pure comp1
    W = np.zeros((100, 2))
    W[:50, 0] = 1.0 + np.random.rand(50)
    W[50:, 1] = 1.0 + np.random.rand(50)
    flat = _synthetic(refs, W)
    out = br.component_fit(flat, refs, method="NNLS")
    dom = out["conc"].argmax(1)
    truth = np.r_[np.zeros(50), np.ones(50)]
    assert (dom == truth).mean() > 0.98


# ── particle statistics ─────────────────────────────────────────────────────
def test_particle_stats_counts(br):
    img = np.zeros((50, 50))
    img[5:10, 5:10] = 1.0        # particle 1  (5x5)
    img[20:28, 20:28] = 1.0      # particle 2  (8x8)
    img[40:45, 40:45] = 1.0      # particle 3  (5x5)
    res = br.particle_stats(img, auto=False, threshold_pct=50,
                            remove_edge=True, min_size_pct=1, px_um=1.0)
    assert res["n"] == 3
    areas = sorted(p["area_px"] for p in res["props"])
    assert areas == [25, 25, 64]
    # ECD of the 64-px square ~ 2*sqrt(64/pi) ≈ 9.03 µm
    big = max(res["props"], key=lambda p: p["area_px"])
    assert abs(big["ecd_um"] - 2 * np.sqrt(64 / np.pi)) < 1e-6


def test_particle_stats_edge_removal(br):
    img = np.zeros((30, 30))
    img[0:5, 0:5] = 1.0          # touches the edge -> removed
    img[15:20, 15:20] = 1.0      # interior -> kept
    res = br.particle_stats(img, auto=False, threshold_pct=50,
                            remove_edge=True, min_size_pct=1)
    assert res["n"] == 1


# ── library-style correlation ───────────────────────────────────────────────
def test_correlation_ranks_true_match(br):
    x = np.linspace(400, 3000, 200)

    def pk(c, w):
        return np.exp(-((x - c) ** 2) / (2 * w ** 2))
    lib = {"A": pk(1000, 25) + pk(1600, 18),
           "B": pk(2900, 40), "C": pk(700, 25) + pk(1300, 18)}
    query = lib["A"] + 0.02 * np.random.rand(200)

    def prep(v):
        v = (v - v.mean()) / (v.std() + 1e-9)
        return v
    q = prep(query)
    scores = {k: float(np.corrcoef(q, prep(v))[0, 1]) for k, v in lib.items()}
    best = max(scores, key=scores.get)
    assert best == "A" and scores["A"] > 0.95
