"""
BioRaman — Group Comparison & Group Mean-Spectra module
=======================================================
Adds two capabilities to BioRaman:

  1. Group Comparison  — per-cell custom band-ratio across multiple maps/files,
     grouped by condition, with inferential statistics (Mann–Whitney U,
     Hodges–Lehmann median difference + 95% CI, Cliff's delta) and a
     publication-style box plot  (Figure-3a style).

  2. Group Mean-Spectra Overlay — mean intracellular spectrum per group with
     ±SD envelope and shaded band-A / band-B windows  (Figure-3b style).

Works three ways:
  • Imported by bioraman.py and launched as a Tkinter window (see INTEGRATION).
  • Run standalone as a Tkinter app:   python bioraman_group_compare.py
  • Headless from the command line:     python bioraman_group_compare.py \
        --group PBS pbs-1.wdf pbs-2.wdf --group JK100 jk100-1.wdf jk100-2.wdf \
        --banda 3085 3165 --bandb 3350 3550 --out ./out

Preprocessing matches the BioRaman GUI defaults:
    cosmic-ray removal (modified Z-score on 1st derivative, thr 12, half-width 3)
    → per-spectrum minimum subtraction
    → AsLS baseline (lambda 1e5, p 0.001)
    → Savitzky–Golay smoothing (11-pt, order 3)
    → area normalisation.

Author: Akalabya Bissoyi  •  github.com/Akalabyabissoyi/BioRaman
"""
from __future__ import annotations
import os, json
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu

# ---- optional deps (match BioRaman) -----------------------------------------
# Optional local (git-ignored) presets file. Lets a user keep their own default
# band windows (e.g. cryo ice/liquid) without shipping assay-specific defaults in
# the public build. Format: {"default_preset": "<name>", "presets": {"<name>":[a0,a1,b0,b1]}}
_PRESET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".bioraman_presets.json")

def _load_presets():
    try:
        return json.load(open(_PRESET_FILE))
    except Exception:
        return {}

try:
    from pybaselines.whittaker import asls as _asls
except Exception:                      # pragma: no cover
    _asls = None
try:
    from renishawWiRE import WDFReader
    HAS_WDF = True
except Exception:                      # pragma: no cover
    HAS_WDF = False
try:
    from skimage.filters import threshold_otsu
    from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, label, center_of_mass
    HAS_SKI = True
except Exception:                      # pragma: no cover
    HAS_SKI = False

# ============================================================================ #
#  Preprocessing (BioRaman defaults)
# ============================================================================ #
def _cosmic_clean(y, thr=12.0, hw=3):
    d = np.diff(y, prepend=y[0]); med = np.median(d); mad = np.median(np.abs(d - med)) + 1e-9
    M = 0.6745 * (d - med) / mad; bad = np.abs(M) > thr
    if bad.any():
        idx = np.where(bad)[0]; mask = np.zeros_like(y, bool)
        for i in idx:
            mask[max(0, i - hw):min(len(y), i + hw + 1)] = True
        good = ~mask
        if good.sum() > 2:
            y = y.copy(); y[mask] = np.interp(np.where(mask)[0], np.where(good)[0], y[good])
    return y

def preprocess(spec, waves, asls_lam=1e5, asls_p=1e-3, sg_win=11, sg_ord=3,
               cr_thr=12.0, cr_hw=3):
    """Apply the BioRaman default pipeline to a (n_pix, n_wave) array."""
    out = np.empty_like(spec, float)
    for i in range(spec.shape[0]):
        y = _cosmic_clean(spec[i].astype(float), cr_thr, cr_hw)
        y = y - y.min()
        if _asls is not None:
            y = y - _asls(y, lam=asls_lam, p=asls_p, max_iter=30)[0]
        if sg_win and sg_win < len(y):
            y = savgol_filter(y, sg_win, sg_ord)
        a = np.trapezoid(np.clip(y, 0, None), waves) if hasattr(np, "trapezoid") \
            else np.trapz(np.clip(y, 0, None), waves)
        if a > 0:
            y = y / a
        out[i] = y
    return out

def _area(Y, w, lo, hi):
    m = (w >= lo) & (w <= hi)
    seg = np.clip(Y[:, m], 0, None)
    return (np.trapezoid(seg, w[m], axis=1) if hasattr(np, "trapezoid")
            else np.trapz(seg, w[m], axis=1))

# ============================================================================ #
#  I/O + per-cell ratio
# ============================================================================ #
def load_map(path):
    """Return (X[n_pix,n_wave], waves, (ny,nx)) from a .wdf file."""
    if not HAS_WDF:
        raise RuntimeError("renishawWiRE not installed — cannot read .wdf")
    r = WDFReader(path)
    w = np.asarray(r.xdata, float); sp = np.asarray(r.spectra, float)
    ny, nx, nw = sp.shape
    order = np.argsort(w)
    return sp.reshape(-1, nw)[:, order], w[order], (ny, nx)

def segment_cell(Yp, waves, shape, ch=(2850, 3000), min_frac=0.01,
                 method="otsu", percentile=85.0):
    """Segment cell pixels from the CH-stretch (lipid/protein) signal.

    method="otsu"        -> automatic Otsu threshold + morphological cleanup
    method="percentile"  -> keep pixels brighter than the given CH percentile
                            (manual control; use a higher value for a tighter ROI)
    Returns (mask_flat, mask2d, chmap2d).
    """
    ny, nx = shape
    chmap = _area(Yp, waves, *ch).reshape(ny, nx)
    chs = gaussian_filter(chmap, 1.0) if HAS_SKI else chmap
    if method == "percentile" or not HAS_SKI:
        mask = chs > np.percentile(chs, percentile)
    else:                                   # Otsu (default)
        mask = chs > threshold_otsu(chs)
        mask = binary_closing(binary_opening(mask, np.ones((2, 2))), np.ones((3, 3)))
        lab, n = label(mask)
        if n:
            keep = [k + 1 for k in range(n) if (lab == k + 1).sum() >= max(15, min_frac * mask.size)]
            mask = np.isin(lab, keep)
        if mask.sum() < 10:
            mask = chs > np.percentile(chs, percentile)
    return mask.reshape(-1), mask, chmap

def cell_ratio(path, band_a, band_b, seg=None, **pp):
    """Mean intracellular band_a/band_b ratio for one map.

    seg : optional dict passed to segment_cell, e.g.
          {"method":"percentile","percentile":90,"ch":(2850,3000)}
    Returns (mean_ratio, waves, roi_mean_spectrum, roi_spectra, mask2d, chmap2d, abmap2d).
    """
    seg = dict(seg or {})
    X, w, shape = load_map(path)
    Yp = preprocess(X, w, **pp)
    mask_flat, mask2d, chmap2d = segment_cell(Yp, w, shape, **seg)
    roi = Yp[mask_flat]
    ab = _area(roi, w, *band_a) / (_area(roi, w, *band_b) + 1e-12)
    abmap2d = (_area(Yp, w, *band_a) / (_area(Yp, w, *band_b) + 1e-12)).reshape(shape)
    return float(np.mean(ab)), w, roi.mean(0), roi, mask2d, chmap2d, abmap2d


def _center_crop(img, mask, S):
    """Crop img (and matching mask) to an S x S window centred on the mask centroid."""
    try:
        from scipy.ndimage import center_of_mass
        ny, nx = mask.shape
        cy, cx = center_of_mass(mask) if mask.any() else (ny / 2, nx / 2)
    except Exception:
        ny, nx = mask.shape; cy, cx = ny / 2, nx / 2
    r0 = max(0, min(int(round(cy - S / 2)), img.shape[0] - S))
    c0 = max(0, min(int(round(cx - S / 2)), img.shape[1] - S))
    return img[r0:r0 + S, c0:c0 + S], mask[r0:r0 + S, c0:c0 + S]

# ============================================================================ #
#  Statistics
# ============================================================================ #
def cliffs_delta(x, y):
    x = np.asarray(x)[:, None]; y = np.asarray(y)[None, :]
    return float((np.sum(x > y) - np.sum(x < y)) / (x.shape[0] * y.shape[1]))

def hodges_lehmann(x, y):
    d = np.subtract.outer(np.asarray(x), np.asarray(y)).ravel()
    return float(np.median(d)), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))

def compare_two_groups(vals_a, vals_b):
    U, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    hl, ci = hodges_lehmann(vals_a, vals_b)
    return dict(n_a=len(vals_a), n_b=len(vals_b),
                mean_a=float(np.mean(vals_a)), sd_a=float(np.std(vals_a, ddof=1)) if len(vals_a) > 1 else 0.0,
                mean_b=float(np.mean(vals_b)), sd_b=float(np.std(vals_b, ddof=1)) if len(vals_b) > 1 else 0.0,
                U=float(U), p=float(p),
                hodges_lehmann=hl, hl_ci=list(ci), cliffs_delta=cliffs_delta(vals_a, vals_b))

# ============================================================================ #
#  Figures
# ============================================================================ #
_PALETTE = ["#d1495b", "#3a7ca5", "#66a182", "#e3a72f", "#8d5a97", "#476c9b"]

def figure_boxplot(groups_vals, ylabel="Mean intracellular band A/B", title=None):
    """groups_vals: dict{label -> list of per-cell values}. Returns matplotlib Figure."""
    import matplotlib.pyplot as plt
    labels = list(groups_vals); data = [groups_vals[k] for k in labels]
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(labels), 4.6))
    bp = ax.boxplot(data, widths=0.55, patch_artist=True,
                    positions=range(1, len(labels) + 1))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)          # version-agnostic (no tick_labels/labels kwarg)
    for patch, c in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(c); patch.set_alpha(0.35)
    for med in bp["medians"]:
        med.set_color("#222"); med.set_linewidth(1.6)
    for i, (k, c) in enumerate(zip(labels, _PALETTE), 1):
        v = groups_vals[k]
        ax.scatter(np.full(len(v), i) + np.random.uniform(-0.07, 0.07, len(v)),
                   v, color=c, s=32, zorder=3, edgecolor="k", lw=0.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.5)
    ymax = max(max(v) for v in data); ax.set_ylim(0, ymax * 1.3)
    if len(labels) == 2:
        st = compare_two_groups(data[0], data[1])
        y = ymax * 1.12; ax.plot([1, 1, 2, 2], [y, y * 1.03, y * 1.03, y], c="k", lw=1.1)
        ax.text(1.5, y * 1.05, f"U={st['U']:.0f}, p={st['p']:.3f}", ha="center", fontsize=9)
        ax.text(1.5, y * 0.62,
                f"Δ(HL)={st['hodges_lehmann']:.2f}\n95%CI[{st['hl_ci'][0]:.2f},{st['hl_ci'][1]:.2f}]"
                f"\nCliff's δ={st['cliffs_delta']:.2f}", ha="center", fontsize=7.5, color="#333")
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    return fig

def figure_mean_spectra(groups_spectra, waves, band_a=None, band_b=None,
                        xlim=(2700, 3800), title=None):
    """groups_spectra: dict{label -> list of per-map mean ROI spectra (each len n_wave)}."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    if band_a: ax.axvspan(*band_a, color="#d1495b", alpha=0.10)
    if band_b: ax.axvspan(*band_b, color="#3a7ca5", alpha=0.10)
    m = (waves >= xlim[0]) & (waves <= xlim[1])
    for (k, specs), c in zip(groups_spectra.items(), _PALETTE):
        arr = np.asarray(specs); mu = arr.mean(0); sd = arr.std(0)
        ax.plot(waves[m], mu[m], color=c, lw=1.6, label=k)
        if arr.shape[0] > 1:
            ax.fill_between(waves[m], (mu - sd)[m], (mu + sd)[m], color=c, alpha=0.18)
    ax.set_xlabel("Raman shift (cm⁻¹)", fontsize=10)
    ax.set_ylabel("Mean intracellular intensity (a.u.)", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    return fig

# ============================================================================ #
#  High-level driver (headless)
# ============================================================================ #
def _prep(img, mask, smooth, crop_S):
    """Optionally centre-crop to crop_S and Gaussian-smooth an image for display."""
    if crop_S:
        img, mask = _center_crop(img, mask, crop_S)
    if smooth:
        try:
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img.astype(float), 0.7)
        except Exception:
            pass
    return img, mask

def figure_mask_preview(previews, title="Cell segmentation preview", smooth=True, equal_size=True):
    """previews: list of dicts {name, chmap, mask}. Grid of CH maps with ROI contour."""
    import matplotlib.pyplot as plt
    interp = "bilinear" if smooth else "nearest"
    S = min(min(p["mask"].shape) for p in previews) if (equal_size and previews) else None
    n = len(previews); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.6 * nrow), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, p in zip(axs.ravel(), previews):
        img, mask = _prep(p["chmap"], p["mask"], smooth, S)
        ax.imshow(img, cmap="magma", interpolation=interp, aspect="equal")
        ax.contour(mask, levels=[0.5], colors="w", linewidths=0.8)
        ax.set_title("%s  (%d px)" % (p["name"], int(p["mask"].sum())), fontsize=8)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=11, fontweight="bold"); fig.tight_layout()
    return fig

def figure_ab_maps(entries, vmin, vmax, band_a, band_b, smooth=True, equal_size=True,
                   title="Ice / liquid-water A/B maps (shared scale)"):
    """entries: list of dicts {name, group, abmap, mask}. Shared-scale jet maps + ROI."""
    import matplotlib.pyplot as plt
    interp = "bicubic" if smooth else "nearest"
    S = min(min(e["mask"].shape) for e in entries) if (equal_size and entries) else None
    n = len(entries); ncol = min(4, n); nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.7 * nrow), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    im = None
    for ax, e in zip(axs.ravel(), entries):
        img, mask = _prep(e["abmap"], e["mask"], smooth, S)
        im = ax.imshow(img, cmap="jet", vmin=vmin, vmax=vmax, interpolation=interp, aspect="equal")
        ax.contour(mask, levels=[0.5], colors="w", linewidths=0.8)
        ax.set_title("%s\n%s" % (e["name"], e["group"]), fontsize=8)
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    if im is not None:
        cb = fig.colorbar(im, ax=axs.ravel().tolist(), fraction=0.03, pad=0.02)
        cb.set_label("A/B  (A %d–%d / B %d–%d cm⁻¹)" % (band_a[0], band_a[1], band_b[0], band_b[1]),
                     fontsize=8)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    return fig

def run(groups_files, band_a=(3085, 3165), band_b=(3350, 3550), outdir=None, seg=None, **pp):
    """groups_files: dict{label -> [wdf paths]}. Computes ratios, stats, figures, CSV/JSON.
    seg: optional dict forwarded to segment_cell (method / percentile / ch)."""
    per_cell, group_vals, group_specs, waves = {}, {}, {}, None
    previews, ab_entries, pool = [], [], []
    for label, files in groups_files.items():
        vals, specs = [], []
        for f in files:
            ab, w, mean_spec, _roi, mask2d, chmap2d, abmap2d = cell_ratio(f, band_a, band_b, seg=seg, **pp)
            # maps can have slightly different wavenumber axes -> resample onto a
            # common reference grid (first map) before averaging/plotting
            if waves is None:
                waves = np.asarray(w, float)
            mean_spec = np.interp(waves, np.asarray(w, float), mean_spec)
            vals.append(ab); specs.append(mean_spec)
            name = os.path.basename(f)
            per_cell[name] = (label, ab)
            previews.append({"name": name, "group": label, "chmap": chmap2d, "mask": mask2d})
            ab_entries.append({"name": name, "group": label, "abmap": abmap2d, "mask": mask2d})
            pool.append(abmap2d[mask2d])
        group_vals[label] = vals; group_specs[label] = specs
    poolv = np.concatenate(pool) if pool else np.array([0, 1])
    vmin, vmax = float(np.percentile(poolv, 2)), float(np.percentile(poolv, 98))
    result = {"band_a": list(band_a), "band_b": list(band_b), "ab_vmin": vmin, "ab_vmax": vmax,
              "groups": {k: {"n": len(v), "mean": float(np.mean(v)),
                             "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                             "values": v} for k, v in group_vals.items()}}
    labs = list(group_vals)
    if len(labs) == 2:
        result["comparison"] = compare_two_groups(group_vals[labs[0]], group_vals[labs[1]])
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fb = figure_boxplot(group_vals, title="Per-cell band-ratio comparison")
        fb.savefig(os.path.join(outdir, "group_boxplot.png"), dpi=400)
        fb.savefig(os.path.join(outdir, "group_boxplot.pdf"))
        fs = figure_mean_spectra(group_specs, waves, band_a, band_b,
                                 title="Group mean intracellular spectra")
        fs.savefig(os.path.join(outdir, "group_mean_spectra.png"), dpi=400)
        fs.savefig(os.path.join(outdir, "group_mean_spectra.pdf"))
        fm = figure_ab_maps(ab_entries, vmin, vmax, band_a, band_b)
        fm.savefig(os.path.join(outdir, "group_AB_maps.png"), dpi=400)
        fm.savefig(os.path.join(outdir, "group_AB_maps.pdf"))
        with open(os.path.join(outdir, "per_cell_ratios.csv"), "w") as fh:
            fh.write("file,group,ratio\n")
            for f, (g, v) in per_cell.items():
                fh.write(f"{f},{g},{v:.6f}\n")
        json.dump(result, open(os.path.join(outdir, "group_comparison.json"), "w"), indent=2)
    return result, group_vals, group_specs, waves, previews, ab_entries

# ============================================================================ #
#  Tkinter window  (call open_group_compare(parent) from BioRaman)
# ============================================================================ #
def open_group_compare(parent=None):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    win = tk.Toplevel(parent) if parent is not None else tk.Tk()
    win.title("BioRaman — Group Comparison")
    win.geometry("1120x760")
    groups = {}     # label -> list[path]

    # scrollable left panel + pinned bottom action bar (so buttons never get cut off)
    left_outer = ttk.Frame(win, width=310); left_outer.pack(side="left", fill="y")
    left_outer.pack_propagate(False)
    actions = ttk.Frame(left_outer, padding=(8, 6)); actions.pack(side="bottom", fill="x")
    _lc = tk.Canvas(left_outer, highlightthickness=0)
    _sb = ttk.Scrollbar(left_outer, orient="vertical", command=_lc.yview)
    _sb.pack(side="right", fill="y"); _lc.pack(side="left", fill="both", expand=True)
    _lc.configure(yscrollcommand=_sb.set)
    left = ttk.Frame(_lc, padding=8)
    _win_id = _lc.create_window((0, 0), window=left, anchor="nw")
    left.bind("<Configure>", lambda e: _lc.configure(scrollregion=_lc.bbox("all")))
    _lc.bind("<Configure>", lambda e: _lc.itemconfig(_win_id, width=e.width))
    for _w in (_lc, left):
        _w.bind("<MouseWheel>", lambda e: _lc.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        _w.bind("<Button-4>", lambda e: _lc.yview_scroll(-1, "units"))
        _w.bind("<Button-5>", lambda e: _lc.yview_scroll(1, "units"))
    ttk.Label(left, text="Groups", font=("", 11, "bold")).pack(anchor="w")
    tree = ttk.Treeview(left, columns=("g",), show="tree", height=8); tree.pack(fill="both", expand=False)

    lbl = tk.StringVar(value="PBS")
    row = ttk.Frame(left); row.pack(fill="x", pady=4)
    ttk.Label(row, text="Label:").pack(side="left")
    ttk.Entry(row, textvariable=lbl, width=12).pack(side="left", padx=4)

    def add_files():
        fs = filedialog.askopenfilenames(title="Select .wdf maps for group '%s'" % lbl.get(),
                                         filetypes=[("Renishaw WDF", "*.wdf"), ("All", "*.*")])
        if not fs: return
        groups.setdefault(lbl.get(), [])
        node = None
        for n in tree.get_children():
            if tree.item(n, "text") == lbl.get(): node = n
        if node is None:
            node = tree.insert("", "end", text=lbl.get(), open=True)
        for f in fs:
            groups[lbl.get()].append(f); tree.insert(node, "end", text=os.path.basename(f))
    ttk.Button(left, text="＋ Add files to group", command=add_files).pack(fill="x", pady=2)

    def remove_selected():
        for iid in list(tree.selection()):
            if not tree.exists(iid):
                continue
            parent = tree.parent(iid)
            if parent == "":                       # a whole group node
                groups.pop(tree.item(iid, "text"), None)
                tree.delete(iid)
            else:                                   # a single file node
                lab = tree.item(parent, "text"); fn = tree.item(iid, "text")
                if lab in groups:
                    groups[lab] = [p for p in groups[lab] if os.path.basename(p) != fn]
                    if not groups[lab]:
                        groups.pop(lab); tree.delete(parent)
                    else:
                        tree.delete(iid)

    def clear_all():
        groups.clear()
        for n in tree.get_children():
            tree.delete(n)

    rowbtn = ttk.Frame(left); rowbtn.pack(fill="x", pady=2)
    ttk.Button(rowbtn, text="✖ Remove selected", command=remove_selected).pack(side="left", expand=True, fill="x", padx=(0, 2))
    ttk.Button(rowbtn, text="🗑 Clear all", command=clear_all).pack(side="left", expand=True, fill="x", padx=(2, 0))
    ttk.Label(left, text="Tip: set Label, then Add files — once per condition.",
              foreground="#666").pack(anchor="w", pady=(2, 0))

    # Bands: no assay-specific default. User picks a preset or types the windows.
    band = {"a0": tk.StringVar(), "a1": tk.StringVar(),
            "b0": tk.StringVar(), "b1": tk.StringVar()}
    PRESETS = {"Custom (enter values)": None,
               "Cryo — ice / liquid OH  (3085–3165 / 3350–3550)": (3085, 3165, 3350, 3550)}
    _pf = _load_presets()
    for _nm, _v in (_pf.get("presets") or {}).items():
        try:
            PRESETS[_nm] = tuple(float(x) for x in _v)
        except Exception:
            pass
    bf = ttk.LabelFrame(left, text="Bands (cm⁻¹)", padding=6); bf.pack(fill="x", pady=6)
    preset_var = tk.StringVar(value="Custom (enter values)")
    def apply_preset(*_):
        v = PRESETS.get(preset_var.get())
        if v:
            for k, val in zip(("a0", "a1", "b0", "b1"), v):
                band[k].set(str(val))
    pr = ttk.Frame(bf); pr.pack(fill="x", pady=(0, 3))
    ttk.Label(pr, text="Preset", width=6).pack(side="left")
    ttk.Combobox(pr, textvariable=preset_var, values=list(PRESETS), state="readonly",
                 width=24).pack(side="left")
    preset_var.trace_add("write", apply_preset)
    for t, k in [("A lo", "a0"), ("A hi", "a1"), ("B lo", "b0"), ("B hi", "b1")]:
        r = ttk.Frame(bf); r.pack(fill="x")
        ttk.Label(r, text=t, width=6).pack(side="left")
        ttk.Entry(r, textvariable=band[k], width=8).pack(side="left")
    # personal default (git-ignored file only): pre-select a preset / prefill bands
    if _pf.get("default_preset") in PRESETS:
        preset_var.set(_pf["default_preset"])
    elif _pf.get("default_bands"):
        for k, val in zip(("a0", "a1", "b0", "b1"), _pf["default_bands"]):
            band[k].set(str(val))

    def get_bands():
        try:
            v = [float(band[k].get()) for k in ("a0", "a1", "b0", "b1")]
        except ValueError:
            raise ValueError("Set the band windows (A lo/hi, B lo/hi) or choose a preset.")
        return (v[0], v[1]), (v[2], v[3])

    # ── Cell segmentation controls ────────────────────────────────────────
    seg_method = tk.StringVar(value="otsu")
    seg_pct = tk.DoubleVar(value=85.0)
    seg_ch0 = tk.DoubleVar(value=2850); seg_ch1 = tk.DoubleVar(value=3000)
    sf = ttk.LabelFrame(left, text="Cell segmentation", padding=6); sf.pack(fill="x", pady=6)
    ttk.Radiobutton(sf, text="Auto (Otsu)", variable=seg_method, value="otsu").pack(anchor="w")
    rowp = ttk.Frame(sf); rowp.pack(fill="x")
    ttk.Radiobutton(rowp, text="Manual percentile", variable=seg_method, value="percentile").pack(side="left")
    ttk.Entry(rowp, textvariable=seg_pct, width=6).pack(side="left", padx=4)
    rowc = ttk.Frame(sf); rowc.pack(fill="x", pady=(2, 0))
    ttk.Label(rowc, text="CH band").pack(side="left")
    ttk.Entry(rowc, textvariable=seg_ch0, width=7).pack(side="left", padx=2)
    ttk.Entry(rowc, textvariable=seg_ch1, width=7).pack(side="left")
    ttk.Label(sf, text="Higher percentile = tighter ROI.", foreground="#666").pack(anchor="w", pady=(2, 0))

    def get_seg():
        return dict(method=seg_method.get(), percentile=seg_pct.get(),
                    ch=(seg_ch0.get(), seg_ch1.get()))

    # ── Display options ───────────────────────────────────────────────────
    smooth_var = tk.BooleanVar(value=True)
    equal_var = tk.BooleanVar(value=True)
    df = ttk.LabelFrame(left, text="Map display", padding=6); df.pack(fill="x", pady=6)
    ttk.Checkbutton(df, text="Smooth maps (interpolation)", variable=smooth_var).pack(anchor="w")
    ttk.Checkbutton(df, text="Equal size (crop to common)", variable=equal_var).pack(anchor="w")

    nb = ttk.Notebook(win); nb.pack(side="right", fill="both", expand=True)
    tab_box = ttk.Frame(nb); tab_spec = ttk.Frame(nb); tab_maps = ttk.Frame(nb); tab_seg = ttk.Frame(nb)
    nb.add(tab_box, text="Box plot + stats"); nb.add(tab_spec, text="Mean spectra")
    nb.add(tab_maps, text="A/B maps"); nb.add(tab_seg, text="Segmentation preview")
    state = {"figs": {}}

    def _show(fig, tab):
        for c in tab.winfo_children():
            c.destroy()
        cv = FigureCanvasTkAgg(fig, master=tab); cv.draw()
        cv.get_tk_widget().pack(fill="both", expand=True)

    def preview_masks():
        if not groups:
            messagebox.showwarning("No data", "Add at least one group of .wdf files."); return
        prev = []
        try:
            ba, bb = get_bands()
            for label, files in groups.items():
                for f in files:
                    _ab, _w, _ms, _roi, mask2d, chmap2d, _abmap = cell_ratio(
                        f, ba, bb, seg=get_seg())
                    prev.append({"name": os.path.basename(f), "group": label,
                                 "chmap": chmap2d, "mask": mask2d})
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        fig = figure_mask_preview(prev, smooth=smooth_var.get(), equal_size=equal_var.get())
        _show(fig, tab_seg); state.setdefault("figs", {})["seg"] = fig
        nb.select(tab_seg)

    def run_analysis():
        if not groups:
            messagebox.showwarning("No data", "Add at least one group of .wdf files."); return
        try:
            ba, bb = get_bands()
            res, gv, gs, w, prev, abent = run(groups, band_a=ba, band_b=bb, seg=get_seg())
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        sm, eq = smooth_var.get(), equal_var.get()
        fb = figure_boxplot(gv, title="Per-cell band-ratio comparison"); _show(fb, tab_box)
        fs = figure_mean_spectra(gs, w, ba, bb, title="Group mean intracellular spectra"); _show(fs, tab_spec)
        fmaps = figure_ab_maps(abent, res["ab_vmin"], res["ab_vmax"], ba, bb, smooth=sm, equal_size=eq); _show(fmaps, tab_maps)
        fp = figure_mask_preview(prev, smooth=sm, equal_size=eq); _show(fp, tab_seg)
        state["figs"] = {"box": fb, "spec": fs, "maps": fmaps, "seg": fp}
        if "comparison" in res:
            c = res["comparison"]
            messagebox.showinfo("Result",
                f"U={c['U']:.0f}, p={c['p']:.3f}\nHL diff={c['hodges_lehmann']:.3f} "
                f"CI[{c['hl_ci'][0]:.3f},{c['hl_ci'][1]:.3f}]\nCliff's δ={c['cliffs_delta']:.2f}")

    def export():
        if not state["figs"]:
            messagebox.showinfo("Nothing to export", "Run the analysis first."); return
        d = filedialog.askdirectory(title="Export figures to…")
        if not d: return
        state["figs"]["box"].savefig(os.path.join(d, "group_boxplot.png"), dpi=400)
        state["figs"]["box"].savefig(os.path.join(d, "group_boxplot.pdf"))
        state["figs"]["spec"].savefig(os.path.join(d, "group_mean_spectra.png"), dpi=400)
        state["figs"]["spec"].savefig(os.path.join(d, "group_mean_spectra.pdf"))
        if state["figs"].get("maps") is not None:
            state["figs"]["maps"].savefig(os.path.join(d, "group_AB_maps.png"), dpi=400)
            state["figs"]["maps"].savefig(os.path.join(d, "group_AB_maps.pdf"))
        if state["figs"].get("seg") is not None:
            state["figs"]["seg"].savefig(os.path.join(d, "segmentation_preview.png"), dpi=300)
        messagebox.showinfo("Exported", "Saved PNG + PDF to\n" + d)

    ttk.Button(actions, text="👁 Preview cell masks", command=preview_masks).pack(fill="x", pady=(0, 2))
    ttk.Button(actions, text="▶ Run comparison", command=run_analysis).pack(fill="x", pady=2)
    ttk.Button(actions, text="⬇ Export figures", command=export).pack(fill="x", pady=(0, 0))
    return win

# ============================================================================ #
#  CLI
# ============================================================================ #
def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="BioRaman group comparison (headless)")
    ap.add_argument("--group", nargs="+", action="append", metavar=("LABEL", "FILE"),
                    help="--group LABEL file1.wdf file2.wdf  (repeatable)")
    ap.add_argument("--banda", nargs=2, type=float, default=[3085, 3165])
    ap.add_argument("--bandb", nargs=2, type=float, default=[3350, 3550])
    ap.add_argument("--segmethod", choices=["otsu", "percentile"], default="otsu",
                    help="cell segmentation method")
    ap.add_argument("--percentile", type=float, default=85.0,
                    help="CH percentile for manual segmentation (higher = tighter ROI)")
    ap.add_argument("--chband", nargs=2, type=float, default=[2850, 3000],
                    help="CH cell-marker band for segmentation")
    ap.add_argument("--out", default="./group_out")
    a = ap.parse_args()
    if not a.group:
        open_group_compare(); import tkinter as tk; tk.mainloop(); return
    groups = {g[0]: g[1:] for g in a.group}
    seg = dict(method=a.segmethod, percentile=a.percentile, ch=tuple(a.chband))
    res, *_ = run(groups, band_a=tuple(a.banda), band_b=tuple(a.bandb), seg=seg, outdir=a.out)
    print(json.dumps(res.get("comparison", res), indent=2))
    print("Outputs →", a.out)

if __name__ == "__main__":
    _cli()

# ============================================================================ #
INTEGRATION = """
To expose this inside BioRaman's GUI, add to bioraman.py:

    import bioraman_group_compare as bgc
    # ... in the Analysis menu setup:
    am.add_command(label="⚖  Group Comparison…",
                   command=lambda: bgc.open_group_compare(self))

That opens the window; add groups of .wdf files, set the band windows,
Run, then Export. No other changes to bioraman.py are required.
"""
