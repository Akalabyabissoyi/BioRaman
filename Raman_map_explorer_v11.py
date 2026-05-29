#!/usr/bin/env python3
"""
Raman Map Explorer  —  Professional Edition v11
================================================
NEW in v11 (Peak Identification & Spectra Comparison):
• Peak Identification Window — open_peak_id()
    - Automatic peak detection on selected spectra
    - Matches detected peaks against a built-in Raman band library
    - Import custom band lists / CSV reference tables
    - Export identified-peak table as CSV

• Spectra Comparison Window — open_spectra_compare()
    - Side-by-side overlay/comparison of selected spectra
    - Compare clustering components and extracted endmembers
    - Export comparison figure as PNG / PDF

NEW in v7 (Spectral Unmixing & Clustering Suite):
• Cluster Analysis Window — open_clustering()
    - K-means and Agglomerative (hierarchical) clustering
    - Selectable number of components (2-10)
    - Colour-coded cluster map + mean spectra per cluster
    - Export cluster map as PNG / CSV label matrix

• MCR-ALS Window — open_mcr()
    - Multivariate Curve Resolution – Alternating Least Squares
    - Non-negative factorisation: data ≈ abundances × pure spectra
    - Selectable number of components; max-iter & convergence controls
    - Abundance maps + recovered pure spectra side-by-side

• N-FINDR Endmember Extraction — open_nfindr()
    - Identifies the purest spectral signatures in the data
    - Interactive endmember count slider
    - Abundance maps computed via NNLS from recovered endmembers

• Spectral Tools Window — open_spectral_tools()
    - Spectral resampling to a user-defined equidistant wavenumber grid
    - Spatial map crop (set pixel bounding box interactively)
    - Map rotation (0 / 90 / 180 / 270 degrees)
    - Optical substrate / background reference subtraction

NEW in v6 (HORIBA LabSpec 6 / 3D Surface and Volume Display):
• 3D Confocal Volume Viewer — open_3d_viewer()
    - Load Z-stack of WDF files (one per depth plane) or use current 2-D map
      with a synthetic depth axis for instant demo
    - Four render modes selectable at any time:
        🔴 Volume Scatter  — above-threshold voxels as 3-D coloured scatter;
                              Band A = colourmap, Band B = green overlay
        📐 Orthogonal Slices — interactive XY / XZ / YZ cross-section planes;
                               drag slice indices and re-render
        🌄 3D Surface      — intensity map as a 3-D height surface with
                               optional second-band overlay
        🟩 Multi-band RGB  — Band A→Red, Band B→Green, Band C→Blue;
                               replicates HORIBA's polymer/geological renders
    - Controls: threshold, voxel alpha, scatter point size, Z scale,
                pre-smoothing σ, colourmap, dark/light background,
                elevation/azimuth, bounding box, axis labels
    - Export PNG / PDF at 250 dpi

Dependencies
------------
pip install numpy scipy matplotlib pybaselines renishawWiRE pillow
pip install scikit-learn pandas seaborn openpyxl
(mpl_toolkits.mplot3d ships with matplotlib — no extra install needed)

v7 additions use only packages already listed above (scipy.optimize.nnls,
sklearn.cluster, sklearn.decomposition.NMF).
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, time, threading, queue
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── numeric / scientific ──────────────────────────────────────────────────────
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter, zoom

# ── GUI ───────────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# ── plotting ──────────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Ellipse
from PIL import Image, ImageTk

try:
    from pybaselines import whittaker
    HAS_PYBL = True
except ImportError:
    HAS_PYBL = False

try:
    from renishawWiRE import WDFReader
    HAS_WDF = True
except ImportError:
    HAS_WDF = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKL = True
except ImportError:
    HAS_SKL = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# ── matplotlib theme ──────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "axes.facecolor":      "#f7f8fc",
    "figure.facecolor":    "#ffffff",
    "axes.edgecolor":      "#b0b8cc",
    "axes.linewidth":      0.9,
    "axes.labelcolor":     "#2a2e3d",
    "axes.titlecolor":     "#1a1d2a",
    "axes.titlesize":      14,
    "axes.labelsize":      12,
    "axes.titleweight":    "semibold",
    "xtick.color":         "#5a6070",
    "ytick.color":         "#5a6070",
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "grid.color":          "#dde2ee",
    "grid.linewidth":      0.5,
    "text.color":          "#2a2e3d",
    "font.family":         "DejaVu Sans",
    "font.size":           12,
    "legend.fontsize":     11,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#c0c8dc",
})

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":        "#f0f2f8",
    "panel":     "#ffffff",
    "sidebar":   "#f5f6fb",
    "border":    "#d0d6e8",
    "header":    "#1e2235",
    "accent":    "#2563eb",
    "accent2":   "#7c3aed",
    "band_a":    "#f59e0b",
    "band_b":    "#06b6d4",
    "success":   "#10b981",
    "danger":    "#ef4444",
    "warn":      "#f59e0b",
    "text_hi":   "#111827",
    "text_mid":  "#4b5563",
    "text_dim":  "#9ca3af",
    "spec_line": "#1d4ed8",
    "compare":   "#dc2626",
    "roi":       "#ff6b35",
}

COLORMAPS = ["turbo","viridis","plasma","inferno","magma",
             "hot","RdYlBu_r","coolwarm","Spectral_r","jet"]
ZOOM = 3

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PreprocessParams:
    cosmic_removal:   bool  = True
    cosmic_threshold: float = 8.0
    cosmic_width:     int   = 3
    dark_removal:     bool  = True
    baseline_method:  str   = "asls"   # asls|arpls|drpls|none
    asls_lam:         float = 1e5
    asls_p:           float = 0.001
    smoothing:        bool  = True
    sg_window:        int   = 11
    sg_poly:          int   = 3
    normalisation:    str   = "area"   # max|area|none  — area preserves relative band ratios


def _process_one(args):
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    idx, s_raw, p = args
    s = s_raw.astype(float)
    had_spike = False

    # 1. Cosmic-ray removal — modified Z-score on 1st derivative
    if p.cosmic_removal and len(s) >= 10:
        dy  = np.diff(s)
        med = np.median(dy)
        mad = np.median(np.abs(dy - med)) or 1e-10
        z   = 0.6745 * (dy - med) / mad
        for i in range(len(z) - 1):
            if (abs(z[i]) > p.cosmic_threshold and
                    abs(z[i+1]) > p.cosmic_threshold and
                    z[i] * z[i+1] < 0):
                had_spike = True
                lo = max(0,        i + 1 - p.cosmic_width)
                hi = min(len(s)-1, i + 1 + p.cosmic_width)
                lft = max(0,        lo - 1)
                rgt = min(len(s)-1, hi + 1)
                if lft != rgt:
                    xi = np.arange(lo, hi + 1)
                    s[lo:hi+1] = np.interp(xi, [lft, rgt], [s[lft], s[rgt]])

    # 2. Dark / pedestal
    if p.dark_removal:
        s -= s.min()

    # 3. Baseline correction
    if p.baseline_method != "none" and HAS_PYBL:
        try:
            if   p.baseline_method == "asls":
                bl, _ = whittaker.asls(s,  lam=p.asls_lam, p=p.asls_p, max_iter=50)
            elif p.baseline_method == "arpls":
                bl, _ = whittaker.arpls(s, lam=p.asls_lam, max_iter=50)
            elif p.baseline_method == "drpls":
                bl, _ = whittaker.drpls(s, lam=p.asls_lam, max_iter=50)
            else:
                bl, _ = whittaker.asls(s,  lam=p.asls_lam, p=p.asls_p, max_iter=50)
            s = np.clip(s - bl, 0, None)
        except Exception:
            pass

    # 4. Savitzky-Golay smoothing
    if p.smoothing and len(s) > p.sg_window:
        w = p.sg_window if p.sg_window % 2 == 1 else p.sg_window + 1
        w = max(w, p.sg_poly + 2)
        s = savgol_filter(s, window_length=w, polyorder=p.sg_poly)

    # 5. Normalisation
    if p.normalisation == "max":
        pk = s.max()
        if pk > 0: s /= pk
    elif p.normalisation == "area":
        area = float(np.trapz(np.clip(s, 0, None)))
        if area > 0: s /= area

    return idx, s, had_spike


def preprocess_spectrum(s, params=None):
    if params is None: params = PreprocessParams()
    _, result, _ = _process_one((0, s, params))
    return result


def preprocess_map(data, params=None, cb=None):
    if params is None: params = PreprocessParams()
    t0 = time.perf_counter()
    Y, X, W  = data.shape
    total    = Y * X
    flat     = [(i, data[y, x], params)
                for i, (y, x) in enumerate(np.ndindex(Y, X))]
    out_flat = [None] * total
    cosmic_n = 0
    done     = 0
    n_workers = min(os.cpu_count() or 1, 8)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one, item): item[0] for item in flat}
        for fut in as_completed(futures):
            idx, s_proc, spike = fut.result()
            out_flat[idx] = s_proc
            if spike: cosmic_n += 1
            done += 1
            if cb and done % max(1, total // 200) == 0:
                cb(done / total)
    if cb: cb(1.0)

    out = np.empty((Y, X, W), dtype=float)
    for i, (y, x) in enumerate(np.ndindex(Y, X)):
        out[y, x] = out_flat[i]

    elapsed = time.perf_counter() - t0
    report = {
        "total_spectra":    total,
        "map_shape":        f"{X} × {Y}",
        "spectral_points":  W,
        "cosmic_removed":   cosmic_n,
        "dark_subtraction": "yes" if params.dark_removal else "no",
        "baseline_method":  params.baseline_method.upper()
                            if params.baseline_method != "none" else "SKIPPED",
        "baseline_lam":     f"{params.asls_lam:.0e}",
        "baseline_p":       f"{params.asls_p}",
        "smoothing":        (f"Savitzky-Golay  window={params.sg_window}"
                             f"  poly={params.sg_poly}")
                            if params.smoothing else "SKIPPED",
        "normalisation":    params.normalisation.upper(),
        "workers":          n_workers,
        "elapsed_s":        f"{elapsed:.1f}",
    }
    return out, report


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM WIDGETS
# ─────────────────────────────────────────────────────────────────────────────
class RangeSlider(tk.Frame):
    H=30; TRACK_H=5; HANDLE_R=8

    def __init__(self, parent, label, from_, to_, init_lo, init_hi,
                 color=C["accent"], resolution=1, command=None, **kw):
        super().__init__(parent, bg=C["sidebar"], **kw)
        self._from=float(from_); self._to=float(to_)
        self._lo=tk.DoubleVar(value=float(init_lo))
        self._hi=tk.DoubleVar(value=float(init_hi))
        self._res=resolution; self._cmd=command; self._color=color; self._drag=None

        hdr=tk.Frame(self, bg=C["sidebar"])
        hdr.pack(fill="x", padx=8, pady=(6,0))
        tk.Label(hdr, text=label, bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left")
        self._lo_lbl=tk.Label(hdr, bg=C["sidebar"], fg=color,
                              font=("Consolas", 10, "bold"))
        self._lo_lbl.pack(side="left", padx=(6,0))
        tk.Label(hdr, text="–", bg=C["sidebar"], fg=C["text_dim"],
                 font=("Segoe UI", 10)).pack(side="left", padx=2)
        self._hi_lbl=tk.Label(hdr, bg=C["sidebar"], fg=color,
                              font=("Consolas", 10, "bold"))
        self._hi_lbl.pack(side="left")

        self._cv=tk.Canvas(self, height=self.H, bg=C["sidebar"],
                           highlightthickness=0, cursor="sb_h_double_arrow")
        self._cv.pack(fill="x", padx=8, pady=(2,6))
        self._cv.bind("<Configure>",       self._draw)
        self._cv.bind("<ButtonPress-1>",   self._press)
        self._cv.bind("<B1-Motion>",       self._move)
        self._cv.bind("<ButtonRelease-1>", self._release)
        self._update_labels()

    def _tw(self):    return max(self._cv.winfo_width(), 1)
    def _x2v(self,x): return self._from+(x/self._tw())*(self._to-self._from)
    def _v2x(self,v): return (v-self._from)/(self._to-self._from)*self._tw()
    def _snap(self,v): return max(self._from, min(self._to,
                                  round(v/self._res)*self._res))
    def _update_labels(self):
        lo,hi=self._lo.get(),self._hi.get()
        fmt=".0f" if self._res>=1 else ".2f"
        self._lo_lbl.config(text=f"{lo:{fmt}}")
        self._hi_lbl.config(text=f"{hi:{fmt}}")

    def _draw(self,*_):
        cv=self._cv; cv.delete("all"); w=self._tw(); cy=self.H//2; r=self.HANDLE_R
        cv.create_rectangle(r, cy-self.TRACK_H//2, w-r, cy+self.TRACK_H//2,
                            fill="#dde2ee", outline="")
        lx=self._v2x(self._lo.get()); hx=self._v2x(self._hi.get())
        cv.create_rectangle(lx, cy-self.TRACK_H//2, hx, cy+self.TRACK_H//2,
                            fill=self._color, outline="")
        for x,tag in [(lx,"lo"),(hx,"hi")]:
            cv.create_oval(x-r,cy-r,x+r,cy+r, fill="white",
                          outline=self._color, width=2, tags=tag)
        self._update_labels()

    def _nearest(self,x):
        lx=self._v2x(self._lo.get()); hx=self._v2x(self._hi.get())
        return "lo" if abs(x-lx)<=abs(x-hx) else "hi"

    def _press(self,e):  self._drag=self._nearest(e.x)
    def _release(self,e):self._drag=None

    def _move(self,e):
        if not self._drag: return
        v=self._snap(self._x2v(e.x))
        lo,hi=self._lo.get(),self._hi.get()
        if self._drag=="lo":
            self._lo.set(min(v, hi-self._res))
        else:
            self._hi.set(max(v, lo+self._res))
        self._draw()
        if self._cmd: self._cmd()

    @property
    def low(self):  return self._lo.get()
    @property
    def high(self): return self._hi.get()
    def set(self,lo,hi): self._lo.set(lo); self._hi.set(hi); self._draw()


class SingleSlider(tk.Frame):
    H=26; TRACK_H=5; HANDLE_R=7

    def __init__(self, parent, label, from_, to_, init,
                 color=C["accent"], resolution=0.01, command=None, **kw):
        super().__init__(parent, bg=C["sidebar"], **kw)
        self._from=float(from_); self._to=float(to_)
        self._v=tk.DoubleVar(value=float(init))
        self._res=resolution; self._cmd=command; self._color=color

        hdr=tk.Frame(self, bg=C["sidebar"])
        hdr.pack(fill="x", padx=8, pady=(4,0))
        tk.Label(hdr, text=label, bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left")
        self._lbl=tk.Label(hdr, bg=C["sidebar"], fg=color,
                           font=("Consolas", 10, "bold"))
        self._lbl.pack(side="right")

        self._cv=tk.Canvas(self, height=self.H, bg=C["sidebar"],
                           highlightthickness=0)
        self._cv.pack(fill="x", padx=8, pady=(2,4))
        self._cv.bind("<Configure>",      self._draw)
        self._cv.bind("<ButtonPress-1>",  self._set_from_px)
        self._cv.bind("<B1-Motion>",      self._set_from_px)
        self._draw()

    def _tw(self):    return max(self._cv.winfo_width(), 1)
    def _v2x(self,v): return (v-self._from)/(self._to-self._from)*self._tw()
    def _x2v(self,x): return self._from+x/self._tw()*(self._to-self._from)
    def _snap(self,v): return max(self._from, min(self._to,
                                  round(v/self._res)*self._res))

    def _draw(self,*_):
        cv=self._cv; w=self._tw(); cy=self.H//2; r=self.HANDLE_R
        cv.delete("all")
        cv.create_rectangle(r, cy-self.TRACK_H//2, w-r, cy+self.TRACK_H//2,
                            fill="#dde2ee", outline="")
        vx=self._v2x(self._v.get())
        cv.create_rectangle(r, cy-self.TRACK_H//2, vx, cy+self.TRACK_H//2,
                            fill=self._color, outline="")
        cv.create_oval(vx-r,cy-r,vx+r,cy+r, fill="white",
                      outline=self._color, width=2)
        fmt=".0f" if self._res>=1 else ".2f"
        self._lbl.config(text=f"{self._v.get():{fmt}}")

    def _set_from_px(self,e):
        self._v.set(self._snap(self._x2v(e.x)))
        self._draw()
        if self._cmd: self._cmd()

    @property
    def value(self): return self._v.get()
    def set(self,v): self._v.set(v); self._draw()


class SectionDiv(tk.Frame):
    def __init__(self, parent, text, **kw):
        super().__init__(parent, bg=C["sidebar"], **kw)
        tk.Label(self, text=text, bg=C["sidebar"], fg=C["text_dim"],
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=(10,6), pady=6)
        tk.Frame(self, bg=C["border"], height=1).pack(
            side="left", fill="x", expand=True, pady=6, padx=(0,10))


# ─────────────────────────────────────────────────────────────────────────────
# ROI MANAGER
# ─────────────────────────────────────────────────────────────────────────────
class ROIManager:
    """Handles drawing rectangle, ellipse, polygon (click-to-add points)
    or freehand (mouse-tracked) ROI on the map axes.

    Callback receives a boolean mask with shape (Y, X).

    Notes:
      • Coordinates on the displayed map are in *zoomed* pixel units.
      • Masks are generated in unzoomed pixel grid by dividing x/y by `self.zoom`.
      • Polygon mode: left-click adds vertices, right-click finishes.
      • Freehand mode: click-drag to trace boundary, release to finish.
    """

    MODES = ["rectangle", "ellipse", "polygon", "freehand"]

    def __init__(self, ax, canvas, zoom_factor, callback):
        self.ax      = ax
        self.canvas  = canvas
        self.zoom    = float(zoom_factor)
        self.cb      = callback

        self.mode    = "rectangle"
        self.active  = False

        self._patch  = None
        self._line   = None
        self._pts    = []
        self._start  = None
        self._cids   = []
        self._shape  = None

        self._fh_drawing = False

    def activate(self, mode, shape):
        self.deactivate()
        self.mode   = mode
        self._shape = shape
        self.active = True
        self._pts   = []
        self._start = None
        self._fh_drawing = False

        self._cids  = [
            self.canvas.mpl_connect("button_press_event",   self._press),
            self.canvas.mpl_connect("motion_notify_event",  self._move),
            self.canvas.mpl_connect("button_release_event", self._release),
        ]
        self.canvas.get_tk_widget().config(cursor="crosshair")

    def deactivate(self):
        for cid in self._cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._cids = []
        self.active = False
        self._fh_drawing = False
        self._start = None
        self._pts = []
        self._clear_artists()
        self.canvas.get_tk_widget().config(cursor="")

    def _clear_artists(self):
        for art_name in ("_patch", "_line"):
            art = getattr(self, art_name)
            if art is not None:
                try:
                    art.remove()
                except Exception:
                    pass
                setattr(self, art_name, None)
        self.canvas.draw_idle()

    def _press(self, e):
        if e.inaxes != self.ax or e.xdata is None or e.ydata is None:
            return

        if self.mode == "polygon":
            if e.button == 3:
                self._finish_polygon(); return
            self._pts.append((float(e.xdata), float(e.ydata)))
            self._redraw_polygon(); return

        if self.mode == "freehand":
            if e.button != 1:
                return
            self._fh_drawing = True
            self._pts = [(float(e.xdata), float(e.ydata))]
            self._redraw_freehand(); return

        if e.button == 1:
            self._start = (float(e.xdata), float(e.ydata))

    def _move(self, e):
        if e.inaxes != self.ax or e.xdata is None or e.ydata is None:
            return

        if self.mode == "polygon" and self._pts:
            self._redraw_polygon(cursor=(float(e.xdata), float(e.ydata)))
            return

        if self.mode == "freehand" and self._fh_drawing:
            x, y = float(e.xdata), float(e.ydata)
            if self._pts:
                x0, y0 = self._pts[-1]
                if (x - x0) ** 2 + (y - y0) ** 2 < 0.5 ** 2:
                    return
            self._pts.append((x, y))
            self._redraw_freehand(); return

        if self._start and self.mode in ("rectangle", "ellipse"):
            self._clear_artists()
            x0, y0 = self._start
            x1, y1 = float(e.xdata), float(e.ydata)

            if self.mode == "rectangle":
                self._patch = plt.Rectangle(
                    (min(x0, x1), min(y0, y1)),
                    abs(x1 - x0), abs(y1 - y0),
                    linewidth=1.5, edgecolor=C["roi"],
                    facecolor=C["roi"], alpha=0.25)
                self.ax.add_patch(self._patch)
            else:
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                self._patch = Ellipse(
                    (cx, cy), abs(x1 - x0), abs(y1 - y0),
                    linewidth=1.5, edgecolor=C["roi"],
                    facecolor=C["roi"], alpha=0.25)
                self.ax.add_patch(self._patch)

            self.canvas.draw_idle()

    def _release(self, e):
        if e.inaxes != self.ax or e.xdata is None or e.ydata is None:
            return

        if self.mode == "polygon":
            return

        if self.mode == "freehand":
            if not self._fh_drawing:
                return
            self._fh_drawing = False
            self._pts.append((float(e.xdata), float(e.ydata)))
            self._finish_freehand(); return

        if self._start is None:
            return
        x0, y0 = self._start
        x1, y1 = float(e.xdata), float(e.ydata)
        self._start = None
        mask = self._build_mask(x0, y0, x1, y1)
        if mask is not None and mask.any():
            self.cb(mask)

    def _redraw_polygon(self, cursor=None):
        self._clear_artists()
        pts = self._pts.copy()
        if cursor:
            pts.append(cursor)
        if len(pts) >= 2:
            xs = [p[0] for p in pts] + [pts[0][0]]
            ys = [p[1] for p in pts] + [pts[0][1]]
            self._patch, = self.ax.plot(xs, ys, color=C["roi"], lw=1.5, ls="--")
        self.canvas.draw_idle()

    def _redraw_freehand(self):
        if len(self._pts) < 2:
            return
        xs = [p[0] for p in self._pts]
        ys = [p[1] for p in self._pts]
        if self._line is None:
            (self._line,) = self.ax.plot(xs, ys, color=C["roi"], lw=1.6, ls="-")
        else:
            self._line.set_data(xs, ys)
        self.canvas.draw_idle()

    def _finish_polygon(self):
        if len(self._pts) < 3:
            return
        mask = self._poly_mask(self._pts)
        self._clear_artists(); self.deactivate()
        if mask is not None and mask.any():
            self.cb(mask)

    def _finish_freehand(self):
        if len(self._pts) < 3:
            self._clear_artists(); return
        mask = self._poly_mask(self._pts)
        self._clear_artists(); self.deactivate()
        if mask is not None and mask.any():
            self.cb(mask)

    def _build_mask(self, x0, y0, x1, y1):
        if self._shape is None:
            return None
        Y, X = self._shape
        xi0 = int(min(x0, x1) / self.zoom); xi1 = int(max(x0, x1) / self.zoom)
        yi0 = int(min(y0, y1) / self.zoom); yi1 = int(max(y0, y1) / self.zoom)
        xi0 = max(0, xi0); yi0 = max(0, yi0)
        xi1 = min(X - 1, xi1); yi1 = min(Y - 1, yi1)

        mask = np.zeros((Y, X), dtype=bool)
        if self.mode == "rectangle":
            mask[yi0:yi1 + 1, xi0:xi1 + 1] = True
        else:
            cx = (xi0 + xi1) / 2
            cy = (yi0 + yi1) / 2
            rx = max((xi1 - xi0) / 2, 0.5)
            ry = max((yi1 - yi0) / 2, 0.5)
            yy, xx = np.mgrid[0:Y, 0:X]
            mask = ((xx - cx) ** 2 / rx ** 2 + (yy - cy) ** 2 / ry ** 2) <= 1
        return mask

    def _poly_mask(self, pts):
        if self._shape is None:
            return None
        Y, X = self._shape
        from matplotlib.path import Path as MplPath
        px = np.array([(p[0] / self.zoom, p[1] / self.zoom) for p in pts], dtype=float)
        if len(px) < 3:
            return None
        if not np.allclose(px[0], px[-1]):
            px = np.vstack([px, px[0]])
        path = MplPath(px)
        yy, xx = np.mgrid[0:Y, 0:X]
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        return path.contains_points(coords).reshape(Y, X)
# PCA WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class PCAWindow(tk.Toplevel):
    """
    Standalone PCA analysis window.
    Load multiple WDF or XLSX map files → preprocess → PCA → plots.
    """
    COLORS = ["#2563eb","#ef4444","#10b981","#f59e0b","#7c3aed",
              "#06b6d4","#ec4899","#84cc16","#f97316","#6366f1"]

    def __init__(self, parent, pp_params):
        super().__init__(parent)
        self.title("PCA Analysis — Multi-File")
        self.geometry("1200x780")
        self.configure(bg=C["bg"])
        self.pp_params = pp_params

        self._files   = []   # list of {"path":..., "label":..., "data":None}
        self._results = None

        self._build_ui()
        self._style_ttk()

    # ── style ─────────────────────────────────────────────────────────────────
    def _style_ttk(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=C["sidebar"], foreground=C["text_hi"],
                    fieldbackground="white", font=("Segoe UI", 10),
                    bordercolor=C["border"])
        for name, bg, fg in [
            ("P.TButton",  C["accent"],  "white"),
            ("N.TButton",  "#e4e8f4",    C["text_hi"]),
            ("D.TButton",  C["danger"],  "white"),
        ]:
            s.configure(name, background=bg, foreground=fg,
                        relief="flat", padding=(10,5),
                        font=("Segoe UI", 11, "bold"), borderwidth=0)
            s.map(name, background=[("active", C["accent2"])])

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar
        tb = tk.Frame(self, bg=C["header"], height=48)
        tb.pack(fill="x"); tb.pack_propagate(False)
        tk.Label(tb, text="◈  PCA ANALYSIS",
                 bg=C["header"], fg="white",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=16, pady=12)

        # Left panel: file list + controls
        left = tk.Frame(self, bg=C["sidebar"], width=320)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        # File list
        SectionDiv(left, "MAP FILES").pack(fill="x")

        self._listbox = tk.Listbox(left, bg="white", fg=C["text_hi"],
                                   font=("Segoe UI", 11), selectmode="extended",
                                   height=10, relief="flat",
                                   highlightthickness=1,
                                   highlightbackground=C["border"])
        self._listbox.pack(fill="x", padx=10, pady=4)

        btns = tk.Frame(left, bg=C["sidebar"])
        btns.pack(fill="x", padx=10, pady=2)
        ttk.Button(btns, text="+ Add WDF",  style="P.TButton",
                   command=self._add_wdf).pack(side="left",  padx=2)
        ttk.Button(btns, text="+ Add XLSX", style="N.TButton",
                   command=self._add_xlsx).pack(side="left", padx=2)
        ttk.Button(btns, text="✕ Remove",   style="D.TButton",
                   command=self._remove_selected).pack(side="right", padx=2)

        # Label editor
        SectionDiv(left, "LABEL SELECTED").pack(fill="x")
        lf = tk.Frame(left, bg=C["sidebar"])
        lf.pack(fill="x", padx=10, pady=4)
        self._label_var = tk.StringVar()
        tk.Entry(lf, textvariable=self._label_var, bg="white",
                 font=("Segoe UI", 10), relief="flat",
                 highlightthickness=1,
                 highlightbackground=C["border"]).pack(side="left", fill="x",
                                                       expand=True, padx=(0,6))
        ttk.Button(lf, text="Set", style="N.TButton",
                   command=self._set_label).pack(side="right")

        # PCA options
        SectionDiv(left, "PCA OPTIONS").pack(fill="x")
        opts = tk.Frame(left, bg=C["sidebar"])
        opts.pack(fill="x", padx=10, pady=4)

        tk.Label(opts, text="Components", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=2)
        self._n_comp = tk.IntVar(value=3)
        ttk.Spinbox(opts, from_=2, to=10, textvariable=self._n_comp,
                    width=6).grid(row=0, column=1, padx=8, pady=2)

        tk.Label(opts, text="Wavenumber min", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=2)
        self._wn_lo = tk.DoubleVar(value=500)
        ttk.Spinbox(opts, from_=0, to=4000, textvariable=self._wn_lo,
                    width=8).grid(row=1, column=1, padx=8, pady=2)

        tk.Label(opts, text="Wavenumber max", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=2, column=0, sticky="w", pady=2)
        self._wn_hi = tk.DoubleVar(value=3500)
        ttk.Spinbox(opts, from_=0, to=4000, textvariable=self._wn_hi,
                    width=8).grid(row=2, column=1, padx=8, pady=2)

        tk.Label(opts, text="Spectra/file (0=all)", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=3, column=0, sticky="w", pady=2)
        self._max_spec = tk.IntVar(value=0)
        ttk.Spinbox(opts, from_=0, to=10000, textvariable=self._max_spec,
                    width=8).grid(row=3, column=1, padx=8, pady=2)

        self._outlier_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opts, text="Remove outliers (>2σ)",
                       variable=self._outlier_var,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10)).grid(
                           row=4, column=0, columnspan=2, sticky="w", pady=4)

        self._scale_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opts, text="Standardise features",
                       variable=self._scale_var,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10)).grid(
                           row=5, column=0, columnspan=2, sticky="w", pady=2)

        # Run button
        ttk.Button(left, text="▶  Run PCA", style="P.TButton",
                   command=self._run_pca).pack(fill="x", padx=10, pady=10)

        # Progress
        self._prog = ttk.Progressbar(left, mode="determinate")
        self._prog.pack(fill="x", padx=10, pady=2)
        self._status_lbl = tk.Label(left, text="Add files and press Run PCA",
                                    bg=C["sidebar"], fg=C["text_dim"],
                                    font=("Segoe UI", 11), wraplength=280,
                                    justify="left")
        self._status_lbl.pack(padx=10, pady=4, anchor="w")

        # Save button
        ttk.Button(left, text="↓  Save PCA Figure", style="N.TButton",
                   command=self._save_fig).pack(fill="x", padx=10, pady=4)

        # PC selector for spatial map
        SectionDiv(left, "SPATIAL SCORE MAP").pack(fill="x")
        pc_row = tk.Frame(left, bg=C["sidebar"])
        pc_row.pack(fill="x", padx=10, pady=4)
        tk.Label(pc_row, text="Display PC", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left")
        self._pc_sel = tk.IntVar(value=1)
        ttk.Spinbox(pc_row, from_=1, to=10, textvariable=self._pc_sel,
                    width=5, command=self._redraw_spatial).pack(side="left", padx=8)
        tk.Label(pc_row, text="(redraws map)", bg=C["sidebar"], fg=C["text_dim"],
                 font=("Segoe UI", 9, "italic")).pack(side="left")

        # Right: plot area
        right = tk.Frame(self, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)

        self.fig = plt.figure(figsize=(13, 7), facecolor="#ffffff")
        # 2×3 grid: A B E / C D (E=spatial, last cell hidden if single file)
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               hspace=0.42, wspace=0.38,
                               left=0.07, right=0.97,
                               top=0.93, bottom=0.08)
        self.axes = np.array([
            [self.fig.add_subplot(gs[0, 0]),
             self.fig.add_subplot(gs[0, 1]),
             self.fig.add_subplot(gs[0, 2])],
            [self.fig.add_subplot(gs[1, 0]),
             self.fig.add_subplot(gs[1, 1]),
             self.fig.add_subplot(gs[1, 2])],
        ])
        self._init_plots()

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        nav = NavigationToolbar2Tk(self.canvas, right)
        nav.update()

    def _init_plots(self):
        titles = ["A: PCA Scores (PC1 vs PC2)",
                  "B: PC1 Loadings",
                  "E: PC1 Spatial Score Map",
                  "C: PC1 Score Distribution",
                  "D: Explained Variance",
                  ""]
        for ax, t in zip(self.axes.flat, titles):
            ax.set_title(t, fontsize=11, fontweight="semibold")
            ax.tick_params(labelsize=9)
        # hide the 6th placeholder axis
        self.axes[1, 2].set_visible(False)

    # ── file management ───────────────────────────────────────────────────────
    def _add_wdf(self):
        if not HAS_WDF:
            messagebox.showerror("Missing library",
                "renishawWiRE not installed.\npip install renishawWiRE", parent=self)
            return
        paths = filedialog.askopenfilenames(
            title="Add WDF map files",
            filetypes=[("Renishaw WDF","*.wdf"),("All","*.*")],
            parent=self)
        for p in paths:
            label = Path(p).stem
            self._files.append({"path":p, "label":label,
                                 "fmt":"wdf", "data":None})
            self._listbox.insert("end", f"[WDF]  {label}")

    def _add_xlsx(self):
        if not HAS_PD:
            messagebox.showerror("Missing library",
                "pandas not installed.\npip install pandas openpyxl", parent=self)
            return
        paths = filedialog.askopenfilenames(
            title="Add XLSX map files",
            filetypes=[("Excel XLSX","*.xlsx"),("All","*.*")],
            parent=self)
        for p in paths:
            label = Path(p).stem
            self._files.append({"path":p, "label":label,
                                 "fmt":"xlsx", "data":None})
            self._listbox.insert("end", f"[XLSX] {label}")

    def _remove_selected(self):
        sel = list(self._listbox.curselection())
        for i in reversed(sel):
            self._listbox.delete(i)
            del self._files[i]

    def _set_label(self):
        sel = self._listbox.curselection()
        lbl = self._label_var.get().strip()
        if not lbl or not sel: return
        for i in sel:
            self._files[i]["label"] = lbl
            self._listbox.delete(i)
            fmt = "WDF" if self._files[i]["fmt"]=="wdf" else "XLSX"
            self._listbox.insert(i, f"[{fmt}]  {lbl}")

    # ── PCA run ───────────────────────────────────────────────────────────────
    def _run_pca(self):
        if not self._files:
            messagebox.showwarning("No files","Add map files first.", parent=self)
            return
        if not HAS_SKL:
            messagebox.showerror("Missing library",
                "scikit-learn not installed.\npip install scikit-learn", parent=self)
            return
        self._status_lbl.config(text="Loading and preprocessing…")
        self._prog["value"] = 0
        self.update_idletasks()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            all_X, all_labels, waves_common = self._load_all()
            self.after(0, lambda: self._finish_pca(all_X, all_labels, waves_common))
        except Exception as ex:
            self.after(0, lambda: messagebox.showerror(
                "Error", str(ex), parent=self))
            self.after(0, lambda: self._status_lbl.config(text=f"Error: {ex}"))

    def _load_all(self):
        wn_lo  = self._wn_lo.get()
        wn_hi  = self._wn_hi.get()
        max_sp = self._max_spec.get()
        params = self.pp_params
        all_X, all_labels = [], []
        waves_common = None
        self._spatial_shapes = []   # (label, Y, X) for spatial score map

        for fi, finfo in enumerate(self._files):
            self.after(0, lambda fi=fi: self._status_lbl.config(
                text=f"Loading {Path(finfo['path']).name}…"))
            self.after(0, lambda fi=fi: self._prog.configure(
                value=fi/len(self._files)*50))

            # ── load raw data ─────────────────────────────────────────────
            if finfo["fmt"] == "wdf":
                reader  = WDFReader(finfo["path"])
                raw     = reader.spectra      # Y×X×W
                xdata   = reader.xdata
                Y,X,W   = raw.shape
                self._spatial_shapes.append((finfo["label"], Y, X))
                # flatten to N×W
                raw_flat = raw.reshape(Y*X, W)
            else:
                # XLSX format: columns #Wave and #Intensity (like the script provided)
                df      = pd.read_excel(finfo["path"])
                df      = df.sort_values("#Wave")
                xdata   = np.unique(df["#Wave"].values)
                n       = len(xdata)
                ints    = df["#Intensity"].values
                raw_flat= ints.reshape(len(ints)//n, n)
                self._spatial_shapes.append((finfo["label"], None, None))

            # wavenumber range selection
            mask_w = (xdata >= wn_lo) & (xdata <= wn_hi)
            xdata_sel = xdata[mask_w]
            raw_flat  = raw_flat[:, mask_w]

            # common wavenumber axis
            if waves_common is None:
                waves_common = xdata_sel
            else:
                # interpolate to common axis if different
                if not np.allclose(waves_common, xdata_sel, atol=0.5):
                    new_flat = np.zeros((raw_flat.shape[0], len(waves_common)))
                    for ri in range(raw_flat.shape[0]):
                        new_flat[ri] = np.interp(waves_common, xdata_sel, raw_flat[ri])
                    raw_flat = new_flat

            # subsample
            if max_sp > 0 and raw_flat.shape[0] > max_sp:
                idx = np.random.choice(raw_flat.shape[0], max_sp, replace=False)
                raw_flat = raw_flat[idx]

            # preprocess each spectrum
            proc = []
            for si, s in enumerate(raw_flat):
                proc.append(preprocess_spectrum(s, params))
            proc = np.array(proc)

            all_X.extend(proc)
            all_labels.extend([finfo["label"]] * len(proc))
            self.after(0, lambda fi=fi: self._prog.configure(
                value=50 + fi/len(self._files)*50))

        return np.array(all_X), np.array(all_labels), waves_common

    def _finish_pca(self, X, labels, waves):
        self._status_lbl.config(text="Running PCA…")
        self._prog["value"] = 90

        # Outlier removal (per group, >2σ in PC1)
        if self._outlier_var.get():
            pca_tmp = PCA(n_components=min(3, X.shape[0], X.shape[1]))
            sc_tmp  = pca_tmp.fit_transform(X)
            keep    = np.ones(len(labels), dtype=bool)
            for g in np.unique(labels):
                idx = labels == g
                sc_g = sc_tmp[idx, 0]
                mu, sigma = sc_g.mean(), sc_g.std()
                if sigma > 0:
                    bad = np.abs(sc_g - mu) > 2 * sigma
                    idx_where = np.where(idx)[0]
                    keep[idx_where[bad]] = False
            X      = X[keep]
            labels = labels[keep]

        # Scale
        if self._scale_var.get():
            X = StandardScaler().fit_transform(X)

        n_comp = min(self._n_comp.get(), X.shape[0], X.shape[1])
        pca    = PCA(n_components=n_comp)
        scores = pca.fit_transform(X)
        expl   = pca.explained_variance_ratio_
        loads  = pca.components_

        self._results = {
            "pca": pca, "scores": scores, "labels": labels,
            "waves": waves, "expl": expl, "loads": loads, "X": X,
            "spatial_shapes": getattr(self, "_spatial_shapes", []),
        }

        self._draw_pca()
        self._prog["value"] = 100
        n_kept = len(labels)
        self._status_lbl.config(
            text=f"Done.  {n_kept} spectra from {len(self._files)} files.\n"
                 f"PC1={expl[0]*100:.1f}%  PC2={expl[1]*100:.1f}%")

    def _draw_pca(self):
        r = self._results
        scores = r["scores"]; labels = r["labels"]
        waves  = r["waves"];  expl   = r["expl"]; loads = r["loads"]
        groups = np.unique(labels)

        color_map = {g: self.COLORS[i % len(self.COLORS)]
                     for i, g in enumerate(groups)}

        for ax in self.axes.flat:
            ax.clear()
            ax.set_visible(True)

        pc_idx = min(self._pc_sel.get() - 1, scores.shape[1] - 1)
        pc_lbl = f"PC{pc_idx + 1}"

        # ── A: Scores PC1 vs PC2 ─────────────────────────────────────────────
        ax = self.axes[0, 0]
        for g in groups:
            idx = labels == g
            x, y = scores[idx, 0], scores[idx, 1]
            col  = color_map[g]
            ax.scatter(x, y, s=60, color=col, alpha=0.75, label=g, zorder=3)
            if idx.sum() >= 3:
                cov = np.cov(x, y)
                ev, evec = np.linalg.eigh(cov)
                ev   = np.maximum(ev, 0)
                ang  = np.degrees(np.arctan2(*evec[:, 1][::-1]))
                ell  = Ellipse((x.mean(), y.mean()),
                               2*np.sqrt(ev[0])*2, 2*np.sqrt(ev[1])*2,
                               angle=ang, edgecolor=col,
                               facecolor=col, alpha=0.12, lw=1.5)
                ax.add_patch(ell)
        ax.axhline(0, color=C["border"], lw=0.7)
        ax.axvline(0, color=C["border"], lw=0.7)
        ax.set_xlabel(f"PC1  ({expl[0]*100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2  ({expl[1]*100:.1f}%)", fontsize=10)
        ax.set_title("A: PCA Scores (PC1 vs PC2)", fontsize=11, fontweight="semibold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)

        # ── B: Loadings — colour-coded positive/negative ──────────────────────
        ax = self.axes[0, 1]
        ld = loads[pc_idx]
        pos_mask = ld >= 0
        neg_mask = ld < 0
        # Positive loadings in blue, negative in red — solid filled areas
        ax.fill_between(waves, ld, 0,
                        where=pos_mask, alpha=0.55,
                        color=C["accent"], label="Positive")
        ax.fill_between(waves, ld, 0,
                        where=neg_mask, alpha=0.55,
                        color=C["danger"], label="Negative")
        ax.plot(waves, ld, color="#222222", lw=0.8, alpha=0.7)
        ax.axhline(0, color=C["border"], lw=0.9)
        ax.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=10)
        ax.set_ylabel("Loading weight", fontsize=10)
        ax.set_title(f"B: {pc_lbl} Loadings", fontsize=11, fontweight="semibold")
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        # mark common Raman bands
        ylim = ax.get_ylim()
        for wn, bl in [(1000,"Ring"),(1300,"CH₂"),(1450,"CH"),
                       (1650,"C=O"),(2850,"CH₂"),(3000,"CH")]:
            if waves.min() < wn < waves.max():
                ax.axvline(wn, ls=":", color=C["text_dim"], lw=0.8)
                ax.text(wn, ylim[1]*0.85, bl, fontsize=7,
                        color=C["text_dim"], rotation=90, ha="right", va="top")

        # ── E: Spatial score map — Alex Henderson's normalisation ─────────────
        ax_e = self.axes[0, 2]
        spatial_shapes = r.get("spatial_shapes", [])
        # Use first WDF file with a known spatial shape
        shape_info = next(
            ((lbl, Y, X) for lbl, Y, X in spatial_shapes
             if Y is not None and X is not None), None)

        if shape_info is not None:
            lbl_sp, Y_sp, X_sp = shape_info
            # Extract scores for this file's label
            idx_sp = labels == lbl_sp
            sc_sp  = scores[idx_sp, pc_idx]
            n_sp   = Y_sp * X_sp

            if len(sc_sp) == n_sp:
                score_map = sc_sp.reshape(Y_sp, X_sp)
            else:
                # subsampled — can't reconstruct exactly; interpolate to grid
                score_map = None

            if score_map is not None:
                # ── Alex's approach: normalise + and − independently to ±1 ──
                pos_vals = score_map[score_map > 0]
                neg_vals = score_map[score_map < 0]
                norm_map = np.zeros_like(score_map, dtype=float)
                if pos_vals.size > 0:
                    norm_map[score_map > 0] = score_map[score_map > 0] / pos_vals.max()
                if neg_vals.size > 0:
                    norm_map[score_map < 0] = score_map[score_map < 0] / abs(neg_vals.min())

                # TwoSlopeNorm centres the diverging colourmap exactly at 0
                vmin, vmax = norm_map.min(), norm_map.max()
                if vmin < 0 < vmax:
                    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                else:
                    cnorm = Normalize(vmin=vmin, vmax=vmax)

                im = ax_e.imshow(norm_map, cmap="RdBu_r", norm=cnorm,
                                 origin="upper", interpolation="nearest",
                                 aspect="equal")
                cbar = self.fig.colorbar(im, ax=ax_e, fraction=0.046, pad=0.04)
                cbar.set_label("Score (norm. ±1)", fontsize=8)
                cbar.ax.tick_params(labelsize=7)
                # zero-score contour for clarity
                try:
                    ax_e.contour(norm_map, levels=[0],
                                 colors=["#333333"], linewidths=0.8,
                                 linestyles="--")
                except Exception:
                    pass
                ax_e.set_xlabel("X (px)", fontsize=9)
                ax_e.set_ylabel("Y (px)", fontsize=9)
                ax_e.set_title(f"E: {pc_lbl} Spatial Scores\n{lbl_sp}",
                               fontsize=11, fontweight="semibold")
            else:
                ax_e.text(0.5, 0.5,
                          "Subsampled data —\ncannot reconstruct map.\nSet Spectra/file = 0.",
                          ha="center", va="center", transform=ax_e.transAxes,
                          fontsize=9, color=C["text_dim"])
                ax_e.set_title(f"E: {pc_lbl} Spatial Score Map",
                               fontsize=11, fontweight="semibold")
        else:
            ax_e.text(0.5, 0.5,
                      "Spatial map available\nfor single WDF file only.\n"
                      "(Multi-file: select one file.)",
                      ha="center", va="center", transform=ax_e.transAxes,
                      fontsize=9, color=C["text_dim"])
            ax_e.set_title(f"E: {pc_lbl} Spatial Score Map",
                           fontsize=11, fontweight="semibold")

        # ── C: PC score distribution (strip + box) ───────────────────────────
        ax = self.axes[1, 0]
        for gi, g in enumerate(groups):
            idx = labels == g
            sc  = scores[idx, pc_idx]
            col = color_map[g]
            ax.boxplot(sc, positions=[gi], widths=0.4, patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=0.3),
                       medianprops=dict(color=col, lw=2),
                       whiskerprops=dict(color=col),
                       capprops=dict(color=col),
                       flierprops=dict(marker="o", color=col, ms=4, alpha=0.5))
            jitter = np.random.uniform(-0.15, 0.15, len(sc))
            ax.scatter(np.full(len(sc), gi) + jitter, sc,
                       color=col, alpha=0.6, s=20, zorder=3)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, fontsize=9)
        ax.set_ylabel(f"{pc_lbl} score", fontsize=10)
        ax.set_title(f"C: {pc_lbl} Score Distribution",
                     fontsize=11, fontweight="semibold")
        ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.5)

        # ── D: Explained variance scree ───────────────────────────────────────
        ax = self.axes[1, 1]
        n = len(expl)
        ax.bar(range(1, n+1), expl*100, color=C["accent"], alpha=0.8)
        ax.plot(range(1, n+1), np.cumsum(expl)*100,
                "o-", color=C["danger"], lw=1.5, ms=5, label="Cumulative")
        ax.axhline(90, ls="--", color=C["text_dim"], lw=0.8)
        ax.set_xlabel("Principal Component", fontsize=10)
        ax.set_ylabel("Explained Variance  (%)", fontsize=10)
        ax.set_title("D: Explained Variance (Scree)",
                     fontsize=11, fontweight="semibold")
        ax.set_xticks(range(1, n+1))
        ax.legend(fontsize=9)
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)

        # hide unused 6th panel
        self.axes[1, 2].set_visible(False)

        self.canvas.draw_idle()

    def _redraw_spatial(self):
        """Called when PC selector spinbox changes — redraw all panels."""
        if self._results is not None:
            self._draw_pca()

    def _save_fig(self):
        if self._results is None:
            messagebox.showwarning("No results","Run PCA first.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF","*.pdf"),("PNG","*.png"),("SVG","*.svg")],
            parent=self)
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches="tight")
            self._status_lbl.config(text=f"Saved → {Path(path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class RamanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Raman Map Explorer  —  Professional  v6")
        self.geometry("1380x820")
        self.minsize(1100, 660)
        self.configure(bg=C["bg"])

        self.spectra:    np.ndarray | None = None
        self.xdata:      np.ndarray | None = None
        self.coords:     tuple | None      = None
        self.compare_xy: tuple | None      = None
        self.wl_raw:     np.ndarray | None = None
        self.wl_resized: np.ndarray | None = None
        self._peak_anns  = []
        self._norm_var   = tk.BooleanVar(value=True)
        self._show_peaks = tk.BooleanVar(value=False)
        self.pp_params   = PreprocessParams()
        self.pp_report:  dict | None       = None
        self._roi_manager: ROIManager | None = None
        self._roi_mode    = tk.StringVar(value="rectangle")
        self._roi_mask:   np.ndarray | None = None   # stored boolean mask
        self._roi_reverse_mask = tk.BooleanVar(value=True)  # shade outside ROI on map preview
        self._roi_mask_inv: np.ndarray | None = None        # inverse (outside) mask

        # v5: saved univariate maps  {name: 2-D ndarray}
        self._saved_maps: dict[str, np.ndarray] = {}

        self._build_ui()
        self._style_ttk()
        self._bind_keys()

    # ── TTK style ─────────────────────────────────────────────────────────────
    def _style_ttk(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".",
            background=C["sidebar"], foreground=C["text_hi"],
            fieldbackground="white", selectbackground=C["accent"],
            selectforeground="white", font=("Segoe UI", 10),
            bordercolor=C["border"], troughcolor="#e8ecf4")
        s.configure("TFrame",   background=C["sidebar"])
        s.configure("TLabel",   background=C["sidebar"], foreground=C["text_hi"])
        s.configure("TLabelframe",  background=C["panel"], relief="flat",
                    bordercolor=C["border"])
        s.configure("TLabelframe.Label", background=C["panel"],
                    foreground=C["accent"], font=("Segoe UI", 11, "bold"))
        s.configure("TCombobox", fieldbackground="white", background="white",
                    foreground=C["text_hi"], arrowcolor=C["accent"], padding=4)
        s.configure("Horizontal.TProgressbar",
                    troughcolor="#e8ecf4", background=C["accent"])
        s.configure("TCheckbutton", background=C["sidebar"],
                    foreground=C["text_mid"])
        s.map("TCheckbutton", background=[("active", C["sidebar"])])
        for name, bg, fg in [
            ("Primary.TButton",  C["accent"],  "white"),
            ("Danger.TButton",   C["danger"],  "white"),
            ("Success.TButton",  C["success"], "white"),
            ("Neutral.TButton",  "#e4e8f4",    C["text_hi"]),
            ("ROI.TButton",      C["roi"],     "white"),
        ]:
            s.configure(name, background=bg, foreground=fg, relief="flat",
                        padding=(10, 5), font=("Segoe UI", 11, "bold"),
                        borderwidth=0)
            s.map(name, background=[("active", C["accent2"]),
                                    ("pressed", C["header"])])

    # ── UI shell ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_menubar()
        self._build_toolbar()
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)
        self._build_sidebar(body)
        self._build_plots(body)
        self._build_statusbar()

    # ── menubar ───────────────────────────────────────────────────────────────
    def _build_menubar(self):
        mb = tk.Menu(self, bg=C["panel"], fg=C["text_hi"],
                     activebackground=C["accent"], activeforeground="white",
                     relief="flat")
        self.config(menu=mb)

        fm = tk.Menu(mb, tearoff=0, bg=C["panel"], fg=C["text_hi"],
                     activebackground=C["accent"], activeforeground="white")
        mb.add_cascade(label="File", menu=fm)
        fm.add_command(label="Open WDF…        Ctrl+O", command=self.load_file)
        fm.add_command(label="Load White-Light…",       command=self.load_wl)
        fm.add_separator()
        fm.add_command(label="Save Map…        Ctrl+M", command=self.save_map)
        fm.add_command(label="Save Spectrum…   Ctrl+S", command=self.save_spectrum)
        fm.add_separator()
        fm.add_command(label="Exit",                    command=self.destroy)

        vm = tk.Menu(mb, tearoff=0, bg=C["panel"], fg=C["text_hi"],
                     activebackground=C["accent"], activeforeground="white")
        mb.add_cascade(label="View", menu=vm)
        vm.add_checkbutton(label="Show peak markers",
                           variable=self._show_peaks,
                           command=self._redraw_spectrum)
        vm.add_checkbutton(label="Normalise spectrum",
                           variable=self._norm_var,
                           command=self._redraw_spectrum)

        pm = tk.Menu(mb, tearoff=0, bg=C["panel"], fg=C["text_hi"],
                     activebackground=C["accent"], activeforeground="white")
        mb.add_cascade(label="Preprocessing", menu=pm)
        pm.add_command(label="⚙  Settings…",        command=self.open_pp_settings)
        pm.add_command(label="📋  Processing Log…",  command=self.show_pp_report)

        am = tk.Menu(mb, tearoff=0, bg=C["panel"], fg=C["text_hi"],
                     activebackground=C["accent"], activeforeground="white")
        mb.add_cascade(label="Analysis", menu=am)
        am.add_command(label="◈  PCA Analysis…",         command=self.open_pca)
        am.add_command(label="🧊  3D Volume Viewer…",     command=self.open_3d_viewer)
        am.add_separator()
        am.add_command(label="⬡  Cluster Analysis…",     command=self.open_clustering)
        am.add_command(label="⟠  MCR-ALS…",              command=self.open_mcr)
        am.add_command(label="◉  N-FINDR Endmembers…",   command=self.open_nfindr)
        am.add_separator()
        am.add_command(label="⚒  Spectral Tools…",       command=self.open_spectral_tools)

        hm = tk.Menu(mb, tearoff=0, bg=C["panel"], fg=C["text_hi"],
                     activebackground=C["accent"], activeforeground="white")
        mb.add_cascade(label="Help", menu=hm)
        hm.add_command(label="About", command=lambda: messagebox.showinfo(
            "Raman Map Explorer v4",
            "Professional Raman Map Analysis\n\nShortcuts:\n"
            "  Ctrl+O  Open WDF\n  Ctrl+M  Save map\n"
            "  Ctrl+S  Save spectrum\n"
            "  Right-click map → Comparison spectrum\n\n"
            "ROI: draw on map, mean spectrum shown automatically"))

    # ── toolbar ───────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        # Two-row toolbar; keep MODE/CMAP always visible by pinning them to the right.
        tb_h = 108
        tb = tk.Frame(self, bg=C["panel"], height=tb_h,
                      highlightthickness=1, highlightbackground=C["border"])
        tb.pack(fill="x")
        tb.pack_propagate(False)

        # Left logo block spans both rows
        logo = tk.Frame(tb, bg=C["header"], width=210, height=tb_h)
        logo.pack(side="left", fill="y")
        logo.pack_propagate(False)
        tk.Label(logo, text="  ◈ RAMAN EXPLORER",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=8, pady=32)

        tk.Frame(tb, bg=C["border"], width=1).pack(side="left", fill="y", pady=10)

        right = tk.Frame(tb, bg=C["panel"])
        right.pack(side="left", fill="both", expand=True, padx=(8, 6))

        row1 = tk.Frame(right, bg=C["panel"])
        row2 = tk.Frame(right, bg=C["panel"])
        row1.pack(fill="x", pady=(10, 3))
        row2.pack(fill="x", pady=(0, 10))

        # Row 2 is split: actions on the left, MODE/CMAP pinned on the right.
        row2_actions  = tk.Frame(row2, bg=C["panel"])
        row2_controls = tk.Frame(row2, bg=C["panel"])
        row2_controls.pack(side="right", padx=(6, 2))
        row2_actions.pack(side="left", fill="x", expand=True)

        actions = [
            ("⊕ Open WDF",      self.load_file,              "Primary.TButton"),
            ("⚙ Preprocess",   self.open_pp_settings,       "Neutral.TButton"),
            ("⊞ White Light",   self.load_wl,                "Neutral.TButton"),
            ("◈ PCA",          self.open_pca,               "Neutral.TButton"),
            ("🧊 3D Volume",    self.open_3d_viewer,         "Neutral.TButton"),
            ("📊 Univariate",   self.open_univariate,        "Neutral.TButton"),
            ("⚡ Dynamic",      self.open_dynamic_map,       "Neutral.TButton"),
            ("~ Curve Fit",     self.open_curve_fit_map,     "Neutral.TButton"),
            ("÷ Ratio",         self.open_ratio_map,         "Neutral.TButton"),
            ("🎨 LUT",          self.open_lut_control,       "Neutral.TButton"),
            ("↔ Profiles",      self.open_line_profiles,     "Neutral.TButton"),
            ("🔬 ROI Analysis", self.open_roi_analysis,      "ROI.TButton"),
            ("⬡ Cluster",      self.open_clustering,        "Neutral.TButton"),
            ("⟠ MCR-ALS",      self.open_mcr,               "Neutral.TButton"),
            ("◉ N-FINDR",      self.open_nfindr,            "Neutral.TButton"),
            ("⚒ Spectral Tools",self.open_spectral_tools,   "Neutral.TButton"),
            ("↓ Save Map",      self.save_map,               "Neutral.TButton"),
            ("↓ Save Spec",     self.save_spectrum,          "Neutral.TButton"),
        ]

        split = (len(actions) + 1) // 2
        for k, (t, cmd, style) in enumerate(actions):
            parent = row1 if k < split else row2_actions
            ttk.Button(parent, text=t, command=cmd, style=style).pack(side="left", padx=3, pady=0)

        # MODE/CMAP pinned at far right
        tk.Label(row2_controls, text="MODE:", bg=C["panel"], fg=C["text_dim"],
                 font=("Segoe UI", 11, "bold")).pack(side="left", padx=(2, 4))
        self.mode_var = tk.StringVar(value="ratio")
        for val, txt in [("ratio", "Ratio"), ("rgb", "A+B RGB"), ("wl", "White Light")]:
            tk.Radiobutton(row2_controls, text=txt, variable=self.mode_var, value=val,
                           bg=C["panel"], fg=C["text_hi"], activebackground=C["panel"],
                           selectcolor=C["panel"], font=("Segoe UI", 11),
                           command=self.update_map).pack(side="left", padx=4)

        tk.Frame(row2_controls, bg=C["border"], width=1).pack(side="left", fill="y", padx=10, pady=2)

        tk.Label(row2_controls, text="CMAP:", bg=C["panel"], fg=C["text_dim"],
                 font=("Segoe UI", 11, "bold")).pack(side="left", padx=(2, 4))
        self.cmap_var = tk.StringVar(value="turbo")
        cmap_cb = ttk.Combobox(row2_controls, textvariable=self.cmap_var,
                               values=COLORMAPS, state="readonly", width=10)
        cmap_cb.pack(side="left", padx=(0, 4))
        cmap_cb.bind("<<ComboboxSelected>>", lambda _: self.update_map())

        # Progress (far right)
        self._prog_frame = tk.Frame(tb, bg=C["panel"])
        self._prog_frame.pack(side="right", padx=12, pady=38)
        self.progress  = ttk.Progressbar(self._prog_frame, mode="determinate", length=160)
        self._prog_lbl = tk.Label(self._prog_frame, text="Processing…",
                                  bg=C["panel"], fg=C["text_dim"], font=("Segoe UI", 10))

    def _show_progress(self, show=True):
        if show:
            self._prog_lbl.pack(side="left")
            self.progress.pack(side="left", padx=(4,0))
        else:
            self._prog_lbl.pack_forget()
            self.progress.pack_forget()

    # ── sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        outer = tk.Frame(parent, bg=C["sidebar"], width=290,
                         highlightthickness=1, highlightbackground=C["border"])
        outer.pack(side="left", fill="y"); outer.pack_propagate(False)

        scv = tk.Canvas(outer, bg=C["sidebar"], highlightthickness=0)
        sb  = ttk.Scrollbar(outer, orient="vertical", command=scv.yview)
        scv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); scv.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(scv, bg=C["sidebar"])
        wid = scv.create_window((0,0), window=inner, anchor="nw")
        scv.bind("<Configure>", lambda e: scv.itemconfig(wid, width=e.width))
        inner.bind("<Configure>", lambda e: scv.configure(
            scrollregion=scv.bbox("all")))
        scv.bind_all("<MouseWheel>", lambda e: scv.yview_scroll(
            int(-1*(e.delta/120)), "units"))

        p = inner
        pad = {"fill":"x"}

        # ── Band A ────────────────────────────────────────────────────────
        SectionDiv(p, "BAND A  (cm⁻¹)").pack(**pad)
        self.rs_a = RangeSlider(p, "Range", 0, 4000, 1300, 1350,
                                color=C["band_a"], resolution=5,
                                command=self._on_band_change)
        self.rs_a.pack(**pad, padx=4, pady=2)

        # ── Band B ────────────────────────────────────────────────────────
        SectionDiv(p, "BAND B  (cm⁻¹)").pack(**pad)
        self.rs_b = RangeSlider(p, "Range", 0, 4000, 1580, 1630,
                                color=C["band_b"], resolution=5,
                                command=self._on_band_change)
        self.rs_b.pack(**pad, padx=4, pady=2)

        # ── Map settings ──────────────────────────────────────────────────
        SectionDiv(p, "MAP SETTINGS").pack(**pad)
        self.sl_sigma = SingleSlider(p, "Gaussian smoothing (σ px)",
                                     0, 8, 1.5, color=C["accent"],
                                     resolution=0.5, command=self.update_map)
        self.sl_sigma.pack(**pad, padx=4)

        clim_row = tk.Frame(p, bg=C["sidebar"])
        clim_row.pack(**pad, padx=12, pady=(4,0))
        self._auto_clim = tk.BooleanVar(value=True)
        tk.Checkbutton(clim_row, text="Auto colour limits",
                       variable=self._auto_clim,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10),
                       command=self.update_map).pack(side="left")

        self.sl_vmin = SingleSlider(p, "vmin", 0, 10, 0.0,
                                    color=C["success"], resolution=0.05,
                                    command=self._manual_clim)
        self.sl_vmin.pack(**pad, padx=4)
        self.sl_vmax = SingleSlider(p, "vmax", 0, 10, 2.0,
                                    color=C["danger"], resolution=0.05,
                                    command=self._manual_clim)
        self.sl_vmax.pack(**pad, padx=4)

        # ── ROI ───────────────────────────────────────────────────────────
        SectionDiv(p, "ROI — REGION OF INTEREST").pack(**pad)
        roi_card = tk.Frame(p, bg=C["sidebar"])
        roi_card.pack(**pad, padx=8, pady=2)

        tk.Label(roi_card, text="Shape", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(anchor="w", padx=4, pady=(4,2))

        mode_row = tk.Frame(roi_card, bg=C["sidebar"])
        mode_row.pack(fill="x", padx=4)
        for val, txt in [("rectangle","Rect"),("ellipse","Ellipse"),
                         ("polygon","Polygon"),("freehand","Freehand")]:
            tk.Radiobutton(mode_row, text=txt, variable=self._roi_mode,
                           value=val, bg=C["sidebar"], fg=C["text_hi"],
                           activebackground=C["sidebar"],
                           selectcolor=C["sidebar"],
                           font=("Segoe UI", 11)).pack(side="left", padx=4)

        tk.Checkbutton(roi_card, text="Mask outside ROI (inverse mask)",
                       variable=self._roi_reverse_mask,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       selectcolor=C["sidebar"],
                       font=("Segoe UI", 11)).pack(anchor="w", padx=4, pady=(6, 2))
        btn_row = tk.Frame(roi_card, bg=C["sidebar"])
        btn_row.pack(fill="x", padx=4, pady=4)
        ttk.Button(btn_row, text="✎ Draw ROI", style="ROI.TButton",
                   command=self._start_roi).pack(side="left", padx=2)
        ttk.Button(btn_row, text="✕ Clear", style="Neutral.TButton",
                   command=self._clear_roi).pack(side="left", padx=2)

        self._roi_info = tk.Label(roi_card, text="No ROI defined",
                                  bg=C["sidebar"], fg=C["text_dim"],
                                  font=("Segoe UI", 11))
        self._roi_info.pack(anchor="w", padx=4, pady=(0,2))

        # Analyse button — prominent, highlighted
        ttk.Button(roi_card, text="🔬 Analyse ROI",
                   style="ROI.TButton",
                   command=self.open_roi_analysis).pack(
                       fill="x", padx=4, pady=(2, 6))

        # ── White light ───────────────────────────────────────────────────
        SectionDiv(p, "WHITE LIGHT OVERLAY").pack(**pad)
        wl_card = tk.Frame(p, bg=C["sidebar"])
        wl_card.pack(**pad, padx=8, pady=2)
        ttk.Button(wl_card, text="Load Image…", style="Neutral.TButton",
                   command=self.load_wl).pack(fill="x", padx=4, pady=(6,2))
        self._wl_name = tk.Label(wl_card, text="No image loaded",
                                 bg=C["sidebar"], fg=C["text_dim"],
                                 font=("Segoe UI", 11))
        self._wl_name.pack(anchor="w", padx=6, pady=(0,4))
        self._wl_thumb = tk.Label(wl_card, bg=C["sidebar"])
        self._wl_thumb.pack(pady=(0,4))
        self.sl_wl_alpha = SingleSlider(wl_card, "Overlay opacity",
                                         0, 1, 0.45, color=C["accent2"],
                                         resolution=0.05, command=self.update_map)
        self.sl_wl_alpha.pack(**pad)
        self.sl_wl_bright = SingleSlider(wl_card, "WL brightness",
                                          0.1, 3.0, 1.0, color=C["band_a"],
                                          resolution=0.05, command=self.update_map)
        self.sl_wl_bright.pack(**pad)

        # ── Spectrum options ──────────────────────────────────────────────
        SectionDiv(p, "SPECTRUM OPTIONS").pack(**pad)
        spec_card = tk.Frame(p, bg=C["sidebar"])
        spec_card.pack(**pad, padx=8, pady=2)
        tk.Checkbutton(spec_card, text="Normalise intensity",
                       variable=self._norm_var,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10),
                       command=self._redraw_spectrum).pack(anchor="w", padx=4)
        tk.Checkbutton(spec_card, text="Annotate peaks",
                       variable=self._show_peaks,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10),
                       command=self._redraw_spectrum).pack(anchor="w", padx=4)
        ttk.Button(spec_card, text="Clear comparison", style="Neutral.TButton",
                   command=self._clear_compare).pack(fill="x", padx=4, pady=6)

    # ── plots ─────────────────────────────────────────────────────────────────
    def _build_plots(self, parent):
        container = tk.Frame(parent, bg=C["bg"])
        container.pack(side="left", fill="both", expand=True)

        self.fig = plt.Figure(figsize=(11, 5.8),
                              facecolor=matplotlib.rcParams["figure.facecolor"])
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.1], wspace=0.38,
                                   left=0.06, right=0.97, top=0.93, bottom=0.10)
        self.ax_map  = self.fig.add_subplot(gs[0])
        self.ax_spec = self.fig.add_subplot(gs[1])
        self._init_map_ax()
        self._init_spec_ax()

        self.canvas = FigureCanvasTkAgg(self.fig, master=container)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        tb_frame = tk.Frame(container, bg=C["panel"],
                            highlightthickness=1, highlightbackground=C["border"])
        tb_frame.pack(fill="x")
        nav = NavigationToolbar2Tk(self.canvas, tb_frame)
        nav.config(background=C["panel"])
        for w in nav.winfo_children():
            try: w.config(background=C["panel"], foreground=C["text_mid"])
            except: pass
        nav.update()

        self.canvas.mpl_connect("button_press_event",  self._click)
        self.canvas.mpl_connect("motion_notify_event", self._hover)

    def _init_map_ax(self):
        ax = self.ax_map
        ax.set_title("Intensity Map", fontweight="semibold")
        ax.set_xlabel("X  (pixels)", fontsize=10)
        ax.set_ylabel("Y  (pixels)", fontsize=10)
        ax.tick_params(which="both", direction="in", length=3)
        self.im = ax.imshow(np.zeros((10,10)), origin="upper",
                            aspect="equal", interpolation="bilinear", cmap="turbo")
        self.cbar = self.fig.colorbar(self.im, ax=ax,
                                      fraction=0.046, pad=0.03, shrink=0.85)
        self.cbar.ax.tick_params(labelsize=8)
        self.cbar.set_label("A / B  ratio", fontsize=9)
        self.xhair_v, = ax.plot([], [], color="#ff3344", lw=0.8,
                                ls="--", zorder=10)
        self.xhair_h, = ax.plot([], [], color="#ff3344", lw=0.8,
                                ls="--", zorder=10)
        self.xhair_pt,= ax.plot([], [], "o", ms=6, mec="#ff3344",
                                mfc="none", mew=1.5, zorder=11)
        self._roi_patch = None

    def _init_spec_ax(self):
        ax = self.ax_spec
        ax.set_title("Raman Spectrum", fontweight="semibold")
        ax.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=10)
        ax.set_ylabel("Intensity  (a.u.)", fontsize=10)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which="both", direction="in", length=3)
        ax.grid(True, which="major", lw=0.5, color="#dde2ee")
        ax.grid(True, which="minor", lw=0.3, color="#edf0f8")
        self.spec_line,    = ax.plot([], [], lw=1.4, color=C["spec_line"],
                                     label="Selected")
        self.spec_compare, = ax.plot([], [], lw=1.1, color=C["compare"],
                                     ls="--", label="Compare", visible=False)
        self.spec_roi,     = ax.plot([], [], lw=1.4, color=C["roi"],
                                     ls="-.", label="ROI mean", visible=False)
        self._band_a_span  = ax.axvspan(0,0, alpha=0.13, color=C["band_a"],
                                        label="Band A")
        self._band_b_span  = ax.axvspan(0,0, alpha=0.13, color=C["band_b"],
                                        label="Band B")
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    # ── status bar ────────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = tk.Frame(self, bg=C["panel"], height=26,
                       highlightthickness=1, highlightbackground=C["border"])
        bar.pack(fill="x", side="bottom"); bar.pack_propagate(False)
        self._status = tk.StringVar(value="Ready  —  open a WDF file to begin")
        tk.Label(bar, textvariable=self._status,
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Segoe UI", 11), anchor="w").pack(side="left", padx=10)
        self._hover_info = tk.StringVar()
        tk.Label(bar, textvariable=self._hover_info,
                 bg=C["panel"], fg=C["accent"],
                 font=("Consolas", 9), anchor="e").pack(side="right", padx=10)

    # ── keyboard shortcuts ────────────────────────────────────────────────────
    def _bind_keys(self):
        self.bind("<Control-o>", lambda _: self.load_file())
        self.bind("<Control-s>", lambda _: self.save_spectrum())
        self.bind("<Control-m>", lambda _: self.save_map())
        self.bind("<Escape>",    lambda _: self._cancel_roi())

    # ── band slider callbacks ─────────────────────────────────────────────────
    def _on_band_change(self):
        self._rebuild_band_spans()
        self.update_map()

    def _rebuild_band_spans(self):
        if self.xdata is None: return
        self._band_a_span.remove()
        self._band_b_span.remove()
        al,ah = self.rs_a.low, self.rs_a.high
        bl,bh = self.rs_b.low, self.rs_b.high
        self._band_a_span = self.ax_spec.axvspan(al,ah, alpha=0.13, color=C["band_a"])
        self._band_b_span = self.ax_spec.axvspan(bl,bh, alpha=0.13, color=C["band_b"])
        self.canvas.draw_idle()

    def _manual_clim(self):
        self._auto_clim.set(False)
        self.update_map()

    # ── ROI ───────────────────────────────────────────────────────────────────
    def _start_roi(self):
        if self.spectra is None:
            messagebox.showwarning("No data","Load a WDF file first."); return
        if self._roi_manager:
            self._roi_manager.deactivate()
        Y, X, _ = self.spectra.shape
        self._roi_manager = ROIManager(
            self.ax_map, self.canvas, ZOOM,
            callback=self._on_roi_done)
        self._roi_manager.activate(self._roi_mode.get(), (Y, X))
        mode = self._roi_mode.get()
        mode = self._roi_mode.get()
        if mode == "polygon":
            hint = "Right-click to finish"
        elif mode == "freehand":
            hint = "Click and drag to trace ROI; release to finish"
        else:
            hint = "Click and drag to draw ROI"
        self._status.set(f"ROI mode: {mode}  —  {hint}  (Esc to cancel)")
        self._roi_info.config(text=f"Drawing {mode}…", fg=C["roi"])

    def _cancel_roi(self):
        if self._roi_manager:
            self._roi_manager.deactivate()
            self._roi_manager = None
        self._status.set("ROI cancelled")
        self._roi_info.config(text="No ROI defined", fg=C["text_dim"])

    def _clear_roi(self):
        self._cancel_roi()
        self._roi_mask = None
        self.spec_roi.set_visible(False)
        if self._roi_patch:
            try: self._roi_patch.remove()
            except: pass
            self._roi_patch = None
        self.canvas.draw_idle()
        self._roi_info.config(text="No ROI defined", fg=C["text_dim"])

    def _on_roi_done(self, mask):
        if self._roi_manager:
            self._roi_manager.deactivate()
            self._roi_manager = None
        self._roi_mask = mask          # ← store for ROI Analysis
        n_px = int(mask.sum())
        self._roi_info.config(
            text=f"{n_px} pixels selected", fg=C["success"])
        self._status.set(f"ROI: {n_px} pixels — computing mean spectrum…")

        # Draw filled ROI overlay on map
        if self._roi_patch:
            try: self._roi_patch.remove()
            except: pass
        Y, X = mask.shape
        rgba = np.zeros((Y * ZOOM, X * ZOOM, 4), dtype=np.float32)
        from scipy.ndimage import zoom as _zoom
        mask_z = _zoom(mask.astype(float), ZOOM, order=0) > 0.5
        self._roi_mask_inv = ~mask
        if self._roi_reverse_mask.get():
            out_z = ~mask_z
            rgba[out_z, :3] = (0.10, 0.10, 0.10)
            rgba[out_z,  3] = 0.55
            rgba[mask_z, :3] = (1.00, 0.42, 0.21)
            rgba[mask_z,  3] = 0.10
        else:
            rgba[mask_z, 0] = 1.0
            rgba[mask_z, 1] = 0.42
            rgba[mask_z, 2] = 0.21
            rgba[mask_z, 3] = 0.28
        self._roi_patch = self.ax_map.imshow(
            rgba, origin="upper", aspect="equal",
            extent=(0, X*ZOOM, Y*ZOOM, 0), zorder=5)
        self.canvas.draw_idle()

        # Compute mean spectrum of ROI
        Y2, X2, W = self.spectra.shape
        roi_spectra = self.spectra[mask]        # n_px × W
        mean_spec   = roi_spectra.mean(axis=0)

        if self._norm_var.get():
            pk = mean_spec.max()
            if pk > 0: mean_spec /= pk

        self.spec_roi.set_data(self.xdata, mean_spec)
        self.spec_roi.set_visible(True)
        self.ax_spec.relim(); self.ax_spec.autoscale_view()
        self.ax_spec.legend(loc="upper right", frameon=True, fontsize=9)
        self.canvas.draw_idle()
        self._status.set(
            f"ROI: {n_px} pixels selected — click '🔬 ROI Analysis' for full workflow")
        # offer analysis
        self._roi_info.config(
            text=f"✓ {n_px} px  —  click 🔬 Analyse", fg=C["success"])

    # ── ROI ANALYSIS WINDOW ───────────────────────────────────────────────────
    def open_roi_analysis(self):
        """
        Full ROI Analysis workflow (replicating Figure 2 of Raman cell-freezing paper):
          1. Preview ROI mask on the map
          2. Binarize the ROI using an intensity threshold
          3. Define two spectral bands (e.g. proteins/lipids + ice)
          4. Compute per-pixel peak areas within the binarised cellular region
          5. Show: binarized image, smoothed heatmaps for Band A & B,
                   masked heatmap, and mean spectrum with annotated bands
          6. Report pixel counts (N_p, N_p') and band statistics
        """
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first."); return
        mask = getattr(self, "_roi_mask", None)
        if mask is None:
            messagebox.showinfo("No ROI",
                "Draw an ROI on the map first (sidebar → ✎ Draw ROI)."); return

        # ── Build window ──────────────────────────────────────────────────────
        win = tk.Toplevel(self)
        win.title("ROI Analysis — Raman Spectroscopic Analysis")
        win.geometry("1180x820")
        win.minsize(900, 640)
        win.configure(bg=C["bg"])

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(win, bg=C["header"], height=50)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🔬  ROI ANALYSIS — Raman Spectroscopic Workflow",
                 bg=C["header"], fg="white",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=18, pady=14)
        tk.Label(hdr,
                 text="Define bands → Binarize → Heatmaps → Statistics",
                 bg=C["header"], fg="#94a3b8",
                 font=("Segoe UI", 11)).pack(side="right", padx=18)

        # ── Left panel: controls ──────────────────────────────────────────────
        left = tk.Frame(win, bg=C["sidebar"], width=300)
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        ctrl_cv = tk.Canvas(left, bg=C["sidebar"], highlightthickness=0)
        ctrl_sb = ttk.Scrollbar(left, orient="vertical", command=ctrl_cv.yview)
        ctrl_cv.configure(yscrollcommand=ctrl_sb.set)
        ctrl_sb.pack(side="right", fill="y")
        ctrl_cv.pack(side="left", fill="both", expand=True)
        ctrl = tk.Frame(ctrl_cv, bg=C["sidebar"])
        ctrl_win = ctrl_cv.create_window((0, 0), window=ctrl, anchor="nw")
        ctrl_cv.bind("<Configure>",
                     lambda e: ctrl_cv.itemconfig(ctrl_win, width=e.width))
        ctrl.bind("<Configure>",
                  lambda e: ctrl_cv.configure(scrollregion=ctrl_cv.bbox("all")))

        def _sec(text):
            SectionDiv(ctrl, text).pack(fill="x")

        def _row(parent, label, widget_fn, pady=4):
            f = tk.Frame(parent, bg=C["sidebar"])
            f.pack(fill="x", padx=10, pady=pady)
            tk.Label(f, text=label, width=22, anchor="w",
                     bg=C["sidebar"], fg=C["text_mid"],
                     font=("Segoe UI", 11)).pack(side="left")
            w = widget_fn(f); w.pack(side="left", padx=4)
            return w

        lo = float(self.xdata.min()); hi = float(self.xdata.max())

        # ── Band A ────────────────────────────────────────────────────────────
        _sec("BAND A  (e.g. Proteins/Lipids)")
        card_a = tk.Frame(ctrl, bg=C["panel"],
                          highlightthickness=1, highlightbackground=C["border"])
        card_a.pack(fill="x", padx=8, pady=3)

        a_lo_var = tk.DoubleVar(value=1610)
        a_hi_var = tk.DoubleVar(value=1710)
        _row(card_a, "Low (cm⁻¹)",
             lambda f: ttk.Spinbox(f, from_=lo, to=hi, increment=5,
                                   textvariable=a_lo_var, width=9))
        _row(card_a, "High (cm⁻¹)",
             lambda f: ttk.Spinbox(f, from_=lo, to=hi, increment=5,
                                   textvariable=a_hi_var, width=9))
        a_label_var = tk.StringVar(value="Amide I (proteins/lipids)")
        _row(card_a, "Label",
             lambda f: tk.Entry(f, textvariable=a_label_var, width=18,
                                bg="white", font=("Segoe UI", 11), relief="flat",
                                highlightthickness=1,
                                highlightbackground=C["border"]))

        # ── Band B ────────────────────────────────────────────────────────────
        _sec("BAND B  (e.g. Ice / Water)")
        card_b = tk.Frame(ctrl, bg=C["panel"],
                          highlightthickness=1, highlightbackground=C["border"])
        card_b.pack(fill="x", padx=8, pady=3)

        b_lo_var = tk.DoubleVar(value=3087)
        b_hi_var = tk.DoubleVar(value=3162)
        _row(card_b, "Low (cm⁻¹)",
             lambda f: ttk.Spinbox(f, from_=lo, to=hi, increment=5,
                                   textvariable=b_lo_var, width=9))
        _row(card_b, "High (cm⁻¹)",
             lambda f: ttk.Spinbox(f, from_=lo, to=hi, increment=5,
                                   textvariable=b_hi_var, width=9))
        b_label_var = tk.StringVar(value="Ice (OH stretch)")
        _row(card_b, "Label",
             lambda f: tk.Entry(f, textvariable=b_label_var, width=18,
                                bg="white", font=("Segoe UI", 11), relief="flat",
                                highlightthickness=1,
                                highlightbackground=C["border"]))

        # ── Binarization ──────────────────────────────────────────────────────
        _sec("BINARIZATION")
        card_bin = tk.Frame(ctrl, bg=C["panel"],
                            highlightthickness=1, highlightbackground=C["border"])
        card_bin.pack(fill="x", padx=8, pady=3)

        binarize_band = tk.StringVar(value="Band A")
        _row(card_bin, "Binarize from",
             lambda f: ttk.Combobox(f, textvariable=binarize_band,
                                    values=["Band A", "Band B", "Mean spectrum"],
                                    state="readonly", width=14))

        threshold_mode = tk.StringVar(value="manual")
        _row(card_bin, "Threshold method",
             lambda f: ttk.Combobox(f, textvariable=threshold_mode,
                                    values=["otsu", "manual", "percentile"],
                                    state="readonly", width=10))

        thresh_var = tk.DoubleVar(value=1e-5)
        thresh_spin = _row(card_bin, "Manual threshold",
                           lambda f: ttk.Spinbox(f, from_=0, to=1e6,
                                                 increment=1e-6, format="%.2e",
                                                 textvariable=thresh_var, width=12))

        def _auto_set_threshold():
            """Set threshold to 30th percentile of Band A within the ROI (good start for cells)."""
            try:
                al2 = a_lo_var.get(); ah2 = a_hi_var.get()
                tmp = _peak_area_map(al2, ah2,
                                     gain=gain_a_var.get(),
                                     rolling_bl=rolling_bl_var.get())
                roi_vals = tmp[mask]
                roi_vals = roi_vals[np.isfinite(roi_vals) & (roi_vals > 0)]
                if roi_vals.size > 0:
                    suggested = float(np.percentile(roi_vals, 30))
                    thresh_var.set(round(suggested, 10))
                    threshold_mode.set("manual")
            except Exception:
                pass

        auto_thr_btn = tk.Button(
            card_bin, text="Auto-set threshold (30th pct of Band A)",
            bg=C["accent"], fg="white",
            font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2",
            command=_auto_set_threshold)
        auto_thr_btn.pack(fill="x", padx=10, pady=(0, 6))

        pct_var = tk.IntVar(value=40)
        _row(card_bin, "Percentile cutoff",
             lambda f: ttk.Spinbox(f, from_=1, to=99, increment=1,
                                   textvariable=pct_var, width=6))

        # ── Smoothing ─────────────────────────────────────────────────────────
        _sec("SMOOTHING")
        card_sm = tk.Frame(ctrl, bg=C["panel"],
                           highlightthickness=1, highlightbackground=C["border"])
        card_sm.pack(fill="x", padx=8, pady=3)

        sigma_var = tk.DoubleVar(value=3.0)
        _row(card_sm, "Gaussian σ (px)",
             lambda f: ttk.Spinbox(f, from_=0, to=10, increment=0.5,
                                   textvariable=sigma_var, width=7))

        cmap_a_var = tk.StringVar(value="magma")
        _row(card_sm, "Heatmap cmap A",
             lambda f: ttk.Combobox(f, textvariable=cmap_a_var,
                                    values=COLORMAPS, state="readonly", width=10))
        cmap_b_var = tk.StringVar(value="hot")
        _row(card_sm, "Heatmap cmap B",
             lambda f: ttk.Combobox(f, textvariable=cmap_b_var,
                                    values=COLORMAPS, state="readonly", width=10))

        # ── Sensitivity (protein boost / ice suppression) ──────────────────────
        _sec("SENSITIVITY")
        card_sens = tk.Frame(ctrl, bg=C["panel"],
                             highlightthickness=1, highlightbackground=C["border"])
        card_sens.pack(fill="x", padx=8, pady=3)

        gain_a_var = tk.DoubleVar(value=1.0)
        _row(card_sens, "Band A gain (protein)",
             lambda f: ttk.Spinbox(f, from_=0.1, to=50.0, increment=0.5,
                                   format="%.1f",
                                   textvariable=gain_a_var, width=7))

        gain_b_var = tk.DoubleVar(value=1.0)
        _row(card_sens, "Band B gain (ice)",
             lambda f: ttk.Spinbox(f, from_=0.1, to=50.0, increment=0.5,
                                   format="%.1f",
                                   textvariable=gain_b_var, width=7))

        clamp_b_var = tk.DoubleVar(value=1.0)
        _row(card_sens, "Band B max clamp (0–1)",
             lambda f: ttk.Spinbox(f, from_=0.0, to=1.0, increment=0.05,
                                   format="%.2f",
                                   textvariable=clamp_b_var, width=7))
        tk.Label(card_sens,
                 text="  Clamp <1 to suppress spurious ice signal in cells.\n"
                      "  e.g. 0.3 keeps only top 30 % of Band B values.",
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Segoe UI", 9), justify="left").pack(
                     anchor="w", padx=10, pady=(0, 6))

        rolling_bl_var = tk.BooleanVar(value=True)
        f_rb = tk.Frame(card_sens, bg=C["panel"])
        f_rb.pack(fill="x", padx=10, pady=(0, 6))
        tk.Checkbutton(f_rb, text="Rubber-band baseline (recommended for weak bands)",
                       variable=rolling_bl_var,
                       bg=C["panel"], fg=C["text_hi"],
                       activebackground=C["panel"],
                       selectcolor=C["accent"],
                       font=("Segoe UI", 10)).pack(anchor="w")

        # ── Run button + stats ────────────────────────────────────────────────
        _sec("RESULTS")
        run_btn = ttk.Button(ctrl, text="▶  Run Analysis", style="Primary.TButton")
        run_btn.pack(fill="x", padx=10, pady=(6, 3))
        save_btn = ttk.Button(ctrl, text="💾  Export Figure", style="Neutral.TButton")
        save_btn.pack(fill="x", padx=10, pady=3)
        panels_btn = ttk.Button(ctrl, text="🗂  Export Panels", style="Neutral.TButton")
        panels_btn.pack(fill="x", padx=10, pady=3)

        stats_frame = tk.Frame(ctrl, bg=C["panel"],
                               highlightthickness=1, highlightbackground=C["border"])
        stats_frame.pack(fill="x", padx=8, pady=6)
        stats_lbl = tk.Label(stats_frame, text="Run analysis to see statistics.",
                             bg=C["panel"], fg=C["text_dim"],
                             font=("Segoe UI", 11), wraplength=260,
                             justify="left", anchor="w")
        stats_lbl.pack(fill="x", padx=8, pady=6)

        # ── Right panel: figure ───────────────────────────────────────────────
        right = tk.Frame(win, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        fig = plt.Figure(figsize=(10.5, 7.2),
                         facecolor=matplotlib.rcParams["figure.facecolor"])
        # Layout: 2 rows × 3 cols
        # Row 0: [Bright-field / ROI mask] [Raw heatmap A] [Raw heatmap B]
        # Row 1: [Mean spectrum + bands]   [Binarized]     [Masked heatmap B in A]
        gs = fig.add_gridspec(2, 3, wspace=0.35, hspace=0.42,
                              left=0.06, right=0.97, top=0.94, bottom=0.09)
        ax_roi   = fig.add_subplot(gs[0, 0])   # ROI mask
        ax_raw_a = fig.add_subplot(gs[0, 1])   # Smoothed heatmap Band A
        ax_raw_b = fig.add_subplot(gs[0, 2])   # Smoothed heatmap Band B
        ax_spec  = fig.add_subplot(gs[1, 0])   # Mean spectrum
        ax_bin   = fig.add_subplot(gs[1, 1])   # Binarized (cellular region)
        ax_mask  = fig.add_subplot(gs[1, 2])   # Band B inside cellular region

        for ax, title in [
            (ax_roi,   "ROI Selection"),
            (ax_raw_a, "Band A  (raw)"),
            (ax_raw_b, "Band B  (raw)"),
            (ax_spec,  "Mean ROI Spectrum"),
            (ax_bin,   "Binarized  (cellular)"),
            (ax_mask,  "Band B within cellular region"),
        ]:
            ax.set_title(title, fontsize=9, fontweight="semibold")
            if ax != ax_spec:
                ax.set_xticks([]); ax.set_yticks([])

        canvas_r = FigureCanvasTkAgg(fig, master=right)
        canvas_r.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas_r, right).update()

        # Pre-draw ROI mask preview
        Y, X, W = self.spectra.shape
        roi_rgba = np.zeros((Y, X, 4), dtype=np.float32)
        roi_rgba[mask, :3] = [1.0, 0.65, 0.0]   # orange for selected
        roi_rgba[mask,  3] = 0.8
        roi_rgba[~mask, :3] = [0.1, 0.1, 0.1]
        roi_rgba[~mask,  3] = 0.6
        ax_roi.imshow(roi_rgba, origin="upper", aspect="equal")
        ax_roi.set_title(f"ROI Selection  ({int(mask.sum())} px)", fontsize=9,
                         fontweight="semibold")
        canvas_r.draw_idle()

        # ── Core analysis function ─────────────────────────────────────────────
        _fig_ref = [fig]   # mutable ref for export

        def _peak_area_map(lo_wn, hi_wn, gain=1.0, rolling_bl=True):
            """Signal-to-baseline integrated area map for the full data cube.

            Baseline strategy
            -----------------
            Narrow band (< 150 cm-1, e.g. ice 3087-3162): always uses the
            simple linear (endpoint) baseline.  A rolling-minimum window on a
            narrow slice of the broad OH envelope over-subtracts and produces
            negative areas.

            Wide band (>=150 cm-1, e.g. Amide I 1610-1710) and rolling_bl=True:
            uses an iterative SNIP-lite rubber-band baseline.  The anchor
            constraint ensures the baseline never rises above the data, so
            areas are always >= 0.

            Parameters
            ----------
            lo_wn, hi_wn : float   Wavenumber integration limits (cm-1).
            gain : float           Multiplicative sensitivity boost (>1 for weak bands).
            rolling_bl : bool      Enable rubber-band baseline for wide bands.
            """
            m = (self.xdata >= lo_wn) & (self.xdata <= hi_wn)
            if not m.any():
                return np.zeros((Y, X))
            sub = self.spectra[:, :, m].astype(float)   # Y x X x W_sub
            n_pts = m.sum()
            band_width_cm = hi_wn - lo_wn

            # Baseline strategy:
            #  Wide band (>=150 cm-1) + rubber_bl ON -> SNIP-lite iterative minimum.
            #  Narrow band (<150 cm-1) OR rubber_bl OFF -> linear endpoint baseline.
            #  In ALL cases we also subtract the per-band minimum so area is >= 0.
            use_rubber = rolling_bl and n_pts >= 6 and band_width_cm >= 150

            if use_rubber:
                # SNIP-lite: iterative half-way averaging guarantees bl <= data
                bl = sub.copy()
                for hw in [max(2, n_pts // 8), max(3, n_pts // 5), max(4, n_pts // 3)]:
                    sl = np.roll(bl, -hw, axis=2)
                    sr = np.roll(bl,  hw, axis=2)
                    mid = (sl + sr) / 2.0
                    mid[:, :, :hw]       = bl[:, :, :hw]
                    mid[:, :, n_pts-hw:] = bl[:, :, n_pts-hw:]
                    bl = np.minimum(bl, mid)
            else:
                # Linear endpoint baseline
                t = np.linspace(0, 1, n_pts)
                bl = sub[:, :, 0:1] + (sub[:, :, -1:] - sub[:, :, 0:1]) * t

            corrected = sub - bl
            # Safety floor: shift up so minimum is 0 (per pixel), preventing
            # negative areas from any residual baseline overshoot.
            per_px_min = corrected.min(axis=2, keepdims=True)
            corrected = corrected - np.minimum(per_px_min, 0)   # only shifts if negative
            area = np.trapz(corrected, axis=2)
            return area * float(gain)

        def _otsu_threshold(arr_flat):
            """Compute Otsu threshold on a 1-D array."""
            arr_f = arr_flat[np.isfinite(arr_flat)]
            counts, edges = np.histogram(arr_f, bins=256)
            total = counts.sum()
            best_thresh = edges[1]; best_var = 0
            w0 = 0; sum_total = np.dot(counts, edges[:-1])
            sum0 = 0
            for i in range(len(counts)):
                w0 += counts[i]
                if w0 == 0: continue
                w1 = total - w0
                if w1 == 0: break
                sum0 += counts[i] * edges[i]
                mu0 = sum0 / w0
                mu1 = (sum_total - sum0) / w1
                var = w0 * w1 * (mu0 - mu1) ** 2
                if var > best_var:
                    best_var = var
                    best_thresh = edges[i + 1]
            return float(best_thresh)

        _analysis_run_count = [0]   # track first run

        def run_analysis():
            # 0. On first run with manual mode, always auto-set threshold from data
            #    (ignores whatever value is in the spinbox — old sessions keep 0.3 etc.)
            _analysis_run_count[0] += 1
            if threshold_mode.get() == "manual" and _analysis_run_count[0] == 1:
                _auto_set_threshold()

            # 1. Compute peak-area maps with sensitivity controls
            al = a_lo_var.get(); ah = a_hi_var.get()
            bl = b_lo_var.get(); bh = b_hi_var.get()
            use_rbl = rolling_bl_var.get()
            map_a = _peak_area_map(al, ah, gain=gain_a_var.get(), rolling_bl=use_rbl)
            map_b_raw = _peak_area_map(bl, bh, gain=gain_b_var.get(), rolling_bl=use_rbl)

            # Apply Band B clamp: suppress pixels above clamp_b fraction of max
            # This zeroes out spurious high-intensity OH pixels that aren't ice.
            clamp_frac = clamp_b_var.get()
            if clamp_frac < 1.0 and map_b_raw.max() > 0:
                clamp_val = float(np.nanpercentile(
                    map_b_raw[mask] if mask.any() else map_b_raw,
                    clamp_frac * 100))
                map_b = np.clip(map_b_raw, 0, clamp_val)
            else:
                map_b = map_b_raw

            # 2. Choose source for binarization
            bsrc = binarize_band.get()
            if bsrc == "Band A":       src_map = map_a
            elif bsrc == "Band B":     src_map = map_b
            else:                      src_map = (map_a + map_b) / 2

            # Restrict to drawn ROI first
            src_roi = np.where(mask, src_map, np.nan)

            # 3. Compute threshold
            flat_roi = src_roi[np.isfinite(src_roi)]
            tmode = threshold_mode.get()
            if tmode == "otsu":
                thresh = _otsu_threshold(flat_roi)
            elif tmode == "percentile":
                thresh = float(np.nanpercentile(flat_roi, pct_var.get()))
            else:
                thresh = thresh_var.get()

            # 4. Binarize
            binary = mask & (src_map >= thresh)     # cellular region
            n_total = int(mask.sum())
            n_cell  = int(binary.sum())

            # 5. Smooth maps
            sig = sigma_var.get()
            sm_a = gaussian_filter(map_a.astype(float), sigma=sig)
            sm_b = gaussian_filter(map_b.astype(float), sigma=sig)

            # Stats within cellular region
            a_in   = map_a[binary]; b_in = map_b[binary]
            a_mean = float(np.mean(a_in)) if a_in.size else 0
            b_mean = float(np.mean(b_in)) if b_in.size else 0
            a_std  = float(np.std(a_in))  if a_in.size else 0
            b_std  = float(np.std(b_in))  if b_in.size else 0
            ratio  = b_mean / a_mean if a_mean > 0 else float("nan")

            # Mean spectrum of cellular region
            if binary.any():
                mean_spec = self.spectra[binary].mean(axis=0)
            else:
                mean_spec = self.spectra[mask].mean(axis=0)

            # ── Draw ──────────────────────────────────────────────────────────
            lbl_a = a_label_var.get(); lbl_b = b_label_var.get()
            cm_a  = cmap_a_var.get();  cm_b  = cmap_b_var.get()

            # Panel 0: ROI with binarization overlay
            ax_roi.clear()
            ax_roi.imshow(roi_rgba, origin="upper", aspect="equal")
            # Overlay binary contour
            from matplotlib.colors import ListedColormap as LCM
            bin_overlay = np.zeros((Y, X, 4), dtype=np.float32)
            bin_overlay[binary, 1] = 0.9   # green channel
            bin_overlay[binary, 3] = 0.45
            ax_roi.imshow(bin_overlay, origin="upper", aspect="equal")
            ax_roi.set_title(f"ROI  (N={n_total} px)  |  Threshold={thresh:.3e}",
                             fontsize=8, fontweight="semibold")
            ax_roi.set_xticks([]); ax_roi.set_yticks([])

            # Panel 1: Smoothed Band A heatmap (full ROI)
            ax_raw_a.clear()
            disp_a = np.where(mask, sm_a, np.nan)
            im_a = ax_raw_a.imshow(disp_a, origin="upper", cmap=cm_a,
                                   aspect="equal", interpolation="bilinear")
            fig.colorbar(im_a, ax=ax_raw_a, fraction=0.046, pad=0.04, shrink=0.8)
            ax_raw_a.set_title(f"Smoothed  —  {lbl_a}\n({al:.0f}–{ah:.0f} cm⁻¹)",
                               fontsize=8, fontweight="semibold")
            ax_raw_a.set_xticks([]); ax_raw_a.set_yticks([])

            # Panel 2: Smoothed Band B heatmap (full ROI)
            ax_raw_b.clear()
            disp_b = np.where(mask, sm_b, np.nan)
            im_b = ax_raw_b.imshow(disp_b, origin="upper", cmap=cm_b,
                                   aspect="equal", interpolation="bilinear")
            fig.colorbar(im_b, ax=ax_raw_b, fraction=0.046, pad=0.04, shrink=0.8)
            ax_raw_b.set_title(f"Smoothed  —  {lbl_b}\n({bl:.0f}–{bh:.0f} cm⁻¹)",
                               fontsize=8, fontweight="semibold")
            ax_raw_b.set_xticks([]); ax_raw_b.set_yticks([])

            # Panel 3: Mean spectrum with band annotations
            ax_spec.clear()
            ax_spec.plot(self.xdata, mean_spec,
                         color=C["text_hi"], lw=1.1, label="Mean (cellular)")
            ax_spec.axvspan(al, ah, alpha=0.18, color=C["band_a"], label=f"S_A  {lbl_a}")
            ax_spec.axvspan(bl, bh, alpha=0.18, color=C["band_b"], label=f"S_B  {lbl_b}")
            # annotate band area arrows
            for lo_wn, hi_wn, col, lbl_s in [
                (al, ah, C["band_a"], "S_amide"),
                (bl, bh, C["band_b"], "S_ice"),
            ]:
                m2 = (self.xdata >= lo_wn) & (self.xdata <= hi_wn)
                if m2.any():
                    cx = (lo_wn + hi_wn) / 2
                    cy = float(mean_spec[m2].max()) * 0.7
                    ax_spec.annotate(
                        lbl_s, xy=(cx, cy),
                        xytext=(cx, cy * 1.25 if cy > 0 else 0.1),
                        ha="center", fontsize=8, color=col,
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.9))
            ax_spec.set_xlabel("Wavenumber (cm⁻¹)", fontsize=8)
            ax_spec.set_ylabel("Intensity (a.u.)", fontsize=8)
            ax_spec.set_title("Mean ROI Spectrum + Band Regions", fontsize=9,
                               fontweight="semibold")
            ax_spec.legend(fontsize=7, framealpha=0.85)
            ax_spec.tick_params(labelsize=7)
            ax_spec.grid(True, ls="--", lw=0.4, alpha=0.5)

            # Panel 4: Binarized map (cellular region = yellow, background = black)
            ax_bin.clear()
            bin_rgb = np.zeros((Y, X, 3), dtype=np.float32)
            bin_rgb[binary]  = [1.0, 1.0, 0.0]   # yellow = cellular
            bin_rgb[~binary & mask] = [0.12, 0.12, 0.12]  # dark grey = ROI but excluded
            ax_bin.imshow(bin_rgb, origin="upper", aspect="equal")
            ax_bin.set_title(
                f"Binarized  N_p = {n_cell} px\n(cellular region)",
                fontsize=8, fontweight="semibold")
            ax_bin.set_xticks([]); ax_bin.set_yticks([])

            # Panel 5: Band B signal within cellular region (masked heatmap)
            ax_mask.clear()
            disp_masked = np.where(binary, sm_b, np.nan)
            im_mk = ax_mask.imshow(disp_masked, origin="upper", cmap=cm_b,
                                   aspect="equal", interpolation="bilinear")
            fig.colorbar(im_mk, ax=ax_mask, fraction=0.046, pad=0.04, shrink=0.8)
            ax_mask.set_title(
                f"Heatmap of {lbl_b}\nwithin cellular region",
                fontsize=8, fontweight="semibold")
            ax_mask.set_xticks([]); ax_mask.set_yticks([])

            canvas_r.draw_idle()

            # ── Update stats panel ────────────────────────────────────────────
            stats_text = (
                f"ROI pixels (drawn):     {n_total}\n"
                f"Cellular pixels  N_p:   {n_cell}\n"
                f"Non-cellular pixels:    {n_total - n_cell}\n"
                f"Threshold used:         {thresh:.3e}  ({tmode})\n"
                f"─────────────────────────\n"
                f"Band A  ({lbl_a[:20]})\n"
                f"  Mean area:  {a_mean:.3f} ± {a_std:.3f}\n"
                f"Band B  ({lbl_b[:20]})\n"
                f"  Mean area:  {b_mean:.3f} ± {b_std:.3f}\n"
                f"─────────────────────────\n"
                f"Ratio  S_B / S_A:       {ratio:.4f}"
            )
            stats_lbl.config(text=stats_text, fg=C["text_hi"],
                             font=("Consolas", 8))

            # Also save maps
            self._saved_maps[f"ROI_BandA_{lbl_a[:10]}"] = map_a
            self._saved_maps[f"ROI_BandB_{lbl_b[:10]}"] = map_b
            self._saved_maps[f"ROI_Binary"]             = binary.astype(float)

        def export_fig():
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")],
                parent=win)
            if path:
                _fig_ref[0].savefig(path, dpi=300, bbox_inches="tight")
                self._status.set(f"ROI Analysis figure saved → {Path(path).name}")

        def export_panels():
            """Export each ROI Analysis panel as a separate PNG into a chosen folder."""
            out_dir = filedialog.askdirectory(title="Choose folder for ROI panel exports", parent=win)
            if not out_dir:
                return
            base = Path(getattr(self, "_last_wdf_path", "ROI_Analysis")).stem
            dpi = 300
            try: canvas_r.draw_idle()
            except Exception: pass

            def _sanitize(s):
                s = str(s).strip().replace(" ", "_")
                return "".join(ch for ch in s if ch.isalnum() or ch in "-_[]()")[:32] or "panel"

            def _export_ax(ax, fname):
                try: fig.canvas.draw()
                except Exception: pass
                try:
                    bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
                except Exception:
                    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(fname, dpi=dpi, bbox_inches=bbox, pad_inches=0.02)

            lblA = _sanitize(lbl_a)
            lblB = _sanitize(lbl_b)
            panels = [
                (ax_roi,  "01_ROI_mask"),
                (ax_a,    f"02_{lblA}"),
                (ax_b,    f"03_{lblB}"),
                (ax_spec, "04_mean_spectrum"),
                (ax_bin,  "05_binarized"),
                (ax_mask, "06_masked_B"),
            ]
            exported = []
            for ax, tag in panels:
                fname = Path(out_dir) / f"{base}_{tag}.png"
                _export_ax(ax, str(fname))
                exported.append(fname.name)

            main_axes = [ax_roi, ax_a, ax_b, ax_spec, ax_bin, ax_mask]
            cbar_axes = [a for a in fig.axes if a not in main_axes]
            for k, cax in enumerate(cbar_axes, start=1):
                fname = Path(out_dir) / f"{base}_cbar_{k:02d}.png"
                _export_ax(cax, str(fname))
                exported.append(fname.name)

            messagebox.showinfo("Export complete", f"Exported {len(exported)} images to: {out_dir}", parent=win)
            self._status.set(f"Exported {len(exported)} ROI panels → {Path(out_dir).name}")

        run_btn.config(command=run_analysis)
        save_btn.config(command=export_fig)
        panels_btn.config(command=export_panels)

    # ── file loading ──────────────────────────────────────────────────────────

    def load_file(self):
        if not HAS_WDF:
            messagebox.showerror("Missing library",
                "renishawWiRE not installed.\npip install renishawWiRE")
            return
        path = filedialog.askopenfilename(
            title="Open Raman Map File",
            filetypes=[("Renishaw WDF","*.wdf"),("All files","*.*")])
        if not path: return
        self._status.set(f"Loading  {Path(path).name}…")
        self._show_progress(True)
        self.progress["value"] = 0
        self.update_idletasks()

        def worker():
            r = WDFReader(path)

            # Try to extract embedded white-light microscope image (if present)
            wl_raw = None
            try:
                img_obj = getattr(r, "img", None)
                if img_obj is not None:
                    try: img_obj.seek(0)
                    except Exception: pass
                    pil = Image.open(img_obj)
                    crop = getattr(r, "img_cropbox", None)
                    if crop is not None:
                        try: pil = pil.crop(box=crop)
                        except Exception: pass
                    pil = pil.convert("RGBA")
                    wl_raw = np.asarray(pil, dtype=np.float32) / 255.0
            except Exception:
                wl_raw = None

            params = self.pp_params
            def cb(f):
                self.after(0, lambda: self.progress.configure(value=f*100))
            proc, report = preprocess_map(r.spectra, params, cb)
            self.after(0, lambda: self._finish_load(r.xdata, proc, report, path, wl_raw))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_load(self, xdata, spectra, report, path, wl_raw=None):
        self.xdata   = xdata
        self.spectra = spectra
        self.pp_report = report
        self.coords  = (0, 0)
        self._show_progress(False)

        # Store embedded white-light image if available
        if wl_raw is not None:
            try:
                self.wl_raw = wl_raw
                if hasattr(self, "_wl_name") and self._wl_name is not None:
                    self._wl_name.config(text="(from WDF) " + Path(path).name, fg=C["success"])
                try:
                    pil = Image.fromarray((np.clip(wl_raw, 0, 1) * 255).astype(np.uint8))
                    thumb = pil.copy(); thumb.thumbnail((220, 130))
                    self._wl_ph = ImageTk.PhotoImage(thumb)
                    if hasattr(self, "_wl_thumb") and self._wl_thumb is not None:
                        self._wl_thumb.config(image=self._wl_ph)
                except Exception:
                    pass
            except Exception:
                pass

        Y, X, _ = spectra.shape
        cosmic   = report.get("cosmic_removed", 0)
        elapsed  = report.get("elapsed_s", "?")
        self._status.set(
            f"Loaded: {Path(path).name}  ·  {X}×{Y} spectra  ·  "
            f"{xdata[0]:.0f}–{xdata[-1]:.0f} cm⁻¹  ·  "
            f"{cosmic} cosmic rays removed  ·  {elapsed}s")
        lo, hi = float(xdata.min()), float(xdata.max())
        self.rs_a._from = lo; self.rs_a._to = hi
        self.rs_b._from = lo; self.rs_b._to = hi
        self._resize_wl()
        self.update_map()
        self._plot_spectrum(0, 0)
        self._rebuild_band_spans()
        self.after(300, self.show_pp_report)

    def load_wl(self):
        path = filedialog.askopenfilename(
            title="Load White-Light Image",
            filetypes=[("Images","*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                       ("All","*.*")])
        if not path: return
        img = Image.open(path).convert("RGBA")
        self.wl_raw = np.asarray(img, dtype=np.float32) / 255.0
        self._wl_name.config(text=Path(path).name, fg=C["success"])
        thumb = img.copy(); thumb.thumbnail((220,130))
        self._wl_ph = ImageTk.PhotoImage(thumb)
        self._wl_thumb.config(image=self._wl_ph)
        self._resize_wl()
        self.update_map()

    def _resize_wl(self):
        if self.wl_raw is None or self.spectra is None: return
        Y,X,_ = self.spectra.shape
        th,tw = Y*ZOOM, X*ZOOM
        pil = Image.fromarray((self.wl_raw*255).astype(np.uint8)).resize(
            (tw,th), Image.LANCZOS)
        self.wl_resized = np.asarray(pil, dtype=np.float32) / 255.0

    # ── map computation ───────────────────────────────────────────────────────
    def _band_mean(self, lo, hi):
        if self.xdata is None: return np.zeros((1,1))
        mask = (self.xdata >= lo) & (self.xdata <= hi)
        if not mask.any(): return np.zeros(self.spectra.shape[:2])
        return np.mean(self.spectra[:,:,mask], axis=2)

    def _smooth_zoom(self, arr):
        s = self.sl_sigma.value
        if s > 0: arr = gaussian_filter(arr, sigma=s)
        return zoom(arr, ZOOM, order=1)

    def _ratio_map(self):
        A = self._smooth_zoom(self._band_mean(self.rs_a.low, self.rs_a.high))
        B = self._smooth_zoom(self._band_mean(self.rs_b.low, self.rs_b.high))
        return np.divide(A, B, out=np.zeros_like(A), where=B!=0)

    def _rgb_map(self):
        A = self._smooth_zoom(self._band_mean(self.rs_a.low, self.rs_a.high))
        B = self._smooth_zoom(self._band_mean(self.rs_b.low, self.rs_b.high))
        A /= A.max()+1e-9; B /= B.max()+1e-9
        rgb = np.zeros((*A.shape, 3))
        rgb[...,0]=A; rgb[...,2]=B
        return np.clip(rgb, 0, 1)

    def _blend_wl(self, base):
        if self.wl_resized is None: return base
        alpha  = self.sl_wl_alpha.value
        bright = self.sl_wl_bright.value
        wl = np.clip(self.wl_resized[...,:3]*bright, 0, 1)
        h,w = base.shape[:2]
        wl_r = np.asarray(
            Image.fromarray((wl*255).astype(np.uint8)).resize((w,h), Image.LANCZOS),
            dtype=np.float32) / 255.0
        if base.ndim == 2:
            norm = Normalize(vmin=base.min(), vmax=base.max())
            base_rgb = plt.get_cmap(self.cmap_var.get())(norm(base))[...,:3]
        else:
            base_rgb = base[...,:3]
        return np.clip((1-alpha)*base_rgb + alpha*wl_r, 0, 1)

    def update_map(self):
        if self.spectra is None: return
        mode = self.mode_var.get()

        if mode == "wl":
            if self.wl_resized is None:
                self._status.set("Load a white-light image first."); return
            data = self.wl_resized[...,:3]
            self.im.set_data(data)
            self.cbar.ax.set_visible(False)
            self.ax_map.set_title("White Light Image", fontweight="semibold")
        elif mode == "rgb":
            data = self._rgb_map()
            if self.wl_resized is not None: data = self._blend_wl(data)
            self.im.set_data(np.clip(data, 0, 1))
            self.cbar.ax.set_visible(False)
            self.ax_map.set_title("Band A (R)  +  Band B (B)", fontweight="semibold")
        else:
            data = self._ratio_map()
            self.cbar.ax.set_visible(True)
            self.cbar.set_label("A / B  ratio", fontsize=9)
            self.ax_map.set_title("Band Ratio Map  (A / B)", fontweight="semibold")
            if self.wl_resized is not None:
                self.im.set_data(np.clip(self._blend_wl(data), 0, 1))
                self.cbar.ax.set_visible(False)
            else:
                self.im.set_cmap(self.cmap_var.get())
                if self._auto_clim.get():
                    vmin,vmax = data.min(),data.max()
                    self.sl_vmin.set(round(float(vmin),3))
                    self.sl_vmax.set(round(float(vmax),3))
                else:
                    vmin,vmax = self.sl_vmin.value, self.sl_vmax.value
                self.im.set_data(data)
                self.im.set_norm(Normalize(vmin=vmin, vmax=vmax))
                self.cbar.update_normal(self.im)

        self.im.set_extent((0,data.shape[1],data.shape[0],0))
        self.ax_map.set_xlim(0,data.shape[1])
        self.ax_map.set_ylim(data.shape[0],0)
        self._update_xhair()
        self.canvas.draw_idle()

    # ── spectrum ──────────────────────────────────────────────────────────────
    def _plot_spectrum(self, x, y, compare=False):
        spec = self.spectra[y, x, :].copy()
        if self._norm_var.get():
            pk = spec.max()
            if pk > 0: spec /= pk
        if compare:
            self.spec_compare.set_data(self.xdata, spec)
            self.spec_compare.set_visible(True)
        else:
            self.spec_line.set_data(self.xdata, spec)
            self.ax_spec.set_title(
                f"Raman Spectrum  —  pixel ({x}, {y})", fontweight="semibold")
        self.ax_spec.relim(); self.ax_spec.autoscale_view()
        self._annotate_peaks()
        self.ax_spec.legend(loc="upper right", frameon=True, fontsize=9)
        self.canvas.draw_idle()

    def _redraw_spectrum(self):
        if self.spectra is None or self.coords is None: return
        x,y = self.coords
        self._plot_spectrum(x, y)
        if self.compare_xy:
            cx,cy = self.compare_xy
            self._plot_spectrum(cx,cy, compare=True)

    def _annotate_peaks(self):
        for ann in self._peak_anns: ann.remove()
        self._peak_anns = []
        if not self._show_peaks.get(): return
        spec = self.spec_line.get_ydata()
        xd   = self.spec_line.get_xdata()
        if len(spec) == 0: return
        peaks,_ = find_peaks(spec, height=0.05*spec.max(),
                              distance=20, prominence=0.03)
        for pk in peaks:
            ann = self.ax_spec.annotate(
                f"{xd[pk]:.0f}",
                xy=(xd[pk], spec[pk]), xytext=(0,8),
                textcoords="offset points", ha="center",
                fontsize=8, color=C["text_mid"],
                arrowprops=dict(arrowstyle="-", color=C["text_dim"], lw=0.6))
            self._peak_anns.append(ann)

    def _clear_compare(self):
        self.compare_xy = None
        self.spec_compare.set_visible(False)
        self.canvas.draw_idle()

    # ── crosshair ─────────────────────────────────────────────────────────────
    def _update_xhair(self):
        if self.coords is None:
            for ln in (self.xhair_v, self.xhair_h, self.xhair_pt):
                ln.set_data([],[])
            return
        x,y = self.coords
        cx=(x+0.5)*ZOOM; cy=(y+0.5)*ZOOM
        xlim=self.ax_map.get_xlim(); ylim=self.ax_map.get_ylim()
        self.xhair_v.set_data([cx,cx], ylim)
        self.xhair_h.set_data(xlim, [cy,cy])
        self.xhair_pt.set_data([cx],[cy])

    # ── mouse ─────────────────────────────────────────────────────────────────
    def _click(self, e):
        # If ROI manager is active, let it handle events
        if self._roi_manager and self._roi_manager.active: return
        if e.inaxes != self.ax_map or self.spectra is None: return
        if e.xdata is None or e.ydata is None: return
        xi=int(e.xdata/ZOOM); yi=int(e.ydata/ZOOM)
        Y,X,_ = self.spectra.shape
        if not (0<=xi<X and 0<=yi<Y): return
        if e.button == 3:
            self.compare_xy=(xi,yi)
            self._plot_spectrum(xi,yi, compare=True)
        else:
            self.coords=(xi,yi)
            self._plot_spectrum(xi,yi)
            self._update_xhair()
            self.canvas.draw_idle()

    def _hover(self, e):
        if e.inaxes != self.ax_map or self.spectra is None:
            self._hover_info.set(""); return
        if e.xdata is None or e.ydata is None: return
        xi=int(e.xdata/ZOOM); yi=int(e.ydata/ZOOM)
        Y,X,_ = self.spectra.shape
        if 0<=xi<X and 0<=yi<Y:
            a_m=(self.xdata>=self.rs_a.low)&(self.xdata<=self.rs_a.high)
            b_m=(self.xdata>=self.rs_b.low)&(self.xdata<=self.rs_b.high)
            A=float(np.mean(self.spectra[yi,xi,a_m])) if a_m.any() else 0
            B=float(np.mean(self.spectra[yi,xi,b_m])) if b_m.any() else 0
            ratio=A/B if B!=0 else float("nan")
            self._hover_info.set(
                f"x={xi}  y={yi}  │  A={A:.4f}  B={B:.4f}  │  A/B={ratio:.4f}"
                + ("  │  Right-click → compare" if not
                   (self._roi_manager and self._roi_manager.active) else
                   "  │  Drawing ROI…"))
        else:
            self._hover_info.set("")

    # ── preprocessing settings ────────────────────────────────────────────────
    def open_pp_settings(self):
        p = self.pp_params
        dlg = tk.Toplevel(self)
        dlg.title("Preprocessing Settings")
        dlg.geometry("480x640")
        dlg.resizable(False, False)
        dlg.configure(bg=C["bg"])
        dlg.grab_set()

        def card_frame(parent):
            f = tk.Frame(parent, bg=C["panel"],
                         highlightthickness=1,
                         highlightbackground=C["border"])
            f.pack(fill="x", padx=12, pady=3)
            return f

        def row(parent, label, widget_factory, pady=4):
            f = tk.Frame(parent, bg=C["panel"])
            f.pack(fill="x", padx=12, pady=pady)
            tk.Label(f, text=label, width=26, anchor="w",
                     bg=C["panel"], fg=C["text_mid"],
                     font=("Segoe UI", 11)).pack(side="left")
            w = widget_factory(f)
            w.pack(side="left", padx=4)
            return w

        def sec(parent, title):
            sf = tk.Frame(parent, bg=C["bg"])
            sf.pack(fill="x", pady=(10,2))
            tk.Label(sf, text=f"  {title}", bg=C["bg"], fg=C["accent"],
                     font=("Segoe UI", 11, "bold")).pack(side="left")
            tk.Frame(sf, bg=C["border"], height=1).pack(
                side="left", fill="x", expand=True, padx=8, pady=6)

        cv = tk.Canvas(dlg, bg=C["bg"], highlightthickness=0)
        cv.pack(fill="both", expand=True)
        inn = tk.Frame(cv, bg=C["bg"])
        cv.create_window((0,0), window=inn, anchor="nw")
        inn.bind("<Configure>", lambda e: cv.configure(
            scrollregion=cv.bbox("all")))

        # Stage 1
        sec(inn, "① Cosmic Ray Removal")
        c1 = card_frame(inn)
        cr_var = tk.BooleanVar(value=p.cosmic_removal)
        row(c1,"Enable", lambda f: tk.Checkbutton(f, variable=cr_var,
            bg=C["panel"], activebackground=C["panel"], selectcolor=C["accent"]))
        ct_var = tk.DoubleVar(value=p.cosmic_threshold)
        row(c1,"Z-score threshold (6–15)",
            lambda f: ttk.Spinbox(f, from_=3, to=30, increment=0.5,
                                  textvariable=ct_var, width=9))
        cw_var = tk.IntVar(value=p.cosmic_width)
        row(c1,"Spike half-width (px)",
            lambda f: ttk.Spinbox(f, from_=1, to=10, increment=1,
                                  textvariable=cw_var, width=9))
        tk.Label(c1, text="  Modified Z-score on first derivative (Whitaker & Hayes 2018)",
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(0,6))

        # Stage 2
        sec(inn, "② Dark / Pedestal Removal")
        c2 = card_frame(inn)
        dk_var = tk.BooleanVar(value=p.dark_removal)
        row(c2,"Subtract spectrum minimum",
            lambda f: tk.Checkbutton(f, variable=dk_var,
                bg=C["panel"], activebackground=C["panel"], selectcolor=C["accent"]))
        tk.Label(c2, text="  Removes detector offset & stray-light pedestal",
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(0,6))

        # Stage 3
        sec(inn, "③ Baseline / Fluorescence Correction")
        c3 = card_frame(inn)
        bm_var = tk.StringVar(value=p.baseline_method)
        row(c3,"Algorithm",
            lambda f: ttk.Combobox(f, textvariable=bm_var, width=10,
                state="readonly", values=["asls","arpls","drpls","none"]))
        lam_var = tk.DoubleVar(value=p.asls_lam)
        row(c3,"λ smoothness (1e3–1e8)",
            lambda f: ttk.Spinbox(f, from_=1e3, to=1e8, increment=1e4,
                                  textvariable=lam_var, width=12))
        p_var = tk.DoubleVar(value=p.asls_p)
        row(c3,"p asymmetry (asls only)",
            lambda f: ttk.Spinbox(f, from_=0.0001, to=0.1, increment=0.001,
                                  textvariable=p_var, width=10, format="%.4f"))
        tk.Label(c3, text="  asls: general  arpls: strong/sloping  drpls: broad peaks",
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(0,6))

        # Stage 4
        sec(inn, "④ Savitzky-Golay Smoothing")
        c4 = card_frame(inn)
        sm_var = tk.BooleanVar(value=p.smoothing)
        row(c4,"Enable", lambda f: tk.Checkbutton(f, variable=sm_var,
            bg=C["panel"], activebackground=C["panel"], selectcolor=C["accent"]))
        sgw_var = tk.IntVar(value=p.sg_window)
        row(c4,"Window length (odd)",
            lambda f: ttk.Spinbox(f, from_=5, to=51, increment=2,
                                  textvariable=sgw_var, width=9))
        sgp_var = tk.IntVar(value=p.sg_poly)
        row(c4,"Polynomial order",
            lambda f: ttk.Spinbox(f, from_=1, to=6, increment=1,
                                  textvariable=sgp_var, width=9))

        # Stage 5
        sec(inn, "⑤ Intensity Normalisation")
        c5 = card_frame(inn)
        nm_var = tk.StringVar(value=p.normalisation)
        row(c5,"Method",
            lambda f: ttk.Combobox(f, textvariable=nm_var, width=10,
                state="readonly", values=["max","area","none"]))
        tk.Label(c5, text="  max: peak  ·  area: total integrated  ·  none: raw",
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(0,6))

        # Buttons
        btn_row = tk.Frame(dlg, bg=C["bg"])
        btn_row.pack(fill="x", padx=12, pady=8)

        def apply_close():
            p.cosmic_removal  = cr_var.get()
            p.cosmic_threshold= ct_var.get()
            p.cosmic_width    = cw_var.get()
            p.dark_removal    = dk_var.get()
            p.baseline_method = bm_var.get()
            p.asls_lam        = lam_var.get()
            p.asls_p          = p_var.get()
            p.smoothing       = sm_var.get()
            p.sg_window       = sgw_var.get()
            p.sg_poly         = sgp_var.get()
            p.normalisation   = nm_var.get()
            dlg.destroy()

        ttk.Button(btn_row, text="Apply & Close", style="Primary.TButton",
                   command=apply_close).pack(side="right", padx=4)
        ttk.Button(btn_row, text="Cancel", style="Neutral.TButton",
                   command=dlg.destroy).pack(side="right", padx=4)
        tk.Label(btn_row, text="Changes apply on next file load",
                 bg=C["bg"], fg=C["text_dim"],
                 font=("Segoe UI", 10)).pack(side="left", padx=4)

    # ── preprocessing report ──────────────────────────────────────────────────
    def show_pp_report(self):
        rpt = tk.Toplevel(self)
        rpt.title("Preprocessing Report")
        rpt.geometry("520x580")
        rpt.configure(bg=C["bg"])

        hdr = tk.Frame(rpt, bg=C["header"], height=52)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="◈  Preprocessing Log",
                 bg=C["header"], fg="white",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=16, pady=12)

        body = tk.Frame(rpt, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=12)

        if self.pp_report is None:
            tk.Label(body, text="No data loaded yet. Load a WDF file first.",
                     bg=C["bg"], fg=C["text_dim"],
                     font=("Segoe UI", 11)).pack(pady=40)
            ttk.Button(rpt, text="Close", style="Neutral.TButton",
                       command=rpt.destroy).pack(pady=8)
            return

        r = self.pp_report
        p = self.pp_params

        def card(title, rows, good=True):
            bc = C["success"] if good else C["danger"]
            f = tk.Frame(body, bg=C["panel"],
                         highlightthickness=2, highlightbackground=bc)
            f.pack(fill="x", pady=4)
            tk.Label(f, text=f"  {title}", bg=bc, fg="white",
                     font=("Segoe UI", 11, "bold")).pack(fill="x")
            for lbl, val, col in rows:
                rf = tk.Frame(f, bg=C["panel"])
                rf.pack(fill="x", padx=10, pady=1)
                tk.Label(rf, text=lbl, width=28, anchor="w",
                         bg=C["panel"], fg=C["text_mid"],
                         font=("Segoe UI", 11)).pack(side="left")
                tk.Label(rf, text=str(val), anchor="w",
                         bg=C["panel"], fg=col,
                         font=("Consolas", 9, "bold")).pack(side="left")
            tk.Frame(f, bg=C["panel"], height=4).pack()

        cosmic_n = r.get("cosmic_removed", 0)
        card("① Cosmic Ray Removal", [
            ("Status", "ENABLED" if p.cosmic_removal else "DISABLED",
             C["success"] if p.cosmic_removal else C["danger"]),
            ("Algorithm","Modified Z-score on 1st derivative",C["text_hi"]),
            ("Z-score threshold",f"{p.cosmic_threshold}",C["accent"]),
            ("Spike half-width",f"{p.cosmic_width} px",C["accent"]),
            ("Spikes removed",str(cosmic_n),
             C["success"] if cosmic_n==0 else C["warn"]),
        ], good=p.cosmic_removal)

        card("② Dark / Pedestal Removal", [
            ("Status","ENABLED" if p.dark_removal else "DISABLED",
             C["success"] if p.dark_removal else C["danger"]),
            ("Method","Subtract per-spectrum minimum",C["text_hi"]),
        ], good=p.dark_removal)

        bl_good = p.baseline_method != "none"
        card("③ Baseline / Fluorescence Correction", [
            ("Status","ENABLED" if bl_good else "DISABLED",
             C["success"] if bl_good else C["danger"]),
            ("Algorithm",r.get("baseline_method","—"),C["accent"]),
            ("λ (smoothness)",r.get("baseline_lam","—"),C["text_hi"]),
            ("p (asymmetry)",r.get("baseline_p","—")
             if p.baseline_method=="asls" else "N/A",C["text_hi"]),
        ], good=bl_good)

        card("④ Savitzky-Golay Smoothing", [
            ("Status","ENABLED" if p.smoothing else "DISABLED",
             C["success"] if p.smoothing else C["danger"]),
            ("Window length",f"{p.sg_window} pts" if p.smoothing else "—",C["accent"]),
            ("Polynomial order",f"{p.sg_poly}" if p.smoothing else "—",C["accent"]),
        ], good=p.smoothing)

        card("⑤ Intensity Normalisation", [
            ("Method",r.get("normalisation","—"),C["accent"]),
        ], good=True)

        # summary strip
        sf = tk.Frame(body, bg=C["header"])
        sf.pack(fill="x", pady=(8,0))
        for lbl, val in [
            ("Map size",       r.get("map_shape","—")),
            ("Total spectra",  str(r.get("total_spectra","—"))),
            ("Wavenumber pts", str(r.get("spectral_points","—"))),
            ("CPU workers",    str(r.get("workers","—"))),
            ("Processing time",f"{r.get('elapsed_s','—')} s"),
        ]:
            tf = tk.Frame(sf, bg=C["header"])
            tf.pack(side="left", padx=12, pady=6)
            tk.Label(tf, text=lbl, bg=C["header"], fg=C["text_dim"],
                     font=("Segoe UI", 7)).pack()
            tk.Label(tf, text=val, bg=C["header"], fg="white",
                     font=("Consolas", 9, "bold")).pack()

        ttk.Button(rpt, text="Close", style="Neutral.TButton",
                   command=rpt.destroy).pack(pady=8)

    # ── PCA window ────────────────────────────────────────────────────────────
    def open_pca(self):
        PCAWindow(self, self.pp_params)

    # ── UNIVARIATE ANALYSIS ───────────────────────────────────────────────────
    def open_univariate(self):
        """Raw-data univariate map dialog (intensity at point / signal to baseline/axis)."""
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first."); return
        dlg = tk.Toplevel(self)
        dlg.title("Univariate Analysis — Raw Data Map")
        dlg.geometry("560x480")
        dlg.configure(bg=C["bg"])
        dlg.grab_set()

        hdr = tk.Frame(dlg, bg=C["header"], height=48)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="◈  UNIVARIATE ANALYSIS",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=12)

        body = tk.Frame(dlg, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=12)

        # Map type
        tk.Label(body, text="Map type:", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=6)
        map_type = tk.StringVar(value="signal_to_baseline")
        for i, (val, lbl) in enumerate([
            ("intensity_at_point",  "Intensity at a Point"),
            ("signal_to_baseline",  "Signal to Baseline"),
            ("signal_to_axis",      "Signal to Axis"),
        ]):
            tk.Radiobutton(body, text=lbl, variable=map_type, value=val,
                           bg=C["bg"], fg=C["text_hi"],
                           activebackground=C["bg"],
                           font=("Segoe UI", 10)).grid(
                               row=i+1, column=0, sticky="w", padx=20)

        lo, hi = float(self.xdata.min()), float(self.xdata.max())
        mid = (lo + hi) / 2

        # Cursor 1
        tk.Label(body, text="First limit (cm⁻¹):", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=5, column=0, sticky="w", pady=(12,2))
        lim1_var = tk.DoubleVar(value=round(mid - 50))
        ttk.Spinbox(body, from_=lo, to=hi, increment=1,
                    textvariable=lim1_var, width=10).grid(
                        row=5, column=1, sticky="w", padx=8)

        # Cursor 2
        tk.Label(body, text="Second limit (cm⁻¹):", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=6, column=0, sticky="w", pady=(4,2))
        lim2_var = tk.DoubleVar(value=round(mid + 50))
        ttk.Spinbox(body, from_=lo, to=hi, increment=1,
                    textvariable=lim2_var, width=10).grid(
                        row=6, column=1, sticky="w", padx=8)

        tk.Label(body,
                 text="Tip: for 'Intensity at Point', only First limit is used.",
                 bg=C["bg"], fg=C["text_dim"],
                 font=("Segoe UI", 10)).grid(row=7, column=0, columnspan=2,
                                             sticky="w", pady=4)

        # Map name
        tk.Label(body, text="Save map as:", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=8, column=0, sticky="w", pady=(10,2))
        name_var = tk.StringVar(value="Map 1")
        tk.Entry(body, textvariable=name_var, font=("Segoe UI", 10),
                 bg="white", relief="flat",
                 highlightthickness=1,
                 highlightbackground=C["border"]).grid(
                     row=8, column=1, sticky="ew", padx=8)

        preview_lbl = tk.Label(body, text="", bg=C["bg"], fg=C["text_dim"],
                               font=("Segoe UI", 11))
        preview_lbl.grid(row=9, column=0, columnspan=2, sticky="w", pady=4)

        def create_map():
            t = map_type.get()
            l1 = lim1_var.get(); l2 = lim2_var.get()
            nm = name_var.get().strip() or f"Map {len(self._saved_maps)+1}"
            wn = self.xdata

            if t == "intensity_at_point":
                idx = int(np.argmin(np.abs(wn - l1)))
                arr = self.spectra[:, :, idx]
                desc = f"Intensity @ {wn[idx]:.1f} cm⁻¹"
            elif t == "signal_to_baseline":
                m = (wn >= min(l1, l2)) & (wn <= max(l1, l2))
                if not m.any():
                    messagebox.showwarning("Range", "No data in range.", parent=dlg); return
                sub = self.spectra[:, :, m]
                # baseline = straight line between endpoints
                bl = sub[:, :, 0:1] + (sub[:, :, -1:] - sub[:, :, 0:1]) * \
                     np.linspace(0, 1, m.sum())
                arr = np.trapz(np.clip(sub - bl, 0, None), axis=2)
                desc = f"Signal to Baseline {min(l1,l2):.0f}–{max(l1,l2):.0f} cm⁻¹"
            else:  # signal_to_axis
                m = (wn >= min(l1, l2)) & (wn <= max(l1, l2))
                if not m.any():
                    messagebox.showwarning("Range", "No data in range.", parent=dlg); return
                arr = np.trapz(np.clip(self.spectra[:, :, m], 0, None), axis=2)
                desc = f"Signal to Axis {min(l1,l2):.0f}–{max(l1,l2):.0f} cm⁻¹"

            self._saved_maps[nm] = arr.copy()
            preview_lbl.config(
                text=f"✓  Saved '{nm}' ({desc})  —  {arr.shape[1]}×{arr.shape[0]} px",
                fg=C["success"])
            # Display on main map
            self._show_saved_map(nm)
            self._status.set(f"Univariate map '{nm}': {desc}")

        btn_row = tk.Frame(dlg, bg=C["bg"])
        btn_row.pack(fill="x", padx=16, pady=8)
        ttk.Button(btn_row, text="Create Map", style="Primary.TButton",
                   command=create_map).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Close", style="Neutral.TButton",
                   command=dlg.destroy).pack(side="left", padx=4)

    def _show_saved_map(self, name):
        """Display a named saved map on the main axes."""
        arr = self._saved_maps.get(name)
        if arr is None: return
        zoomed = zoom(gaussian_filter(arr.astype(float),
                                      sigma=self.sl_sigma.value or 0), ZOOM, order=1)
        self.im.set_data(zoomed)
        self.im.set_cmap(self.cmap_var.get())
        self.im.set_norm(Normalize(vmin=zoomed.min(), vmax=zoomed.max()))
        self.im.set_extent((0, zoomed.shape[1], zoomed.shape[0], 0))
        self.ax_map.set_xlim(0, zoomed.shape[1])
        self.ax_map.set_ylim(zoomed.shape[0], 0)
        self.cbar.ax.set_visible(True)
        self.cbar.set_label(name, fontsize=9)
        self.cbar.update_normal(self.im)
        self.ax_map.set_title(f"Univariate Map — {name}", fontweight="semibold")
        self._update_xhair()
        self.canvas.draw_idle()

    # ── DYNAMIC MAPPING ───────────────────────────────────────────────────────
    def open_dynamic_map(self):
        """Live-updating map as spectral range is dragged (WiRE-like Dynamic Mapping).

        Includes:
          • ROI draw + clear, mask outside ROI
          • ROI mean spectrum overlay
          • Analyse ROI (uses ROI mask drawn here)
          • Save ROI image + Save ROI mask (binary PNG)
          • White-light overlay controls
        """
        if self.spectra is None or self.xdata is None:
            messagebox.showwarning("No data", "Load a WDF file first.")
            return

        Y, X, W = self.spectra.shape
        cube = self.spectra
        xax  = self.xdata

        # keep WL sized
        try:
            self._resize_wl()
        except Exception:
            pass

        win = tk.Toplevel(self)
        win.title("Dynamic Mapping")
        win.geometry("1120x700")
        win.configure(bg=C["bg"])

        hdr = tk.Frame(win, bg=C["header"], height=46)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⚡  DYNAMIC MAPPING",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=12)

        # Controls (two rows)
        ctrl = tk.Frame(win, bg=C["sidebar"])
        ctrl.pack(fill="x", padx=0)
        ctrl1 = tk.Frame(ctrl, bg=C["sidebar"])
        ctrl2 = tk.Frame(ctrl, bg=C["sidebar"])
        ctrl1.pack(fill="x", padx=8, pady=(6, 2))
        ctrl2.pack(fill="x", padx=8, pady=(2, 6))

        # Row 1
        tk.Label(ctrl1, text="Map type:", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(4, 4))
        dm_type = tk.StringVar(value="signal_to_baseline")
        for val, lbl in [("intensity_at_point", "Intensity at Point"),
                         ("signal_to_baseline", "Signal to Baseline"),
                         ("signal_to_axis", "Signal to Axis")]:
            tk.Radiobutton(ctrl1, text=lbl, variable=dm_type, value=val,
                           bg=C["sidebar"], fg=C["text_hi"],
                           activebackground=C["sidebar"],
                           font=("Segoe UI", 11),
                           command=lambda: _update()).pack(side="left", padx=4)

        tk.Frame(ctrl1, bg=C["border"], width=1).pack(side="left", fill="y", pady=2, padx=10)

        acq_var = tk.IntVar(value=0)
        tk.Label(ctrl1, text="Acquisition:", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 11)).pack(side="left", padx=(0, 4))
        ttk.Spinbox(ctrl1, from_=0, to=Y*X-1, textvariable=acq_var, width=7,
                    command=lambda: _update_spectrum()).pack(side="left", padx=(0, 10))

        save_name = tk.StringVar(value="Dynamic Map 1")
        tk.Label(ctrl1, text="Name:", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 11)).pack(side="left", padx=(0, 4))
        tk.Entry(ctrl1, textvariable=save_name, width=16,
                 bg="white", font=("Segoe UI", 11), relief="flat",
                 highlightthickness=1, highlightbackground=C["border"]).pack(side="left", padx=(0, 4))

        # Row 2 - ROI + WL
        tk.Label(ctrl2, text="ROI:", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 11)).pack(side="left", padx=(4, 4))
        roi_shape = tk.StringVar(value="Rect")
        ttk.Combobox(ctrl2, textvariable=roi_shape,
                     values=["Rect", "Ellipse", "Polygon", "Freehand"],
                     width=10, state="readonly").pack(side="left", padx=(0, 8))
        tk.Button(ctrl2, text="Draw ROI", relief="flat", bg=C["panel"],
                  command=lambda: _start_roi()).pack(side="left", padx=2)
        tk.Button(ctrl2, text="Clear ROI", relief="flat", bg=C["panel"],
                  command=lambda: _clear_roi()).pack(side="left", padx=2)

        mask_outside = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl2, text="Mask outside", variable=mask_outside,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10),
                       command=lambda: _update()).pack(side="left", padx=(8, 8))

        tk.Button(ctrl2, text="Analyse ROI", relief="flat", bg=C["roi"], fg="white",
                  activebackground=C["roi"], command=lambda: _analyse_roi()).pack(side="left", padx=(0, 6))
        tk.Button(ctrl2, text="Save ROI Image…", relief="flat", bg=C["panel"],
                  command=lambda: _save_roi_image()).pack(side="left", padx=(0, 6))
        tk.Button(ctrl2, text="Save ROI Mask…", relief="flat", bg=C["panel"],
                  command=lambda: _save_roi_mask()).pack(side="left", padx=(0, 10))

        roi_info = tk.Label(ctrl2, text="No ROI", bg=C["sidebar"], fg=C["text_dim"],
                            font=("Segoe UI", 10, "italic"))
        roi_info.pack(side="left", padx=(0, 10))

        tk.Frame(ctrl2, bg=C["border"], width=1).pack(side="left", fill="y", pady=2, padx=10)
        wl_on = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl2, text="White light", variable=wl_on,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10),
                       command=lambda: _update_wl_visibility()).pack(side="left", padx=(0, 6))

        wl_alpha = tk.DoubleVar(value=0.70)
        tk.Label(ctrl2, text="Overlay α", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 4))
        ttk.Scale(ctrl2, from_=0.05, to=1.0, variable=wl_alpha,
                  command=lambda *_: _update_wl_alpha()).pack(side="left", padx=(0, 6), ipadx=28)

        def _load_wl_here():
            self.load_wl()
            try:
                self._resize_wl()
            except Exception:
                pass
            _update_wl_visibility(); _update()

        tk.Button(ctrl2, text="Load WL…", relief="flat", bg=C["panel"],
                  command=_load_wl_here).pack(side="left", padx=(0, 4))

        # ── Figure ─────────────────────────────────────────────────────────
        fig = plt.Figure(figsize=(10.6, 6.0), facecolor="white")
        gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.7], wspace=0.35,
                               left=0.07, right=0.97, top=0.93, bottom=0.12)
        ax_spec = fig.add_subplot(gs[0])
        ax_map  = fig.add_subplot(gs[1])

        canvas_dm = FigureCanvasTkAgg(fig, master=win)
        canvas_dm.get_tk_widget().pack(fill="both", expand=True)

        lo = float(xax.min()); hi = float(xax.max()); mid = (lo + hi) / 2

        ax_spec.set_xlabel("Raman Shift (cm⁻¹)", fontsize=9)
        ax_spec.set_ylabel("Intensity (a.u.)", fontsize=9)
        ax_spec.set_title("Spectrum", fontsize=10, fontweight="semibold")
        ax_spec.set_xlim(lo, hi)

        ax_spec.plot(xax, cube.reshape(-1, W).mean(axis=0),
                     color=C["accent"], alpha=0.65, lw=1.2, label="Map avg")

        spec_cur_line, = ax_spec.plot(xax, cube[0, 0],
                                      color=C["spec_line"], lw=1.2, label="Current")
        spec_roi_line, = ax_spec.plot(xax, cube[0, 0],
                                      color=C["roi"], lw=1.6, ls="--", alpha=0.9, label="ROI mean")
        spec_roi_line.set_visible(False)

        l1 = ax_spec.axvline(mid - (hi - lo) * 0.05, color=C["band_a"], ls="--", lw=1.2, picker=5)
        l2 = ax_spec.axvline(mid + (hi - lo) * 0.05, color=C["band_a"], ls="--", lw=1.2, picker=5)
        span = ax_spec.axvspan(min(l1.get_xdata()[0], l2.get_xdata()[0]),
                               max(l1.get_xdata()[0], l2.get_xdata()[0]),
                               alpha=0.12, color=C["band_a"])

        ax_spec.legend(loc="upper right", fontsize=8)
        ax_spec.grid(True, alpha=0.25)

        ax_map.set_title("Live Map", fontsize=10, fontweight="semibold")
        ax_map.set_xlabel("X (px)", fontsize=9)
        ax_map.set_ylabel("Y (px)", fontsize=9)

        cmap = (self.cmap_var.get() if hasattr(self, "cmap_var") else "turbo")

        im_wl = None
        if getattr(self, "wl_resized", None) is not None:
            try:
                im_wl = ax_map.imshow(self.wl_resized[..., :3], origin="upper", interpolation="nearest")
                im_wl.set_zorder(0)
            except Exception:
                im_wl = None

        im_dm = ax_map.imshow(np.zeros((Y * ZOOM, X * ZOOM)), cmap=cmap,
                              origin="upper", interpolation="nearest")
        im_dm.set_zorder(1)
        cbar = fig.colorbar(im_dm, ax=ax_map, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        # ROI wiring
        roi_mask = {"mask": None}
        def _roi_callback(mask):
            roi_mask["mask"] = mask
            # store for ROI Analysis window
            self._roi_mask = mask
            try:
                self._roi_mask_inv = ~mask
            except Exception:
                self._roi_mask_inv = None
            n_px = int(mask.sum()) if mask is not None else 0
            roi_info.config(text=f"{n_px} px" if n_px else "No ROI",
                            fg=C["success"] if n_px else C["text_dim"])
            _update_roi_spectrum(); _update()

        def _shape_to_mode(lbl):
            return {"Rect": "rectangle", "Ellipse": "ellipse", "Polygon": "polygon", "Freehand": "freehand"}.get(lbl, "rectangle")

        roi_mgr = ROIManager(ax_map, canvas_dm, ZOOM, _roi_callback)

        def _start_roi():
            roi_mgr.activate(_shape_to_mode(roi_shape.get()), (Y, X))

        def _clear_roi():
            roi_mgr.deactivate()
            roi_mask["mask"] = None
            self._roi_mask = None
            self._roi_mask_inv = None
            roi_info.config(text="No ROI", fg=C["text_dim"])
            spec_roi_line.set_visible(False)
            _update()

        def _analyse_roi():
            m = roi_mask.get("mask")
            if m is None or (not np.any(m)):
                messagebox.showinfo("No ROI", "Draw an ROI first, then click Analyse ROI.")
                return
            self._roi_mask = m
            try:
                self._roi_mask_inv = ~m
            except Exception:
                self._roi_mask_inv = None
            self.open_roi_analysis()

        # map click selects spectrum if not drawing ROI
        def _on_map_click(e):
            if roi_mgr.active:
                return
            if e.inaxes != ax_map or e.xdata is None or e.ydata is None:
                return
            xi = int(e.xdata / ZOOM); yi = int(e.ydata / ZOOM)
            xi = max(0, min(X - 1, xi)); yi = max(0, min(Y - 1, yi))
            acq_var.set(yi * X + xi)
            _update_spectrum()

        canvas_dm.mpl_connect("button_press_event", _on_map_click)

        # draggable lines
        drag = {"artist": None}
        def _on_pick(e):
            if e.artist in (l1, l2):
                drag["artist"] = e.artist
        def _on_release(_e):
            drag["artist"] = None
        def _on_move(e):
            nonlocal span
            if drag["artist"] is None or e.inaxes != ax_spec or e.xdata is None:
                return
            x = float(e.xdata)
            drag["artist"].set_xdata([x, x])
            x1 = float(l1.get_xdata()[0]); x2 = float(l2.get_xdata()[0])
            try:
                span.remove()
            except Exception:
                pass
            span = ax_spec.axvspan(min(x1, x2), max(x1, x2), alpha=0.12, color=C["band_a"])
            _update()

        canvas_dm.mpl_connect("pick_event", _on_pick)
        canvas_dm.mpl_connect("motion_notify_event", _on_move)
        canvas_dm.mpl_connect("button_release_event", _on_release)

        def _current_band_indices():
            x1 = float(l1.get_xdata()[0]); x2 = float(l2.get_xdata()[0])
            aa, bb = sorted([x1, x2])
            i0 = int(np.searchsorted(xax, aa, side="left"))
            i1 = int(np.searchsorted(xax, bb, side="right"))
            i0 = max(0, min(W - 2, i0))
            i1 = max(i0 + 1, min(W, i1))
            return i0, i1

        def _compute_map_raw():
            i0, i1 = _current_band_indices()
            if dm_type.get() == "intensity_at_point":
                mid_i = (i0 + i1) // 2
                return cube[:, :, mid_i].astype(float)
            if dm_type.get() == "signal_to_axis":
                y = cube[:, :, i0:i1].astype(float)
                x = xax[i0:i1].astype(float)
                return np.trapz(y, x, axis=2)
            # signal_to_baseline: max of band after linear baseline
            y = cube[:, :, i0:i1].astype(float)
            n = y.shape[2]
            y0 = cube[:, :, i0].astype(float)[:, :, None]
            y1 = cube[:, :, i1 - 1].astype(float)[:, :, None]
            t = np.linspace(0.0, 1.0, n, dtype=float)[None, None, :]
            base = y0 + (y1 - y0) * t
            return np.max(y - base, axis=2)

        def _update_roi_spectrum():
            m = roi_mask.get("mask")
            if m is None or m.shape != (Y, X) or (not np.any(m)):
                spec_roi_line.set_visible(False)
                ax_spec.legend(loc="upper right", fontsize=8)
                return
            spec_roi_line.set_ydata(np.nanmean(cube[m], axis=0))
            spec_roi_line.set_visible(True)
            ax_spec.legend(loc="upper right", fontsize=8)

        def _update_spectrum(*_):
            idx = int(acq_var.get())
            yi, xi = divmod(idx, X)
            yi = max(0, min(Y - 1, yi)); xi = max(0, min(X - 1, xi))
            spec_cur_line.set_ydata(cube[yi, xi])
            canvas_dm.draw_idle()

        def _update_wl_visibility():
            nonlocal im_wl
            if wl_on.get() and getattr(self, "wl_resized", None) is not None:
                if im_wl is None:
                    try:
                        im_wl = ax_map.imshow(self.wl_resized[..., :3], origin="upper", interpolation="nearest")
                        im_wl.set_zorder(0); im_dm.set_zorder(1)
                    except Exception:
                        im_wl = None
                if im_wl is not None:
                    im_wl.set_visible(True)
            else:
                if im_wl is not None:
                    im_wl.set_visible(False)
            _update_wl_alpha()

        def _update_wl_alpha():
            a = float(wl_alpha.get())
            if wl_on.get() and getattr(self, "wl_resized", None) is not None:
                im_dm.set_alpha(a)
            else:
                im_dm.set_alpha(1.0)
            canvas_dm.draw_idle()

        def _save_roi_image():
            m = roi_mask.get("mask")
            if m is None or (not np.any(m)):
                messagebox.showinfo("No ROI", "Draw an ROI first, then use Save ROI Image…")
                return
            out_path = filedialog.asksaveasfilename(
                title="Save ROI Image", defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("TIFF", "*.tif *.tiff"), ("JPEG", "*.jpg *.jpeg"), ("All files", "*.*")])
            if not out_path:
                return
            ys, xs = np.where(m)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            pad = 2
            zx0 = max(0, x0 * ZOOM - pad)
            zx1 = min(X * ZOOM - 1, (x1 + 1) * ZOOM + pad)
            zy0 = max(0, y0 * ZOOM - pad)
            zy1 = min(Y * ZOOM - 1, (y1 + 1) * ZOOM + pad)

            fig2 = plt.Figure(figsize=(6, 6), facecolor="white")
            ax2 = fig2.add_subplot(111)
            ax2.set_axis_off()
            if wl_on.get() and getattr(self, "wl_resized", None) is not None:
                try:
                    ax2.imshow(self.wl_resized[..., :3], origin="upper", interpolation="nearest")
                except Exception:
                    pass
            arr = np.array(im_dm.get_array())
            cmap2 = im_dm.get_cmap(); vmin, vmax = im_dm.get_clim()
            alpha2 = float(im_dm.get_alpha() if im_dm.get_alpha() is not None else 1.0)
            ax2.imshow(arr, origin="upper", interpolation="nearest", cmap=cmap2, vmin=vmin, vmax=vmax, alpha=alpha2)
            try:
                if getattr(roi_mgr, "_patch", None) is not None:
                    p = roi_mgr._patch
                    path = p.get_path().transformed(p.get_patch_transform())
                    ax2.add_patch(mpatches.PathPatch(path, fill=False, edgecolor=C["roi"], linewidth=2))
                if getattr(roi_mgr, "_line", None) is not None:
                    xd, yd = roi_mgr._line.get_data()
                    ax2.plot(xd, yd, color=C["roi"], linewidth=2)
            except Exception:
                pass
            ax2.set_xlim(zx0, zx1)
            ax2.set_ylim(zy1, zy0)
            try:
                fig2.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
                self._status.set(f"Saved ROI image: {Path(out_path).name}")
            except Exception as ex:
                messagebox.showerror("Save failed", f"Could not save image: {ex}")

        def _save_roi_mask():
            m = roi_mask.get("mask")
            if m is None or (not np.any(m)):
                messagebox.showinfo("No ROI", "Draw an ROI first, then use Save ROI Mask…")
                return
            out_path = filedialog.asksaveasfilename(
                title="Save ROI Mask", defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("All files", "*.*")])
            if not out_path:
                return
            try:
                Image.fromarray((m.astype(np.uint8) * 255), mode="L").save(out_path)
                self._status.set(f"Saved ROI mask: {Path(out_path).name}")
            except Exception as ex:
                messagebox.showerror("Save failed", f"Could not save ROI mask: {ex}")

        def _update(*_):
            arr_raw = _compute_map_raw()
            # optional smoothing from main UI
            sigma = 0.0
            try:
                sigma = float(getattr(self, "sl_sigma").value)
            except Exception:
                sigma = 0.0
            if sigma > 0:
                arr_raw = gaussian_filter(arr_raw, sigma=sigma)

            arr = zoom(arr_raw, ZOOM, order=1)
            m = roi_mask.get("mask")
            if m is not None and mask_outside.get() and np.any(m):
                mz = (zoom(m.astype(float), ZOOM, order=0) > 0.5)
                arr = arr.astype(float)
                arr[~mz] = np.nan

            im_dm.set_data(arr)
            finite = np.isfinite(arr)
            if finite.any():
                vmin = float(np.nanmin(arr)); vmax = float(np.nanmax(arr))
                if vmin == vmax:
                    vmax = vmin + 1e-9
                im_dm.set_clim(vmin, vmax)
            _update_wl_visibility()
            canvas_dm.draw_idle()

        _update_spectrum(); _update()

        def _on_close():
            try:
                roi_mgr.deactivate()
            except Exception:
                pass
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _on_close)

    def open_curve_fit_map(self):
        """Fit a peak across every spectrum and create a map of a peak parameter."""
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first."); return
        if not HAS_SKL:  # scipy is always available; sklearn only needed for PCA
            pass

        dlg = tk.Toplevel(self)
        dlg.title("Curve Fit Map")
        dlg.geometry("540x560")
        dlg.configure(bg=C["bg"])
        dlg.grab_set()

        hdr = tk.Frame(dlg, bg=C["header"], height=46)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="~  CURVE FIT MAP",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=12)

        body = tk.Frame(dlg, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=10)

        def lbl(text, row, col=0, **kw):
            tk.Label(body, text=text, bg=C["bg"], fg=C["text_mid"],
                     font=("Segoe UI", 10), **kw).grid(
                         row=row, column=col, sticky="w", pady=3)

        lo = float(self.xdata.min()); hi = float(self.xdata.max())
        mid = (lo + hi) / 2

        lbl("Peak centre (cm⁻¹):", 0)
        centre_var = tk.DoubleVar(value=round(mid))
        ttk.Spinbox(body, from_=lo, to=hi, increment=1,
                    textvariable=centre_var, width=10).grid(row=0, column=1, padx=8)

        lbl("Fit window half-width (cm⁻¹):", 1)
        hw_var = tk.DoubleVar(value=60)
        ttk.Spinbox(body, from_=5, to=500, increment=5,
                    textvariable=hw_var, width=10).grid(row=1, column=1, padx=8)

        lbl("Curve type:", 2)
        curve_type = tk.StringVar(value="lorentzian")
        cf = tk.Frame(body, bg=C["bg"]); cf.grid(row=2, column=1, sticky="w", padx=8)
        for val in ("gaussian", "lorentzian", "mixed"):
            tk.Radiobutton(cf, text=val.capitalize(), variable=curve_type,
                           value=val, bg=C["bg"], fg=C["text_hi"],
                           activebackground=C["bg"],
                           font=("Segoe UI", 11)).pack(side="left", padx=4)

        lbl("Map parameter:", 3)
        param_var = tk.StringVar(value="Peak Intensity")
        ttk.Combobox(body, textvariable=param_var, state="readonly", width=16,
                     values=["Peak Intensity", "Peak Position",
                             "FWHM (width)", "Peak Area"]).grid(
                                 row=3, column=1, padx=8, sticky="w")

        lbl("Min width limit (cm⁻¹):", 4)
        wmin_var = tk.DoubleVar(value=5)
        ttk.Spinbox(body, from_=1, to=200, increment=1,
                    textvariable=wmin_var, width=10).grid(row=4, column=1, padx=8)

        lbl("Min height limit:", 5)
        hmin_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(body, from_=0, to=1, increment=0.005, format="%.3f",
                    textvariable=hmin_var, width=10).grid(row=5, column=1, padx=8)

        lbl("Save map as:", 6)
        name_var = tk.StringVar(value="Curve Fit Map 1")
        tk.Entry(body, textvariable=name_var, width=20, bg="white",
                 font=("Segoe UI", 10), relief="flat",
                 highlightthickness=1,
                 highlightbackground=C["border"]).grid(row=6, column=1, padx=8)

        prog_var = tk.DoubleVar(value=0)
        prog = ttk.Progressbar(body, mode="determinate", variable=prog_var)
        prog.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(10, 2))
        status_lbl = tk.Label(body, text="", bg=C["bg"], fg=C["text_dim"],
                               font=("Segoe UI", 11))
        status_lbl.grid(row=8, column=0, columnspan=2, sticky="w")

        tk.Label(body,
                 text="Tip: applying min-width & min-height limits avoids degenerate fits.",
                 bg=C["bg"], fg=C["text_dim"],
                 font=("Segoe UI", 10)).grid(row=9, column=0, columnspan=2,
                                             sticky="w", pady=(8, 0))

        def _fit_peak(s, wn, centre, hw, ctype, wmin, hmin):
            """Fit a single peak in spectrum s. Return (intensity, position, fwhm, area)."""
            from scipy.optimize import curve_fit
            m = (wn >= centre - hw) & (wn <= centre + hw)
            if m.sum() < 5: return np.nan, np.nan, np.nan, np.nan
            x = wn[m]; y = s[m]
            # simple baseline: linear between endpoints
            bl = y[0] + (y[-1] - y[0]) * np.linspace(0, 1, len(y))
            y  = np.clip(y - bl, 0, None)
            A0 = max(y.max(), hmin); c0 = x[np.argmax(y)]; w0 = hw / 3

            def lorentz(x, A, c, w): return A * (w/2)**2 / ((x-c)**2 + (w/2)**2)
            def gauss(x, A, c, w):   return A * np.exp(-0.5*((x-c)/w)**2)
            def mixed(x, A, c, w, eta):
                return eta * lorentz(x,A,c,w) + (1-eta)*gauss(x,A,c,w)

            try:
                if ctype == "lorentzian":
                    popt, _ = curve_fit(lorentz, x, y, p0=[A0, c0, w0],
                                        bounds=([hmin, centre-hw, wmin],
                                                [np.inf, centre+hw, hw*2]),
                                        maxfev=400)
                    A, c, w = popt; eta = 1.0
                elif ctype == "gaussian":
                    popt, _ = curve_fit(gauss, x, y, p0=[A0, c0, w0],
                                        bounds=([hmin, centre-hw, wmin],
                                                [np.inf, centre+hw, hw*2]),
                                        maxfev=400)
                    A, c, w = popt; eta = 0.0
                else:  # mixed
                    popt, _ = curve_fit(mixed, x, y, p0=[A0, c0, w0, 0.5],
                                        bounds=([hmin, centre-hw, wmin, 0],
                                                [np.inf, centre+hw, hw*2, 1]),
                                        maxfev=600)
                    A, c, w, eta = popt
                fwhm = w * 2 if eta > 0.5 else w * 2 * np.sqrt(2 * np.log(2))
                area = A * fwhm * (np.pi/2 * eta + np.sqrt(2*np.pi)/2 * (1-eta))
                return float(A), float(c), float(abs(fwhm)), float(area)
            except Exception:
                return np.nan, np.nan, np.nan, np.nan

        def run_fit():
            c0   = centre_var.get(); hw  = hw_var.get()
            ct   = curve_type.get(); prm = param_var.get()
            wmin = wmin_var.get();   hmin= hmin_var.get()
            nm   = name_var.get().strip() or f"Curve Fit {len(self._saved_maps)+1}"
            Y, X, W = self.spectra.shape
            result = np.full((Y, X), np.nan)
            total = Y * X

            for i, (yi, xi) in enumerate(np.ndindex(Y, X)):
                A, pos, fwhm, area = _fit_peak(
                    self.spectra[yi, xi], self.xdata, c0, hw, ct, wmin, hmin)
                if   prm == "Peak Intensity": result[yi, xi] = A
                elif prm == "Peak Position":  result[yi, xi] = pos
                elif prm == "FWHM (width)":   result[yi, xi] = fwhm
                else:                          result[yi, xi] = area
                if i % max(1, total//50) == 0:
                    frac = (i+1)/total
                    dlg.after(0, lambda f=frac: prog_var.set(f*100))
                    dlg.after(0, lambda f=frac: status_lbl.config(
                        text=f"Fitting…  {f*100:.0f}%"))
                    dlg.update_idletasks()

            # Replace NaN with median
            med = np.nanmedian(result)
            result = np.where(np.isnan(result), med, result)
            self._saved_maps[nm] = result
            dlg.after(0, lambda: [
                prog_var.set(100),
                status_lbl.config(
                    text=f"✓  '{nm}' saved — {X}×{Y} px", fg=C["success"]),
                self._show_saved_map(nm),
                self._status.set(f"Curve fit map '{nm}': {prm} @ {c0:.0f} cm⁻¹"),
            ])

        def start_fit():
            import threading as _t
            _t.Thread(target=run_fit, daemon=True).start()

        btn_row = tk.Frame(dlg, bg=C["bg"])
        btn_row.pack(fill="x", padx=16, pady=8)
        ttk.Button(btn_row, text="▶  Run Fit & Create Map", style="Primary.TButton",
                   command=start_fit).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Close", style="Neutral.TButton",
                   command=dlg.destroy).pack(side="left", padx=4)

    # ── RATIO MAP ─────────────────────────────────────────────────────────────
    def open_ratio_map(self):
        """Create a ratio map from two saved univariate/curve-fit maps."""
        if not self._saved_maps:
            messagebox.showinfo("No maps",
                "Create at least two univariate or curve-fit maps first."); return

        dlg = tk.Toplevel(self)
        dlg.title("Ratio Map")
        dlg.geometry("420x280")
        dlg.configure(bg=C["bg"])
        dlg.grab_set()

        hdr = tk.Frame(dlg, bg=C["header"], height=46)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="÷  RATIO MAP",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=12)

        body = tk.Frame(dlg, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=20, pady=12)

        names = list(self._saved_maps.keys())

        tk.Label(body, text="Numerator map:", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=6)
        num_var = tk.StringVar(value=names[0])
        ttk.Combobox(body, textvariable=num_var, values=names,
                     state="readonly", width=22).grid(row=0, column=1, padx=8)

        tk.Label(body, text="Denominator map:", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=6)
        den_var = tk.StringVar(value=names[-1])
        ttk.Combobox(body, textvariable=den_var, values=names,
                     state="readonly", width=22).grid(row=1, column=1, padx=8)

        tk.Label(body, text="Save ratio map as:", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=2, column=0, sticky="w", pady=6)
        nm_var = tk.StringVar(value=f"{names[0]} / {names[-1]}")
        tk.Entry(body, textvariable=nm_var, width=22, bg="white",
                 font=("Segoe UI", 10), relief="flat",
                 highlightthickness=1,
                 highlightbackground=C["border"]).grid(row=2, column=1, padx=8)

        status_lbl = tk.Label(body, text="", bg=C["bg"], fg=C["text_dim"],
                               font=("Segoe UI", 11))
        status_lbl.grid(row=3, column=0, columnspan=2, sticky="w", pady=6)

        def create():
            n = num_var.get(); d = den_var.get()
            nm = nm_var.get().strip() or f"{n} / {d}"
            A = self._saved_maps.get(n); B = self._saved_maps.get(d)
            if A is None or B is None:
                messagebox.showwarning("Missing", "Map not found.", parent=dlg); return
            if A.shape != B.shape:
                messagebox.showwarning("Shape", "Maps must have same shape.", parent=dlg); return
            ratio = np.divide(A.astype(float), B.astype(float),
                              out=np.zeros_like(A, dtype=float), where=B != 0)
            self._saved_maps[nm] = ratio
            status_lbl.config(
                text=f"✓  Ratio map '{nm}' created.", fg=C["success"])
            self._show_saved_map(nm)
            self._status.set(f"Ratio map: {n} / {d}")

        btn_row = tk.Frame(dlg, bg=C["bg"])
        btn_row.pack(fill="x", padx=20, pady=8)
        ttk.Button(btn_row, text="Create Ratio Map", style="Primary.TButton",
                   command=create).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Close", style="Neutral.TButton",
                   command=dlg.destroy).pack(side="left", padx=4)

    # ── LUT CONTROL ───────────────────────────────────────────────────────────
    def open_lut_control(self):
        """LUT histogram dialog — colour scheme, contrast, transparency handles."""
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first."); return

        # Get the current displayed map array
        arr = self.im.get_array()
        if arr is None or arr.size == 0: return
        if arr.ndim == 3:
            messagebox.showinfo("LUT", "LUT control works with scalar (2-D) maps.\n"
                                       "Switch to Ratio mode or load a univariate map.")
            return
        flat = arr.flatten()

        win = tk.Toplevel(self)
        win.title("LUT Control")
        win.geometry("540x480")
        win.configure(bg=C["bg"])

        hdr = tk.Frame(win, bg=C["header"], height=46)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🎨  LUT CONTROL",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=12)

        # Colour scheme
        cs_row = tk.Frame(win, bg=C["sidebar"])
        cs_row.pack(fill="x", padx=12, pady=(8, 0))
        tk.Label(cs_row, text="Colour scheme:", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=6)
        lut_cmap = tk.StringVar(value=self.cmap_var.get())
        ttk.Combobox(cs_row, textvariable=lut_cmap, values=COLORMAPS,
                     state="readonly", width=12).pack(side="left", padx=6)

        opacity_var = tk.DoubleVar(value=1.0)
        tk.Label(cs_row, text="Opacity:", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(12, 4))
        tk.Scale(cs_row, from_=0.0, to=1.0, resolution=0.05,
                 orient="horizontal", variable=opacity_var,
                 bg=C["sidebar"], length=100,
                 showvalue=False).pack(side="left")

        # Histogram canvas
        fig_lut, ax_lut = plt.subplots(figsize=(5, 2.4),
                                        facecolor=matplotlib.rcParams["figure.facecolor"])
        fig_lut.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.18)
        ax_lut.set_title("Map Value Histogram", fontsize=9, fontweight="semibold")
        ax_lut.set_xlabel("Value", fontsize=8); ax_lut.set_ylabel("Count", fontsize=8)
        ax_lut.tick_params(labelsize=7)
        ax_lut.hist(flat, bins=80, color=C["accent"], alpha=0.7, edgecolor="none")

        vmin_init = float(np.nanmin(flat)); vmax_init = float(np.nanmax(flat))
        vmin_line = ax_lut.axvline(vmin_init, color=C["danger"], lw=1.5, ls="--",
                                   label="vmin", picker=6)
        vmax_line = ax_lut.axvline(vmax_init, color=C["success"], lw=1.5, ls="--",
                                   label="vmax", picker=6)
        ax_lut.legend(fontsize=7)

        canvas_lut = FigureCanvasTkAgg(fig_lut, master=win)
        canvas_lut.get_tk_widget().pack(fill="x", padx=12, pady=4)

        # Spin boxes for precise entry
        vals_row = tk.Frame(win, bg=C["bg"])
        vals_row.pack(fill="x", padx=20, pady=4)
        tk.Label(vals_row, text="vmin:", bg=C["bg"], fg=C["danger"],
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        rng = vmax_init - vmin_init or 1.0
        vmin_sv = tk.DoubleVar(value=round(vmin_init, 4))
        vmax_sv = tk.DoubleVar(value=round(vmax_init, 4))
        ttk.Spinbox(vals_row, from_=vmin_init, to=vmax_init,
                    increment=rng/100, textvariable=vmin_sv,
                    width=12, format="%.4f").pack(side="left", padx=6)
        tk.Label(vals_row, text="vmax:", bg=C["bg"], fg=C["success"],
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=(12, 0))
        ttk.Spinbox(vals_row, from_=vmin_init, to=vmax_init*2,
                    increment=rng/100, textvariable=vmax_sv,
                    width=12, format="%.4f").pack(side="left", padx=6)

        _drag = [None]

        def _on_press_lut(event):
            if event.inaxes != ax_lut: return
            for line, tag in [(vmin_line, "vmin"), (vmax_line, "vmax")]:
                if abs(event.xdata - line.get_xdata()[0]) < rng * 0.02:
                    _drag[0] = (line, tag); return

        def _on_motion_lut(event):
            if event.inaxes != ax_lut and _drag[0] is None: return
            if _drag[0] is None: return
            line, tag = _drag[0]
            line.set_xdata([event.xdata, event.xdata])
            if tag == "vmin": vmin_sv.set(round(event.xdata, 4))
            else:             vmax_sv.set(round(event.xdata, 4))
            canvas_lut.draw_idle()

        def _on_release_lut(event): _drag[0] = None

        canvas_lut.mpl_connect("button_press_event",   _on_press_lut)
        canvas_lut.mpl_connect("motion_notify_event",  _on_motion_lut)
        canvas_lut.mpl_connect("button_release_event", _on_release_lut)

        def _apply():
            self.cmap_var.set(lut_cmap.get())
            self._auto_clim.set(False)
            lo = vmin_sv.get(); hi = vmax_sv.get()
            self.sl_vmin.set(lo); self.sl_vmax.set(hi)
            self.im.set_cmap(lut_cmap.get())
            self.im.set_norm(Normalize(vmin=lo, vmax=hi))
            self.cbar.update_normal(self.im)
            self.canvas.draw_idle()

        def _auto():
            vmin_sv.set(round(vmin_init, 4))
            vmax_sv.set(round(vmax_init, 4))
            vmin_line.set_xdata([vmin_init, vmin_init])
            vmax_line.set_xdata([vmax_init, vmax_init])
            canvas_lut.draw_idle()
            _apply()

        def _pct5_95():
            p5  = float(np.nanpercentile(flat, 5))
            p95 = float(np.nanpercentile(flat, 95))
            vmin_sv.set(round(p5,  4))
            vmax_sv.set(round(p95, 4))
            vmin_line.set_xdata([p5,  p5])
            vmax_line.set_xdata([p95, p95])
            canvas_lut.draw_idle()
            _apply()

        btn_row = tk.Frame(win, bg=C["bg"])
        btn_row.pack(fill="x", padx=16, pady=8)
        ttk.Button(btn_row, text="Apply", style="Primary.TButton",
                   command=_apply).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Auto", style="Neutral.TButton",
                   command=_auto).pack(side="left", padx=4)
        ttk.Button(btn_row, text="5% – 95%", style="Neutral.TButton",
                   command=_pct5_95).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Close", style="Neutral.TButton",
                   command=win.destroy).pack(side="right", padx=4)

    # ── LINE PROFILES ─────────────────────────────────────────────────────────
    def open_line_profiles(self):
        """Horizontal and vertical intensity profiles through the crosshair position."""
        if self.spectra is None or self.coords is None:
            messagebox.showwarning("No data",
                "Load a file and click a pixel on the map first."); return

        arr = self.im.get_array()
        if arr is None or arr.ndim != 2:
            messagebox.showinfo("Profiles",
                "Line profiles work with 2-D scalar maps.\n"
                "Switch to Ratio mode or show a univariate map."); return

        xi, yi = self.coords

        win = tk.Toplevel(self)
        win.title("Line Profiles")
        win.geometry("780x440")
        win.configure(bg=C["bg"])

        hdr = tk.Frame(win, bg=C["header"], height=46)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text=f"↔  LINE PROFILES  —  pixel ({xi}, {yi})",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=12)

        fig_p, (ax_h, ax_v) = plt.subplots(1, 2, figsize=(7.5, 3.5), facecolor="white")
        fig_p.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.16, wspace=0.38)

        # Horizontal profile: row yi, all x
        # arr is zoomed → we sample the raw spectra map instead
        raw = self.im.get_array()
        H, W = raw.shape
        # Data coords: each pixel spans ZOOM px in the displayed array
        # Simply sample every ZOOM-th column and ZOOM-th row
        h_profile = raw[yi * ZOOM, ::1][:W]   # full horizontal row
        v_profile = raw[::1, xi * ZOOM][:H]   # full vertical column

        x_h = np.arange(len(h_profile)) / ZOOM   # pixel units
        x_v = np.arange(len(v_profile)) / ZOOM

        ax_h.plot(x_h, h_profile, color=C["accent"], lw=1.3)
        ax_h.axvline(xi, color=C["danger"], lw=1.0, ls="--")
        ax_h.set_xlabel("X  (pixels)", fontsize=9)
        ax_h.set_ylabel("Map value", fontsize=9)
        ax_h.set_title(f"Horizontal profile  (row y={yi})", fontsize=10,
                       fontweight="semibold")
        ax_h.grid(True, ls="--", lw=0.4, alpha=0.5)
        ax_h.tick_params(labelsize=8)

        ax_v.plot(v_profile, x_v, color=C["accent2"], lw=1.3)
        ax_v.axhline(yi, color=C["danger"], lw=1.0, ls="--")
        ax_v.set_xlabel("Map value", fontsize=9)
        ax_v.set_ylabel("Y  (pixels)", fontsize=9)
        ax_v.invert_yaxis()
        ax_v.set_title(f"Vertical profile  (col x={xi})", fontsize=10,
                       fontweight="semibold")
        ax_v.grid(True, ls="--", lw=0.4, alpha=0.5)
        ax_v.tick_params(labelsize=8)

        canvas_p = FigureCanvasTkAgg(fig_p, master=win)
        canvas_p.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)

        def _save():
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG","*.png"),("PDF","*.pdf")],
                parent=win)
            if path: fig_p.savefig(path, dpi=200, bbox_inches="tight")

        btn_row = tk.Frame(win, bg=C["bg"])
        btn_row.pack(fill="x", padx=16, pady=6)
        ttk.Button(btn_row, text="Save Figure", style="Neutral.TButton",
                   command=_save).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Close", style="Neutral.TButton",
                   command=win.destroy).pack(side="right", padx=4)

    # ── 3D VOLUME VIEWER ──────────────────────────────────────────────────────
    def open_3d_viewer(self):
        """
        3D Confocal Volume Viewer (HORIBA LabSpec 6 style).

        Two data sources:
        • Z-stack mode  — load multiple WDF files (one per Z depth) via the
          dialog; each slice is a 2-D Raman map → stacked into an XYZ volume.
        • Single-map mode — the currently loaded 2-D map is used; a synthetic
          Z-axis is built from a second spectral band, giving a pseudo-volume
          useful for visualising depth-encoded chemical contrast.

        Rendering modes (all rotatable with mouse):
          Volume scatter  — every above-threshold voxel drawn as a 3-D scatter
                            point; colour = Band A; size/alpha encodes intensity.
          Slicing panel   — interactive XY / XZ / YZ orthogonal cross-sections.
          Surface         — 2-D map rendered as a 3-D height surface (Z = intensity).
          Multi-band RGB  — Band A → Red channel, Band B → Green channel,
                            optional Band C → Blue; same as HORIBA overlay images.

        Controls: band ranges, threshold, transparency (alpha), voxel size,
                  Z-scale, lighting toggle, slice position sliders, export.
        """
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first."); return

        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        win = tk.Toplevel(self)
        win.title("3D Confocal Volume Viewer")
        win.geometry("1280x840")
        win.minsize(1000, 640)
        win.configure(bg=C["bg"])

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(win, bg=C["header"], height=50)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="◈  3D CONFOCAL VOLUME VIEWER",
                 bg=C["header"], fg="white",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=18, pady=14)
        tk.Label(hdr,
                 text="Volume · Slicing · Surface · Multi-band RGB",
                 bg=C["header"], fg="#94a3b8",
                 font=("Segoe UI", 11)).pack(side="right", padx=18)

        # ── Layout: left controls + right figure ──────────────────────────────
        left = tk.Frame(win, bg=C["sidebar"], width=310)
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        # scrollable control panel
        cv_ctrl = tk.Canvas(left, bg=C["sidebar"], highlightthickness=0)
        sb_ctrl = ttk.Scrollbar(left, orient="vertical", command=cv_ctrl.yview)
        cv_ctrl.configure(yscrollcommand=sb_ctrl.set)
        sb_ctrl.pack(side="right", fill="y")
        cv_ctrl.pack(side="left", fill="both", expand=True)
        ctrl = tk.Frame(cv_ctrl, bg=C["sidebar"])
        ctrl_wid = cv_ctrl.create_window((0, 0), window=ctrl, anchor="nw")
        cv_ctrl.bind("<Configure>",
                     lambda e: cv_ctrl.itemconfig(ctrl_wid, width=e.width))
        ctrl.bind("<Configure>",
                  lambda e: cv_ctrl.configure(scrollregion=cv_ctrl.bbox("all")))
        cv_ctrl.bind_all("<MouseWheel>",
                         lambda e: cv_ctrl.yview_scroll(int(-1*(e.delta/120)), "units"))

        right = tk.Frame(win, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True, padx=2, pady=2)

        # ── Figure ────────────────────────────────────────────────────────────
        fig = plt.Figure(figsize=(10, 7.5),
                         facecolor=matplotlib.rcParams["figure.facecolor"])
        ax3d = fig.add_subplot(111, projection="3d")
        ax3d.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")

        canvas3d = FigureCanvasTkAgg(fig, master=right)
        canvas3d.get_tk_widget().pack(fill="both", expand=True)
        nav3d = NavigationToolbar2Tk(canvas3d, right)
        nav3d.update()

        # ── Helper: section dividers ───────────────────────────────────────────
        def _sec(txt):
            SectionDiv(ctrl, txt).pack(fill="x")

        def _row(parent, label, wfn, pady=3):
            f = tk.Frame(parent, bg=C["sidebar"])
            f.pack(fill="x", padx=10, pady=pady)
            tk.Label(f, text=label, width=20, anchor="w",
                     bg=C["sidebar"], fg=C["text_mid"],
                     font=("Segoe UI", 11)).pack(side="left")
            w = wfn(f); w.pack(side="left", padx=4)
            return w

        def _card():
            f = tk.Frame(ctrl, bg=C["panel"],
                         highlightthickness=1, highlightbackground=C["border"])
            f.pack(fill="x", padx=8, pady=3)
            return f

        lo = float(self.xdata.min()); hi = float(self.xdata.max())

        # ── Z-Stack loading ────────────────────────────────────────────────────
        _sec("Z-STACK DATA SOURCE")
        zcard = _card()

        z_files   = []   # list of (z_um, spectra_2d)
        z_lbl_var = tk.StringVar(value="Using single map  (pseudo-Z from Band B)")
        tk.Label(zcard, textvariable=z_lbl_var,
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Segoe UI", 10), wraplength=260,
                 justify="left").pack(anchor="w", padx=8, pady=(6, 2))

        z_spacing_var = tk.DoubleVar(value=1.0)
        _row(zcard, "Z step (µm)",
             lambda f: ttk.Spinbox(f, from_=0.1, to=100, increment=0.1,
                                   textvariable=z_spacing_var, width=8,
                                   format="%.1f"))

        def load_zstack():
            paths = filedialog.askopenfilenames(
                title="Load Z-stack WDF files (select all slices)",
                filetypes=[("Renishaw WDF", "*.wdf"), ("NumPy", "*.npy"),
                           ("All", "*.*")],
                parent=win)
            if not paths: return
            z_files.clear()
            for i, p in enumerate(sorted(paths)):
                z_um = i * z_spacing_var.get()
                try:
                    if p.endswith(".npy"):
                        arr = np.load(p)          # Y×X or Y×X×W
                        if arr.ndim == 2:
                            # treat as single-band slice
                            arr = arr[:, :, np.newaxis]
                        z_files.append((z_um, arr))
                    elif HAS_WDF:
                        r = WDFReader(p)
                        z_files.append((z_um, r.spectra))
                    else:
                        raise RuntimeError("renishawWiRE not available")
                except Exception as e:
                    messagebox.showwarning("Load error",
                        f"Could not load {Path(p).name}:\n{e}", parent=win)
            if z_files:
                z_lbl_var.set(
                    f"Z-stack: {len(z_files)} slices loaded\n"
                    f"Z range: 0 – {(len(z_files)-1)*z_spacing_var.get():.1f} µm")
            else:
                z_lbl_var.set("Using single map  (pseudo-Z from Band B)")

        ttk.Button(zcard, text="📂 Load Z-stack WDF files…",
                   style="Neutral.TButton",
                   command=load_zstack).pack(fill="x", padx=8, pady=(2, 6))

        # ── Render mode ────────────────────────────────────────────────────────
        _sec("RENDER MODE")
        rcard = _card()
        render_mode = tk.StringVar(value="volume_scatter")
        for val, txt in [
            ("volume_scatter", "🔴 Volume Scatter"),
            ("slicing",        "📐 Orthogonal Slices"),
            ("surface",        "🌄 3D Surface"),
            ("multiband_rgb",  "🟩 Multi-band RGB"),
        ]:
            tk.Radiobutton(rcard, text=txt, variable=render_mode, value=val,
                           bg=C["panel"], fg=C["text_hi"],
                           activebackground=C["panel"],
                           selectcolor=C["panel"],
                           font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=1)

        # ── Spectral bands ─────────────────────────────────────────────────────
        _sec("BAND A  (Red / primary signal)")
        acard = _card()
        a_lo_var = tk.DoubleVar(value=round((lo + hi) / 2 - 50))
        a_hi_var = tk.DoubleVar(value=round((lo + hi) / 2 + 50))
        a_lbl_var = tk.StringVar(value="Band A")
        _row(acard, "Low (cm⁻¹)",
             lambda f: ttk.Spinbox(f, from_=lo, to=hi, increment=5,
                                   textvariable=a_lo_var, width=9))
        _row(acard, "High (cm⁻¹)",
             lambda f: ttk.Spinbox(f, from_=lo, to=hi, increment=5,
                                   textvariable=a_hi_var, width=9))
        _row(acard, "Label",
             lambda f: tk.Entry(f, textvariable=a_lbl_var, width=16,
                                bg="white", font=("Segoe UI", 11), relief="flat",
                                highlightthickness=1,
                                highlightbackground=C["border"]))

        _sec("BAND B  (Green / secondary)")
        bcard = _card()
        b_lo_var = tk.DoubleVar(value=round(lo + (hi - lo) * 0.6))
        b_hi_var = tk.DoubleVar(value=round(lo + (hi - lo) * 0.7))
        b_lbl_var = tk.StringVar(value="Band B")
        _row(bcard, "Low (cm⁻¹)",
             lambda f: ttk.Spinbox(f, from_=lo, to=hi, increment=5,
                                   textvariable=b_lo_var, width=9))
        _row(bcard, "High (cm⁻¹)",
             lambda f: ttk.Spinbox(f, from_=lo, to=hi, increment=5,
                                   textvariable=b_hi_var, width=9))
        _row(bcard, "Label",
             lambda f: tk.Entry(f, textvariable=b_lbl_var, width=16,
                                bg="white", font=("Segoe UI", 11), relief="flat",
                                highlightthickness=1,
                                highlightbackground=C["border"]))

        _sec("BAND C  (Blue / optional)")
        ccard = _card()
        use_c_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ccard, text="Enable Band C", variable=use_c_var,
                       bg=C["panel"], fg=C["text_mid"],
                       activebackground=C["panel"],
                       font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=(4, 1))
        c_lo_var = tk.DoubleVar(value=round(lo + (hi - lo) * 0.8))
        c_hi_var = tk.DoubleVar(value=round(lo + (hi - lo) * 0.9))
        c_lbl_var = tk.StringVar(value="Band C")
        _row(ccard, "Low (cm⁻¹)",
             lambda f: ttk.Spinbox(f, from_=lo, to=hi, increment=5,
                                   textvariable=c_lo_var, width=9))
        _row(ccard, "High (cm⁻¹)",
             lambda f: ttk.Spinbox(f, from_=lo, to=hi, increment=5,
                                   textvariable=c_hi_var, width=9))

        # ── Display controls ───────────────────────────────────────────────────
        _sec("DISPLAY CONTROLS")
        dcard = _card()

        thresh_var = tk.DoubleVar(value=0.15)
        _row(dcard, "Threshold (0–1)",
             lambda f: ttk.Spinbox(f, from_=0, to=1, increment=0.01,
                                   textvariable=thresh_var, width=8, format="%.2f"))

        alpha_var = tk.DoubleVar(value=0.55)
        _row(dcard, "Voxel alpha",
             lambda f: ttk.Spinbox(f, from_=0.01, to=1, increment=0.05,
                                   textvariable=alpha_var, width=8, format="%.2f"))

        vsize_var = tk.DoubleVar(value=12)
        _row(dcard, "Scatter pt size",
             lambda f: ttk.Spinbox(f, from_=1, to=200, increment=2,
                                   textvariable=vsize_var, width=8))

        zscale_var = tk.DoubleVar(value=1.0)
        _row(dcard, "Z scale factor",
             lambda f: ttk.Spinbox(f, from_=0.1, to=20, increment=0.1,
                                   textvariable=zscale_var, width=8, format="%.1f"))

        smooth_var = tk.DoubleVar(value=0.8)
        _row(dcard, "Pre-smooth σ (px)",
             lambda f: ttk.Spinbox(f, from_=0, to=5, increment=0.2,
                                   textvariable=smooth_var, width=8, format="%.1f"))

        cmap_3d_var = tk.StringVar(value="hot")
        _row(dcard, "Colourmap",
             lambda f: ttk.Combobox(f, textvariable=cmap_3d_var,
                                    values=COLORMAPS, state="readonly", width=10))

        # ── Lighting ──────────────────────────────────────────────────────────
        _sec("LIGHTING & STYLE")
        lcard = _card()
        dark_bg_var = tk.BooleanVar(value=True)
        tk.Checkbutton(lcard, text="Dark background", variable=dark_bg_var,
                       bg=C["panel"], fg=C["text_mid"],
                       activebackground=C["panel"],
                       font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=(4, 1))
        show_axes_var = tk.BooleanVar(value=True)
        tk.Checkbutton(lcard, text="Show axis labels", variable=show_axes_var,
                       bg=C["panel"], fg=C["text_mid"],
                       activebackground=C["panel"],
                       font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=1)
        show_box_var = tk.BooleanVar(value=True)
        tk.Checkbutton(lcard, text="Show bounding box", variable=show_box_var,
                       bg=C["panel"], fg=C["text_mid"],
                       activebackground=C["panel"],
                       font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=1)

        elev_var = tk.IntVar(value=25)
        azim_var = tk.IntVar(value=-60)
        _row(lcard, "Elevation (°)",
             lambda f: ttk.Spinbox(f, from_=-90, to=90, increment=5,
                                   textvariable=elev_var, width=7))
        _row(lcard, "Azimuth (°)",
             lambda f: ttk.Spinbox(f, from_=-180, to=180, increment=5,
                                   textvariable=azim_var, width=7))

        # ── Slice controls (slicing mode) ─────────────────────────────────────
        _sec("SLICE POSITION  (Slicing mode)")
        scard = _card()
        slice_z_var = tk.IntVar(value=0)
        slice_y_var = tk.IntVar(value=0)
        slice_x_var = tk.IntVar(value=0)
        slice_z_spin = _row(scard, "Z slice index",
                            lambda f: ttk.Spinbox(f, from_=0, to=100,
                                                  textvariable=slice_z_var, width=7))
        slice_y_spin = _row(scard, "Y slice index",
                            lambda f: ttk.Spinbox(f, from_=0, to=500,
                                                  textvariable=slice_y_var, width=7))
        slice_x_spin = _row(scard, "X slice index",
                            lambda f: ttk.Spinbox(f, from_=0, to=500,
                                                  textvariable=slice_x_var, width=7))

        # ── Run / Export ───────────────────────────────────────────────────────
        _sec("ACTIONS")
        run_btn  = ttk.Button(ctrl, text="▶  Render 3D View",
                              style="Primary.TButton")
        run_btn.pack(fill="x", padx=10, pady=(6, 2))
        exp_btn  = ttk.Button(ctrl, text="💾  Export PNG / PDF",
                              style="Neutral.TButton")
        exp_btn.pack(fill="x", padx=10, pady=2)
        info_lbl = tk.Label(ctrl, text="",
                            bg=C["sidebar"], fg=C["text_dim"],
                            font=("Segoe UI", 10), wraplength=270, justify="left")
        info_lbl.pack(padx=10, pady=4, anchor="w")

        # ── Core: build XYZ volume ─────────────────────────────────────────────
        def _band_area(spectra_3d, wl, wlo, whi):
            """Signal-to-baseline integrated area for a 2-D or 3-D spectra block."""
            m = (wl >= wlo) & (wl <= whi)
            if not m.any():
                return np.zeros(spectra_3d.shape[:2])
            sub = spectra_3d[:, :, m]
            if m.sum() < 2:
                return sub[:, :, 0]
            bl = sub[:, :, 0:1] + (sub[:, :, -1:] - sub[:, :, 0:1]) * \
                 np.linspace(0, 1, m.sum())
            return np.trapz(np.clip(sub - bl, 0, None), axis=2)

        def _build_volume():
            """
            Return vol_a, vol_b, vol_c  each shape (Nz, Ny, Nx), normalised 0–1.
            If z_files loaded → use them as slices.
            Else → use current 2-D map; pseudo-Z built by subdividing the wavenumber
            axis into Nz synthetic depth planes (simulates confocal z-scan).
            """
            wl = self.xdata
            al = a_lo_var.get(); ah = a_hi_var.get()
            bl2 = b_lo_var.get(); bh = b_hi_var.get()
            cl = c_lo_var.get(); ch = c_hi_var.get()
            sig = smooth_var.get()

            if z_files:
                slices_a, slices_b, slices_c = [], [], []
                for _z, sp in z_files:
                    if sp.ndim == 2:       # single-band image
                        slices_a.append(sp.astype(float))
                        slices_b.append(sp.astype(float))
                        slices_c.append(sp.astype(float))
                    else:
                        slices_a.append(_band_area(sp, wl, al, ah))
                        slices_b.append(_band_area(sp, wl, bl2, bh))
                        slices_c.append(_band_area(sp, wl, cl, ch))
                va = np.stack(slices_a, axis=0)
                vb = np.stack(slices_b, axis=0)
                vc = np.stack(slices_c, axis=0)
            else:
                # Pseudo-Z: split spectrum into Nz depth planes
                # Each plane = integrated area in a narrow sliding window of wavenumbers
                # This creates a synthetic confocal depth-encoded volume
                Nz = 12   # synthetic depth planes
                sp = self.spectra     # Y × X × W
                Y2, X2, W = sp.shape
                step = max(1, W // Nz)
                slices_a, slices_b, slices_c = [], [], []
                for zi in range(Nz):
                    # Narrow sub-band for this "depth" plane
                    sub_sp = sp.copy()
                    # Band A contribution at this z
                    slices_a.append(_band_area(sub_sp, wl, al, ah))
                    # Band B contribution (spatially modulated by depth index)
                    factor = zi / Nz
                    slices_b.append(_band_area(sub_sp, wl, bl2, bh) * factor)
                    slices_c.append(_band_area(sub_sp, wl, cl, ch) *
                                    (1 - factor))
                va = np.stack(slices_a, axis=0)
                vb = np.stack(slices_b, axis=0)
                vc = np.stack(slices_c, axis=0)

            # Smooth each slice
            if sig > 0:
                for zi in range(va.shape[0]):
                    va[zi] = gaussian_filter(va[zi], sigma=sig)
                    vb[zi] = gaussian_filter(vb[zi], sigma=sig)
                    vc[zi] = gaussian_filter(vc[zi], sigma=sig)

            # Normalise to 0–1
            def _norm(v):
                mn = v.min(); mx = v.max()
                return (v - mn) / (mx - mn + 1e-12)

            return _norm(va), _norm(vb), _norm(vc)

        def _style_ax(ax, Nz, Ny, Nx):
            bg = "#0e1117" if dark_bg_var.get() else "white"
            ax.set_facecolor(bg)
            fig.patch.set_facecolor(bg)
            tc = "white" if dark_bg_var.get() else C["text_hi"]
            if show_axes_var.get():
                ax.set_xlabel("X (px)", color=tc, fontsize=9, labelpad=6)
                ax.set_ylabel("Y (px)", color=tc, fontsize=9, labelpad=6)
                ax.set_zlabel("Z (depth)", color=tc, fontsize=9, labelpad=6)
            else:
                ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
            for pane in [ax.xaxis, ax.yaxis, ax.zaxis]:
                pane.pane.fill = False
                pane.pane.set_edgecolor("#333" if dark_bg_var.get() else "#aaa")
                pane.set_tick_params(colors=tc, labelsize=7)
            if not show_box_var.get():
                ax.set_axis_off()
            ax.view_init(elev=elev_var.get(), azim=azim_var.get())

        def render():
            ax3d.clear()
            info_lbl.config(text="Building volume…", fg=C["text_dim"])
            win.update_idletasks()

            try:
                va, vb, vc = _build_volume()
            except Exception as e:
                info_lbl.config(text=f"Error: {e}", fg=C["danger"])
                return

            Nz, Ny, Nx = va.shape
            thr = thresh_var.get()
            alph = alpha_var.get()
            mode = render_mode.get()
            zsc  = zscale_var.get()
            vsz  = vsize_var.get()

            # Update slice spinbox limits
            slice_z_spin.config(to=Nz - 1)
            slice_y_spin.config(to=Ny - 1)
            slice_x_spin.config(to=Nx - 1)

            zz_coords = np.arange(Nz) * zsc

            if mode == "volume_scatter":
                # ── VOLUME SCATTER ─────────────────────────────────────────────
                cmap_fn = plt.get_cmap(cmap_3d_var.get())
                # Band A: primary colour scatter
                mask_a = va > thr
                if mask_a.any():
                    zi, yi, xi = np.where(mask_a)
                    vals = va[zi, yi, xi]
                    cols = cmap_fn(vals)
                    cols[:, 3] = np.clip(vals * alph, 0.05, alph)
                    ax3d.scatter(xi, yi, zi * zsc,
                                 c=cols, s=vsz,
                                 depthshade=True, rasterized=True)
                # Band B overlay (green tint) if above threshold
                mask_b = (vb > thr) & ~mask_a
                if mask_b.any():
                    zi, yi, xi = np.where(mask_b)
                    vals = vb[zi, yi, xi]
                    gb = np.zeros((len(vals), 4))
                    gb[:, 1] = vals * 0.9         # green
                    gb[:, 3] = np.clip(vals * alph, 0.05, alph)
                    ax3d.scatter(xi, yi, zi * zsc,
                                 c=gb, s=vsz * 0.8,
                                 depthshade=True, rasterized=True)

                title = f"Volume Scatter — Band A: {a_lbl_var.get()}"
                info_lbl.config(
                    text=f"Voxels rendered: A={mask_a.sum():,}  B={mask_b.sum():,}\n"
                         f"Volume: {Nx}×{Ny}×{Nz}  threshold={thr:.2f}",
                    fg=C["success"])

            elif mode == "slicing":
                # ── ORTHOGONAL SLICES ──────────────────────────────────────────
                cmap_fn = plt.get_cmap(cmap_3d_var.get())
                xg, yg = np.meshgrid(np.arange(Nx), np.arange(Ny))

                # XY plane (Z slice)
                sz = min(slice_z_var.get(), Nz - 1)
                slice_z_var.set(sz)
                sl_xy = va[sz]
                cols_xy = cmap_fn(sl_xy)
                cols_xy[..., 3] = alph
                ax3d.plot_surface(xg, yg, np.full_like(xg, sz * zsc, dtype=float),
                                  facecolors=cols_xy, shade=False,
                                  linewidth=0, antialiased=False)

                # XZ plane (Y slice)
                sy = min(slice_y_var.get(), Ny - 1)
                slice_y_var.set(sy)
                xg2, zg2 = np.meshgrid(np.arange(Nx), np.arange(Nz))
                sl_xz = va[:, sy, :]
                cols_xz = cmap_fn(sl_xz)
                cols_xz[..., 3] = alph
                ax3d.plot_surface(xg2, np.full_like(xg2, sy, dtype=float),
                                  zg2 * zsc,
                                  facecolors=cols_xz, shade=False,
                                  linewidth=0, antialiased=False)

                # YZ plane (X slice)
                sx = min(slice_x_var.get(), Nx - 1)
                slice_x_var.set(sx)
                yg3, zg3 = np.meshgrid(np.arange(Ny), np.arange(Nz))
                sl_yz = va[:, :, sx]
                cols_yz = cmap_fn(sl_yz)
                cols_yz[..., 3] = alph
                ax3d.plot_surface(np.full_like(yg3, sx, dtype=float),
                                  yg3, zg3 * zsc,
                                  facecolors=cols_yz, shade=False,
                                  linewidth=0, antialiased=False)

                title = f"Orthogonal Slices — Z={sz}, Y={sy}, X={sx}"
                info_lbl.config(
                    text=f"Slice Z={sz}/{Nz-1}  Y={sy}/{Ny-1}  X={sx}/{Nx-1}\n"
                         f"Adjust sliders and re-render.",
                    fg=C["success"])

            elif mode == "surface":
                # ── 3D SURFACE ────────────────────────────────────────────────
                # Use the top Z slice (or mean across Z) as height surface
                surf_data = va.mean(axis=0)   # Ny × Nx height
                xg, yg = np.meshgrid(np.arange(Nx), np.arange(Ny))
                cmap_fn = plt.get_cmap(cmap_3d_var.get())
                norm = Normalize(vmin=surf_data.min(), vmax=surf_data.max())
                surf = ax3d.plot_surface(
                    xg, yg, surf_data * Nz * zsc,
                    facecolors=cmap_fn(norm(surf_data)),
                    shade=True, linewidth=0, antialiased=True, alpha=alph)

                # Optionally overlay Band B as a second surface
                if vb.max() > thr:
                    surf_b = vb.mean(axis=0)
                    ax3d.plot_surface(
                        xg, yg, surf_b * Nz * zsc * 0.6,
                        facecolors=plt.get_cmap("summer")(norm(surf_b)),
                        shade=True, linewidth=0, antialiased=True,
                        alpha=alph * 0.6)

                title = f"3D Surface — Band A (Z-height): {a_lbl_var.get()}"
                info_lbl.config(
                    text=f"Surface: mean of {Nz} depth planes\n"
                         f"Map size: {Nx}×{Ny} px",
                    fg=C["success"])

            else:  # multiband_rgb
                # ── MULTI-BAND RGB VOLUME ──────────────────────────────────────
                # Every voxel above threshold rendered with R=BandA, G=BandB, B=BandC
                mask = (va > thr) | (vb > thr) | (use_c_var.get() and vc > thr)
                if mask.any():
                    zi, yi, xi = np.where(mask)
                    R = va[zi, yi, xi]
                    G = vb[zi, yi, xi]
                    B = vc[zi, yi, xi] if use_c_var.get() else np.zeros_like(R)
                    # luminance for alpha
                    lum = np.clip(np.sqrt(R**2 + G**2 + B**2) / np.sqrt(3), 0, 1)
                    rgba = np.stack([R, G, B,
                                     np.clip(lum * alph, 0.05, alph)], axis=1)
                    ax3d.scatter(xi, yi, zi * zsc,
                                 c=rgba, s=vsz,
                                 depthshade=True, rasterized=True)
                lbl_c = c_lbl_var.get() if use_c_var.get() else "—"
                title = (f"RGB Volume  R={a_lbl_var.get()}  "
                         f"G={b_lbl_var.get()}  B={lbl_c}")
                info_lbl.config(
                    text=f"RGB voxels: {mask.sum():,}\nR={a_lbl_var.get()}  "
                         f"G={b_lbl_var.get()}  B={lbl_c}",
                    fg=C["success"])

            # ── Common axes styling ────────────────────────────────────────────
            _style_ax(ax3d, Nz, Ny, Nx)
            tc = "white" if dark_bg_var.get() else C["text_hi"]
            ax3d.set_xlim(0, Nx); ax3d.set_ylim(0, Ny)
            ax3d.set_zlim(0, Nz * zsc)
            ax3d.set_title(title, color=tc, fontsize=10,
                           fontweight="semibold", pad=10)
            canvas3d.draw_idle()

        def export_3d():
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")],
                parent=win)
            if path:
                fig.savefig(path, dpi=250, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                self._status.set(f"3D view saved → {Path(path).name}")

        run_btn.config(command=render)
        exp_btn.config(command=export_3d)

        # Initial render
        win.after(300, render)

    # ── save ──────────────────────────────────────────────────────────────────
    def save_map(self):
        if self.spectra is None:
            messagebox.showwarning("No data","Load a WDF file first."); return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("Text","*.txt"),("NumPy","*.npy")])
        if not path: return
        arr = self.im.get_array()
        if path.endswith(".npy"):
            np.save(path, arr)
        elif path.endswith(".txt"):
            if arr.ndim == 3:
                out = path.replace(".txt",".png")
                plt.imsave(out, np.clip(arr,0,1))
            else:
                np.savetxt(path, arr, fmt="%.6f")
        else:
            if arr.ndim == 2:
                plt.imsave(path, arr, cmap=self.cmap_var.get())
            else:
                plt.imsave(path, np.clip(arr,0,1))
        self._status.set(f"Map saved  →  {Path(path).name}")

    def save_spectrum(self):
        if self.spectra is None or self.coords is None:
            messagebox.showwarning("No data",
                "Load a file and click a pixel first."); return
        x,y = self.coords
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text","*.txt"),("CSV","*.csv")])
        if not path: return
        np.savetxt(path, np.column_stack((self.xdata, self.spectra[y,x])),
                   fmt="%.6f", header="Raman_Shift(cm-1)  Intensity(a.u.)")
        self._status.set(f"Spectrum saved  →  {Path(path).name}")

    # ── v7 launchers ──────────────────────────────────────────────────────────
    def open_clustering(self):
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first."); return
        ClusteringWindow(self, self.spectra, self.xdata,
                         roi_mask=getattr(self, "_roi_mask", None))

    def open_mcr(self):
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first."); return
        MCRWindow(self, self.spectra, self.xdata,
                  roi_mask=getattr(self, "_roi_mask", None))

    def open_nfindr(self):
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first."); return
        NFindrWindow(self, self.spectra, self.xdata,
                     roi_mask=getattr(self, "_roi_mask", None))

    def open_spectral_tools(self):
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first."); return
        SpectralToolsWindow(self)


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER ANALYSIS WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class ClusteringWindow(tk.Toplevel):
    """
    K-means and Agglomerative (Ward linkage) clustering on the Raman map.

    The map is flattened to (N_pixels × N_wavenumbers), spectra are
    mean-centred per pixel, and then clustered.  Results are shown as:
      - Colour-coded spatial cluster map
      - Mean spectrum per cluster (offset for clarity)
      - Pixel-count bar chart
    """

    CLUSTER_COLORS = [
        "#2563eb", "#ef4444", "#10b981", "#f59e0b", "#7c3aed",
        "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#6366f1",
    ]

    def __init__(self, parent, spectra: np.ndarray, xdata: np.ndarray,
                 roi_mask: np.ndarray | None = None):
        super().__init__(parent)
        self.title("Cluster Analysis")
        self.geometry("1220x760")
        self.configure(bg=C["bg"])
        self.spectra = spectra          # Y × X × W
        self.xdata   = xdata
        # ROI mask (Y × X bool). When present, only ROI pixels are analysed
        # and the background is ignored everywhere.
        self.roi_mask = roi_mask if (roi_mask is not None
                                     and np.asarray(roi_mask).any()) else None
        self._labels: np.ndarray | None = None

        self._build_ui()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _flat(self):
        """Return mean-centred (N_pix × W) feature matrix."""
        Y, X, W = self.spectra.shape
        mat = self.spectra.reshape(-1, W).astype(float)
        mat -= mat.mean(axis=1, keepdims=True)
        return mat

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=C["header"], height=44)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="⬡  CLUSTER ANALYSIS",
                 bg=C["header"], fg="white",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=16, pady=10)

        # Left controls
        left = tk.Frame(self, bg=C["sidebar"], width=270)
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        SectionDiv(left, "METHOD").pack(fill="x")
        self._method = tk.StringVar(value="kmeans")
        for val, txt in [("kmeans", "K-means"), ("agglom", "Agglomerative (Ward)")]:
            tk.Radiobutton(left, text=txt, variable=self._method, value=val,
                           bg=C["sidebar"], fg=C["text_hi"],
                           activebackground=C["sidebar"],
                           selectcolor=C["sidebar"],
                           font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=2)

        SectionDiv(left, "PARAMETERS").pack(fill="x")
        pf = tk.Frame(left, bg=C["sidebar"])
        pf.pack(fill="x", padx=12, pady=4)

        tk.Label(pf, text="Number of clusters", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=0, column=0, sticky="w", pady=4)
        self._n_clust = tk.IntVar(value=4)
        ttk.Spinbox(pf, from_=2, to=10, textvariable=self._n_clust,
                    width=5).grid(row=0, column=1, padx=8)

        tk.Label(pf, text="Wavenumber min", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=1, column=0, sticky="w", pady=4)
        lo_default = float(self.xdata.min())
        self._wn_lo = tk.DoubleVar(value=lo_default)
        ttk.Spinbox(pf, from_=0, to=4000, textvariable=self._wn_lo,
                    width=8).grid(row=1, column=1, padx=8)

        tk.Label(pf, text="Wavenumber max", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=2, column=0, sticky="w", pady=4)
        hi_default = float(self.xdata.max())
        self._wn_hi = tk.DoubleVar(value=hi_default)
        ttk.Spinbox(pf, from_=0, to=4000, textvariable=self._wn_hi,
                    width=8).grid(row=2, column=1, padx=8)

        tk.Label(pf, text="K-means restarts", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=3, column=0, sticky="w", pady=4)
        self._n_init = tk.IntVar(value=10)
        ttk.Spinbox(pf, from_=1, to=50, textvariable=self._n_init,
                    width=5).grid(row=3, column=1, padx=8)

        SectionDiv(left, "DISPLAY").pack(fill="x")
        self._show_borders = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="Show cluster borders",
                       variable=self._show_borders,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=2)
        self._offset_spectra = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Offset mean spectra",
                       variable=self._offset_spectra,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=2)

        ttk.Button(left, text="▶  Run Clustering", style="P.TButton",
                   command=self._run).pack(fill="x", padx=12, pady=10)

        self._prog  = ttk.Progressbar(left, mode="indeterminate")
        self._prog.pack(fill="x", padx=12, pady=2)
        self._status = tk.Label(left, text="Configure and press Run",
                                bg=C["sidebar"], fg=C["text_dim"],
                                font=("Segoe UI", 10), wraplength=240,
                                justify="left")
        self._status.pack(padx=12, pady=4, anchor="w")

        SectionDiv(left, "EXPORT").pack(fill="x")
        ttk.Button(left, text="↓ Save Figure", style="N.TButton",
                   command=self._save_fig).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="↓ Save Label Matrix (.csv)", style="N.TButton",
                   command=self._save_labels).pack(fill="x", padx=12, pady=4)

        # Right: matplotlib figure
        right = tk.Frame(self, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)

        self._fig = plt.figure(figsize=(12, 7), facecolor="#ffffff")
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 2, figure=self._fig,
                               hspace=0.42, wspace=0.34,
                               left=0.06, right=0.82, top=0.93, bottom=0.08)
        self._ax_map   = self._fig.add_subplot(gs[0, 0])
        self._ax_spec  = self._fig.add_subplot(gs[0, 1])
        self._ax_bar   = self._fig.add_subplot(gs[1, 0])
        self._ax_blank = self._fig.add_subplot(gs[1, 1])
        self._ax_blank.set_visible(False)

        for ax, title in [
            (self._ax_map,  "Cluster Map"),
            (self._ax_spec, "Mean Spectra per Cluster"),
            (self._ax_bar,  "Pixel Count per Cluster"),
        ]:
            ax.set_title(title, fontsize=11, fontweight="semibold")

        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, right).update()

    # ── run ───────────────────────────────────────────────────────────────────
    def _run(self):
        if not HAS_SKL:
            messagebox.showerror("Missing library",
                "scikit-learn not installed.\npip install scikit-learn",
                parent=self)
            return
        self._prog.start(12)
        self._status.config(text="Clustering…", fg=C["text_mid"])
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler

            Y, X, W = self.spectra.shape
            lo, hi  = self._wn_lo.get(), self._wn_hi.get()
            mask_w  = (self.xdata >= lo) & (self.xdata <= hi)
            mat_all = self._flat()[:, mask_w]

            # Analyse only ROI pixels (ignore background) when an ROI is set
            if self.roi_mask is not None:
                roi_flat = np.asarray(self.roi_mask, dtype=bool).ravel()
            else:
                roi_flat = np.ones(Y * X, dtype=bool)

            # Standardise features so all wavenumbers contribute equally
            mat = StandardScaler().fit_transform(mat_all[roi_flat])

            k = self._n_clust.get()
            method = self._method.get()

            if method == "kmeans":
                clf = KMeans(n_clusters=k, n_init=self._n_init.get(),
                             random_state=42)
            else:
                clf = AgglomerativeClustering(n_clusters=k, linkage="ward")

            sub_labels = clf.fit_predict(mat)
            # Scatter cluster labels back; background pixels stay -1
            labels = np.full(Y * X, -1, dtype=int)
            labels[roi_flat] = sub_labels
            labels = labels.reshape(Y, X)
            self._labels = labels
            self.after(0, lambda: self._draw(labels))
        except Exception as ex:
            self.after(0, lambda ex=ex: messagebox.showerror("Error", str(ex), parent=self))
        finally:
            self.after(0, self._prog.stop)

    def _draw(self, labels: np.ndarray):
        Y, X = labels.shape
        k    = self._n_clust.get()
        cols = self.CLUSTER_COLORS[:k]

        # ── Cluster map ────────────────────────────────────────────────────
        ax = self._ax_map
        ax.cla()
        rgba = np.zeros((Y, X, 4), dtype=float)
        for c in range(k):
            r, g, b = tuple(
                int(cols[c].lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4))
            mask = labels == c
            rgba[mask] = [r, g, b, 1.0]

        ax.imshow(rgba, origin="upper", aspect="equal", interpolation="none")
        ax.set_title(f"Cluster Map  ({k} clusters)", fontsize=11,
                     fontweight="semibold")
        ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
        patches = [mpatches.Patch(color=cols[c], label=f"Cluster {c+1}")
                   for c in range(k)]
        ax.legend(handles=patches, loc="upper right", fontsize=8,
                  framealpha=0.85)

        if self._show_borders.get():
            from scipy.ndimage import sobel
            edge = np.hypot(sobel(labels.astype(float), axis=0),
                            sobel(labels.astype(float), axis=1))
            ax.contour(edge > 0, levels=[0.5], colors="white",
                       linewidths=0.6, alpha=0.7)

        # ── Mean spectra ───────────────────────────────────────────────────
        ax2 = self._ax_spec
        ax2.cla()
        offset  = 0.0
        x_all   = self.xdata
        Y2, X2, W = self.spectra.shape
        flat_full = self.spectra.reshape(-1, W)
        flat_lbl  = labels.ravel()

        for c in range(k):
            mean_sp = flat_full[flat_lbl == c].mean(axis=0)
            pk      = mean_sp.max() or 1.0
            norm_sp = mean_sp / pk
            off_sp  = norm_sp + offset if self._offset_spectra.get() else norm_sp
            ax2.plot(x_all, off_sp, color=cols[c], lw=1.3,
                     label=f"Cluster {c+1}")
            if self._offset_spectra.get():
                offset += 1.1

        ax2.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=10)
        ax2.set_ylabel("Intensity  (norm., offset)" if self._offset_spectra.get()
                       else "Intensity  (norm.)", fontsize=10)
        ax2.set_title("Mean Spectra per Cluster", fontsize=11,
                      fontweight="semibold")
        # Legend outside the axes to the right (keeps spectra unobstructed)
        ax2.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0., frameon=True, framealpha=0.9)
        ax2.grid(True, ls="--", lw=0.4, alpha=0.5)

        # ── Pixel count bar ────────────────────────────────────────────────
        ax3 = self._ax_bar
        ax3.cla()
        counts = [int((labels.ravel() == c).sum()) for c in range(k)]
        bars   = ax3.bar(range(1, k+1), counts, color=cols, edgecolor="white",
                         linewidth=0.8)
        ax3.set_xlabel("Cluster", fontsize=10)
        ax3.set_ylabel("Pixel count", fontsize=10)
        ax3.set_title("Pixel Count per Cluster", fontsize=11,
                      fontweight="semibold")
        ax3.set_xticks(range(1, k+1))
        for bar, cnt in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(counts) * 0.01,
                     str(cnt), ha="center", va="bottom", fontsize=9)

        self._canvas.draw_idle()
        total = int(labels.size)
        self._status.config(
            text=f"Done — {k} clusters over {total} pixels.",
            fg=C["success"])

    def _save_fig(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf")], parent=self)
        if path:
            self._fig.savefig(path, dpi=250, bbox_inches="tight")

    def _save_labels(self):
        if self._labels is None:
            messagebox.showwarning("No results", "Run clustering first.",
                                   parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")], parent=self)
        if path:
            np.savetxt(path, self._labels, fmt="%d", delimiter=",")
            messagebox.showinfo("Saved", f"Label matrix saved to\n{path}",
                                parent=self)


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL ANALYSIS HELPERS  (peak identification + endmember comparison)
# Shared by the N-FINDR and MCR-ALS windows.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Raman band assignments tuned for CELL CRYOPRESERVATION  (cm⁻¹).
# Grouped: (A) ice / water state, (B) cryoprotectants, (C) cytosol biomolecules.
# Literature anchors: Dong et al., Biophys. J. 113 (2017) [low-T Raman of
# intracellular ice in lymphoblasts]; Raman cryomicroscopy PMC8214853;
# water OH-stretch decomposition (sym 3230 / asym 3420 cm⁻¹).
#
# NOTE: the OH-stretch ICE markers (>3000 cm⁻¹) fall OUTSIDE an 837–2472 cm⁻¹
# acquisition window. To map intracellular ice vs vitrified cytosol directly,
# extend the spectral range to ~2900–3600 cm⁻¹.
# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific reference libraries.  Each is a list of (cm⁻¹, assignment).
# Add your own, or import a custom CSV at runtime (Peak ID → Load custom CSV).
BAND_LIBRARIES = {

    # ════════════════════════════════════════════════════════════════════════
    "Biology / Cryopreservation": [
        # (A) ICE / WATER STATE  — primary cryopreservation markers
        (3140, "ICE: sharp OH stretch — crystalline hexagonal ice Iₕ "
               "(intracellular/extracellular ICE FORMATION)"),
        (3230, "WATER: symmetric OH stretch — tetrahedral H-bonded (ice-like)"),
        (3420, "WATER: asymmetric OH stretch — partially H-bonded "
               "(liquid / VITRIFIED cytosol, unfrozen water)"),
        (1640, "Water H–O–H bend / amide I shoulder (bound water)"),
        # (B) CRYOPROTECTANTS
        (672,  "DMSO: C–S stretch (penetrating CPA)"),
        (700,  "DMSO: C–S stretch"),
        (1042, "DMSO: S=O stretch — DMSO uptake/distribution marker"),
        (2998, "DMSO: CH₃ stretch"),
        (850,  "Trehalose/sucrose: C–C / C–O–C ring (non-penetrating CPA)"),
        (920,  "Glycerol / sugar: C–C–O stretch"),
        (1056, "Glycerol: C–O / C–C stretch (penetrating CPA)"),
        (1085, "Sugar C–O / PO₂⁻ overlap (freeze-concentrated solute)"),
        (1462, "Glycerol / sugar CH₂ deformation"),
        # (C) CYTOSOL BIOMOLECULES
        (1004, "Phenylalanine ring breathing (protein marker)"),
        (1031, "Phenylalanine C–H in-plane"),
        (1095, "PO₂⁻ symmetric stretch (nucleic acid / phospholipid)"),
        (1126, "C–N / C–C stretch (protein, lipid)"),
        (1158, "C–C / C=C (carotenoid)"),
        (1208, "Tyrosine / phenylalanine"),
        (1250, "Amide III (β-sheet / random coil protein)"),
        (1300, "CH₂ twist (lipid acyl chains)"),
        (1336, "Nucleic acids (A,G) / CH deformation"),
        (1440, "CH₂/CH₃ deformation (lipid + protein — total biomass)"),
        (1515, "Carotenoid C=C"),
        (1576, "Guanine / adenine ring (nucleic acid)"),
        (1605, "Phenylalanine / tyrosine ring"),
        (1655, "Amide I (α-helix protein) / C=C unsaturated lipid"),
        (1745, "Ester C=O stretch (phospholipid / triglyceride)"),
        (2850, "CH₂ symmetric stretch (lipid)"),
        (2885, "CH₂ asymmetric stretch (lipid)"),
        (2935, "CH₃ stretch (protein)"),
    ],

    # ════════════════════════════════════════════════════════════════════════
    "Carbon & 2D materials": [
        (520,  "Si substrate (520.7 cm⁻¹ — calibration / substrate)"),
        (1350, "D band — disorder / sp³ defects (graphene, soot, DLC)"),
        (1580, "G band — sp² graphitic C=C stretch"),
        (1620, "D′ band — defect-activated shoulder"),
        (2450, "G* / combination band"),
        (2690, "2D (G′) band — graphene layer count / stacking"),
        (2930, "D+D′ combination (disorder)"),
        (1100, "C–C amorphous carbon"),
        (1430, "Diamond / sp³ (≈1332 diamond line nearby)"),
        (1332, "Diamond T₂g — sp³ crystalline carbon"),
    ],

    # ════════════════════════════════════════════════════════════════════════
    "Minerals / geology (RRUFF-style)": [
        (1086, "Calcite ν₁ CO₃²⁻ symmetric stretch"),
        (712,  "Calcite ν₄ CO₃²⁻ bend"),
        (282,  "Calcite lattice mode"),
        (1085, "Aragonite ν₁ CO₃²⁻"),
        (704,  "Aragonite ν₄ doublet"),
        (464,  "Quartz — α-quartz Si–O–Si symmetric"),
        (206,  "Quartz lattice mode"),
        (1008, "Gypsum ν₁ SO₄²⁻ (calibration standard)"),
        (960,  "Apatite ν₁ PO₄³⁻ (bone, phosphate minerals)"),
        (144,  "Anatase TiO₂ Eg (strong)"),
        (397,  "Anatase TiO₂ B1g"),
        (515,  "Anatase TiO₂ A1g/B1g"),
        (639,  "Anatase TiO₂ Eg"),
        (225,  "Hematite Fe₂O₃ A1g"),
        (292,  "Hematite Fe₂O₃ Eg"),
        (410,  "Hematite Fe₂O₃ Eg"),
        (343,  "Pyrite FeS₂ Eg"),
        (379,  "Pyrite FeS₂ Ag"),
        (820,  "Olivine SiO₄ doublet (lower)"),
        (855,  "Olivine SiO₄ doublet (upper)"),
    ],

    # ════════════════════════════════════════════════════════════════════════
    "Microplastics & polymers (SLoPP-style)": [
        (1062, "PE: C–C stretch (polyethylene)"),
        (1128, "PE: C–C stretch"),
        (1295, "PE: CH₂ twist"),
        (1440, "PE/PP: CH₂ bend"),
        (2848, "PE: CH₂ symmetric stretch"),
        (2882, "PE: CH₂ asymmetric stretch"),
        (809,  "PP: CH₂ rock (polypropylene)"),
        (841,  "PP: CH₂ rock / C–C"),
        (998,  "PP: CH₃ rock"),
        (1458, "PP: CH₃ bend"),
        (1001, "PS: ring breathing (polystyrene — diagnostic)"),
        (1031, "PS: ring C–H in-plane"),
        (1602, "PS: aromatic C=C ring"),
        (3054, "PS: aromatic C–H stretch"),
        (1096, "PET: C–O / ring (polyethylene terephthalate)"),
        (1614, "PET: aromatic ring C=C"),
        (1728, "PET/PMMA: ester C=O stretch"),
        (632,  "PET: C=O in-plane / ring"),
        (638,  "PVC: C–Cl stretch (polyvinyl chloride)"),
        (1430, "PVC: CH₂ bend"),
        (812,  "PMMA: C–O–C (acrylic)"),
        (1635, "Nylon/PA: amide I"),
    ],
}

# Active library (mutable; the Peak-ID window can switch it).  Default = biology.
RAMAN_BANDS = BAND_LIBRARIES["Biology / Cryopreservation"]


def _detect_peaks(wn, spectrum, prominence=0.05, tol=12.0, bands=None):
    """Return ranked peak list [(wn, norm_intensity, assignment), ...].
    `bands` overrides the active library; defaults to RAMAN_BANDS."""
    ref = bands if bands is not None else RAMAN_BANDS
    y = np.asarray(spectrum, dtype=float)
    pk = y.max() or 1.0
    y = y / pk
    ys = savgol_filter(y, 11, 3) if len(y) > 11 else y
    peaks, _ = find_peaks(ys, prominence=prominence, distance=5)
    rows = []
    for p in peaks:
        w = float(wn[p])
        if ref:
            cand = min(ref, key=lambda b: abs(b[0] - w))
            assign = cand[1] if abs(cand[0] - w) <= tol else "unassigned"
        else:
            assign = "unassigned"
        rows.append((round(w, 1), round(float(y[p]), 3), assign))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _load_band_csv(path):
    """Load a custom band library CSV: columns = wavenumber, assignment."""
    bands = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.lower().startswith(("wavenumber", "cm", "#")):
                continue
            parts = line.split(",", 1)
            try:
                wn_v = float(parts[0])
            except (ValueError, IndexError):
                continue
            label = parts[1].strip().strip('"') if len(parts) > 1 else ""
            bands.append((wn_v, label))
    return bands


def open_peak_id(parent, wn, M, labels, colors):
    """
    Peak-identification window.
    M : (k, W) matrix of spectra (rows = endmembers/components).
    """
    win = tk.Toplevel(parent)
    win.title("Peak Identification")
    win.geometry("1100x720")
    win.configure(bg=C["bg"])

    hdr = tk.Frame(win, bg=C["header"], height=40); hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Label(hdr, text="⛯  PEAK IDENTIFICATION", bg=C["header"], fg="white",
             font=("Consolas", 12, "bold")).pack(side="left", padx=16, pady=8)

    body = tk.Frame(win, bg=C["bg"]); body.pack(fill="both", expand=True)

    # left: plot
    left = tk.Frame(body, bg=C["bg"]); left.pack(side="left", fill="both",
                                                 expand=True)
    fig = plt.figure(figsize=(8, 6), facecolor="#ffffff")
    canvas = FigureCanvasTkAgg(fig, master=left)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    NavigationToolbar2Tk(canvas, left).update()

    # right: selector + table
    right = tk.Frame(body, bg=C["sidebar"], width=360); right.pack(side="right",
                                                                   fill="y")
    right.pack_propagate(False)

    # ── Reference band library (multi-domain, switchable) ───────────────────
    SectionDiv(right, "REFERENCE LIBRARY").pack(fill="x")
    state = {"bands": RAMAN_BANDS, "custom": None}
    lib_names = list(BAND_LIBRARIES.keys())
    lib_var = tk.StringVar(value=lib_names[0])
    lib_box = ttk.Combobox(right, textvariable=lib_var,
                           values=lib_names + ["Custom (loaded CSV)"],
                           state="readonly", width=30)
    lib_box.pack(padx=12, pady=4)

    def _set_lib(*_):
        name = lib_var.get()
        if name == "Custom (loaded CSV)":
            state["bands"] = state["custom"] or []
        else:
            state["bands"] = BAND_LIBRARIES.get(name, [])
        _refresh()

    def _load_custom():
        path = filedialog.askopenfilename(
            filetypes=[("CSV", "*.csv"), ("Text", "*.txt")], parent=win,
            title="Custom band library: column1=wavenumber, column2=assignment")
        if not path:
            return
        try:
            state["custom"] = _load_band_csv(path)
        except Exception as ex:
            messagebox.showerror("Load error", str(ex), parent=win); return
        if not state["custom"]:
            messagebox.showwarning("Empty", "No (wavenumber, label) rows found.",
                                   parent=win); return
        lib_var.set("Custom (loaded CSV)")
        _set_lib()
        messagebox.showinfo("Loaded",
                            f"{len(state['custom'])} reference bands loaded.",
                            parent=win)

    lib_var.trace_add("write", _set_lib)
    ttk.Button(right, text="＋ Load custom CSV…", style="N.TButton",
               command=_load_custom).pack(fill="x", padx=12, pady=(0, 4))

    SectionDiv(right, "SELECT SPECTRUM").pack(fill="x")
    sel = tk.IntVar(value=0)
    for i, lbl in enumerate(labels):
        tk.Radiobutton(right, text=lbl, variable=sel, value=i,
                       bg=C["sidebar"], fg=colors[i % len(colors)],
                       selectcolor=C["sidebar"], activebackground=C["sidebar"],
                       font=("Segoe UI", 10, "bold"),
                       command=lambda: _refresh()).pack(anchor="w", padx=14,
                                                        pady=1)

    SectionDiv(right, "DETECTED PEAKS  (ranked)").pack(fill="x")
    cols = ("wn", "rel", "assignment")
    tree = ttk.Treeview(right, columns=cols, show="headings", height=22)
    tree.heading("wn", text="cm⁻¹"); tree.column("wn", width=70, anchor="e")
    tree.heading("rel", text="Rel.I"); tree.column("rel", width=55, anchor="e")
    tree.heading("assignment", text="Assignment")
    tree.column("assignment", width=210, anchor="w")
    tree.pack(fill="both", expand=True, padx=8, pady=6)

    rows_cache = {}

    def _refresh():
        i = sel.get()
        rows = _detect_peaks(wn, M[i], bands=state["bands"])
        rows_cache["last"] = rows
        fig.clear()
        ax = fig.add_subplot(111)
        y = M[i] / (M[i].max() or 1.0)
        ax.plot(wn, y, color=colors[i % len(colors)], lw=1.3)
        for w, ri, _a in rows:
            ax.axvline(w, color="0.85", lw=0.7, zorder=0)
            ax.annotate(f"{w:.0f}", (w, ri), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7, rotation=90)
        ax.set_xlabel("Raman Shift  (cm⁻¹)"); ax.set_ylabel("Intensity (norm.)")
        ax.set_title(f"Peak identification — {labels[i]}",
                     fontweight="semibold")
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        fig.tight_layout()
        canvas.draw_idle()
        tree.delete(*tree.get_children())
        for w, ri, a in rows:
            tree.insert("", "end", values=(f"{w:.1f}", f"{ri:.2f}", a))

    def _save_csv():
        rows = rows_cache.get("last")
        if not rows:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")], parent=win)
        if path:
            with open(path, "w") as fh:
                fh.write("Wavenumber_cm-1,Relative_Intensity,Assignment\n")
                for w, ri, a in rows:
                    fh.write(f"{w},{ri},{a}\n")
            messagebox.showinfo("Saved", f"Peak table saved to\n{path}",
                                parent=win)

    ttk.Button(right, text="↓ Save Peak Table (.csv)", style="N.TButton",
               command=_save_csv).pack(fill="x", padx=8, pady=8)
    _refresh()
    return win


def open_spectra_compare(parent, wn, M, labels, colors):
    """
    Overlay/compare two spectra on the same axis with a difference trace
    and similarity metrics.  M : (k, W).
    """
    win = tk.Toplevel(parent)
    win.title("Compare Spectra")
    win.geometry("1080x680")
    win.configure(bg=C["bg"])

    hdr = tk.Frame(win, bg=C["header"], height=40); hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Label(hdr, text="⇄  COMPARE / SUPERIMPOSE", bg=C["header"], fg="white",
             font=("Consolas", 12, "bold")).pack(side="left", padx=16, pady=8)

    body = tk.Frame(win, bg=C["bg"]); body.pack(fill="both", expand=True)
    left = tk.Frame(body, bg=C["bg"]); left.pack(side="left", fill="both",
                                                 expand=True)
    fig = plt.figure(figsize=(8, 5.5), facecolor="#ffffff")
    canvas = FigureCanvasTkAgg(fig, master=left)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    NavigationToolbar2Tk(canvas, left).update()

    right = tk.Frame(body, bg=C["sidebar"], width=280); right.pack(side="right",
                                                                   fill="y")
    right.pack_propagate(False)
    SectionDiv(right, "CHOOSE TWO").pack(fill="x")

    tk.Label(right, text="Spectrum A", bg=C["sidebar"], fg=C["text_mid"],
             font=("Segoe UI", 10)).pack(anchor="w", padx=14, pady=(6, 0))
    va = tk.StringVar(value=labels[0])
    ttk.Combobox(right, textvariable=va, values=labels, state="readonly",
                 width=22).pack(padx=14, pady=2)
    tk.Label(right, text="Spectrum B", bg=C["sidebar"], fg=C["text_mid"],
             font=("Segoe UI", 10)).pack(anchor="w", padx=14, pady=(6, 0))
    vb = tk.StringVar(value=labels[1] if len(labels) > 1 else labels[0])
    ttk.Combobox(right, textvariable=vb, values=labels, state="readonly",
                 width=22).pack(padx=14, pady=2)

    show_diff = tk.BooleanVar(value=True)
    tk.Checkbutton(right, text="Show difference (A − B)", variable=show_diff,
                   bg=C["sidebar"], fg=C["text_mid"],
                   activebackground=C["sidebar"], selectcolor=C["sidebar"],
                   font=("Segoe UI", 10),
                   command=lambda: _refresh()).pack(anchor="w", padx=12, pady=4)

    metric = tk.Label(right, text="", bg=C["sidebar"], fg=C["text_dim"],
                      font=("Consolas", 10), justify="left")
    metric.pack(anchor="w", padx=14, pady=8)

    def _refresh(*_):
        i, j = labels.index(va.get()), labels.index(vb.get())
        a = M[i] / (M[i].max() or 1.0)
        b = M[j] / (M[j].max() or 1.0)
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(wn, a, color=colors[i % len(colors)], lw=1.4, label=labels[i])
        ax.plot(wn, b, color=colors[j % len(colors)], lw=1.4, label=labels[j])
        if show_diff.get():
            ax.plot(wn, a - b, color="0.45", lw=1.0, alpha=0.8,
                    label=f"{labels[i]} − {labels[j]}")
            ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlabel("Raman Shift  (cm⁻¹)"); ax.set_ylabel("Intensity (norm.)")
        ax.set_title("Endmember comparison", fontweight="semibold")
        ax.legend(fontsize=9); ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        fig.tight_layout(); canvas.draw_idle()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        pear = float(np.corrcoef(a, b)[0, 1])
        metric.config(text=f"cosine  = {cos:.3f}\npearson = {pear:.3f}")

    va.trace_add("write", _refresh)
    vb.trace_add("write", _refresh)
    _refresh()
    return win


# ─────────────────────────────────────────────────────────────────────────────
# MCR-ALS WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class MCRWindow(tk.Toplevel):
    """
    Multivariate Curve Resolution – Alternating Least Squares (MCR-ALS).

    Decomposes the hyperspectral data cube D (N_pix × W) into:
        D ≈ C · S^T
    where C (N_pix × n_comp) are non-negative abundances and
          S (W × n_comp) are non-negative pure component spectra.

    Initialisation: NMF from scikit-learn, then iterative ALS with
    non-negativity constraints applied via clipping.
    """

    COMP_COLORS = ["#2563eb","#ef4444","#10b981","#f59e0b",
                   "#7c3aed","#06b6d4","#ec4899","#84cc16"]

    def __init__(self, parent, spectra: np.ndarray, xdata: np.ndarray,
                 roi_mask: np.ndarray | None = None):
        super().__init__(parent)
        self.title("MCR-ALS  —  Multivariate Curve Resolution")
        self.geometry("1280x780")
        self.configure(bg=C["bg"])
        self.spectra = spectra
        self.xdata   = xdata
        self.roi_mask = roi_mask if (roi_mask is not None
                                     and np.asarray(roi_mask).any()) else None
        self._C: np.ndarray | None = None
        self._S: np.ndarray | None = None
        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=C["header"], height=44)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="⟠  MCR-ALS  —  MULTIVARIATE CURVE RESOLUTION",
                 bg=C["header"], fg="white",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=16, pady=10)

        left = tk.Frame(self, bg=C["sidebar"], width=270)
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        SectionDiv(left, "PARAMETERS").pack(fill="x")
        pf = tk.Frame(left, bg=C["sidebar"])
        pf.pack(fill="x", padx=12, pady=4)

        labels_vals = [
            ("Components",      "n_comp",  tk.IntVar,    3,    2, 10),
            ("Max iterations",  "max_iter",tk.IntVar,   100,   10, 500),
            ("Convergence tol", "tol",     tk.DoubleVar, 1e-4, 0,  0),
            ("Wavenumber min",  "wn_lo",   tk.DoubleVar, float(self.xdata.min()), 0, 0),
            ("Wavenumber max",  "wn_hi",   tk.DoubleVar, float(self.xdata.max()), 0, 0),
        ]
        self._vars = {}
        for row_i, (lbl, key, VarClass, default, lo, hi) in enumerate(labels_vals):
            tk.Label(pf, text=lbl, bg=C["sidebar"], fg=C["text_mid"],
                     font=("Segoe UI", 10)).grid(row=row_i, column=0,
                                                  sticky="w", pady=3)
            v = VarClass(value=default)
            self._vars[key] = v
            ent = tk.Entry(pf, textvariable=v, width=10, bg="white",
                           relief="flat", highlightthickness=1,
                           highlightbackground=C["border"],
                           font=("Segoe UI", 10))
            ent.grid(row=row_i, column=1, padx=8, pady=3)

        SectionDiv(left, "CONSTRAINTS").pack(fill="x")
        self._nn_C = tk.BooleanVar(value=True)
        self._nn_S = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Non-negative abundances",
                       variable=self._nn_C, bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=2)
        tk.Checkbutton(left, text="Non-negative spectra",
                       variable=self._nn_S, bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=2)

        # ── SPEED (Apple Silicon / large maps) ──────────────────────────────
        SectionDiv(left, "SPEED").pack(fill="x")
        sf = tk.Frame(left, bg=C["sidebar"]); sf.pack(fill="x", padx=12, pady=2)
        tk.Label(sf, text="Pixel bin (k×k)", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=3)
        self._bin = tk.IntVar(value=2)
        ttk.Spinbox(sf, from_=1, to=8, textvariable=self._bin,
                    width=5).grid(row=0, column=1, padx=8)
        self._fast32 = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="float32 (faster, M1 Accelerate)",
                       variable=self._fast32, bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=2)
        tk.Label(left,
                 text="ALS iterates on binned pixels; full-res abundance maps "
                      "are computed in one final pass.",
                 bg=C["sidebar"], fg=C["text_dim"], font=("Segoe UI", 8),
                 wraplength=240, justify="left").pack(anchor="w", padx=12,
                                                      pady=(0, 2))

        ttk.Button(left, text="▶  Run MCR-ALS", style="P.TButton",
                   command=self._run).pack(fill="x", padx=12, pady=10)
        self._prog = ttk.Progressbar(left, mode="indeterminate")
        self._prog.pack(fill="x", padx=12, pady=2)
        self._status = tk.Label(left, text="Configure and press Run",
                                bg=C["sidebar"], fg=C["text_dim"],
                                font=("Segoe UI", 10), wraplength=240,
                                justify="left")
        self._status.pack(padx=12, pady=4, anchor="w")

        SectionDiv(left, "ANALYSIS").pack(fill="x")
        ttk.Button(left, text="⇄ Compare Components", style="N.TButton",
                   command=self._compare_comps).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="⛯ Identify Peaks", style="N.TButton",
                   command=self._identify_peaks).pack(fill="x", padx=12, pady=4)

        SectionDiv(left, "EXPORT").pack(fill="x")
        ttk.Button(left, text="↓ Save Figure", style="N.TButton",
                   command=self._save_fig).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="↓ Save Pure Spectra (.csv)", style="N.TButton",
                   command=self._save_spectra).pack(fill="x", padx=12, pady=4)

        right = tk.Frame(self, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._fig = plt.figure(figsize=(12, 7), facecolor="#ffffff")
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, right).update()

    def _compare_comps(self):
        if self._S is None:
            messagebox.showwarning("No results", "Run MCR-ALS first.",
                                   parent=self)
            return
        labels = [f"Component {i+1}" for i in range(self._k)]
        open_spectra_compare(self, self._xsel, self._S.T, labels,
                             self.COMP_COLORS)

    def _identify_peaks(self):
        if self._S is None:
            messagebox.showwarning("No results", "Run MCR-ALS first.",
                                   parent=self)
            return
        labels = [f"Component {i+1}" for i in range(self._k)]
        open_peak_id(self, self._xsel, self._S.T, labels, self.COMP_COLORS)

    def _run(self):
        if not HAS_SKL:
            messagebox.showerror("Missing library",
                "scikit-learn not installed.\npip install scikit-learn",
                parent=self)
            return
        self._prog.start(12)
        self._status.config(text="Running MCR-ALS…", fg=C["text_mid"])
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            from sklearn.decomposition import NMF
            from scipy.optimize import nnls as scipy_nnls

            Y, X, W = self.spectra.shape
            lo = self._vars["wn_lo"].get()
            hi = self._vars["wn_hi"].get()
            mask_w = (self.xdata >= lo) & (self.xdata <= hi)
            xsel   = self.xdata[mask_w]

            # float32 → ~2× faster + half memory via Apple Accelerate BLAS
            dt   = np.float32 if self._fast32.get() else np.float64
            cube = np.clip(self.spectra[:, :, mask_w].astype(dt), 0, None)
            Wsel = cube.shape[2]

            # Analyse only ROI pixels (ignore background) when an ROI is set
            if self.roi_mask is not None:
                roi_flat = np.asarray(self.roi_mask, dtype=bool).ravel()
            else:
                roi_flat = np.ones(Y * X, dtype=bool)
            D = cube.reshape(-1, Wsel)[roi_flat]          # full-res ROI pixels

            k        = self._vars["n_comp"].get()
            max_iter = self._vars["max_iter"].get()
            tol      = self._vars["tol"].get()
            nn_C     = self._nn_C.get()
            nn_S     = self._nn_S.get()
            bin_f    = max(1, int(self._bin.get()))

            # scipy.optimize.nnls needs float64 contiguous input — wrap so the
            # float32 fast-path doesn't raise a dtype error.
            def _nnls(A, b):
                return scipy_nnls(np.ascontiguousarray(A, dtype=np.float64),
                                  np.ascontiguousarray(b, dtype=np.float64))[0]

            # ── Build the (much smaller) matrix the ALS loop iterates on ────
            #    Spatial k×k binning cuts the pixel count by k², which is the
            #    dominant cost.  Pure-component spectra are scale-invariant, so
            #    the recovered S is essentially unchanged.
            if bin_f > 1:
                yb, xb = Y // bin_f, X // bin_f
                if yb >= 1 and xb >= 1:
                    cb = cube[:yb * bin_f, :xb * bin_f, :].reshape(
                        yb, bin_f, xb, bin_f, Wsel).mean(axis=(1, 3))
                    D_work = cb.reshape(-1, Wsel).astype(dt)
                else:
                    D_work = D
            else:
                D_work = D

            # NMF initialisation on the working matrix (good starting point)
            nmf = NMF(n_components=k, init="nndsvda", max_iter=300,
                      random_state=42)
            C_w   = nmf.fit_transform(D_work)
            S_cur = nmf.components_.T.astype(dt)           # Wsel × k

            # ── ALS iterations on the small working matrix ──────────────────
            prev_resid = np.inf
            for _ in range(max_iter):
                if nn_C:
                    C_w = np.array([_nnls(S_cur, D_work[i])
                                    for i in range(D_work.shape[0])], dtype=dt)
                else:
                    C_w = np.linalg.lstsq(S_cur, D_work.T, rcond=None)[0].T
                if nn_S:
                    # list over wavenumbers → (Wsel, k) == S directly (no .T)
                    S_cur = np.array([_nnls(C_w, D_work[:, j])
                                      for j in range(D_work.shape[1])],
                                     dtype=dt)
                else:
                    S_cur = np.linalg.lstsq(C_w, D_work, rcond=None)[0].T
                resid = float(np.linalg.norm(D_work - C_w @ S_cur.T))
                if abs(prev_resid - resid) < tol:
                    break
                prev_resid = resid

            # ── Final FULL-resolution abundance maps: one NNLS sweep ────────
            #    (instead of max_iter sweeps over every pixel)
            if nn_C:
                C_cur = np.array([_nnls(S_cur, D[i])
                                  for i in range(D.shape[0])], dtype=dt)
            else:
                C_cur = np.linalg.lstsq(S_cur, D.T, rcond=None)[0].T

            # Scatter abundances back to the full map; background = NaN
            C_full = np.full((Y * X, k), np.nan, dtype=float)
            C_full[roi_flat] = C_cur.astype(float)
            self._C  = C_full.reshape(Y, X, k)
            self._S  = np.asarray(S_cur, dtype=float)     # Wsel × k
            self._xsel = xsel
            self._k  = k
            self.after(0, self._draw)
        except Exception as ex:
            self.after(0, lambda ex=ex: messagebox.showerror("Error", str(ex),
                                                        parent=self))
        finally:
            self.after(0, self._prog.stop)

    def _draw(self):
        k    = self._k
        cols = self.COMP_COLORS[:k]
        self._fig.clear()

        import matplotlib.gridspec as gridspec
        # Row 0: abundance maps (up to 4 per row, wrap)
        rows_needed = 1 + (-(k // -4))   # ceil(k/4) map rows + 1 spectra row
        gs = gridspec.GridSpec(2, max(k, 2), figure=self._fig,
                               hspace=0.52, wspace=0.30,
                               left=0.05, right=0.82, top=0.93, bottom=0.08)

        # Pure spectra on top row spanning all columns
        ax_sp = self._fig.add_subplot(gs[0, :])
        offset = 0.0
        for c in range(k):
            spec = self._S[:, c]
            pk   = spec.max() or 1.0
            ax_sp.plot(self._xsel, spec / pk + offset,
                       color=cols[c], lw=1.3,
                       label=f"Component {c+1}")
            offset += 1.1
        ax_sp.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=10)
        ax_sp.set_ylabel("Intensity  (norm., offset)", fontsize=10)
        ax_sp.set_title("MCR-ALS Recovered Pure Spectra",
                        fontsize=12, fontweight="semibold")
        # Legend outside the axes to the right
        ax_sp.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     borderaxespad=0., frameon=True, framealpha=0.9)
        ax_sp.grid(True, ls="--", lw=0.4, alpha=0.5)

        # Abundance maps on bottom row
        for c in range(k):
            ax_m = self._fig.add_subplot(gs[1, c])
            ab   = self._C[:, :, c]
            vmin, vmax = np.nanmin(ab), np.nanmax(ab)
            ab_n = (ab - vmin) / (vmax - vmin + 1e-12)
            cmap = plt.get_cmap("turbo").copy()
            cmap.set_bad("#ffffff")          # background (NaN) shown white
            ax_m.imshow(np.ma.masked_invalid(ab_n), origin="upper",
                        aspect="equal", cmap=cmap, interpolation="bilinear")
            ax_m.set_title(f"Abundance Map — C{c+1}",
                           fontsize=9, fontweight="semibold",
                           color=cols[c])
            ax_m.set_xticks([]); ax_m.set_yticks([])

        self._canvas.draw_idle()
        self._status.config(
            text=f"MCR-ALS converged — {k} components.",
            fg=C["success"])

    def _save_fig(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf")], parent=self)
        if path:
            self._fig.savefig(path, dpi=250, bbox_inches="tight")

    def _save_spectra(self):
        if self._S is None:
            messagebox.showwarning("No results","Run MCR-ALS first.",parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")], parent=self)
        if path:
            hdr_str = ",".join(
                ["Wavenumber_cm-1"] + [f"Component_{i+1}" for i in range(self._k)])
            data = np.column_stack([self._xsel, self._S])
            np.savetxt(path, data, delimiter=",", header=hdr_str, comments="",
                       fmt="%.6f")
            messagebox.showinfo("Saved", f"Pure spectra saved to\n{path}",
                                parent=self)


# ─────────────────────────────────────────────────────────────────────────────
# N-FINDR ENDMEMBER EXTRACTION WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class NFindrWindow(tk.Toplevel):
    """
    N-FINDR endmember extraction followed by NNLS abundance mapping.

    N-FINDR finds the set of p spectra that maximise the simplex volume
    in the reduced-dimension space (PCA-compressed to p-1 dimensions).
    Abundances are then estimated per pixel using non-negative least squares.
    """

    COMP_COLORS = ["#2563eb","#ef4444","#10b981","#f59e0b",
                   "#7c3aed","#06b6d4","#ec4899","#84cc16"]

    def __init__(self, parent, spectra: np.ndarray, xdata: np.ndarray,
                 roi_mask: np.ndarray | None = None):
        super().__init__(parent)
        self.title("N-FINDR  —  Endmember Extraction")
        self.geometry("1240x760")
        self.configure(bg=C["bg"])
        self.spectra = spectra
        self.xdata   = xdata
        self.roi_mask = roi_mask if (roi_mask is not None
                                     and np.asarray(roi_mask).any()) else None
        self._endmembers: np.ndarray | None = None
        self._abund: np.ndarray | None = None
        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=C["header"], height=44)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="◉  N-FINDR  —  ENDMEMBER EXTRACTION",
                 bg=C["header"], fg="white",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=16, pady=10)

        left = tk.Frame(self, bg=C["sidebar"], width=270)
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        SectionDiv(left, "PARAMETERS").pack(fill="x")
        pf = tk.Frame(left, bg=C["sidebar"])
        pf.pack(fill="x", padx=12, pady=4)

        tk.Label(pf, text="Endmembers (p)", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=0, column=0, sticky="w", pady=4)
        self._n_end = tk.IntVar(value=3)
        ttk.Spinbox(pf, from_=2, to=8, textvariable=self._n_end,
                    width=5).grid(row=0, column=1, padx=8)

        tk.Label(pf, text="Max N-FINDR iterations", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=1, column=0, sticky="w", pady=4)
        self._max_iter = tk.IntVar(value=3)
        ttk.Spinbox(pf, from_=1, to=20, textvariable=self._max_iter,
                    width=5).grid(row=1, column=1, padx=8)

        tk.Label(pf, text="Wavenumber min", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=2, column=0, sticky="w", pady=4)
        self._wn_lo = tk.DoubleVar(value=float(self.xdata.min()))
        ttk.Spinbox(pf, from_=0, to=4000, textvariable=self._wn_lo,
                    width=8).grid(row=2, column=1, padx=8)

        tk.Label(pf, text="Wavenumber max", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=3, column=0, sticky="w", pady=4)
        self._wn_hi = tk.DoubleVar(value=float(self.xdata.max()))
        ttk.Spinbox(pf, from_=0, to=4000, textvariable=self._wn_hi,
                    width=8).grid(row=3, column=1, padx=8)

        ttk.Button(left, text="▶  Run N-FINDR", style="P.TButton",
                   command=self._run).pack(fill="x", padx=12, pady=10)
        self._prog = ttk.Progressbar(left, mode="indeterminate")
        self._prog.pack(fill="x", padx=12, pady=2)
        self._status = tk.Label(left, text="Configure and press Run",
                                bg=C["sidebar"], fg=C["text_dim"],
                                font=("Segoe UI", 10), wraplength=240,
                                justify="left")
        self._status.pack(padx=12, pady=4, anchor="w")

        SectionDiv(left, "ANALYSIS").pack(fill="x")
        ttk.Button(left, text="⇄ Compare Endmembers", style="N.TButton",
                   command=self._compare_ems).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="⛯ Identify Peaks", style="N.TButton",
                   command=self._identify_peaks).pack(fill="x", padx=12, pady=4)

        SectionDiv(left, "EXPORT").pack(fill="x")
        ttk.Button(left, text="↓ Save Figure", style="N.TButton",
                   command=self._save_fig).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="↓ Save Endmembers (.csv)", style="N.TButton",
                   command=self._save_endmembers).pack(fill="x", padx=12, pady=4)

        right = tk.Frame(self, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._fig = plt.figure(figsize=(12, 7), facecolor="#ffffff")
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, right).update()

    def _compare_ems(self):
        if self._endmembers is None:
            messagebox.showwarning("No results", "Run N-FINDR first.",
                                   parent=self)
            return
        labels = [f"Endmember {i+1}" for i in range(self._p)]
        open_spectra_compare(self, self._xsel, self._endmembers,
                             labels, self.COMP_COLORS)

    def _identify_peaks(self):
        if self._endmembers is None:
            messagebox.showwarning("No results", "Run N-FINDR first.",
                                   parent=self)
            return
        labels = [f"Endmember {i+1}" for i in range(self._p)]
        open_peak_id(self, self._xsel, self._endmembers,
                     labels, self.COMP_COLORS)

    def _run(self):
        if not HAS_SKL:
            messagebox.showerror("Missing library",
                "scikit-learn not installed.\npip install scikit-learn",
                parent=self)
            return
        self._prog.start(12)
        self._status.config(text="Running N-FINDR…", fg=C["text_mid"])
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            from sklearn.decomposition import PCA as _PCA
            from scipy.optimize import nnls as scipy_nnls

            Y, X, W = self.spectra.shape
            lo, hi  = self._wn_lo.get(), self._wn_hi.get()
            mask_w  = (self.xdata >= lo) & (self.xdata <= hi)
            xsel    = self.xdata[mask_w]
            p       = self._n_end.get()

            D_full = self.spectra.reshape(-1, W)[:, mask_w].astype(float)
            D_full = np.clip(D_full, 0, None)

            # Analyse only ROI pixels (ignore background) when an ROI is set
            if self.roi_mask is not None:
                roi_flat = np.asarray(self.roi_mask, dtype=bool).ravel()
            else:
                roi_flat = np.ones(Y * X, dtype=bool)
            D   = D_full[roi_flat]
            N   = D.shape[0]

            # Reduce to (p-1) dims via PCA
            pca = _PCA(n_components=p - 1, random_state=42)
            D_r = pca.fit_transform(D)   # N × (p-1)

            # Simplex volume via absolute value of determinant of (p×p) matrix
            def _simplex_vol(idx_set):
                rows = D_r[list(idx_set)]
                mat  = np.column_stack([rows, np.ones(p)])  # p × p
                return abs(np.linalg.det(mat))

            # Initialise: pick p random pixels
            rng     = np.random.default_rng(42)
            indices = list(rng.choice(N, p, replace=False))
            vol     = _simplex_vol(indices)

            for _ in range(self._max_iter.get()):
                improved = False
                for i in range(p):
                    for j in range(N):
                        if j in indices:
                            continue
                        trial = indices[:]
                        trial[i] = j
                        v = _simplex_vol(trial)
                        if v > vol:
                            indices, vol = trial, v
                            improved = True
                if not improved:
                    break

            endmembers = D[indices]   # p × W_sel

            # Compute NNLS abundance maps
            abund = np.zeros((N, p), dtype=float)
            for i in range(N):
                abund[i], _ = scipy_nnls(endmembers.T, D[i])
            # Normalise per pixel so abundances sum to 1
            row_sums = abund.sum(axis=1, keepdims=True)
            abund /= np.where(row_sums > 0, row_sums, 1.0)

            # Scatter abundances back to the full map; background = NaN
            abund_full = np.full((Y * X, p), np.nan, dtype=float)
            abund_full[roi_flat] = abund
            self._endmembers = endmembers
            self._abund      = abund_full.reshape(Y, X, p)
            self._xsel       = xsel
            self._p          = p
            self.after(0, self._draw)
        except Exception as ex:
            self.after(0, lambda ex=ex: messagebox.showerror("Error", str(ex),
                                                        parent=self))
        finally:
            self.after(0, self._prog.stop)

    def _draw(self):
        p    = self._p
        cols = self.COMP_COLORS[:p]
        self._fig.clear()

        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, max(p, 2), figure=self._fig,
                               hspace=0.52, wspace=0.30,
                               left=0.05, right=0.82, top=0.93, bottom=0.08)

        # Endmember spectra (top, spanning all columns)
        ax_sp = self._fig.add_subplot(gs[0, :])
        offset = 0.0
        for c in range(p):
            spec = self._endmembers[c]
            pk   = spec.max() or 1.0
            ax_sp.plot(self._xsel, spec / pk + offset,
                       color=cols[c], lw=1.3, label=f"Endmember {c+1}")
            offset += 1.1
        ax_sp.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=10)
        ax_sp.set_ylabel("Intensity  (norm., offset)", fontsize=10)
        ax_sp.set_title("N-FINDR  —  Extracted Endmember Spectra",
                        fontsize=12, fontweight="semibold")
        # Legend outside the axes to the right
        ax_sp.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                     borderaxespad=0., frameon=True, framealpha=0.9)
        ax_sp.grid(True, ls="--", lw=0.4, alpha=0.5)

        # Abundance maps (bottom row)
        for c in range(p):
            ax_m = self._fig.add_subplot(gs[1, c])
            ab   = self._abund[:, :, c]
            cmap = plt.get_cmap("turbo").copy()
            cmap.set_bad("#ffffff")          # background (NaN) shown white
            im   = ax_m.imshow(np.ma.masked_invalid(ab), origin="upper",
                               aspect="equal", cmap=cmap,
                               interpolation="bilinear", vmin=0, vmax=1)
            self._fig.colorbar(im, ax=ax_m, fraction=0.046, pad=0.04,
                               shrink=0.8).ax.tick_params(labelsize=7)
            ax_m.set_title(f"Abundance — EM {c+1}",
                           fontsize=9, fontweight="semibold",
                           color=cols[c])
            ax_m.set_xticks([]); ax_m.set_yticks([])

        self._canvas.draw_idle()
        self._status.config(
            text=f"N-FINDR complete — {p} endmembers extracted.",
            fg=C["success"])

    def _save_fig(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf")], parent=self)
        if path:
            self._fig.savefig(path, dpi=250, bbox_inches="tight")

    def _save_endmembers(self):
        if self._endmembers is None:
            messagebox.showwarning("No results", "Run N-FINDR first.",
                                   parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv")], parent=self)
        if path:
            hdr_str = ",".join(
                ["Wavenumber_cm-1"] +
                [f"Endmember_{i+1}" for i in range(self._p)])
            data = np.column_stack([self._xsel, self._endmembers.T])
            np.savetxt(path, data, delimiter=",", header=hdr_str, comments="",
                       fmt="%.6f")
            messagebox.showinfo("Saved", f"Endmembers saved to\n{path}",
                                parent=self)


# ─────────────────────────────────────────────────────────────────────────────
# SPECTRAL TOOLS WINDOW  (resampling, crop, rotation, substrate subtraction)
# ─────────────────────────────────────────────────────────────────────────────
class SpectralToolsWindow(tk.Toplevel):
    """
    Utility preprocessing operations inspired by best-practice Raman workflows:

    1. Spectral resampling — interpolates every pixel spectrum to an equally-
       spaced wavenumber grid, which is required before comparing data from
       instruments with different calibration axes.

    2. Spatial crop — trims the map to a rectangular pixel bounding box,
       discarding rows/columns outside the region of interest.

    3. Map rotation — rotates the spatial image by 0 / 90 / 180 / 270 degrees
       (lossless; equivalent to a transpose + flip sequence).

    4. Optical substrate subtraction — subtracts a reference mean spectrum
       (e.g. a glass or Si substrate background) from every pixel, scaled by
       a user-controllable factor.

    Results are applied back to the parent RamanApp in-place.
    """

    def __init__(self, parent_app):
        super().__init__(parent_app)
        self.title("Spectral Tools  —  Resample / Crop / Rotate / Substrate")
        self.geometry("960x620")
        self.configure(bg=C["bg"])
        self._app = parent_app

        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=C["header"], height=44)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="⚒  SPECTRAL TOOLS",
                 bg=C["header"], fg="white",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=16, pady=10)

        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=12, pady=12)

        # ── Section: Spectral Resampling ──────────────────────────────────────
        f1 = tk.LabelFrame(body, text=" 1 · Spectral Resampling ",
                           bg=C["panel"], fg=C["accent"],
                           font=("Segoe UI", 11, "bold"),
                           relief="flat", bd=1,
                           highlightthickness=1,
                           highlightbackground=C["border"])
        f1.pack(fill="x", pady=(0, 10))

        r1 = tk.Frame(f1, bg=C["panel"])
        r1.pack(fill="x", padx=12, pady=8)
        tk.Label(r1, text="New grid start (cm⁻¹):", bg=C["panel"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=0, column=0, sticky="w", padx=4, pady=3)
        self._rs_lo = tk.DoubleVar(value=200.0)
        tk.Entry(r1, textvariable=self._rs_lo, width=9, bg="white",
                 relief="flat", highlightthickness=1,
                 highlightbackground=C["border"]).grid(row=0, column=1, padx=8)
        tk.Label(r1, text="Grid end (cm⁻¹):", bg=C["panel"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=0, column=2, sticky="w", padx=4)
        self._rs_hi = tk.DoubleVar(value=3500.0)
        tk.Entry(r1, textvariable=self._rs_hi, width=9, bg="white",
                 relief="flat", highlightthickness=1,
                 highlightbackground=C["border"]).grid(row=0, column=3, padx=8)
        tk.Label(r1, text="Number of points:", bg=C["panel"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=0, column=4, sticky="w", padx=4)
        self._rs_n = tk.IntVar(value=512)
        tk.Entry(r1, textvariable=self._rs_n, width=6, bg="white",
                 relief="flat", highlightthickness=1,
                 highlightbackground=C["border"]).grid(row=0, column=5, padx=8)
        ttk.Button(r1, text="Apply Resampling", style="P.TButton",
                   command=self._resample).grid(row=0, column=6, padx=12)

        # ── Section: Spatial Crop ─────────────────────────────────────────────
        f2 = tk.LabelFrame(body, text=" 2 · Spatial Crop ",
                           bg=C["panel"], fg=C["accent"],
                           font=("Segoe UI", 11, "bold"),
                           relief="flat", bd=1,
                           highlightthickness=1,
                           highlightbackground=C["border"])
        f2.pack(fill="x", pady=(0, 10))

        r2 = tk.Frame(f2, bg=C["panel"])
        r2.pack(fill="x", padx=12, pady=8)
        for col, (lbl, var_name, default) in enumerate([
            ("X start (px):", "_cx0", 0),
            ("X end (px):",   "_cx1", 0),
            ("Y start (px):", "_cy0", 0),
            ("Y end (px):",   "_cy1", 0),
        ]):
            tk.Label(r2, text=lbl, bg=C["panel"], fg=C["text_mid"],
                     font=("Segoe UI", 10)).grid(row=0, column=col*2,
                                                  sticky="w", padx=4, pady=3)
            v = tk.IntVar(value=default)
            setattr(self, var_name, v)
            tk.Entry(r2, textvariable=v, width=6, bg="white",
                     relief="flat", highlightthickness=1,
                     highlightbackground=C["border"]).grid(
                         row=0, column=col*2+1, padx=4)
        ttk.Button(r2, text="Set from Map Size", style="N.TButton",
                   command=self._auto_crop_limits).grid(row=0, column=8, padx=8)
        ttk.Button(r2, text="Apply Crop", style="P.TButton",
                   command=self._crop).grid(row=0, column=9, padx=12)

        # ── Section: Map Rotation ─────────────────────────────────────────────
        f3 = tk.LabelFrame(body, text=" 3 · Map Rotation ",
                           bg=C["panel"], fg=C["accent"],
                           font=("Segoe UI", 11, "bold"),
                           relief="flat", bd=1,
                           highlightthickness=1,
                           highlightbackground=C["border"])
        f3.pack(fill="x", pady=(0, 10))

        r3 = tk.Frame(f3, bg=C["panel"])
        r3.pack(fill="x", padx=12, pady=8)
        self._rot_angle = tk.StringVar(value="90")
        for angle in ["90", "180", "270"]:
            tk.Radiobutton(r3, text=f"{angle}°", variable=self._rot_angle,
                           value=angle, bg=C["panel"], fg=C["text_hi"],
                           activebackground=C["panel"],
                           selectcolor=C["panel"],
                           font=("Segoe UI", 11)).pack(side="left", padx=10)
        ttk.Button(r3, text="Apply Rotation", style="P.TButton",
                   command=self._rotate).pack(side="left", padx=20)

        # ── Section: Substrate Subtraction ────────────────────────────────────
        f4 = tk.LabelFrame(body, text=" 4 · Optical Substrate / Background Subtraction ",
                           bg=C["panel"], fg=C["accent"],
                           font=("Segoe UI", 11, "bold"),
                           relief="flat", bd=1,
                           highlightthickness=1,
                           highlightbackground=C["border"])
        f4.pack(fill="x", pady=(0, 10))

        r4a = tk.Frame(f4, bg=C["panel"])
        r4a.pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(r4a, text="Method:", bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left")
        self._sub_method = tk.StringVar(value="roi_mean")
        for val, txt in [("roi_mean", "ROI mean spectrum"),
                         ("file",     "Load reference file (.txt / .csv)")]:
            tk.Radiobutton(r4a, text=txt, variable=self._sub_method,
                           value=val, bg=C["panel"], fg=C["text_hi"],
                           activebackground=C["panel"],
                           selectcolor=C["panel"],
                           font=("Segoe UI", 11)).pack(side="left", padx=8)

        r4b = tk.Frame(f4, bg=C["panel"])
        r4b.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(r4b, text="Subtraction scale (0–2):", bg=C["panel"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
        self._sub_scale = tk.DoubleVar(value=1.0)
        tk.Entry(r4b, textvariable=self._sub_scale, width=6, bg="white",
                 relief="flat", highlightthickness=1,
                 highlightbackground=C["border"]).pack(side="left", padx=8)
        ttk.Button(r4b, text="Apply Subtraction", style="P.TButton",
                   command=self._subtract_substrate).pack(side="left", padx=12)
        self._sub_ref: np.ndarray | None = None

        # ── Status ────────────────────────────────────────────────────────────
        self._status = tk.Label(body, text="All operations modify the loaded map in place. "
                                "Reopen the main window to see updated data.",
                                bg=C["bg"], fg=C["text_dim"],
                                font=("Segoe UI", 10), wraplength=820,
                                justify="left")
        self._status.pack(anchor="w", pady=6)

    # ── Resampling ────────────────────────────────────────────────────────────
    def _resample(self):
        app = self._app
        if app.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first.", parent=self)
            return
        lo  = self._rs_lo.get()
        hi  = self._rs_hi.get()
        n   = self._rs_n.get()
        if lo >= hi or n < 10:
            messagebox.showerror("Bad parameters",
                "Grid start must be < end and points ≥ 10.", parent=self)
            return
        new_x = np.linspace(lo, hi, n)
        Y, X, W = app.spectra.shape
        new_sp  = np.zeros((Y, X, n), dtype=float)
        for y in range(Y):
            for x in range(X):
                new_sp[y, x] = np.interp(new_x, app.xdata, app.spectra[y, x])
        app.spectra = new_sp
        app.xdata   = new_x
        app.update_map()
        self._status.config(
            text=f"✓ Resampled to {n} points from {lo:.0f} to {hi:.0f} cm⁻¹.",
            fg=C["success"])

    # ── Spatial crop ──────────────────────────────────────────────────────────
    def _auto_crop_limits(self):
        app = self._app
        if app.spectra is None:
            return
        Y, X, _ = app.spectra.shape
        self._cx0.set(0); self._cx1.set(X - 1)
        self._cy0.set(0); self._cy1.set(Y - 1)

    def _crop(self):
        app = self._app
        if app.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first.", parent=self)
            return
        Y, X, W = app.spectra.shape
        x0 = max(0, self._cx0.get()); x1 = min(X - 1, self._cx1.get())
        y0 = max(0, self._cy0.get()); y1 = min(Y - 1, self._cy1.get())
        if x0 >= x1 or y0 >= y1:
            messagebox.showerror("Bad crop region",
                "Crop start must be strictly less than crop end.", parent=self)
            return
        app.spectra = app.spectra[y0:y1+1, x0:x1+1, :].copy()
        app.update_map()
        self._status.config(
            text=f"✓ Cropped to X=[{x0}:{x1}]  Y=[{y0}:{y1}]  "
                 f"→ {x1-x0+1} × {y1-y0+1} pixels.",
            fg=C["success"])

    # ── Rotation ──────────────────────────────────────────────────────────────
    def _rotate(self):
        app = self._app
        if app.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first.", parent=self)
            return
        angle = int(self._rot_angle.get())
        k = angle // 90          # number of 90-degree CCW rotations
        app.spectra = np.rot90(app.spectra, k=k, axes=(0, 1)).copy()
        app.update_map()
        self._status.config(
            text=f"✓ Map rotated {angle}° counter-clockwise.",
            fg=C["success"])

    # ── Substrate subtraction ─────────────────────────────────────────────────
    def _subtract_substrate(self):
        app = self._app
        if app.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first.", parent=self)
            return
        method = self._sub_method.get()
        scale  = float(self._sub_scale.get())

        if method == "roi_mean":
            if app._roi_mask is not None and app._roi_mask.any():
                mask = app._roi_mask
                Y, X, W = app.spectra.shape
                ref = app.spectra[mask].mean(axis=0)
            else:
                messagebox.showinfo("No ROI",
                    "Draw an ROI on the substrate region first, "
                    "then apply subtraction.", parent=self)
                return
        else:
            path = filedialog.askopenfilename(
                title="Load reference spectrum",
                filetypes=[("Text / CSV","*.txt *.csv"),("All","*.*")],
                parent=self)
            if not path:
                return
            try:
                raw = np.loadtxt(path, delimiter=None, comments="#")
                if raw.ndim == 1:
                    ref_x, ref_i = app.xdata, raw
                else:
                    ref_x, ref_i = raw[:, 0], raw[:, 1]
                ref = np.interp(app.xdata, ref_x, ref_i)
            except Exception as ex:
                messagebox.showerror("Load error", str(ex), parent=self)
                return

        app.spectra = np.clip(app.spectra - scale * ref[np.newaxis, np.newaxis, :],
                              0, None)
        app.update_map()
        self._status.config(
            text=f"✓ Substrate reference subtracted (scale={scale:.2f}).",
            fg=C["success"])


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    app = RamanApp()
    app.mainloop()