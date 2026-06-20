#!/usr/bin/env python3
"""
PCA Studio — interactive, scalable, publication-grade PCA for spectroscopy.

A standalone companion to BioRaman. Load many spectral maps / spectra
(.wdf, .csv, .txt, .dpt, .xlsx), assign groups, and run modern PCA:

  • scales either way — exact randomized SVD for small N, out-of-core
    IncrementalPCA for 10^5–10^6 spectra (engine: pca_core.py);
  • standard / robust / sparse variants;
  • T² and Q-residual (SPE) diagnostics with 95% control limits;
  • PLS-DA / LDA classification and k-means / agglomerative / HDBSCAN
    clustering on the scores;
  • Nature/Science-quality figures: colour-blind-safe Okabe–Ito palette,
    600 dpi PNG and vector PDF/SVG export, density rendering for huge clouds.

Run:  python pca_studio.py
MIT License — part of the BioRaman project.  Author: Akalabya Bissoyi.
"""
from __future__ import annotations

import os
import threading
import traceback
from pathlib import Path

import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
import matplotlib.transforms as mtransforms

import pca_core as core

# optional readers
try:
    from renishawWiRE import WDFReader
    HAS_WDF = True
except Exception:
    HAS_WDF = False
try:
    import pandas as pd
    HAS_PD = True
except Exception:
    HAS_PD = False


# ───────────────────────────── publication style ───────────────────────────
# Okabe–Ito colour-blind-safe qualitative palette (8 colours).
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]
# Selectable qualitative palettes for the figure controls.
PALETTES = {
    "Okabe–Ito (colour-blind safe)": OKABE_ITO,
    "Tableau 10": ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1",
                   "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F", "#BAB0AC"],
    "Set1 (bold)": ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00",
                    "#FFD92F", "#A65628", "#F781BF"],
    "Dark2": ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
              "#E6AB02", "#A6761D", "#666666"],
    "Viridis (sampled)": ["#440154", "#46327e", "#365c8d", "#277f8e",
                          "#1fa187", "#4ac16d", "#a0da39", "#fde725"],
    "Greyscale": ["#000000", "#444444", "#777777", "#999999", "#bbbbbb",
                  "#222222", "#555555", "#888888"],
}
# matplotlib legend location choices.
LEGEND_LOCS = ["best", "upper right", "upper left", "lower left", "lower right",
               "right", "center left", "center right", "lower center",
               "upper center", "center", "outside right"]

# UI theme
C = dict(bg="#0f1419", sidebar="#1a2129", header="#11161c", panel="#222c38",
         accent="#2563eb", accent2="#1e4fd0", danger="#b3261e",
         text_hi="#f3f6fb", text_mid="#c3ccd9", text_dim="#8a97a8",
         border="#33414f")

# Common kwargs for plain tk.Checkbutton/tk.Radiobutton so labels stay
# readable on macOS (ttk's Aqua theme ignores style colours for these
# widgets, which can leave light-grey text on a white native background).
CKW = dict(bg=C["sidebar"], fg=C["text_mid"], selectcolor=C["sidebar"],
            activebackground=C["sidebar"], activeforeground=C["text_hi"],
            highlightthickness=0, bd=0)


def set_pub_style():
    """Journal-quality matplotlib defaults (vector-friendly, readable at column width)."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,      # embed TrueType (editable text in Illustrator)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# ───────────────────────────────── readers ─────────────────────────────────
# Robust spectral-file loaders live in spectra_io.py (importable / testable).
from spectra_io import read_spectra, DELIM_MAP  # noqa: E402


# ───────────────────────────── the application ─────────────────────────────
class PCAStudio(tk.Toplevel):
    """Scalable publication-grade PCA window.

    Runs standalone (``master=None`` → owns a hidden root) or embedded inside
    another Tk app such as BioRaman (``master`` = the parent app).  When
    ``external_data`` is provided it is added as an in-memory dataset so the
    spectra already loaded in the host app can be analysed without re-reading
    files.  ``external_data`` is a dict with keys:
        ``spectra`` : (n_samples, n_bands) float array
        ``waves``   : (n_bands,) wavenumber axis
        ``group``   : str label for the dataset (default "BioRaman data")
    """

    def __init__(self, master=None, external_data=None):
        # Own a hidden root when launched standalone so Tk has a main window;
        # when embedded, attach to the host application's event loop.
        self._own_root = None
        if master is None:
            self._own_root = tk.Tk()
            self._own_root.withdraw()
            super().__init__(self._own_root)
        else:
            super().__init__(master)
        self.title("PCA Studio — scalable publication-grade PCA")
        self.geometry("1680x1000")
        self.configure(bg=C["bg"])
        set_pub_style()

        self._files = []        # [{"path","label","group","color"}]
        self._result = None     # core.PCAResult
        self._labels = None
        self._groups = None
        self._waves = None
        self._cluster = None
        self._X = None                   # retained preprocessed matrix (subset PCA)
        self._scaling_used = "none"
        self._view = tk.StringVar(value="scores")
        self._legend_names = {}          # {group_value: custom display label}

        # density-cloud / subset-highlight controls
        self.sub_highlight = tk.StringVar(value="")   # comma-sep groups to highlight
        self.sub_deflate   = tk.BooleanVar(value=False)
        self.sub_ndeflate  = tk.IntVar(value=2)
        self.sub_bw        = tk.DoubleVar(value=1.0)   # density smoothing factor
        self.sub_inset     = tk.BooleanVar(value=True)

        # ── user-controllable figure / legend settings ──────────────────────
        self.r_palette   = tk.StringVar(value="Okabe–Ito (colour-blind safe)")
        self.r_title     = tk.StringVar(value="")    # blank → auto title
        self.r_xlabel    = tk.StringVar(value="")    # blank → auto label
        self.r_ylabel    = tk.StringVar(value="")
        self.r_legend_on = tk.BooleanVar(value=True)
        self.r_legend_loc= tk.StringVar(value="best")
        self.r_legend_ncol = tk.IntVar(value=1)
        self.r_legend_fs = tk.DoubleVar(value=8)
        self.r_legend_frame = tk.BooleanVar(value=False)
        self.r_legend_title = tk.StringVar(value="")
        self.r_marker    = tk.DoubleVar(value=0)     # 0 → auto by N
        self.r_alpha     = tk.DoubleVar(value=0.6)
        self.r_fontsize  = tk.DoubleVar(value=9)
        self.r_grid      = tk.BooleanVar(value=False)
        self.r_dpi       = tk.IntVar(value=600)
        self.r_w_in      = tk.DoubleVar(value=8.0)
        self.r_h_in      = tk.DoubleVar(value=6.0)
        self.r_apply_size = tk.BooleanVar(value=False)  # apply WxH on screen too

        # ── text / CSV / Excel import options ───────────────────────────────
        self.in_delim  = tk.StringVar(value="auto")   # auto/comma/tab/semicolon/space
        self.in_header = tk.StringVar(value="auto")   # auto/yes/no
        self.in_orient = tk.StringVar(value="auto")   # auto/cols/rows
        self.in_sheet  = tk.IntVar(value=0)           # Excel sheet index

        self._build_style()
        self._build_ui()

        # Clean shutdown: destroy the hidden root too when standalone.
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Optionally ingest spectra handed over by the host app (BioRaman).
        if external_data is not None:
            try:
                self.add_external_dataset(external_data)
            except Exception:
                traceback.print_exc()

    def _on_close(self):
        try:
            self.destroy()
        finally:
            if self._own_root is not None:
                self._own_root.destroy()

    def add_external_dataset(self, data):
        """Register an in-memory dataset (no file on disk) and list it.

        ``data`` keys: ``spectra`` (n×bands), ``waves`` (bands,), ``group`` (str).
        """
        spec = np.asarray(data["spectra"], dtype=np.float32)
        if spec.ndim == 3:                      # Y×X×W cube → (Y*X, W)
            spec = spec.reshape(-1, spec.shape[-1])
        x = np.asarray(data["waves"], dtype=float)
        group = str(data.get("group", "BioRaman data"))
        self._files.append({"path": None, "label": group, "group": group,
                            "_data": (spec, x)})
        self.listbox.insert("end", f"[in-memory] {group}  "
                                   f"({spec.shape[0]:,}×{spec.shape[1]})  [{group}]")

    # ── ttk styling ─────────────────────────────────────────────────────────
    def _build_style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure(".", background=C["sidebar"], foreground=C["text_hi"],
                    fieldbackground="white", font=("Segoe UI", 10),
                    bordercolor=C["border"])
        for name, bg in [("P.TButton", C["accent"]), ("N.TButton", "#33414f"),
                         ("D.TButton", C["danger"])]:
            s.configure(name, background=bg, foreground="white", relief="flat",
                        padding=(8, 5), font=("Segoe UI", 10, "bold"),
                        borderwidth=0)
            s.map(name, background=[("active", C["accent2"])])
        s.configure("TRadiobutton", background=C["sidebar"], foreground=C["text_mid"])
        s.configure("TCheckbutton", background=C["sidebar"], foreground=C["text_mid"])

    # ── layout ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        bar = tk.Frame(self, bg=C["header"], height=46); bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="◈  PCA STUDIO", bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16)
        tk.Label(bar, text="scalable · robust/sparse · T²+Q diagnostics · "
                 "publication export", bg=C["header"], fg=C["text_dim"],
                 font=("Segoe UI", 9)).pack(side="left", padx=8)

        # scrollable left control column
        left_outer = tk.Frame(self, bg=C["sidebar"], width=340)
        left_outer.pack(side="left", fill="y"); left_outer.pack_propagate(False)
        cv = tk.Canvas(left_outer, bg=C["sidebar"], highlightthickness=0, width=340)
        sb = ttk.Scrollbar(left_outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); cv.pack(side="left", fill="both", expand=True)
        left = tk.Frame(cv, bg=C["sidebar"])
        cv.create_window((0, 0), window=left, anchor="nw", width=322)
        left.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind_all("<MouseWheel>", lambda e: cv.yview_scroll(int(-e.delta/120), "units"))
        cv.bind_all("<Button-4>", lambda e: cv.yview_scroll(-1, "units"))
        cv.bind_all("<Button-5>", lambda e: cv.yview_scroll(1, "units"))

        self._build_controls(left)

        # right plot area
        right = tk.Frame(self, bg="white"); right.pack(side="left", fill="both",
                                                       expand=True)
        self.fig = plt.figure(figsize=(12, 9), facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        tbf = tk.Frame(right, bg="white"); tbf.pack(fill="x")
        NavigationToolbar2Tk(self.canvas, tbf)
        self._welcome()

    def _section(self, parent, text):
        f = tk.Frame(parent, bg=C["panel"]); f.pack(fill="x", pady=(10, 2))
        tk.Label(f, text=text, bg=C["panel"], fg=C["text_hi"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=3)

    def _build_controls(self, left):
        # FILES
        self._section(left, "MAP / SPECTRA FILES")
        self.listbox = tk.Listbox(left, bg="white", fg="#1a2129",
                                  selectmode="extended", height=8, relief="flat",
                                  highlightthickness=1, highlightbackground=C["border"])
        self.listbox.pack(fill="x", padx=10, pady=4)
        b = tk.Frame(left, bg=C["sidebar"]); b.pack(fill="x", padx=10)
        ttk.Button(b, text="+ Add files", style="P.TButton",
                   command=self._add_files).pack(side="left")
        ttk.Button(b, text="✕ Remove", style="D.TButton",
                   command=self._remove).pack(side="right")
        tk.Label(left, text=".wdf · .csv · .txt · .tsv · .dpt · .asc · .xlsx/.xls",
                 bg=C["sidebar"], fg=C["text_dim"],
                 font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=(2, 0))

        # TEXT / CSV / EXCEL IMPORT OPTIONS
        self._section(left, "TEXT / CSV / EXCEL OPTIONS")
        io = tk.Frame(left, bg=C["sidebar"]); io.pack(fill="x", padx=10, pady=2)
        tk.Label(io, text="Delimiter", bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(io, textvariable=self.in_delim, width=14, state="readonly",
                     values=list(DELIM_MAP.keys())).grid(row=0, column=1, padx=4)
        tk.Label(io, text="Header row", bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(io, textvariable=self.in_header, width=14, state="readonly",
                     values=["auto", "yes", "no"]).grid(row=1, column=1, padx=4)
        tk.Label(io, text="Spectra are in", bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=2, column=0, sticky="w", pady=2)
        ttk.Combobox(io, textvariable=self.in_orient, width=14, state="readonly",
                     values=["auto", "cols", "rows"]).grid(row=2, column=1, padx=4)
        self._spin_var(io, "Excel sheet #", self.in_sheet, 3, 0, 50, 1)
        ttk.Button(left, text="🔍 Preview selected file", style="N.TButton",
                   command=self._preview_file).pack(fill="x", padx=10, pady=(2, 0))
        tk.Label(left, text="auto-detects layout; override above if a file "
                 "loads wrong, then Preview to check.",
                 bg=C["sidebar"], fg=C["text_dim"], wraplength=300,
                 font=("Segoe UI", 8), justify="left").pack(anchor="w", padx=12)

        # GROUP LABEL
        self._section(left, "GROUP (for colour / classify)")
        g = tk.Frame(left, bg=C["sidebar"]); g.pack(fill="x", padx=10, pady=2)
        self.group_var = tk.StringVar()
        tk.Entry(g, textvariable=self.group_var, bg="white", relief="flat",
                 highlightthickness=1, highlightbackground=C["border"]).pack(
                     side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(g, text="Set", style="N.TButton",
                   command=self._set_group).pack(side="right")

        # PREPROCESS
        self._section(left, "PREPROCESS")
        pf = tk.Frame(left, bg=C["sidebar"]); pf.pack(fill="x", padx=10, pady=2)
        self._spin(pf, "Wavenumber min", 0, 0, 4000, "wn_lo", 400)
        self._spin(pf, "Wavenumber max", 1, 0, 4000, "wn_hi", 1800)
        self._spin(pf, "Max spectra/file (0=all)", 2, 0, 1_000_000, "max_spec", 0)
        tk.Label(pf, text="Scaling", bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=3, column=0, sticky="w", pady=2)
        self.scaling = tk.StringVar(value="snv")
        ttk.Combobox(pf, textvariable=self.scaling, width=12, state="readonly",
                     values=list(core.SCALINGS)).grid(row=3, column=1, padx=6)

        # PCA OPTIONS
        self._section(left, "PCA")
        of = tk.Frame(left, bg=C["sidebar"]); of.pack(fill="x", padx=10, pady=2)
        self._spin(of, "Components", 0, 2, 30, "n_comp", 5)
        tk.Label(of, text="Variant", bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=1, column=0, sticky="w", pady=2)
        self.variant = tk.StringVar(value="standard")
        ttk.Combobox(of, textvariable=self.variant, width=12, state="readonly",
                     values=["standard", "robust", "sparse"]).grid(row=1, column=1,
                                                                   padx=6)
        tk.Label(of, text="Solver", bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=2, column=0, sticky="w", pady=2)
        self.solver = tk.StringVar(value="auto")
        ttk.Combobox(of, textvariable=self.solver, width=12, state="readonly",
                     values=["auto", "exact", "incremental"]).grid(row=2, column=1,
                                                                   padx=6)
        self._spin(of, "Sparse α", 3, 0.0, 50.0, "alpha", 1.0, dbl=True, inc=0.5)

        ttk.Button(left, text="▶  Run PCA", style="P.TButton",
                   command=self._run).pack(fill="x", padx=10, pady=(10, 4))
        self.prog = ttk.Progressbar(left, mode="determinate")
        self.prog.pack(fill="x", padx=10)
        self.status = tk.Label(left, text="Add files, set groups, Run PCA.",
                               bg=C["sidebar"], fg=C["text_dim"], wraplength=300,
                               justify="left", font=("Segoe UI", 9))
        self.status.pack(fill="x", padx=10, pady=4)

        # VIEW
        self._section(left, "VIEW")
        for val, txt in [("scores", "Scores (2D)"),
                         ("scores3d", "Scores 3D — global map (PC1·2·3)"),
                         ("pcgrid", "PC pair grid (PC1-2 / 1-3 / 2-3)"),
                         ("subset", "Density cloud + subset highlight (Fig 4)"),
                         ("loadings", "Loadings"),
                         ("scree", "Scree + cumulative variance"),
                         ("diag", "Diagnostics  (T² vs Q)")]:
            tk.Radiobutton(left, text=txt, value=val, variable=self._view,
                           command=self._redraw, bg=C["sidebar"],
                           fg=C["text_mid"], selectcolor=C["sidebar"],
                           activebackground=C["sidebar"],
                           activeforeground=C["text_hi"],
                           highlightthickness=0, bd=0,
                           ).pack(anchor="w", padx=14)
        pcr = tk.Frame(left, bg=C["sidebar"]); pcr.pack(fill="x", padx=10, pady=4)
        self._spin(pcr, "PC x", 0, 1, 30, "pc_x", 1)
        self._spin(pcr, "PC y", 1, 1, 30, "pc_y", 2)
        self._spin(pcr, "PC z (3D)", 2, 1, 30, "pc_z", 3)
        self._spin(pcr, "Max points to render", 3, 2000, 500000, "max_render", 40000)
        self.density = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Density render when N large",
                       variable=self.density, command=self._redraw,
                       **CKW).pack(anchor="w", padx=14)
        self.ellipse = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="95% Hotelling ellipses",
                       variable=self.ellipse, command=self._redraw,
                       **CKW).pack(anchor="w", padx=14)

        # DENSITY-CLOUD / SUBSET-HIGHLIGHT (Fig 4)
        self._section(left, "DENSITY CLOUD + SUBSET")
        su = tk.Frame(left, bg=C["sidebar"]); su.pack(fill="x", padx=10, pady=2)
        tk.Label(su, text="Highlight groups", bg=C["sidebar"],
                 fg=C["text_mid"]).grid(row=0, column=0, sticky="w", pady=2)
        tk.Entry(su, textvariable=self.sub_highlight, bg="white", width=16,
                 relief="flat", highlightthickness=1,
                 highlightbackground=C["border"]).grid(row=0, column=1, padx=4)
        tk.Label(su, text="comma-separated; blank = all background",
                 bg=C["sidebar"], fg=C["text_dim"], font=("Segoe UI", 8)).grid(
                     row=1, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(su, text="Residual-subset inset",
                       variable=self.sub_inset, **CKW).grid(
                           row=2, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(su, text="Deflate global PCs in inset",
                       variable=self.sub_deflate, **CKW).grid(
                           row=3, column=0, columnspan=2, sticky="w")
        self._spin_var(su, "PCs to deflate", self.sub_ndeflate, 4, 1, 10, 1)
        self._spin_var(su, "Cloud smoothing", self.sub_bw, 5, 0.3, 4.0, 0.1)
        ttk.Button(left, text="↻ Draw density view", style="N.TButton",
                   command=lambda: (self._view.set("subset"), self._redraw())
                   ).pack(fill="x", padx=10, pady=(2, 0))

        # FIGURE & LEGEND CONTROLS
        self._section(left, "FIGURE & LEGEND")
        ff = tk.Frame(left, bg=C["sidebar"]); ff.pack(fill="x", padx=10, pady=2)
        tk.Label(ff, text="Palette", bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(ff, textvariable=self.r_palette, width=16, state="readonly",
                     values=list(PALETTES.keys())).grid(row=0, column=1, padx=4)
        self._labelled_entry(ff, "Title (blank=auto)", self.r_title, 1)
        self._labelled_entry(ff, "X label (blank=auto)", self.r_xlabel, 2)
        self._labelled_entry(ff, "Y label (blank=auto)", self.r_ylabel, 3)
        self._spin_var(ff, "Marker size (0=auto)", self.r_marker, 4, 0, 200, 1)
        self._spin_var(ff, "Transparency", self.r_alpha, 5, 0.05, 1.0, 0.05)
        self._spin_var(ff, "Font size", self.r_fontsize, 6, 5, 24, 1)
        tk.Checkbutton(ff, text="Grid", variable=self.r_grid, **CKW).grid(
            row=7, column=0, columnspan=2, sticky="w")

        lg = tk.Frame(left, bg=C["sidebar"]); lg.pack(fill="x", padx=10, pady=2)
        tk.Checkbutton(lg, text="Show legend", variable=self.r_legend_on,
                       **CKW).grid(row=0, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(lg, text="Legend frame", variable=self.r_legend_frame,
                       **CKW).grid(row=0, column=1, sticky="w")
        tk.Label(lg, text="Legend pos", bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(lg, textvariable=self.r_legend_loc, width=14, state="readonly",
                     values=LEGEND_LOCS).grid(row=1, column=1, padx=4)
        self._spin_var(lg, "Legend columns", self.r_legend_ncol, 2, 1, 6, 1)
        self._spin_var(lg, "Legend font", self.r_legend_fs, 3, 5, 20, 1)
        self._labelled_entry(lg, "Legend title", self.r_legend_title, 4)
        ttk.Button(lg, text="✎ Rename legend entries…", style="N.TButton",
                   command=self._edit_legend_names).grid(
                       row=5, column=0, columnspan=2, sticky="ew", pady=4)

        ez = tk.Frame(left, bg=C["sidebar"]); ez.pack(fill="x", padx=10, pady=2)
        self._spin_var(ez, "Export DPI", self.r_dpi, 0, 72, 1200, 50)
        self._spin_var(ez, "Width (in)", self.r_w_in, 1, 1.5, 20, 0.5)
        self._spin_var(ez, "Height (in)", self.r_h_in, 2, 1.5, 20, 0.5)
        tk.Checkbutton(ez, text="Apply size on screen too",
                       variable=self.r_apply_size, **CKW).grid(
                           row=3, column=0, columnspan=2, sticky="w")
        ttk.Button(left, text="↻ Apply / Redraw", style="P.TButton",
                   command=self._redraw).pack(fill="x", padx=10, pady=(4, 2))

        # DOWNSTREAM
        self._section(left, "DOWNSTREAM")
        ttk.Button(left, text="◎ Classify (LDA / PLS-DA)", style="N.TButton",
                   command=self._classify).pack(fill="x", padx=10, pady=2)
        ttk.Button(left, text="⬡ Cluster scores", style="N.TButton",
                   command=self._cluster).pack(fill="x", padx=10, pady=2)

        # EXPORT
        self._section(left, "EXPORT (publication)")
        ttk.Button(left, text="↓ Save figure (PNG/PDF/SVG)", style="N.TButton",
                   command=self._save_fig).pack(fill="x", padx=10, pady=2)
        ttk.Button(left, text="↓ Save scores + diagnostics (CSV)", style="N.TButton",
                   command=self._save_csv).pack(fill="x", padx=10, pady=2)
        ttk.Button(left, text="↓ Save loadings (CSV)", style="N.TButton",
                   command=self._save_loadings).pack(fill="x", padx=10, pady=(2, 12))

    def _spin(self, parent, label, row, lo, hi, attr, init, dbl=False, inc=1):
        tk.Label(parent, text=label, bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=row, column=0, sticky="w", pady=2)
        var = tk.DoubleVar(value=init) if dbl else tk.IntVar(value=init)
        setattr(self, attr, var)
        ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=9,
                    increment=inc).grid(row=row, column=1, padx=6, pady=2)

    def _spin_var(self, parent, label, var, row, lo, hi, inc):
        tk.Label(parent, text=label, bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=9,
                    increment=inc).grid(row=row, column=1, padx=6, pady=2)

    def _labelled_entry(self, parent, label, var, row):
        tk.Label(parent, text=label, bg=C["sidebar"], fg=C["text_mid"]).grid(
            row=row, column=0, sticky="w", pady=2)
        tk.Entry(parent, textvariable=var, bg="white", relief="flat", width=16,
                 highlightthickness=1, highlightbackground=C["border"]).grid(
                     row=row, column=1, padx=4, pady=2)

    # ── render-settings helpers used by the draw methods ─────────────────────
    def _palette(self):
        return PALETTES.get(self.r_palette.get(), OKABE_ITO)

    def _disp(self, g):
        """Display label for a group/cluster value (honours user renames)."""
        return self._legend_names.get(str(g), str(g))

    def _marker_size(self, n, big_default=6, small_default=16):
        m = float(self.r_marker.get())
        if m > 0:
            return m
        return big_default if n > 5000 else small_default

    def _apply_legend(self, ax, three_d=False):
        """Apply legend visibility / position / style from the controls."""
        if not self.r_legend_on.get():
            leg = ax.get_legend()
            if leg:
                leg.remove()
            return
        loc = self.r_legend_loc.get()
        kw = dict(markerscale=2, ncol=int(self.r_legend_ncol.get()),
                  fontsize=float(self.r_legend_fs.get()),
                  frameon=bool(self.r_legend_frame.get()))
        if self.r_legend_title.get().strip():
            kw["title"] = self.r_legend_title.get().strip()
        if loc == "outside right":
            kw["loc"] = "center left"; kw["bbox_to_anchor"] = (1.02, 0.5)
        else:
            kw["loc"] = loc
        ax.legend(**kw)

    def _finalize_axes(self, ax, auto_title="", auto_x="", auto_y=""):
        """Apply title / labels / grid overrides (blank fields keep auto text)."""
        t = self.r_title.get().strip()
        ax.set_title(t if t else auto_title)
        if auto_x:
            xl = self.r_xlabel.get().strip()
            ax.set_xlabel(xl if xl else auto_x)
        if auto_y:
            yl = self.r_ylabel.get().strip()
            ax.set_ylabel(yl if yl else auto_y)
        ax.grid(self.r_grid.get(), color="#e6e6e6", lw=0.5)

    def _edit_legend_names(self):
        """Dialog to rename each group/cluster as it appears in the legend."""
        if self._groups is None and self._cluster is None:
            messagebox.showinfo("Rename", "Run PCA first.", parent=self); return
        vals = (sorted(set(self._cluster)) if self._cluster is not None
                else list(dict.fromkeys(self._groups)))
        win = tk.Toplevel(self); win.title("Rename legend entries")
        win.configure(bg=C["sidebar"])
        tk.Label(win, text="Custom legend labels (blank = original)",
                 bg=C["sidebar"], fg=C["text_hi"],
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2,
                                                     padx=10, pady=8, sticky="w")
        entries = {}
        for i, v in enumerate(vals):
            tk.Label(win, text=str(v), bg=C["sidebar"], fg=C["text_mid"]).grid(
                row=i + 1, column=0, sticky="w", padx=10, pady=2)
            var = tk.StringVar(value=self._legend_names.get(str(v), ""))
            tk.Entry(win, textvariable=var, bg="white", width=24).grid(
                row=i + 1, column=1, padx=10, pady=2)
            entries[str(v)] = var

        def _save():
            for k, var in entries.items():
                txt = var.get().strip()
                if txt:
                    self._legend_names[k] = txt
                else:
                    self._legend_names.pop(k, None)
            win.destroy(); self._redraw()
        ttk.Button(win, text="Apply", style="P.TButton", command=_save).grid(
            row=len(vals) + 1, column=0, columnspan=2, pady=10, padx=10, sticky="ew")

    # ── file handling ────────────────────────────────────────────────────────
    def _import_opts(self):
        return {"delim": self.in_delim.get(), "header": self.in_header.get(),
                "orient": self.in_orient.get(), "sheet": int(self.in_sheet.get())}

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="Add spectral files",
            filetypes=[("Spectra",
                        "*.wdf *.csv *.txt *.tsv *.dpt *.asc *.xlsx *.xls"),
                       ("Text / CSV", "*.csv *.txt *.tsv *.dpt *.asc"),
                       ("Excel", "*.xlsx *.xls"),
                       ("Renishaw WDF", "*.wdf"),
                       ("All", "*.*")])
        for p in paths:
            self._files.append({"path": p, "label": Path(p).stem,
                                "group": Path(p).stem})
            self.listbox.insert("end", f"{Path(p).name}  [{Path(p).stem}]")

    def _preview_file(self):
        """Load the selected file with the current import options and report the
        parsed shape + a quick plot, so layout problems are caught before PCA."""
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("Preview", "Select a file in the list first.",
                                parent=self); return
        f = self._files[sel[0]]
        try:
            if f.get("_data") is not None:
                spec, x = f["_data"]; shape = None
            else:
                spec, x, shape = read_spectra(f["path"], self._import_opts())
        except Exception as e:
            messagebox.showerror("Preview failed",
                                 f"{Path(f['path']).name}\n\n{e}\n\n"
                                 "Try overriding Delimiter / Header / orientation.",
                                 parent=self); return
        win = tk.Toplevel(self); win.title(f"Preview — {Path(f['path']).name}")
        win.configure(bg="white")
        info = (f"Parsed {spec.shape[0]:,} spectra × {spec.shape[1]} points\n"
                f"Wavenumber range: {np.nanmin(x):.1f} – {np.nanmax(x):.1f} "
                f"({'ascending' if x[0] < x[-1] else 'descending'})"
                + (f"\nMap grid: {shape}" if shape else ""))
        tk.Label(win, text=info, bg="white", fg="#1a2129", justify="left",
                 font=("Segoe UI", 10)).pack(anchor="w", padx=10, pady=6)
        fig = plt.figure(figsize=(6.5, 3.6)); ax = fig.add_subplot(111)
        rng = np.random.default_rng(0)
        idx = rng.choice(spec.shape[0], min(8, spec.shape[0]), replace=False)
        for i in idx:
            ax.plot(x, spec[i], lw=0.8, alpha=0.8)
        ax.plot(x, spec.mean(0), color="k", lw=1.6, label="mean")
        ax.set_xlabel("Wavenumber (cm$^{-1}$)"); ax.set_ylabel("Intensity")
        ax.set_title("Sample of parsed spectra"); ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=win).get_tk_widget().pack(fill="both",
                                                                expand=True)

    def _remove(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i); del self._files[i]

    def _set_group(self):
        sel = self.listbox.curselection()
        g = self.group_var.get().strip()
        if not sel or not g:
            return
        for i in sel:
            self._files[i]["group"] = g
            p = self._files[i]["path"]
            nm = Path(p).name if p else f"[in-memory] {self._files[i]['label']}"
            self.listbox.delete(i)
            self.listbox.insert(i, f"{nm}  [{g}]")
        for i in sel:
            self.listbox.selection_set(i)

    # ── run PCA (threaded) ────────────────────────────────────────────────────
    def _run(self):
        if not self._files:
            messagebox.showwarning("No files", "Add files first.", parent=self)
            return
        self.status.config(text="Loading & preprocessing…"); self.prog["value"] = 0
        self.update_idletasks()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            X, labels, groups, waves = self._load_all()
            variant = self.variant.get()
            self.after(0, lambda: self.status.config(text="Decomposing…"))
            self.after(0, lambda: self.prog.configure(value=80))
            res = core.run_pca(
                X, n_components=int(self.n_comp.get()),
                scaling=self.scaling.get(), variant=variant,
                method=self.solver.get(), alpha=float(self.alpha.get()))
            self._result = res
            self._labels = labels
            self._groups = groups
            self._waves = waves
            self._X = X                      # retained for subset / residual PCA
            self._scaling_used = self.scaling.get()
            self._cluster = None
            self.after(0, self._after_run)
        except Exception as ex:
            tb = traceback.format_exc()
            self.after(0, lambda: messagebox.showerror("Error", f"{ex}\n\n{tb}",
                                                       parent=self))
            self.after(0, lambda: self.status.config(text=f"Error: {ex}"))

    def _load_all(self):
        wn_lo, wn_hi = float(self.wn_lo.get()), float(self.wn_hi.get())
        max_sp = int(self.max_spec.get())
        opts = self._import_opts()
        rng = np.random.default_rng(42)
        Xparts, labels, groups = [], [], []
        waves_common = None
        n = len(self._files)
        for fi, f in enumerate(self._files):
            self.after(0, lambda f=f: self.status.config(
                text=f"Loading {Path(f['path']).name}…"))
            self.after(0, lambda fi=fi: self.prog.configure(value=fi / n * 70))
            if f.get("_data") is not None:
                spec, x = f["_data"]
            else:
                spec, x, _shape = read_spectra(f["path"], opts)
            m = (x >= wn_lo) & (x <= wn_hi)
            x, spec = x[m], spec[:, m]
            if waves_common is None:
                waves_common = x
            elif not (x.shape == waves_common.shape and np.allclose(x, waves_common, atol=0.5)):
                spec = np.vstack([np.interp(waves_common, x, row) for row in spec])
            if max_sp > 0 and spec.shape[0] > max_sp:
                idx = rng.choice(spec.shape[0], max_sp, replace=False)
                spec = spec[idx]
            Xparts.append(spec.astype(np.float32))
            labels += [f["label"]] * spec.shape[0]
            groups += [f["group"]] * spec.shape[0]
        X = np.vstack(Xparts)                       # single allocation
        return X, np.array(labels), np.array(groups), waves_common

    def _after_run(self):
        r = self._result
        self.prog["value"] = 100
        sug = ""
        try:
            from pca_core import suggest_n_components
            # need full evr; recompute quick guidance from retained ratios
        except Exception:
            pass
        self.status.config(
            text=(f"Done. N={r.n_samples:,} spectra · {r.loadings.shape[1]} bands · "
                  f"method={r.method} · scaling={r.scaling}. "
                  f"PC1–PC{r.n_components} capture "
                  f"{100*r.cumulative_variance[-1]:.1f}% variance."))
        self._redraw()

    # ── colours ───────────────────────────────────────────────────────────────
    def _group_colors(self):
        pal = self._palette()
        uniq = list(dict.fromkeys(self._groups))
        return {g: pal[i % len(pal)] for i, g in enumerate(uniq)}, uniq

    def _color_assignments(self):
        """Return (values_array, cmap_dict, uniq_list, title_extra) — colours by
        cluster if a clustering exists, else by group."""
        pal = self._palette()
        if self._cluster is not None:
            uniq = sorted(set(self._cluster))
            cmap = {c: (pal[i % len(pal)] if c != -1 else "#999999")
                    for i, c in enumerate(uniq)}
            return np.asarray(self._cluster), cmap, uniq, " · coloured by cluster"
        cmap, uniq = self._group_colors()
        return np.asarray(self._groups), cmap, uniq, ""

    def _render_idx(self, n):
        """Indices of a representative random subsample for plotting huge clouds
        (the full data is always kept for the maths)."""
        cap = int(self.max_render.get())
        if n <= cap:
            return np.arange(n), False
        rng = np.random.default_rng(42)
        return np.sort(rng.choice(n, cap, replace=False)), True

    # ── drawing ───────────────────────────────────────────────────────────────
    def _welcome(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.6, "PCA Studio", ha="center", fontsize=22, weight="bold")
        ax.text(0.5, 0.48, "Add spectral files · set groups · Run PCA",
                ha="center", fontsize=11, color="#555")
        ax.text(0.5, 0.36, "scales to millions of spectra · robust & sparse · "
                "T² + Q diagnostics · 600 dpi vector export",
                ha="center", fontsize=9, color="#888")
        self.canvas.draw()

    def _redraw(self):
        if self._result is None:
            return
        plt.rcParams.update({"font.size": float(self.r_fontsize.get())})
        if self.r_apply_size.get():
            self.fig.set_size_inches(float(self.r_w_in.get()),
                                     float(self.r_h_in.get()))
        v = self._view.get()
        self.fig.clear()
        if v == "scores":
            self._draw_scores()
        elif v == "scores3d":
            self._draw_scores3d()
        elif v == "pcgrid":
            self._draw_pcgrid()
        elif v == "subset":
            self._draw_subset()
        elif v == "loadings":
            self._draw_loadings()
        elif v == "scree":
            self._draw_scree()
        elif v == "diag":
            self._draw_diag()
        self.fig.tight_layout()
        self.canvas.draw()

    def _draw_scores(self):
        r = self._result
        ax = self.fig.add_subplot(111)
        ix = int(self.pc_x.get()) - 1
        iy = int(self.pc_y.get()) - 1
        ix = min(ix, r.n_components - 1); iy = min(iy, r.n_components - 1)
        sx, sy = r.scores[:, ix], r.scores[:, iy]
        n = sx.size
        evr = r.explained_variance_ratio

        if self._cluster is not None:
            color_by, title_extra = self._cluster, " · coloured by cluster"
        else:
            color_by, title_extra = self._groups, ""

        cby, cmap, uniq, _ = self._color_assignments()
        alpha = float(self.r_alpha.get())
        big = n > 20000 and self.density.get()
        if big and self._cluster is None and len(set(self._groups)) == 1:
            hb = ax.hexbin(sx, sy, gridsize=80, cmap="viridis", mincnt=1, linewidths=0)
            self.fig.colorbar(hb, ax=ax, label="spectra per bin", shrink=0.8)
        else:
            s = self._marker_size(n)
            for g in uniq:
                m = cby == g
                ax.scatter(sx[m], sy[m], s=s, c=cmap[g], alpha=alpha,
                           edgecolors="none", rasterized=n > 5000,
                           label=self._disp(g))
                if self.ellipse.get() and m.sum() > 3:
                    self._conf_ellipse(ax, sx[m], sy[m], cmap[g])
            self._apply_legend(ax)

        ax.axhline(0, color="#bbb", lw=0.6, zorder=0)
        ax.axvline(0, color="#bbb", lw=0.6, zorder=0)
        self._finalize_axes(ax,
                            auto_title=f"PCA scores  (N = {n:,}){title_extra}",
                            auto_x=f"PC{ix+1}  ({100*evr[ix]:.1f}%)",
                            auto_y=f"PC{iy+1}  ({100*evr[iy]:.1f}%)")

    def _conf_ellipse(self, ax, x, y, color, conf=0.95):
        if x.size < 3:
            return
        cov = np.cov(x, y)
        try:
            from scipy.stats import chi2
            s = np.sqrt(chi2.ppf(conf, df=2))
        except Exception:
            s = 2.4477
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * s * np.sqrt(np.maximum(vals, 0))
        e = Ellipse((x.mean(), y.mean()), w, h, angle=theta, facecolor="none",
                    edgecolor=color, lw=1.3, ls="--", alpha=0.9)
        ax.add_patch(e)

    def _draw_scores3d(self):
        """3D scores cloud (PC1·PC2·PC3) — the Schuppert 2016 Fig. 1a 'global
        map' look: groups separating into lobes of a central cloud."""
        r = self._result
        ax = self.fig.add_subplot(111, projection="3d")
        ix = min(int(self.pc_x.get()) - 1, r.n_components - 1)
        iy = min(int(self.pc_y.get()) - 1, r.n_components - 1)
        iz = min(int(self.pc_z.get()) - 1, r.n_components - 1)
        evr = r.explained_variance_ratio
        cby, cmap, uniq, extra = self._color_assignments()
        idx, sub = self._render_idx(r.scores.shape[0])
        alpha = float(self.r_alpha.get())
        s = self._marker_size(idx.size, big_default=4, small_default=12)
        for g in uniq:
            m = idx[cby[idx] == g]
            if m.size == 0:
                continue
            ax.scatter(r.scores[m, ix], r.scores[m, iy], r.scores[m, iz],
                       s=s, c=cmap[g], alpha=alpha, depthshade=True,
                       edgecolors="none", rasterized=True, label=self._disp(g))
        ax.set_xlabel(f"PC{ix+1} ({100*evr[ix]:.1f}%)", labelpad=2)
        ax.set_ylabel(f"PC{iy+1} ({100*evr[iy]:.1f}%)", labelpad=2)
        ax.set_zlabel(f"PC{iz+1} ({100*evr[iz]:.1f}%)", labelpad=2)
        ax.view_init(elev=18, azim=-60)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_edgecolor("#dddddd")
            pane.pane.set_facecolor((1, 1, 1, 0))
            pane._axinfo["grid"].update(color="#eeeeee", linewidth=0.5)
        self._apply_legend(ax, three_d=True)
        note = f"  (showing {idx.size:,} of {r.scores.shape[0]:,})" if sub else ""
        t = self.r_title.get().strip()
        ax.set_title(t if t else f"3D PCA scores — global map{extra}{note}")

    def _draw_pcgrid(self):
        """Small-multiples of the first PC pairs — a compact 'global map' figure."""
        r = self._result
        evr = r.explained_variance_ratio
        cby, cmap, uniq, extra = self._color_assignments()
        idx, sub = self._render_idx(r.scores.shape[0])
        pal = self._palette()
        alpha = float(self.r_alpha.get())
        pairs = [(0, 1), (0, 2), (1, 2)]
        pairs = [(a, b) for a, b in pairs if b < r.n_components]
        n = len(pairs)
        s = self._marker_size(idx.size, big_default=4, small_default=12)
        axes = []
        for k, (a, b) in enumerate(pairs):
            ax = self.fig.add_subplot(1, n + 1, k + 1)
            for g in uniq:
                m = idx[cby[idx] == g]
                if m.size == 0:
                    continue
                ax.scatter(r.scores[m, a], r.scores[m, b], s=s, c=cmap[g],
                           alpha=alpha, edgecolors="none", rasterized=True,
                           label=self._disp(g))
            ax.axhline(0, color="#ccc", lw=0.5); ax.axvline(0, color="#ccc", lw=0.5)
            ax.set_xlabel(f"PC{a+1} ({100*evr[a]:.1f}%)")
            ax.set_ylabel(f"PC{b+1} ({100*evr[b]:.1f}%)")
            ax.grid(self.r_grid.get(), color="#e6e6e6", lw=0.5)
            axes.append(ax)
        # scree panel on the right
        axs = self.fig.add_subplot(1, n + 1, n + 1)
        kk = np.arange(1, r.n_components + 1)
        axs.bar(kk, 100 * evr, color=pal[0], alpha=0.85)
        axs.plot(kk, 100 * r.cumulative_variance, "-o", color=pal[1 % len(pal)],
                 lw=1.2, ms=3)
        axs.set_xlabel("PC"); axs.set_ylabel("Variance (%)"); axs.set_xticks(kk)
        if axes and self.r_legend_on.get():
            self._apply_legend(axes[-1])
        t = self.r_title.get().strip()
        self.fig.suptitle(t if t else f"PCA global map — PC pair grid{extra}",
                          y=0.99)

    # ── density cloud + subset highlight (Schuppert Fig. 4) ───────────────────
    def _density_layer(self, ax, sx, sy, color, bw=1.0, extent=None,
                       z=1, gamma=None, strength=None):
        """Draw a soft, translucent density 'cloud' for one group as a smooth
        glow (finely-binned, Gaussian-smoothed KDE rendered via imshow with an
        alpha ramp — no stepped contour bands).

        ``strength`` (0-1, defaults to the Transparency slider) controls how
        visible the cloud is: higher = more opaque/bolder background, and
        also lowers the gamma so faint outskirts are lifted more."""
        from matplotlib.colors import LinearSegmentedColormap, to_rgb
        if strength is None:
            try:
                strength = float(self.r_alpha.get())
            except Exception:
                strength = 0.6
        strength = min(max(strength, 0.05), 1.0)
        if gamma is None:
            gamma = 0.65 - 0.35 * strength   # more strength -> lower gamma -> brighter outskirts
        if sx.size < 8:
            ax.scatter(sx, sy, s=8, color=color, alpha=0.3, edgecolors="none",
                       zorder=z)
            return
        xmin, xmax, ymin, ymax = extent
        H, xe, ye = np.histogram2d(sx, sy, bins=240,
                                   range=[[xmin, xmax], [ymin, ymax]])
        try:
            from scipy.ndimage import gaussian_filter
            H = gaussian_filter(H, sigma=max(2.0, 6.0 * bw))
        except Exception:
            from numpy import convolve
            s = max(2.0, 6.0 * bw)
            k = np.exp(-0.5 * (np.arange(-int(3*s), int(3*s)+1) / s) ** 2); k /= k.sum()
            H = np.apply_along_axis(lambda m: convolve(m, k, "same"), 0, H)
            H = np.apply_along_axis(lambda m: convolve(m, k, "same"), 1, H)
        if H.max() <= 0:
            return
        H = (H.T / H.max()) ** gamma            # gamma lifts the faint outskirts
        r, g, b = to_rgb(color)
        # transparent → solid colour alpha ramp gives the soft "glow" look;
        # mid/top alpha scale with `strength` so the cloud reads more boldly
        a_mid = 0.20 + 0.35 * strength
        a_top = 0.55 + 0.40 * strength
        cmap = LinearSegmentedColormap.from_list(
            "d", [(r, g, b, 0.0), (r, g, b, a_mid), (r, g, b, min(a_top, 1.0))])
        Hm = np.ma.masked_less(H, max(0.04 - 0.03 * strength, 0.005))  # hide empty bg
        ax.imshow(Hm, origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]),
                  cmap=cmap, interpolation="bilinear", aspect="auto", zorder=z)

    def _draw_subset(self):
        r = self._result
        ix = min(int(self.pc_x.get()) - 1, r.n_components - 1)
        iy = min(int(self.pc_y.get()) - 1, r.n_components - 1)
        sx, sy = r.scores[:, ix], r.scores[:, iy]
        evr = r.explained_variance_ratio
        cmap, uniq = self._group_colors()
        groups = np.asarray(self._groups)

        # which groups to highlight (bright points) vs leave as background cloud
        raw = [s.strip() for s in self.sub_highlight.get().split(",") if s.strip()]
        hi = [g for g in uniq if str(g) in raw] if raw else []
        bg = [g for g in uniq if g not in hi] if hi else list(uniq)

        # If a residual-subset inset will be drawn, give it its own panel
        # alongside the main axes instead of floating on top of the cloud.
        show_inset = bool(hi) and self.sub_inset.get() and self._X is not None
        if show_inset:
            gs = self.fig.add_gridspec(1, 2, width_ratios=(2.4, 1.0), wspace=0.32)
            ax = self.fig.add_subplot(gs[0, 0])
            axin = self.fig.add_subplot(gs[0, 1])
        else:
            ax = self.fig.add_subplot(111)
            axin = None

        pad_x = 0.05 * (sx.max() - sx.min() + 1e-9)
        pad_y = 0.05 * (sy.max() - sy.min() + 1e-9)
        extent = (sx.min() - pad_x, sx.max() + pad_x,
                  sy.min() - pad_y, sy.max() + pad_y)

        # background density clouds (deeper colours drawn last so all show)
        for zi, g in enumerate(bg):
            m = groups == g
            self._density_layer(ax, sx[m], sy[m], cmap[g],
                                 bw=float(self.sub_bw.get()), extent=extent,
                                 z=zi + 1)
        # foreground highlighted points
        idx_all, _ = self._render_idx(sx.size)
        for g in hi:
            m = (groups == g)
            sel = idx_all[m[idx_all]]
            ax.scatter(sx[sel], sy[sel], s=self._marker_size(sel.size, 10, 22),
                       c=cmap[g], alpha=0.95, edgecolors="white", linewidths=0.3,
                       rasterized=sel.size > 5000, label=self._disp(g), zorder=5)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        if hi:
            self._apply_legend(ax)
        self._finalize_axes(
            ax,
            auto_title=("Density map" + (f" — {', '.join(map(str, hi))} highlighted"
                                         if hi else " (all groups)")),
            auto_x=f"PC{ix+1} ({100*evr[ix]:.1f}%)",
            auto_y=f"PC{iy+1} ({100*evr[iy]:.1f}%)")

        # ── residual-subset PCA panel (separate from the main axes, so it
        #    never overlaps/hinders the density-cloud view) ─────────────────
        if axin is not None:
            try:
                mask = np.isin(groups, hi)
                ndef = int(self.sub_ndeflate.get()) if self.sub_deflate.get() else 0
                sub = core.subset_pca(
                    self._X[mask], n_components=2, scaling=self._scaling_used,
                    deflate_loadings=r.loadings, deflate_mean=r.mean,
                    n_deflate=ndef)
                gsub = groups[mask]
                for g in hi:
                    mm = gsub == g
                    axin.scatter(sub.scores[mm, 0], sub.scores[mm, 1], s=8,
                                 c=cmap[g], alpha=0.85, edgecolors="none",
                                 label=self._disp(g))
                axin.set_xlabel(f"{sub.method} PC1", fontsize=9)
                axin.set_ylabel(f"{sub.method} PC2", fontsize=9)
                axin.tick_params(labelsize=8)
                axin.set_title("Subset re-analysed", fontsize=10)
                for s in ("top", "right"):
                    axin.spines[s].set_visible(False)
            except Exception as e:
                axin.text(0.02, 0.98, f"inset unavailable: {e}", fontsize=8,
                          transform=axin.transAxes, color="#999", va="top",
                          wrap=True)
                axin.axis("off")

    def _draw_loadings(self):
        r = self._result
        ax = self.fig.add_subplot(111)
        x = self._waves if self._waves is not None else np.arange(r.loadings.shape[1])
        pal = self._palette()
        show = min(r.n_components, 4)
        off = 0.0
        amp = np.nanmax(np.abs(r.loadings[:show])) or 1.0
        for k in range(show):
            ld = r.loadings[k]
            ax.plot(x, ld + off, color=pal[k % len(pal)], lw=1.1,
                    label=f"PC{k+1} ({100*r.explained_variance_ratio[k]:.1f}%)")
            ax.axhline(off, color="#ddd", lw=0.5)
            off += 1.2 * amp
        self._apply_legend(ax)
        self._finalize_axes(ax, auto_title=f"PCA loadings  ({r.method})",
                            auto_x="Wavenumber (cm$^{-1}$)",
                            auto_y="Loading (offset)")
        if self._waves is not None:
            ax.set_xlim(float(np.min(x)), float(np.max(x)))

    def _draw_scree(self):
        r = self._result
        ax = self.fig.add_subplot(111)
        pal = self._palette()
        k = np.arange(1, r.n_components + 1)
        evr = 100 * r.explained_variance_ratio
        ax.bar(k, evr, color=pal[0], alpha=0.85, label="individual")
        ax.set_ylabel("Explained variance (%)", color=pal[0])
        ax2 = ax.twinx()
        ax2.plot(k, 100 * r.cumulative_variance, "-o", color=pal[1 % len(pal)],
                 lw=1.4, label="cumulative")
        ax2.set_ylabel("Cumulative variance (%)", color=pal[1 % len(pal)])
        ax2.axhline(95, color="#888", ls="--", lw=0.8)
        ax2.set_ylim(0, 101)
        ax.set_xticks(k)
        self._finalize_axes(ax, auto_title="Scree & cumulative variance",
                            auto_x="Principal component", auto_y="")

    def _draw_diag(self):
        r = self._result
        ax = self.fig.add_subplot(111)
        if r.t2 is None or r.q is None:
            ax.text(0.5, 0.5, "Diagnostics unavailable for this variant",
                    ha="center"); ax.axis("off"); return
        t2, q = r.t2, r.q
        n = t2.size
        big = n > 20000 and self.density.get()
        if big and len(set(self._groups)) == 1:
            hb = ax.hexbin(t2, q, gridsize=70, cmap="magma", mincnt=1, bins="log")
            self.fig.colorbar(hb, ax=ax, label="log$_{10}$ count", shrink=0.8)
        else:
            cmap, uniq = self._group_colors()
            s = self._marker_size(n, big_default=8, small_default=14)
            for g in uniq:
                m = self._groups == g
                ax.scatter(t2[m], q[m], s=s, c=cmap[g], alpha=float(self.r_alpha.get()),
                           edgecolors="none", rasterized=n > 5000,
                           label=self._disp(g))
            self._apply_legend(ax)
        out = 0
        if r.t2_limit and np.isfinite(r.t2_limit):
            ax.axvline(r.t2_limit, color="#b3261e", ls="--", lw=1)
        if r.q_limit and np.isfinite(r.q_limit):
            ax.axhline(r.q_limit, color="#b3261e", ls=":", lw=1)
        if r.t2_limit and r.q_limit and np.isfinite(r.t2_limit) and np.isfinite(r.q_limit):
            out = int(np.sum((t2 > r.t2_limit) & (q > r.q_limit)))
        self._finalize_axes(
            ax, auto_title=f"Influence plot — {out:,} spectra beyond both 95% limits",
            auto_x="Hotelling $T^2$", auto_y="Q-residual (SPE)")

    # ── downstream ─────────────────────────────────────────────────────────────
    def _classify(self):
        if self._result is None:
            return
        if len(set(self._groups)) < 2:
            messagebox.showinfo("Classify",
                                "Need ≥2 groups. Use 'Set' to assign group names.",
                                parent=self); return
        win = tk.Toplevel(self); win.title("Classification"); win.configure(bg="white")
        method = "lda"
        try:
            out = core.classify_scores(self._result.scores, self._groups,
                                       method=method, groups=self._groups)
        except Exception as e:
            messagebox.showerror("Classify", str(e), parent=self); return
        fig = plt.figure(figsize=(5.5, 4.6)); ax = fig.add_subplot(111)
        cm = out["confusion"]; cls = out["classes"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(cls))); ax.set_yticks(range(len(cls)))
        ax.set_xticklabels(cls, rotation=45, ha="right"); ax.set_yticklabels(cls)
        for i in range(len(cls)):
            for j in range(len(cls)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{out['method'].upper()} — grouped CV acc "
                     f"{100*out['accuracy']:.1f}%")
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=win).get_tk_widget().pack(fill="both",
                                                                expand=True)

    def _cluster(self):
        if self._result is None:
            return
        try:
            out = core.cluster_scores(self._result.scores, method="kmeans",
                                      k=max(2, len(set(self._groups))))
        except Exception as e:
            messagebox.showerror("Cluster", str(e), parent=self); return
        self._cluster = out["labels"]
        sil = out["silhouette"]
        self.status.config(text=f"k-means: {len(set(out['labels']))} clusters"
                           + (f" · silhouette {sil:.2f}" if sil else ""))
        self._view.set("scores"); self._redraw()

    # ── export ─────────────────────────────────────────────────────────────────
    def _save_fig(self):
        if self._result is None:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF (vector)", "*.pdf"), ("SVG (vector)", "*.svg"),
                       ("PNG 600 dpi", "*.png")])
        if p:
            orig = self.fig.get_size_inches().copy()
            try:
                self.fig.set_size_inches(float(self.r_w_in.get()),
                                         float(self.r_h_in.get()))
                self.fig.savefig(p, dpi=int(self.r_dpi.get()), bbox_inches="tight")
            finally:
                self.fig.set_size_inches(orig)        # restore on-screen size
                self.canvas.draw_idle()
            self.status.config(
                text=f"Saved {Path(p).name} @ {int(self.r_dpi.get())} dpi, "
                     f"{self.r_w_in.get()}×{self.r_h_in.get()} in")

    def _save_csv(self):
        if self._result is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        r = self._result
        cols = [f"PC{i+1}" for i in range(r.n_components)]
        data = r.scores
        hdr = ["group", "label"] + cols
        rows = []
        for i in range(data.shape[0]):
            rows.append([self._groups[i], self._labels[i]] + list(data[i]))
        extra = ""
        if r.t2 is not None:
            hdr += ["T2", "Q"]
            for i, rr in enumerate(rows):
                rr += [r.t2[i], r.q[i]]
            extra = (f"# T2_limit_95={r.t2_limit}, Q_limit_95={r.q_limit}, "
                     f"method={r.method}, scaling={r.scaling}\n")
        with open(p, "w") as fh:
            fh.write(extra)
            fh.write(",".join(hdr) + "\n")
            for rr in rows:
                fh.write(",".join(map(str, rr)) + "\n")
        self.status.config(text=f"Saved {Path(p).name}")

    def _save_loadings(self):
        if self._result is None:
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        r = self._result
        x = self._waves if self._waves is not None else np.arange(r.loadings.shape[1])
        hdr = ["wavenumber"] + [f"PC{i+1}" for i in range(r.n_components)]
        M = np.column_stack([x, r.loadings.T])
        np.savetxt(p, M, delimiter=",", header=",".join(hdr), comments="")
        self.status.config(text=f"Saved {Path(p).name}")


def main():
    app = PCAStudio()
    app.mainloop()


if __name__ == "__main__":
    main()
