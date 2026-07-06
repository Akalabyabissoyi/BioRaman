#!/usr/bin/env python3
"""
BioRaman  —  Raman Hyperspectral Map Analysis for Biophysics
================================================
Created by: Akalabya Bissoyi
            <akalabya.bissoyi@manchester.ac.uk>  ·  <bissoyi.akalabya@gmail.com>
Gibson Group, University of Manchester  ·  https://gibsongroupresearch.com/

Copyright (c) 2026  Akalabya Bissoyi and the Gibson Group, University of Manchester

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. See the MIT
License (LICENSE file) for the full text.
================================================
NEW in v1.2.0 (Figure export & multi-region ROI):
• Individual panel export — every analysis window can now save each plot as
    its own publication-ready file (PNG 600 dpi + optional vector PDF), not
    just the whole page:
    - MCR-ALS & N-FINDR: combined spectra, each component/endmember spectrum,
      and each abundance map (with its own colorbar + µm scale bar)
    - Clustering: cluster map, mean spectra, pixel-count panels
    - PCA: scores, loadings, distribution, scree, spatial map (pre-existing)
• Multi-region ROI — "Multi-region" checkbox + "＋ Add Region" button let you
    union several ROIs (any mix of rect/ellipse/polygon/freehand) into a single
    selection used by ROI Analysis, MCR-ALS, Clustering and N-FINDR

NEW in v13 (Reproducibility, Batch & QC):
• Preprocessing recipes — save_recipe() / load_recipe()
    - Save/load all preprocessing parameters as a JSON recipe for reproducible,
      shareable analyses
• Non-destructive reprocessing — reprocess()
    - Re-apply a changed recipe to the retained raw cube without reloading
• Batch processing — open_batch() / run_batch()
    - Apply one recipe to a whole folder of files; writes processed outputs
      plus a batch_summary.csv
• Quality-control maps — open_qc_map()
    - Per-pixel SNR, total/max intensity and detector-saturation maps
• Analysis report — save_report()
    - One-click self-contained HTML report (map, mean spectrum, recipe, log)
• Session save/restore — save_session() / load_session()
• Headless CLI — run `python bioraman.py --input … --out … [--recipe …]`
    for GUI-free batch processing in pipelines/servers
• Cluster validation — mean silhouette score reported after clustering
• HDF5 export added to Save Processed Data

NEW in v12 (Processed-Data Export):
• Save Processed Data — save_processed()
    - Exports the full preprocessed spectral cube (baseline-corrected,
      smoothed, normalised, cosmic-ray-cleaned) plus the wavenumber axis
    - Formats: .npz (lossless, reloadable), .csv/.txt/.dpt (Renishaw long
      format for maps — round-trips through the built-in reader), .mat (SciPy)
    - Available from File menu, the toolbar, and Ctrl+Shift+S
    - Note: proprietary instrument containers (.wdf/.wip/.spc) cannot be
      rewritten, so processed data is exported to open formats instead

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

__author__  = "Akalabya Bissoyi"
__email__   = "akalabya.bissoyi@manchester.ac.uk, bissoyi.akalabya@gmail.com"
__affiliation__ = "Gibson Group, University of Manchester"
__url__     = "https://gibsongroupresearch.com/"
__license__ = "MIT"
__version__ = "1.4.0"  # v6: WL overlay metadata alignment + per-panel image/spectrum export

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, sys, time, threading, queue
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _ensure_package(import_name, pip_name=None):
    """Import a package, transparently pip-installing it into the *current*
    interpreter (sys.executable) if it is missing.

    Returns the imported module, or None if it is unavailable and could not be
    installed. Set the environment variable BIORAMAN_NO_AUTOINSTALL=1 to disable
    automatic installation (e.g. for offline or frozen builds).
    """
    import importlib
    try:
        return importlib.import_module(import_name)
    except Exception:
        pass
    # Never attempt a pip install from inside a frozen/standalone build
    # (sys.executable is the app itself, so the install hangs and can leave
    # readers unavailable). All needed packages are bundled at build time.
    if os.environ.get("BIORAMAN_NO_AUTOINSTALL") or getattr(sys, "frozen", False):
        return None
    pip_name = pip_name or import_name
    try:
        import subprocess
        print(f"[BioRaman] '{pip_name}' not found — installing it for "
              f"{sys.executable} …", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", pip_name])
        importlib.invalidate_caches()
        mod = importlib.import_module(import_name)
        print(f"[BioRaman] Installed '{pip_name}'.", flush=True)
        return mod
    except Exception as exc:
        print(f"[BioRaman] Could not auto-install '{pip_name}': {exc}\n"
              f"           Install it manually with:\n"
              f"           {sys.executable} -m pip install {pip_name}",
              flush=True)
        return None

# ── numeric / scientific ──────────────────────────────────────────────────────
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter, gaussian_filter1d, zoom, median_filter

# NumPy 2.0 renamed ``np.trapz`` → ``np.trapezoid`` (the old name was removed).
# Bind to whichever exists so the app works on both NumPy 1.x and 2.x.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _bundled_library_dir():
    """Return the path to the bundled PCRS starter reference library, or None.

    Looks for a ``pcrs_library`` folder next to this script (and next to a
    PyInstaller bundle, via ``sys._MEIPASS``)."""
    cands = []
    try:
        cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "pcrs_library"))
    except Exception:
        pass
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands.append(os.path.join(meipass, "pcrs_library"))
    for d in cands:
        if os.path.isdir(d):
            return d
    return None

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

# Native Renishaw .wdf reader — auto-installed on first run if missing, because
# it is what recovers the true X×Y map geometry from .wdf files.
if _ensure_package("renishawWiRE") is not None:
    try:
        from renishawWiRE import WDFReader
        HAS_WDF = True
    except Exception:
        HAS_WDF = False
else:
    HAS_WDF = False

# Universal multi-format reader (wdf/spc/jdx/dpt/dat/txt/csv …)
try:
    from raman_io import open_raman, SUPPORTED_PATTERNS
    HAS_RAMANIO = True
except Exception:
    HAS_RAMANIO = False
    SUPPORTED_PATTERNS = [("Renishaw WDF", "*.wdf"), ("All files", "*.*")]

# WITec .wip reader (via RamanSPy or the photonicdata wip_loader)
try:
    import ramanspy as _ramanspy
    HAS_RAMANSPY = True
except Exception:
    HAS_RAMANSPY = False

try:
    from wip_loader import load as _wip_load   # photonicdata-files-wip
    HAS_WIPLOADER = True
except Exception:
    HAS_WIPLOADER = False

HAS_WITEC = HAS_RAMANSPY or HAS_WIPLOADER

# Add WITec to the file-open patterns no matter which base reader is active.
if not any("wip" in pat.lower() for _, pat in SUPPORTED_PATTERNS):
    SUPPORTED_PATTERNS = (
        [("Raman data", "*.wdf *.wip *.spc *.jdx *.txt *.csv *.dpt *.dat"),
         ("WITec WIP", "*.wip"), ("Renishaw WDF", "*.wdf")]
        + [p for p in SUPPORTED_PATTERNS if "wdf" not in p[1].lower()]
    )

# v4 — add the open columnar formats to the master open dialog.
if not any("parquet" in pat.lower() for _, pat in SUPPORTED_PATTERNS):
    if SUPPORTED_PATTERNS and SUPPORTED_PATTERNS[0][0] == "Raman data":
        _lbl, _pat = SUPPORTED_PATTERNS[0]
        SUPPORTED_PATTERNS[0] = (_lbl, _pat + " *.parquet *.feather *.arrow")
    SUPPORTED_PATTERNS = (SUPPORTED_PATTERNS
        + [("Parquet/Feather cube", "*.parquet *.feather *.arrow")])

# Files at or above this size (MB) trigger a "this may be slow" caution on load.
LARGE_FILE_WARN_MB = 500.0


class _WITecReader:
    """Adapter for WITec ``.wip`` files exposing the same minimal interface as
    the WDF/raman_io readers: ``.xdata`` (1-D wavenumbers, cm⁻¹) and
    ``.spectra`` (a Y×X×W intensity cube). Single spectra and line scans are
    promoted to a cube so the existing map pipeline can consume them."""

    def __init__(self, path):
        if not HAS_WITEC:
            raise RuntimeError(
                "Reading WITec .wip files needs RamanSPy.\n"
                "Install it with:\n    pip install ramanspy\n"
                "(or the photonicdata 'wip_loader' package).")
        axis, data = self._read(path)
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:                 # single spectrum  -> 1×1×W
            data = data[None, None, :]
        elif data.ndim == 2:               # line scan (N×W)   -> 1×N×W
            data = data[None, :, :]
        self.xdata   = np.asarray(axis, dtype=float)
        self.spectra = data
        self.img     = None                # no embedded white-light image

    @staticmethod
    def _read(path):
        # Preferred: RamanSPy returns a SpectralContainer (or a list of them).
        if HAS_RAMANSPY:
            obj = _ramanspy.load.witec(path)
            if isinstance(obj, (list, tuple)):
                obj = obj[0]
            return obj.spectral_axis, obj.spectral_data
        # Fallback: photonicdata wip_loader (dict-like).
        wip = _wip_load(path)
        graph = None
        for v in (wip.values() if hasattr(wip, "values") else []):
            if hasattr(v, "spectral_data") or hasattr(v, "data"):
                graph = v; break
        if graph is None:
            raise RuntimeError("No spectral graph found in WIP file.")
        axis = getattr(graph, "spectral_axis", getattr(graph, "x", None))
        data = getattr(graph, "spectral_data", getattr(graph, "data", None))
        return axis, data


class _TextReader:
    """Native reader for ASCII Raman files (``.txt``/``.csv``/``.dpt``/``.dat``/
    ``.jdx``/``.asc``) following the RAMANMETRIX data conventions:

    * Single spectrum  — two columns: wavenumber, intensity.
    * Multiple spectra — wavenumber axis in the first column (preferred) or the
      first row; the remaining columns/rows are intensities.
    * Renishaw "long" export with named columns — a header containing
      ``#Wave`` and ``#Intensity`` (optionally ``#X``/``#Y`` for a scan or
      ``#Time`` for a time series). The long table is reshaped back into a
      proper spectrum / line / map cube.

    Delimiters (tab / comma / semicolon / whitespace) are auto-detected and
    comment lines beginning with ``%`` are ignored. Exposes ``.xdata``
    (1-D wavenumbers) and ``.spectra`` (Y×X×W cube)."""

    def __init__(self, path):
        axis, cube = self._parse(path)
        axis = np.asarray(axis, dtype=float)
        cube = np.asarray(cube, dtype=float)
        if cube.ndim == 1:
            cube = cube[None, None, :]
        elif cube.ndim == 2:
            cube = cube[None, :, :]
        # keep wavenumbers ascending (rest of the app assumes this)
        if axis.size > 1 and axis[0] > axis[-1]:
            axis = axis[::-1]
            cube = cube[:, :, ::-1]
        self.xdata   = axis
        self.spectra = cube
        self.img     = None

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _read_lines(path):
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            return [ln.rstrip("\n\r") for ln in fh]

    @staticmethod
    def _pick_delim(sample):
        for d in ("\t", ";", ","):     # tab/semicolon win over comma (EU decimals)
            if d in sample:
                return d
        return None

    @classmethod
    def _parse(cls, path):
        raw = cls._read_lines(path)

        # 1) Renishaw named-column long format?  (#Wave + #Intensity header)
        for i, ln in enumerate(raw):
            low = ln.lower()
            if "#wave" in low and "#intensity" in low:
                return cls._parse_named(raw, i)

        # 2) Plain numeric table (the common cases).
        return cls._parse_numeric(raw)

    # ── plain numeric tables ────────────────────────────────────────────────
    @classmethod
    def _parse_numeric(cls, raw):
        body = [ln for ln in raw
                if ln.strip() and not ln.lstrip().startswith(("#", "%"))]
        if not body:
            raise RuntimeError("File contains no numeric data.")
        delim = cls._pick_delim(body[min(len(body) - 1, 5)])

        def split(line):
            return line.split(delim) if delim else line.split()

        def to_float(tokens):
            out = []
            for t in tokens:
                t = t.strip()
                if delim != ",":
                    t = t.replace(",", ".")
                try:
                    out.append(float(t))
                except ValueError:
                    return None
            return out

        if to_float(split(body[0])) is None:    # drop a text header line
            body = body[1:]
        rows = [v for v in (to_float(split(ln)) for ln in body) if v]
        if not rows:
            raise RuntimeError("Could not parse any numeric rows.")
        # F2 — keep the MODAL (most common) row width, not the maximum. A single
        # corrupt line with extra columns must not be allowed to discard every
        # well-formed row and silently return the lone anomalous one.
        from collections import Counter
        width = Counter(len(r) for r in rows).most_common(1)[0][0]
        kept  = [r for r in rows if len(r) == width]
        if len(kept) != len(rows):
            print(f"[BioRaman] ASCII reader: skipped {len(rows) - len(kept)} row(s) "
                  f"with a non-standard column count (kept {len(kept)} rows of "
                  f"width {width}).", flush=True)
        arr = np.asarray(kept, dtype=float)

        if arr.shape[1] == 2:                    # single spectrum: wn, intensity
            order = np.argsort(arr[:, 0])
            return arr[order, 0], arr[order, 1]
        if arr.shape[1] > 2:
            axis = arr[:, 0]
            if np.all(np.diff(axis) > 0) or np.all(np.diff(axis) < 0):
                return axis, arr[:, 1:].T        # spectra in columns
            return arr[0, :], arr[1:, :]         # else wavenumbers in first row
        raise RuntimeError("Unrecognised text layout (need ≥2 columns).")

    # ── Renishaw long format (#X #Y #Wave #Intensity) ───────────────────────
    @classmethod
    def _parse_named(cls, raw, hdr_idx):
        header = raw[hdr_idx]
        names = [t.strip().lstrip("#").lower() for t in header.split()]
        def idx(name):
            return names.index(name) if name in names else None
        wi, ii = idx("wave"), idx("intensity")
        xi, yi, ti = idx("x"), idx("y"), idx("time")
        if wi is None or ii is None:
            raise RuntimeError("Named Renishaw export missing #Wave/#Intensity.")

        rows = []
        for ln in raw[hdr_idx + 1:]:
            if not ln.strip() or ln.lstrip().startswith(("#", "%")):
                continue
            toks = ln.split()
            try:
                rows.append([float(t) for t in toks])
            except ValueError:
                continue
        if not rows:
            raise RuntimeError("No numeric data rows after the header.")
        # F2 — modal column count (see _parse_numeric); guard against one stray
        # over-wide line wiping out the whole table.
        from collections import Counter
        ncol = Counter(len(r) for r in rows).most_common(1)[0][0]
        kept = [r for r in rows if len(r) == ncol]
        if len(kept) != len(rows):
            print(f"[BioRaman] Renishaw reader: skipped {len(rows) - len(kept)} "
                  f"row(s) with a non-standard column count.", flush=True)
        arr = np.asarray(kept, dtype=float)

        wave = arr[:, wi]
        inten = arr[:, ii]

        # build the per-point grouping key
        if xi is not None and yi is not None:
            keys = list(zip(arr[:, xi], arr[:, yi]))
        elif ti is not None:
            keys = list(zip(arr[:, ti]))
        else:
            keys = None

        if keys is None:                          # single spectrum
            return wave, inten

        # split the long table into one spectrum per (changing) key
        spectra, axes, cur = [], [], object()
        buf_w, buf_i = [], []
        coords = []
        for k, w, it in zip(keys, wave, inten):
            if k != cur:
                if buf_i:
                    spectra.append(buf_i); axes.append(buf_w); coords.append(cur)
                cur, buf_w, buf_i = k, [], []
            buf_w.append(w); buf_i.append(it)
        if buf_i:
            spectra.append(buf_i); axes.append(buf_w); coords.append(cur)

        W = min(len(s) for s in spectra)
        spectra = [s[:W] for s in spectra]
        axis = np.asarray(axes[0][:W], dtype=float)
        mat = np.asarray(spectra, dtype=float)    # (n_spectra, W)

        # reshape to a 2-D map when (X, Y) form a regular grid
        if xi is not None and yi is not None:
            xs = sorted({c[0] for c in coords})
            ys = sorted({c[1] for c in coords})
            if len(xs) * len(ys) == mat.shape[0] and len(xs) > 1 and len(ys) > 1:
                xpos = {v: j for j, v in enumerate(xs)}
                ypos = {v: j for j, v in enumerate(ys)}
                cube = np.zeros((len(ys), len(xs), W), dtype=float)
                for (cx, cy), spec in zip(coords, mat):
                    cube[ypos[cy], xpos[cx]] = spec
                return axis, cube
        return axis, mat                          # else 1×N×W line/series


def _reader_map_shape(r):
    """Best-effort (rows, cols) map geometry from a reader object.

    Stage X/Y positions are used first because some Renishaw StreamLine maps
    report a degenerate ``map_shape`` of 1×N (a flat line) even though the data
    is a true 2-D grid — the real geometry is then only recoverable from the
    per-spectrum xpos/ypos arrays.
    """
    # 1) derive the grid from the stage X/Y position arrays (most reliable)
    for ax, ay in (("xpos", "ypos"), ("x_pos", "y_pos"),
                   ("map_xpos", "map_ypos")):
        xp, yp = getattr(r, ax, None), getattr(r, ay, None)
        try:
            if xp is not None and yp is not None:
                xp = np.round(np.asarray(xp, dtype=float).ravel(), 3)
                yp = np.round(np.asarray(yp, dtype=float).ravel(), 3)
                if xp.size == yp.size and xp.size > 1:
                    nx, ny = np.unique(xp).size, np.unique(yp).size
                    if nx > 1 and ny > 1 and nx * ny == xp.size:
                        return ny, nx
        except Exception:
            pass
    # 2) explicit shape attributes — but reject degenerate 1×N "line" shapes
    for attr in ("map_shape", "map_size", "spatial_shape"):
        ms = getattr(r, attr, None)
        if ms is not None:
            try:
                a, b = int(ms[0]), int(ms[1])
                if a > 1 and b > 1:
                    return a, b
            except Exception:
                pass
    # 3) renishawWiRE per-axis dimension counts
    for ax, ay in (("map_x", "map_y"), ("ncollected_x", "ncollected_y")):
        nx, ny = getattr(r, ax, None), getattr(r, ay, None)
        try:
            if nx and ny and int(nx) > 1 and int(ny) > 1:
                return int(ny), int(nx)
        except Exception:
            pass
    return None


def _to_cube(spectra, map_shape=None, n_points=None):
    """Promote any reader's spectra to a Y×X×W cube.

    Uses ``n_points`` (the number of wavenumbers, i.e. len(xdata)) to recover
    the spectral axis when the backend returns a flattened array:

    1-D (W,)            -> 1×1×W                       (single spectrum)
    1-D (N*W,)          -> map_shape×W or 1×N×W        (flattened map)
    2-D (N, W)          -> map_shape×W or 1×N×W
    2-D (W, N)          -> transposed first if it matches n_points
    3-D (Y, X, W)       -> unchanged
    """
    s = np.asarray(spectra, dtype=float)
    if s.ndim == 3:
        return s

    def _shape_NW(N, W, flat):
        if map_shape is not None:
            a, b = map_shape
            if a * b == N:
                return flat.reshape(a, b, W)
        return flat.reshape(1, N, W)

    if s.ndim == 2:
        N, W = s.shape
        # detect a transposed (W, N) layout using the known wavenumber count
        if n_points and W != n_points and N == n_points:
            s = s.T; N, W = s.shape
        return _shape_NW(N, W, s)

    if s.ndim == 1:
        if n_points and s.size % n_points == 0 and s.size != n_points:
            N = s.size // n_points          # flattened map: N pixels × W points
            return _shape_NW(N, n_points, s.reshape(N, n_points))
        return s.reshape(1, 1, s.size)      # single spectrum

    raise RuntimeError(f"Unsupported spectra array with {s.ndim} dimensions.")


def _cube_from_positions(spectra2d, xp, yp):
    """Build a Y×X×W cube by assigning each spectrum to its (x, y) stage cell.

    Handles maps where several spectra share a grid position (e.g. depth or
    repeat acquisitions): collisions at the same cell are averaged. Robust to
    acquisition order and to a multiplicity factor (count = nx·ny·k).
    Returns None if the positions do not form a usable 2-D grid.
    """
    s = np.asarray(spectra2d, dtype=float)
    xp = np.round(np.asarray(xp, dtype=float).ravel(), 3)
    yp = np.round(np.asarray(yp, dtype=float).ravel(), 3)
    N, W = s.shape
    if xp.size != N or yp.size != N:
        return None
    xs = np.unique(xp); ys = np.unique(yp)
    nx, ny = xs.size, ys.size
    if nx < 2 or ny < 2 or nx * ny > N:
        return None
    xi = np.searchsorted(xs, xp)
    yi = np.searchsorted(ys, yp)
    cube = np.zeros((ny, nx, W), dtype=float)
    cnt  = np.zeros((ny, nx), dtype=float)
    np.add.at(cube, (yi, xi), s)
    np.add.at(cnt,  (yi, xi), 1.0)
    cnt[cnt == 0] = 1.0
    cube /= cnt[:, :, None]
    return cube


def _volume_from_positions(spectra2d, xp, yp, zp):
    """Build a Z×Y×X×W confocal volume from per-spectrum stage positions.

    Returns (volume, zvals, yvals, xvals) or None if the positions do not form
    a 3-D grid with more than one depth plane.
    """
    s = np.asarray(spectra2d, dtype=float)
    xp = np.round(np.asarray(xp, float).ravel(), 3)
    yp = np.round(np.asarray(yp, float).ravel(), 3)
    zp = np.round(np.asarray(zp, float).ravel(), 3)
    N, W = s.shape
    if not (xp.size == yp.size == zp.size == N):
        return None
    xs, ys, zs = np.unique(xp), np.unique(yp), np.unique(zp)
    nx, ny, nz = xs.size, ys.size, zs.size
    if nz < 2 or nx < 2 or ny < 2 or nx * ny * nz > N:
        return None
    xi = np.searchsorted(xs, xp)
    yi = np.searchsorted(ys, yp)
    zi = np.searchsorted(zs, zp)
    vol = np.zeros((nz, ny, nx, W), dtype=float)
    cnt = np.zeros((nz, ny, nx), dtype=float)
    np.add.at(vol, (zi, yi, xi), s)
    np.add.at(cnt, (zi, yi, xi), 1.0)
    cnt[cnt == 0] = 1.0
    vol /= cnt[:, :, :, None]
    return vol, zs, ys, xs


def _normalise_reader(r):
    """Ensure a reader object exposes a 3-D ``.spectra`` cube.

    Prefers stage-position gridding (xpos/ypos) so that StreamLine / depth
    maps — where ``count`` is a multiple of the spatial grid — are folded into
    the correct Y×X×W shape instead of a flat line. When a depth axis (zpos)
    with several planes is present, the full Z×Y×X×W confocal volume is also
    reconstructed and attached as ``r._volume`` (with ``r._zvals``)."""
    s = np.asarray(getattr(r, "spectra"), dtype=float)
    try:
        x = np.asarray(getattr(r, "xdata", None))
        W = int(x.size) if x is not None and x.ndim >= 1 else None
    except Exception:
        W = None

    # 1) position-based gridding (most reliable for real instrument maps)
    xp, yp = getattr(r, "xpos", None), getattr(r, "ypos", None)
    if W and xp is not None and yp is not None and s.ndim < 3:
        try:
            xp = np.asarray(xp, dtype=float).ravel()
            yp = np.asarray(yp, dtype=float).ravel()
            N = xp.size
            s2 = None
            if s.ndim == 1 and s.size == N * W:
                s2 = s.reshape(N, W)
            elif s.ndim == 2 and s.shape == (N, W):
                s2 = s
            elif s.ndim == 2 and s.shape == (W, N):
                s2 = s.T
            if s2 is not None:
                cube = _cube_from_positions(s2, xp, yp)
                if cube is not None:
                    # physical pixel size (µm) from stage-position spacing
                    try:
                        xs = np.unique(np.round(xp, 3)); ys = np.unique(np.round(yp, 3))
                        dx = np.median(np.diff(xs)) if xs.size > 1 else 0
                        dy = np.median(np.diff(ys)) if ys.size > 1 else 0
                        vals = [v for v in (abs(dx), abs(dy)) if v > 0]
                        if vals:
                            r._px_um = float(np.mean(vals))
                    except Exception:
                        pass
                    # reconstruct the true confocal volume when a depth axis exists
                    zp = getattr(r, "zpos", None)
                    if zp is not None:
                        try:
                            volpack = _volume_from_positions(s2, xp, yp, zp)
                            if volpack is not None:
                                r._volume, r._zvals, r._yvals, r._xvals = volpack
                        except Exception:
                            pass
                    r.spectra = cube
                    return r
        except Exception:
            pass

    # 2) fall back to shape/attribute-based promotion
    try:
        r.spectra = _to_cube(s, _reader_map_shape(r), W)
    except Exception:
        s2 = np.atleast_2d(s)
        r.spectra = s2[None, :, :] if s2.ndim == 2 else s2
    return r


def _open_raman_any(path):
    """Open any supported Raman file, routing by extension.

    Whatever the backend returns, the result is normalised so ``.spectra`` is
    always a 3-D (Y × X × W) cube — single spectra and line scans are promoted,
    and 2-D map exports are reshaped using the reader's map geometry."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wip":
        return _WITecReader(path)
    if ext in (".parquet", ".feather", ".arrow"):
        return read_cube_table(path)
    if ext in (".txt", ".csv", ".dpt", ".dat", ".jdx", ".asc"):
        if HAS_RAMANIO:
            try:
                return _normalise_reader(open_raman(path))
            except Exception:
                pass
        # Try the generic text reader, then fall back to the Horiba map parser
        try:
            return _TextReader(path)
        except Exception:
            return _read_horiba_map_text(path)
    if ext == ".wdf":
        # Prefer the native Renishaw reader: it reconstructs proper 2-D map
        # cubes (Y×X×W). Fall back to the universal raman_io reader.
        if HAS_WDF:
            try:
                return _normalise_reader(WDFReader(path))
            except Exception:
                pass
        if HAS_RAMANIO:
            return _normalise_reader(open_raman(path))
        raise RuntimeError(
            "Reading .wdf needs renishawWiRE (pip install renishawWiRE) "
            "or the raman_io package.")
    if HAS_RAMANIO:
        return _normalise_reader(open_raman(path))
    if HAS_WDF:
        return _normalise_reader(WDFReader(path))
    raise RuntimeError(f"No reader available for {ext} files.")


# RAMANMETRIX-compatible metadata workflow (ZIP + metadata table)
try:
    import raman_metadata as rmeta
    HAS_RMETA = True
except Exception:
    HAS_RMETA = False

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
# PUPAE EXPERIMENT — filename → (diet, temperature) parser
# Recognises stems like  Con5-1  Gly15-2  Pro5-3  Tre15-1
# ─────────────────────────────────────────────────────────────────────────────
import re as _re
_DIET_MAP = {"con": "Control", "gly": "Glycerol",
             "pro": "Proline", "tre": "Trehalose"}

def parse_pupae_label(stem):
    """UNIVERSAL build: factor auto-parsing disabled. Every file keeps its
    filename stem as its group label; use the 'Set' label box to group files.
    (See bioraman_pupae.py for the diet/temperature-aware version.)"""
    return None


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PreprocessParams:
    cosmic_removal:   bool  = True
    cosmic_threshold: float = 12.0  # median-residual modified-Z; higher = more conservative
    cosmic_width:     int   = 3     # max contiguous width (px) treated as a cosmic spike
    dark_removal:     bool  = True
    baseline_method:  str   = "asls"   # asls|arpls|drpls|none
    asls_lam:         float = 1e5
    asls_p:           float = 0.001
    smoothing:        bool  = True
    sg_window:        int   = 11
    sg_poly:          int   = 3
    normalisation:    str   = "area"   # max|area|snv|vector|none  — SNV/vector best when CH-stretch dominates


def _detect_cosmic(s, thr, width):
    """Width-aware cosmic-ray detector (F1).

    Cosmic rays are sub-resolution (1-2 px) POSITIVE spikes. We flag points
    whose residual from a size-3 median filter is a strong positive outlier
    (modified Z-score > ``thr``), then only correct contiguous runs no wider
    than ``width`` px — so genuine, multi-point Raman bands are preserved.
    The previous detector used the modified Z-score of the first derivative,
    which fires on the apex of any sharp real band and could erase it.

    Returns (corrected_spectrum, n_points_removed, removed_index_list).
    """
    s = s.astype(float).copy()
    removed = []
    if len(s) < 5:
        return s, 0, removed
    med   = median_filter(s, size=3, mode="nearest")
    resid = s - med
    rmad  = np.median(np.abs(resid - np.median(resid))) or 1e-10
    sigma = 1.4826 * rmad                          # robust noise scale
    flag  = resid > thr * sigma                    # positive-going only
    if not flag.any():
        return s, 0, removed
    width = max(1, int(width))
    idx   = np.where(flag)[0]
    # group consecutive flagged indices; correct only narrow (cosmic) runs
    for g in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if len(g) <= width:
            lo, hi = int(g[0]), int(g[-1])
            lft = max(0, lo - 1); rgt = min(len(s) - 1, hi + 1)
            if lft != rgt:
                s[lo:hi + 1] = np.interp(np.arange(lo, hi + 1),
                                         [lft, rgt], [s[lft], s[rgt]])
                removed.extend(range(lo, hi + 1))
    return s, len(removed), removed


def _process_one(args):
    """Top-level worker for ProcessPoolExecutor (must be picklable).

    Returns ``(idx, s, info)`` where ``info`` is a dict with keys
    ``spike`` (bool), ``n_removed`` (int) and ``baseline_ok`` (bool|None,
    None when no baseline was requested).
    """
    idx, s_raw, p = args
    s = s_raw.astype(float)
    # F6 — sanitise non-finite input so NaN/Inf cannot poison downstream maths
    if not np.all(np.isfinite(s)):
        finite = np.isfinite(s)
        if finite.any():
            xi = np.arange(len(s))
            s = np.interp(xi, xi[finite], s[finite])
        else:
            s = np.zeros_like(s)
    n_removed = 0; baseline_ok = None

    # 1. Cosmic-ray removal — width-aware median-residual outlier rejection (F1)
    if p.cosmic_removal:
        s, n_removed, _ = _detect_cosmic(s, p.cosmic_threshold, p.cosmic_width)
    had_spike = n_removed > 0

    # 2. Dark / pedestal
    if p.dark_removal:
        s = s - s.min()

    # 3. Baseline correction
    if p.baseline_method != "none" and HAS_PYBL:
        baseline_ok = False
        try:
            if   p.baseline_method == "asls":
                bl, _ = whittaker.asls(s,  lam=p.asls_lam, p=p.asls_p, max_iter=50)
            elif p.baseline_method == "arpls":
                bl, _ = whittaker.arpls(s, lam=p.asls_lam, max_iter=50)
            elif p.baseline_method == "drpls":
                bl, _ = whittaker.drpls(s, lam=p.asls_lam, max_iter=50)
            else:
                bl, _ = whittaker.asls(s,  lam=p.asls_lam, p=p.asls_p, max_iter=50)
            # F3 — subtract the baseline WITHOUT clipping negatives to zero.
            # Half-wave rectifying the noise floor here biases the area used
            # for normalisation upward in proportion to the noise level. We
            # keep the (small, zero-mean) residual noise so area/normalisation
            # stay quantitatively faithful; downstream solvers tolerate it.
            s = s - bl
            baseline_ok = True
        except Exception:
            # F5 — do not silently swallow. Leave the spectrum uncorrected and
            # record the failure so the report reflects what actually happened.
            baseline_ok = False

    # 4. Savitzky-Golay smoothing
    if p.smoothing and len(s) > p.sg_window:
        w = p.sg_window if p.sg_window % 2 == 1 else p.sg_window + 1
        w = max(w, p.sg_poly + 2)
        if w % 2 == 0:                 # F7 — savgol requires an ODD window
            w += 1
        poly = min(p.sg_poly, w - 1)   # and polyorder < window_length
        if w <= len(s):
            s = savgol_filter(s, window_length=w, polyorder=poly)

    # 5. Normalisation
    if p.normalisation == "max":
        pk = s.max()
        if pk > 0: s /= pk
    elif p.normalisation == "area":
        # F3 — integrate the un-rectified signal. With baseline-corrected,
        # roughly zero-mean noise this is an unbiased estimate of the true
        # band area and is independent of the noise level (unlike clip-then-
        # integrate, which inflates the denominator for noisier spectra).
        area = float(_trapz(s))
        if area > 0: s /= area
    elif p.normalisation == "snv":
        # Standard Normal Variate: (x - mean) / std per spectrum.
        # Removes multiplicative scaling + additive offset; robust when the
        # CH-stretch dominates overall intensity.
        mu = s.mean(); sd = s.std()
        if sd > 0: s = (s - mu) / sd
    elif p.normalisation == "vector":
        # L2 / unit-vector normalisation: x / ||x||
        nrm = float(np.sqrt(np.sum(s ** 2)))
        if nrm > 0: s = s / nrm

    info = {"spike": had_spike, "n_removed": n_removed, "baseline_ok": baseline_ok}
    return idx, s, info


def preprocess_spectrum(s, params=None):
    if params is None: params = PreprocessParams()
    _, result, _ = _process_one((0, s, params))
    return result


def preprocess_map(data, params=None, cb=None):
    if params is None: params = PreprocessParams()
    t0 = time.perf_counter()
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:          # single spectrum  -> 1×1×W
        data = data[None, None, :]
    elif data.ndim == 2:        # line scan (N×W)   -> 1×N×W
        data = data[None, :, :]
    Y, X, W  = data.shape
    total    = Y * X
    flat     = [(i, data[y, x], params)
                for i, (y, x) in enumerate(np.ndindex(Y, X))]
    out_flat = [None] * total
    cosmic_n = 0          # spectra with >=1 cosmic spike removed
    cosmic_pts = 0        # total cosmic points interpolated
    baseline_fail = 0     # spectra where baseline correction was requested but failed
    done     = 0
    n_workers = min(os.cpu_count() or 1, 8)

    def _tally(info):
        nonlocal cosmic_n, cosmic_pts, baseline_fail
        if info.get("spike"):            cosmic_n   += 1
        cosmic_pts += int(info.get("n_removed", 0))
        if info.get("baseline_ok") is False: baseline_fail += 1

    def _run_serial():
        """Process every spectrum in-process. Always works; the safe fallback."""
        nonlocal done
        for item in flat:
            idx, s_proc, info = _process_one(item)
            out_flat[idx] = s_proc
            _tally(info)
            done += 1
            if cb and done % max(1, total // 200) == 0:
                cb(done / total)

    # Process-based parallelism is only worth it for large maps AND is unsafe in
    # a frozen/standalone build: with the "spawn" start method each worker
    # re-launches sys.executable (here the BioRaman app itself), which commonly
    # hangs and leaves the loader stuck on "Loading…" forever. So: run serially
    # when frozen or for small jobs, and fall back to serial if the pool errors.
    use_pool = (not getattr(sys, "frozen", False)) and total >= 64 and n_workers > 1
    if use_pool:
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_process_one, item): item[0]
                           for item in flat}
                for fut in as_completed(futures):
                    idx, s_proc, info = fut.result()
                    out_flat[idx] = s_proc
                    _tally(info)
                    done += 1
                    if cb and done % max(1, total // 200) == 0:
                        cb(done / total)
        except Exception as exc:
            # Pool unavailable (frozen app, sandbox, pickling/spawn failure …) —
            # reset partial state and reprocess everything serially.
            print(f"[BioRaman] Parallel preprocessing unavailable "
                  f"({exc!r}); falling back to serial.", flush=True)
            out_flat = [None] * total
            cosmic_n = 0; cosmic_pts = 0; baseline_fail = 0
            done     = 0
            ran_pool = False
            _run_serial()
        else:
            ran_pool = True
    else:
        ran_pool = False
        _run_serial()
    if cb: cb(1.0)

    out = np.empty((Y, X, W), dtype=float)
    for i, (y, x) in enumerate(np.ndindex(Y, X)):
        out[y, x] = out_flat[i]

    elapsed = time.perf_counter() - t0
    if params.baseline_method == "none":
        baseline_status = "SKIPPED"
    elif baseline_fail == 0:
        baseline_status = params.baseline_method.upper()
    elif baseline_fail >= total:
        baseline_status = f"{params.baseline_method.upper()} — FAILED on all spectra (left uncorrected)"
    else:
        baseline_status = (f"{params.baseline_method.upper()} — "
                           f"FAILED on {baseline_fail}/{total} spectra (those left uncorrected)")
    report = {
        "total_spectra":    total,
        "map_shape":        f"{X} × {Y}",
        "spectral_points":  W,
        "cosmic_removed":   cosmic_n,
        "cosmic_points":    cosmic_pts,
        "dark_subtraction": "yes" if params.dark_removal else "no",
        "baseline_method":  baseline_status,
        "baseline_failures": baseline_fail,
        "baseline_lam":     f"{params.asls_lam:.0e}",
        "baseline_p":       f"{params.asls_p}",
        "smoothing":        (f"Savitzky-Golay  window={params.sg_window}"
                             f"  poly={params.sg_poly}")
                            if params.smoothing else "SKIPPED",
        "normalisation":    params.normalisation.upper(),
        "workers":          n_workers if ran_pool else 1,
        "elapsed_s":        f"{elapsed:.1f}",
    }
    return out, report


# ─────────────────────────────────────────────────────────────────────────────
# RECIPE / EXPORT / BATCH HELPERS  (shared by the GUI and the headless CLI)
# ─────────────────────────────────────────────────────────────────────────────
def recipe_to_dict(params):
    """Serialise a PreprocessParams dataclass to a plain dict."""
    from dataclasses import asdict
    return asdict(params)


def recipe_from_dict(d):
    """Build a PreprocessParams from a dict, ignoring unknown keys."""
    p = PreprocessParams()
    for k, v in (d or {}).items():
        if hasattr(p, k):
            setattr(p, k, v)
    return p


def save_recipe_file(path, params):
    import json
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"bioraman_recipe_version": 1,
                   "params": recipe_to_dict(params)}, fh, indent=2)


def load_recipe_file(path):
    return load_pipeline_file(path)


# ── v4: full pipeline export / replay (JSON or YAML) ──────────────────────────
def _has_yaml():
    try:
        import yaml  # noqa: F401
        return True
    except Exception:
        return False


def pipeline_to_dict(params, source="", output_format=".npz"):
    """Build a self-describing, reproducible pipeline document: app version,
    creation time, data source, chosen output format and the full
    preprocessing recipe. Re-runnable with :func:`run_pipeline`."""
    import datetime
    return {
        "bioraman_pipeline_version": 1,
        "app_version": __version__,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": str(source or ""),
        "output_format": output_format,
        "preprocessing": recipe_to_dict(params),
    }


def save_pipeline_file(path, params, source="", output_format=".npz"):
    """Serialise the pipeline to .json or .yaml/.yml (chosen by extension)."""
    doc = pipeline_to_dict(params, source=source, output_format=output_format)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        if not _has_yaml():
            raise RuntimeError("YAML export needs PyYAML (pip install pyyaml).")
        import yaml
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
    else:
        import json
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)


def load_pipeline_file(path):
    """Load a pipeline/recipe (.json/.yaml/.yml) and return PreprocessParams.
    Accepts new pipeline docs, legacy recipe files, and bare param dicts."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if ext in (".yaml", ".yml"):
        if not _has_yaml():
            raise RuntimeError("Reading YAML needs PyYAML (pip install pyyaml).")
        import yaml
        d = yaml.safe_load(text) or {}
    else:
        import json
        d = json.loads(text)
    if isinstance(d, dict):
        section = d.get("preprocessing") or d.get("params") or d
    else:
        section = {}
    return recipe_from_dict(section)


def run_pipeline(pipeline_path, in_dir, out_dir, log=print):
    """Replay an exported pipeline file over every supported map in a folder."""
    import os as _os
    params = load_pipeline_file(pipeline_path)
    out_fmt = ".npz"
    try:
        import json
        ext = _os.path.splitext(pipeline_path)[1].lower()
        with open(pipeline_path, "r", encoding="utf-8") as fh:
            doc = (__import__("yaml").safe_load(fh) if ext in (".yaml", ".yml")
                   else json.load(fh))
        out_fmt = (doc or {}).get("output_format", ".npz")
    except Exception:
        pass
    return run_batch(in_dir, out_dir, params, out_format=out_fmt, log=log)


def write_cube(path, cube, x, report=None, source=""):
    """Write a preprocessed spectral cube (Y×X×W) + wavenumber axis to disk.

    Format is chosen from the file extension: .npz, .h5/.hdf5, .mat,
    or ASCII (.csv/.txt/.dpt). Maps are written in the Renishaw long format
    (#X #Y #Wave #Intensity) so they round-trip through the built-in reader.
    """
    cube = np.asarray(cube, dtype=float)
    x    = np.asarray(x, dtype=float)
    Y, X, W = cube.shape
    ext  = os.path.splitext(path)[1].lower()
    report = report or {}

    if ext in (".parquet", ".feather", ".arrow"):
        write_cube_table(path, cube, x, source=source)

    elif ext == ".npz":
        np.savez_compressed(
            path, xdata=x, spectra=cube,
            report=np.array(list(report.items()), dtype=object),
            source=str(source or ""))

    elif ext in (".h5", ".hdf5"):
        try:
            import h5py
        except Exception as exc:
            raise RuntimeError("Saving HDF5 needs h5py (pip install h5py).") from exc
        with h5py.File(path, "w") as hf:
            hf.create_dataset("wavenumber", data=x)
            hf.create_dataset("spectra", data=cube, compression="gzip")
            hf.attrs["source"] = str(source or "")
            hf.attrs["map_shape"] = [Y, X]
            for k, v in report.items():
                try: hf.attrs[f"report/{k}"] = str(v)
                except Exception: pass

    elif ext == ".mat":
        try:
            from scipy.io import savemat
        except Exception as exc:
            raise RuntimeError("Saving .mat needs SciPy (pip install scipy).") from exc
        savemat(path, {"wavenumber": x, "spectra": cube,
                       "map_shape": np.array([Y, X], dtype=int),
                       "source": str(source or "")})

    else:  # ASCII
        delim = "," if ext == ".csv" else "\t"
        if Y == 1 and X == 1:
            np.savetxt(path, np.column_stack((x, cube[0, 0])),
                       fmt="%.6f", delimiter=delim,
                       header=f"Raman_Shift(cm-1){delim}Intensity(a.u.)",
                       comments="")
        else:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(delim.join(["#X", "#Y", "#Wave", "#Intensity"]) + "\n")
                for yy in range(Y):
                    for xx in range(X):
                        for wn, iv in zip(x, cube[yy, xx]):
                            fh.write(delim.join((f"{xx}", f"{yy}",
                                     f"{wn:.6f}", f"{iv:.6f}")) + "\n")


def process_file(in_path, params, cb=None):
    """Open any supported Raman file and apply the preprocessing recipe.
    Returns (xdata, processed_cube, report)."""
    r = _open_raman_any(in_path)
    proc, report = preprocess_map(r.spectra, params, cb)
    return r.xdata, proc, report


def run_batch(in_dir, out_dir, params, out_format=".npz",
              recursive=False, log=print):
    """Apply one recipe to every supported file in a folder.

    Writes one processed file per input plus a ``batch_summary.csv``.
    Returns (n_ok, n_fail).
    """
    exts = (".wdf", ".wip", ".spc", ".jdx", ".txt", ".csv", ".dpt", ".dat",
            ".asc", ".parquet", ".feather", ".arrow")
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = (sorted(in_dir.rglob("*")) if recursive else sorted(in_dir.iterdir()))
    files = [f for f in files if f.is_file() and f.suffix.lower() in exts]

    summary, n_ok, n_fail = [], 0, 0
    for i, f in enumerate(files, 1):
        try:
            log(f"[{i}/{len(files)}] {f.name}")
            x, cube, report = process_file(str(f), params)
            out_path = out_dir / (f.stem + "_processed" + out_format)
            write_cube(str(out_path), cube, x, report, source=str(f))
            Y, X, W = cube.shape
            summary.append({"file": f.name, "status": "ok",
                            "X": X, "Y": Y, "points": W,
                            "cosmic_removed": report.get("cosmic_removed", ""),
                            "output": out_path.name})
            n_ok += 1
        except Exception as exc:
            log(f"    FAILED: {exc}")
            summary.append({"file": f.name, "status": f"failed: {exc}"})
            n_fail += 1

    # write summary CSV
    try:
        import csv
        cols = ["file", "status", "X", "Y", "points", "cosmic_removed", "output"]
        with open(out_dir / "batch_summary.csv", "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for row in summary:
                w.writerow({c: row.get(c, "") for c in cols})
    except Exception:
        pass
    log(f"Done: {n_ok} ok, {n_fail} failed  →  {out_dir}")
    return n_ok, n_fail


# ─────────────────────────────────────────────────────────────────────────────
# v4 — PARQUET / FEATHER CUBE I/O  (open, columnar, language-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
class _SimpleCube:
    """Lightweight reader-like object so loaded cubes flow through the same
    pipeline as instrument readers (needs only ``.spectra`` and ``.xdata``)."""
    def __init__(self, spectra, xdata, px_um=None, source=""):
        self.spectra = np.asarray(spectra, dtype=float)
        self.xdata   = np.asarray(xdata, dtype=float)
        if px_um is not None:
            self._px_um = px_um
        self.source = source


def write_cube_table(path, cube, x, px_um=None, source=""):
    """Write a Y×X×W cube to Apache Parquet (.parquet) or Feather
    (.feather/.arrow). One row per pixel: columns ``y``, ``x`` and an
    ``intensity`` list column; the wavenumber axis, map shape, pixel size and
    source are stored in the file's schema metadata so the cube round-trips
    losslessly and is also readable from pandas/R/Julia."""
    try:
        import pyarrow as pa
    except Exception as exc:
        raise RuntimeError(
            "Parquet/Feather I/O needs pyarrow (pip install pyarrow).") from exc
    import json
    cube = np.asarray(cube, dtype=np.float32)
    x    = np.asarray(x, dtype=float)
    Y, X, W = cube.shape
    flat = cube.reshape(Y * X, W)
    ys, xs = np.divmod(np.arange(Y * X), X)
    table = pa.table({
        "y": pa.array(ys.astype(np.int32)),
        "x": pa.array(xs.astype(np.int32)),
        "intensity": pa.array(list(flat)),
    })
    meta = {
        b"format":     b"bioraman-cube-v1",
        b"wavenumber": json.dumps([float(v) for v in x]).encode("utf-8"),
        b"map_shape":  json.dumps([int(Y), int(X)]).encode("utf-8"),
        b"px_um":      json.dumps(px_um).encode("utf-8"),
        b"source":     str(source or "").encode("utf-8"),
    }
    table = table.replace_schema_metadata(meta)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        import pyarrow.parquet as pq
        pq.write_table(table, path, compression="zstd")
    else:  # .feather / .arrow
        import pyarrow.feather as feather
        feather.write_feather(table, path, compression="zstd")


def read_cube_table(path):
    """Read a Parquet/Feather cube written by :func:`write_cube_table`
    (or any table exposing y/x/intensity + wavenumber metadata)."""
    try:
        import pyarrow as pa  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Parquet/Feather I/O needs pyarrow (pip install pyarrow).") from exc
    import json
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(path)
    else:
        import pyarrow.feather as feather
        table = feather.read_table(path)
    meta = table.schema.metadata or {}
    if b"wavenumber" not in meta:
        raise RuntimeError(
            "This table has no bioraman cube metadata (wavenumber axis). "
            "It may not be a hyperspectral cube exported by BioRaman.")
    x = np.asarray(json.loads(meta[b"wavenumber"].decode("utf-8")), dtype=float)
    Y, X = json.loads(meta[b"map_shape"].decode("utf-8"))
    px_um = None
    if meta.get(b"px_um"):
        try: px_um = json.loads(meta[b"px_um"].decode("utf-8"))
        except Exception: px_um = None
    src = meta.get(b"source", b"").decode("utf-8")
    cols = {n: i for i, n in enumerate(table.column_names)}
    flat = np.vstack([np.asarray(r, dtype=float)
                      for r in table.column("intensity").to_pylist()])
    # Honour explicit y/x ordering if present (robust to row reordering)
    if "y" in cols and "x" in cols:
        yy = np.asarray(table.column("y").to_pylist(), dtype=int)
        xx = np.asarray(table.column("x").to_pylist(), dtype=int)
        cube = np.zeros((Y, X, x.size), dtype=float)
        cube[yy, xx] = flat
    else:
        cube = flat.reshape(Y, X, x.size)
    return _SimpleCube(cube, x, px_um=px_um, source=src)


# ─────────────────────────────────────────────────────────────────────────────
# v4 — HORIBA LabSpec ASCII / text map reader  (open export path)
# Horiba's binary .ngc is proprietary and undocumented; LabSpec can export an
# open ASCII map (and HDF5 from LabSpec 6.3+). This parses the standard ASCII
# map: first row = Raman shifts (preceded by blank coordinate cells), each
# following row = X-pos, Y-pos, then the intensities for that pixel.
# ─────────────────────────────────────────────────────────────────────────────
def _read_horiba_map_text(path):
    import io
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    # Detect delimiter (tab is the Horiba default; fall back to comma/space)
    sample = "\n".join(raw.splitlines()[:5])
    if "\t" in sample:
        delim = "\t"
    elif ";" in sample:
        delim = ";"
    elif "," in sample:
        delim = ","
    else:
        delim = None  # whitespace
    def _split(line):
        return line.split(delim) if delim else line.split()
    lines = [ln for ln in raw.splitlines() if ln.strip()
             and not ln.lstrip().startswith(("#", "//"))]
    if len(lines) < 2:
        raise RuntimeError("Not a Horiba ASCII map (need a header + data rows).")

    def _to_float(tok):
        try: return float(tok.replace(",", ".")) if delim != "," else float(tok)
        except Exception: return np.nan

    header = [_to_float(t) for t in _split(lines[0])]
    # Leading non-finite / empty cells are the X/Y coordinate placeholders.
    n_lead = 0
    for v in header:
        if not np.isfinite(v):
            n_lead += 1
        else:
            break
    wn = np.asarray(header[n_lead:], dtype=float)
    W = wn.size
    if W < 3:
        raise RuntimeError("Horiba header has too few wavenumber columns.")

    xs, ys, specs = [], [], []
    for ln in lines[1:]:
        toks = _split(ln)
        vals = [_to_float(t) for t in toks]
        if len(vals) < n_lead + W:
            continue
        if n_lead >= 2:
            xs.append(vals[0]); ys.append(vals[1])
        elif n_lead == 1:
            xs.append(vals[0]); ys.append(0.0)
        else:
            xs.append(len(xs)); ys.append(0.0)
        specs.append(vals[len(vals) - W:])
    if not specs:
        raise RuntimeError("No data rows parsed from the Horiba map.")
    spectra_flat = np.asarray(specs, dtype=float)

    # Reconstruct the 2-D grid from the unique X/Y stage positions.
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    ux = np.unique(xs); uy = np.unique(ys)
    if ux.size * uy.size == spectra_flat.shape[0] and ux.size > 1 and uy.size > 1:
        xi = {v: i for i, v in enumerate(ux)}
        yi = {v: i for i, v in enumerate(uy)}
        cube = np.zeros((uy.size, ux.size, W), dtype=float)
        for k in range(spectra_flat.shape[0]):
            cube[yi[ys[k]], xi[xs[k]]] = spectra_flat[k]
        px_um = None
        if ux.size > 1:
            step = np.median(np.diff(ux))
            if np.isfinite(step) and step > 0:
                px_um = float(abs(step))
    else:
        # Could not infer a 2-D grid — treat as a line scan (1 × N × W).
        cube = spectra_flat[None, :, :]
        px_um = None
    # Ensure ascending wavenumber axis
    if wn.size > 1 and wn[0] > wn[-1]:
        wn = wn[::-1]; cube = cube[:, :, ::-1]
    return _SimpleCube(cube, wn, px_um=px_um, source=str(path))


# ─────────────────────────────────────────────────────────────────────────────
# MAP DISPLAY QUALITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _smooth_nan(a, sigma):
    """NaN-aware Gaussian smoothing for nicer map display (background kept NaN)."""
    a = np.asarray(a, dtype=float)
    if sigma <= 0:
        return a
    m = np.isfinite(a)
    if not m.any():
        return a
    a0 = np.where(m, a, 0.0)
    num = gaussian_filter(a0, sigma)
    den = gaussian_filter(m.astype(float), sigma)
    out = np.where(den > 0, num / den, np.nan)
    out[~m] = np.nan
    return out


def _add_scale_bar(ax, shape, px_um):
    """Draw a µm scale bar (bottom-left) on a map axis."""
    if not px_um or px_um <= 0:
        return
    import matplotlib.patheffects as pe
    ny, nx = shape[:2]
    target = nx * px_um * 0.25                      # ~quarter of the width
    nice = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    L = min(nice, key=lambda v: abs(v - target))
    Lpx = L / px_um
    if Lpx >= nx * 0.9:
        return
    x0 = nx * 0.06; y0 = ny * 0.92
    ln = ax.plot([x0, x0 + Lpx], [y0, y0], color="white", lw=3,
                 solid_capstyle="butt")[0]
    ln.set_path_effects([pe.withStroke(linewidth=5, foreground="black")])
    lbl = f"{L:g} µm"
    tx = ax.text(x0 + Lpx / 2, y0 - ny * 0.02, lbl, color="white",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    tx.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])


def show_map(ax, fig, arr, cmap="turbo", sigma=0.8, robust=True,
             vmin=None, vmax=None, title=None, title_color=None,
             colorbar=True, equal=True, px_um=None, interpolation="bilinear"):
    """Render a 2-D map the way instrument software does: light NaN-aware
    Gaussian smoothing and bilinear interpolation for a smooth, presentation-
    quality view, robust percentile contrast, an equal aspect ratio and an
    optional µm scale bar. Returns the AxesImage.

    For a faithful, pixel-exact view (each measured pixel shown as a discrete
    cell) set the window's *Display σ* control to 0 and/or pass
    ``interpolation="nearest"``."""
    a = _smooth_nan(arr, sigma)
    finite = a[np.isfinite(a)]
    if vmin is None or vmax is None:
        if robust and finite.size:
            lo, hi = np.percentile(finite, [2, 98])
            if hi <= lo:
                lo, hi = float(finite.min()), float(finite.max())
        elif finite.size:
            lo, hi = float(finite.min()), float(finite.max())
        else:
            lo, hi = 0.0, 1.0
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("#ffffff")
    im = ax.imshow(np.ma.masked_invalid(a), origin="upper",
                   aspect=("equal" if equal else "auto"), cmap=cm,
                   interpolation=interpolation, vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title, fontsize=9, fontweight="semibold",
                     color=(title_color or "#1b2333"))
    ax.set_xticks([]); ax.set_yticks([])
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     shrink=0.85).ax.tick_params(labelsize=7)
    _add_scale_bar(ax, a.shape, px_um)
    return im


_LUT_CMAPS = ["turbo", "viridis", "plasma", "inferno", "magma", "cividis",
              "jet", "hot", "coolwarm", "RdBu_r", "gray", "gist_earth",
              "nipy_spectral"]


def _lut_clim(arr, setting):
    """vmin/vmax from a panel LUT setting's low/high percentiles."""
    f = np.asarray(arr)[np.isfinite(arr)]
    if f.size == 0:
        return None, None
    lo, hi = np.percentile(f, [setting["lo"], setting["hi"]])
    if hi <= lo:
        lo, hi = float(f.min()), float(f.max())
    return float(lo), float(hi)


def _lut_for_win(win, key, default_cmap="turbo"):
    d = getattr(win, "_lut", {}).get(key)
    if d:
        return d
    return {"cmap": default_cmap, "lo": 2.0, "hi": 98.0}


def _build_lut_panel(win, parent, ready_fn, n_panels=8):
    """Attach a per-panel LUT control (colour map + contrast percentiles) to a
    map window. Stores vars on *win* and redraws via win._draw()."""
    from functools import partial
    SectionDiv(parent, "PANEL LUT").pack(fill="x")
    r1 = tk.Frame(parent, bg=C["sidebar"]); r1.pack(fill="x", padx=10, pady=2)
    tk.Label(r1, text="Panel #", width=8, anchor="w", bg=C["sidebar"],
             fg=C["text_mid"], font=("Segoe UI", 9)).pack(side="left")
    win._lut_idx = tk.IntVar(value=1)
    ttk.Spinbox(r1, from_=1, to=n_panels, width=4,
                textvariable=win._lut_idx).pack(side="left")
    win._lut_all = tk.BooleanVar(value=False)
    ttk.Checkbutton(r1, variable=win._lut_all, text="all").pack(side="left",
                                                                padx=6)
    r2 = tk.Frame(parent, bg=C["sidebar"]); r2.pack(fill="x", padx=10, pady=2)
    tk.Label(r2, text="Colour map", width=8, anchor="w", bg=C["sidebar"],
             fg=C["text_mid"], font=("Segoe UI", 9)).pack(side="left")
    win._lut_cmap = tk.StringVar(value="turbo")
    ttk.Combobox(r2, textvariable=win._lut_cmap, state="readonly", width=12,
                 values=_LUT_CMAPS).pack(side="left")
    r3 = tk.Frame(parent, bg=C["sidebar"]); r3.pack(fill="x", padx=10, pady=2)
    tk.Label(r3, text="Contrast %", width=8, anchor="w", bg=C["sidebar"],
             fg=C["text_mid"], font=("Segoe UI", 9)).pack(side="left")
    win._lut_lo = tk.DoubleVar(value=2.0); win._lut_hi = tk.DoubleVar(value=98.0)
    ttk.Spinbox(r3, from_=0, to=49, width=5, textvariable=win._lut_lo).pack(
        side="left")
    tk.Label(r3, text="–", bg=C["sidebar"], fg=C["text_dim"]).pack(side="left")
    ttk.Spinbox(r3, from_=51, to=100, width=5, textvariable=win._lut_hi).pack(
        side="left")

    def _apply():
        if not ready_fn():
            return
        s = {"cmap": win._lut_cmap.get(), "lo": float(win._lut_lo.get()),
             "hi": float(win._lut_hi.get())}
        if win._lut_all.get():
            for k in range(n_panels):
                win._lut[k] = dict(s)
        else:
            win._lut[int(win._lut_idx.get()) - 1] = dict(s)
        win._draw()
    ttk.Button(parent, text="🎨 Apply LUT", command=_apply).pack(
        fill="x", padx=10, pady=3)


# ─────────────────────────────────────────────────────────────────────────────
# TESTABLE CORE ANALYSIS  (used by the GUI, the validation harness and tests)
# ─────────────────────────────────────────────────────────────────────────────
def pca_component_suggestions(evr, eigvals, scaled, var_target=0.95):
    """Objective number-of-PC criteria for a scree plot.

    Improper selection of the number of principal components is one of the
    most common, result-invalidating mistakes in Raman chemometrics
    (Hanson 2017; Khristoforova 2022; Vajna 2011). This returns three
    standard, transparent criteria so the user is not left guessing.

    Parameters
    ----------
    evr : 1-D explained-variance-ratio array (length = #components computed).
    eigvals : 1-D eigenvalues (explained_variance_) for the same components.
    scaled : whether the data were autoscaled (correlation-matrix PCA).
    var_target : cumulative-variance target (default 0.95).

    Returns a dict with keys ``cumvar``, ``kaiser`` and ``broken_stick``
    (each an int #PCs, clamped to ≥1), plus ``var_target``.
    """
    evr = np.asarray(evr, dtype=float)
    eig = np.asarray(eigvals, dtype=float)
    p   = len(evr)
    out = {"var_target": var_target}

    # 1) Cumulative variance — smallest k reaching the target.
    cum = np.cumsum(evr)
    k_cum = int(np.searchsorted(cum, var_target) + 1)
    out["cumvar"] = int(min(max(k_cum, 1), p))

    # 2) Kaiser rule. On autoscaled (correlation) data: eigenvalue > 1.
    #    On mean-centred (covariance) data the equivalent is the
    #    average-eigenvalue rule: eigenvalue > mean(eigenvalues).
    thresh = 1.0 if scaled else float(np.mean(eig)) if eig.size else 0.0
    out["kaiser"] = int(min(max(int(np.sum(eig > thresh)), 1), p))

    # 3) Broken-stick: keep PCs whose variance share exceeds the expected
    #    share under a random partition  b_k = (1/p) Σ_{i=k..p} 1/i.
    idx = np.arange(1, p + 1)
    bstick = np.array([np.sum(1.0 / idx[k:]) / p for k in range(p)])
    keep = evr > bstick
    out["broken_stick"] = int(max(1, np.argmin(keep) if (not keep.all() and keep.any())
                                  else (np.sum(keep) if keep.any() else 1)))
    return out


def hotelling_t2_radius(n, conf=0.95):
    """Scale factor for a Hotelling-T² confidence ellipse of 2-D scores.

    A point lies inside the ellipse if its Mahalanobis distance ≤ this radius.
    Uses the finite-sample F-distribution form for an individual observation;
    falls back to the χ² form when n is too small or SciPy is unavailable.
    Multiply each covariance eigenvalue's sqrt by this radius for the semi-axes.
    """
    try:
        from scipy.stats import f as _f, chi2 as _chi2
        if n > 3:
            return float(np.sqrt(2 * (n - 1) / (n - 2) * _f.ppf(conf, 2, n - 2)))
        return float(np.sqrt(_chi2.ppf(conf, df=2)))
    except Exception:
        # χ²(0.95, 2) ≈ 5.991 → radius ≈ 2.448; safe constant fallback.
        return 2.4477


def prep_spectra(M, preprocess="Spectrum", normalise="None"):
    """Apply derivative + normalisation to a spectra matrix (last axis = W)."""
    M = np.asarray(M, dtype=float)
    if preprocess == "1st derivative":
        M = np.gradient(M, axis=-1)
    elif preprocess == "2nd derivative":
        M = np.gradient(np.gradient(M, axis=-1), axis=-1)
    if normalise == "Vector":
        n = np.linalg.norm(M, axis=-1, keepdims=True)
        M = np.divide(M, n, out=np.zeros_like(M), where=n > 0)
    elif normalise.startswith("Mean"):
        mu = M.mean(axis=-1, keepdims=True); sd = M.std(axis=-1, keepdims=True)
        M = np.divide(M - mu, sd, out=np.zeros_like(M), where=sd > 0)
    return M


def component_fit(flat, refs, method="NNLS", preprocess="Spectrum",
                  normalise="None", background_order=0):
    """Component analysis (DCLS / NNLS) of spectra against reference spectra.

    Parameters
    ----------
    flat : (N, W) array of spectra (rows are pixels).
    refs : (K, W) array of reference component spectra on the same axis.
    method : 'NNLS' (non-negative) or 'DCLS' (unconstrained least squares).
    preprocess : 'Spectrum' | '1st derivative' | '2nd derivative'.
    normalise : 'None…' | 'Vector' | 'Mean…' (see prep_spectra).
    background_order : polynomial background order modelled in the fit
        (Spectrum method only; the background coefficients are unconstrained).

    Returns a dict with:
        conc    : (N, K) component abundances (clipped at 0 for ≥0 reporting)
        raw     : (N, K) raw fit coefficients (may be negative for DCLS)
        rel     : (N, K) per-pixel relative concentration in percent
        lof     : (N,)   percentage lack of fit
        overall : (K,)   overall concentration estimate in percent
    """
    flat = np.asarray(flat, dtype=float)
    refs = np.asarray(refs, dtype=float)
    if refs.ndim == 1:
        refs = refs[None, :]
    N, W = flat.shape
    K = refs.shape[0]
    Fq = prep_spectra(flat, preprocess, normalise)
    R = prep_spectra(refs, preprocess, normalise).T          # W × K
    A = R
    if preprocess == "Spectrum" and background_order > 0:
        xn = np.linspace(-1, 1, W)
        P = np.vander(xn, background_order + 1, increasing=True)
        A = np.hstack([R, P])
    M = A.shape[1]
    C = np.zeros((N, K)); recon = np.zeros((N, W))
    if method == "DCLS":
        coef, *_ = np.linalg.lstsq(A, Fq.T, rcond=None)      # M × N
        C = coef[:K].T
        recon = (A @ coef).T
    else:
        from scipy.optimize import nnls
        if M > K:
            from scipy.optimize import lsq_linear
            lb = np.concatenate([np.zeros(K), -np.inf * np.ones(M - K)])
            ub = np.inf * np.ones(M)
            for i in range(N):
                sol = lsq_linear(A, Fq[i], bounds=(lb, ub), method="bvls")
                C[i] = sol.x[:K]; recon[i] = A @ sol.x
        else:
            for i in range(N):
                c, _ = nnls(A, Fq[i]); C[i] = c; recon[i] = A @ c
    resid = Fq - recon
    lof = np.linalg.norm(resid, axis=1) / (np.linalg.norm(Fq, axis=1) + 1e-9) * 100
    Cc = np.clip(C, 0, None)
    tot = Cc.sum(axis=1, keepdims=True)
    rel = np.divide(Cc, tot, out=np.zeros_like(Cc), where=tot > 0) * 100
    overall = Cc.sum(axis=0) / (Cc.sum() + 1e-9) * 100
    return dict(conc=Cc, raw=C, rel=rel, lof=lof, overall=overall)


def particle_stats(image, auto=True, threshold_pct=50.0, remove_edge=True,
                   min_size_pct=1.0, px_um=1.0):
    """Label and measure particles/domains in a 2-D image.

    Returns a dict with: labels (int array), mask (bool), props (list of dicts
    with label/area_px/area_um2/ecd_um/cx/cy), area_pct, n.
    """
    from scipy import ndimage
    img = np.asarray(image, dtype=float)
    finite = np.isfinite(img)
    img = np.where(finite, img, np.nanmin(img[finite]) if finite.any() else 0.0)
    if auto:
        thr = _otsu_threshold(img)
    else:
        lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
        thr = lo + (hi - lo) * threshold_pct / 100.0
    mask = img >= thr
    lbl, n = ndimage.label(mask)
    if remove_edge and n:
        border = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
        for b in border:
            if b:
                lbl[lbl == b] = 0
    counts = np.bincount(lbl.ravel())
    areas = {i: counts[i] for i in range(1, len(counts)) if counts[i] > 0}
    keep = set()
    if areas:
        biggest = max(areas.values())
        keep = {i for i, a in areas.items()
                if a >= biggest * min_size_pct / 100.0}
    coms = ndimage.center_of_mass(mask, lbl, sorted(keep)) if keep else []
    props = []
    for idx, lab in enumerate(sorted(keep)):
        a_px = int(areas[lab]); a_um = a_px * px_um * px_um
        ecd = 2.0 * np.sqrt(a_um / np.pi)
        cy, cx = coms[idx] if len(coms) else (0, 0)
        props.append(dict(label=lab, area_px=a_px, area_um2=a_um,
                          ecd_um=ecd, cx=cx, cy=cy))
    area_pct = (np.isin(lbl, list(keep)).sum() / lbl.size * 100
                if lbl.size else 0.0)
    return dict(labels=lbl, mask=np.isin(lbl, list(keep)), props=props,
                area_pct=area_pct, n=len(props))


def band_range_status(xdata, lo, hi):
    """Classify a band window [lo, hi] against the acquired wavenumber axis.

    Returns (status, message) where status is one of:
      'ok'       — the whole window lies inside the measured range.
      'partial'  — the window overlaps the data but extends past one/both ends
                   (results are silently truncated to the data edge).
      'outside'  — the window lies entirely beyond the measured range; there is
                   NO data, so any intensity reported there is an artifact.
    This is what catches the classic mistake of integrating e.g. an OH-stretch
    band (~3100 cm⁻¹) when acquisition stopped at ~2500 cm⁻¹.
    """
    if xdata is None or np.size(xdata) == 0:
        return ("outside", "No spectral axis loaded.")
    lo, hi = (lo, hi) if lo <= hi else (hi, lo)
    xmin = float(np.nanmin(xdata)); xmax = float(np.nanmax(xdata))
    rng = f"data range {xmin:.0f}–{xmax:.0f} cm⁻¹"
    if hi < xmin or lo > xmax:
        return ("outside",
                f"Band {lo:.0f}–{hi:.0f} cm⁻¹ is OUTSIDE the {rng}. "
                f"No spectra were collected here — any signal shown is an "
                f"artifact, not a measurement.")
    if lo < xmin or hi > xmax:
        return ("partial",
                f"Band {lo:.0f}–{hi:.0f} cm⁻¹ extends past the {rng}; it is "
                f"truncated to the measured edge — interpret with caution.")
    return ("ok", f"Band {lo:.0f}–{hi:.0f} cm⁻¹ within {rng}.")


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


def make_scroll_sidebar(parent, width=270):
    """Create a fixed-width, vertically scrollable sidebar so long control
    stacks never clip off on short windows. Returns the inner frame to pack
    controls into (use exactly like the old ``left`` frame)."""
    outer = tk.Frame(parent, bg=C["sidebar"], width=width)
    outer.pack(side="left", fill="y"); outer.pack_propagate(False)
    cv = tk.Canvas(outer, bg=C["sidebar"], highlightthickness=0)
    sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
    cv.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    cv.pack(side="left", fill="both", expand=True)
    inner = tk.Frame(cv, bg=C["sidebar"])
    wid = cv.create_window((0, 0), window=inner, anchor="nw")
    cv.bind("<Configure>", lambda e: cv.itemconfig(wid, width=e.width))
    inner.bind("<Configure>",
               lambda e: cv.configure(scrollregion=cv.bbox("all")))
    # wheel scrolls only while the pointer is over this sidebar
    cv.bind("<Enter>", lambda e: cv.bind_all(
        "<MouseWheel>",
        lambda ev: cv.yview_scroll(int(-1 * (ev.delta / 120)), "units")))
    cv.bind("<Leave>", lambda e: cv.unbind_all("<MouseWheel>"))
    return inner


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
# ─────────────────────────────────────────────────────────────────────────────
# PANEL SAVE MIXIN — tick a checkbox on each plot panel, save only the ticked
# ones. Reused by every analysis window (PCA, Clustering, N-FINDR, …).
# ─────────────────────────────────────────────────────────────────────────────
class PanelSaveMixin:
    """Overlay a small checkbox at the top-left of every content panel of an
    embedded matplotlib canvas, and save only the ticked panels as standalone
    images. A window enables it with one call after creating its canvas:

        self._enable_panel_ticks(canvas, basename="PCA")

    Detection is automatic via the canvas 'draw_event', so individual windows
    do not need to register their panels manually."""

    def _enable_panel_ticks(self, canvas, basename="panel"):
        self._tick_canvas = canvas
        self._tick_fig = canvas.figure
        self._panel_basename = basename
        self._panel_vars = getattr(self, "_panel_vars", {})
        self._panel_checks = []
        w = canvas.get_tk_widget()
        w.bind("<Configure>", lambda _e: self._place_panel_ticks())
        try:
            canvas.mpl_connect("draw_event",
                               lambda _e: self._place_panel_ticks())
        except Exception:
            pass
        try:
            ttk.Button(w.master, text="☑ Save Ticked Panels…",
                       command=self._save_ticked_panels).pack(
                           side="bottom", fill="x", padx=6, pady=4)
        except Exception:
            pass

    def _content_axes(self):
        fig = getattr(self, "_tick_fig", None)
        if fig is None:
            return []
        # Skip colorbar axes (matplotlib labels them '<colorbar>').
        return [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]

    def _place_panel_ticks(self):
        for cb in getattr(self, "_panel_checks", []):
            try:
                cb.destroy()
            except Exception:
                pass
        self._panel_checks = []
        axes = self._content_axes()
        if not axes:
            return
        w = self._tick_canvas.get_tk_widget()
        fw, fh = w.winfo_width(), w.winfo_height()
        if fw <= 1 or fh <= 1:
            return
        seen = {}
        for idx, ax in enumerate(axes):
            tag = (ax.get_title() or "").strip() or f"panel{idx+1}"
            n = seen.get(tag, 0) + 1
            seen[tag] = n
            if n > 1:
                tag = f"{tag} ({n})"
            ax._save_tag = tag
            var = self._panel_vars.setdefault(tag, tk.BooleanVar(value=True))
            pos = ax.get_position()
            x = int(pos.x0 * fw) + 2
            y = int((1.0 - pos.y1) * fh) + 2
            cb = tk.Checkbutton(w, variable=var, bg="white",
                                activebackground="white", bd=0,
                                highlightthickness=0, padx=0, pady=0)
            cb.place(x=x, y=y)
            self._panel_checks.append(cb)

    def _save_ticked_panels(self):
        fig = getattr(self, "_tick_fig", None)
        if fig is None or not self._content_axes():
            messagebox.showwarning("Nothing to save",
                                   "Generate a plot first.", parent=self)
            return
        self._place_panel_ticks()          # ensure tags are current
        chosen = [ax for ax in self._content_axes()
                  if getattr(ax, "_save_tag", None)
                  and self._panel_vars.get(ax._save_tag)
                  and self._panel_vars[ax._save_tag].get()]
        if not chosen:
            messagebox.showinfo(
                "Nothing selected",
                "Tick the box at the top-left of at least one panel.",
                parent=self)
            return
        out_dir = filedialog.askdirectory(
            title="Choose a folder for the ticked panels", parent=self)
        if not out_dir:
            return
        save_pdf = messagebox.askyesno(
            "Vector copy?",
            "Also save a vector PDF of each panel?\n\n"
            "Yes → PNG (600 dpi) + PDF      No → PNG only", parent=self)

        import re
        from matplotlib.transforms import Bbox
        try:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
        except Exception:
            renderer = None
        base = getattr(self, "_panel_basename", None) or "panel"
        exported = []
        for ax in chosen:
            tag = getattr(ax, "_save_tag", "panel")
            safe = re.sub(r"[^A-Za-z0-9._-]+", "_", tag).strip("_") or "panel"
            try:
                boxes = [ax.get_tightbbox(renderer)]
                for im in ax.images:
                    cb = getattr(im, "colorbar", None)
                    if cb is not None:
                        boxes.append(cb.ax.get_tightbbox(renderer))
                for coll in ax.collections:
                    cb = getattr(coll, "colorbar", None)
                    if cb is not None:
                        boxes.append(cb.ax.get_tightbbox(renderer))
                bb = Bbox.union([b for b in boxes if b is not None])
                ext = bb.expanded(1.06, 1.10).transformed(
                    fig.dpi_scale_trans.inverted())
                png = Path(out_dir) / f"{base}_{safe}.png"
                fig.savefig(png, dpi=600, bbox_inches=ext, facecolor="white")
                exported.append(png.name)
                if save_pdf:
                    pdf = Path(out_dir) / f"{base}_{safe}.pdf"
                    fig.savefig(pdf, bbox_inches=ext, facecolor="white")
                    exported.append(pdf.name)
            except Exception as ex:
                messagebox.showerror("Export error", f"{tag}: {ex}",
                                     parent=self)
        if exported:
            messagebox.showinfo(
                "Panels saved",
                f"Saved {len(exported)} files to:\n{out_dir}", parent=self)


def attach_panel_saver(win, canvas, basename="panel"):
    """Give an arbitrary Toplevel the PanelSaveMixin behaviour (tick boxes on
    each panel + 'Save Ticked Panels' button) without it subclassing the mixin.
    Used for figures built inside plain functions rather than mixin windows."""
    import types
    for name in ("_enable_panel_ticks", "_content_axes",
                 "_place_panel_ticks", "_save_ticked_panels"):
        setattr(win, name, types.MethodType(getattr(PanelSaveMixin, name), win))
    win._enable_panel_ticks(canvas, basename=basename)


# PCA WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class PCAWindow(PanelSaveMixin, tk.Toplevel):
    """
    Standalone PCA analysis window.
    Load multiple WDF or XLSX map files → preprocess → PCA → plots.
    """
    COLORS = ["#2563eb","#ef4444","#10b981","#f59e0b","#7c3aed",
              "#06b6d4","#ec4899","#84cc16","#f97316","#6366f1"]

    def __init__(self, parent, pp_params):
        super().__init__(parent)
        self.title("PCA Analysis — Multi-File")
        self.geometry("1650x980")
        self.configure(bg=C["bg"])
        self.pp_params = pp_params

        self._files   = []   # list of {"path":..., "label":..., "data":None}
        self._results = None
        self._seed    = 42   # fixed RNG seed for reproducibility

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

        # Left panel: file list + controls (scrollable so nothing clips off)
        left_outer = tk.Frame(self, bg=C["sidebar"], width=320)
        left_outer.pack(side="left", fill="y")
        left_outer.pack_propagate(False)

        _lcanvas = tk.Canvas(left_outer, bg=C["sidebar"],
                             highlightthickness=0, width=320)
        _lscroll = ttk.Scrollbar(left_outer, orient="vertical",
                                 command=_lcanvas.yview)
        _lcanvas.configure(yscrollcommand=_lscroll.set)
        _lscroll.pack(side="right", fill="y")
        _lcanvas.pack(side="left", fill="both", expand=True)

        # All controls go into this inner frame (keeps every .pack(in left) below)
        left = tk.Frame(_lcanvas, bg=C["sidebar"])
        _lcanvas.create_window((0, 0), window=left, anchor="nw", width=302)
        left.bind("<Configure>",
                  lambda e: _lcanvas.configure(scrollregion=_lcanvas.bbox("all")))

        # Mouse-wheel scrolling while the pointer is over the sidebar
        def _wheel(e):
            delta = -1 if getattr(e, "num", None) == 4 else (
                1 if getattr(e, "num", None) == 5 else int(-e.delta / 120))
            _lcanvas.yview_scroll(delta, "units")

        def _bind_wheel(_):
            _lcanvas.bind_all("<MouseWheel>", _wheel)
            _lcanvas.bind_all("<Button-4>", _wheel)
            _lcanvas.bind_all("<Button-5>", _wheel)

        def _unbind_wheel(_):
            _lcanvas.unbind_all("<MouseWheel>")
            _lcanvas.unbind_all("<Button-4>")
            _lcanvas.unbind_all("<Button-5>")

        left_outer.bind("<Enter>", _bind_wheel)
        left_outer.bind("<Leave>", _unbind_wheel)

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
        ttk.Button(btns, text="+ Add files",  style="P.TButton",
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

        self._outlier_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opts, text="Remove outliers (Hotelling T²)",
                       variable=self._outlier_var,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10)).grid(
                           row=4, column=0, columnspan=2, sticky="w", pady=4)

        # FlyHash sparse-code outlier scan (fruit-fly novelty detection).
        self._flyhash_out_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opts, text="🪰 Exclude FlyHash outliers",
                       variable=self._flyhash_out_var,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10)).grid(
                           row=5, column=0, sticky="w", pady=2)
        self._flyhash_pct = tk.DoubleVar(value=2.0)
        ttk.Spinbox(opts, from_=0.5, to=20.0, increment=0.5,
                    textvariable=self._flyhash_pct, width=6).grid(
                        row=5, column=1, padx=8, pady=2)
        tk.Label(opts, text="(% most-novel spectra flagged & dropped)",
                 bg=C["sidebar"], fg=C["text_dim"],
                 font=("Segoe UI", 8)).grid(row=6, column=0, columnspan=2,
                                            sticky="w")

        self._scale_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opts, text="Standardise features",
                       variable=self._scale_var,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 10)).grid(
                           row=7, column=0, columnspan=2, sticky="w", pady=2)

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
        ttk.Button(left, text="↓  Save Panels (publication)", style="N.TButton",
                   command=self._save_panels).pack(fill="x", padx=10, pady=(0,4))
        # ── class-imbalance handling for the classifier (SMOTE family) ──────
        # Inspired by Servert Lerdo de Tejada et al. (2026), Spectrochim. Acta
        # A — resampling is applied INSIDE each CV training fold only, never to
        # the held-out test fold, so reported accuracy stays leakage-free.
        imb = tk.Frame(left, bg=C["sidebar"])
        imb.pack(fill="x", padx=10, pady=(2, 0))
        tk.Label(imb, text="Class imbalance", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left")
        self._clf_imbalance = tk.StringVar(value="None")
        ttk.Combobox(imb, textvariable=self._clf_imbalance, width=16,
                     state="readonly",
                     values=("None", "Random over-sample", "SMOTE",
                             "Borderline-SMOTE", "ADASYN", "SMOTE-ENN",
                             "SMOTE-Tomek")).pack(side="right")
        ttk.Button(left, text="◎  Classify (PLS-DA / LDA)", style="N.TButton",
                   command=self._run_classifier).pack(fill="x", padx=10, pady=(0,4))
        ttk.Button(left, text="≈  Band-ratio export (CSV)", style="N.TButton",
                   command=self._export_band_ratios).pack(fill="x", padx=10, pady=(0,4))
        ttk.Button(left, text="～  Mean spectrum + bands", style="N.TButton",
                   command=self._show_mean_spectrum).pack(fill="x", padx=10, pady=(0,4))

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

        # Scores plot (PC1 vs PC2) appearance controls
        SectionDiv(left, "SCORES PLOT (PC1 vs PC2)").pack(fill="x")
        sp = tk.Frame(left, bg=C["sidebar"])
        sp.pack(fill="x", padx=10, pady=4)

        tk.Label(sp, text="Point size", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=2)
        self._pt_size = tk.DoubleVar(value=18)
        ttk.Spinbox(sp, from_=2, to=120, increment=2, textvariable=self._pt_size,
                    width=6, command=self._redraw_spatial).grid(
                        row=0, column=1, padx=8, pady=2)

        tk.Label(sp, text="Aspect (H/W, 0=auto)", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).grid(
                     row=1, column=0, sticky="w", pady=2)
        self._aspect = tk.DoubleVar(value=0.0)
        ttk.Spinbox(sp, from_=0.0, to=5.0, increment=0.1, textvariable=self._aspect,
                    width=6, command=self._redraw_spatial).grid(
                        row=1, column=1, padx=8, pady=2)
        tk.Label(sp, text="Adjust size/aspect, then it redraws live",
                 bg=C["sidebar"], fg=C["text_dim"],
                 font=("Segoe UI", 9, "italic")).grid(
                     row=2, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # Right: plot area
        right = tk.Frame(self, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)

        self.fig = plt.figure(figsize=(17, 9.6), facecolor="#ffffff")
        # 2×3 grid: A B E / C D (E=spatial, last cell hidden if single file)
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(2, 3, figure=self.fig,
                               hspace=0.50, wspace=0.30,
                               left=0.055, right=0.985,
                               top=0.93, bottom=0.12)
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
        self._enable_panel_ticks(self.canvas, basename="PCA")

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
        # Universal: accepts wdf/spc/jdx/dpt/dat/txt/csv … via raman_io
        if not (HAS_RAMANIO or HAS_WDF):
            messagebox.showerror("Missing library",
                "No reader available.\npip install renishawWiRE  (and spc-spectra "
                "for .spc files)", parent=self)
            return
        paths = filedialog.askopenfilenames(
            title="Add Raman files (any format)",
            filetypes=SUPPORTED_PATTERNS, parent=self)
        for p in paths:
            stem = Path(p).stem
            parsed = parse_pupae_label(stem)
            if parsed:
                diet, temp, group = parsed
            else:
                diet, temp, group = None, None, stem
            ext = Path(p).suffix.lstrip(".").upper() or "RAMAN"
            self._files.append({"path":p, "label":group, "diet":diet,
                                 "temp":temp, "fmt":"raman", "data":None})
            self._listbox.insert("end", f"[{ext}]  {group}")

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
            stem = Path(p).stem
            parsed = parse_pupae_label(stem)
            if parsed:
                diet, temp, group = parsed
            else:
                diet, temp, group = None, None, stem
            self._files.append({"path":p, "label":group, "diet":diet,
                                 "temp":temp, "fmt":"xlsx", "data":None})
            self._listbox.insert("end", f"[XLSX] {group}")

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
        if self.pp_params.baseline_method != "none" and not HAS_PYBL:
            messagebox.showwarning("Baseline skipped",
                "pybaselines is not installed — baseline correction will be "
                "SKIPPED even though it is selected.\n\npip install pybaselines",
                parent=self)
        np.random.seed(self._seed)
        self._status_lbl.config(text="Loading and preprocessing…")
        self._prog["value"] = 0
        self.update_idletasks()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            (all_X, all_labels, waves_common,
             all_diets, all_temps, all_pupae) = self._load_all()
            self.after(0, lambda: self._finish_pca(
                all_X, all_labels, waves_common, all_diets, all_temps, all_pupae))
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
        all_diets, all_temps = [], []
        all_pupae = []   # per-spectrum source-file (pupa) ID for grouped CV
        waves_common = None
        self._spatial_shapes = []   # (label, Y, X) for spatial score map

        for fi, finfo in enumerate(self._files):
            self.after(0, lambda fi=fi: self._status_lbl.config(
                text=f"Loading {Path(finfo['path']).name}…"))
            self.after(0, lambda fi=fi: self._prog.configure(
                value=fi/len(self._files)*50))

            # ── load raw data ─────────────────────────────────────────────
            if finfo["fmt"] in ("wdf", "raman"):
                reader  = (open_raman(finfo["path"]) if HAS_RAMANIO
                           else WDFReader(finfo["path"]))
                raw     = np.asarray(reader.spectra)
                xdata   = np.asarray(reader.xdata)
                if raw.ndim == 3:                         # Y×X×W spatial map
                    Y, X, W = raw.shape
                    self._spatial_shapes.append((finfo["label"], Y, X))
                    raw_flat = raw.reshape(Y * X, W)
                elif raw.ndim == 2:                       # N×W series of spectra
                    raw_flat = raw
                    self._spatial_shapes.append((finfo["label"], None, None))
                else:                                     # single 1-D spectrum
                    raw_flat = raw.reshape(1, -1)
                    self._spatial_shapes.append((finfo["label"], None, None))
                # guard wavenumber axis length / orientation
                if xdata.ndim != 1 or xdata.shape[0] != raw_flat.shape[1]:
                    xdata = np.arange(raw_flat.shape[1], dtype=float)
                if xdata[0] > xdata[-1]:                  # ensure ascending
                    xdata    = xdata[::-1]
                    raw_flat = raw_flat[:, ::-1]
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

            pupa_id = Path(finfo["path"]).stem
            all_X.extend(proc)
            all_labels.extend([finfo["label"]] * len(proc))
            all_diets.extend([finfo.get("diet")] * len(proc))
            all_temps.extend([finfo.get("temp")] * len(proc))
            all_pupae.extend([pupa_id] * len(proc))
            self.after(0, lambda fi=fi: self._prog.configure(
                value=50 + fi/len(self._files)*50))

        return (np.array(all_X), np.array(all_labels), waves_common,
                np.array(all_diets, dtype=object), np.array(all_temps, dtype=object),
                np.array(all_pupae, dtype=object))

    def _finish_pca(self, X, labels, waves, diets=None, temps=None, pupae=None):
        self._status_lbl.config(text="Running PCA…")
        self._prog["value"] = 90

        diets = np.array(diets, dtype=object) if diets is not None else None
        temps = np.array(temps, dtype=object) if temps is not None else None
        pupae = np.array(pupae, dtype=object) if pupae is not None else None

        scaled = bool(self._scale_var.get())
        # Retain the UNSCALED features. The supervised classifier (F4) must fit
        # its scaler inside each CV fold; scaling the whole matrix here would
        # leak test-fold statistics into training. PCA/plotting below still use
        # the globally-scaled matrix, which is fine (unsupervised, descriptive).
        X_unscaled = X.copy()
        # Scale BEFORE outlier screening so both use the same feature space
        if scaled:
            X = StandardScaler().fit_transform(X)

        # FlyHash outlier scan — fruit-fly sparse-code novelty detection.
        # Flags the most-novel spectra (top N%) and drops them before PCA so a
        # few artefact/contaminant spectra can't hijack the principal components.
        self._flyhash_dropped = 0
        if self._flyhash_out_var.get() and X.shape[0] > 10:
            nov = flyhash_novelty(X_unscaled)
            pct = float(self._flyhash_pct.get())
            cut = np.percentile(nov, 100.0 - pct)
            keep = nov < cut
            if keep.sum() >= 5 and keep.sum() < len(keep):
                self._flyhash_dropped = int((~keep).sum())
                X = X[keep]; X_unscaled = X_unscaled[keep]; labels = labels[keep]
                if diets is not None: diets = diets[keep]
                if temps is not None: temps = temps[keep]
                if pupae is not None: pupae = pupae[keep]

        # Outlier removal — Hotelling T² across the first k PCs (95% χ² limit).
        # NOTE: this is a global, unsupervised screen applied before CV. It can
        # still exert a mild optimistic bias on cross-validated metrics; it is
        # user-toggled (Outlier removal) and can be disabled for a strictly
        # leak-free estimate.
        if self._outlier_var.get() and X.shape[0] > 5:
            from scipy.stats import chi2
            k = int(min(self._n_comp.get(), X.shape[0] - 1, X.shape[1]))
            sc_tmp = PCA(n_components=k).fit_transform(X)
            sd = sc_tmp.std(axis=0, ddof=1)
            sd[sd == 0] = 1.0
            t2   = np.sum((sc_tmp / sd) ** 2, axis=1)
            keep = t2 <= chi2.ppf(0.975, df=k)
            X = X[keep]; X_unscaled = X_unscaled[keep]; labels = labels[keep]
            if diets is not None: diets = diets[keep]
            if temps is not None: temps = temps[keep]
            if pupae is not None: pupae = pupae[keep]

        n_comp = min(self._n_comp.get(), X.shape[0], X.shape[1])
        # Fit enough components (capped) to drive objective scree guidance,
        # then expose the user-selected n_comp for scores/loadings.
        k_guide = int(min(X.shape[0] - 1, X.shape[1], 30))
        k_guide = max(k_guide, n_comp)
        pca    = PCA(n_components=k_guide)
        scores_full = pca.fit_transform(X)
        expl_full   = pca.explained_variance_ratio_
        eig_full    = pca.explained_variance_
        scores = scores_full[:, :n_comp]
        expl   = expl_full[:n_comp]
        loads  = pca.components_[:n_comp]
        suggest = pca_component_suggestions(expl_full, eig_full, scaled)

        self._results = {
            "pca": pca, "scores": scores, "labels": labels,
            "waves": waves, "expl": expl, "loads": loads, "X": X,
            "expl_full": expl_full, "eig_full": eig_full, "suggest": suggest,
            "n_comp": n_comp,
            "X_unscaled": X_unscaled, "scaled": scaled,
            "diets": diets, "temps": temps, "pupae": pupae,
            "spatial_shapes": getattr(self, "_spatial_shapes", []),
        }

        self._draw_pca()
        self._prog["value"] = 100
        n_kept = len(labels)
        _fly = getattr(self, "_flyhash_dropped", 0)
        _flytxt = f"  ·  🪰 dropped {_fly} outlier(s)" if _fly else ""
        self._status_lbl.config(
            text=f"Done.  {n_kept} spectra from {len(self._files)} files.{_flytxt}\n"
                 f"PC1={expl[0]*100:.1f}%  PC2={expl[1]*100:.1f}%  ·  "
                 f"suggested PCs: {suggest['cumvar']} (95% var) / "
                 f"{suggest['kaiser']} (Kaiser) / {suggest['broken_stick']} (broken-stick)")

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
        # User-adjustable marker size (default smaller so dense clouds resolve)
        try:
            _pt_sz = max(2.0, float(self._pt_size.get()))
        except Exception:
            _pt_sz = 18.0
        diets = r.get("diets"); temps = r.get("temps")
        use_factor = (diets is not None and temps is not None
                      and any(d is not None for d in diets))

        def _draw_ellipse(x, y, col):
            # 95% Hotelling-T² confidence ellipse (F-distribution, finite n).
            # Replaces the old fixed 2·√λ (~86%) ellipse; T² is the chemometrics
            # standard for PCA score plots and is interpretable as a 95% region.
            if len(x) >= 3:
                cov = np.cov(x, y)
                ev, evec = np.linalg.eigh(cov)
                ev  = np.maximum(ev, 0)
                ang = np.degrees(np.arctan2(*evec[:, 1][::-1]))
                rad = hotelling_t2_radius(len(x), conf=0.95)
                ax.add_patch(Ellipse((x.mean(), y.mean()),
                             2*rad*np.sqrt(ev[1]), 2*rad*np.sqrt(ev[0]),
                             angle=ang, edgecolor=col, facecolor="none",
                             alpha=0.9, lw=1.8, ls="-"))

        if use_factor:
            from matplotlib.lines import Line2D
            diet_order = ["Control", "Glycerol", "Proline", "Trehalose"]
            uniq_diet  = [d for d in diet_order if d in set(diets)]
            uniq_diet += [d for d in sorted(set(map(str, diets)))
                          if d not in uniq_diet and d != "None"]
            diet_col = {d: self.COLORS[i % len(self.COLORS)]
                        for i, d in enumerate(uniq_diet)}
            temp_marker = {"5°C": "o", "15°C": "^"}
            uniq_temp   = [t for t in ["5°C", "15°C"] if t in set(temps)]
            for d in uniq_diet:
                for t in uniq_temp:
                    idx = np.array([(dd == d and tt == t)
                                    for dd, tt in zip(diets, temps)])
                    if not idx.any():
                        continue
                    x, y = scores[idx, 0], scores[idx, 1]
                    ax.scatter(x, y, s=_pt_sz, color=diet_col[d],
                               marker=temp_marker.get(t, "s"), alpha=0.55,
                               edgecolor="white", linewidth=0.35, zorder=3)
                    _draw_ellipse(x, y, diet_col[d])
            # Defer legends to the empty 6th panel so they never cover data
            self._legend_handles = (
                [mpatches.Patch(color=diet_col[d], label=d) for d in uniq_diet],
                [Line2D([0], [0], marker=temp_marker.get(t, "s"),
                        color="#444444", linestyle="none", markersize=9,
                        label=t) for t in uniq_temp])
        else:
            for g in groups:
                idx = labels == g
                x, y = scores[idx, 0], scores[idx, 1]
                col  = color_map[g]
                ax.scatter(x, y, s=_pt_sz, color=col, alpha=0.75, label=g, zorder=3)
                _draw_ellipse(x, y, col)
            self._legend_handles = (
                [mpatches.Patch(color=color_map[g], label=str(g))
                 for g in groups], [])
        ax.axhline(0, color=C["border"], lw=0.7)
        ax.axvline(0, color=C["border"], lw=0.7)
        ax.set_xlabel(f"PC1  ({expl[0]*100:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC2  ({expl[1]*100:.1f}%)", fontsize=10)
        ax.set_title("A: PCA Scores (PC1 vs PC2) · 95% T² ellipse",
                     fontsize=11, fontweight="semibold")
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        # Manual aspect ratio: 0 = auto (fill panel); >0 sets box height/width
        try:
            _asp = float(self._aspect.get())
        except Exception:
            _asp = 0.0
        if _asp and _asp > 0:
            ax.set_box_aspect(_asp)
        else:
            ax.set_box_aspect(None)

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
        ax.set_xticklabels(groups, fontsize=8, rotation=35, ha="right")
        ax.set_ylabel(f"{pc_lbl} score", fontsize=10)
        ax.set_title(f"C: {pc_lbl} Score Distribution",
                     fontsize=11, fontweight="semibold")
        ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.5)

        # ── D: Explained variance scree + objective PC-number guidance ─────────
        ax = self.axes[1, 1]
        evr_full = r.get("expl_full", expl)
        sug      = r.get("suggest", {})
        nshow    = int(min(len(evr_full), 15))     # keep the bar plot readable
        ev = np.asarray(evr_full[:nshow]) * 100
        xs = range(1, nshow + 1)
        ax.bar(xs, ev, color=C["accent"], alpha=0.8)
        ax.plot(xs, np.cumsum(evr_full)[:nshow] * 100,
                "o-", color=C["danger"], lw=1.5, ms=4, label="Cumulative")
        tgt = sug.get("var_target", 0.95) * 100
        ax.axhline(tgt, ls="--", color=C["text_dim"], lw=0.8)
        ax.text(nshow, tgt, f" {tgt:.0f}%", fontsize=7, color=C["text_dim"],
                va="bottom", ha="right")
        # mark the three objective criteria
        crit = [("cumvar", "95% var", C["danger"]),
                ("kaiser", "Kaiser", "#2E8B57"),
                ("broken_stick", "broken-stick", "#8E44AD")]
        seen = {}
        for key, lab, col in crit:
            k = sug.get(key)
            if k and k <= nshow:
                off = seen.get(k, 0); seen[k] = off + 1
                ax.axvline(k, ls=":", color=col, lw=1.3, alpha=0.9)
                ax.text(k, 96 - off*9, f"{lab}={k}", rotation=90, fontsize=6.5,
                        color=col, ha="right", va="top")
        # mark the currently-selected n_comp
        ncs = r.get("n_comp")
        if ncs and ncs <= nshow:
            ax.axvline(ncs, color="#222222", lw=1.0, alpha=0.5)
        ax.set_xlabel("Principal Component", fontsize=10)
        ax.set_ylabel("Explained Variance  (%)", fontsize=10)
        ax.set_title("D: Scree + suggested #PCs", fontsize=11, fontweight="semibold")
        ax.set_xticks(list(xs))
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8, loc="center right")
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)

        # ── 6th panel → dedicated legend area (never overlaps data) ───────────
        lax = self.axes[1, 2]
        lax.clear()
        lax.set_visible(True)
        lax.axis("off")
        diet_h, temp_h = getattr(self, "_legend_handles", ([], []))
        if diet_h:
            leg1 = lax.legend(handles=diet_h, title="Diet", fontsize=10,
                              title_fontsize=11, loc="upper left",
                              bbox_to_anchor=(0.0, 1.0), framealpha=0.95,
                              borderpad=0.6, labelspacing=0.5)
            lax.add_artist(leg1)
        if temp_h:
            lax.legend(handles=temp_h, title="Temperature", fontsize=10,
                       title_fontsize=11, loc="upper left",
                       bbox_to_anchor=(0.0, 0.55), framealpha=0.95,
                       borderpad=0.6, labelspacing=0.5)

        self.canvas.draw_idle()

    def _redraw_spatial(self):
        """Called when PC selector spinbox changes — redraw all panels."""
        if self._results is not None:
            self._draw_pca()

    # ── one-click mean spectrum + Raman band annotation ───────────────────────
    # Standard biological Raman band library (cm⁻¹, label, assignment)
    _BAND_LIB = [
        (480,  "480",  "Carbohydrate / trehalose (C–O–C, ring)"),
        (855,  "855",  "Proline / C–C, tyrosine ring"),
        (937,  "937",  "C–C backbone (proline, protein)"),
        (1004, "1004", "Phenylalanine ring breathing (protein ref)"),
        (1080, "1080", "C–C / C–O (lipid chain, carbohydrate)"),
        (1265, "1265", "=C–H deformation (unsat. lipid) / amide III"),
        (1300, "1300", "CH₂ twist (lipid acyl chains)"),
        (1440, "1440", "CH₂/CH₃ deformation (lipid + protein)"),
        (1656, "1656", "Amide I / C=C stretch (protein, unsat.)"),
        (1745, "1745", "C=O ester (triglyceride)"),
        (2850, "2850", "CH₂ sym stretch (lipid)"),
        (2885, "2885", "CH₂ asym stretch (lipid order)"),
        (2930, "2930", "CH₃ stretch (protein/lipid)"),
    ]

    def _show_mean_spectrum(self):
        """Plot mean ± SD spectrum per group with annotated Raman bands.
        Lets the user see where real signal lies before choosing cutoffs."""
        if self._results is None:
            messagebox.showwarning("No results",
                "Run PCA first (loads + preprocesses the spectra).", parent=self)
            return
        r = self._results
        waves = np.asarray(r["waves"], dtype=float)
        X = np.asarray(r["X"]); labels = np.asarray(r["labels"])
        groups = np.unique(labels)
        color_map = {g: self.COLORS[i % len(self.COLORS)]
                     for i, g in enumerate(groups)}

        win = tk.Toplevel(self)
        win.title("Mean spectrum + Raman band annotation")
        win.geometry("1150x680")
        win.configure(bg=C["bg"])

        fig = plt.figure(figsize=(12, 6.4), facecolor="#ffffff")
        ax = fig.add_subplot(111)

        ymax = 0.0
        for g in groups:
            idx = labels == g
            mu = X[idx].mean(axis=0)
            sd = X[idx].std(axis=0)
            col = color_map[g]
            ax.plot(waves, mu, color=col, lw=1.3, label=f"{g} (n={int(idx.sum())})",
                    zorder=3)
            ax.fill_between(waves, mu - sd, mu + sd, color=col, alpha=0.15, zorder=2)
            ymax = max(ymax, float(np.nanmax(mu + sd)))

        # annotate standard bands that fall inside the measured range
        wlo, whi = float(waves.min()), float(waves.max())
        for wn, lbl, _assign in self._BAND_LIB:
            if wlo < wn < whi:
                ax.axvline(wn, ls=":", color=C["text_dim"], lw=0.7, zorder=1)
                ax.text(wn, ymax * 1.01, lbl, rotation=90, fontsize=7,
                        ha="center", va="bottom", color=C["text_mid"])

        # shade the non-informative "silent" region 1800–2700 if present
        if wlo < 1800 and whi > 2700:
            ax.axvspan(1800, 2700, color="#bbbbbb", alpha=0.12, zorder=0)
            ax.text((1800 + 2700) / 2, ymax * 0.5, "silent region\n(noise only)",
                    ha="center", va="center", fontsize=8, color=C["text_dim"],
                    style="italic")

        ax.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=11)
        ax.set_ylabel("Normalised intensity (mean ± SD)", fontsize=11)
        ax.set_title("Group mean spectra with biological Raman band assignments",
                     fontsize=12, fontweight="semibold")
        ax.legend(fontsize=8, framealpha=0.9, ncol=2)
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        ax.margins(x=0.01)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, win).update()

        bar = tk.Frame(win, bg=C["bg"]); bar.pack(fill="x", pady=4)
        info = tk.Label(bar, bg=C["bg"], fg=C["text_mid"], font=("Segoe UI", 9),
                        text=("Tip: use this to pick Wavenumber min/max. "
                              "Fingerprint 600–1800 = primary; CH-stretch 2800–3030 "
                              "= separate run; skip the shaded silent region."))
        info.pack(side="left", padx=8)

        def _save():
            p = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf"), ("PNG", "*.png"), ("SVG", "*.svg")],
                parent=win)
            if p:
                fig.savefig(p, dpi=600, bbox_inches="tight")
        ttk.Button(bar, text="↓ Save figure", style="N.TButton",
                   command=_save).pack(side="right", padx=8)
        self._mean_fig = fig

    # ── class-imbalance resampler factory (SMOTE family) ───────────────────────
    @staticmethod
    def _build_sampler(name, y_train, seed):
        """Return an imbalanced-learn sampler for the given training labels, or
        None if no resampling is requested / possible. k-neighbours is capped to
        the smallest class so SMOTE never asks for more neighbours than exist."""
        if name in (None, "None"):
            return None
        from collections import Counter
        counts = Counter(y_train)
        min_n = min(counts.values())
        # need ≥2 samples in the smallest class to interpolate new ones
        k = max(1, min(5, min_n - 1))
        from imblearn.over_sampling import (RandomOverSampler, SMOTE,
                                            BorderlineSMOTE, ADASYN)
        from imblearn.combine import SMOTEENN, SMOTETomek
        if name == "Random over-sample":
            return RandomOverSampler(random_state=seed)
        if min_n < 2:
            # SMOTE-style interpolation impossible with a singleton class;
            # degrade gracefully to plain random over-sampling.
            return RandomOverSampler(random_state=seed)
        if name == "SMOTE":
            return SMOTE(k_neighbors=k, random_state=seed)
        if name == "Borderline-SMOTE":
            return BorderlineSMOTE(k_neighbors=k, random_state=seed)
        if name == "ADASYN":
            return ADASYN(n_neighbors=k, random_state=seed)
        if name == "SMOTE-ENN":
            return SMOTEENN(random_state=seed, smote=SMOTE(k_neighbors=k,
                                                           random_state=seed))
        if name == "SMOTE-Tomek":
            return SMOTETomek(random_state=seed, smote=SMOTE(k_neighbors=k,
                                                            random_state=seed))
        return None

    # ── supervised classification: PLS-DA / LDA with cross-validation ──────────
    def _run_classifier(self):
        if self._results is None:
            messagebox.showwarning("No results", "Run PCA first.", parent=self)
            return
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.model_selection import (StratifiedKFold, GroupKFold,
                                                 cross_val_predict)
            from sklearn.preprocessing import LabelBinarizer, StandardScaler
            from sklearn.pipeline import make_pipeline
            from sklearn.metrics import (accuracy_score, confusion_matrix,
                                         balanced_accuracy_score)
        except Exception as ex:
            messagebox.showerror("Missing library",
                                 f"scikit-learn required.\n{ex}", parent=self)
            return

        # ── class-imbalance handling (SMOTE family) ───────────────────────────
        samp_name = (self._clf_imbalance.get()
                     if hasattr(self, "_clf_imbalance") else "None")
        if samp_name != "None":
            try:
                import imblearn  # noqa: F401
            except Exception:
                messagebox.showerror(
                    "Missing library",
                    "Class-imbalance resampling needs imbalanced-learn.\n\n"
                    "Install it with:\n    pip install imbalanced-learn\n\n"
                    "Then re-run, or set 'Class imbalance' back to None.",
                    parent=self)
                return

        r = self._results
        # F4 — classify the UNSCALED features and fold scaling into a per-fold
        # Pipeline, so StandardScaler is fit on training data only (no leakage).
        # Fall back to the stored matrix for older sessions without the key.
        X = np.asarray(r.get("X_unscaled", r["X"]))
        clf_scaled = bool(r.get("scaled", False))
        def _pipe(est):
            return make_pipeline(StandardScaler(), est) if clf_scaled else est
        y = np.asarray(r["labels"])
        classes = np.unique(y)
        if len(classes) < 2:
            messagebox.showwarning("Classification",
                "Need at least 2 groups.", parent=self)
            return
        # smallest class size sets the CV fold count (cap at 5)
        min_n = min(int(np.sum(y == c)) for c in classes)
        if min_n < 2:
            messagebox.showwarning("Classification",
                "Each group needs ≥2 spectra for cross-validation.", parent=self)
            return
        n_splits = int(min(5, min_n))

        # ── pupa-grouped CV: keep all spectra from one pupa in the same fold ──
        groups = r.get("pupae")
        use_groups = groups is not None and len(np.unique(groups)) >= 2
        if use_groups:
            groups = np.asarray(groups)
            # need ≥2 pupae per class, and folds ≤ min pupae-per-class
            per_class_pupae = [len(np.unique(groups[y == c])) for c in classes]
            if min(per_class_pupae) >= 2:
                n_splits = int(min(n_splits, min(per_class_pupae)))
            else:
                use_groups = False   # not enough pupae per group → fall back
        cv_kind = ("pupa-grouped" if use_groups else "spectrum-level")

        win = tk.Toplevel(self)
        win.title("Supervised Classification — PLS-DA / LDA")
        win.geometry("1150x560")
        win.configure(bg=C["bg"])

        method = (self._clf_method.get()
                  if hasattr(self, "_clf_method") else "PLS-DA")
        def _make_cv(seed):
            if use_groups:
                return GroupKFold(n_splits=n_splits)
            return StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        cv = _make_cv(self._seed)
        cv_groups = groups if use_groups else None

        # ── ask how many permutations (heavy step) — 0 to skip ────────────────
        from tkinter import simpledialog
        n_spec = X.shape[0]
        suggested = 100 if n_spec <= 3000 else (50 if n_spec <= 8000 else 0)
        n_perm = simpledialog.askinteger(
            "Permutation test",
            (f"{n_spec} spectra loaded.\n\n"
             "How many label permutations for the significance test?\n"
             "  • 0   = skip (fast — accuracy + confusion matrix only)\n"
             "  • 100 = recommended\n"
             "  • 200 = thorough (slower)\n\n"
             "The test now runs in the background; you can keep using the app."),
            parent=self, minvalue=0, maxvalue=1000, initialvalue=suggested)
        if n_perm is None:
            return   # user cancelled

        # ── progress / cancel window ──────────────────────────────────────────
        win = tk.Toplevel(self)
        win.title("Supervised Classification — PLS-DA / LDA")
        win.geometry("1150x600")
        win.configure(bg=C["bg"])
        status = tk.Label(win, text="Running cross-validation…",
                          bg=C["bg"], fg=C["text_hi"], font=("Segoe UI", 11))
        status.pack(pady=8)
        prog = ttk.Progressbar(win, mode="determinate", maximum=100)
        prog.pack(fill="x", padx=20, pady=4)
        cancel_evt = threading.Event()
        ttk.Button(win, text="Cancel", style="D.TButton",
                   command=cancel_evt.set).pack(pady=4)

        def _set_status(txt, pct=None):
            self.after(0, lambda: status.config(text=txt))
            if pct is not None:
                self.after(0, lambda: prog.configure(value=pct))

        def _compute():
            results = {}
            try:
                lb = LabelBinarizer()
                Yfull = lb.fit_transform(y)
                if Yfull.shape[1] == 1:
                    Yfull = np.hstack([1 - Yfull, Yfull])
                n_comp = int(min(self._n_comp.get(), X.shape[1],
                                 max(2, len(classes))))
                ug = np.unique(groups) if use_groups else None
                total_steps = 2 * (1 + n_perm)
                step = [0]
                def _tick(label):
                    step[0] += 1
                    _set_status(label, 100.0 * step[0] / max(1, total_steps))

                for name in ("PLS-DA", "LDA"):
                    if cancel_evt.is_set():
                        return
                    def _oof_resampled(yv, cvv):
                        # Out-of-fold predictions with SMOTE-family resampling
                        # applied to each TRAINING fold only (test fold is never
                        # touched → no optimistic leakage). Scaling is likewise
                        # fit on the (resampled) training fold only.
                        yv = np.asarray(yv, dtype=object)
                        pred = np.empty(len(yv), dtype=object)
                        for tr, te in cvv.split(X, yv, groups=cv_groups):
                            Xtr, ytr = X[tr], yv[tr]
                            try:
                                samp = self._build_sampler(samp_name, ytr,
                                                           self._seed)
                                if samp is not None:
                                    Xtr, ytr = samp.fit_resample(Xtr, ytr)
                            except Exception:
                                Xtr, ytr = X[tr], yv[tr]   # fall back: no resample
                            Xte = X[te]
                            if clf_scaled:
                                sc = StandardScaler().fit(Xtr)
                                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
                            if name == "PLS-DA":
                                lbf = LabelBinarizer()
                                Ytr = lbf.fit_transform(ytr)
                                if Ytr.shape[1] == 1:
                                    Ytr = np.hstack([1 - Ytr, Ytr])
                                est = PLSRegression(
                                    n_components=min(n_comp, Xtr.shape[1],
                                                     max(2, len(lbf.classes_)))
                                ).fit(Xtr, Ytr)
                                Yh = np.asarray(est.predict(Xte))
                                pred[te] = lbf.classes_[np.argmax(Yh, axis=1)]
                            else:
                                est = LinearDiscriminantAnalysis().fit(Xtr, ytr)
                                pred[te] = est.predict(Xte)
                        return pred

                    def fit(yv, cvv):
                        if samp_name != "None":
                            return _oof_resampled(yv, cvv)
                        if name == "PLS-DA":
                            Yv = lb.fit_transform(yv)
                            if Yv.shape[1] == 1:
                                Yv = np.hstack([1 - Yv, Yv])
                            Yh = cross_val_predict(
                                _pipe(PLSRegression(n_components=n_comp)), X, Yv,
                                cv=cvv, groups=cv_groups)
                            if len(classes) > 2:
                                return np.array(classes)[np.argmax(Yh, axis=1)]
                            return lb.classes_[(Yh[:, 1] > Yh[:, 0]).astype(int)]
                        return cross_val_predict(
                            _pipe(LinearDiscriminantAnalysis()), X, yv,
                            cv=cvv, groups=cv_groups)

                    _tick(f"{name}: observed fit…")
                    pred = fit(y, cv)
                    bacc_obs = balanced_accuracy_score(y, pred)
                    res = {
                        "pred": pred,
                        "acc": accuracy_score(y, pred),
                        "bacc": bacc_obs,
                        "cm": confusion_matrix(y, pred, labels=classes),
                        "sampler": samp_name,
                    }
                    if n_perm > 0:
                        rng = np.random.default_rng(self._seed)
                        null = np.empty(n_perm)
                        for pi in range(n_perm):
                            if cancel_evt.is_set():
                                return
                            if use_groups:
                                glabel = {g: y[groups == g][0] for g in ug}
                                permvals = rng.permutation(list(glabel.values()))
                                gmap = dict(zip(ug, permvals))
                                yp = np.array([gmap[g] for g in groups],
                                              dtype=object)
                            else:
                                yp = rng.permutation(y)
                            null[pi] = balanced_accuracy_score(
                                yp, fit(yp, _make_cv(int(rng.integers(1e9)))))
                            if pi % 5 == 0 or pi == n_perm - 1:
                                _tick(f"{name}: permutation {pi+1}/{n_perm}…")
                        res["p_perm"] = float((np.sum(null >= bacc_obs) + 1)
                                              / (n_perm + 1))
                        res["null_mean"] = float(null.mean())
                        res["n_perm"] = n_perm
                    results[name] = res
            except Exception as ex:
                results["__error__"] = str(ex)
            if not cancel_evt.is_set():
                self.after(0, lambda: _draw_results(results))

        def _draw_results(results):
            if "__error__" in results:
                status.config(text=f"Error: {results['__error__']}",
                              fg=C["danger"])
                prog.destroy(); return
            status.destroy(); prog.destroy()
            self._render_clf(win, results, classes, n_splits, cv_kind)

        threading.Thread(target=_compute, daemon=True).start()
        return

    def _render_clf(self, win, results, classes, n_splits, cv_kind):
        fig = plt.figure(figsize=(11.5, 5.2), facecolor="#ffffff")
        axes = [fig.add_subplot(1, 2, i + 1) for i in range(2)]
        for ax, name in zip(axes, ("PLS-DA", "LDA")):
            res = results.get(name, {})
            if "error" in res:
                ax.text(0.5, 0.5, f"{name}\nfailed:\n{res['error']}",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color=C["danger"]); ax.axis("off"); continue
            cm = res["cm"]
            cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(len(classes)))
            ax.set_yticks(range(len(classes)))
            ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=7)
            ax.set_yticklabels(classes, fontsize=7)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f"{cm[i, j]}",
                            ha="center", va="center", fontsize=7,
                            color="white" if cmn[i, j] > 0.5 else "#222")
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            pstr = ""
            if "p_perm" in res:
                pv = res["p_perm"]
                ptxt = f"p<{1.0/(res['n_perm']+1):.3f}" if pv <= 1.0/(res['n_perm']+1) \
                    else f"p={pv:.3f}"
                pstr = (f"\nperm. test ({res['n_perm']}×): {ptxt}  "
                        f"(null={res['null_mean']*100:.1f}%)")
            sstr = ""
            if res.get("sampler", "None") != "None":
                sstr = f"  ·  resampling: {res['sampler']}"
            ax.set_title(f"{name}\n{n_splits}-fold {cv_kind} CV{sstr}  ·  "
                         f"acc={res['acc']*100:.1f}%  ·  "
                         f"balanced={res['bacc']*100:.1f}%" + pstr,
                         fontsize=9, fontweight="semibold")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, win).update()
        self._clf_fig = fig
        self._clf_results = (results, classes, n_splits)

        bar = tk.Frame(win, bg=C["bg"]); bar.pack(fill="x", pady=4)
        def _save_clf():
            p = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf"), ("PNG", "*.png")], parent=win)
            if p:
                fig.savefig(p, dpi=600, bbox_inches="tight")
        def _save_clf_csv():
            p = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")], parent=win)
            if not p:
                return
            with open(p, "w", encoding="utf-8") as fh:
                for name in ("PLS-DA", "LDA"):
                    res = results.get(name, {})
                    if "error" in res:
                        fh.write(f"{name},error,{res['error']}\n\n"); continue
                    fh.write(f"{name},{n_splits}-fold {cv_kind} CV,"
                             f"resampling={res.get('sampler','None')},"
                             f"accuracy={res['acc']:.4f},"
                             f"balanced_accuracy={res['bacc']:.4f}")
                    if "p_perm" in res:
                        fh.write(f",permutation_p={res['p_perm']:.4f},"
                                 f"n_permutations={res['n_perm']},"
                                 f"null_mean_balanced_acc={res['null_mean']:.4f}")
                    fh.write("\n")
                    fh.write("true\\pred," + ",".join(map(str, classes)) + "\n")
                    for i, c in enumerate(classes):
                        fh.write(str(c) + "," +
                                 ",".join(map(str, res["cm"][i])) + "\n")
                    fh.write("\n")
        ttk.Button(bar, text="↓ Save figure", style="N.TButton",
                   command=_save_clf).pack(side="left", padx=8)
        ttk.Button(bar, text="↓ Save confusion CSV", style="N.TButton",
                   command=_save_clf_csv).pack(side="left", padx=4)

    # ── automated diagnostic band-ratio export ────────────────────────────────
    def _export_band_ratios(self):
        if self._results is None:
            messagebox.showwarning("No results", "Run PCA first.", parent=self)
            return
        r = self._results
        waves = np.asarray(r["waves"])
        X = np.asarray(r["X"]); labels = np.asarray(r["labels"])
        diets = r.get("diets"); temps = r.get("temps")

        def band(w_lo, w_hi):
            m = (waves >= w_lo) & (waves <= w_hi)
            if not m.any():
                return np.full(X.shape[0], np.nan)
            return _trapz(np.clip(X[:, m], 0, None), waves[m], axis=1)

        # Diagnostic windows (cm⁻¹) — standard biological Raman assignments
        I2850 = band(2840, 2860)   # CH2 sym str (lipid acyl chains)
        I2885 = band(2875, 2900)   # CH2 asym str
        I1656 = band(1640, 1680)   # C=C / amide I (unsaturation + protein)
        I1440 = band(1430, 1460)   # CH2/CH3 deformation (total lipid+protein)
        I1004 = band(995, 1010)    # phenylalanine ring-breathing (protein marker)
        Ilip  = band(2840, 2900)   # total CH-stretch lipid envelope
        Iprot = band(1640, 1680)   # amide I (protein)
        Icarb = band(470, 560)     # carbohydrate / trehalose-rich region
        eps = 1e-12
        ratios = {
            "Lipid_order_2850_2885":   I2850 / (I2885 + eps),
            "Unsaturation_1656_1440":  I1656 / (I1440 + eps),
            "Protein_lipid_1004_2850": I1004 / (I2850 + eps),
            "Amide_lipid_1656_2850":   Iprot / (Ilip + eps),
            "Carb_protein_500_1004":   Icarb / (I1004 + eps),
        }

        outdir = filedialog.askdirectory(
            title="Choose folder for band-ratio outputs", parent=self)
        if not outdir:
            return
        # ── CSV (per-spectrum) ────────────────────────────────────────────────
        csv_path = os.path.join(outdir, "band_ratios_per_spectrum.csv")
        cols = list(ratios.keys())
        with open(csv_path, "w", encoding="utf-8") as fh:
            hdr = ["index", "group", "diet", "temperature"] + cols
            fh.write(",".join(hdr) + "\n")
            for i in range(X.shape[0]):
                d = (diets[i] if diets is not None else "")
                t = (temps[i] if temps is not None else "")
                row = [str(i), str(labels[i]), str(d), str(t)] + \
                      [f"{ratios[c][i]:.6f}" for c in cols]
                fh.write(",".join(row) + "\n")

        # ── CSV (per-group summary mean ± SD) ─────────────────────────────────
        groups = np.unique(labels)
        sum_path = os.path.join(outdir, "band_ratios_group_summary.csv")
        with open(sum_path, "w", encoding="utf-8") as fh:
            fh.write("group,n," +
                     ",".join([f"{c}_mean,{c}_sd" for c in cols]) + "\n")
            for g in groups:
                idx = labels == g
                cells = [str(g), str(int(idx.sum()))]
                for c in cols:
                    v = ratios[c][idx]
                    cells += [f"{np.nanmean(v):.6f}", f"{np.nanstd(v):.6f}"]
                fh.write(",".join(cells) + "\n")

        # ── boxplot figure ────────────────────────────────────────────────────
        ncols = len(cols)
        fig, axs = plt.subplots(1, ncols, figsize=(3.4 * ncols, 4.6),
                                facecolor="#ffffff")
        if ncols == 1:
            axs = [axs]
        palette = {g: self.COLORS[i % len(self.COLORS)]
                   for i, g in enumerate(groups)}
        for ax, c in zip(axs, cols):
            data = [ratios[c][labels == g] for g in groups]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6)
            for patch, g in zip(bp["boxes"], groups):
                patch.set_facecolor(palette[g]); patch.set_alpha(0.35)
            for med in bp["medians"]:
                med.set_color("#222"); med.set_linewidth(1.5)
            for gi, g in enumerate(groups):
                v = ratios[c][labels == g]
                jit = np.random.uniform(-0.12, 0.12, len(v))
                ax.scatter(np.full(len(v), gi + 1) + jit, v,
                           color=palette[g], s=10, alpha=0.5, zorder=3)
            ax.set_xticks(range(1, len(groups) + 1))
            ax.set_xticklabels(groups, rotation=40, ha="right", fontsize=7)
            ax.set_title(c.replace("_", " "), fontsize=9, fontweight="semibold")
            ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.5)
        fig.tight_layout()
        fig_path = os.path.join(outdir, "band_ratios_boxplots.pdf")
        fig.savefig(fig_path, bbox_inches="tight")
        png_path = os.path.join(outdir, "band_ratios_boxplots.png")
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        plt.close(fig)

        self._status_lbl.config(
            text=f"Band ratios → {Path(outdir).name}  "
                 f"(per-spectrum + summary CSV + boxplots)")
        messagebox.showinfo("Band-ratio export",
            "Saved:\n• band_ratios_per_spectrum.csv\n"
            "• band_ratios_group_summary.csv\n"
            "• band_ratios_boxplots.pdf / .png", parent=self)

    def _save_fig(self):
        if self._results is None:
            messagebox.showwarning("No results","Run PCA first.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF","*.pdf"),("PNG","*.png"),("SVG","*.svg")],
            parent=self)
        if path:
            self.fig.savefig(path, dpi=600, bbox_inches="tight")
            self._status_lbl.config(text=f"Saved → {Path(path).name}")

    def _save_panels(self):
        """Export each PCA panel as a standalone publication-quality file."""
        if self._results is None:
            messagebox.showwarning("No results", "Run PCA first.", parent=self)
            return
        outdir = filedialog.askdirectory(
            title="Choose folder for individual panels", parent=self)
        if not outdir:
            return
        ans = messagebox.askyesno(
            "File format",
            "Save as PDF (vector, best for publication)?\n\n"
            "Yes = PDF   ·   No = PNG (600 dpi).", parent=self)
        ext = ".pdf" if ans else ".png"
        kinds = [("A_scores_PC1_PC2", "scores"),
                 ("B_loadings",       "loadings"),
                 ("C_score_distribution", "dist"),
                 ("D_explained_variance", "scree"),
                 ("E_spatial_score_map",  "spatial")]
        saved = 0
        for nm, kind in kinds:
            try:
                fig = self._render_single_panel(kind)
                if fig is None:
                    continue
                fname = os.path.join(outdir, f"PCA_{nm}{ext}")
                if ext == ".pdf":
                    fig.savefig(fname, bbox_inches="tight")
                else:
                    fig.savefig(fname, dpi=600, bbox_inches="tight")
                plt.close(fig)
                saved += 1
            except Exception as ex:
                messagebox.showerror("Save error", f"{nm}: {ex}", parent=self)
        self._status_lbl.config(
            text=f"Saved {saved} panels ({ext}) → {Path(outdir).name}")

    def _render_single_panel(self, kind):
        """Build a clean standalone figure for one panel (publication quality).
        Legends sit OUTSIDE the axes so they never overlap the data."""
        r = self._results
        if r is None:
            return None
        scores = r["scores"]; labels = r["labels"]
        waves  = r["waves"];  expl = r["expl"]; loads = r["loads"]
        groups = np.unique(labels)
        color_map = {g: self.COLORS[i % len(self.COLORS)]
                     for i, g in enumerate(groups)}
        pc_idx = min(self._pc_sel.get() - 1, scores.shape[1] - 1)
        pc_lbl = f"PC{pc_idx + 1}"

        def _ellipse(ax, x, y, col):
            if len(x) >= 3:
                cov = np.cov(x, y)
                ev, evec = np.linalg.eigh(cov)
                ev = np.maximum(ev, 0)
                ang = np.degrees(np.arctan2(*evec[:, 1][::-1]))
                rad = hotelling_t2_radius(len(x), conf=0.95)   # 95% T² ellipse
                ax.add_patch(Ellipse((x.mean(), y.mean()),
                             2*rad*np.sqrt(ev[1]), 2*rad*np.sqrt(ev[0]),
                             angle=ang, edgecolor=col, facecolor="none",
                             alpha=0.9, lw=1.8))

        if kind == "scores":
            fig, ax = plt.subplots(figsize=(8.5, 6.5))
            try:
                _pt_sz = max(2.0, float(self._pt_size.get()))
            except Exception:
                _pt_sz = 18.0
            diets = r.get("diets"); temps = r.get("temps")
            use_factor = (diets is not None and temps is not None
                          and any(d is not None for d in diets))
            if use_factor:
                from matplotlib.lines import Line2D
                diet_order = ["Control", "Glycerol", "Proline", "Trehalose"]
                uniq_diet  = [d for d in diet_order if d in set(diets)]
                uniq_diet += [d for d in sorted(set(map(str, diets)))
                              if d not in uniq_diet and d != "None"]
                diet_col = {d: self.COLORS[i % len(self.COLORS)]
                            for i, d in enumerate(uniq_diet)}
                temp_marker = {"5°C": "o", "15°C": "^"}
                uniq_temp = [t for t in ["5°C", "15°C"] if t in set(temps)]
                for d in uniq_diet:
                    for t in uniq_temp:
                        idx = np.array([(dd == d and tt == t)
                                        for dd, tt in zip(diets, temps)])
                        if not idx.any():
                            continue
                        x, y = scores[idx, 0], scores[idx, 1]
                        ax.scatter(x, y, s=_pt_sz, color=diet_col[d],
                                   marker=temp_marker.get(t, "s"), alpha=0.55,
                                   edgecolor="white", linewidth=0.35, zorder=3)
                        _ellipse(ax, x, y, diet_col[d])
                diet_handles = [mpatches.Patch(color=diet_col[d], label=d)
                                for d in uniq_diet]
                temp_handles = [Line2D([0], [0], marker=temp_marker.get(t, "s"),
                                color="#444444", linestyle="none", markersize=8,
                                label=t) for t in uniq_temp]
                # legends OUTSIDE the axes, stacked on the right
                leg1 = ax.legend(handles=diet_handles, title="Diet",
                                 fontsize=10, title_fontsize=10.5,
                                 loc="upper left", bbox_to_anchor=(1.02, 1.0),
                                 framealpha=0.95, borderaxespad=0)
                ax.add_artist(leg1)
                ax.legend(handles=temp_handles, title="Temperature",
                          fontsize=10, title_fontsize=10.5,
                          loc="upper left", bbox_to_anchor=(1.02, 0.55),
                          framealpha=0.95, borderaxespad=0)
            else:
                for g in groups:
                    idx = labels == g
                    x, y = scores[idx, 0], scores[idx, 1]
                    ax.scatter(x, y, s=_pt_sz, color=color_map[g], alpha=0.6,
                               edgecolor="white", linewidth=0.35,
                               label=g, zorder=3)
                    _ellipse(ax, x, y, color_map[g])
                ax.legend(fontsize=9, loc="upper left",
                          bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
            ax.axhline(0, color=C["border"], lw=0.7)
            ax.axvline(0, color=C["border"], lw=0.7)
            ax.set_xlabel(f"PC1  ({expl[0]*100:.1f}%)", fontsize=12)
            ax.set_ylabel(f"PC2  ({expl[1]*100:.1f}%)", fontsize=12)
            ax.set_title("PCA Scores (PC1 vs PC2)", fontsize=13,
                         fontweight="semibold")
            ax.grid(True, ls="--", lw=0.4, alpha=0.5)
            try:
                _asp = float(self._aspect.get())
            except Exception:
                _asp = 0.0
            if _asp and _asp > 0:
                ax.set_box_aspect(_asp)

        elif kind == "loadings":
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ld = loads[pc_idx]
            ax.fill_between(waves, ld, 0, where=ld >= 0, alpha=0.55,
                            color=C["accent"], label="Positive")
            ax.fill_between(waves, ld, 0, where=ld < 0, alpha=0.55,
                            color=C["danger"], label="Negative")
            ax.plot(waves, ld, color="#222222", lw=0.8, alpha=0.7)
            ax.axhline(0, color=C["border"], lw=0.9)
            ax.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=12)
            ax.set_ylabel("Loading weight", fontsize=12)
            ax.set_title(f"{pc_lbl} Loadings", fontsize=13, fontweight="semibold")
            ax.legend(fontsize=10, loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
            ax.grid(True, ls="--", lw=0.4, alpha=0.5)
            ylim = ax.get_ylim()
            for wn, bl in [(1000,"Ring"),(1300,"CH₂"),(1450,"CH"),
                           (1650,"C=O"),(2850,"CH₂"),(3000,"CH")]:
                if waves.min() < wn < waves.max():
                    ax.axvline(wn, ls=":", color=C["text_dim"], lw=0.8)
                    ax.text(wn, ylim[1]*0.85, bl, fontsize=8,
                            color=C["text_dim"], rotation=90, ha="right", va="top")

        elif kind == "dist":
            fig, ax = plt.subplots(figsize=(8.5, 6))
            for gi, g in enumerate(groups):
                idx = labels == g
                sc = scores[idx, pc_idx]; col = color_map[g]
                ax.boxplot(sc, positions=[gi], widths=0.4, patch_artist=True,
                           boxprops=dict(facecolor=col, alpha=0.3),
                           medianprops=dict(color=col, lw=2),
                           whiskerprops=dict(color=col),
                           capprops=dict(color=col),
                           flierprops=dict(marker="o", color=col, ms=4, alpha=0.5))
                jit = np.random.uniform(-0.15, 0.15, len(sc))
                ax.scatter(np.full(len(sc), gi) + jit, sc, color=col,
                           alpha=0.6, s=20, zorder=3)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, fontsize=9, rotation=35, ha="right")
            ax.set_ylabel(f"{pc_lbl} score", fontsize=12)
            ax.set_title(f"{pc_lbl} Score Distribution", fontsize=13,
                         fontweight="semibold")
            ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.5)

        elif kind == "scree":
            fig, ax = plt.subplots(figsize=(8, 5.5))
            evr_full = r.get("expl_full", expl)
            sug      = r.get("suggest", {})
            nshow = int(min(len(evr_full), 15))
            xs = range(1, nshow + 1)
            ax.bar(xs, np.asarray(evr_full[:nshow])*100, color=C["accent"], alpha=0.8)
            ax.plot(xs, np.cumsum(evr_full)[:nshow]*100, "o-",
                    color=C["danger"], lw=1.5, ms=5, label="Cumulative")
            tgt = sug.get("var_target", 0.95)*100
            ax.axhline(tgt, ls="--", color=C["text_dim"], lw=0.8)
            for key, lab, col in (("cumvar","95% var",C["danger"]),
                                  ("kaiser","Kaiser","#2E8B57"),
                                  ("broken_stick","broken-stick","#8E44AD")):
                k = sug.get(key)
                if k and k <= nshow:
                    ax.axvline(k, ls=":", color=col, lw=1.4, alpha=0.9, label=f"{lab}={k}")
            ax.set_xlabel("Principal Component", fontsize=12)
            ax.set_ylabel("Explained Variance  (%)", fontsize=12)
            ax.set_title("Explained Variance (Scree) + suggested #PCs", fontsize=13,
                         fontweight="semibold")
            ax.set_xticks(list(xs)); ax.set_ylim(0, 105)
            ax.legend(fontsize=9)
            ax.grid(True, ls="--", lw=0.4, alpha=0.5)

        elif kind == "spatial":
            spatial_shapes = r.get("spatial_shapes", [])
            shape_info = next(((lbl, Y, X) for lbl, Y, X in spatial_shapes
                               if Y is not None and X is not None), None)
            if shape_info is None:
                return None
            lbl_sp, Y_sp, X_sp = shape_info
            idx_sp = labels == lbl_sp
            sc_sp = scores[idx_sp, pc_idx]
            if len(sc_sp) != Y_sp * X_sp:
                return None
            fig, ax = plt.subplots(figsize=(7, 6))
            score_map = sc_sp.reshape(Y_sp, X_sp)
            norm_map = np.zeros_like(score_map, dtype=float)
            pos = score_map[score_map > 0]; neg = score_map[score_map < 0]
            if pos.size: norm_map[score_map > 0] = score_map[score_map > 0]/pos.max()
            if neg.size: norm_map[score_map < 0] = score_map[score_map < 0]/abs(neg.min())
            vmin, vmax = norm_map.min(), norm_map.max()
            cnorm = (TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                     if vmin < 0 < vmax else Normalize(vmin=vmin, vmax=vmax))
            im = ax.imshow(norm_map, cmap="RdBu_r", norm=cnorm,
                           origin="upper", interpolation="nearest", aspect="equal")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Score (norm. ±1)", fontsize=10)
            ax.set_xlabel("X (px)", fontsize=12); ax.set_ylabel("Y (px)", fontsize=12)
            ax.set_title(f"{pc_lbl} Spatial Scores\n{lbl_sp}", fontsize=13,
                         fontweight="semibold")
        else:
            return None

        fig.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# FLYHASH — fruit-fly mushroom-body sparse hashing for fast region + anomaly maps
# ─────────────────────────────────────────────────────────────────────────────
def make_connectome_connectivity(n_glom=50, n_kc=256, claws_mean=6.0,
                                 bias_strength=1.0, seed=0):
    """PN->KC connectivity from published fly mushroom-body wiring statistics:
    ~50 glomerular inputs, n_kc Kenyon cells, each KC sampling ~6 inputs
    ('claws') with non-uniform glomerular bias. Returns (n_glom, n_kc).
    To use the REAL measured wiring, load a FlyWire/neuPrint PN->KC matrix
    of shape (n_glom, n_kc) and pass it to flyhash_maps(connectivity=...)."""
    rng = np.random.default_rng(seed)
    base = rng.gamma(shape=max(1e-3, 1.0 / max(bias_strength, 1e-3)),
                     scale=1.0, size=n_glom)
    pref = base / base.sum()
    C = np.zeros((n_glom, n_kc))
    for kc in range(n_kc):
        n_claws = max(2, int(rng.poisson(claws_mean)))
        inputs = rng.choice(n_glom, size=min(n_claws, n_glom),
                            replace=False, p=pref)
        C[inputs, kc] = rng.lognormal(mean=0.0, sigma=0.4, size=len(inputs))
    return C


def _spectrum_to_glomeruli(flat, n_glom):
    """Reduce (n_pixels, B) spectra to (n_pixels, n_glom) channels by averaging
    contiguous spectral bands (a receptor->PN dimensionality reduction)."""
    B = flat.shape[1]
    edges = np.linspace(0, B, n_glom + 1).astype(int)
    return np.stack([flat[:, edges[g]:edges[g + 1]].mean(axis=1)
                     for g in range(n_glom)], axis=1)


def flyhash_maps(cube, n_bits=256, sparseness=0.05, n_regions=4,
                 do_regions=True, do_anomaly=True, seed=42,
                 projection="random", connectivity=None, n_glom=50):
    """Fly-brain-inspired analysis of a Raman hyperspectral cube.

    Mimics the fruit fly mushroom body: each pixel's spectrum is projected to a
    high-dim space, then only the strongest few percent of projections are kept
    (a sparse binary 'fingerprint'). These fingerprints are cheap to compare,
    which makes two things fast and robust:

      - REGION map  : cluster fingerprints into chemical domains (KMeans).
      - ANOMALY map : score each pixel by how few others share its fingerprint;
                      rare pixels (contaminants, defects, trace phases) light up.

    projection : "random"     -> dense Gaussian random matrix (standard FlyHash).
                 "connectome"  -> sparse PN->KC wiring prior (real fly structure):
                                  spectrum is reduced to n_glom channels, then
                                  passed through a biased ~6-claw connectivity
                                  matrix. Pass a real matrix via `connectivity`.
    cube : ndarray (H, W, B)   hyperspectral map (already preprocessed).
    Returns (region_map | None, anomaly_map | None), each shape (H, W).
    """
    cube = np.asarray(cube, dtype=float)
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)

    # Normalise each spectrum to unit area so overall brightness doesn't dominate.
    flat = flat - np.nanmin(flat, axis=1, keepdims=True)
    areas = np.nansum(flat, axis=1, keepdims=True)
    areas[areas == 0] = 1.0
    flat = np.nan_to_num(flat / areas)

    # The fly hash: project -> keep top-k -> sparse binary code.
    rng = np.random.default_rng(seed)
    if projection == "connectome":
        C = connectivity if connectivity is not None else \
            make_connectome_connectivity(n_glom=n_glom, n_kc=int(n_bits), seed=seed)
        activity = _spectrum_to_glomeruli(flat, C.shape[0]) @ C
    else:
        proj = rng.standard_normal((B, int(n_bits)))
        activity = flat @ proj
    n_on = max(1, int(int(n_bits) * float(sparseness)))
    thresh = np.sort(activity, axis=1)[:, -n_on][:, None]
    hashes = (activity >= thresh).astype(np.float32)

    region_map = None
    if do_regions:
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=int(n_regions), n_init=10,
                        random_state=0).fit_predict(hashes)
        region_map = labels.reshape(H, W).astype(float)

    anomaly_map = None
    if do_anomaly:
        # Novelty = inverse of how many 'on' bits a pixel shares with a random
        # reference sample of pixels. Fewer shared bits  =>  more unusual.
        n_ref = min(500, len(hashes))
        ref = hashes[rng.choice(len(hashes), size=n_ref, replace=False)]
        shared = hashes @ ref.T
        novelty = -shared.mean(axis=1)
        novelty -= novelty.min()
        if novelty.max() > 0:
            novelty /= novelty.max()
        anomaly_map = novelty.reshape(H, W)

    return region_map, anomaly_map


def flyhash_novelty(X, n_bits=256, sparseness=0.05, seed=42):
    """Per-row novelty score using the fly mushroom-body sparse hash.

    X : ndarray (n_samples, n_features)  — e.g. the PCA spectra matrix.
    Returns a 1-D array of novelty scores (0..1); high = unlike the rest of
    the dataset (candidate outlier / contaminant / artefact spectrum).
    """
    X = np.asarray(X, dtype=float)
    n, B = X.shape
    Xn = X - np.nanmin(X, axis=1, keepdims=True)
    areas = np.nansum(Xn, axis=1, keepdims=True)
    areas[areas == 0] = 1.0
    Xn = np.nan_to_num(Xn / areas)

    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((B, int(n_bits)))
    act = Xn @ proj
    n_on = max(1, int(int(n_bits) * float(sparseness)))
    thr = np.sort(act, axis=1)[:, -n_on][:, None]
    hashes = (act >= thr).astype(np.float32)

    n_ref = min(500, n)
    ref = hashes[rng.choice(n, size=n_ref, replace=False)]
    shared = hashes @ ref.T
    novelty = -shared.mean(axis=1)
    novelty -= novelty.min()
    if novelty.max() > 0:
        novelty /= novelty.max()
    return novelty


# ─────────────────────────────────────────────────────────────────────────────
# CELL MASK / BACKGROUND REJECTION — separate cell pixels from buffer (e.g. PBS)
# ─────────────────────────────────────────────────────────────────────────────
def _band_height_area(cube, wn, lo, hi, local_baseline=True):
    """Return (height_map, area_map) over [lo, hi] cm⁻¹ with an optional linear
    local baseline subtracted (endpoints of the band). Both shape (Y, X)."""
    m = (wn >= lo) & (wn <= hi)
    if m.sum() < 2:
        return None, None
    x = wn[m]
    sub = cube[:, :, m].astype(float)
    if local_baseline:
        t = (x - x[0]) / (x[-1] - x[0] + 1e-12)
        base = (sub[:, :, :1] * (1 - t)[None, None, :]
                + sub[:, :, -1:] * t[None, None, :])
        sub = np.clip(sub - base, 0, None)
    trap = getattr(np, "trapezoid", np.trapz)
    return sub.max(axis=2), trap(sub, x, axis=2)


def _noise_map(cube, wn, lo, hi):
    """Per-pixel noise: std in a signal-free window if available, else a robust
    high-frequency estimate from successive spectral differences."""
    m = (wn >= lo) & (wn <= hi)
    if m.sum() >= 5:
        return cube[:, :, m].std(axis=2)
    d = np.diff(cube.astype(float), axis=2)
    med = np.median(d, axis=2, keepdims=True)
    return 1.4826 * np.median(np.abs(d - med), axis=2) / np.sqrt(2.0)


def _otsu_threshold(vals):
    """Otsu's between-class-variance threshold for a 1-D array of values."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or v.ptp() == 0:
        return float(np.nanmedian(vals)) if v.size else 0.0
    hist, edges = np.histogram(v, bins=256)
    centres = (edges[:-1] + edges[1:]) / 2.0
    wb = np.cumsum(hist).astype(float)
    wf = wb[-1] - wb
    cs = np.cumsum(hist * centres)
    mb = cs / np.maximum(wb, 1)
    mf = (cs[-1] - cs) / np.maximum(wf, 1)
    between = wb * wf * (mb - mf) ** 2
    return float(centres[int(np.argmax(between))])


def cell_mask_analysis(cube, wn, cell_lo=2850.0, cell_hi=2950.0,
                       noise_lo=1750.0, noise_hi=1850.0, min_snr=3.0,
                       method="Otsu", percentile=75.0):
    """Auto cell-vs-background separation for a Raman map.

    Strategy: cells are rich in CH-stretch (2850–2950 cm⁻¹) where buffer/PBS is
    nearly silent, so an SNR map of that band cleanly marks cell pixels. Falls
    back to amide I (1620–1700) if the CH-stretch band is outside the data.

    Returns dict with: area, height, noise, snr (all (Y,X)); mask (bool Y,X);
    background_spectrum (W,); threshold (float); band (lo,hi) actually used.
    """
    lo, hi = cell_lo, cell_hi
    height, area = _band_height_area(cube, wn, lo, hi)
    if height is None:                       # CH-stretch absent → amide I
        lo, hi = 1620.0, 1700.0
        height, area = _band_height_area(cube, wn, lo, hi)
    if height is None:
        raise ValueError("Neither CH-stretch nor amide I band is within the "
                         "spectral range of this map.")
    noise = _noise_map(cube, wn, noise_lo, noise_hi)
    snr = height / (noise + 1e-12)

    if method == "Percentile":
        thr = float(np.nanpercentile(snr, percentile))
    else:
        thr = max(_otsu_threshold(snr), float(min_snr))
    mask = snr >= thr

    flat = cube.reshape(-1, cube.shape[2]).astype(float)
    bg_pix = ~mask.reshape(-1)
    if bg_pix.sum() >= 5:
        bg_spectrum = np.median(flat[bg_pix], axis=0)
    else:                                    # almost all cell → use global min env
        bg_spectrum = np.percentile(flat, 5, axis=0)

    return {"area": area, "height": height, "noise": noise, "snr": snr,
            "mask": mask, "background_spectrum": bg_spectrum,
            "threshold": thr, "band": (lo, hi)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class RamanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"BioRaman  —  Raman Hyperspectral Map Analysis for Biophysics  (v{__version__})")
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
        self._roi_smooth_edge  = tk.BooleanVar(value=True)  # anti-aliased ROI boundary on map
        self._roi_mask_inv: np.ndarray | None = None        # inverse (outside) mask
        self._roi_multi   = tk.BooleanVar(value=False)      # accumulate multiple ROIs
        self._roi_count   = 0                               # number of regions in current mask

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
        fm.add_command(label="Open Raman file (WDF/WIP/Parquet/…)   Ctrl+O", command=self.load_file)
        fm.add_command(label="Import Horiba LabSpec map (ASCII/text)…",
                       command=self.import_horiba)
        fm.add_command(label="Load White-Light…",       command=self.load_wl)
        fm.add_separator()
        fm.add_command(label="📦 RAMANMETRIX Dataset (ZIP + metadata)…",
                       command=self.open_metrix_dataset)
        fm.add_separator()
        fm.add_command(label="Save Map…        Ctrl+M", command=self.save_map)
        fm.add_command(label="Save Spectrum…   Ctrl+S", command=self.save_spectrum)
        fm.add_command(label="Save Processed Data…   Ctrl+Shift+S",
                       command=self.save_processed)
        fm.add_command(label="📂 Batch Process Folder…", command=self.open_batch)
        fm.add_command(label="📝 Save Analysis Report (HTML)…",
                       command=self.save_report)
        fm.add_separator()
        fm.add_command(label="💾 Save Session…",  command=self.save_session)
        fm.add_command(label="📤 Load Session…",  command=self.load_session)
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
        pm.add_separator()
        pm.add_command(label="🔁  Reprocess (apply recipe to raw)",
                       command=self.reprocess)
        pm.add_command(label="💾  Export Pipeline (JSON/YAML)…", command=self.save_recipe)
        pm.add_command(label="📥  Load Pipeline / Recipe…",      command=self.load_recipe)
        pm.add_command(label="▶  Run Pipeline on folder…",       command=self.run_pipeline_folder)
        pm.add_separator()
        pm.add_command(label="✓  Quality Control Maps…", command=self.open_qc_map)
        pm.add_command(label="🧫  Cell Mask / Background Rejection…",
                       command=self.open_cell_mask)

        am = tk.Menu(mb, tearoff=0, bg=C["panel"], fg=C["text_hi"],
                     activebackground=C["accent"], activeforeground="white")
        mb.add_cascade(label="Analysis", menu=am)
        am.add_command(label="◈  PCA Analysis…",         command=self.open_pca)
        am.add_command(label="◇  PCA Studio (scalable, large datasets)…",
                       command=self.open_pca_studio)
        am.add_command(label="🧊  3D Volume Viewer…",     command=self.open_3d_viewer)
        am.add_command(label="✨  Publication Volume (Plotly)…",
                       command=self.open_volume_render)
        am.add_separator()
        am.add_command(label="⬡  Cluster Analysis…",     command=self.open_clustering)
        am.add_command(label="⟠  MCR-ALS…",              command=self.open_mcr)
        am.add_command(label="◉  N-FINDR Endmembers…",   command=self.open_nfindr)
        am.add_command(label="🪰  FlyHash (Region + Anomaly)…", command=self.open_flyhash)
        am.add_separator()
        am.add_command(label="⚗  Component Analysis (DCLS/NNLS)…",
                       command=self.open_component_analysis)
        am.add_command(label="▦  Multi-Map Analysis…",  command=self.open_multimap)
        am.add_command(label="◍  Particle Statistics…",
                       command=self.open_particle_stats)
        am.add_command(label="🔎  Library Search (full-spectrum)…",
                       command=self.open_library_search)
        am.add_command(label="⚒  Spectral Tools…",       command=self.open_spectral_tools)
        am.add_command(label="⚖  Group Comparison (stats + mean spectra)…",
                       command=self.open_group_compare)
        am.add_command(label="🧭  Analysis Planner (suggests a workflow)…",
                       command=self.open_planner)
        am.add_command(label="╱  Line Profile (line scan across a map)…",
                       command=self.open_line_profile)

        # ── PCRS (Particle Characterisation & Raman Spectroscopy) ──────────────
        pm = tk.Menu(mb, tearoff=0, bg=C["panel"], fg=C["text_hi"],
                     activebackground=C["accent"], activeforeground="white")
        mb.add_cascade(label="PCRS", menu=pm)
        pm.add_command(label="◍  ParticleFinder + IDFinder…",
                       command=self.open_pcrs_particlefinder)

        hm = tk.Menu(mb, tearoff=0, bg=C["panel"], fg=C["text_hi"],
                     activebackground=C["accent"], activeforeground="white")
        mb.add_cascade(label="Help", menu=hm)
        hm.add_command(label="About", command=lambda: messagebox.showinfo(
            "BioRaman",
            f"BioRaman — Raman Hyperspectral Map Analysis for Biophysics  (v{__version__})\n\n"
            "Created by Akalabya Bissoyi\n"
            "Gibson Group, University of Manchester\n"
            "https://gibsongroupresearch.com/\n"
            "akalabya.bissoyi@manchester.ac.uk · bissoyi.akalabya@gmail.com\n\n"
            "Copyright (c) 2026 Akalabya Bissoyi and the Gibson Group,\n"
            "University of Manchester.\n"
            "Released under the MIT License. This program comes with ABSOLUTELY\n"
            "NO WARRANTY; see the LICENSE file for details.\n\n"
            "Shortcuts:\n"
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
        tk.Label(logo, text="  ◈ BIORAMAN",
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
            ("⊕ Open file",     self.load_file,              "Primary.TButton"),
            ("⚙ Preprocess",   self.open_pp_settings,       "Neutral.TButton"),
            ("🔁 Reprocess",    self.reprocess,              "Neutral.TButton"),
            ("⊞ White Light",   self.load_wl,                "Neutral.TButton"),
            ("◈ PCA",          self.open_pca,               "Neutral.TButton"),
            ("◇ PCA Studio",   self.open_pca_studio,        "Primary.TButton"),
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
            ("🪰 FlyHash",      self.open_flyhash,           "Neutral.TButton"),
            ("⚒ Spectral Tools",self.open_spectral_tools,   "Neutral.TButton"),
            ("✓ QC Maps",       self.open_qc_map,            "Neutral.TButton"),
            ("🧫 Cell Mask",    self.open_cell_mask,         "Neutral.TButton"),
            ("✨ HQ Volume",    self.open_volume_render,     "Neutral.TButton"),
            ("📂 Batch",        self.open_batch,             "Neutral.TButton"),
            ("↓ Save Map",      self.save_map,               "Neutral.TButton"),
            ("↓ Save Spec",     self.save_spectrum,          "Neutral.TButton"),
            ("💾 Save Processed",self.save_processed,         "Neutral.TButton"),
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
        tk.Checkbutton(roi_card, text="Smooth ROI edge (anti-aliased)",
                       variable=self._roi_smooth_edge,
                       command=self._redraw_roi_overlay,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       selectcolor=C["sidebar"],
                       font=("Segoe UI", 11)).pack(anchor="w", padx=4, pady=(0, 2))
        tk.Checkbutton(roi_card,
                       text="Multi-region (add each ROI to selection)",
                       variable=self._roi_multi,
                       bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       selectcolor=C["sidebar"],
                       font=("Segoe UI", 11)).pack(anchor="w", padx=4, pady=(0, 2))
        btn_row = tk.Frame(roi_card, bg=C["sidebar"])
        btn_row.pack(fill="x", padx=4, pady=4)
        ttk.Button(btn_row, text="✎ Draw ROI", style="ROI.TButton",
                   command=self._start_roi).pack(side="left", padx=2)
        ttk.Button(btn_row, text="＋ Add Region", style="ROI.TButton",
                   command=self._add_roi).pack(side="left", padx=2)
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
        self._wl_align_note = tk.Label(
            wl_card, text="", bg=C["sidebar"], fg=C["text_dim"],
            font=("Segoe UI", 9), wraplength=210, justify="left")
        self._wl_align_note.pack(anchor="w", padx=6, pady=(0,4))
        self.sl_wl_alpha = SingleSlider(wl_card, "Overlay opacity",
                                         0, 1, 0.45, color=C["accent2"],
                                         resolution=0.05, command=self.update_map)
        self.sl_wl_alpha.pack(**pad)
        self.sl_wl_bright = SingleSlider(wl_card, "WL brightness",
                                          0.1, 3.0, 1.0, color=C["band_a"],
                                          resolution=0.05, command=self.update_map)
        self.sl_wl_bright.pack(**pad)

        # ── Alignment / registration ──────────────────────────────────────
        # The embedded white-light image is stretched onto the Raman pixel
        # grid. If the file's crop metadata is slightly off (or the image was
        # loaded manually) the overlay drifts. These controls let the user
        # shift/scale the image to register it against the map.
        def _wl_realign(*_):
            self._resize_wl(); self.update_map()
        self.sl_wl_dx = SingleSlider(wl_card, "Align shift X (px)",
                                     -60, 60, 0, color=C["accent2"],
                                     resolution=1, command=_wl_realign)
        self.sl_wl_dx.pack(**pad)
        self.sl_wl_dy = SingleSlider(wl_card, "Align shift Y (px)",
                                     -60, 60, 0, color=C["accent2"],
                                     resolution=1, command=_wl_realign)
        self.sl_wl_dy.pack(**pad)
        self.sl_wl_scale = SingleSlider(wl_card, "Align scale",
                                        0.5, 1.5, 1.0, color=C["band_a"],
                                        resolution=0.01, command=_wl_realign)
        self.sl_wl_scale.pack(**pad)
        ttk.Button(wl_card, text="Reset alignment", style="Neutral.TButton",
                   command=self._reset_wl_align).pack(fill="x", padx=4, pady=(2,6))

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
            except Exception: pass
        nav.update()

        # Per-panel export (the toolbar disk icon only saves the whole figure)
        ttk.Button(tb_frame, text="↓ Save Spectrum",
                   style="Neutral.TButton",
                   command=lambda: self._save_panel(self.ax_spec, "spectrum")
                   ).pack(side="right", padx=4, pady=2)
        ttk.Button(tb_frame, text="↓ Save Map",
                   style="Neutral.TButton",
                   command=lambda: self._save_panel(self.ax_map, "map")
                   ).pack(side="right", padx=4, pady=2)

        self.canvas.mpl_connect("button_press_event",  self._click)
        self.canvas.mpl_connect("motion_notify_event", self._hover)

    def _save_panel(self, ax, tag):
        """Save a single panel (map or spectrum) to PNG/PDF at 600 dpi."""
        path = filedialog.asksaveasfilename(
            title=f"Save {tag} image",
            defaultextension=".png",
            initialfile=f"BioRaman_{tag}",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"),
                       ("SVG vector", "*.svg"), ("TIFF image", "*.tif")])
        if not path:
            return
        try:
            # tight bounding box around just this panel (+ its colourbar/labels)
            ext = ax.get_tightbbox(self.fig.canvas.get_renderer()).transformed(
                self.fig.dpi_scale_trans.inverted()).expanded(1.18, 1.12)
            self.fig.savefig(path, dpi=600, bbox_inches=ext, facecolor="white")
            self._status.set(f"Saved {tag} → {Path(path).name}")
        except Exception as ex:
            messagebox.showerror("Save error", f"Could not save {tag}:\n{ex}",
                                 parent=self)

    def _init_map_ax(self):
        ax = self.ax_map
        ax.set_title("Intensity Map", fontweight="semibold")
        ax.set_xlabel("X  (pixels)", fontsize=10)
        ax.set_ylabel("Y  (pixels)", fontsize=10)
        ax.tick_params(which="both", direction="in", length=3)
        self.im = ax.imshow(np.zeros((10,10)), origin="upper",
                            aspect="equal", interpolation="bicubic", cmap="turbo")
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
        self.bind("<Control-S>", lambda _: self.save_processed())
        self.bind("<Control-Shift-S>", lambda _: self.save_processed())
        self.bind("<Control-m>", lambda _: self.save_map())
        self.bind("<Escape>",    lambda _: self._cancel_roi())

    # ── band slider callbacks ─────────────────────────────────────────────────
    def _on_band_change(self):
        self._rebuild_band_spans()
        self._warn_band_range()
        self.update_map()

    def _warn_band_range(self):
        """Flag, in the status bar, any active band that falls outside (or
        partly outside) the acquired wavenumber range — so out-of-range bands
        like an OH-stretch window beyond the cutoff can't masquerade as data."""
        if self.xdata is None:
            return
        mode = self.mode_var.get() if hasattr(self, "mode_var") else "ratio"
        if mode == "wl":          # white-light only — bands unused
            return
        checks = [("A", self.rs_a.low, self.rs_a.high)]
        # Band B is used by the Ratio and A+B RGB modes.
        if mode in ("ratio", "rgb"):
            checks.append(("B", self.rs_b.low, self.rs_b.high))
        worst = "ok"; msg = ""
        rank = {"ok": 0, "partial": 1, "outside": 2}
        for tag, lo, hi in checks:
            status, m = band_range_status(self.xdata, lo, hi)
            if rank[status] > rank[worst]:
                worst, msg = status, f"Band {tag}: {m}"
        if worst == "outside":
            self._status.set("⛔  " + msg)
        elif worst == "partial":
            self._status.set("⚠  " + msg)

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

    def _add_roi(self):
        """Draw another region and add it to the current selection."""
        self._roi_multi.set(True)
        if self._roi_mask is None:
            self._status.set("No ROI yet — draw the first region.")
        self._start_roi()

    def _cancel_roi(self):
        if self._roi_manager:
            self._roi_manager.deactivate()
            self._roi_manager = None
        self._status.set("ROI cancelled")
        if self._roi_mask is None:
            self._roi_info.config(text="No ROI defined", fg=C["text_dim"])

    def _clear_roi(self):
        self._cancel_roi()
        self._roi_mask = None
        self._roi_count = 0
        self.spec_roi.set_visible(False)
        if self._roi_patch:
            try: self._roi_patch.remove()
            except Exception: pass
            self._roi_patch = None
        self.canvas.draw_idle()
        self._roi_info.config(text="No ROI defined", fg=C["text_dim"])

    def _draw_roi_overlay(self, mask):
        """Paint the ROI overlay on the map. Honors the 'Smooth ROI edge'
        toggle (smooth = anti-aliased boundary, off = hard pixel edge). This is
        display-only; the stored ROI mask and all analysis are unaffected."""
        if mask is None:
            return
        from scipy.ndimage import zoom as _zoom
        if self._roi_patch:
            try: self._roi_patch.remove()
            except Exception: pass
            self._roi_patch = None
        self._roi_mask_inv = ~mask
        Y, X = mask.shape
        rgba = np.zeros((Y * ZOOM, X * ZOOM, 4), dtype=np.float32)
        if self._roi_smooth_edge.get():
            # Anti-aliased boundary: bilinear upsample + light blur → soft 0→1
            # alpha ramp so the edge reads as a smooth curve, not stair-steps.
            soft = _zoom(mask.astype(np.float32), ZOOM, order=1)
            soft = gaussian_filter(soft, sigma=ZOOM * 0.7)
            soft = np.clip(soft, 0.0, 1.0)[..., None]
            if self._roi_reverse_mask.get():
                dark = np.array([0.10, 0.10, 0.10], np.float32)
                warm = np.array([1.00, 0.42, 0.21], np.float32)
                rgba[..., :3] = (1.0 - soft) * dark + soft * warm
                rgba[...,  3] = ((1.0 - soft) * 0.55 + soft * 0.10)[..., 0]
            else:
                rgba[..., 0] = 1.00; rgba[..., 1] = 0.42; rgba[..., 2] = 0.21
                rgba[...,  3] = (soft * 0.28)[..., 0]
            interp = "bilinear"
        else:
            # Hard, pixel-accurate boundary (literal whole-pixel mask).
            mask_z = _zoom(mask.astype(float), ZOOM, order=0) > 0.5
            if self._roi_reverse_mask.get():
                out_z = ~mask_z
                rgba[out_z, :3] = (0.10, 0.10, 0.10); rgba[out_z, 3] = 0.55
                rgba[mask_z, :3] = (1.00, 0.42, 0.21); rgba[mask_z, 3] = 0.10
            else:
                rgba[mask_z, 0] = 1.0; rgba[mask_z, 1] = 0.42
                rgba[mask_z, 2] = 0.21; rgba[mask_z, 3] = 0.28
            interp = "nearest"
        self._roi_patch = self.ax_map.imshow(
            rgba, origin="upper", aspect="equal", interpolation=interp,
            extent=(0, X*ZOOM, Y*ZOOM, 0), zorder=5)
        self.canvas.draw_idle()

    def _redraw_roi_overlay(self):
        """Re-paint the current ROI overlay (e.g. when the smooth-edge toggle is
        switched) using the stored mask, if any."""
        m = getattr(self, "_roi_mask", None)
        if m is not None and np.any(m):
            self._draw_roi_overlay(m)

    def _on_roi_done(self, mask):
        if self._roi_manager:
            self._roi_manager.deactivate()
            self._roi_manager = None
        # Multi-region: union the new region into the running selection.
        if self._roi_multi.get() and self._roi_mask is not None \
                and self._roi_mask.shape == mask.shape:
            mask = self._roi_mask | mask
            self._roi_count += 1
        else:
            self._roi_count = 1
        self._roi_mask = mask          # ← store for ROI Analysis
        n_px = int(mask.sum())
        reg = self._roi_count
        reg_txt = f"{reg} regions" if reg > 1 else "1 region"
        self._roi_info.config(
            text=f"{n_px} pixels selected ({reg_txt})", fg=C["success"])
        self._status.set(f"ROI: {n_px} pixels in {reg_txt} — computing mean spectrum…")

        # Draw filled ROI overlay on map
        self._draw_roi_overlay(mask)

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
            f"ROI: {n_px} pixels in {reg_txt} — '＋ Add Region' to add more, "
            f"or '🔬 ROI Analysis' for full workflow")
        # offer analysis
        self._roi_info.config(
            text=f"✓ {n_px} px in {reg_txt}  —  click 🔬 Analyse",
            fg=C["success"])

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

        b_lo_var = tk.DoubleVar(value=3100)
        b_hi_var = tk.DoubleVar(value=3450)
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
        gs = fig.add_gridspec(2, 3, wspace=0.55, hspace=0.48,
                              left=0.05, right=0.90, top=0.92, bottom=0.08)
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
        attach_panel_saver(win, canvas_r, basename="ROI_analysis")

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
                # Band entirely outside the acquired range — return NaN so the
                # panel renders BLANK rather than a misleading flat-zero field.
                return np.full((Y, X), np.nan)
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
            area = _trapz(corrected, axis=2)
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

            # ── Out-of-range guard ─────────────────────────────────────────────
            # Flag any band that falls outside the acquired wavenumber range so
            # e.g. an ice OH-stretch window (3087-3162) requested on data that
            # stops at ~2500 cm⁻¹ cannot masquerade as a real measurement.
            _bad = []
            for _tag, _lbl, _lo, _hi in (
                    ("A", a_label_var.get(), al, ah),
                    ("B", b_label_var.get(), bl, bh)):
                _st, _msg = band_range_status(self.xdata, _lo, _hi)
                if _st != "ok":
                    _bad.append(f"Band {_tag} ({_lbl}): {_msg}")
            if _bad:
                _outside = any("OUTSIDE" in b for b in _bad)
                _title = ("Band outside data range" if _outside
                          else "Band partly outside data range")
                _body = "\n\n".join(_bad)
                if _outside:
                    messagebox.showwarning(
                        _title, _body + "\n\nThat band's panel will be left "
                        "blank (no data exists there).", parent=win)
                else:
                    if not messagebox.askyesno(
                            _title, _body + "\n\nRun the analysis anyway?",
                            parent=win):
                        return

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
            disp_a = np.ma.masked_where(~mask, sm_a)
            cmap_a = plt.get_cmap(cm_a).copy(); cmap_a.set_bad("#bfbfbf")
            ax_raw_a.set_facecolor("#bfbfbf")
            im_a = ax_raw_a.imshow(disp_a, origin="upper", cmap=cmap_a,
                                   aspect="equal", interpolation="bilinear")
            fig.colorbar(im_a, ax=ax_raw_a, fraction=0.046, pad=0.04, shrink=0.8)
            ax_raw_a.set_title(f"Smoothed  —  {lbl_a}\n({al:.0f}–{ah:.0f} cm⁻¹)",
                               fontsize=8, fontweight="semibold")
            ax_raw_a.set_xticks([]); ax_raw_a.set_yticks([])

            # Panel 2: Smoothed Band B heatmap (full ROI)
            ax_raw_b.clear()
            disp_b = np.ma.masked_where(~mask, sm_b)
            cmap_b = plt.get_cmap(cm_b).copy(); cmap_b.set_bad("#bfbfbf")
            ax_raw_b.set_facecolor("#bfbfbf")
            im_b = ax_raw_b.imshow(disp_b, origin="upper", cmap=cmap_b,
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
            # annotate band area arrows (use the user's actual band labels,
            # not fixed assignments, so any A/B choice is described correctly)
            for lo_wn, hi_wn, col, lbl_s in [
                (al, ah, C["band_a"], f"S_A ({lbl_a})"),
                (bl, bh, C["band_b"], f"S_B ({lbl_b})"),
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

            # Panel 5: per-pixel ratio Band B / Band A within cellular region.
            # Shows where ice is high RELATIVE to cell material, rather than
            # echoing the binary mask shape.
            ax_mask.clear()
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_map = sm_b / sm_a
            ratio_map[~np.isfinite(ratio_map)] = np.nan
            disp_masked = np.ma.masked_where(~binary | ~np.isfinite(ratio_map),
                                             ratio_map)
            cmap_mk = plt.get_cmap(cm_b).copy()
            cmap_mk.set_bad("#bfbfbf")
            # Scale contrast to the WITHIN-CELL ratio values.
            rv = ratio_map[binary & np.isfinite(ratio_map)]
            if rv.size:
                vlo, vhi = np.nanpercentile(rv, [2, 98])
                if not np.isfinite(vlo) or vhi <= vlo:
                    vlo, vhi = None, None
            else:
                vlo, vhi = None, None
            im_mk = ax_mask.imshow(disp_masked, origin="upper", cmap=cmap_mk,
                                   aspect="equal", interpolation="bilinear",
                                   vmin=vlo, vmax=vhi)
            ax_mask.set_facecolor("#bfbfbf")
            fig.colorbar(im_mk, ax=ax_mask, fraction=0.046, pad=0.04, shrink=0.8)
            ax_mask.set_title(
                f"{lbl_b} / {lbl_a}  ratio\nper pixel (cellular region)",
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

    # ── RAMANMETRIX dataset (ZIP + metadata) ──────────────────────────────────
    def open_metrix_dataset(self):
        """Import a RAMANMETRIX-style ZIP (spectra + metadata table) and/or
        generate a metadata template for it.

        Implements the conventions documented at
        https://docs.ramanmetrix.eu/documentation/Data.html
        (Providing Metadata → Generate metadata template / Metadata Table).
        """
        if not HAS_RMETA:
            messagebox.showerror(
                "Unavailable",
                "raman_metadata.py could not be imported.\n"
                "Make sure it sits next to bioraman.py.", parent=self)
            return

        dlg = tk.Toplevel(self)
        dlg.title("RAMANMETRIX Dataset — ZIP + Metadata")
        dlg.geometry("860x620")
        dlg.configure(bg=C["bg"])
        dlg.grab_set()

        hdr = tk.Frame(dlg, bg=C["header"], height=48)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="📦  RAMANMETRIX DATASET",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=12)

        # state
        state = {"zip": None, "spectra": [], "meta_files": [], "resolved": {}}

        top = tk.Frame(dlg, bg=C["bg"]); top.pack(fill="x", padx=16, pady=(12, 4))
        zip_var = tk.StringVar(value="No ZIP selected.")
        tk.Label(top, textvariable=zip_var, bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10), anchor="w").pack(side="left", fill="x", expand=True)

        # options
        opt = tk.Frame(dlg, bg=C["bg"]); opt.pack(fill="x", padx=16, pady=4)
        colset = tk.StringVar(value="full")
        kind   = tk.StringVar(value="auto")
        tk.Label(opt, text="Columns:", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        tk.Radiobutton(opt, text="Full default set", variable=colset, value="full",
                       bg=C["bg"]).grid(row=0, column=1, sticky="w", padx=6)
        tk.Radiobutton(opt, text="Core columns", variable=colset, value="core",
                       bg=C["bg"]).grid(row=0, column=2, sticky="w", padx=6)
        tk.Label(opt, text="Template:", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10, "bold")).grid(row=1, column=0, sticky="w")
        for i, (lbl, val) in enumerate([("Auto", "auto"), ("Long (per file)", "long"),
                                        ("Short (per folder)", "short")]):
            tk.Radiobutton(opt, text=lbl, variable=kind, value=val,
                           bg=C["bg"]).grid(row=1, column=1 + i, sticky="w", padx=6)

        # table
        tbl_frame = tk.Frame(dlg, bg=C["bg"]); tbl_frame.pack(fill="both", expand=True,
                                                              padx=16, pady=8)
        cols = ("file", "type", "batch", "date", "device", "standard", "include")
        tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=14)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=110 if c != "file" else 280, anchor="w")
        vsb = ttk.Scrollbar(tbl_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        status = tk.StringVar(value="Select a ZIP archive to begin.")
        tk.Label(dlg, textvariable=status, bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=16)

        def _refresh_table(mapping):
            tree.delete(*tree.get_children())
            for path in state["spectra"]:
                m = mapping.get(path, {})
                tree.insert("", "end", values=(
                    path,
                    m.get("type", ""), m.get("batch", ""), m.get("date", ""),
                    m.get("device", ""),
                    m.get("standard", ""), m.get("include", "")))

        def _select_zip():
            p = filedialog.askopenfilename(
                title="Select dataset ZIP", filetypes=[("ZIP archive", "*.zip")],
                parent=dlg)
            if not p:
                return
            try:
                state["zip"] = p
                state["spectra"] = rmeta.list_spectra_in_zip(p)
                state["meta_files"] = rmeta.list_metadata_in_zip(p)
            except Exception as exc:
                messagebox.showerror("ZIP error", str(exc), parent=dlg); return
            zip_var.set(Path(p).name)
            # show inferred labels immediately (folder-structure parsing)
            inferred = {sp: rmeta.infer_from_path(sp) for sp in state["spectra"]}
            state["resolved"] = inferred
            _refresh_table(inferred)
            status.set(f"{len(state['spectra'])} spectra · "
                       f"{len(state['meta_files'])} metadata file(s) detected"
                       + (f": {', '.join(state['meta_files'])}"
                          if state["meta_files"] else " — labels inferred from folders."))

        def _apply_metadata():
            if not state["zip"]:
                messagebox.showwarning("No ZIP", "Select a ZIP first.", parent=dlg); return
            if not state["meta_files"]:
                messagebox.showinfo(
                    "No metadata file",
                    "No 'metadata' CSV/XLSX found inside the ZIP.\n"
                    "Showing labels inferred from the folder structure instead.",
                    parent=dlg)
                return
            all_rows = []
            for mf in state["meta_files"]:
                try:
                    all_rows.extend(rmeta.read_metadata_from_zip(state["zip"], mf))
                except Exception as exc:
                    messagebox.showerror("Metadata error",
                                         f"{mf}: {exc}", parent=dlg); return
            resolved = rmeta.match_metadata(state["spectra"], all_rows)
            # merge folder-structure inference as a base layer, metadata wins
            merged = {}
            for sp in state["spectra"]:
                base = dict(rmeta.infer_from_path(sp))
                if sp in resolved:
                    base.update({k: v for k, v in resolved[sp].items()})
                    merged[sp] = base
                # spectra excluded by include=False are dropped from `resolved`
                elif not any((sp == _norm_key(r.get("file", r.get("path", "")))
                              or sp.startswith(_norm_key(r.get("file", r.get("path", "")))))
                             and rmeta._coerce_bool(r.get("include", True)) is False
                             for r in all_rows):
                    merged[sp] = base
            state["resolved"] = merged
            _refresh_table(merged)
            self._metrix_dataset = {"zip": state["zip"], "metadata": merged}
            status.set(f"Applied metadata · {len(merged)} spectra included "
                       f"(of {len(state['spectra'])}).")

        def _norm_key(k):
            return str(k).replace("\\", "/").lstrip("./").lstrip("/") if k else ""

        def _generate_template():
            if not state["spectra"]:
                messagebox.showwarning("No data",
                                       "Select a ZIP with spectra first.", parent=dlg)
                return
            columns = rmeta.DEFAULT_COLUMNS if colset.get() == "full" else rmeta.CORE_COLUMNS
            cols2, rows = rmeta.generate_template(
                state["spectra"], columns=columns, kind=kind.get())
            out = filedialog.asksaveasfilename(
                title="Save metadata template", defaultextension=".csv",
                initialfile="metadata_template.csv",
                filetypes=[("CSV", "*.csv")], parent=dlg)
            if not out:
                return
            rmeta.write_template_csv(cols2, rows, out)
            status.set(f"Template written: {Path(out).name} ({len(rows)} rows).")
            messagebox.showinfo("Template saved",
                                f"Metadata template saved to:\n{out}\n\n"
                                f"{len(rows)} rows · {kind.get()} layout.",
                                parent=dlg)

        def _load_selected():
            sel = tree.focus()
            if not sel:
                messagebox.showinfo("Select a row",
                                    "Pick a spectrum row to load into the viewer.",
                                    parent=dlg); return
            member = tree.item(sel, "values")[0]
            if member.endswith("/") or "*" in member:
                messagebox.showinfo("Not a file",
                                    "This row is a folder pattern, not a single file.",
                                    parent=dlg); return
            try:
                import tempfile, zipfile as _zf
                with _zf.ZipFile(state["zip"]) as z:
                    raw = z.read(member)
                suffix = os.path.splitext(member)[1] or ".txt"
                tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tf.write(raw); tf.close()
                dlg.destroy()
                self._load_path(tf.name, display_name=member)
            except Exception as exc:
                messagebox.showerror("Load error", str(exc), parent=dlg)

        btns = tk.Frame(dlg, bg=C["bg"]); btns.pack(fill="x", padx=16, pady=12)
        ttk.Button(top, text="📂 Select ZIP…", command=_select_zip).pack(side="right")
        ttk.Button(btns, text="🧬 Apply metadata", command=_apply_metadata
                   ).pack(side="left", padx=4)
        ttk.Button(btns, text="📝 Generate template CSV…", command=_generate_template
                   ).pack(side="left", padx=4)
        ttk.Button(btns, text="📈 Load selected spectrum", command=_load_selected
                   ).pack(side="left", padx=4)
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side="right", padx=4)

    def load_file(self):
        path = filedialog.askopenfilename(
            title="Open Raman File (any format)",
            filetypes=SUPPORTED_PATTERNS)
        if not path: return
        self._load_path(path)

    def import_horiba(self):
        """Open a Horiba LabSpec ASCII/text map export using the dedicated
        Horiba parser (Horiba's binary .ngc is proprietary/undocumented; the
        ASCII map and LabSpec 6.3+ HDF5 are the open export paths)."""
        path = filedialog.askopenfilename(
            title="Import Horiba LabSpec map (ASCII/text export)",
            filetypes=[("Horiba text map", "*.txt *.csv *.dat *.asc"),
                       ("All", "*.*")])
        if not path:
            return
        self._load_path(path, reader=_read_horiba_map_text)

    def _load_path(self, path, display_name=None, reader=None):
        """Load a single spectrum/map file by path (reused by the dataset dialog)."""
        label = display_name or Path(path).name

        # Caution for large mapping files: loading + multivariate/3D analyses on a
        # big hyperspectral cube can be slow and memory-hungry. Let the user decide.
        try:
            size_mb = Path(path).stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0.0
        if size_mb >= LARGE_FILE_WARN_MB:
            proceed = messagebox.askyesno(
                "Large file — this may be slow",
                f"“{label}” is {size_mb:.0f} MB.\n\n"
                "Large hyperspectral maps are fully supported, but loading and the "
                "heavier analyses (MCR-ALS, N-FINDR, clustering, 3D rendering) can be "
                "slow and use a lot of memory. The window may appear to freeze while "
                "it computes — this is normal.\n\n"
                "Load this file now?",
                icon="warning", default="yes", parent=self)
            if not proceed:
                self._status.set("Load cancelled.")
                return

        self._status.set(f"Loading  {label}…")
        self._show_progress(True)
        self.progress["value"] = 0
        self.update_idletasks()

        def worker():
          try:
            r = reader(path) if reader is not None else _open_raman_any(path)

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
            # Retain the raw cube so preprocessing can be re-tuned without a reload
            self._raw_spectra = np.asarray(r.spectra, dtype=float)
            self._raw_xdata   = np.asarray(r.xdata,   dtype=float)
            # Retain the true confocal volume (Z×Y×X×W) when present
            self._volume = getattr(r, "_volume", None)
            self._zvals  = getattr(r, "_zvals", None)
            self._px_um  = getattr(r, "_px_um", None)   # µm per pixel (scale bar)
            def cb(f):
                self.after(0, lambda: self.progress.configure(value=f*100))
            proc, report = preprocess_map(r.spectra, params, cb)
            self.after(0, lambda: self._finish_load(r.xdata, proc, report, path, wl_raw))
          except Exception as exc:
            # Surface the failure instead of leaving the UI stuck on "Loading…".
            import traceback
            traceback.print_exc()
            msg = f"{type(exc).__name__}: {exc}"
            def _fail():
                self._show_progress(False)
                self._status.set(f"Failed to load {label}: {msg}")
                messagebox.showerror(
                    "Could not load file",
                    f"BioRaman could not load:\n{label}\n\n{msg}\n\n"
                    "If this is a map file, check that it is a supported format "
                    "(.wdf / .wip / ASCII table). The full traceback was printed "
                    "to the console.",
                    parent=self)
            self.after(0, _fail)

        threading.Thread(target=worker, daemon=True).start()

    def _finish_load(self, xdata, spectra, report, path, wl_raw=None):
        self.xdata   = xdata
        self.spectra = spectra
        self.pp_report = report
        self._loaded_path = path
        self.coords  = (0, 0)
        self._show_progress(False)

        # Store embedded white-light image if available
        if wl_raw is not None:
            try:
                self.wl_raw = wl_raw
                self._wl_from_wdf = True
                if hasattr(self, "_wl_name") and self._wl_name is not None:
                    self._wl_name.config(text="✓ from WDF: " + Path(path).name, fg=C["success"])
                if hasattr(self, "_wl_align_note"):
                    self._wl_align_note.config(
                        text="Auto-registered from stage metadata (img_cropbox). "
                             "Alignment sliders only fine-tune residual drift.",
                        fg=C["text_dim"])
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
        self._wl_from_wdf = False
        self._wl_name.config(text="⚠ manual: " + Path(path).name, fg=C["band_a"])
        if hasattr(self, "_wl_align_note"):
            self._wl_align_note.config(
                text="Manually loaded — no stage metadata, so the image is "
                     "stretched to the scan area and may be offset. Use the "
                     "alignment sliders, or load the .wdf for auto-registration.",
                fg=C["band_a"])
        thumb = img.copy(); thumb.thumbnail((220,130))
        self._wl_ph = ImageTk.PhotoImage(thumb)
        self._wl_thumb.config(image=self._wl_ph)
        self._resize_wl()
        self.update_map()

    def _reset_wl_align(self):
        """Return the white-light overlay to the file's native registration."""
        for sl in ("sl_wl_dx", "sl_wl_dy"):
            if hasattr(self, sl): getattr(self, sl).set(0)
        if hasattr(self, "sl_wl_scale"): self.sl_wl_scale.set(1.0)
        self._resize_wl(); self.update_map()

    def _resize_wl(self):
        if self.wl_raw is None or self.spectra is None: return
        Y,X,_ = self.spectra.shape
        th,tw = Y*ZOOM, X*ZOOM
        # Base: stretch the full WL image onto the Raman pixel grid.
        pil = Image.fromarray((self.wl_raw*255).astype(np.uint8)).resize(
            (tw,th), Image.LANCZOS)
        arr = np.asarray(pil, dtype=np.float32) / 255.0

        # Manual registration (shift X/Y in pixels, scale about the centre)
        # so the user can correct residual deviation from the map.
        s  = float(getattr(self, "sl_wl_scale", None).value) if hasattr(self, "sl_wl_scale") else 1.0
        dx = int(round(getattr(self, "sl_wl_dx", None).value)) if hasattr(self, "sl_wl_dx") else 0
        dy = int(round(getattr(self, "sl_wl_dy", None).value)) if hasattr(self, "sl_wl_dy") else 0

        if abs(s - 1.0) > 1e-3:
            zoomed = zoom(arr, (s, s, 1), order=1)
            zh, zw = zoomed.shape[:2]
            out = np.zeros_like(arr)
            # centre the scaled image back into the original frame (crop or pad)
            oy, ox = (th - zh)//2, (tw - zw)//2
            sy0, sx0 = max(0, -oy), max(0, -ox)
            dy0, dx0 = max(0, oy),  max(0, ox)
            hh = min(zh - sy0, th - dy0); ww = min(zw - sx0, tw - dx0)
            out[dy0:dy0+hh, dx0:dx0+ww] = zoomed[sy0:sy0+hh, sx0:sx0+ww]
            arr = out

        if dx or dy:
            arr = np.roll(arr, (dy, dx), axis=(0, 1))
            # blank the wrapped-around edges so shifted regions read as empty
            if dy > 0:   arr[:dy, :] = 0
            elif dy < 0: arr[dy:, :] = 0
            if dx > 0:   arr[:, :dx] = 0
            elif dx < 0: arr[:, dx:] = 0

        self.wl_resized = arr

    # ── map computation ───────────────────────────────────────────────────────
    def _band_mean(self, lo, hi):
        if self.xdata is None: return np.zeros((1,1))
        mask = (self.xdata >= lo) & (self.xdata <= hi)
        if not mask.any():
            # Band is entirely outside the acquired range — return NaN so the
            # map renders blank instead of a misleading flat-zero field.
            return np.full(self.spectra.shape[:2], np.nan)
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

    def _wl_cover_bbox(self, shape):
        """Bounding box (x0, x1, y0, y1) of the white-light–covered region in
        map-pixel coords, or None if fully covered / no data. Used to auto-crop
        the map view to the covered area after an alignment shift."""
        wl = self.wl_resized
        if wl is None:
            return None
        cov = (wl[..., 3] > 0.0) if wl.shape[-1] >= 4 \
            else np.any(wl[..., :3] > 0.0, axis=-1)
        if cov.all() or not cov.any():
            return None
        h, w = shape
        if cov.shape != (h, w):
            cov = np.asarray(Image.fromarray((cov * 255).astype(np.uint8)).resize(
                (w, h), Image.NEAREST)) > 127
            if cov.all() or not cov.any():
                return None
        rows = np.where(cov.any(axis=1))[0]
        cols = np.where(cov.any(axis=0))[0]
        return int(cols.min()), int(cols.max()) + 1, int(rows.min()), int(rows.max()) + 1

    def _blend_wl(self, base):
        if self.wl_resized is None: return base
        alpha  = self.sl_wl_alpha.value
        bright = self.sl_wl_bright.value
        wl_full = self.wl_resized
        wl = np.clip(wl_full[...,:3]*bright, 0, 1)
        h,w = base.shape[:2]
        wl_r = np.asarray(
            Image.fromarray((wl*255).astype(np.uint8)).resize((w,h), Image.LANCZOS),
            dtype=np.float32) / 255.0
        # Coverage mask: where the (possibly shifted/scaled) white-light image
        # actually has data.  Regions exposed by alignment shifts are blanked to
        # zero (incl. alpha); show the base map there instead of blending in
        # black — that avoids the odd solid-black band in saved figures.
        if wl_full.shape[-1] >= 4:
            cover_src = (wl_full[..., 3] > 0.0)
        else:
            cover_src = np.any(wl_full[..., :3] > 0.0, axis=-1)
        cover = np.asarray(
            Image.fromarray((cover_src * 255).astype(np.uint8)).resize(
                (w, h), Image.NEAREST), dtype=np.float32) / 255.0
        cover = cover[..., None]                        # (h, w, 1)
        if base.ndim == 2:
            norm = Normalize(vmin=base.min(), vmax=base.max())
            base_rgb = plt.get_cmap(self.cmap_var.get())(norm(base))[...,:3]
        else:
            base_rgb = base[...,:3]
        blended = (1 - alpha) * base_rgb + alpha * wl_r
        out = np.where(cover > 0.5, blended, base_rgb)  # exposed → base map
        return np.clip(out, 0, 1)

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
        # Auto-crop the view to the white-light–covered area when an alignment
        # shift has exposed an uncovered strip, so saved figures don't show a
        # blank/black border.  Falls back to the full frame when fully covered.
        bb = self._wl_cover_bbox(data.shape[:2]) if self.wl_resized is not None else None
        if bb is not None:
            x0, x1, y0, y1 = bb
            self.ax_map.set_xlim(x0, x1)
            self.ax_map.set_ylim(y1, y0)          # inverted y (origin="upper")
        else:
            self.ax_map.set_xlim(0, data.shape[1])
            self.ax_map.set_ylim(data.shape[0], 0)
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
        row(c1,"Z-score threshold (10–20)",
            lambda f: ttk.Spinbox(f, from_=4, to=40, increment=0.5,
                                  textvariable=ct_var, width=9))
        cw_var = tk.IntVar(value=p.cosmic_width)
        row(c1,"Max spike width (px)",
            lambda f: ttk.Spinbox(f, from_=1, to=10, increment=1,
                                  textvariable=cw_var, width=9))
        tk.Label(c1, text="  Median-residual outlier rejection: removes narrow (≤width px)\n"
                          "  positive spikes only; higher threshold = safer for sharp bands",
                 bg=C["panel"], fg=C["text_dim"], justify="left",
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
                state="readonly", values=["max","area","snv","vector","none"]))
        tk.Label(c5, text="  area/max: ratio  ·  snv/vector: removes CH-stretch dominance  ·  none: raw",
                 bg=C["panel"], fg=C["text_dim"],
                 font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(0,6))

        # Buttons
        btn_row = tk.Frame(dlg, bg=C["bg"])
        btn_row.pack(fill="x", padx=12, pady=8)

        def _commit():
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

        def apply_close():
            _commit()
            dlg.destroy()

        def apply_reprocess():
            # User-initiated preprocessing: commit the recipe and immediately
            # re-run it on the raw data kept in memory, then show what it did.
            _commit()
            if getattr(self, "_raw_spectra", None) is None:
                messagebox.showinfo(
                    "Nothing loaded yet",
                    "Recipe saved. Load a file to apply it — or reopen this "
                    "dialog after loading to reprocess on demand.", parent=dlg)
                dlg.destroy(); return
            dlg.destroy()
            self.reprocess(show_summary=True)

        ttk.Button(btn_row, text="Apply & Reprocess Now", style="Primary.TButton",
                   command=apply_reprocess).pack(side="right", padx=4)
        ttk.Button(btn_row, text="Apply (next load)", style="Neutral.TButton",
                   command=apply_close).pack(side="right", padx=4)
        ttk.Button(btn_row, text="Cancel", style="Neutral.TButton",
                   command=dlg.destroy).pack(side="right", padx=4)
        tk.Label(btn_row, text="“Reprocess Now” re-runs on the loaded raw data",
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

    def open_pca_studio(self):
        """Launch the scalable PCA Studio window (out-of-core IncrementalPCA for
        very large datasets). If a map is currently loaded, offer to hand the
        loaded spectra over; the user can also add files inside PCA Studio."""
        try:
            from pca_studio import PCAStudio
        except Exception as ex:
            messagebox.showerror(
                "PCA Studio unavailable",
                "Could not import PCA Studio. Make sure pca_studio.py, "
                "pca_core.py and spectra_io.py are in the same folder as "
                f"BioRaman.\n\n{ex}")
            return

        external = None
        if getattr(self, "spectra", None) is not None and \
                getattr(self, "xdata", None) is not None:
            use = messagebox.askyesno(
                "PCA Studio",
                "Send the currently loaded map into PCA Studio?\n\n"
                "Yes — analyse the loaded spectra (you can still add more files).\n"
                "No  — open empty and load files inside PCA Studio.")
            if use:
                cube = np.asarray(self.spectra)
                spectra2d = cube.reshape(-1, cube.shape[-1]) if cube.ndim == 3 else cube
                external = {"spectra": spectra2d,
                            "waves": np.asarray(self.xdata, dtype=float),
                            "group": "BioRaman map"}
        try:
            PCAStudio(master=self, external_data=external)
        except Exception as ex:
            import traceback as _tb
            messagebox.showerror("PCA Studio failed",
                                 f"{ex}\n\n{_tb.format_exc()}")

    # ── FLYHASH ANALYSIS ──────────────────────────────────────────────────────
    def open_flyhash(self):
        """Fruit-fly mushroom-body sparse-hash analysis.

        Builds two maps from the current cube:
          • FlyHash Regions  — fast unsupervised chemical segmentation.
          • FlyHash Anomaly  — highlights rare/unusual pixels (contaminants,
                               defects, trace phases) with no labels needed.
        Results are stored as named maps (same system as Univariate), so the
        usual LUT / colormap / save controls all work on them.
        """
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a WDF file first."); return

        dlg = tk.Toplevel(self)
        dlg.title("FlyHash — Sparse-Code Region & Anomaly Maps")
        dlg.geometry("560x520")
        dlg.configure(bg=C["bg"])
        dlg.grab_set()
        # holder for an optional real PN->KC connectivity matrix
        self._flyhash_connectivity = getattr(self, "_flyhash_connectivity", None)

        hdr = tk.Frame(dlg, bg=C["header"], height=48)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🪰  FLYHASH ANALYSIS",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=12)

        body = tk.Frame(dlg, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=12)

        tk.Label(body,
                 text="Fly-brain sparse hashing: fast region segmentation +\n"
                      "automatic rare-pixel (anomaly) detection.",
                 bg=C["bg"], fg=C["text_dim"], justify="left",
                 font=("Segoe UI", 10)).grid(row=0, column=0, columnspan=2,
                                             sticky="w", pady=(0, 8))

        # Outputs
        do_regions = tk.BooleanVar(value=True)
        do_anomaly = tk.BooleanVar(value=True)
        tk.Checkbutton(body, text="Build chemical region map", variable=do_regions,
                       bg=C["bg"], fg=C["text_hi"], activebackground=C["bg"],
                       selectcolor=C["panel"], font=("Segoe UI", 10)).grid(
                           row=1, column=0, columnspan=2, sticky="w")
        tk.Checkbutton(body, text="Build anomaly (rare-pixel) map", variable=do_anomaly,
                       bg=C["bg"], fg=C["text_hi"], activebackground=C["bg"],
                       selectcolor=C["panel"], font=("Segoe UI", 10)).grid(
                           row=2, column=0, columnspan=2, sticky="w", pady=(0, 6))

        # Number of regions
        tk.Label(body, text="Number of regions:", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=3, column=0, sticky="w", pady=4)
        n_regions = tk.IntVar(value=4)
        ttk.Spinbox(body, from_=2, to=20, increment=1, textvariable=n_regions,
                    width=8).grid(row=3, column=1, sticky="w", padx=8)

        # Hash size
        tk.Label(body, text="Hash size (bits):", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=4, column=0, sticky="w", pady=4)
        n_bits = tk.IntVar(value=256)
        ttk.Spinbox(body, from_=32, to=2048, increment=32, textvariable=n_bits,
                    width=8).grid(row=4, column=1, sticky="w", padx=8)

        # Sparseness
        tk.Label(body, text="Sparseness (% bits on):", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=5, column=0, sticky="w", pady=4)
        sparse_pct = tk.DoubleVar(value=5.0)
        ttk.Spinbox(body, from_=1.0, to=50.0, increment=1.0, textvariable=sparse_pct,
                    width=8).grid(row=5, column=1, sticky="w", padx=8)
        tk.Label(body, text="(fly uses ~5%; lower = sharper, more distinct codes)",
                 bg=C["bg"], fg=C["text_dim"], font=("Segoe UI", 9)).grid(
                     row=6, column=0, columnspan=2, sticky="w")

        # Projection prior: random vs measured fly connectome (research option)
        tk.Label(body, text="Projection:", bg=C["bg"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).grid(row=7, column=0, sticky="w", pady=(8, 2))
        proj_var = tk.StringVar(value="random")
        ttk.Combobox(body, textvariable=proj_var, state="readonly", width=14,
                     values=["random", "connectome"]).grid(
                         row=7, column=1, sticky="w", padx=8, pady=(8, 2))

        conn_lbl = tk.Label(
            body,
            text="random = standard FlyHash · connectome = fly PN→KC wiring prior",
            bg=C["bg"], fg=C["text_dim"], font=("Segoe UI", 9))
        conn_lbl.grid(row=8, column=0, columnspan=2, sticky="w")

        def load_real_conn():
            path = filedialog.askopenfilename(
                title="Load real PN→KC connectivity (CSV: dense or pn,kc,weight)",
                filetypes=[("CSV", "*.csv"), ("All", "*.*")], parent=dlg)
            if not path:
                return
            try:
                raw = pd.read_csv(path)
                if raw.shape[1] == 3:
                    raw.columns = ["pn", "kc", "w"]
                    pn = {v: i for i, v in enumerate(sorted(raw.pn.unique()))}
                    kc = {v: i for i, v in enumerate(sorted(raw.kc.unique()))}
                    M = np.zeros((len(pn), len(kc)))
                    for _, r in raw.iterrows():
                        M[pn[r.pn], kc[r.kc]] = r.w
                else:
                    M = raw.values.astype(float)
                self._flyhash_connectivity = M
                proj_var.set("connectome")
                conn_lbl.config(text=f"✓ real connectome loaded: "
                                     f"{M.shape[0]} glomeruli × {M.shape[1]} KCs",
                                fg=C["success"])
            except Exception as exc:
                messagebox.showerror("Load failed", str(exc), parent=dlg)

        ttk.Button(body, text="Load real PN→KC…", style="Neutral.TButton",
                   command=load_real_conn).grid(row=9, column=0, columnspan=2,
                                                sticky="w", pady=(4, 2))

        status_lbl = tk.Label(body, text="", bg=C["bg"], fg=C["text_dim"],
                              font=("Segoe UI", 11))
        status_lbl.grid(row=10, column=0, columnspan=2, sticky="w", pady=8)

        def run():
            if not (do_regions.get() or do_anomaly.get()):
                messagebox.showwarning("Nothing to do",
                                       "Tick at least one output map.", parent=dlg)
                return
            try:
                status_lbl.config(text="Computing fly-hash…", fg=C["text_dim"])
                dlg.update_idletasks()
                self._show_progress(True)
                _conn = (self._flyhash_connectivity
                         if proj_var.get() == "connectome" else None)
                reg, ano = flyhash_maps(
                    self.spectra,
                    n_bits=n_bits.get(),
                    sparseness=max(0.01, sparse_pct.get() / 100.0),
                    n_regions=n_regions.get(),
                    do_regions=do_regions.get(),
                    do_anomaly=do_anomaly.get(),
                    projection=proj_var.get(),
                    connectivity=_conn,
                )
            except Exception as exc:
                self._show_progress(False)
                messagebox.showerror("FlyHash failed", str(exc), parent=dlg)
                return
            self._show_progress(False)

            shown = None
            if reg is not None:
                self._saved_maps["FlyHash Regions"] = reg.copy()
                shown = "FlyHash Regions"
            if ano is not None:
                self._saved_maps["FlyHash Anomaly"] = ano.copy()
                shown = "FlyHash Anomaly"   # show anomaly last (usually most useful)

            made = ", ".join(n for n, v in
                             [("Regions", reg is not None), ("Anomaly", ano is not None)]
                             if v)
            status_lbl.config(text=f"✓  Saved FlyHash {made}. "
                                   f"Bright = rare pixels.", fg=C["success"])
            if shown:
                self._show_saved_map(shown)
            self._status.set(f"FlyHash maps created ({made}).")

        btn_row = tk.Frame(dlg, bg=C["bg"])
        btn_row.pack(fill="x", padx=16, pady=8)
        ttk.Button(btn_row, text="Run FlyHash", style="Primary.TButton",
                   command=run).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Close", style="Neutral.TButton",
                   command=dlg.destroy).pack(side="left", padx=4)

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

            # ── out-of-range guard ───────────────────────────────────────────
            _lo, _hi = (l1, l1) if t == "intensity_at_point" else \
                (min(l1, l2), max(l1, l2))
            _st, _msg = band_range_status(wn, _lo, _hi)
            if _st == "outside":
                messagebox.showerror("Band outside data range", _msg, parent=dlg)
                return
            if _st == "partial":
                if not messagebox.askyesno(
                        "Band partly outside data range",
                        _msg + "\n\nBuild the map anyway?", parent=dlg):
                    return

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
                arr = _trapz(np.clip(sub - bl, 0, None), axis=2)
                desc = f"Signal to Baseline {min(l1,l2):.0f}–{max(l1,l2):.0f} cm⁻¹"
            else:  # signal_to_axis
                m = (wn >= min(l1, l2)) & (wn <= max(l1, l2))
                if not m.any():
                    messagebox.showwarning("Range", "No data in range.", parent=dlg); return
                arr = _trapz(np.clip(self.spectra[:, :, m], 0, None), axis=2)
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
                              origin="upper", interpolation="bicubic")
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
                return _trapz(y, x, axis=2)
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
            ax2.imshow(arr, origin="upper", interpolation="bicubic", cmap=cmap2, vmin=vmin, vmax=vmax, alpha=alpha2)
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

    # ── CELL MASK / BACKGROUND REJECTION ───────────────────────────────────────
    def open_cell_mask(self):
        """Auto-separate cell pixels from buffer (e.g. PBS) and reject background.

        Builds a CH-stretch (2850–2950 cm⁻¹) SNR map — where cells are bright and
        buffer is nearly silent — thresholds it into a cell mask, estimates the
        background spectrum from the non-cell pixels, and can subtract it from the
        whole cube so downstream analysis sees clean, background-free spectra.
        """
        if self.spectra is None or self.xdata is None:
            messagebox.showwarning("No data", "Load and preprocess a map first.",
                                   parent=self)
            return
        cube = self.spectra
        wn = np.asarray(self.xdata, dtype=float)

        dlg = tk.Toplevel(self)
        dlg.title("Cell Mask / Background Rejection")
        dlg.geometry("1080x720")
        dlg.configure(bg=C["bg"])

        hdr = tk.Frame(dlg, bg=C["header"], height=46)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="🧫  CELL MASK / BACKGROUND REJECTION",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=12)

        ctl = tk.Frame(dlg, bg=C["sidebar"])
        ctl.pack(fill="x", padx=0, pady=0)
        def _sp(parent, label, val, frm, to, inc=1.0, w=7):
            tk.Label(parent, text=label, bg=C["sidebar"], fg=C["text_mid"],
                     font=("Segoe UI", 9)).pack(side="left", padx=(8, 2))
            v = tk.DoubleVar(value=val)
            ttk.Spinbox(parent, from_=frm, to=to, increment=inc,
                        textvariable=v, width=w).pack(side="left")
            return v

        row1 = tk.Frame(ctl, bg=C["sidebar"]); row1.pack(fill="x", pady=(8, 2))
        cell_lo = _sp(row1, "Cell band lo", 2850, 0, 4000, 10)
        cell_hi = _sp(row1, "hi", 2950, 0, 4000, 10)
        noise_lo = _sp(row1, "Noise band lo", 1750, 0, 4000, 10)
        noise_hi = _sp(row1, "hi", 1850, 0, 4000, 10)

        row2 = tk.Frame(ctl, bg=C["sidebar"]); row2.pack(fill="x", pady=(2, 8))
        tk.Label(row2, text="Threshold", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(8, 2))
        method = tk.StringVar(value="Otsu")
        ttk.Combobox(row2, textvariable=method, values=("Otsu", "Percentile"),
                     state="readonly", width=11).pack(side="left")
        min_snr = _sp(row2, "min SNR", 3.0, 0, 50, 0.5, w=6)
        pct = _sp(row2, "percentile", 75.0, 0, 100, 1.0, w=6)
        nan_bg = tk.BooleanVar(value=True)
        tk.Checkbutton(row2, text="blank background pixels (NaN)",
                       variable=nan_bg, bg=C["sidebar"], fg=C["text_mid"],
                       activebackground=C["sidebar"],
                       font=("Segoe UI", 9)).pack(side="left", padx=10)

        fig = plt.figure(figsize=(10.5, 4.6), facecolor="#ffffff")
        axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
        canvas = FigureCanvasTkAgg(fig, master=dlg)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)
        NavigationToolbar2Tk(canvas, dlg).update()

        state = {"res": None}

        def compute():
            try:
                res = cell_mask_analysis(
                    cube, wn,
                    cell_lo=cell_lo.get(), cell_hi=cell_hi.get(),
                    noise_lo=noise_lo.get(), noise_hi=noise_hi.get(),
                    min_snr=min_snr.get(), method=method.get(),
                    percentile=pct.get())
            except Exception as ex:
                messagebox.showerror("Cell mask", str(ex), parent=dlg)
                return
            state["res"] = res
            for ax in axes:
                ax.clear()
            cell_band = res["area"]
            im0 = axes[0].imshow(cell_band, cmap="magma")
            axes[0].set_title(f"Cell-band area\n{res['band'][0]:.0f}–"
                              f"{res['band'][1]:.0f} cm⁻¹", fontsize=9)
            im1 = axes[1].imshow(res["mask"], cmap="Greens")
            frac = 100.0 * res["mask"].mean()
            axes[1].set_title(f"Cell mask  ({frac:.0f}% pixels)\n"
                              f"SNR ≥ {res['threshold']:.1f}", fontsize=9)
            im2 = axes[2].imshow(res["snr"], cmap="viridis")
            axes[2].set_title("Per-pixel SNR", fontsize=9)
            for ax in axes:
                ax.axis("off")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            fig.tight_layout()
            canvas.draw_idle()
            self._status.set(
                f"Cell mask: {frac:.0f}% cell pixels, SNR≥{res['threshold']:.1f}")

        def save_maps():
            res = state["res"]
            if res is None:
                messagebox.showinfo("Cell mask", "Run Compute first.", parent=dlg)
                return
            self._saved_maps["Cell mask"] = res["mask"].astype(float)
            self._saved_maps["Cell SNR"] = res["snr"]
            self._saved_maps["Cell-band area"] = res["area"]
            messagebox.showinfo(
                "Cell mask",
                "Saved 'Cell mask', 'Cell SNR' and 'Cell-band area' to the map "
                "list (use Ratio Map / Save Map).", parent=dlg)

        def apply_bgsub():
            res = state["res"]
            if res is None:
                messagebox.showinfo("Cell mask", "Run Compute first.", parent=dlg)
                return
            if not messagebox.askyesno(
                "Background subtraction",
                "Subtract the estimated background (buffer) spectrum from every "
                "pixel of the current map?\n\nThe raw cube is backed up and can be "
                "restored with Preprocessing → Reprocess.", parent=dlg):
                return
            if not hasattr(self, "_spectra_prebg") or self._spectra_prebg is None:
                self._spectra_prebg = self.spectra.copy()
            cleaned = self.spectra.astype(float) - res["background_spectrum"][None, None, :]
            if nan_bg.get():
                cleaned[~res["mask"]] = np.nan
            self.spectra = cleaned
            self._status.set("Background subtracted"
                             + (" · background pixels blanked" if nan_bg.get() else ""))
            messagebox.showinfo(
                "Background subtraction",
                "Done. Re-run PCA / clustering to use the cleaned, cell-only "
                "spectra. (Preprocessing → Reprocess restores the raw cube.)",
                parent=dlg)

        bar = tk.Frame(dlg, bg=C["bg"]); bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="▶  Compute", style="Primary.TButton",
                   command=compute).pack(side="left", padx=4)
        ttk.Button(bar, text="💾  Save mask + SNR to maps", style="Neutral.TButton",
                   command=save_maps).pack(side="left", padx=4)
        ttk.Button(bar, text="🧹  Apply background subtraction",
                   style="Neutral.TButton", command=apply_bgsub).pack(side="left", padx=4)
        ttk.Button(bar, text="Close", style="Neutral.TButton",
                   command=dlg.destroy).pack(side="right", padx=4)

        compute()

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
            return _trapz(np.clip(sub - bl, 0, None), axis=2)

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
        arr = np.asarray(self.im.get_array())
        # Match the on-screen auto-crop: if an alignment shift exposed an
        # uncovered strip, crop the saved array to the white-light area too.
        if self.wl_resized is not None:
            bb = self._wl_cover_bbox(arr.shape[:2])
            if bb is not None:
                x0, x1, y0, y1 = bb
                arr = arr[y0:y1, x0:x1]
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

    def save_processed(self):
        """Export the *preprocessed* spectral cube (baseline-corrected, smoothed,
        normalised, cosmic-ray-cleaned) together with the wavenumber axis.

        Re-loadable output formats:
          • .npz  — compressed NumPy archive (xdata, spectra cube, report);
                    recommended, lossless, fast to reload.
          • .txt / .csv / .dpt — ASCII table.  Maps are written in the Renishaw
                    long format (#X #Y #Wave #Intensity) so they round-trip back
                    through the built-in text reader; single spectra are written
                    as two columns (wavenumber, intensity).
          • .h5   — HDF5 dataset (requires h5py).
          • .mat  — MATLAB v5 file (requires SciPy).

        Note: the original instrument container (.wdf / .wip / .spc) is a
        proprietary binary format and cannot be rewritten; the processed data is
        therefore exported to the open formats above.
        """
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a file first."); return

        stem = "processed"
        src = getattr(self, "_loaded_path", None)
        if src:
            stem = Path(src).stem + "_processed"
        path = filedialog.asksaveasfilename(
            title="Save Preprocessed Data",
            initialfile=stem,
            defaultextension=".npz",
            filetypes=[("NumPy archive", "*.npz"),
                       ("Parquet cube", "*.parquet"),
                       ("Feather cube", "*.feather"),
                       ("HDF5", "*.h5"),
                       ("CSV", "*.csv"),
                       ("Text", "*.txt"),
                       ("DPT", "*.dpt"),
                       ("MATLAB", "*.mat")])
        if not path:
            return

        cube = np.asarray(self.spectra, dtype=float)   # Y × X × W
        Y, X, W = cube.shape
        report = getattr(self, "pp_report", {}) or {}
        try:
            write_cube(path, cube, self.xdata, report, source=str(src or ""))
        except Exception as exc:
            messagebox.showerror("Save failed",
                                 f"Could not save processed data:\n{exc}")
            return

        self._status.set(
            f"Processed data saved  →  {Path(path).name}  "
            f"({X}×{Y}×{W})")

    # ── v13: recipes, reprocess, QC, batch, report, session ────────────────────
    def save_recipe(self):
        """Export the current processing pipeline to a JSON or YAML file
        (self-describing + re-runnable on a folder via Run Pipeline)."""
        path = filedialog.asksaveasfilename(
            title="Export Pipeline (JSON / YAML)",
            initialfile="pipeline", defaultextension=".json",
            filetypes=[("JSON pipeline", "*.json"),
                       ("YAML pipeline", "*.yaml"),
                       ("YAML pipeline", "*.yml")])
        if not path:
            return
        try:
            save_pipeline_file(path, self.pp_params,
                               source=getattr(self, "_loaded_path", ""))
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc)); return
        self._status.set(f"Pipeline exported  →  {Path(path).name}")

    def load_recipe(self):
        """Load a pipeline/recipe (JSON or YAML) and offer to reapply."""
        path = filedialog.askopenfilename(
            title="Load Pipeline / Recipe",
            filetypes=[("Pipeline / recipe", "*.json *.yaml *.yml"),
                       ("All", "*.*")])
        if not path:
            return
        try:
            self.pp_params = load_pipeline_file(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc)); return
        self._status.set(f"Pipeline loaded  →  {Path(path).name}")
        if getattr(self, "_raw_spectra", None) is not None and messagebox.askyesno(
                "Reprocess", "Re-apply the loaded pipeline to the current data now?"):
            self.reprocess()

    def run_pipeline_folder(self):
        """Replay an exported pipeline file over every map in a chosen folder."""
        pipe = filedialog.askopenfilename(
            title="Choose a pipeline file (JSON / YAML)",
            filetypes=[("Pipeline / recipe", "*.json *.yaml *.yml"),
                       ("All", "*.*")])
        if not pipe:
            return
        in_dir = filedialog.askdirectory(title="Folder of maps to process")
        if not in_dir:
            return
        out_dir = filedialog.askdirectory(title="Output folder for results")
        if not out_dir:
            return
        self._status.set("Running pipeline…")
        def worker():
            try:
                n_ok, n_fail = run_pipeline(pipe, in_dir, out_dir,
                                            log=lambda m: None)
                msg = f"Pipeline done: {n_ok} ok, {n_fail} failed → {Path(out_dir).name}"
            except Exception as exc:
                msg = f"Pipeline failed: {exc}"
            self.after(0, lambda: self._status.set(msg))
        threading.Thread(target=worker, daemon=True).start()

    def reprocess(self, show_summary=True):
        """Re-run preprocessing on the retained raw cube (no disk reload).

        This is the user-initiated preprocessing entry point: the recipe in
        Preprocessing → Settings is applied on demand to the raw data that was
        kept in memory at load time, so the user can tune and re-run the
        pipeline interactively without re-opening the file. When
        ``show_summary`` is true a short report of what the run actually did
        (cosmic points removed, any baseline failures) is shown afterwards so
        aggressive settings cannot silently alter the spectra.
        """
        raw = getattr(self, "_raw_spectra", None)
        if raw is None:
            messagebox.showwarning(
                "No raw data",
                "Load a file first (raw data is kept only for files loaded "
                "in this session)."); return
        self._status.set("Reprocessing…")
        self._show_progress(True); self.progress["value"] = 0
        self.update_idletasks()

        def worker():
            params = self.pp_params
            def cb(f):
                self.after(0, lambda: self.progress.configure(value=f*100))
            proc, report = preprocess_map(raw, params, cb)
            xdata = getattr(self, "_raw_xdata", self.xdata)
            path = getattr(self, "_loaded_path", "reprocessed")
            def _done():
                self._finish_load(xdata, proc, report, path, None)
                if show_summary:
                    self._show_reprocess_summary(report)
            self.after(0, _done)
        threading.Thread(target=worker, daemon=True).start()

    def _show_reprocess_summary(self, report):
        """Concise, user-facing summary of what the preprocessing run did —
        the 'report/preview removed points' safety net for cosmic removal."""
        total = report.get("total_spectra", 0) or 1
        cn    = report.get("cosmic_removed", 0)
        cpts  = report.get("cosmic_points", 0)
        bfail = report.get("baseline_failures", 0)
        lines = [
            f"Spectra processed: {report.get('total_spectra', '?')}",
            f"Cosmic spikes removed: {cn} spectra ({cpts} points interpolated)",
            f"Baseline: {report.get('baseline_method', '—')}",
            f"Normalisation: {report.get('normalisation', '—')}",
        ]
        warn = []
        if cpts > 0.05 * total * max(1, report.get("spectral_points", 1)) / 100:
            pass  # placeholder; fraction check below is the meaningful one
        frac_spectra = cn / total
        if frac_spectra > 0.5:
            warn.append(
                f"Cosmic removal touched {frac_spectra*100:.0f}% of spectra — if "
                "your bands are very sharp, raise the Z-score threshold in "
                "Preprocessing → Settings or disable cosmic removal.")
        if bfail > 0:
            warn.append(f"Baseline correction FAILED on {bfail} spectrum(s); "
                        "those were left uncorrected.")
        msg = "\n".join(lines)
        if warn:
            messagebox.showwarning("Preprocessing applied",
                                   msg + "\n\n⚠ " + "\n\n⚠ ".join(warn))
        else:
            messagebox.showinfo("Preprocessing applied", msg)

    def open_qc_map(self):
        """Show per-pixel quality-control maps (SNR, total intensity,
        saturation, cosmic-affected pixels)."""
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a file first."); return
        QCMapWindow(self)

    def open_volume_render(self):
        """Publication-quality translucent volume rendering (Plotly) of a
        depth-resolved confocal Raman dataset."""
        vol = getattr(self, "_volume", None)
        if vol is None:
            messagebox.showinfo(
                "No 3-D volume",
                "This file is a single 2-D map, not a depth-resolved volume, "
                "so there is no real Z axis to render.\n\n"
                "The publication volume renderer needs a confocal volume "
                "(a file with multiple Z planes) or a loaded Z-stack.")
            return
        VolumeRenderWindow(self, vol, getattr(self, "_zvals", None), self.xdata)

    def open_batch(self):
        """Open the batch-processing dialog (apply current recipe to a folder)."""
        BatchWindow(self)

    def open_multimap(self):
        """Multi-map analysis: compare average spectra across maps and build
        band-ratio image series with a normalised quantification trend."""
        MultiMapWindow(self)

    def open_library_search(self):
        """Full-spectrum library search against a user-supplied reference
        library (RRUFF / Raman Open Database / SLoPP / any spectra folder)."""
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a file first."); return
        LibrarySearchWindow(self)

    def open_component_analysis(self):
        """Supervised component analysis (DCLS/NNLS) → concentration maps,
        lack-of-fit and concentration estimates (WiRE-style)."""
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a file first."); return
        ComponentAnalysisWindow(self)

    def open_particle_stats(self):
        """Particle / domain statistics on the current single-band map."""
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a file first."); return
        try:
            arr = np.asarray(self.im.get_array(), dtype=float)
        except Exception:
            arr = None
        if arr is None or arr.ndim != 2:
            messagebox.showinfo(
                "Need a single-band map",
                "Particle statistics needs a single-band intensity map. Build "
                "one first (Univariate, Ratio, or a Component Analysis "
                "concentration map), then try again.")
            return
        ParticleStatsWindow(self, arr, "current map")

    def open_pcrs_particlefinder(self):
        """PCRS ParticleFinder + IDFinder: locate particles on the current
        single-band map, extract each particle's mean Raman spectrum from the
        full hyperspectral cube, and (optionally) identify every particle
        against a user-supplied reference library — all in one panel."""
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a file first."); return
        if np.asarray(self.spectra).ndim != 3:
            messagebox.showinfo(
                "Need a hyperspectral map",
                "ParticleFinder needs a 2-D hyperspectral map (a cube of "
                "spectra), so it can extract a spectrum for each particle.")
            return
        try:
            arr = np.asarray(self.im.get_array(), dtype=float)
        except Exception:
            arr = None
        if arr is None or arr.ndim != 2:
            messagebox.showinfo(
                "Need a single-band map",
                "ParticleFinder locates particles on a single-band intensity "
                "map. Build one first (Univariate, Ratio, or a Component "
                "Analysis concentration map), then try again.")
            return
        ParticleFinderWindow(self, arr, "current map")

    def save_report(self):
        """Write a self-contained HTML report of the current map + recipe."""
        if self.spectra is None:
            messagebox.showwarning("No data", "Load a file first."); return
        stem = "report"
        src = getattr(self, "_loaded_path", None)
        if src:
            stem = Path(src).stem + "_report"
        path = filedialog.asksaveasfilename(
            title="Save Analysis Report", initialfile=stem,
            defaultextension=".html", filetypes=[("HTML report", "*.html")])
        if not path:
            return
        try:
            self._write_html_report(path)
        except Exception as exc:
            messagebox.showerror("Report failed", str(exc)); return
        self._status.set(f"Report saved  →  {Path(path).name}")

    def _fig_to_b64(self, fig):
        import io, base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _write_html_report(self, path):
        import html as _html, datetime
        cube = np.asarray(self.spectra, dtype=float)
        Y, X, W = cube.shape
        x = np.asarray(self.xdata, dtype=float)

        # map image (mean intensity)
        fig1 = plt.figure(figsize=(5, 4))
        ax1 = fig1.add_subplot(111)
        ax1.imshow(cube.mean(axis=2), cmap=self.cmap_var.get(), origin="upper")
        ax1.set_title("Mean intensity map"); ax1.set_xlabel("X (px)"); ax1.set_ylabel("Y (px)")
        img_map = self._fig_to_b64(fig1)

        # mean spectrum +/- std
        fig2 = plt.figure(figsize=(6, 3.2))
        ax2 = fig2.add_subplot(111)
        flat = cube.reshape(-1, W)
        mu, sd = flat.mean(0), flat.std(0)
        ax2.plot(x, mu, color="#2b6cff", lw=1.2)
        ax2.fill_between(x, mu - sd, mu + sd, color="#2b6cff", alpha=0.2)
        ax2.set_xlabel("Raman shift (cm⁻¹)"); ax2.set_ylabel("Intensity (a.u.)")
        ax2.set_title("Mean spectrum ± 1 SD")
        img_spec = self._fig_to_b64(fig2)

        rec = recipe_to_dict(self.pp_params)
        rec_rows = "".join(
            f"<tr><td>{_html.escape(str(k))}</td>"
            f"<td>{_html.escape(str(v))}</td></tr>" for k, v in rec.items())
        report = getattr(self, "pp_report", {}) or {}
        rep_rows = "".join(
            f"<tr><td>{_html.escape(str(k))}</td>"
            f"<td>{_html.escape(str(v))}</td></tr>" for k, v in report.items())
        src = _html.escape(str(getattr(self, "_loaded_path", "—")))
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>BioRaman report</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#1b2333;max-width:900px}}
h1{{color:#2b6cff}} h2{{border-bottom:1px solid #dde;padding-bottom:4px;margin-top:28px}}
table{{border-collapse:collapse;margin:8px 0}} td{{border:1px solid #dde;padding:4px 10px}}
td:first-child{{color:#556;font-weight:600}} img{{max-width:100%;border:1px solid #eee;border-radius:6px}}
.small{{color:#889;font-size:13px}}</style></head><body>
<h1>BioRaman Analysis Report</h1>
<p class="small">Generated {now} · BioRaman v{__version__}</p>
<table><tr><td>Source file</td><td>{src}</td></tr>
<tr><td>Map size</td><td>{X} × {Y} pixels</td></tr>
<tr><td>Spectral points</td><td>{W}</td></tr>
<tr><td>Range</td><td>{x[0]:.0f} – {x[-1]:.0f} cm⁻¹</td></tr></table>
<h2>Mean intensity map</h2><img src="data:image/png;base64,{img_map}">
<h2>Mean spectrum</h2><img src="data:image/png;base64,{img_spec}">
<h2>Preprocessing recipe</h2><table>{rec_rows}</table>
<h2>Processing log</h2><table>{rep_rows}</table>
</body></html>"""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(doc)

    def save_session(self):
        """Save a lightweight session (source path + recipe + view settings)."""
        import json
        path = filedialog.asksaveasfilename(
            title="Save Session", initialfile="session",
            defaultextension=".bioses", filetypes=[("BioRaman session", "*.bioses")])
        if not path:
            return
        sess = {
            "bioraman_session_version": 1,
            "source": str(getattr(self, "_loaded_path", "") or ""),
            "recipe": recipe_to_dict(self.pp_params),
            "cmap": self.cmap_var.get(),
            "normalise_view": bool(self._norm_var.get()),
            "show_peaks": bool(self._show_peaks.get()),
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(sess, fh, indent=2)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc)); return
        self._status.set(f"Session saved  →  {Path(path).name}")

    def load_session(self):
        """Restore a session: apply recipe + view settings and reload the file."""
        import json
        path = filedialog.askopenfilename(
            title="Load Session",
            filetypes=[("BioRaman session", "*.bioses"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                sess = json.load(fh)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc)); return
        self.pp_params = recipe_from_dict(sess.get("recipe", {}))
        try:
            if sess.get("cmap"): self.cmap_var.set(sess["cmap"])
            self._norm_var.set(bool(sess.get("normalise_view", True)))
            self._show_peaks.set(bool(sess.get("show_peaks", False)))
        except Exception:
            pass
        src = sess.get("source", "")
        if src and os.path.exists(src):
            self._load_path(src)
            self._status.set(f"Session restored  →  {Path(path).name}")
        else:
            messagebox.showinfo(
                "Session loaded",
                "Recipe and view settings restored. The original data file "
                "was not found — load it manually to continue.")

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

    def open_group_compare(self):
        """Multi-file group comparison: per-cell custom band ratio with
        Mann-Whitney / Hodges-Lehmann / Cliff's delta + group mean-spectra
        overlay. Implemented in the companion module bioraman_group_compare.py."""
        try:
            import bioraman_group_compare as bgc
        except Exception as e:
            messagebox.showerror(
                "Group Comparison unavailable",
                "Could not load bioraman_group_compare.py.\n\n"
                "Ensure the module sits next to bioraman.py and that scikit-image "
                "is installed.\n\nDetails: %s" % e)
            return
        bgc.open_group_compare(self)

    def open_planner(self):
        """Offline analysis planner: asks about the sample/goal and suggests a
        BioRaman workflow from a built-in knowledge base (no external search)."""
        try:
            import bioraman_planner as bp
        except Exception as e:
            messagebox.showerror(
                "Analysis Planner unavailable",
                "Could not load bioraman_planner.py (keep it next to bioraman.py).\n\n"
                "Details: %s" % e)
            return
        bp.open_planner(self)

    def open_line_profile(self):
        """Draw a line across the current map and plot band intensity (one or two
        bands, dual axis) vs distance — a 'line scan'. See bioraman_lineprofile.py."""
        if self.spectra is None or np.asarray(self.spectra).ndim != 3:
            messagebox.showwarning("No map", "Load a Raman map first (a 2-D map, not a single spectrum).")
            return
        try:
            import bioraman_lineprofile as blp
        except Exception as e:
            messagebox.showerror(
                "Line Profile unavailable",
                "Could not load bioraman_lineprofile.py (keep it next to bioraman.py).\n\n"
                "Details: %s" % e)
            return
        blp.open_line_profile(self)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED: export each panel of a multi-panel figure as its own file
# ─────────────────────────────────────────────────────────────────────────────
def save_panels_dialog(win, fig, panels, base="figure"):
    """Save each (axes, name) in *panels* as its own standalone image by
    cropping the live figure to that axes' tight bounding box. Prompts for a
    folder and (optionally) a vector PDF copy. Returns the list of files."""
    out_dir = filedialog.askdirectory(
        title="Choose a folder for the individual panels", parent=win)
    if not out_dir:
        return []
    save_pdf = messagebox.askyesno(
        "Vector copy?",
        "Also save a vector PDF of each panel (recommended for journals)?\n\n"
        "Yes → PNG (600 dpi) + PDF      No → PNG only",
        parent=win)

    def _sanitize(s):
        s = str(s).strip().replace(" ", "_")
        return "".join(ch for ch in s if ch.isalnum() or ch in "-_[]()")[:40] or "panel"

    try:
        fig.canvas.draw()
    except Exception:
        pass
    renderer = fig.canvas.get_renderer()
    exported = []
    for ax, name in panels:
        if ax is None or not ax.get_visible():
            continue
        try:
            bbox = ax.get_tightbbox(renderer).transformed(
                fig.dpi_scale_trans.inverted())
        except Exception:
            bbox = ax.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted())
        bbox = bbox.expanded(1.06, 1.10)
        tag = _sanitize(name)
        png = Path(out_dir) / f"{base}_{tag}.png"
        fig.savefig(png, dpi=600, bbox_inches=bbox, facecolor="white")
        exported.append(png.name)
        if save_pdf:
            pdf = Path(out_dir) / f"{base}_{tag}.pdf"
            fig.savefig(pdf, bbox_inches=bbox, facecolor="white")
            exported.append(pdf.name)
    messagebox.showinfo(
        "Panels saved",
        f"Saved {len(exported)} files to:\n{out_dir}", parent=win)
    return exported


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER ANALYSIS WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class ClusteringWindow(PanelSaveMixin, tk.Toplevel):
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
        left = make_scroll_sidebar(self, 270)

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
        ttk.Button(left, text="↓ Save Figure (whole page)", style="N.TButton",
                   command=self._save_fig).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="↓ Save Individual Panels…", style="N.TButton",
                   command=self._save_panels).pack(fill="x", padx=12, pady=4)
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
        self._enable_panel_ticks(self._canvas, basename="Clustering")

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
            # Cluster-quality metric (mean silhouette: higher = better separation)
            try:
                from sklearn.metrics import silhouette_score
                self._silhouette = (
                    float(silhouette_score(mat, sub_labels))
                    if k > 1 and len(set(sub_labels)) > 1 else None)
            except Exception:
                self._silhouette = None
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

        sil = getattr(self, "_silhouette", None)
        if sil is not None:
            self._status.config(
                text=f"Silhouette score: {sil:.3f}  (range −1→1, higher = "
                     f"better-separated clusters)", fg=C["success"])

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

    def _save_panels(self):
        if self._labels is None:
            messagebox.showwarning("No results", "Run clustering first.",
                                   parent=self)
            return
        base = Path(getattr(self.master, "_last_wdf_path", "Cluster")).stem or "Cluster"
        panels = [
            (self._ax_map,  "cluster_map"),
            (self._ax_spec, "mean_spectra"),
            (self._ax_bar,  "pixel_counts"),
        ]
        n = save_panels_dialog(self, self._fig, panels, base=base)
        if n:
            self._status.config(
                text=f"Saved {len(n)} individual panels.", fg=C["success"])

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

# ---------------------------------------------------------------------------
# Optional PERSONAL overlay (private; never committed to GitHub, see .gitignore).
# If a local `ramanbiolib_personal.py` sits next to this script, its
# PERSONAL_BANDS list [(wavenumber, "label"), ...] is merged into the active
# biology library, and any PERSONAL_LIBRARIES dict is added as extra selectable
# libraries.  This lets a private build carry additional reference bands (e.g.
# MicrobioRaman-derived or literature-estimated) on top of the public set,
# without changing anything in the public repository.  No-ops if absent.
# ---------------------------------------------------------------------------
try:
    import os as _os, importlib.util as _ilu
    try:
        _here = _os.path.dirname(_os.path.abspath(__file__))
    except NameError:
        _here = _os.getcwd()
    _pp = _os.path.join(_here, "ramanbiolib_personal.py")
    if _os.path.exists(_pp):
        _spec = _ilu.spec_from_file_location("ramanbiolib_personal", _pp)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _extra = list(getattr(_mod, "PERSONAL_BANDS", []) or [])
        if _extra:
            RAMAN_BANDS = RAMAN_BANDS + _extra
            BAND_LIBRARIES["Biology / Cryopreservation"] = RAMAN_BANDS
        _extra_libs = dict(getattr(_mod, "PERSONAL_LIBRARIES", {}) or {})
        for _name, _bands in _extra_libs.items():
            BAND_LIBRARIES[_name] = list(_bands)
        print("[BioRaman] Personal overlay loaded: %d extra bands, %d extra libraries"
              % (len(_extra), len(_extra_libs)))
except Exception as _e:
    print("[BioRaman] Personal overlay not loaded (%s)" % _e)


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
        self._panel_vars = {}     # tag -> BooleanVar (ticked = save)
        self._panel_items = []    # [(ax, tag)] registered each _draw
        self._panel_checks = []   # overlay Checkbutton widgets
        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=C["header"], height=44)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="⟠  MCR-ALS  —  MULTIVARIATE CURVE RESOLUTION",
                 bg=C["header"], fg="white",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=16, pady=10)

        left = make_scroll_sidebar(self, 270)

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
        self._despike = tk.BooleanVar(value=True)
        tk.Checkbutton(left, text="Cosmic-ray despike (pre-MCR)",
                       variable=self._despike, bg=C["sidebar"], fg=C["text_mid"],
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
        ttk.Button(left, text="📉 Lack-of-fit vs components", style="N.TButton",
                   command=self._lof_scan).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="🧭 Flatness / tilt check", style="N.TButton",
                   command=self._flatness_check).pack(fill="x", padx=12, pady=4)
        ds = tk.Frame(left, bg=C["sidebar"]); ds.pack(fill="x", padx=12, pady=4)
        tk.Label(ds, text="Display σ", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left")
        self._dsig = tk.DoubleVar(value=0.8)   # display smoothing; set to 0 for faithful pixels
        ttk.Spinbox(ds, from_=0.0, to=4.0, increment=0.2, width=5,
                    textvariable=self._dsig,
                    command=lambda: getattr(self, "_C", None) is not None
                    and self._draw()).pack(side="left", padx=6)
        self._lut = {}
        _build_lut_panel(self, left, lambda: getattr(self, "_C", None) is not None)

        SectionDiv(left, "EXPORT").pack(fill="x")
        ttk.Button(left, text="↓ Save Figure (whole page)", style="N.TButton",
                   command=self._save_fig).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="↓ Save Individual Panels…  (all)",
                   style="N.TButton",
                   command=self._save_panels).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="☑ Save Ticked Panels…", style="N.TButton",
                   command=self._save_ticked).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="↓ Save Pure Spectra (.csv)", style="N.TButton",
                   command=self._save_spectra).pack(fill="x", padx=12, pady=4)

        right = tk.Frame(self, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._fig = plt.figure(figsize=(12, 7.6), facecolor="#ffffff")
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas.get_tk_widget().bind(
            "<Configure>", lambda _e: self._place_panel_ticks())
        NavigationToolbar2Tk(self._canvas, right).update()

    def _place_panel_ticks(self):
        """Overlay a small checkbox at the top-left of each live panel so the
        user can tick which figures to save."""
        for cb in self._panel_checks:
            cb.destroy()
        self._panel_checks = []
        if not self._panel_items:
            return
        w = self._canvas.get_tk_widget()
        fw, fh = w.winfo_width(), w.winfo_height()
        if fw <= 1 or fh <= 1:
            return
        for ax, tag in self._panel_items:
            var = self._panel_vars.setdefault(tag, tk.BooleanVar(value=True))
            pos = ax.get_position()
            x = int(pos.x0 * fw) + 2
            y = int((1.0 - pos.y1) * fh) + 2
            cb = tk.Checkbutton(w, variable=var, bg="white",
                                activebackground="white", bd=0,
                                highlightthickness=0, padx=0, pady=0)
            cb.place(x=x, y=y)
            self._panel_checks.append(cb)

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
            n_spikes = 0
            if self._despike.get():
                cube, n_spikes = self._despike_cube(cube)
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
            self._n_spikes = n_spikes
            self.after(0, self._draw)
        except Exception as ex:
            self.after(0, lambda ex=ex: messagebox.showerror("Error", str(ex),
                                                        parent=self))
        finally:
            self.after(0, self._prog.stop)

    @staticmethod
    def _despike_cube(cube, win=5, thresh=7.0):
        """Remove cosmic-ray spikes along the spectral axis.

        A spike is a narrow, strongly positive outlier relative to a local
        median. Detection uses a robust (MAD-based) modified z-score on the
        residual to the median-filtered spectrum; flagged points are replaced
        by the median value. Returns (clean_cube, n_points_replaced)."""
        from scipy.ndimage import median_filter
        med = median_filter(cube, size=(1, 1, win))
        diff = cube - med
        # Noise scale from NONZERO residuals: in flat/smooth regions the median
        # filter reproduces values exactly, so a plain median(|diff|) collapses
        # to 0 and would disable detection.
        absd = np.abs(diff)
        nz = absd[absd > 0]
        if nz.size == 0:
            return cube, 0
        mad = float(np.median(nz))
        if not np.isfinite(mad) or mad <= 0:
            return cube, 0
        # Only positive excursions → cosmic rays, not real absorption dips.
        spikes = diff > thresh * 1.4826 * mad
        if not spikes.any():
            return cube, 0
        out = cube.copy()
        out[spikes] = med[spikes]
        return out, int(spikes.sum())

    def _lof_scan(self):
        """Scan k = 1..K and plot lack-of-fit %, to guide component choice."""
        if not HAS_SKL:
            messagebox.showerror("Missing library",
                "scikit-learn not installed.\npip install scikit-learn",
                parent=self)
            return
        self._prog.start(12)
        self._status.config(text="Scanning component counts…", fg=C["text_mid"])
        threading.Thread(target=self._lof_worker, daemon=True).start()

    def _lof_worker(self):
        try:
            from sklearn.decomposition import NMF
            lo = self._vars["wn_lo"].get(); hi = self._vars["wn_hi"].get()
            mask_w = (self.xdata >= lo) & (self.xdata <= hi)
            dt = np.float32 if self._fast32.get() else np.float64
            cube = np.clip(self.spectra[:, :, mask_w].astype(dt), 0, None)
            if self._despike.get():
                cube, _ = self._despike_cube(cube)
            Y, X, Wsel = cube.shape
            bin_f = max(1, int(self._bin.get()))
            if bin_f > 1:
                yb, xb = Y // bin_f, X // bin_f
                if yb >= 1 and xb >= 1:
                    cube = cube[:yb*bin_f, :xb*bin_f, :].reshape(
                        yb, bin_f, xb, bin_f, Wsel).mean(axis=(1, 3))
            Dw = cube.reshape(-1, Wsel).astype(dt)
            if self.roi_mask is not None and bin_f == 1:
                Dw = Dw[np.asarray(self.roi_mask, dtype=bool).ravel()]
            kmax = min(8, max(self._vars["n_comp"].get() + 3, 5),
                       Dw.shape[0] - 1, Wsel - 1)
            normD = float(np.linalg.norm(Dw)) or 1.0
            ks, lofs = [], []
            for k in range(1, kmax + 1):
                nmf = NMF(n_components=k, init="nndsvda", max_iter=300,
                          random_state=42)
                Cw = nmf.fit_transform(Dw)
                Sw = nmf.components_
                lof = 100.0 * float(np.linalg.norm(Dw - Cw @ Sw)) / normD
                ks.append(k); lofs.append(lof)
            self.after(0, lambda: self._show_lof(ks, lofs))
        except Exception as ex:
            self.after(0, lambda ex=ex: messagebox.showerror(
                "LOF scan error", str(ex), parent=self))
        finally:
            self.after(0, self._prog.stop)

    def _show_lof(self, ks, lofs):
        win = tk.Toplevel(self); win.title("Lack-of-fit vs number of components")
        win.configure(bg=C["bg"]); win.geometry("640x480")
        fig = plt.Figure(figsize=(6, 4.4), facecolor="white")
        ax = fig.add_subplot(111)
        ax.plot(ks, lofs, "o-", color="#2563eb", lw=1.6)
        for k, l in zip(ks, lofs):
            ax.annotate(f"{l:.1f}%", (k, l), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
        ax.set_xlabel("Number of components (k)")
        ax.set_ylabel("Lack of fit  (% of ‖D‖)")
        ax.set_title("MCR-ALS / NMF lack-of-fit vs k", fontweight="semibold")
        ax.set_xticks(ks)
        ax.grid(True, ls="--", lw=0.4, alpha=0.5)
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(cv, win).update()
        drops = [lofs[i-1] - lofs[i] for i in range(1, len(lofs))]
        hint = ""
        if drops:
            for i, d in enumerate(drops):
                if d < 1.0:
                    hint = (f"Improvement past k={ks[i]} is < 1%; "
                            f"k={ks[i]} is likely sufficient.")
                    break
        tk.Label(win, text=hint or "No clear plateau in range — inspect curve.",
                 bg=C["bg"], fg=C["text_mid"], font=("Segoe UI", 10),
                 wraplength=600, justify="left").pack(padx=10, pady=6)
        self._status.config(text="Lack-of-fit scan complete.", fg=C["success"])

    def _flatness_check(self):
        """Check whether the map intensity is a flat field or a tilted plane.

        A tilted / defocused sample makes the total per-pixel intensity drift
        smoothly across the field. We fit a plane I = a·x + b·y + c to the
        total-intensity map; a high fraction of variance explained by that
        plane (plus a non-trivial slope) indicates tilt rather than chemistry.
        """
        cube = np.asarray(self.spectra, dtype=float)
        if cube.ndim != 3:
            messagebox.showwarning("Flatness", "Need a 2-D map cube.",
                                   parent=self)
            return
        Y, X, _ = cube.shape
        if Y < 3 or X < 3:
            messagebox.showwarning("Flatness", "Map too small to assess.",
                                   parent=self)
            return
        inten = np.nansum(np.clip(cube, 0, None), axis=2)   # Y × X
        yy, xx = np.mgrid[0:Y, 0:X]
        m = np.isfinite(inten)
        A = np.column_stack([xx[m].ravel(), yy[m].ravel(),
                             np.ones(m.sum())])
        b = inten[m].ravel()
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        plane = (coef[0] * xx + coef[1] * yy + coef[2])
        resid = inten - plane
        ss_tot = float(np.nansum((inten - np.nanmean(inten)) ** 2)) or 1.0
        ss_res = float(np.nansum(resid[m] ** 2))
        r2 = 1.0 - ss_res / ss_tot
        # Slope expressed as % intensity change across the full field.
        span = abs(coef[0]) * (X - 1) + abs(coef[1]) * (Y - 1)
        mean_i = float(np.nanmean(inten)) or 1.0
        tilt_pct = 100.0 * span / mean_i

        if r2 >= 0.6 and tilt_pct >= 15:
            verdict = ("LIKELY TILTED / defocused: the intensity field is "
                       "mostly a smooth plane.")
            col = "#b91c1c"
        elif r2 >= 0.4 or tilt_pct >= 10:
            verdict = ("BORDERLINE: some planar gradient present — inspect "
                       "and consider focus tracking.")
            col = "#b45309"
        else:
            verdict = "LOOKS FLAT: little planar gradient in intensity."
            col = "#15803d"

        win = tk.Toplevel(self); win.title("Flatness / tilt check")
        win.configure(bg=C["bg"]); win.geometry("900x420")
        fig = plt.Figure(figsize=(8.6, 3.6), facecolor="white")
        a1 = fig.add_subplot(1, 3, 1); a2 = fig.add_subplot(1, 3, 2)
        a3 = fig.add_subplot(1, 3, 3)
        im1 = a1.imshow(inten, cmap="turbo"); a1.set_title("Total intensity")
        fig.colorbar(im1, ax=a1, fraction=0.046)
        im2 = a2.imshow(plane, cmap="turbo"); a2.set_title("Fitted plane")
        fig.colorbar(im2, ax=a2, fraction=0.046)
        im3 = a3.imshow(resid, cmap="coolwarm")
        a3.set_title("Residual (real structure)")
        fig.colorbar(im3, ax=a3, fraction=0.046)
        for ax in (a1, a2, a3):
            ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=win)
        cv.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(cv, win).update()
        tk.Label(win,
                 text=(f"{verdict}\n"
                       f"Plane fit R² = {r2:.2f}   |   "
                       f"tilt ≈ {tilt_pct:.0f}% of mean intensity across field"),
                 bg=C["bg"], fg=col, font=("Segoe UI", 10, "bold"),
                 wraplength=860, justify="left").pack(padx=10, pady=6)
        self._status.config(text="Flatness check complete.", fg=C["success"])

    def _draw(self):
        k    = self._k
        cols = self.COMP_COLORS[:k]
        self._fig.clear()

        import matplotlib.gridspec as gridspec
        self._panel_items = []
        # 3 rows: combined spectra (top), per-component spectra (middle),
        # abundance maps (bottom). Each panel gets its own save-tick.
        gs = gridspec.GridSpec(3, max(k, 2), figure=self._fig,
                               height_ratios=[1.15, 0.85, 1.05],
                               hspace=0.55, wspace=0.30,
                               left=0.05, right=0.82, top=0.95, bottom=0.05)

        # Pure spectra on top row spanning all columns
        ax_sp = self._fig.add_subplot(gs[0, :])
        self._panel_items.append((ax_sp, "pure_spectra_all"))
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

        # Middle row: each component spectrum on its own — individually tickable
        for c in range(k):
            ax_c = self._fig.add_subplot(gs[1, c])
            self._panel_items.append((ax_c, f"spectrum_C{c+1}"))
            spec = self._S[:, c]
            pk   = spec.max() or 1.0
            ax_c.plot(self._xsel, spec / pk, color=cols[c], lw=1.2)
            ax_c.set_title(f"Spectrum — C{c+1}", fontsize=9,
                           fontweight="semibold", color=cols[c])
            ax_c.set_xlabel("Raman Shift (cm⁻¹)", fontsize=8)
            ax_c.tick_params(labelsize=7)
            ax_c.grid(True, ls="--", lw=0.4, alpha=0.5)

        # Abundance maps on bottom row — smoothed, robust-contrast display
        for c in range(k):
            ax_m = self._fig.add_subplot(gs[2, c])
            self._panel_items.append((ax_m, f"abundance_C{c+1}"))
            s = _lut_for_win(self, c)
            vmin, vmax = _lut_clim(self._C[:, :, c], s)
            show_map(ax_m, self._fig, self._C[:, :, c], cmap=s["cmap"],
                     sigma=float(getattr(self, "_dsig", None).get()
                                 if getattr(self, "_dsig", None) else 0.8),
                     robust=False, vmin=vmin, vmax=vmax,
                     title=f"Abundance Map — C{c+1}", title_color=cols[c],
                     colorbar=True, px_um=getattr(self.master, "_px_um", None))

        self._canvas.draw_idle()
        self.after_idle(self._place_panel_ticks)
        ns = getattr(self, "_n_spikes", 0)
        spike_txt = f"  ({ns} cosmic-ray points removed)" if ns else ""
        self._status.config(
            text=f"MCR-ALS converged — {k} components.{spike_txt}",
            fg=C["success"])

    def _save_fig(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf")], parent=self)
        if path:
            self._fig.savefig(path, dpi=250, bbox_inches="tight")

    def _save_panels(self):
        """Save every plot in the MCR-ALS window as its own standalone,
        publication-ready file: the combined pure-spectra plot, each
        component spectrum on its own, and each abundance map (with its own
        colorbar + scale bar). Each panel is re-rendered into a fresh figure
        so nothing is cropped or clipped."""
        if self._S is None or self._C is None:
            messagebox.showwarning("No results", "Run MCR-ALS first.",
                                   parent=self)
            return
        out_dir = filedialog.askdirectory(
            title="Choose a folder for the individual MCR-ALS panels",
            parent=self)
        if not out_dir:
            return

        # File format: vector PDF is best for journals; PNG also written.
        save_pdf = messagebox.askyesno(
            "Vector copy?",
            "Also save a vector PDF of each panel (recommended for journals)?\n\n"
            "Yes → PNG (600 dpi) + PDF      No → PNG only",
            parent=self)

        k    = self._k
        cols = self.COMP_COLORS[:k]
        base = Path(getattr(self.master, "_last_wdf_path", "MCR")).stem or "MCR"
        sigma = float(getattr(self, "_dsig", None).get()
                      if getattr(self, "_dsig", None) else 0.8)
        px_um = getattr(self.master, "_px_um", None)
        exported = []

        def _write(fig, tag):
            png = Path(out_dir) / f"{base}_{tag}.png"
            fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
            exported.append(png.name)
            if save_pdf:
                pdf = Path(out_dir) / f"{base}_{tag}.pdf"
                fig.savefig(pdf, bbox_inches="tight", facecolor="white")
                exported.append(pdf.name)
            plt.close(fig)

        try:
            # 1) Combined pure-spectra plot (all components, offset)
            f1 = plt.figure(figsize=(7, 4.2), facecolor="white")
            ax = f1.add_subplot(111)
            offset = 0.0
            for c in range(k):
                spec = self._S[:, c]
                pk = spec.max() or 1.0
                ax.plot(self._xsel, spec / pk + offset, color=cols[c],
                        lw=1.3, label=f"Component {c+1}")
                offset += 1.1
            ax.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=11)
            ax.set_ylabel("Intensity  (norm., offset)", fontsize=11)
            ax.set_title("MCR-ALS Recovered Pure Spectra",
                         fontsize=12, fontweight="semibold")
            ax.legend(fontsize=9, frameon=True, framealpha=0.9)
            ax.grid(True, ls="--", lw=0.4, alpha=0.5)
            _write(f1, "pure_spectra_all")

            # 2) Each component spectrum on its own (no offset)
            for c in range(k):
                fc = plt.figure(figsize=(7, 3.4), facecolor="white")
                axc = fc.add_subplot(111)
                spec = self._S[:, c]
                pk = spec.max() or 1.0
                axc.plot(self._xsel, spec / pk, color=cols[c], lw=1.4)
                axc.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=11)
                axc.set_ylabel("Intensity  (norm.)", fontsize=11)
                axc.set_title(f"Pure Spectrum — Component {c+1}",
                              fontsize=12, fontweight="semibold",
                              color=cols[c])
                axc.grid(True, ls="--", lw=0.4, alpha=0.5)
                _write(fc, f"spectrum_C{c+1}")

            # 3) Each abundance map on its own (with colorbar + scale bar)
            for c in range(k):
                fm = plt.figure(figsize=(4.6, 4.0), facecolor="white")
                axm = fm.add_subplot(111)
                s = _lut_for_win(self, c)
                vmin, vmax = _lut_clim(self._C[:, :, c], s)
                show_map(axm, fm, self._C[:, :, c], cmap=s["cmap"],
                         sigma=sigma, robust=False, vmin=vmin, vmax=vmax,
                         title=f"Abundance Map — C{c+1}", title_color=cols[c],
                         colorbar=True, px_um=px_um)
                _write(fm, f"abundance_C{c+1}")
        except Exception as ex:
            messagebox.showerror("Export error", str(ex), parent=self)
            return

        messagebox.showinfo(
            "Panels saved",
            f"Saved {len(exported)} files to:\n{out_dir}", parent=self)
        self._status.config(
            text=f"Saved {len(exported)} individual panels → "
                 f"{Path(out_dir).name}", fg=C["success"])

    def _save_ticked(self):
        """Save only the panels whose top-left checkbox is ticked."""
        if self._S is None or self._C is None:
            messagebox.showwarning("No results", "Run MCR-ALS first.",
                                   parent=self)
            return
        ticked = [tag for tag, var in self._panel_vars.items() if var.get()]
        if not ticked:
            messagebox.showinfo(
                "Nothing selected",
                "Tick the checkbox at the top-left of at least one panel "
                "to choose what to save.", parent=self)
            return
        out_dir = filedialog.askdirectory(
            title="Choose a folder for the ticked panels", parent=self)
        if not out_dir:
            return
        save_pdf = messagebox.askyesno(
            "Vector copy?",
            "Also save a vector PDF of each panel?\n\n"
            "Yes → PNG (600 dpi) + PDF      No → PNG only", parent=self)

        k = self._k
        cols = self.COMP_COLORS[:k]
        base = Path(getattr(self.master, "_last_wdf_path", "MCR")).stem or "MCR"
        sigma = float(getattr(self, "_dsig", None).get()
                      if getattr(self, "_dsig", None) else 0.8)
        px_um = getattr(self.master, "_px_um", None)
        exported = []

        def _write(fig, tag):
            png = Path(out_dir) / f"{base}_{tag}.png"
            fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
            exported.append(png.name)
            if save_pdf:
                pdf = Path(out_dir) / f"{base}_{tag}.pdf"
                fig.savefig(pdf, bbox_inches="tight", facecolor="white")
                exported.append(pdf.name)
            plt.close(fig)

        try:
            if "pure_spectra_all" in ticked:
                f1 = plt.figure(figsize=(7, 4.2), facecolor="white")
                ax = f1.add_subplot(111)
                offset = 0.0
                for c in range(k):
                    spec = self._S[:, c]; pk = spec.max() or 1.0
                    ax.plot(self._xsel, spec / pk + offset, color=cols[c],
                            lw=1.3, label=f"Component {c+1}")
                    offset += 1.1
                ax.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=11)
                ax.set_ylabel("Intensity  (norm., offset)", fontsize=11)
                ax.set_title("MCR-ALS Recovered Pure Spectra",
                             fontsize=12, fontweight="semibold")
                ax.legend(fontsize=9, frameon=True, framealpha=0.9)
                ax.grid(True, ls="--", lw=0.4, alpha=0.5)
                _write(f1, "pure_spectra_all")
            # Each individually-ticked component spectrum (no offset)
            for c in range(k):
                if f"spectrum_C{c+1}" not in ticked:
                    continue
                fc = plt.figure(figsize=(7, 3.4), facecolor="white")
                axc = fc.add_subplot(111)
                spec = self._S[:, c]; pk = spec.max() or 1.0
                axc.plot(self._xsel, spec / pk, color=cols[c], lw=1.4)
                axc.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=11)
                axc.set_ylabel("Intensity  (norm.)", fontsize=11)
                axc.set_title(f"Pure Spectrum — Component {c+1}",
                              fontsize=12, fontweight="semibold", color=cols[c])
                axc.grid(True, ls="--", lw=0.4, alpha=0.5)
                _write(fc, f"spectrum_C{c+1}")
            for c in range(k):
                if f"abundance_C{c+1}" not in ticked:
                    continue
                fm = plt.figure(figsize=(4.6, 4.0), facecolor="white")
                axm = fm.add_subplot(111)
                s = _lut_for_win(self, c)
                vmin, vmax = _lut_clim(self._C[:, :, c], s)
                show_map(axm, fm, self._C[:, :, c], cmap=s["cmap"],
                         sigma=sigma, robust=False, vmin=vmin, vmax=vmax,
                         title=f"Abundance Map — C{c+1}", title_color=cols[c],
                         colorbar=True, px_um=px_um)
                _write(fm, f"abundance_C{c+1}")
        except Exception as ex:
            messagebox.showerror("Export error", str(ex), parent=self)
            return

        messagebox.showinfo(
            "Panels saved",
            f"Saved {len(exported)} files to:\n{out_dir}", parent=self)
        self._status.config(
            text=f"Saved {len(exported)} ticked panels → "
                 f"{Path(out_dir).name}", fg=C["success"])

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
class NFindrWindow(PanelSaveMixin, tk.Toplevel):
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
        self._lut = {}
        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self, bg=C["header"], height=44)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="◉  N-FINDR  —  ENDMEMBER EXTRACTION",
                 bg=C["header"], fg="white",
                 font=("Consolas", 12, "bold")).pack(side="left", padx=16, pady=10)

        left = make_scroll_sidebar(self, 270)

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
        ds = tk.Frame(left, bg=C["sidebar"]); ds.pack(fill="x", padx=12, pady=4)
        tk.Label(ds, text="Display σ", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left")
        self._dsig = tk.DoubleVar(value=0.8)   # display smoothing; set to 0 for faithful pixels
        ttk.Spinbox(ds, from_=0.0, to=4.0, increment=0.2, width=5,
                    textvariable=self._dsig,
                    command=lambda: self._abund is not None and self._draw()
                    ).pack(side="left", padx=6)
        _build_lut_panel(self, left, lambda: self._abund is not None)

        SectionDiv(left, "EXPORT").pack(fill="x")
        ttk.Button(left, text="↓ Save Figure (whole page)", style="N.TButton",
                   command=self._save_fig).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="↓ Save Individual Panels…", style="N.TButton",
                   command=self._save_panels).pack(fill="x", padx=12, pady=4)
        ttk.Button(left, text="↓ Save Endmembers (.csv)", style="N.TButton",
                   command=self._save_endmembers).pack(fill="x", padx=12, pady=4)

        right = tk.Frame(self, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True)
        self._fig = plt.figure(figsize=(12, 7), facecolor="#ffffff")
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, right).update()
        self._enable_panel_ticks(self._canvas, basename="NFINDR")

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

        # Abundance maps (bottom row) — smoothed, robust-contrast display
        for c in range(p):
            ax_m = self._fig.add_subplot(gs[1, c])
            s = _lut_for_win(self, c)
            vmin, vmax = _lut_clim(self._abund[:, :, c], s)
            show_map(ax_m, self._fig, self._abund[:, :, c], cmap=s["cmap"],
                     sigma=float(getattr(self, "_dsig", None).get()
                                 if getattr(self, "_dsig", None) else 0.8),
                     robust=False, vmin=vmin, vmax=vmax,
                     title=f"Abundance — EM {c+1}", title_color=cols[c],
                     px_um=getattr(self.master, "_px_um", None))

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

    def _save_panels(self):
        """Save each N-FINDR plot as its own standalone, publication-ready
        file: the combined endmember spectra, each endmember spectrum on its
        own, and each abundance map (with colorbar + scale bar)."""
        if getattr(self, "_endmembers", None) is None \
                or getattr(self, "_abund", None) is None:
            messagebox.showwarning("No results", "Run N-FINDR first.",
                                   parent=self)
            return
        out_dir = filedialog.askdirectory(
            title="Choose a folder for the individual N-FINDR panels",
            parent=self)
        if not out_dir:
            return
        save_pdf = messagebox.askyesno(
            "Vector copy?",
            "Also save a vector PDF of each panel (recommended for journals)?\n\n"
            "Yes → PNG (600 dpi) + PDF      No → PNG only",
            parent=self)

        p     = self._p
        cols  = self.COMP_COLORS[:p]
        base  = Path(getattr(self.master, "_last_wdf_path", "NFINDR")).stem or "NFINDR"
        sigma = float(getattr(self, "_dsig", None).get()
                      if getattr(self, "_dsig", None) else 0.8)
        px_um = getattr(self.master, "_px_um", None)
        exported = []

        def _write(fig, tag):
            png = Path(out_dir) / f"{base}_{tag}.png"
            fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
            exported.append(png.name)
            if save_pdf:
                pdf = Path(out_dir) / f"{base}_{tag}.pdf"
                fig.savefig(pdf, bbox_inches="tight", facecolor="white")
                exported.append(pdf.name)
            plt.close(fig)

        try:
            # 1) Combined endmember spectra
            f1 = plt.figure(figsize=(7, 4.2), facecolor="white")
            ax = f1.add_subplot(111)
            offset = 0.0
            for c in range(p):
                spec = self._endmembers[c]
                pk = spec.max() or 1.0
                ax.plot(self._xsel, spec / pk + offset, color=cols[c],
                        lw=1.3, label=f"Endmember {c+1}")
                offset += 1.1
            ax.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=11)
            ax.set_ylabel("Intensity  (norm., offset)", fontsize=11)
            ax.set_title("N-FINDR Endmember Spectra",
                         fontsize=12, fontweight="semibold")
            ax.legend(fontsize=9, frameon=True, framealpha=0.9)
            ax.grid(True, ls="--", lw=0.4, alpha=0.5)
            _write(f1, "endmembers_all")

            # 2) Each endmember spectrum on its own
            for c in range(p):
                fc = plt.figure(figsize=(7, 3.4), facecolor="white")
                axc = fc.add_subplot(111)
                spec = self._endmembers[c]
                pk = spec.max() or 1.0
                axc.plot(self._xsel, spec / pk, color=cols[c], lw=1.4)
                axc.set_xlabel("Raman Shift  (cm⁻¹)", fontsize=11)
                axc.set_ylabel("Intensity  (norm.)", fontsize=11)
                axc.set_title(f"Endmember {c+1}", fontsize=12,
                              fontweight="semibold", color=cols[c])
                axc.grid(True, ls="--", lw=0.4, alpha=0.5)
                _write(fc, f"endmember_EM{c+1}")

            # 3) Each abundance map on its own
            for c in range(p):
                fm = plt.figure(figsize=(4.6, 4.0), facecolor="white")
                axm = fm.add_subplot(111)
                s = _lut_for_win(self, c)
                vmin, vmax = _lut_clim(self._abund[:, :, c], s)
                show_map(axm, fm, self._abund[:, :, c], cmap=s["cmap"],
                         sigma=sigma, robust=False, vmin=vmin, vmax=vmax,
                         title=f"Abundance — EM {c+1}", title_color=cols[c],
                         colorbar=True, px_um=px_um)
                _write(fm, f"abundance_EM{c+1}")
        except Exception as ex:
            messagebox.showerror("Export error", str(ex), parent=self)
            return

        messagebox.showinfo(
            "Panels saved",
            f"Saved {len(exported)} files to:\n{out_dir}", parent=self)
        self._status.config(
            text=f"Saved {len(exported)} individual panels → "
                 f"{Path(out_dir).name}", fg=C["success"])

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
# MULTI-MAP ANALYSIS WINDOW
#   • Average-spectra comparison (stacked / offset, labelled peaks)  — like Fig A
#   • Band-ratio image series + normalised quantification trend      — like Fig B
# ─────────────────────────────────────────────────────────────────────────────
class MultiMapWindow(tk.Toplevel):
    """Compare several Raman maps side by side.

    Each entry in ``self.maps`` is a dict with keys:
        name : str                  display label
        wn   : 1-D ndarray (W,)     wavenumber axis
        cube : 3-D ndarray (Y,X,W)  processed hyperspectral cube
        mean : 1-D ndarray (W,)     mean spectrum over all pixels (cached)
    """

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.title("Multi-Map Analysis — compare maps")
        self.configure(bg=C["bg"])
        self.geometry("1180x760")
        self.maps: list[dict] = []
        self._ratio_cache = None          # (names, images, trend) for export

        # ── header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C["header"], height=46)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="▦  MULTI-MAP ANALYSIS",
                 bg=C["header"], fg="white",
                 font=("Consolas", 13, "bold")).pack(side="left", padx=16, pady=12)

        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # ── left: map list + loaders ──────────────────────────────────────────
        left = tk.Frame(body, bg=C["sidebar"], width=270)
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        tk.Label(left, text="Loaded maps", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=(12, 4))

        lb_frame = tk.Frame(left, bg=C["sidebar"])
        lb_frame.pack(fill="both", expand=True, padx=12)
        self.lb = tk.Listbox(lb_frame, selectmode="extended",
                             bg="white", fg=C["text_hi"],
                             font=("Consolas", 10), relief="flat",
                             highlightthickness=1,
                             highlightbackground=C["border"], activestyle="none")
        self.lb.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lb_frame, orient="vertical", command=self.lb.yview)
        sb.pack(side="right", fill="y")
        self.lb.config(yscrollcommand=sb.set)

        btns = tk.Frame(left, bg=C["sidebar"])
        btns.pack(fill="x", padx=12, pady=8)
        ttk.Button(btns, text="＋ Load map files…", style="Primary.TButton",
                   command=self._load_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="＋ Add current map", style="Neutral.TButton",
                   command=self._add_current).pack(fill="x", pady=2)
        ttk.Button(btns, text="↑ Up", style="Neutral.TButton",
                   command=lambda: self._move(-1)).pack(side="left", expand=True, fill="x", padx=(0, 2), pady=2)
        ttk.Button(btns, text="↓ Down", style="Neutral.TButton",
                   command=lambda: self._move(1)).pack(side="left", expand=True, fill="x", padx=(2, 0), pady=2)
        ttk.Button(btns, text="✕ Remove selected", style="Neutral.TButton",
                   command=self._remove).pack(fill="x", pady=2)

        self._status = tk.Label(left, text="No maps loaded.",
                                bg=C["sidebar"], fg=C["text_dim"],
                                font=("Segoe UI", 9), wraplength=240, justify="left")
        self._status.pack(fill="x", padx=12, pady=(4, 10))

        # ── right: notebook with the two tools ────────────────────────────────
        nb = ttk.Notebook(body)
        nb.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self._build_avg_tab(nb)
        self._build_ratio_tab(nb)

    # ── map-list management ───────────────────────────────────────────────────
    def _refresh_list(self):
        self.lb.delete(0, "end")
        for m in self.maps:
            Y, X, W = m["cube"].shape
            self.lb.insert("end", f"{m['name']}  [{X}×{Y}]")
        n = len(self.maps)
        self._status.config(
            text=(f"{n} map{'s' if n != 1 else ''} loaded."
                  if n else "No maps loaded."))

    def _add_map(self, name, wn, cube):
        cube = np.asarray(cube, dtype=float)
        if cube.ndim == 2:
            cube = cube[None, :, :]
        # ensure unique name
        base, i = name, 1
        existing = {m["name"] for m in self.maps}
        while name in existing:
            i += 1; name = f"{base} ({i})"
        first = not self.maps
        self.maps.append({"name": name,
                          "wn": np.asarray(wn, dtype=float),
                          "cube": cube,
                          "mean": cube.reshape(-1, cube.shape[-1]).mean(axis=0)})
        if first:
            self._apply_data_defaults(self.maps[0])

    @staticmethod
    def _auto_peaks(wn, y, n=4):
        """Return up to ``n`` peak positions (cm⁻¹), strongest first, found
        purely from the data — no assumptions about which bands are present."""
        wn = np.asarray(wn, float); y = np.asarray(y, float)
        if y.size < 3:
            return []
        # local maxima
        idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:]))[0] + 1
        if idx.size == 0:
            return []
        # prominence ≈ height above the lower of the two neighbouring minima
        prom = []
        for i in idx:
            l = y[:i].min() if i > 0 else y[i]
            r = y[i:].min() if i < y.size else y[i]
            prom.append(y[i] - max(l, r))
        order = np.argsort(prom)[::-1]
        chosen, span = [], (wn.max() - wn.min())
        for o in order:
            p = wn[idx[o]]
            if all(abs(p - c) > 0.03 * span for c in chosen):   # keep them separated
                chosen.append(p)
            if len(chosen) >= n:
                break
        return chosen

    def _apply_data_defaults(self, m):
        """Seed peak labels and ratio bands from the actual loaded spectrum so
        nothing is tied to any specific Raman band."""
        wn, y = m["wn"], m["mean"]
        span = float(wn.max() - wn.min()) or 1.0
        win = max(span * 0.02, (wn[1] - wn[0]) * 2 if wn.size > 1 else 1.0)
        peaks = self._auto_peaks(wn, y, n=4)
        if peaks:
            self._avg_peaks.set(", ".join(f"{p:.0f}" for p in peaks))
        # numerator = strongest peak; denominator = next strongest (else band edge)
        if peaks:
            c1 = peaks[0]
            self._num_lo.set(round(c1 - win, 1)); self._num_hi.set(round(c1 + win, 1))
            c2 = peaks[1] if len(peaks) > 1 else (wn.min() + 0.9 * span)
            self._den_lo.set(round(c2 - win, 1)); self._den_hi.set(round(c2 + win, 1))
        else:
            lo, hi = float(wn.min()), float(wn.max())
            self._num_lo.set(round(lo + 0.20 * span, 1)); self._num_hi.set(round(lo + 0.30 * span, 1))
            self._den_lo.set(round(lo + 0.70 * span, 1)); self._den_hi.set(round(lo + 0.80 * span, 1))

    def _add_current(self):
        if self.app.spectra is None or self.app.xdata is None:
            messagebox.showinfo("No map", "Load a map in the main window first.",
                                parent=self); return
        nm = Path(getattr(self.app, "_loaded_path", "current") or "current").stem or "current map"
        self._add_map(nm, self.app.xdata, self.app.spectra)
        self._refresh_list()

    def _load_files(self):
        paths = filedialog.askopenfilenames(
            parent=self, title="Load Raman map files",
            filetypes=SUPPORTED_PATTERNS)
        if not paths: return
        self._status.config(text="Loading…"); self.update_idletasks()
        ok, fails = 0, []
        for p in paths:
            try:
                r = _open_raman_any(p)
                proc, _ = preprocess_map(r.spectra, self.app.pp_params)
                self._add_map(Path(p).stem, r.xdata, proc)
                ok += 1
            except Exception as exc:
                fails.append(f"{Path(p).name}: {exc}")
        self._refresh_list()
        if fails:
            messagebox.showwarning(
                "Some files failed",
                f"Loaded {ok} map(s).\n\nFailed:\n" + "\n".join(fails[:8]),
                parent=self)

    def _selected_indices(self):
        return list(self.lb.curselection())

    def _remove(self):
        for i in reversed(self._selected_indices()):
            del self.maps[i]
        self._refresh_list()

    def _move(self, d):
        sel = self._selected_indices()
        if not sel: return
        i = sel[0]; j = i + d
        if 0 <= j < len(self.maps):
            self.maps[i], self.maps[j] = self.maps[j], self.maps[i]
            self._refresh_list()
            self.lb.selection_set(j)

    # ── tab 1: average-spectra comparison ─────────────────────────────────────
    def _build_avg_tab(self, nb):
        tab = tk.Frame(nb, bg=C["bg"]); nb.add(tab, text="  Average Spectra  ")

        ctl = tk.Frame(tab, bg=C["panel"]); ctl.pack(fill="x", padx=6, pady=6)
        tk.Label(ctl, text="Normalise:", bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(8, 4))
        self._avg_norm = tk.StringVar(value="max")
        ttk.Combobox(ctl, textvariable=self._avg_norm, state="readonly", width=8,
                     values=["none", "max", "area", "minmax", "vector"]).pack(side="left")
        tk.Label(ctl, text="Stat:", bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(12, 4))
        self._avg_stat = tk.StringVar(value="mean")
        ttk.Combobox(ctl, textvariable=self._avg_stat, state="readonly", width=8,
                     values=["mean", "median"]).pack(side="left")
        self._avg_band = tk.BooleanVar(value=False)
        tk.Checkbutton(ctl, text="±1 SD", variable=self._avg_band, bg=C["panel"],
                       fg=C["text_mid"], selectcolor="white", activebackground=C["panel"],
                       command=self._draw_avg).pack(side="left", padx=(10, 0))
        tk.Label(ctl, text="Offset:", bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(12, 4))
        self._avg_offset = tk.DoubleVar(value=1.0)
        tk.Scale(ctl, from_=0.0, to=3.0, resolution=0.1, orient="horizontal",
                 variable=self._avg_offset, bg=C["panel"], length=130,
                 showvalue=True, command=lambda _e: self._draw_avg()).pack(side="left")
        tk.Label(ctl, text="Peak labels (cm⁻¹, comma-sep):", bg=C["panel"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left", padx=(12, 4))
        self._avg_peaks = tk.StringVar(value="")   # auto-filled from data
        tk.Entry(ctl, textvariable=self._avg_peaks, width=20, bg="white",
                 relief="flat", highlightthickness=1,
                 highlightbackground=C["border"]).pack(side="left")
        ttk.Button(ctl, text="↻ Plot", style="Primary.TButton",
                   command=self._draw_avg).pack(side="left", padx=8)
        ttk.Button(ctl, text="↓ CSV", style="Neutral.TButton",
                   command=self._export_avg).pack(side="right", padx=6)

        self._avg_fig = plt.figure(figsize=(7, 5.4))
        self._avg_ax = self._avg_fig.add_subplot(111)
        self._avg_cv = FigureCanvasTkAgg(self._avg_fig, master=tab)
        self._avg_cv.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=(0, 4))
        NavigationToolbar2Tk(self._avg_cv, tab).update()

    @staticmethod
    def _norm_factors(y, mode, wn=None):
        """Return (offset, scale) so the normalised spectrum is (y-offset)/scale.
        Returning factors (rather than a normalised array) lets the same scale be
        applied consistently to a spread/SD band."""
        y = np.asarray(y, dtype=float)
        if mode == "max":
            s = np.nanmax(np.abs(y));            return 0.0, (s if s else 1.0)
        if mode == "area":
            s = np.trapz(np.clip(y, 0, None), wn) if wn is not None \
                else np.trapz(np.clip(y, 0, None))
            return 0.0, (s if s else 1.0)
        if mode == "vector":                      # L2 / SNV-style vector norm
            s = float(np.sqrt(np.nansum(y ** 2))); return 0.0, (s if s else 1.0)
        if mode == "minmax":
            lo, hi = np.nanmin(y), np.nanmax(y)
            return float(lo), (float(hi - lo) if hi > lo else 1.0)
        return 0.0, 1.0

    def _repr_spec(self, m, stat):
        """Representative spectrum (mean or robust median over all pixels) plus
        the per-channel standard deviation across pixels (sample heterogeneity)."""
        flat = m["cube"].reshape(-1, m["cube"].shape[-1])
        center = np.nanmedian(flat, axis=0) if stat == "median" \
            else np.nanmean(flat, axis=0)
        sd = np.nanstd(flat, axis=0)
        return center, sd

    def _draw_avg(self):
        ax = self._avg_ax; ax.cla()
        if not self.maps:
            ax.text(0.5, 0.5, "Load maps, then press Plot",
                    ha="center", va="center", transform=ax.transAxes,
                    color=C["text_dim"]); self._avg_cv.draw(); return
        mode = self._avg_norm.get()
        stat = self._avg_stat.get()
        show_band = bool(self._avg_band.get())
        off_step = float(self._avg_offset.get())
        cmap = plt.cm.get_cmap("tab10")
        try:
            peaks = [float(p) for p in self._avg_peaks.get().split(",") if p.strip()]
        except ValueError:
            peaks = []
        ymax_top = 0.0
        for k, m in enumerate(self.maps):
            center, sd = self._repr_spec(m, stat)
            off, scale = self._norm_factors(center, mode, m["wn"])
            y = (center - off) / scale + k * off_step
            col = cmap(k % 10)
            ax.plot(m["wn"], y, lw=1.2, color=col, label=m["name"])
            if show_band:
                sdn = sd / scale
                ax.fill_between(m["wn"], y - sdn, y + sdn, color=col, alpha=0.18,
                                linewidth=0)
                ymax_top = max(ymax_top, np.nanmax(y + sdn))
            else:
                ymax_top = max(ymax_top, np.nanmax(y))
        # peak guide lines + labels across the top
        for pk in peaks:
            ax.axvline(pk, color=C["text_dim"], lw=0.6, ls=":", alpha=0.7)
            ax.text(pk, ymax_top * 1.02, f"{pk:g}", rotation=0, ha="center",
                    va="bottom", fontsize=8, color=C["text_mid"])
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Intensity (a.u.)" + ("  ·  offset stacked" if off_step else ""))
        ax.set_title(f"Comparison of {stat} spectra"
                     + ("  ·  shaded = ±1 SD across pixels" if show_band else ""))
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        ax.margins(x=0.01)
        self._avg_fig.tight_layout()
        self._avg_cv.draw()

    def _export_avg(self):
        if not self.maps:
            messagebox.showinfo("No data", "Nothing to export.", parent=self); return
        path = filedialog.asksaveasfilename(
            parent=self, title="Export average spectra", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not path: return
        # interpolate every map onto the first map's axis for a tidy table
        ref = self.maps[0]["wn"]
        mode = self._avg_norm.get(); stat = self._avg_stat.get()
        cols = [ref]; header = ["wavenumber_cm-1"]
        for m in self.maps:
            center, _ = self._repr_spec(m, stat)
            off, scale = self._norm_factors(center, mode, m["wn"])
            y = (center - off) / scale
            cols.append(np.interp(ref, m["wn"], y))
            header.append(m["name"])
        arr = np.column_stack(cols)
        np.savetxt(path, arr, delimiter=",", header=",".join(header),
                   comments="", fmt="%.6g")
        self._status.config(text=f"Saved {Path(path).name}")

    # ── tab 2: band-ratio image series + trend ────────────────────────────────
    def _build_ratio_tab(self, nb):
        tab = tk.Frame(nb, bg=C["bg"]); nb.add(tab, text="  Ratio Series  ")

        ctl = tk.Frame(tab, bg=C["panel"]); ctl.pack(fill="x", padx=6, pady=6)
        tk.Label(ctl, text="Numerator band:", bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(8, 2))
        self._num_lo = tk.DoubleVar(value=0.0); self._num_hi = tk.DoubleVar(value=0.0)
        for v in (self._num_lo, self._num_hi):
            tk.Entry(ctl, textvariable=v, width=6, bg="white", relief="flat",
                     highlightthickness=1, highlightbackground=C["band_a"]).pack(side="left", padx=1)
        tk.Label(ctl, text="÷ Denominator band:", bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(10, 2))
        self._den_lo = tk.DoubleVar(value=0.0); self._den_hi = tk.DoubleVar(value=0.0)
        for v in (self._den_lo, self._den_hi):
            tk.Entry(ctl, textvariable=v, width=6, bg="white", relief="flat",
                     highlightthickness=1, highlightbackground=C["band_b"]).pack(side="left", padx=1)
        tk.Label(ctl, text="Stat:", bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(10, 2))
        self._ratio_stat = tk.StringVar(value="mean")
        ttk.Combobox(ctl, textvariable=self._ratio_stat, state="readonly", width=7,
                     values=["mean", "median"]).pack(side="left")
        tk.Label(ctl, text="Bkg %:", bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(10, 2))
        self._ratio_bg = tk.DoubleVar(value=5.0)
        tk.Entry(ctl, textvariable=self._ratio_bg, width=4, bg="white", relief="flat",
                 highlightthickness=1, highlightbackground=C["border"]).pack(side="left")
        self._ratio_bl = tk.BooleanVar(value=True)
        tk.Checkbutton(ctl, text="Local baseline", variable=self._ratio_bl,
                       bg=C["panel"], fg=C["text_mid"], selectcolor="white",
                       activebackground=C["panel"]).pack(side="left", padx=(8, 0))
        self._ratio_cmap = tk.StringVar(value="inferno")
        ttk.Combobox(ctl, textvariable=self._ratio_cmap, state="readonly", width=9,
                     values=COLORMAPS).pack(side="left", padx=(6, 0))
        ttk.Button(ctl, text="↻ Compute", style="Primary.TButton",
                   command=self._draw_ratio).pack(side="left", padx=8)
        ttk.Button(ctl, text="↓ Trend CSV", style="Neutral.TButton",
                   command=self._export_ratio).pack(side="right", padx=6)

        self._ratio_fig = plt.figure(figsize=(8, 5.4))
        self._ratio_cv = FigureCanvasTkAgg(self._ratio_fig, master=tab)
        self._ratio_cv.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=(0, 4))
        NavigationToolbar2Tk(self._ratio_cv, tab).update()

    def _band_integral(self, cube, wn, lo, hi, local_baseline=False):
        """Trapezoidal band area on each map's own wavenumber axis. Returns NaN
        where the requested band lies outside the map's range (no silent
        nearest-channel substitution). Optional local linear baseline subtracts
        the straight line between the window endpoints per pixel, so the area is
        insensitive to residual broadband offset."""
        if hi < lo: lo, hi = hi, lo
        mask = (wn >= lo) & (wn <= hi)
        if mask.sum() < 2:                       # band not (sufficiently) covered
            return np.full(cube.shape[:2], np.nan)
        x = wn[mask]
        sub = cube[:, :, mask].astype(float)
        if local_baseline and x[-1] > x[0]:
            y0 = sub[:, :, :1]; y1 = sub[:, :, -1:]
            t = ((x - x[0]) / (x[-1] - x[0]))[None, None, :]
            sub = sub - (y0 + (y1 - y0) * t)
        return np.trapz(np.clip(sub, 0, None), x, axis=2)

    def _ratio_image(self, m):
        bl = bool(self._ratio_bl.get())
        num = self._band_integral(m["cube"], m["wn"],
                                  self._num_lo.get(), self._num_hi.get(), bl)
        den = self._band_integral(m["cube"], m["wn"],
                                  self._den_lo.get(), self._den_hi.get(), bl)
        ratio = np.full(num.shape, np.nan, dtype=float)
        # background rejection: drop pixels whose reference (denominator) band is
        # below a fraction of that map's robust 99th-percentile signal — these are
        # empty/noise pixels where a ratio is meaningless and explosively noisy.
        dvalid = den[np.isfinite(den) & (den > 0)]
        try:
            bg_frac = max(0.0, float(self._ratio_bg.get())) / 100.0
        except (tk.TclError, ValueError):
            bg_frac = 0.05
        thr = (np.percentile(dvalid, 99) * bg_frac) if dvalid.size else 0.0
        good = np.isfinite(num) & np.isfinite(den) & (den > thr) & (den > 0)
        ratio[good] = num[good] / den[good]
        return ratio

    def _draw_ratio(self):
        fig = self._ratio_fig; fig.clf()
        if not self.maps:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Load maps, then press Compute",
                    ha="center", va="center", transform=ax.transAxes,
                    color=C["text_dim"]); self._ratio_cv.draw(); return
        n = len(self.maps)
        images = [self._ratio_image(m) for m in self.maps]
        stat = self._ratio_stat.get()
        # central value + spread per map (NaN background excluded everywhere)
        trend = np.full(n, np.nan); spread = np.full(n, np.nan); npix = np.zeros(n, int)
        for k, im in enumerate(images):
            v = im[np.isfinite(im)]
            npix[k] = v.size
            if v.size:
                if stat == "median":
                    trend[k] = np.median(v)
                    q1, q3 = np.percentile(v, [25, 75]); spread[k] = (q3 - q1) / 2.0
                else:
                    trend[k] = v.mean(); spread[k] = v.std(ddof=1) if v.size > 1 else 0.0
        ref = trend[0]
        if np.isfinite(ref) and ref != 0:
            norm_trend = trend / ref; norm_spread = spread / abs(ref)
        else:
            norm_trend = trend; norm_spread = spread
        # shared colour scale across the series
        finite = np.concatenate([im[np.isfinite(im)].ravel() for im in images]) \
            if any(np.isfinite(im).any() for im in images) else np.array([0, 1])
        vmin, vmax = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
        cmap = self._ratio_cmap.get()

        gs = fig.add_gridspec(2, n, height_ratios=[2.2, 1.4], hspace=0.35, wspace=0.08)
        im0 = None
        for k, im in enumerate(images):
            ax = fig.add_subplot(gs[0, k])
            im0 = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            ax.set_title(self.maps[k]["name"], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        if im0 is not None:
            cax = fig.add_axes([0.92, 0.55, 0.015, 0.33])
            fig.colorbar(im0, cax=cax).ax.tick_params(labelsize=7)
        axt = fig.add_subplot(gs[1, :])
        spread_lbl = "IQR/2" if stat == "median" else "SD"
        axt.errorbar(range(1, n + 1), norm_trend, yerr=norm_spread, fmt="o-",
                     color=C["accent"], lw=1.5, capsize=3, ecolor=C["text_dim"],
                     elinewidth=0.9)
        axt.axhline(1.0, color=C["text_dim"], lw=0.7, ls="--")
        axt.set_xticks(range(1, n + 1))
        axt.set_xticklabels([m["name"] for m in self.maps], rotation=30,
                            ha="right", fontsize=7)
        axt.set_ylabel("Normalised value ratio", fontsize=9)
        axt.set_title(f"Quantification ({stat} band ratio, normalised to first map · "
                      f"error bars = {spread_lbl} across valid pixels)", fontsize=8)
        axt.grid(alpha=0.25)
        fig.subplots_adjust(left=0.06, right=0.90, top=0.93, bottom=0.18)
        self._ratio_cv.draw()
        self._ratio_cache = ([m["name"] for m in self.maps], trend, norm_trend,
                             spread, norm_spread, npix, spread_lbl)
        self._status.config(
            text=f"Ratio: {self._num_lo.get():g}-{self._num_hi.get():g} ÷ "
                 f"{self._den_lo.get():g}-{self._den_hi.get():g}")

    def _export_ratio(self):
        if not self._ratio_cache:
            messagebox.showinfo("No data", "Press Compute first.", parent=self); return
        names, trend, norm, spread, norm_spread, npix, spread_lbl = self._ratio_cache
        path = filedialog.asksaveasfilename(
            parent=self, title="Export ratio trend", defaultextension=".csv",
            filetypes=[("CSV", "*.csv")])
        if not path: return
        sl = spread_lbl.replace("/", "_over_")
        with open(path, "w") as f:
            f.write(f"map,index,band_ratio,{sl},normalised_ratio,"
                    f"normalised_{sl},n_valid_pixels\n")
            for i, nm in enumerate(names):
                f.write(f"{nm},{i+1},{trend[i]:.6g},{spread[i]:.6g},"
                        f"{norm[i]:.6g},{norm_spread[i]:.6g},{npix[i]}\n")
        self._status.config(text=f"Saved {Path(path).name}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# v13: QUALITY-CONTROL MAP WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class QCMapWindow(tk.Toplevel):
    """Per-pixel quality-control maps computed from the loaded data:
    signal-to-noise ratio, total/maximum intensity and (when the raw cube is
    available) detector saturation."""

    def __init__(self, app):
        super().__init__(app)
        self.title("Quality Control — per-pixel maps")
        self.configure(bg=C["bg"]); self.geometry("760x640")
        self.app = app
        self._metrics = self._compute()

        top = tk.Frame(self, bg=C["panel"]); top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="Metric:", bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 11)).pack(side="left", padx=(4, 6))
        self._sel = tk.StringVar(value=list(self._metrics.keys())[0])
        cb = ttk.Combobox(top, textvariable=self._sel, state="readonly",
                          values=list(self._metrics.keys()), width=28)
        cb.pack(side="left"); cb.bind("<<ComboboxSelected>>", lambda _e: self._draw())
        ttk.Button(top, text="↓ Save QC map (CSV)",
                   command=self._save).pack(side="right", padx=4)

        self._fig = plt.figure(figsize=(6, 5))
        self._ax  = self._fig.add_subplot(111)
        self._cv  = FigureCanvasTkAgg(self._fig, master=self)
        self._cv.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=6)
        self._cbar = None
        self._summary = tk.Label(self, bg=C["bg"], fg=C["text_mid"],
                                 font=("Consolas", 10), justify="left")
        self._summary.pack(fill="x", padx=10, pady=(0, 8))
        self._draw()

    def _compute(self):
        P = np.asarray(self.app.spectra, dtype=float)        # Y×X×W processed
        eps = 1e-12
        out = {}
        # noise estimate from successive differences (high-frequency content)
        nd = np.diff(P, axis=2)
        noise = np.std(nd, axis=2) / np.sqrt(2.0)
        signal = P.max(axis=2) - P.min(axis=2)
        out["Signal-to-noise ratio"] = signal / (noise + eps)
        out["Total intensity"] = P.sum(axis=2)
        out["Max intensity"]   = P.max(axis=2)
        raw = getattr(self.app, "_raw_spectra", None)
        if raw is not None and np.asarray(raw).shape[:2] == P.shape[:2]:
            R = np.asarray(raw, dtype=float)
            gmax = float(R.max()) or 1.0
            out["Raw max (saturation)"] = R.max(axis=2) / gmax
        return out

    def _draw(self):
        arr = self._metrics[self._sel.get()]
        self._ax.cla()
        if self._cbar is not None:
            try: self._cbar.remove()
            except Exception: pass
            self._cbar = None
        im = self._ax.imshow(arr, cmap="viridis", origin="upper")
        self._ax.set_title(self._sel.get()); self._ax.set_xlabel("X (px)")
        self._ax.set_ylabel("Y (px)")
        self._cbar = self._fig.colorbar(im, ax=self._ax, fraction=0.046, pad=0.04)
        self._cv.draw()
        a = arr[np.isfinite(arr)]
        if a.size:
            self._summary.config(
                text=(f"min={a.min():.3g}   median={np.median(a):.3g}   "
                      f"mean={a.mean():.3g}   max={a.max():.3g}"))

    def _save(self):
        path = filedialog.asksaveasfilename(
            parent=self, title="Save QC map", defaultextension=".csv",
            initialfile=self._sel.get().replace(" ", "_"),
            filetypes=[("CSV", "*.csv"), ("NumPy", "*.npy")])
        if not path:
            return
        arr = self._metrics[self._sel.get()]
        if path.endswith(".npy"):
            np.save(path, arr)
        else:
            np.savetxt(path, arr, fmt="%.6g", delimiter=",")


# ─────────────────────────────────────────────────────────────────────────────
# v13: BATCH PROCESSING WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class BatchWindow(tk.Toplevel):
    """Apply the current preprocessing recipe to every supported file in a
    folder and export each processed cube, plus a summary CSV."""

    def __init__(self, app):
        super().__init__(app)
        self.title("Batch Processing")
        self.configure(bg=C["bg"]); self.geometry("680x520")
        self.app = app
        self._in  = tk.StringVar()
        self._out = tk.StringVar()
        self._fmt = tk.StringVar(value=".npz")
        self._rec = tk.BooleanVar(value=False)

        frm = tk.Frame(self, bg=C["panel"]); frm.pack(fill="x", padx=10, pady=10)
        def row(label, var, cmd):
            r = tk.Frame(frm, bg=C["panel"]); r.pack(fill="x", pady=4)
            tk.Label(r, text=label, width=14, anchor="w", bg=C["panel"],
                     fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
            tk.Entry(r, textvariable=var).pack(side="left", fill="x", expand=True,
                                               padx=4)
            ttk.Button(r, text="Browse…", command=cmd).pack(side="left")
        row("Input folder", self._in,
            lambda: self._in.set(filedialog.askdirectory(parent=self) or self._in.get()))
        row("Output folder", self._out,
            lambda: self._out.set(filedialog.askdirectory(parent=self) or self._out.get()))

        opt = tk.Frame(frm, bg=C["panel"]); opt.pack(fill="x", pady=6)
        tk.Label(opt, text="Output format", bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 6))
        ttk.Combobox(opt, textvariable=self._fmt, state="readonly", width=8,
                     values=[".npz", ".h5", ".csv", ".txt", ".mat"]).pack(side="left")
        ttk.Checkbutton(opt, text="Recurse into sub-folders",
                        variable=self._rec).pack(side="left", padx=16)
        ttk.Button(opt, text="Load recipe…",
                   command=self._load_recipe).pack(side="right")

        tk.Label(self, text="Uses the recipe currently set in Preprocessing → "
                 "Settings (unless you load one above).", bg=C["bg"],
                 fg=C["text_dim"], font=("Segoe UI", 9)).pack(fill="x", padx=12)

        bar = tk.Frame(self, bg=C["bg"]); bar.pack(fill="x", padx=12, pady=6)
        self._prog = ttk.Progressbar(bar, mode="determinate")
        self._prog.pack(side="left", fill="x", expand=True)
        self._run_btn = ttk.Button(bar, text="▶ Run batch", command=self._run)
        self._run_btn.pack(side="left", padx=8)

        self._log = tk.Text(self, height=12, bg="#0f172a", fg="#cbd5e1",
                            font=("Consolas", 9), wrap="none")
        self._log.pack(fill="both", expand=True, padx=12, pady=(0, 10))

    def _load_recipe(self):
        path = filedialog.askopenfilename(
            parent=self, title="Load recipe",
            filetypes=[("JSON recipe", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            self.app.pp_params = load_recipe_file(path)
            self._logln(f"Loaded recipe: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc), parent=self)

    def _logln(self, msg):
        self._log.insert("end", msg + "\n"); self._log.see("end")
        self.update_idletasks()

    def _run(self):
        in_dir, out_dir = self._in.get().strip(), self._out.get().strip()
        if not in_dir or not os.path.isdir(in_dir):
            messagebox.showwarning("Input", "Choose a valid input folder.",
                                   parent=self); return
        if not out_dir:
            messagebox.showwarning("Output", "Choose an output folder.",
                                   parent=self); return
        self._run_btn.config(state="disabled")
        params = self.app.pp_params
        fmt, rec = self._fmt.get(), self._rec.get()

        def cb(msg):
            self.after(0, lambda: self._logln(msg))

        def worker():
            try:
                n_ok, n_fail = run_batch(in_dir, out_dir, params,
                                         out_format=fmt, recursive=rec, log=cb)
                self.after(0, lambda: messagebox.showinfo(
                    "Batch complete", f"{n_ok} processed, {n_fail} failed.",
                    parent=self))
            except Exception as exc:
                self.after(0, lambda exc=exc: messagebox.showerror(
                    "Batch error", str(exc), parent=self))
            finally:
                self.after(0, lambda: self._run_btn.config(state="normal"))
        threading.Thread(target=worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# v13: PUBLICATION-QUALITY VOLUME RENDERER  (Plotly go.Volume)
# ─────────────────────────────────────────────────────────────────────────────
class VolumeRenderWindow(tk.Toplevel):
    """Publication 3-D renderer for a confocal Raman volume.

    Two modes:
      • Surface — solid iso-surfaces of up to three bands shown together in
        different colours (red / green / blue), reproducing multi-component
        chemical 3-D maps (the WiRE / paper "terrain" look).
      • Cloud   — translucent volume ray-cast of one band (the soft "glow").
    Exports interactive HTML and high-resolution PNG.
    """
    BANDS = [("A", "Red",   "#e8483b"),
             ("B", "Green", "#2ecc55"),
             ("C", "Blue",  "#3b7ae8")]
    REF_COLORS = ["#e8483b", "#2ecc55", "#3b7ae8", "#f59e0b", "#06b6d4"]

    def __init__(self, app, vol4, zvals, xdata):
        super().__init__(app)
        self.title("Publication Volume Renderer")
        self.configure(bg=C["bg"]); self.geometry("450x760")
        self.app = app
        self.vol4 = np.asarray(vol4, dtype=float)        # Z × Y × X × W
        self.zvals = (np.asarray(zvals, dtype=float)
                      if zvals is not None else None)
        self.xdata = np.asarray(xdata, dtype=float)
        nz, ny, nx, W = self.vol4.shape

        xmin, xmax = float(self.xdata.min()), float(self.xdata.max())
        def clamp(v): return max(xmin, min(xmax, v))
        defaults = [(clamp(2850), clamp(2980), True),
                    (clamp(1560), clamp(1680), True),
                    (clamp(1280), clamp(1340), False)]

        pad = tk.Frame(self, bg=C["panel"]); pad.pack(fill="both", expand=True,
                                                      padx=10, pady=10)
        tk.Label(pad, text=f"Volume: {nx} × {ny} × {nz}  (X·Y·Z)",
                 bg=C["panel"], fg=C["text_hi"],
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(2, 6))

        mf = tk.Frame(pad, bg=C["panel"]); mf.pack(fill="x", pady=2)
        tk.Label(mf, text="Mode", width=12, anchor="w", bg=C["panel"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
        self.mode = tk.StringVar(value="solid block (filled, multi-band)")
        ttk.Combobox(mf, textvariable=self.mode, state="readonly", width=26,
                     values=["solid block (filled, multi-band)",
                             "surface (multi-band, hollow)",
                             "cloud (single band, glow)"]).pack(side="left")

        tk.Label(pad, text="Bands (cm⁻¹)  —  colour = chemical component",
                 bg=C["panel"], fg=C["text_mid"],
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 2))
        self.benable = {}; self.blo = {}; self.bhi = {}
        for (key, cname, col), (lo, hi, en) in zip(self.BANDS, defaults):
            r = tk.Frame(pad, bg=C["panel"]); r.pack(fill="x", pady=2)
            v = tk.BooleanVar(value=en); self.benable[key] = v
            ttk.Checkbutton(r, variable=v).pack(side="left")
            tk.Label(r, text=cname, width=6, anchor="w", bg=C["panel"], fg=col,
                     font=("Segoe UI", 10, "bold")).pack(side="left")
            lov = tk.StringVar(value=f"{lo:.0f}"); hiv = tk.StringVar(value=f"{hi:.0f}")
            self.blo[key] = lov; self.bhi[key] = hiv
            tk.Entry(r, textvariable=lov, width=7).pack(side="left", padx=2)
            tk.Label(r, text="–", bg=C["panel"], fg=C["text_dim"]).pack(side="left")
            tk.Entry(r, textvariable=hiv, width=7).pack(side="left", padx=2)

        # ── Reference-spectra concentration analysis (CLS / NNLS) ──────────
        self._refs = []            # list of {name, color, spec(W,)}
        self._cls_result = None
        cf = tk.Frame(pad, bg=C["panel"]); cf.pack(fill="x", pady=(10, 0))
        self._usecls = tk.BooleanVar(value=False)
        ttk.Checkbutton(cf, variable=self._usecls,
                        text="Concentration mode (fit reference spectra)"
                        ).pack(side="left")
        rf = tk.Frame(pad, bg=C["panel"]); rf.pack(fill="x", pady=(2, 0))
        ttk.Button(rf, text="＋ Load…",
                   command=self._load_refs).pack(side="left")
        ttk.Button(rf, text="✕ Clear",
                   command=self._clear_refs).pack(side="left", padx=(4, 0))
        self._reflbl = tk.Label(rf, text="none loaded", bg=C["panel"],
                                fg=C["text_dim"], font=("Segoe UI", 9),
                                wraplength=230, justify="left")
        self._reflbl.pack(side="left", padx=6)
        self._lof = tk.BooleanVar(value=True)
        ttk.Checkbutton(pad, variable=self._lof,
                        text="Show lack-of-fit (LoF) in purple").pack(anchor="w")

        self._vars = {}
        def field(label, key, default):
            r = tk.Frame(pad, bg=C["panel"]); r.pack(fill="x", pady=2)
            tk.Label(r, text=label, width=18, anchor="w", bg=C["panel"],
                     fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
            v = tk.StringVar(value=str(default)); self._vars[key] = v
            tk.Entry(r, textvariable=v, width=10).pack(side="left")
        # depth axis values (µm) and default keep-range
        if self.zvals is not None and self.zvals.size >= 2:
            self._zf = np.sort(self.zvals.astype(float))
        else:
            self._zf = np.arange(nz, dtype=float)
        z0d, z1d = float(self._zf.min()), float(self._zf.max())

        tk.Label(pad, text="", bg=C["panel"]).pack(pady=1)
        field("Z keep ≥ (µm)",   "z0",    f"{z0d:.0f}")
        field("Z keep ≤ (µm)",   "z1",    f"{z1d:.0f}")
        field("Iso level pct",   "iso",   "62")
        field("Smoothing σ (vox)", "sigma", "1.0")
        field("Upsample factor", "up",    "2")
        field("Surface opacity", "op",    "1.0")
        field("Z stretch",       "zst",   "2.0")
        field("LoF threshold",   "lofthr", "0.5")

        wf = tk.Frame(pad, bg=C["panel"]); wf.pack(fill="x", pady=2)
        tk.Label(wf, text="Background", width=18, anchor="w", bg=C["panel"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
        self._bg = tk.StringVar(value="white")
        ttk.Combobox(wf, textvariable=self._bg, state="readonly", width=8,
                     values=["white", "black"]).pack(side="left")

        self._zeq = tk.BooleanVar(value=False)
        ttk.Checkbutton(pad, variable=self._zeq,
                        text="Equalize depth (compensate confocal attenuation)"
                        ).pack(anchor="w", pady=(4, 0))

        bar = tk.Frame(pad, bg=C["panel"]); bar.pack(fill="x", pady=10)
        ttk.Button(bar, text="✨ Render (open in browser)",
                   command=lambda: self._render(save_png=False)).pack(fill="x", pady=2)
        ttk.Button(bar, text="🖼  Render & Save PNG…",
                   command=lambda: self._render(save_png=True)).pack(fill="x", pady=2)
        self._status = tk.Label(pad, text="", bg=C["panel"], fg=C["text_mid"],
                                font=("Segoe UI", 9), wraplength=410,
                                justify="left")
        self._status.pack(anchor="w", pady=(6, 0))

    # ── volume helpers ─────────────────────────────────────────────────────
    def _postgeom(self, V, sigma, up, normalize=True):
        """Apply depth-crop, optional depth-equalisation, upsampling, smoothing
        and (optionally) robust 0-1 normalisation to a Z×Y×X scalar volume."""
        try:
            z0 = float(self._vars["z0"].get()); z1 = float(self._vars["z1"].get())
            if z1 < z0: z0, z1 = z1, z0
            zf = self._zf
            zmask = (zf >= z0) & (zf <= z1)
            if zmask.sum() >= 1:
                V = V[zmask]
                self._zlo_um = float(zf[zmask].min())
                self._zhi_um = float(zf[zmask].max())
        except Exception:
            pass
        if getattr(self, "_zeq", None) is not None and self._zeq.get():
            for zi in range(V.shape[0]):
                ref = np.percentile(V[zi], 99.0)
                if ref > 0:
                    V[zi] = V[zi] / ref
        if up > 1: V = zoom(V, up, order=1)
        if sigma > 0: V = gaussian_filter(V, sigma=sigma * up)
        if normalize:
            vmin = float(np.percentile(V, 20.0)); vmax = float(np.percentile(V, 99.0))
            if vmax <= vmin: vmin, vmax = float(V.min()), float(V.max())
            if vmax > vmin: V = np.clip((V - vmin) / (vmax - vmin), 0.0, 1.0)
        return V

    def _band_volume(self, lo, hi, sigma, up):
        if hi < lo: lo, hi = hi, lo
        mask = (self.xdata >= lo) & (self.xdata <= hi)
        if not mask.any(): mask = np.ones_like(self.xdata, dtype=bool)
        V = self.vol4[:, :, :, mask].mean(axis=3)
        return self._postgeom(V, sigma, up, normalize=True)

    # ── reference-spectra concentration analysis (CLS / NNLS) ───────────────
    def _clear_refs(self):
        self._refs = []
        self._cls_result = None
        self._usecls.set(False)
        self._reflbl.config(text="none loaded", fg=C["text_dim"])
        self._status.config(text="Reference spectra cleared.")

    def _ref_from_file(self, path):
        """Load a single reference spectrum from *any* supported format and
        resample it onto the current data's wavenumber axis.

        Works universally: WDF/WIP/SPC/JDX and ASCII (txt/csv/dpt/dat/asc),
        single spectra or maps/volumes (averaged to a mean reference), with
        any wavenumber ordering, spacing or range.
        """
        r = _open_raman_any(path)
        sx = np.asarray(getattr(r, "xdata"), dtype=float).ravel()
        sd = np.asarray(getattr(r, "spectra"), dtype=float)
        # collapse any map/volume to one mean spectrum over the spectral axis
        spec = sd.reshape(-1, sd.shape[-1]).mean(axis=0)
        # reconcile axis/spectrum lengths
        if sx.size != spec.size:
            if sx.size > spec.size:
                sx = sx[:spec.size]
            else:
                spec = spec[:sx.size]
        if sx.size < 2:
            raise RuntimeError("reference has too few points.")
        # ascending axis, de-duplicated, finite values
        order = np.argsort(sx)
        sx, spec = sx[order], spec[order]
        sx, uniq = np.unique(sx, return_index=True)
        spec = spec[uniq]
        spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
        # warn (caller handles) if there is essentially no spectral overlap
        overlap = min(sx.max(), float(self.xdata.max())) - \
                  max(sx.min(), float(self.xdata.min()))
        si = np.interp(self.xdata, sx, spec)   # clamps outside the ref range
        return si, overlap

    def _load_refs(self):
        paths = filedialog.askopenfilenames(
            parent=self, title="Load reference component spectra (any format)",
            filetypes=[("Raman spectra",
                        "*.txt *.csv *.dpt *.dat *.jdx *.asc *.wdf *.wip *.spc"),
                       ("All files", "*.*")])
        if not paths:
            return
        added = 0
        for p in paths:
            try:
                si, overlap = self._ref_from_file(p)
                if overlap <= 0:
                    messagebox.showwarning(
                        "No spectral overlap",
                        f"{Path(p).name}: its wavenumber range does not overlap "
                        "the loaded data, so it cannot be used as a reference.",
                        parent=self)
                    continue
                col = self.REF_COLORS[len(self._refs) % len(self.REF_COLORS)]
                self._refs.append({"name": Path(p).stem[:18],
                                   "color": col, "spec": si})
                added += 1
            except Exception as e:
                messagebox.showwarning("Reference load error",
                                       f"{Path(p).name}:\n{e}", parent=self)
        if added:
            self._cls_result = None
            self._usecls.set(True)
            self._reflbl.config(
                text=f"{len(self._refs)}: " +
                     ", ".join(r["name"] for r in self._refs),
                fg=C["success"])
            if len(self._refs) < 2:
                messagebox.showinfo(
                    "One component loaded",
                    "Concentration analysis compares components relative to each "
                    "other, so it needs at least TWO reference spectra — with "
                    "only one, every voxel is trivially 100%.\n\nLoad your other "
                    "component spectra (e.g. select them all at once), or use "
                    "✕ Clear to start over.", parent=self)

    def _cls_abundance(self):
        """Per-voxel non-negative least-squares fit of the reference spectra
        (no normalisation). Returns (abundance_volumes, lof_volume, percent)."""
        if self._cls_result is not None:
            return self._cls_result
        from scipy.optimize import nnls
        R = np.stack([r["spec"] for r in self._refs], axis=1)   # W × K
        nz, ny, nx, W = self.vol4.shape
        S = self.vol4.reshape(-1, W)
        K = len(self._refs)
        A = np.zeros((S.shape[0], K), dtype=float)
        res = np.zeros(S.shape[0], dtype=float)
        snorm = np.linalg.norm(S, axis=1) + 1e-9
        for i in range(S.shape[0]):
            c, rnorm = nnls(R, S[i])
            A[i] = c; res[i] = rnorm
        Avol = [A[:, k].reshape(nz, ny, nx) for k in range(K)]
        lof = (res / snorm).reshape(nz, ny, nx)
        tot = A.sum()
        pct = [float(A[:, k].sum() / tot * 100) if tot > 0 else 0.0
               for k in range(K)]
        self._cls_result = (Avol, lof, pct)
        return self._cls_result

    def _grid(self, shape):
        nz, ny, nx = shape
        if getattr(self, "_zlo_um", None) is not None:
            z = np.linspace(self._zlo_um, self._zhi_um, nz)
        elif self.zvals is not None and self.zvals.size >= 2:
            z = np.linspace(self.zvals.min(), self.zvals.max(), nz)
        else:
            z = np.arange(nz, dtype=float)
        return np.meshgrid(z, np.arange(ny), np.arange(nx), indexing="ij")

    def _export(self, fig, save_png):
        out_dir = os.path.dirname(os.path.abspath(
            getattr(self.app, "_loaded_path", "") or "")) or os.getcwd()
        stem = "volume_render"
        src = getattr(self.app, "_loaded_path", None)
        if src: stem = Path(src).stem + "_volume"
        html_path = os.path.join(out_dir, stem + ".html")
        fig.write_html(html_path, include_plotlyjs="cdn")
        url = "file://" + html_path
        opened = False
        import webbrowser
        for name in ("firefox", "chrome", "google-chrome", "chromium",
                     "chromium-browser", "microsoft-edge"):
            try:
                webbrowser.get(name).open(url); opened = True; break
            except Exception:
                continue
        if not opened and sys.platform == "darwin":
            import subprocess
            for app in ("Firefox", "Google Chrome", "Microsoft Edge"):
                try:
                    subprocess.Popen(["open", "-a", app, html_path]); opened = True; break
                except Exception:
                    continue
        if not opened:
            webbrowser.open(url)
        msg = (f"Opened {Path(html_path).name}. If empty, open it in Firefox or "
               f"Chrome (Safari can't render WebGL 3-D).")
        if save_png:
            png = filedialog.asksaveasfilename(
                parent=self, title="Save 3-D PNG", initialfile=stem,
                defaultextension=".png", filetypes=[("PNG", "*.png")])
            if png:
                if _ensure_package("kaleido") is None:
                    messagebox.showwarning("PNG export",
                        "Static PNG needs the 'kaleido' package.\nThe HTML was "
                        "still saved.", parent=self)
                else:
                    fig.write_image(png, width=1500, height=1150, scale=2)
                    msg += f"  PNG saved → {Path(png).name}"
        self._status.config(text=msg)

    # ── render ─────────────────────────────────────────────────────────────
    def _render(self, save_png=False):
        if _ensure_package("plotly") is None:
            messagebox.showerror("Plotly required",
                "Install Plotly:\n    pip install plotly", parent=self)
            return
        self._status.config(text="Rendering…"); self.update_idletasks()
        try:
            import plotly.graph_objects as go
            sigma = float(self._vars["sigma"].get() or 0)
            up    = max(1, int(float(self._vars["up"].get() or 1)))
            zst   = max(0.3, float(self._vars["zst"].get() or 1))
            op    = float(self._vars["op"].get() or 1.0)
            isopct = float(self._vars["iso"].get() or 60)
            bg = self._bg.get()
            mode = self.mode.get()
            solid = mode.startswith("solid")
            surface = mode.startswith("surface")
            use_cls = (getattr(self, "_usecls", None) is not None
                       and self._usecls.get() and len(self._refs) >= 1)
            traces = []

            if use_cls:
                # Concentration estimate — winner-take-all over NNLS abundances.
                self._status.config(text="Fitting reference spectra (NNLS)…")
                self.update_idletasks()
                Avol, lof, pct = self._cls_abundance()
                Ag = [self._postgeom(A.copy(), sigma, up, normalize=False)
                      for A in Avol]
                lofg = self._postgeom(lof.copy(), sigma, up, normalize=False)
                stack = np.stack(Ag, axis=0)          # K × Z × Y × X
                K = len(self._refs)
                tot = stack.sum(axis=0)
                dom = stack.argmax(axis=0)
                thr = float(np.percentile(tot, np.clip(isopct, 1, 99)))
                seg = np.where(tot >= thr, dom + 1, 0).astype(float)
                comps = [(f"{self._refs[k]['name']} ({pct[k]:.1f}%)",
                          self._refs[k]['color']) for k in range(K)]
                if self._lof.get():
                    try: lofthr = float(self._vars["lofthr"].get())
                    except Exception: lofthr = 0.5
                    lm = float(lofg.max()) or 1.0
                    seg = np.where((tot >= thr) & ((lofg / lm) >= lofthr),
                                   K + 1, seg)
                    comps.append(("LoF", "#b026ff"))
                ncat = len(comps)
                Z, Y, X = self._grid(seg.shape)
                oscale = [[0.0, 0.0], [0.5 / ncat, 0.0],
                          [0.5 / ncat + 1e-3, 1.0], [1.0, 1.0]]
                cscale = [[0.0, "rgba(0,0,0,0)"]]
                for i, (n, col) in enumerate(comps):
                    cscale += [[min(0.999, (i + 0.5) / ncat), col],
                               [min(1.0, (i + 1) / ncat), col]]
                traces.append(go.Volume(
                    x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=seg.flatten(),
                    cmin=0, cmax=ncat, colorscale=cscale, opacityscale=oscale,
                    opacity=1.0, surface_count=max(8, 2 * ncat + 2),
                    caps=dict(x_show=True, y_show=True, z_show=True),
                    showscale=False))
                for n, col in comps:
                    traces.append(go.Scatter3d(
                        x=[None], y=[None], z=[None], mode="markers",
                        marker=dict(size=8, color=col), name=n, showlegend=True))
                title = "Raman 3-D Concentration (CLS):  " + "   ".join(
                    f"{self._refs[k]['name']} {pct[k]:.1f}%" for k in range(K))
            elif solid:
                # Winner-take-all segmentation → a FILLED, opaque coloured block.
                enabled = [(k, n, c) for k, n, c in self.BANDS
                           if self.benable[k].get()]
                if not enabled:
                    messagebox.showwarning("No bands",
                        "Enable at least one band.", parent=self)
                    self._status.config(text=""); return
                vols = [self._band_volume(float(self.blo[k].get()),
                                          float(self.bhi[k].get()), sigma, up)
                        for k, _, _ in enabled]
                stack = np.stack(vols, axis=0)            # K × Z × Y × X
                mag = stack.max(axis=0)
                dom = stack.argmax(axis=0)                # 0..K-1
                thr = float(np.percentile(mag, np.clip(isopct, 1, 99)))
                seg = np.where(mag >= thr, dom + 1, 0).astype(float)   # 0=bg
                K = len(enabled)
                Z, Y, X = self._grid(seg.shape)
                # discrete colour ramp (value 0 = transparent background,
                # values 1..K each a solid component colour) + step opacity
                oscale = [[0.0, 0.0], [0.5 / K, 0.0], [0.5 / K + 1e-3, 1.0],
                          [1.0, 1.0]]
                cscale = [[0.0, "rgba(0,0,0,0)"]]
                for i, (k, n, col) in enumerate(enabled):
                    cscale += [[min(0.999, (i + 0.5) / K), col],
                               [min(1.0, (i + 1) / K), col]]
                traces.append(go.Volume(
                    x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=seg.flatten(),
                    cmin=0, cmax=K, colorscale=cscale, opacityscale=oscale,
                    opacity=1.0, surface_count=max(8, 2 * K + 2),
                    caps=dict(x_show=True, y_show=True, z_show=True),
                    showscale=False))
                # legend proxies
                for k, n, col in enabled:
                    traces.append(go.Scatter3d(
                        x=[None], y=[None], z=[None], mode="markers",
                        marker=dict(size=8, color=col), name=n, showlegend=True))
                title = "Raman 3-D Chemical Volume (segmented)"
            elif surface:
                for key, cname, col in self.BANDS:
                    if not self.benable[key].get():
                        continue
                    V = self._band_volume(float(self.blo[key].get()),
                                          float(self.bhi[key].get()), sigma, up)
                    Z, Y, X = self._grid(V.shape)
                    lvl = float(np.percentile(V, np.clip(isopct, 1, 99)))
                    traces.append(go.Isosurface(
                        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                        value=V.flatten(), isomin=lvl, isomax=float(V.max()),
                        surface_count=2, colorscale=[[0, col], [1, col]],
                        showscale=False, opacity=op, flatshading=False,
                        caps=dict(x_show=True, y_show=True, z_show=True),
                        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.25,
                                      roughness=0.9, fresnel=0.1),
                        lightposition=dict(x=100, y=200, z=300),
                        name=cname, showlegend=True))
                title = "Raman 3-D Chemical Surfaces"
            else:
                key = next((k for k, _, _ in self.BANDS
                            if self.benable[k].get()), "A")
                V = self._band_volume(float(self.blo[key].get()),
                                      float(self.bhi[key].get()), sigma, up)
                Z, Y, X = self._grid(V.shape)
                lvl = float(np.percentile(V, np.clip(isopct, 1, 99)))
                cscale = [[0.0, "#04240f"], [0.25, "#0e5a28"], [0.55, "#27a24b"],
                          [0.80, "#5fd97e"], [1.0, "#e6ffe9"]]
                traces.append(go.Volume(
                    x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=V.flatten(),
                    isomin=lvl, isomax=1.0, opacity=min(0.6, op),
                    surface_count=22, colorscale=cscale,
                    opacityscale=[[0, 0], [0.35, 0], [0.55, 0.35],
                                  [0.8, 0.72], [1, 1]],
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    showscale=True))
                title = "Raman Confocal Volume"

            if not traces:
                messagebox.showwarning("No bands",
                    "Enable at least one band.", parent=self)
                self._status.config(text=""); return

            ax_col = "#444" if bg == "white" else "#bbb"
            fig = go.Figure(data=traces)
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="X (px)", yaxis_title="Y (px)",
                    zaxis_title="Z (µm)",
                    aspectmode="manual", aspectratio=dict(x=1, y=1, z=zst),
                    xaxis=dict(showgrid=False, showbackground=False, color=ax_col),
                    yaxis=dict(showgrid=False, showbackground=False, color=ax_col),
                    zaxis=dict(showgrid=False, showbackground=False, color=ax_col),
                    bgcolor=bg),
                paper_bgcolor=bg,
                font=dict(color="black" if bg == "white" else "white"),
                legend=dict(font=dict(color="black" if bg == "white" else "white")),
                margin=dict(l=0, r=0, t=40, b=0), showlegend=True)
            self._export(fig, save_png)
        except Exception as exc:
            messagebox.showerror("Render failed", str(exc), parent=self)
            self._status.config(text="")


# ─────────────────────────────────────────────────────────────────────────────
# v13: FULL-SPECTRUM LIBRARY SEARCH
# ─────────────────────────────────────────────────────────────────────────────
class LibrarySearchWindow(tk.Toplevel):
    """Identify a spectrum by correlating it against a user-supplied reference
    library (RRUFF / Raman Open Database / SLoPP / any folder of spectra).

    No spectral library is bundled with BioRaman — the user loads one they have
    downloaded, so there are no licensing constraints on the software itself.
    Libraries are matched on the overlapping wavenumber range after a chosen
    preprocessing (raw / SNV / 1st-derivative) using Pearson correlation.
    """
    SPEC_EXTS = (".txt", ".csv", ".dpt", ".dat", ".asc", ".jdx", ".spc", ".wdf")

    def __init__(self, app):
        super().__init__(app)
        self.title("Full-Spectrum Library Search")
        self.configure(bg=C["bg"]); self.geometry("960x620")
        self.app = app
        self.x = np.asarray(app.xdata, dtype=float)
        self._lib = []     # list of {name, spec(W,) with NaN outside coverage}

        left = make_scroll_sidebar(self, 300)
        SectionDiv(left, "REFERENCE LIBRARY").pack(fill="x")
        ttk.Button(left, text="★ Load bundled library",
                   command=self._load_bundled).pack(fill="x", padx=10, pady=(8, 2))
        ttk.Button(left, text="📁 Load library folder…",
                   command=self._load_folder).pack(fill="x", padx=10, pady=(2, 2))
        ttk.Button(left, text="＋ Load library files…",
                   command=self._load_files).pack(fill="x", padx=10, pady=2)
        self._libl = tk.Label(left, text="no library loaded", bg=C["sidebar"],
                              fg=C["text_dim"], font=("Segoe UI", 9),
                              wraplength=270, justify="left")
        self._libl.pack(fill="x", padx=10, pady=4)

        SectionDiv(left, "QUERY SPECTRUM").pack(fill="x")
        self._qsrc = tk.StringVar(value="Selected pixel")
        for v in ("Selected pixel", "ROI mean", "Whole-map mean"):
            ttk.Radiobutton(left, text=v, value=v, variable=self._qsrc).pack(
                anchor="w", padx=14)

        SectionDiv(left, "MATCHING").pack(fill="x")
        rp = tk.Frame(left, bg=C["sidebar"]); rp.pack(fill="x", padx=10, pady=3)
        tk.Label(rp, text="Preprocess", width=11, anchor="w", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
        self._prep = tk.StringVar(value="SNV")
        ttk.Combobox(rp, textvariable=self._prep, state="readonly", width=14,
                     values=["raw", "SNV", "1st derivative"]).pack(side="left")
        rt = tk.Frame(left, bg=C["sidebar"]); rt.pack(fill="x", padx=10, pady=3)
        tk.Label(rt, text="Top matches", width=11, anchor="w", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
        self._topn = tk.IntVar(value=15)
        ttk.Spinbox(rt, from_=5, to=100, textvariable=self._topn,
                    width=6).pack(side="left")
        rb = tk.Frame(left, bg=C["sidebar"]); rb.pack(fill="x", padx=10, pady=3)
        tk.Label(rb, text="Broaden σ", width=11, anchor="w", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
        self._sigma = tk.DoubleVar(value=0.0)
        ttk.Spinbox(rb, from_=0.0, to=50.0, increment=1.0,
                    textvariable=self._sigma, width=6).pack(side="left")
        tk.Label(rb, text="cm⁻¹", bg=C["sidebar"], fg=C["text_dim"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(4, 0))
        ttk.Button(left, text="🔎 Search", style="ROI.TButton",
                   command=self._search).pack(fill="x", padx=10, pady=(10, 4))
        ttk.Button(left, text="↓ Export results (CSV)",
                   command=self._export).pack(fill="x", padx=10, pady=2)

        right = tk.Frame(self, bg=C["bg"]); right.pack(side="left", fill="both",
                                                       expand=True)
        cols = ("rank", "name", "score", "act", "conf")
        heads = ("Rank", "Name", "Score", "Act", "Confidence")
        self._tv = ttk.Treeview(right, columns=cols, show="headings", height=10)
        for c, h, w in zip(cols, heads, (50, 300, 80, 80, 90)):
            self._tv.heading(c, text=h); self._tv.column(c, width=w)
        self._tv.pack(fill="x", padx=8, pady=8)
        self._tv.bind("<<TreeviewSelect>>", lambda _e: self._plot_selected())

        self._fig = plt.Figure(figsize=(7, 3.2))
        self._ax = self._fig.add_subplot(111)
        self._cv = FigureCanvasTkAgg(self._fig, master=right)
        self._cv.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._results = []
        self._draw_query()

    # ── library loading ─────────────────────────────────────────────────────
    def _ingest(self, paths):
        n = 0
        for p in paths:
            try:
                r = _open_raman_any(str(p))
                sx = np.asarray(r.xdata, dtype=float).ravel()
                sd = np.asarray(r.spectra, dtype=float)
                spec = sd.reshape(-1, sd.shape[-1]).mean(axis=0)
                if sx.size != spec.size:
                    m = min(sx.size, spec.size); sx, spec = sx[:m], spec[:m]
                if sx.size < 4:
                    continue
                order = np.argsort(sx); sx, spec = sx[order], spec[order]
                sx, uniq = np.unique(sx, return_index=True); spec = spec[uniq]
                spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
                si = np.interp(self.x, sx, spec, left=np.nan, right=np.nan)
                if np.isfinite(si).sum() >= 10:
                    self._lib.append({"name": Path(p).stem[:60], "spec": si})
                    n += 1
            except Exception:
                continue
        if n:
            self._libl.config(text=f"{len(self._lib)} spectra loaded",
                              fg=C["success"])
        else:
            messagebox.showwarning("Library",
                "No usable spectra found (need spectra overlapping your data's "
                "wavenumber range).", parent=self)

    def _load_folder(self):
        d = filedialog.askdirectory(parent=self, title="Reference library folder")
        if not d:
            return
        paths = [p for p in Path(d).rglob("*")
                 if p.is_file() and p.suffix.lower() in self.SPEC_EXTS]
        if not paths:
            messagebox.showwarning("Library", "No spectra files in that folder.",
                                   parent=self); return
        self._libl.config(text=f"loading {len(paths)} files…", fg=C["text_mid"])
        self.update_idletasks()
        self._ingest(paths)

    def _load_files(self):
        paths = filedialog.askopenfilenames(
            parent=self, title="Reference library files",
            filetypes=[("Spectra", " ".join("*" + e for e in self.SPEC_EXTS)),
                       ("All files", "*.*")])
        if paths:
            self._ingest([Path(p) for p in paths])

    def _load_bundled(self):
        d = _bundled_library_dir()
        if not d:
            messagebox.showwarning(
                "Bundled library",
                "Bundled PCRS library not found. Expected a 'pcrs_library' "
                "folder next to the application.", parent=self)
            return
        paths = [p for p in Path(d).rglob("*")
                 if p.is_file() and p.suffix.lower() in self.SPEC_EXTS]
        if not paths:
            messagebox.showwarning("Bundled library",
                "No spectra in the bundled library folder.", parent=self)
            return
        self._libl.config(text=f"loading bundled library ({len(paths)})…",
                          fg=C["text_mid"])
        self.update_idletasks()
        self._ingest(paths)

    # ── query + matching ────────────────────────────────────────────────────
    def _query_spectrum(self):
        S = self.app.spectra
        src = self._qsrc.get()
        if src == "Whole-map mean":
            return S.reshape(-1, S.shape[-1]).mean(axis=0)
        if src == "ROI mean":
            m = getattr(self.app, "_roi_mask", None)
            if m is not None and np.asarray(m).any():
                return S[np.asarray(m, dtype=bool)].mean(axis=0)
        xy = getattr(self.app, "coords", None) or (0, 0)
        x, y = xy
        return S[y, x]

    @staticmethod
    def _prep_vec(v, mode):
        v = np.asarray(v, dtype=float)
        if mode == "1st derivative":
            v = np.gradient(v)
        if mode in ("SNV", "1st derivative"):
            mu = np.nanmean(v); sd = np.nanstd(v)
            if sd > 0: v = (v - mu) / sd
        return v

    def _broaden(self, v):
        """Optional Gaussian smoothing in cm⁻¹ to tolerate resolution /
        small peak-position differences between query and library."""
        sigma = float(self._sigma.get() or 0.0)
        v = np.asarray(v, dtype=float)
        if sigma <= 0:
            return v
        dx = np.median(np.abs(np.diff(self.x)))
        if not np.isfinite(dx) or dx <= 0:
            return v
        nan_mask = ~np.isfinite(v)
        sm = gaussian_filter1d(np.nan_to_num(v, nan=0.0), sigma / dx)
        sm[nan_mask] = np.nan
        return sm

    @staticmethod
    def _corr(q, L, mask):
        """Mean-centred cosine similarity (Pearson r) over masked bins."""
        a = q[mask] - q[mask].mean()
        b = L[mask] - L[mask].mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 0:
            return float("nan")
        return float(a @ b / denom)

    @staticmethod
    def _confidence(score, act):
        """Heuristic qualitative label. Thresholds are generic starting
        points — recalibrate on your own validation set."""
        s = act if act == act else score
        if s != s:
            return "—"
        if s >= 0.75:
            return "high"
        if s >= 0.50:
            return "moderate"
        return "low"

    def _search(self):
        if not self._lib:
            messagebox.showwarning("Library", "Load a reference library first.",
                                   parent=self); return
        mode = self._prep.get()
        qraw = self._query_spectrum()
        q = self._prep_vec(self._broaden(qraw), mode)
        self._q_disp = qraw
        # Active bins: where the peak-normalised query rises above the baseline.
        qn = np.asarray(qraw, dtype=float)
        rng = np.nanmax(qn) - np.nanmin(qn)
        active = (qn - np.nanmin(qn)) > 0.05 * rng if rng > 0 else \
            np.ones_like(qn, dtype=bool)
        results = []
        for entry in self._lib:
            L = self._prep_vec(self._broaden(entry["spec"]), mode)
            m = np.isfinite(L) & np.isfinite(q)
            if m.sum() < 10:
                continue
            score = self._corr(q, L, m)
            if score != score:   # NaN -> degenerate
                continue
            ma = m & active
            act = self._corr(q, L, ma) if ma.sum() >= 10 else float("nan")
            results.append((entry["name"], score, act, entry["spec"]))
        results.sort(key=lambda t: t[1], reverse=True)
        self._results = results[: self._topn.get()]
        self._tv.delete(*self._tv.get_children())
        for i, (name, score, act, _) in enumerate(self._results, 1):
            act_s = f"{act:.3f}" if act == act else "—"
            self._tv.insert("", "end", values=(
                i, name, f"{score:.3f}", act_s, self._confidence(score, act)))
        if self._results:
            kids = self._tv.get_children()
            if kids: self._tv.selection_set(kids[0])
            self._plot_selected()

    def _draw_query(self):
        self._ax.cla()
        self._ax.plot(self.x, self._query_spectrum(), color="#2563eb", lw=1.1,
                      label="query")
        self._ax.set_xlabel("Raman shift (cm⁻¹)"); self._ax.set_ylabel("Intensity")
        self._ax.legend(fontsize=8); self._cv.draw()

    def _plot_selected(self):
        sel = self._tv.selection()
        if not sel or not self._results:
            return
        idx = self._tv.index(sel[0])
        name, score, act, spec = self._results[idx]
        self._ax.cla()
        q = self._query_spectrum()
        qn = (q - np.nanmin(q)) / (np.nanmax(q) - np.nanmin(q) + 1e-9)
        sn = (spec - np.nanmin(spec)) / (np.nanmax(spec) - np.nanmin(spec) + 1e-9)
        self._ax.plot(self.x, qn, color="#2563eb", lw=1.1, label="query")
        self._ax.plot(self.x, sn, color="#ef4444", lw=1.0, alpha=0.8,
                      label=f"{name}  (r={score:.3f})")
        self._ax.set_xlabel("Raman shift (cm⁻¹)")
        self._ax.set_ylabel("Normalised intensity")
        self._ax.legend(fontsize=8); self._cv.draw()

    def _export(self):
        if not self._results:
            messagebox.showwarning("Export", "Run a search first.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".csv",
            filetypes=[("CSV", "*.csv")], initialfile="library_matches")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "name", "correlation", "active_correlation",
                        "confidence"])
            for i, (name, score, act, _) in enumerate(self._results, 1):
                w.writerow([i, name, f"{score:.4f}",
                            f"{act:.4f}" if act == act else "",
                            self._confidence(score, act)])
        messagebox.showinfo("Export", f"Saved {len(self._results)} matches.",
                            parent=self)


# ─────────────────────────────────────────────────────────────────────────────
# v14: COMPONENT ANALYSIS (DCLS / NNLS) + PARTICLE STATISTICS  (WiRE-style)
# ─────────────────────────────────────────────────────────────────────────────
def _load_reference_spectrum(path, xaxis):
    """Load a single reference spectrum (any format) onto a target axis."""
    r = _open_raman_any(path)
    sx = np.asarray(r.xdata, dtype=float).ravel()
    sd = np.asarray(r.spectra, dtype=float)
    spec = sd.reshape(-1, sd.shape[-1]).mean(axis=0)
    if sx.size != spec.size:
        m = min(sx.size, spec.size); sx, spec = sx[:m], spec[:m]
    order = np.argsort(sx); sx, spec = sx[order], spec[order]
    sx, uniq = np.unique(sx, return_index=True); spec = spec[uniq]
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    return np.interp(xaxis, sx, spec)


def _otsu_threshold(img):
    """Otsu's threshold for a 2-D image (numpy only, no skimage)."""
    v = np.asarray(img, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0 or v.max() <= v.min():
        return float(np.nanmedian(img))
    hist, edges = np.histogram(v, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w = hist.astype(float); tot = w.sum()
    wB = np.cumsum(w); wF = tot - wB
    sumtot = (w * centers).sum(); sumB = np.cumsum(w * centers)
    valid = (wB > 0) & (wF > 0)
    mB = np.divide(sumB, wB, out=np.zeros_like(sumB), where=wB > 0)
    mF = np.divide(sumtot - sumB, wF, out=np.zeros_like(sumB), where=wF > 0)
    between = wB * wF * (mB - mF) ** 2
    between[~valid] = -1
    # For well-separated (e.g. bimodal) data the between-class variance is flat
    # across the empty gap between clusters; argmax would return the left edge
    # (right at the upper tail of the dark cluster). Return the MIDPOINT of the
    # maximizing plateau instead — the conventional, better-centred threshold.
    mx = between.max()
    plateau = np.flatnonzero(between >= mx - 1e-9 * max(1.0, abs(mx)))
    return float(centers[int(plateau[len(plateau) // 2])])


class ComponentAnalysisWindow(PanelSaveMixin, tk.Toplevel):
    """Supervised component analysis (DCLS / NNLS) on a 2-D map, producing
    concentration maps, a percentage lack-of-fit map and overall concentration
    estimates — the Renishaw WiRE \"Component analysis\" workflow."""
    COLORS = ["#e8483b", "#2ecc55", "#3b7ae8", "#f59e0b", "#9b59b6", "#06b6d4"]

    def __init__(self, app):
        super().__init__(app)
        self.title("Component Analysis  (DCLS / NNLS)")
        self.configure(bg=C["bg"]); self.geometry("1180x760")
        self.app = app
        self.x = np.asarray(app.xdata, dtype=float)
        self._refs = []
        self._res = None
        self._lut = {}      # per-panel {index|'lof': {cmap, lo, hi}}

        left = make_scroll_sidebar(self, 300)
        SectionDiv(left, "REFERENCE COMPONENTS").pack(fill="x")
        ttk.Button(left, text="＋ Load reference spectra…",
                   command=self._load_refs).pack(fill="x", padx=10, pady=(8, 2))
        ttk.Button(left, text="✕ Clear",
                   command=self._clear).pack(fill="x", padx=10, pady=2)
        self._refl = tk.Label(left, text="none loaded", bg=C["sidebar"],
                              fg=C["text_dim"], font=("Segoe UI", 9),
                              wraplength=270, justify="left")
        self._refl.pack(fill="x", padx=10, pady=4)

        SectionDiv(left, "FIT OPTIONS").pack(fill="x")
        def combo(label, var, vals):
            r = tk.Frame(left, bg=C["sidebar"]); r.pack(fill="x", padx=10, pady=3)
            tk.Label(r, text=label, width=12, anchor="w", bg=C["sidebar"],
                     fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
            ttk.Combobox(r, textvariable=var, state="readonly", width=15,
                         values=vals).pack(side="left")
        self._method = tk.StringVar(value="NNLS")
        combo("Method", self._method, ["NNLS", "DCLS"])
        self._prep = tk.StringVar(value="Spectrum")
        combo("Spectrum", self._prep, ["Spectrum", "1st derivative",
                                       "2nd derivative"])
        self._norm = tk.StringVar(value="None (quantitative)")
        combo("Normalise", self._norm, ["None (quantitative)", "Vector",
                                        "Mean-centre + unit variance"])
        rb = tk.Frame(left, bg=C["sidebar"]); rb.pack(fill="x", padx=10, pady=3)
        tk.Label(rb, text="Background", width=12, anchor="w", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
        self._bg = tk.IntVar(value=4)
        ttk.Spinbox(rb, from_=0, to=7, textvariable=self._bg,
                    width=5).pack(side="left")
        tk.Label(rb, text="poly order", bg=C["sidebar"], fg=C["text_dim"],
                 font=("Segoe UI", 9)).pack(side="left", padx=4)
        self._lof = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, variable=self._lof,
                        text="Calculate % lack of fit").pack(anchor="w", padx=12)
        ttk.Button(left, text="▶ Run component analysis", style="ROI.TButton",
                   command=self._run).pack(fill="x", padx=10, pady=(10, 4))

        SectionDiv(left, "CONCENTRATION ESTIMATE").pack(fill="x")
        self._conc = tk.Label(left, text="(run analysis)", bg=C["sidebar"],
                              fg=C["text_hi"], font=("Consolas", 10),
                              justify="left", anchor="w")
        self._conc.pack(fill="x", padx=10, pady=4)
        pf = tk.Frame(left, bg=C["sidebar"]); pf.pack(fill="x", padx=10, pady=2)
        tk.Label(pf, text="Component", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._pcomp = tk.IntVar(value=1)
        self._pspin = ttk.Spinbox(pf, from_=1, to=1, textvariable=self._pcomp,
                                  width=5)
        self._pspin.pack(side="left", padx=4)
        ttk.Button(left, text="◍ Particle statistics on component",
                   command=self._to_particles).pack(fill="x", padx=10, pady=2)
        dsf = tk.Frame(left, bg=C["sidebar"]); dsf.pack(fill="x", padx=10, pady=2)
        tk.Label(dsf, text="Display σ", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._dsig = tk.DoubleVar(value=0.8)   # display smoothing; set to 0 for faithful pixels
        ttk.Spinbox(dsf, from_=0.0, to=4.0, increment=0.2, width=5,
                    textvariable=self._dsig,
                    command=lambda: self._res and self._draw()).pack(
                        side="left", padx=6)

        SectionDiv(left, "PANEL LUT").pack(fill="x")
        cf = tk.Frame(left, bg=C["sidebar"]); cf.pack(fill="x", padx=10, pady=2)
        tk.Label(cf, text="Colour map", width=9, anchor="w", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 9)).pack(side="left")
        self._lut_cmap = tk.StringVar(value="turbo")
        ttk.Combobox(cf, textvariable=self._lut_cmap, state="readonly", width=12,
                     values=["turbo", "viridis", "plasma", "inferno", "magma",
                             "cividis", "jet", "hot", "coolwarm", "RdBu_r",
                             "gray", "gist_earth", "nipy_spectral"]).pack(
                         side="left")
        ct = tk.Frame(left, bg=C["sidebar"]); ct.pack(fill="x", padx=10, pady=2)
        tk.Label(ct, text="Contrast %", width=9, anchor="w", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 9)).pack(side="left")
        self._lut_lo = tk.DoubleVar(value=2.0)
        self._lut_hi = tk.DoubleVar(value=98.0)
        ttk.Spinbox(ct, from_=0, to=49, increment=1, width=5,
                    textvariable=self._lut_lo).pack(side="left")
        tk.Label(ct, text="–", bg=C["sidebar"], fg=C["text_dim"]).pack(side="left")
        ttk.Spinbox(ct, from_=51, to=100, increment=1, width=5,
                    textvariable=self._lut_hi).pack(side="left")
        self._lut_all = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, variable=self._lut_all,
                        text="Apply to all panels").pack(anchor="w", padx=12)
        ttk.Button(left, text="🎨 Apply LUT to panel (use Component #)",
                   command=self._apply_lut).pack(fill="x", padx=10, pady=2)

        ttk.Button(left, text="↓ Export maps + estimates",
                   command=self._export).pack(fill="x", padx=10, pady=2)
        self._status = tk.Label(left, text="", bg=C["sidebar"], fg=C["text_mid"],
                                font=("Segoe UI", 9), wraplength=270,
                                justify="left")
        self._status.pack(fill="x", padx=10, pady=6)

        right = tk.Frame(self, bg=C["bg"]); right.pack(side="left", fill="both",
                                                       expand=True)
        self._fig = plt.Figure(figsize=(8, 7)); self._cv = FigureCanvasTkAgg(
            self._fig, master=right)
        self._cv.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self._enable_panel_ticks(self._cv, basename="ComponentAnalysis")
        note = ("Tip: load the map with Preprocessing → Normalisation = none and "
                "use 'None (quantitative)' here for quantitative concentrations.")
        tk.Label(right, text=note, bg=C["bg"], fg=C["text_dim"],
                 font=("Segoe UI", 9)).pack(pady=(0, 4))

    # ── references ──────────────────────────────────────────────────────────
    def _load_refs(self):
        paths = filedialog.askopenfilenames(
            parent=self, title="Load reference component spectra (any format)",
            filetypes=[("Spectra", "*.txt *.csv *.dpt *.dat *.asc *.jdx *.spc "
                        "*.wdf *.wip"), ("All files", "*.*")])
        for p in paths or []:
            try:
                spec = _load_reference_spectrum(p, self.x)
                self._refs.append({"name": Path(p).stem[:18], "spec": spec})
            except Exception as e:
                messagebox.showwarning("Load error", f"{Path(p).name}: {e}",
                                       parent=self)
        if self._refs:
            self._refl.config(text=f"{len(self._refs)}: " +
                              ", ".join(r["name"] for r in self._refs),
                              fg=C["success"])
            self._pspin.config(to=len(self._refs))

    def _clear(self):
        self._refs = []; self._res = None
        self._refl.config(text="none loaded", fg=C["text_dim"])
        self._conc.config(text="(run analysis)")

    # ── fit ─────────────────────────────────────────────────────────────────
    def _prep_mat(self, M):
        mode = self._prep.get(); norm = self._norm.get()
        if mode == "1st derivative":
            M = np.gradient(M, axis=-1)
        elif mode == "2nd derivative":
            M = np.gradient(np.gradient(M, axis=-1), axis=-1)
        if norm == "Vector":
            n = np.linalg.norm(M, axis=-1, keepdims=True)
            M = np.divide(M, n, out=np.zeros_like(M), where=n > 0)
        elif norm.startswith("Mean"):
            mu = M.mean(axis=-1, keepdims=True); sd = M.std(axis=-1, keepdims=True)
            M = np.divide(M - mu, sd, out=np.zeros_like(M), where=sd > 0)
        return M

    def _run(self):
        if len(self._refs) < 1:
            messagebox.showwarning("References",
                "Load at least one reference component.", parent=self); return
        self._status.config(text="Fitting…"); self.update_idletasks()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            S = np.asarray(self.app.spectra, dtype=float)
            Y, X, W = S.shape; K = len(self._refs)
            refs = np.stack([r["spec"] for r in self._refs], axis=0)
            out = component_fit(
                S.reshape(Y * X, W), refs,
                method=self._method.get(), preprocess=self._prep.get(),
                normalise=self._norm.get(), background_order=int(self._bg.get()))
            maps = [out["rel"][:, k].reshape(Y, X) for k in range(K)]
            lof = out["lof"].reshape(Y, X)
            self._res = dict(maps=maps, lof=lof, overall=out["overall"],
                             names=[r["name"] for r in self._refs])
            self.after(0, self._draw)
        except Exception as exc:
            self.after(0, lambda exc=exc: messagebox.showerror(
                "Component analysis failed", str(exc), parent=self))
            self.after(0, lambda: self._status.config(text=""))

    def _draw(self):
        res = self._res; self._fig.clf()
        maps = res["maps"]; names = res["names"]
        show_lof = self._lof.get()
        n = len(maps) + (1 if show_lof else 0)
        cols = min(3, n); rows = int(np.ceil(n / cols))
        ds = float(getattr(self, "_dsig", None).get()
                   if getattr(self, "_dsig", None) else 0.8)
        def _clim(arr, s):
            f = arr[np.isfinite(arr)]
            if f.size == 0:
                return None, None
            lo, hi = np.percentile(f, [s["lo"], s["hi"]])
            if hi <= lo:
                lo, hi = float(f.min()), float(f.max())
            return float(lo), float(hi)
        pxu = getattr(self.app, "_px_um", None)
        for k, (m, name) in enumerate(zip(maps, names)):
            ax = self._fig.add_subplot(rows, cols, k + 1)
            s = self._lut_for(k); vmin, vmax = _clim(m, s)
            show_map(ax, self._fig, m, cmap=s["cmap"], sigma=ds, robust=False,
                     vmin=vmin, vmax=vmax, px_um=pxu,
                     title=f"{name}  ({res['overall'][k]:.1f}%)")
        if show_lof:
            ax = self._fig.add_subplot(rows, cols, len(maps) + 1)
            s = self._lut_for("lof"); vmin, vmax = _clim(res["lof"], s)
            show_map(ax, self._fig, res["lof"], cmap=s["cmap"], sigma=ds,
                     robust=False, vmin=vmin, vmax=vmax, px_um=pxu,
                     title="% lack of fit")
        self._fig.tight_layout(); self._cv.draw()
        self._conc.config(text="\n".join(
            f"{nm:<14}{v:6.2f}%" for nm, v in zip(names, res["overall"])))
        self._status.config(text="Done.")

    def _lut_for(self, key):
        d = self._lut.get(key)
        if d:
            return d
        return {"cmap": "magma" if key == "lof" else "turbo",
                "lo": 2.0, "hi": 98.0}

    def _apply_lut(self):
        if not self._res:
            messagebox.showwarning("No result", "Run analysis first.", parent=self)
            return
        setting = {"cmap": self._lut_cmap.get(),
                   "lo": float(self._lut_lo.get()),
                   "hi": float(self._lut_hi.get())}
        if self._lut_all.get():
            for k in range(len(self._res["maps"])):
                self._lut[k] = dict(setting)
            self._lut["lof"] = dict(setting)
        else:
            k = int(self._pcomp.get()) - 1
            n = len(self._res["maps"])
            self._lut[k if k < n else "lof"] = dict(setting)
        self._draw()

    def _to_particles(self):
        if not self._res:
            messagebox.showwarning("No result", "Run analysis first.", parent=self)
            return
        k = int(self._pcomp.get()) - 1
        k = max(0, min(k, len(self._res["maps"]) - 1))
        ParticleStatsWindow(self.app, self._res["maps"][k],
                            f"{self._res['names'][k]} concentration")

    def _export(self):
        if not self._res:
            messagebox.showwarning("No result", "Run analysis first.", parent=self)
            return
        d = filedialog.askdirectory(parent=self, title="Export folder")
        if not d:
            return
        for nm, m in zip(self._res["names"], self._res["maps"]):
            np.savetxt(os.path.join(d, f"conc_{nm}.csv"), m, fmt="%.4f",
                       delimiter=",")
        if self._lof.get():
            np.savetxt(os.path.join(d, "lack_of_fit.csv"), self._res["lof"],
                       fmt="%.4f", delimiter=",")
        import csv
        with open(os.path.join(d, "concentration_estimates.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["component", "percent"])
            for nm, v in zip(self._res["names"], self._res["overall"]):
                w.writerow([nm, f"{v:.3f}"])
        self._status.config(text=f"Exported to {Path(d).name}")


class ParticleStatsWindow(tk.Toplevel):
    """Particle / domain statistics on a 2-D image: Otsu (or manual) binarise,
    label connected regions, and report area, equivalent circle diameter and
    counts with a size histogram and CSV export (WiRE Particle Statistics)."""

    def __init__(self, app, image, name="image"):
        super().__init__(app)
        self.title(f"Particle Statistics — {name}")
        self.configure(bg=C["bg"]); self.geometry("1080x720")
        self.app = app
        self.img = np.asarray(image, dtype=float)
        self.name = name
        self._props = []

        left = make_scroll_sidebar(self, 290)
        SectionDiv(left, "BINARISATION").pack(fill="x")
        self._auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, variable=self._auto, text="Auto-binarise (Otsu)",
                        command=self._run).pack(anchor="w", padx=12, pady=2)
        tk.Label(left, text="Threshold (% of max)", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 9)).pack(anchor="w", padx=12)
        self._thr = tk.DoubleVar(value=50.0)
        ttk.Scale(left, from_=1, to=99, variable=self._thr, orient="horizontal",
                  command=lambda _e: self._run()).pack(fill="x", padx=12)
        SectionDiv(left, "FILTERS").pack(fill="x")
        self._edge = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, variable=self._edge, text="Remove edge particles",
                        command=self._run).pack(anchor="w", padx=12)
        rf = tk.Frame(left, bg=C["sidebar"]); rf.pack(fill="x", padx=12, pady=3)
        tk.Label(rf, text="Min size (% of largest)", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 9)).pack(side="left")
        self._minsz = tk.DoubleVar(value=1.0)
        ttk.Spinbox(rf, from_=0, to=100, textvariable=self._minsz, width=6,
                    command=self._run).pack(side="left", padx=4)
        rp = tk.Frame(left, bg=C["sidebar"]); rp.pack(fill="x", padx=12, pady=3)
        tk.Label(rp, text="Pixel size (µm)", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._px = tk.DoubleVar(value=1.0)
        ttk.Spinbox(rp, from_=0.01, to=1000, increment=0.1, textvariable=self._px,
                    width=7, command=self._run).pack(side="left", padx=4)
        ttk.Button(left, text="↻ Recompute", style="ROI.TButton",
                   command=self._run).pack(fill="x", padx=10, pady=(8, 4))
        ttk.Button(left, text="↓ Export particle table (CSV)",
                   command=self._export).pack(fill="x", padx=10, pady=2)
        self._summary = tk.Label(left, text="", bg=C["sidebar"], fg=C["text_hi"],
                                 font=("Consolas", 9), justify="left", anchor="w")
        self._summary.pack(fill="x", padx=10, pady=8)

        right = tk.Frame(self, bg=C["bg"]); right.pack(side="left", fill="both",
                                                       expand=True)
        self._fig = plt.Figure(figsize=(8, 6.6)); self._cv = FigureCanvasTkAgg(
            self._fig, master=right)
        self._cv.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        self._run()

    def _run(self, *_):
        try:
            import scipy.ndimage  # noqa: F401
        except Exception:
            messagebox.showerror("SciPy required", "Particle statistics needs "
                                 "SciPy.", parent=self); return
        res = particle_stats(
            self.img, auto=self._auto.get(), threshold_pct=self._thr.get(),
            remove_edge=self._edge.get(), min_size_pct=self._minsz.get(),
            px_um=float(self._px.get()))
        props = res["props"]; self._props = props
        area_pct = res["area_pct"]; keepmask = res["mask"]
        ecds = np.array([p["ecd_um"] for p in props]) if props else np.array([])

        self._fig.clf()
        ax1 = self._fig.add_subplot(2, 1, 1)
        ax1.imshow(self.img, cmap="turbo", origin="upper")
        ax1.contour(keepmask, levels=[0.5], colors="white", linewidths=0.6)
        for p in props:
            ax1.text(p["cx"], p["cy"], str(p["label"]), color="white",
                     fontsize=6, ha="center", va="center")
        ax1.set_title(f"{self.name}: {len(props)} particles", fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])
        ax2 = self._fig.add_subplot(2, 1, 2)
        if ecds.size:
            ax2.hist(ecds, bins=min(30, max(5, ecds.size)), color="#2563eb",
                     alpha=0.85)
            ax2.set_xlabel("Equivalent circle diameter (µm)")
            ax2.set_ylabel("Count")
        ax2.set_title("Size distribution", fontsize=10)
        self._fig.tight_layout(); self._cv.draw()

        if ecds.size:
            self._summary.config(text=(
                f"particles : {len(props)}\n"
                f"area %     : {area_pct:6.2f}\n"
                f"mean ECD   : {ecds.mean():6.2f} µm\n"
                f"median ECD : {np.median(ecds):6.2f} µm\n"
                f"min / max  : {ecds.min():.2f} / {ecds.max():.2f} µm"))
        else:
            self._summary.config(text="no particles above threshold")

    def _export(self):
        if not self._props:
            messagebox.showwarning("Export", "No particles to export.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="particles")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["label", "area_px", "area_um2", "ecd_um", "cx", "cy"])
            for p in self._props:
                w.writerow([p["label"], p["area_px"], f"{p['area_um2']:.3f}",
                            f"{p['ecd_um']:.3f}", f"{p['cx']:.1f}", f"{p['cy']:.1f}"])
        messagebox.showinfo("Export", f"Saved {len(self._props)} particles.",
                            parent=self)


# ─────────────────────────────────────────────────────────────────────────────
# PCRS: ParticleFinder + IDFinder  (HORIBA PCRS-style integrated workflow)
# ─────────────────────────────────────────────────────────────────────────────
class ParticleFinderWindow(tk.Toplevel):
    """PCRS-style integrated workflow.

    ParticleFinder — automatically locate and characterise particles on the
    current single-band map (area, equivalent-circle diameter, centroid),
    then pull a mean Raman spectrum for each particle straight out of the
    hyperspectral cube.

    IDFinder — match every particle's spectrum against a user-supplied
    reference library (RRUFF / Raman Open Database / SLoPP / any folder of
    spectra) by Pearson correlation, and tag each particle with its best-match
    identity and a confidence score.

    No spectral library is bundled, so there are no licensing constraints —
    the user loads a library they already have.
    """
    SPEC_EXTS = (".txt", ".csv", ".dpt", ".dat", ".asc", ".jdx", ".spc", ".wdf")

    def __init__(self, app, image, name="image"):
        super().__init__(app)
        self.title(f"PCRS — ParticleFinder + IDFinder — {name}")
        self.configure(bg=C["bg"]); self.geometry("1200x780")
        self.app = app
        self.img = np.asarray(image, dtype=float)
        self.name = name
        self.x = np.asarray(app.xdata, dtype=float)
        self.cube = np.asarray(app.spectra, dtype=float)   # (H, W, n_wave)
        self._props = []          # enriched particle dicts
        self._lib = []            # [{name, spec(W,)}]
        self._labels = None       # full integer label image

        # ── left control panel ───────────────────────────────────────────────
        left = make_scroll_sidebar(self, 300)

        SectionDiv(left, "PARTICLEFINDER").pack(fill="x")
        self._auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, variable=self._auto, text="Auto-binarise (Otsu)",
                        command=self._detect).pack(anchor="w", padx=12, pady=2)
        tk.Label(left, text="Threshold (% of max)", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 9)).pack(anchor="w", padx=12)
        self._thr = tk.DoubleVar(value=50.0)
        ttk.Scale(left, from_=1, to=99, variable=self._thr, orient="horizontal",
                  command=lambda _e: self._detect()).pack(fill="x", padx=12)
        self._edge = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, variable=self._edge, text="Remove edge particles",
                        command=self._detect).pack(anchor="w", padx=12)
        rf = tk.Frame(left, bg=C["sidebar"]); rf.pack(fill="x", padx=12, pady=3)
        tk.Label(rf, text="Min size (% of largest)", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 9)).pack(side="left")
        self._minsz = tk.DoubleVar(value=1.0)
        ttk.Spinbox(rf, from_=0, to=100, textvariable=self._minsz, width=6,
                    command=self._detect).pack(side="left", padx=4)
        rp = tk.Frame(left, bg=C["sidebar"]); rp.pack(fill="x", padx=12, pady=3)
        tk.Label(rp, text="Pixel size (µm)", bg=C["sidebar"], fg=C["text_mid"],
                 font=("Segoe UI", 9)).pack(side="left")
        self._px = tk.DoubleVar(value=1.0)
        ttk.Spinbox(rp, from_=0.01, to=1000, increment=0.1, textvariable=self._px,
                    width=7, command=self._detect).pack(side="left", padx=4)
        ttk.Button(left, text="◍ Find particles", style="ROI.TButton",
                   command=self._detect).pack(fill="x", padx=10, pady=(8, 4))

        SectionDiv(left, "IDFINDER LIBRARY").pack(fill="x")
        ttk.Button(left, text="★ Load bundled library",
                   command=self._load_bundled).pack(fill="x", padx=10, pady=(6, 2))
        ttk.Button(left, text="📁 Load library folder…",
                   command=self._load_folder).pack(fill="x", padx=10, pady=(2, 2))
        ttk.Button(left, text="＋ Load library files…",
                   command=self._load_files).pack(fill="x", padx=10, pady=2)
        self._libl = tk.Label(left, text="no library loaded", bg=C["sidebar"],
                              fg=C["text_dim"], font=("Segoe UI", 9),
                              wraplength=270, justify="left")
        self._libl.pack(fill="x", padx=10, pady=4)
        rpm = tk.Frame(left, bg=C["sidebar"]); rpm.pack(fill="x", padx=10, pady=3)
        tk.Label(rpm, text="Preprocess", width=10, anchor="w", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
        self._prep = tk.StringVar(value="SNV")
        ttk.Combobox(rpm, textvariable=self._prep, state="readonly", width=14,
                     values=["raw", "SNV", "1st derivative"]).pack(side="left")
        rbm = tk.Frame(left, bg=C["sidebar"]); rbm.pack(fill="x", padx=10, pady=3)
        tk.Label(rbm, text="Broaden σ", width=11, anchor="w", bg=C["sidebar"],
                 fg=C["text_mid"], font=("Segoe UI", 10)).pack(side="left")
        self._sigma = tk.DoubleVar(value=0.0)
        ttk.Spinbox(rbm, from_=0.0, to=50.0, increment=1.0,
                    textvariable=self._sigma, width=6).pack(side="left")
        tk.Label(rbm, text="cm⁻¹", bg=C["sidebar"], fg=C["text_dim"],
                 font=("Segoe UI", 9)).pack(side="left", padx=(4, 0))
        ttk.Button(left, text="⛯ Identify all particles", style="ROI.TButton",
                   command=self._identify).pack(fill="x", padx=10, pady=(8, 4))
        ttk.Button(left, text="↓ Export particle table (CSV)",
                   command=self._export).pack(fill="x", padx=10, pady=2)

        self._summary = tk.Label(left, text="", bg=C["sidebar"], fg=C["text_hi"],
                                 font=("Consolas", 9), justify="left", anchor="w")
        self._summary.pack(fill="x", padx=10, pady=8)

        # ── right: results table + figures ───────────────────────────────────
        right = tk.Frame(self, bg=C["bg"]); right.pack(side="left", fill="both",
                                                       expand=True)
        cols = ("id", "ecd_um", "area_um2", "match", "score")
        heads = ("#", "ECD µm", "Area µm²", "Best match (IDFinder)", "Score")
        widths = (40, 70, 80, 320, 70)
        self._tv = ttk.Treeview(right, columns=cols, show="headings", height=9)
        for c, h, w in zip(cols, heads, widths):
            self._tv.heading(c, text=h); self._tv.column(c, width=w)
        self._tv.pack(fill="x", padx=8, pady=(8, 4))
        self._tv.bind("<<TreeviewSelect>>", lambda _e: self._plot_selected())

        self._fig = plt.Figure(figsize=(8.4, 5.4))
        self._cv = FigureCanvasTkAgg(self._fig, master=right)
        self._cv.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._detect()

    # ── ParticleFinder ───────────────────────────────────────────────────────
    def _detect(self, *_):
        try:
            import scipy.ndimage  # noqa: F401
        except Exception:
            messagebox.showerror("SciPy required",
                                 "ParticleFinder needs SciPy.", parent=self)
            return
        res = particle_stats(
            self.img, auto=self._auto.get(), threshold_pct=self._thr.get(),
            remove_edge=self._edge.get(), min_size_pct=self._minsz.get(),
            px_um=float(self._px.get()))
        self._labels = res["labels"]
        props = res["props"]
        # Extract a mean spectrum per particle from the cube (if shapes align).
        H, W = self.img.shape
        cube_ok = (self.cube.ndim == 3 and self.cube.shape[0] == H
                   and self.cube.shape[1] == W)
        for p in props:
            p["match"] = ""
            p["score"] = float("nan")
            if cube_ok:
                pmask = (self._labels == p["label"])
                if pmask.any():
                    p["spec"] = self.cube[pmask].mean(axis=0)
                else:
                    p["spec"] = None
            else:
                p["spec"] = None
        self._props = props
        if not cube_ok:
            self._libl.config(
                text="map/cube shape mismatch — spectra unavailable",
                fg=C["text_dim"])
        self._refresh_table()
        self._draw_overview()
        self._update_summary(res["area_pct"])

    # ── IDFinder library ─────────────────────────────────────────────────────
    @staticmethod
    def _prep_vec(v, mode):
        v = np.asarray(v, dtype=float)
        if mode == "1st derivative":
            v = np.gradient(v)
        if mode in ("SNV", "1st derivative"):
            mu = np.nanmean(v); sd = np.nanstd(v)
            if sd > 0:
                v = (v - mu) / sd
        return v

    def _broaden(self, v):
        """Optional Gaussian smoothing in cm⁻¹ applied before matching."""
        sigma = float(self._sigma.get() or 0.0)
        v = np.asarray(v, dtype=float)
        if sigma <= 0:
            return v
        dx = np.median(np.abs(np.diff(self.x)))
        if not np.isfinite(dx) or dx <= 0:
            return v
        nan_mask = ~np.isfinite(v)
        sm = gaussian_filter1d(np.nan_to_num(v, nan=0.0), sigma / dx)
        sm[nan_mask] = np.nan
        return sm

    def _ingest(self, paths):
        n = 0
        for p in paths:
            try:
                r = _open_raman_any(str(p))
                sx = np.asarray(r.xdata, dtype=float).ravel()
                sd = np.asarray(r.spectra, dtype=float)
                spec = sd.reshape(-1, sd.shape[-1]).mean(axis=0)
                if sx.size != spec.size:
                    m = min(sx.size, spec.size); sx, spec = sx[:m], spec[:m]
                if sx.size < 4:
                    continue
                order = np.argsort(sx); sx, spec = sx[order], spec[order]
                sx, uniq = np.unique(sx, return_index=True); spec = spec[uniq]
                spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
                si = np.interp(self.x, sx, spec, left=np.nan, right=np.nan)
                if np.isfinite(si).sum() >= 10:
                    self._lib.append({"name": Path(p).stem[:60], "spec": si})
                    n += 1
            except Exception:
                continue
        if n:
            self._libl.config(text=f"{len(self._lib)} reference spectra loaded",
                              fg=C["success"])
        else:
            messagebox.showwarning(
                "Library", "No usable spectra found (need spectra overlapping "
                "your data's wavenumber range).", parent=self)

    def _load_folder(self):
        d = filedialog.askdirectory(parent=self, title="Reference library folder")
        if not d:
            return
        paths = [p for p in Path(d).rglob("*")
                 if p.is_file() and p.suffix.lower() in self.SPEC_EXTS]
        if not paths:
            messagebox.showwarning("Library", "No spectra files in that folder.",
                                   parent=self); return
        self._libl.config(text=f"loading {len(paths)} files…", fg=C["text_mid"])
        self.update_idletasks()
        self._ingest(paths)

    def _load_files(self):
        paths = filedialog.askopenfilenames(
            parent=self, title="Reference library files",
            filetypes=[("Spectra", " ".join("*" + e for e in self.SPEC_EXTS)),
                       ("All files", "*.*")])
        if paths:
            self._ingest([Path(p) for p in paths])

    def _load_bundled(self):
        d = _bundled_library_dir()
        if not d:
            messagebox.showwarning(
                "Bundled library",
                "Bundled PCRS library not found. Expected a 'pcrs_library' "
                "folder next to the application.", parent=self)
            return
        paths = [p for p in Path(d).rglob("*")
                 if p.is_file() and p.suffix.lower() in self.SPEC_EXTS]
        if not paths:
            messagebox.showwarning("Bundled library",
                "No spectra in the bundled library folder.", parent=self)
            return
        self._libl.config(text=f"loading bundled library ({len(paths)})…",
                          fg=C["text_mid"])
        self.update_idletasks()
        self._ingest(paths)

    def _best_match(self, spec, mode):
        """Return (name, score) of best library match for one spectrum."""
        if spec is None or not self._lib:
            return ("", float("nan"))
        q = self._prep_vec(self._broaden(spec), mode)
        best_name, best_score = "", -2.0
        for entry in self._lib:
            L = self._prep_vec(self._broaden(entry["spec"]), mode)
            m = np.isfinite(L) & np.isfinite(q)
            if m.sum() < 10:
                continue
            a = q[m] - q[m].mean(); b = L[m] - L[m].mean()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom <= 0:
                continue
            score = float(a @ b / denom)
            if score > best_score:
                best_name, best_score = entry["name"], score
        if best_name == "":
            return ("", float("nan"))
        return (best_name, best_score)

    def _identify(self):
        if not self._props:
            messagebox.showinfo("IDFinder", "Find particles first.", parent=self)
            return
        if not self._lib:
            messagebox.showwarning("IDFinder",
                "Load a reference library first.", parent=self); return
        if all(p.get("spec") is None for p in self._props):
            messagebox.showwarning("IDFinder",
                "No per-particle spectra available (map/cube shape mismatch).",
                parent=self); return
        mode = self._prep.get()
        for p in self._props:
            p["match"], p["score"] = self._best_match(p.get("spec"), mode)
        self._refresh_table()
        self._draw_overview()

    # ── views ────────────────────────────────────────────────────────────────
    def _refresh_table(self):
        self._tv.delete(*self._tv.get_children())
        for p in self._props:
            sc = p.get("score")
            sc_s = f"{sc:.3f}" if sc == sc else ""   # NaN check
            self._tv.insert("", "end", iid=str(p["label"]),
                            values=(p["label"], f"{p['ecd_um']:.2f}",
                                    f"{p['area_um2']:.1f}",
                                    p.get("match", ""), sc_s))

    def _draw_overview(self):
        self._fig.clf()
        ax = self._fig.add_subplot(111)
        ax.imshow(self.img, cmap="turbo", origin="upper")
        if self._labels is not None:
            keep = np.isin(self._labels, [p["label"] for p in self._props])
            ax.contour(keep, levels=[0.5], colors="white", linewidths=0.6)
        for p in self._props:
            tag = p.get("match") or str(p["label"])
            if p.get("match"):
                tag = f"{p['label']}:{p['match'][:14]}"
            ax.text(p["cx"], p["cy"], tag, color="white", fontsize=6,
                    ha="center", va="center")
        ax.set_title(f"{self.name}: {len(self._props)} particles", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        self._fig.tight_layout(); self._cv.draw()

    def _plot_selected(self):
        sel = self._tv.selection()
        if not sel:
            return
        lab = int(sel[0])
        p = next((q for q in self._props if q["label"] == lab), None)
        if p is None or p.get("spec") is None:
            return
        self._fig.clf()
        ax = self._fig.add_subplot(111)
        q = np.asarray(p["spec"], dtype=float)
        qn = (q - np.nanmin(q)) / (np.nanmax(q) - np.nanmin(q) + 1e-9)
        ax.plot(self.x, qn, color="#2563eb", lw=1.1,
                label=f"particle {lab}")
        match = p.get("match")
        if match:
            entry = next((e for e in self._lib if e["name"] == match), None)
            if entry is not None:
                s = np.asarray(entry["spec"], dtype=float)
                sn = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-9)
                ax.plot(self.x, sn, color="#ef4444", lw=1.0, alpha=0.8,
                        label=f"{match} (r={p['score']:.3f})")
        ax.set_xlabel("Raman shift (cm⁻¹)")
        ax.set_ylabel("Normalised intensity")
        ax.set_title(f"Particle {lab} spectrum", fontsize=10)
        ax.legend(fontsize=8)
        self._fig.tight_layout(); self._cv.draw()

    def _update_summary(self, area_pct):
        ecds = np.array([p["ecd_um"] for p in self._props]) if self._props else \
            np.array([])
        n_id = sum(1 for p in self._props if p.get("match"))
        if ecds.size:
            self._summary.config(text=(
                f"particles : {len(self._props)}\n"
                f"identified : {n_id}\n"
                f"area %     : {area_pct:6.2f}\n"
                f"mean ECD   : {ecds.mean():6.2f} µm\n"
                f"median ECD : {np.median(ecds):6.2f} µm\n"
                f"min / max  : {ecds.min():.2f} / {ecds.max():.2f} µm"))
        else:
            self._summary.config(text="no particles above threshold")

    def _export(self):
        if not self._props:
            messagebox.showwarning("Export", "No particles to export.",
                                   parent=self); return
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".csv", filetypes=[("CSV", "*.csv")],
            initialfile="pcrs_particles")
        if not path:
            return
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["label", "area_px", "area_um2", "ecd_um", "cx", "cy",
                        "best_match", "score"])
            for p in self._props:
                sc = p.get("score")
                w.writerow([p["label"], p["area_px"], f"{p['area_um2']:.3f}",
                            f"{p['ecd_um']:.3f}", f"{p['cx']:.1f}",
                            f"{p['cy']:.1f}", p.get("match", ""),
                            f"{sc:.4f}" if sc == sc else ""])
        messagebox.showinfo("Export", f"Saved {len(self._props)} particles.",
                            parent=self)


# ─────────────────────────────────────────────────────────────────────────────
# HEADLESS COMMAND-LINE INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
def _run_cli(argv):
    import argparse
    ap = argparse.ArgumentParser(
        prog="bioraman",
        description="BioRaman headless batch preprocessing. With no arguments "
                    "the graphical interface starts.")
    ap.add_argument("--input", "-i",
                    help="Input file or folder of Raman files.")
    ap.add_argument("--out", "-o",
                    help="Output folder for processed data.")
    ap.add_argument("--recipe", "-r",
                    help="JSON preprocessing recipe (defaults to built-in params).")
    ap.add_argument("--format", "-f", default=".npz",
                    choices=[".npz", ".h5", ".csv", ".txt", ".mat"],
                    help="Output format (default: .npz).")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into sub-folders when --input is a folder.")
    ap.add_argument("--save-recipe", metavar="PATH",
                    help="Write the default recipe to PATH and exit.")
    ap.add_argument("--version", action="version",
                    version=f"BioRaman {__version__}")
    args = ap.parse_args(argv)

    if args.save_recipe:
        save_recipe_file(args.save_recipe, PreprocessParams())
        print(f"Wrote default recipe → {args.save_recipe}")
        return 0

    params = (load_recipe_file(args.recipe) if args.recipe
              else PreprocessParams())

    if not args.input or not args.out:
        ap.error("--input and --out are required for batch processing.")

    if os.path.isdir(args.input):
        n_ok, n_fail = run_batch(args.input, args.out, params,
                                 out_format=args.format,
                                 recursive=args.recursive, log=print)
        return 0 if n_fail == 0 else 1

    # single file
    os.makedirs(args.out, exist_ok=True)
    x, cube, report = process_file(args.input, params)
    out_path = os.path.join(
        args.out, Path(args.input).stem + "_processed" + args.format)
    write_cube(out_path, cube, x, report, source=args.input)
    print(f"Processed {args.input} → {out_path}")
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    if len(sys.argv) > 1:
        sys.exit(_run_cli(sys.argv[1:]))

    app = RamanApp()
    # Offline Analysis Planner pops up at launch (user can disable via its checkbox)
    try:
        import bioraman_planner as _bp
        if _bp.should_show_startup():
            app.after(700, app.open_planner)
    except Exception:
        pass
    app.mainloop()