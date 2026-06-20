"""
spectra_io.py — robust spectral file readers for PCA Studio.

Pure (no-GUI) loaders for Renishaw .wdf, ASCII .csv/.txt/.tsv/.dpt/.asc and
Excel .xlsx/.xls. Text and Excel reading auto-detects the delimiter, header
rows, long form (#Wave/#Intensity) and orientation (spectra in columns vs
rows). All auto-detections can be overridden via the `opts` dict:

    opts = {"delim":  "auto"|"comma"|"tab"|"semicolon"|"space/whitespace",
            "header": "auto"|"yes"|"no",
            "orient": "auto"|"cols"|"rows",     # are spectra in columns or rows
            "sheet":  <int>}                    # Excel sheet index

Every reader returns (spectra (N, W) float32, wavenumbers (W,) float, shape)
where `shape` is (Y, X) for a 2-D map or None otherwise.

MIT License — part of the BioRaman / PCA Studio project.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    import pandas as pd
    HAS_PD = True
except Exception:
    HAS_PD = False

try:
    from renishawWiRE import WDFReader
    HAS_WDF = True
except Exception:
    HAS_WDF = False


DELIM_MAP = {"auto": None, "comma": ",", "tab": "\t", "semicolon": ";",
             "space/whitespace": r"\s+"}


def _looks_like_header(row) -> bool:
    """True if a row is mostly non-numeric text (i.e. a column-name header)."""
    n_num = 0
    for v in row:
        try:
            float(str(v).replace(",", ".") if isinstance(v, str) else v)
            n_num += 1
        except (ValueError, TypeError):
            pass
    return n_num < max(1, len(row) // 2)


def _monotonic(a) -> bool:
    a = np.asarray(a, float)
    d = np.diff(a)
    return a.size >= 3 and (np.all(d > 0) or np.all(d < 0))


def _looks_numeric_axis(names) -> bool:
    """True if column names are themselves numbers (wavenumber axis as header)."""
    try:
        return _monotonic([float(c) for c in names])
    except (ValueError, TypeError):
        return False


def _finalize(spec, x):
    spec = np.atleast_2d(np.asarray(spec, float))
    x = np.asarray(x, float)
    if x.size and x[0] > x[-1]:               # force ascending wavenumber axis
        x = x[::-1]; spec = spec[:, ::-1]
    return spec.astype(np.float32), x, None


def _read_delimited(path, opts):
    """Read CSV/TXT/DPT/TSV/ASC into a raw (header-less) DataFrame."""
    if not HAS_PD:
        sep = DELIM_MAP.get(opts.get("delim", "auto"))
        arr = np.atleast_2d(np.genfromtxt(path, delimiter=sep, comments="#"))
        import types
        # minimal DataFrame-like shim is avoided; require pandas for text parsing
        raise RuntimeError("pandas is required for text/CSV import "
                           "(pip install pandas)")
    sep = DELIM_MAP.get(opts.get("delim", "auto"))
    return pd.read_csv(path, sep=sep, engine="python", header=None,
                       comment="#", skip_blank_lines=True)


def _interpret_table(df, opts):
    """Turn a raw table into (spectra, wavenumbers, None).

    Handles header rows, long form (#Wave/#Intensity), and both orientations,
    with manual overrides via `opts`."""
    header_opt = opts.get("header", "auto")
    orient = opts.get("orient", "auto")

    # ── header detection ────────────────────────────────────────────────────
    colnames = None
    first = list(df.iloc[0].values)
    has_header = (header_opt == "yes") or (header_opt == "auto"
                                           and _looks_like_header(first))
    if has_header:
        colnames = [str(c).strip().lstrip("#").lower() for c in first]
        df = df.iloc[1:].reset_index(drop=True)

    # ── long form: explicit wave/intensity columns ──────────────────────────
    if colnames:
        wkey = next((i for i, c in enumerate(colnames)
                     if c in ("wave", "wavenumber", "raman shift", "shift", "x")),
                    None)
        ikey = next((i for i, c in enumerate(colnames)
                     if c.startswith("inten") or c in ("counts", "y", "intensity")),
                    None)
        if wkey is not None and ikey is not None and df.shape[1] <= 3:
            wv = pd.to_numeric(df.iloc[:, wkey], errors="coerce").values
            it = pd.to_numeric(df.iloc[:, ikey], errors="coerce").values
            good = np.isfinite(wv) & np.isfinite(it)
            wv, it = wv[good], it[good]
            x = np.unique(wv)
            n = len(x)
            spec = it.reshape(len(it) // n, n) if n and len(it) % n == 0 else it[None, :]
            return _finalize(spec, x)

    # ── numeric matrix ──────────────────────────────────────────────────────
    M = df.apply(pd.to_numeric, errors="coerce").values.astype(float)
    M = M[~np.all(np.isnan(M), axis=1)][:, ~np.all(np.isnan(M), axis=0)]
    M = np.nan_to_num(M)

    col0_axis = _monotonic(M[:, 0])
    row0_axis = _monotonic(M[0, :])
    header_is_axis = bool(colnames) and _looks_numeric_axis(colnames)

    if orient == "rows" or (orient == "auto" and header_is_axis):
        if header_is_axis:
            x = np.array([float(c) for c in colnames]); spec = M
        elif row0_axis and not col0_axis:
            x = M[0]; spec = M[1:]
        else:
            x = np.arange(M.shape[1], dtype=float); spec = M
    elif orient == "cols" or (orient == "auto" and col0_axis):
        x = M[:, 0]; spec = M[:, 1:].T
    elif orient == "auto" and row0_axis:
        x = M[0]; spec = M[1:]
    else:
        x = np.arange(M.shape[1], dtype=float); spec = M

    if spec.ndim == 1:
        spec = spec[None, :]
    return _finalize(spec, x)


def read_spectra(path: str, opts: dict = None):
    """Return (spectra (N,W) float32, wavenumbers (W,) float, (Y,X) or None)."""
    opts = opts or {}
    ext = Path(path).suffix.lower()

    if ext == ".wdf":
        if not HAS_WDF:
            raise RuntimeError("renishawWiRE not installed: pip install renishawWiRE")
        r = WDFReader(path)
        raw = np.asarray(r.spectra)
        x = np.asarray(r.xdata)
        shape = None
        if raw.ndim == 3:
            Y, X, W = raw.shape
            shape = (Y, X)
            raw = raw.reshape(Y * X, W)
        elif raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if x.ndim != 1 or x.shape[0] != raw.shape[1]:
            x = np.arange(raw.shape[1], dtype=float)
        if x[0] > x[-1]:
            x = x[::-1]; raw = raw[:, ::-1]
        return raw.astype(np.float32), x.astype(float), shape

    if ext in (".csv", ".txt", ".dpt", ".asc", ".tsv"):
        return _interpret_table(_read_delimited(path, opts), opts)

    if ext in (".xlsx", ".xls"):
        if not HAS_PD:
            raise RuntimeError("pandas/openpyxl required for Excel input "
                               "(pip install pandas openpyxl)")
        df = pd.read_excel(path, sheet_name=opts.get("sheet", 0), header=None)
        return _interpret_table(df, opts)

    raise RuntimeError(f"Unsupported file type: {ext}")
