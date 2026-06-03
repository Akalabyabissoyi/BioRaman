#!/usr/bin/env python3
"""
raman_metadata.py  —  RAMANMETRIX-compatible metadata workflow for BioRaman
===========================================================================

Created by: Akalabya Bissoyi  <akalabya.bissoyi@manchester.ac.uk>
Copyright (C) 2026  Akalabya Bissoyi
Licensed under the GNU GPL v3.0-or-later.

This module implements the data-input conventions described in the
RAMANMETRIX software documentation ("Data Input" → "Providing Metadata"):

    https://docs.ramanmetrix.eu/documentation/Data.html

It provides pure, GUI-independent helpers to:

  * enumerate spectral files inside a ZIP archive,
  * parse the embedded ``YYMMDD_hhmmss`` timestamps,
  * infer ``type`` / ``batch`` / ``standard`` / ``date`` from the
    ``.../$type/$batch/$file`` folder structure,
  * generate a metadata template table (short / long / auto), and
  * read an existing metadata table (CSV / XLS / XLSX) and match its rows
    to the spectral files using the documented "longest pattern wins" rule.

All functions here are deliberately free of any Tkinter / matplotlib
dependencies so that they can be unit-tested in isolation.
"""

from __future__ import annotations

import csv
import io
import os
import re
import zipfile
from datetime import datetime
from pathlib import PurePosixPath

# ---------------------------------------------------------------------------
# Column conventions (case-sensitive, per RAMANMETRIX docs)
# ---------------------------------------------------------------------------

# Full default column set documented under "Metadata Table".
DEFAULT_COLUMNS = [
    "file",
    "include",
    "device_wn",
    "standard",
    "standard_intensity",
    "dark_bg",
    "reference_sample",
    "interferent_sample",
    "date",
    "device",
    "batch",
    "type",
]

# The commonly-used subset (Option 2 "from scratch" in the docs).
CORE_COLUMNS = ["file", "include", "standard", "date", "device", "batch", "type"]

# Boolean-valued columns and their documented defaults.
BOOL_DEFAULTS = {
    "include": True,
    "standard": False,
    "standard_intensity": False,
    "dark_bg": False,
    "reference_sample": False,
    "interferent_sample": False,
}

# Spectral-data extensions recognised by RAMANMETRIX / BioRaman.
SPECTRA_EXTS = {
    ".spc", ".jdx", ".txt", ".csv", ".lpe",
    ".wip", ".wdf", ".dpt", ".dat",
}

# Last occurring YYMMDD_hhmmss (or YYMMDD-hhmmss) pattern is the timestamp.
_TIMESTAMP_RE = re.compile(r"(\d{6})[_-](\d{6})")

# Folder name signalling wavenumber-calibration standard spectra.
_STANDARD_RE = re.compile(r"(?:^|[^0-9a-z])(?:4-?aap|aap)(?:[^0-9a-z]|$)", re.I)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _norm(path: str) -> str:
    """Normalise a path to forward-slash form (ZIP internal style)."""
    return str(path).replace("\\", "/").lstrip("./").lstrip("/")


def is_metadata_file(path: str) -> bool:
    """A metadata file is any csv/xls/xlsx whose name contains 'metadata'."""
    p = _norm(path).lower()
    name = p.rsplit("/", 1)[-1]
    return "metadata" in name and name.endswith((".csv", ".xls", ".xlsx"))


def is_spectrum_file(path: str) -> bool:
    p = _norm(path)
    if p.endswith("/"):
        return False
    if is_metadata_file(p):
        return False
    ext = os.path.splitext(p)[1].lower()
    # A .csv/.txt may legitimately be a spectrum; metadata is filtered above.
    return ext in SPECTRA_EXTS


# ---------------------------------------------------------------------------
# Timestamp + folder-structure inference
# ---------------------------------------------------------------------------

def parse_timestamp(path: str):
    """Return a ``datetime`` from the *last* YYMMDD_hhmmss pattern, or None.

    Mirrors the RAMANMETRIX rule: the last occurring ``YYMMDD_hhmmss``
    (or ``YYMMDD-hhmmss``) pattern in the file path is the timestamp.
    """
    matches = list(_TIMESTAMP_RE.finditer(_norm(path)))
    if not matches:
        return None
    ymd, hms = matches[-1].groups()
    for century in ("20", "19"):
        try:
            return datetime.strptime(century + ymd + hms, "%Y%m%d%H%M%S")
        except ValueError:
            continue
    return None


def infer_from_path(path: str) -> dict:
    """Infer metadata fields from a spectrum's path within the archive.

    Convention: ``.../$type/$batch/$SingleSpectrumFile``. Multi-spectra files
    sit directly in ``.../$type/``. Standard spectra live in folders whose
    name equals 'AAP' or contains '4-AAP' / '4AAP'.

    Returns a dict with keys: type, batch, standard, date.
    """
    p = _norm(path)
    parts = PurePosixPath(p).parts
    folders = list(parts[:-1])  # drop the filename

    out = {"type": "", "batch": "", "standard": False, "date": ""}

    # Standard detection on any folder component.
    if any(_STANDARD_RE.search(f) for f in folders):
        out["standard"] = True

    # type / batch from the two innermost folders (excluding standard folders).
    meaningful = [f for f in folders if not _STANDARD_RE.search(f)]
    if len(meaningful) >= 2:
        out["type"], out["batch"] = meaningful[-2], meaningful[-1]
    elif len(meaningful) == 1:
        out["type"] = meaningful[-1]

    ts = parse_timestamp(p)
    if ts is not None:
        out["date"] = ts.strftime("%Y-%m-%d")
    return out


# ---------------------------------------------------------------------------
# ZIP enumeration
# ---------------------------------------------------------------------------

def list_spectra_in_zip(zip_path: str):
    """Return a sorted list of spectral-file paths inside the ZIP."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    return sorted(_norm(n) for n in names if is_spectrum_file(n))


def list_metadata_in_zip(zip_path: str):
    """Return a sorted list of metadata-table paths inside the ZIP."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    return sorted(_norm(n) for n in names if is_metadata_file(n))


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------

def _blank_row(columns):
    row = {c: "" for c in columns}
    for c, default in BOOL_DEFAULTS.items():
        if c in row:
            row[c] = bool(default)
    return row


def _fill_row(path, columns):
    row = _blank_row(columns)
    inferred = infer_from_path(path)
    if "file" in row:
        row["file"] = path
    for k in ("type", "batch", "date"):
        if k in row and inferred.get(k):
            row[k] = inferred[k]
    if "standard" in row:
        row["standard"] = bool(inferred.get("standard", False))
    return row


def generate_template(spectra_paths, columns=None, kind="long"):
    """Generate metadata template rows for a set of spectral file paths.

    Parameters
    ----------
    spectra_paths : iterable of str
    columns : list of str, optional
        Defaults to :data:`DEFAULT_COLUMNS`.
    kind : {"long", "short", "auto"}
        * ``long``  — one row per file (most explicit).
        * ``short`` — one row per ``$type/$batch`` folder, using a trailing
          ``"<folder>/"`` path pattern (RAMANMETRIX longest-pattern rule).
        * ``auto``  — ``short`` when every folder is internally homogeneous
          in its inferred labels, otherwise ``long``.

    Returns
    -------
    (columns, rows) : (list, list of dict)
    """
    columns = list(columns) if columns else list(DEFAULT_COLUMNS)
    paths = [_norm(p) for p in spectra_paths]

    if kind == "auto":
        kind = "short" if _folders_homogeneous(paths) else "long"

    if kind == "short":
        rows = _short_rows(paths, columns)
    else:
        rows = [_fill_row(p, columns) for p in paths]
    return columns, rows


def _folder_of(path):
    parts = PurePosixPath(_norm(path)).parts
    return "/".join(parts[:-1])


def _folders_homogeneous(paths):
    """True when all files within each folder share the same inferred labels."""
    groups = {}
    for p in paths:
        groups.setdefault(_folder_of(p), []).append(p)
    for members in groups.values():
        sig = {(d["type"], d["batch"], d["standard"])
               for d in (infer_from_path(m) for m in members)}
        if len(sig) > 1:
            return False
    return True


def _short_rows(paths, columns):
    rows = []
    seen = set()
    for p in sorted(paths):
        folder = _folder_of(p)
        pattern = (folder + "/") if folder else "*"
        if pattern in seen:
            continue
        seen.add(pattern)
        row = _blank_row(columns)
        inferred = infer_from_path(p)
        if "file" in row:
            row["file"] = pattern
        for k in ("type", "batch", "date"):
            if k in row and inferred.get(k):
                row[k] = inferred[k]
        if "standard" in row:
            row["standard"] = bool(inferred.get("standard", False))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _fmt(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    return "" if value is None else str(value)


def write_template_csv(columns, rows, out_path):
    """Write a metadata template to a CSV file (UTF-8)."""
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _fmt(row.get(c, "")) for c in columns})
    return out_path


def template_to_csv_string(columns, rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({c: _fmt(row.get(c, "")) for c in columns})
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Reading existing metadata tables
# ---------------------------------------------------------------------------

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f", ""}


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def read_metadata_table(path=None, data=None, ext=None):
    """Read a metadata table from a path or raw bytes/str.

    Supports CSV directly (stdlib) and XLS/XLSX via pandas if available.
    Returns a list of dict rows with original (string) cell values.
    """
    if ext is None and path is not None:
        ext = os.path.splitext(path)[1].lower()
    ext = (ext or ".csv").lower()

    if ext in (".xls", ".xlsx"):
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Reading XLS/XLSX metadata requires pandas (and openpyxl)."
            ) from exc
        src = path if path is not None else io.BytesIO(
            data if isinstance(data, (bytes, bytearray)) else data.encode()
        )
        df = pd.read_excel(src, dtype=str).fillna("")
        return df.to_dict("records")

    # CSV / text
    if path is not None:
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            text = fh.read()
    elif isinstance(data, (bytes, bytearray)):
        text = data.decode("utf-8-sig")
    else:
        text = data or ""
    # Auto-detect delimiter (RAMANMETRIX uses tab or comma in places).
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    return [dict(r) for r in reader]


def read_metadata_from_zip(zip_path, member):
    with zipfile.ZipFile(zip_path) as zf:
        raw = zf.read(member)
    return read_metadata_table(data=raw, ext=os.path.splitext(member)[1])


# ---------------------------------------------------------------------------
# Matching metadata rows to spectra (longest-pattern-wins)
# ---------------------------------------------------------------------------

def match_metadata(spectra_paths, meta_rows):
    """Apply metadata rows to spectra paths per RAMANMETRIX semantics.

    For each spectrum, every metadata row whose ``file`` (or ``path``) value is
    a prefix of the spectrum path applies; longer patterns override shorter
    ones. Special patterns ``"*"`` and ``"."`` match all otherwise-unlisted
    files. Returns a dict ``{path: merged_metadata_dict}`` containing only
    spectra with ``include`` truthy (default True).
    """
    paths = [_norm(p) for p in spectra_paths]

    # Separate wildcard rows from explicit-prefix rows.
    explicit = []   # (pattern, row)
    wildcard = []   # row
    for row in meta_rows:
        key = row.get("file", row.get("path", ""))
        key = _norm(key) if key else ""
        if key in ("*", ".", ""):
            wildcard.append(row)
        else:
            explicit.append((key, row))

    result = {}
    for p in paths:
        applicable = [(len(pat), row) for pat, row in explicit
                      if p == pat or p.startswith(pat)]
        merged = {}
        # Wildcards first (lowest priority), then by ascending pattern length.
        for row in wildcard:
            merged.update(_clean_row(row))
        for _, row in sorted(applicable, key=lambda t: t[0]):
            merged.update(_clean_row(row))

        include = merged.get("include", True)
        inc_bool = _coerce_bool(include)
        if inc_bool is False:
            continue
        result[p] = merged
    return result


def _clean_row(row):
    """Drop the path key and empty cells; coerce known booleans."""
    out = {}
    for k, v in row.items():
        if k in ("file", "path"):
            continue
        if v is None or str(v).strip() == "":
            continue
        if k in BOOL_DEFAULTS:
            b = _coerce_bool(v)
            out[k] = b if b is not None else v
        else:
            out[k] = v
    return out


__all__ = [
    "DEFAULT_COLUMNS", "CORE_COLUMNS", "SPECTRA_EXTS",
    "is_metadata_file", "is_spectrum_file", "parse_timestamp",
    "infer_from_path", "list_spectra_in_zip", "list_metadata_in_zip",
    "generate_template", "write_template_csv", "template_to_csv_string",
    "read_metadata_table", "read_metadata_from_zip", "match_metadata",
]
