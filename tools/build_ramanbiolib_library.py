#!/usr/bin/env python3
"""Convert the RamanBiolib open biomolecule database into BioRaman PCRS
reference-library CSVs.

Source : RamanBiolib  (https://github.com/mteranm/ramanbiolib)
         M. Teran, JJ. Ruiz, P. Loza-Alvarez, D. Masip, D. Merino,
         "Open Raman spectral library for biomolecule identification",
         Chemometrics and Intelligent Laboratory Systems 264 (2025) 105476.
         https://doi.org/10.1016/j.chemolab.2025.105476
Licence: database content is ODbL-1.0 (attribution + share-alike).

Each reference spectrum is reconstructed as a sum of Gaussian peaks placed at
RamanBiolib's published peak positions and relative intensities (the same
approach BioRaman already uses in build_pcrs_library.py). Output files use the
BioRaman library convention:  <Category>__<component>.csv  with two columns
(wavenumber, intensity), min-max normalised on the standard BioRaman axis.

Run:  python build_ramanbiolib_library.py PEAKS_CSV METADATA_CSV
(defaults point at the downloaded RamanBiolib db files.)
"""
import os, sys, csv, ast, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "pcrs_library")

# BioRaman standard synthetic axis (matches build_pcrs_library.py).
X = np.arange(150.0, 3300.0, 2.0)
FWHM = 12.0                       # cm-1, per synthesised band
SIGMA = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gauss(centre, amp):
    return amp * np.exp(-0.5 * ((X - centre) / SIGMA) ** 2)


def safe_name(s):
    s = s.strip()
    s = re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")
    return s or "component"


def category_from_type(t):
    """Top level of RamanBiolib's 'Type/Sub/Sub' tree -> BioRaman category."""
    if not t:
        return "Biomolecule"
    return safe_name(t.split("/")[0])


def load_metadata(path):
    types = {}
    refs = {}
    with open(path, newline="", encoding="utf-8") as f:
        # skip any leading non-csv lines from the saved fetch wrapper
        rows = f.read().splitlines()
    start = next(i for i, r in enumerate(rows) if r.startswith("id,component"))
    reader = csv.DictReader(rows[start:])
    for r in reader:
        cid = r.get("id", "").strip()
        if not cid:
            continue
        types[cid] = r.get("type", "")
        refs[cid] = r.get("reference", "")
    return types, refs


def load_peaks(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = f.read().splitlines()
    start = next(i for i, r in enumerate(rows) if r.startswith("id,component"))
    return list(csv.DictReader(rows[start:]))


def main():
    peaks_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ramanbiolib_raman_peaks_db.csv")
    meta_csv  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "ramanbiolib_metadata_db.csv")

    types, refs = load_metadata(meta_csv)
    peaks = load_peaks(peaks_csv)
    os.makedirs(OUT, exist_ok=True)

    written = []
    seen = {}
    for row in peaks:
        cid = row.get("id", "").strip()
        comp = (row.get("component") or "").strip()
        if not comp:
            continue
        try:
            pk = ast.literal_eval(row["peaks"])
            it = ast.literal_eval(row["intensity"])
        except Exception:
            continue
        if not pk:
            continue
        y = np.zeros_like(X)
        for c, a in zip(pk, it):
            y += gauss(float(c), float(a))
        m = y.max()
        if m <= 0:
            continue
        y = y / m

        cat = category_from_type(types.get(cid, ""))
        base = f"{cat}__{safe_name(comp)}"
        # de-duplicate repeated component names (multiple measurements)
        n = seen.get(base, 0) + 1
        seen[base] = n
        fname = base if n == 1 else f"{base}_{n}"
        path = os.path.join(OUT, fname + ".csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["wavenumber", "intensity"])
            for xi, yi in zip(X, y):
                w.writerow([f"{xi:.1f}", f"{yi:.5f}"])
        written.append(fname)

    print(f"Wrote {len(written)} RamanBiolib reference spectra to {OUT}")
    return written


if __name__ == "__main__":
    main()
