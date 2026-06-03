#!/usr/bin/env python3
"""
BioRaman concentration-analysis validation harness.

Two modes:

1. Synthetic phantom (default, no data needed) — builds a map with KNOWN
   component concentrations from synthetic reference spectra, runs the same
   component_fit() used by the GUI, and reports recovered-vs-true error.
   Exits non-zero if the recovery error exceeds the tolerance, so it can gate
   a release.

       python validation/validate_concentration.py

2. Real data (the Renishaw WiRE "Tablet" exercise or your own sample) — fits a
   real map against real reference spectra and prints the overall %, which you
   compare to the value reported by WiRE's Concentration Estimate procedure.

       python validation/validate_concentration.py \
           --map Tablet.wdf --refs aspirin.wdf caffeine.wdf paracetamol.wdf \
           --method NNLS --background 4 --normalise "None (quantitative)"

See validation/README.md for the full WiRE comparison protocol.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_bioraman():
    spec = importlib.util.spec_from_file_location(
        "bioraman", os.path.join(ROOT, "bioraman.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_validation(br, n_pixels=400, n_points=300, tol_pct=1.0,
                         seed=0):
    """Known-truth phantom: returns (max_abs_error_percent, ok)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(400, 3200, n_points)

    def band(c, w, a=1.0):
        return a * np.exp(-((x - c) ** 2) / (2 * w ** 2))

    # three "pure" reference spectra (multi-band, distinct)
    refs = np.stack([
        band(1003, 22) + band(1600, 16) + band(2940, 38),       # comp A
        band(780, 26) + band(1320, 18) + band(1700, 20),        # comp B
        band(620, 20) + band(1450, 22) + band(2890, 30),        # comp C
    ])
    K = refs.shape[0]

    # random non-negative concentrations per pixel, then build the data
    W_true = rng.random((n_pixels, K))
    flat = W_true @ refs
    flat += 0.002 * flat.max() * rng.standard_normal(flat.shape)   # 0.2% noise

    true_overall = (W_true.sum(0) / W_true.sum() * 100)

    print("Synthetic phantom — known concentrations vs recovered\n")
    worst = 0.0
    for method in ("DCLS", "NNLS"):
        out = br.component_fit(flat, refs, method=method,
                               preprocess="Spectrum", normalise="None",
                               background_order=0)
        rec = out["overall"]
        err = np.abs(rec - true_overall)
        worst = max(worst, float(err.max()))
        print(f"  {method}:")
        for k in range(K):
            print(f"    comp {k+1}:  true {true_overall[k]:6.2f}%   "
                  f"recovered {rec[k]:6.2f}%   |Δ| {err[k]:.2f}")
        # per-pixel dominant-component classification accuracy
        acc = (out["conc"].argmax(1) == W_true.argmax(1)).mean()
        print(f"    overall-sum {rec.sum():.2f}%   "
              f"pixel dominant-class accuracy {acc*100:.1f}%\n")

    ok = worst <= tol_pct
    print(f"Worst overall error: {worst:.3f}%   tolerance: {tol_pct:.2f}%   "
          f"=> {'PASS' if ok else 'FAIL'}")
    return worst, ok


def file_validation(br, map_path, ref_paths, method, background, normalise):
    """Fit a real map against real reference spectra; print overall %."""
    r = br._open_raman_any(map_path)
    # the GUI preprocesses on load; here we use raw spectra (quantitative)
    cube = np.asarray(r.spectra, dtype=float)
    if cube.ndim == 1:
        cube = cube[None, None, :]
    elif cube.ndim == 2:
        cube = cube[None, :, :]
    Y, X, Wn = cube.shape
    xax = np.asarray(r.xdata, dtype=float)
    refs = np.stack([br._load_reference_spectrum(p, xax) for p in ref_paths])
    out = br.component_fit(cube.reshape(Y * X, Wn), refs, method=method,
                           preprocess="Spectrum", normalise=normalise,
                           background_order=background)
    print(f"\nMap: {os.path.basename(map_path)}   ({X}×{Y} pixels)")
    print(f"Method: {method}   background poly: {background}   "
          f"normalise: {normalise}\n")
    print("Concentration estimate (compare to WiRE):")
    for p, v in zip(ref_paths, out["overall"]):
        print(f"  {os.path.basename(p):<24} {v:6.2f}%")
    print(f"  {'sum':<24} {out['overall'].sum():6.2f}%")
    print(f"  mean lack-of-fit: {out['lof'].mean():.2f}%")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", help="real map file (e.g. Tablet.wdf)")
    ap.add_argument("--refs", nargs="+", default=[],
                    help="reference component spectra files")
    ap.add_argument("--method", default="NNLS", choices=["NNLS", "DCLS"])
    ap.add_argument("--background", type=int, default=0)
    ap.add_argument("--normalise", default="None (quantitative)")
    ap.add_argument("--tol", type=float, default=1.0,
                    help="synthetic-phantom tolerance in percent")
    args = ap.parse_args(argv)

    br = _load_bioraman()

    if args.map and args.refs:
        file_validation(br, args.map, args.refs, args.method,
                        args.background, args.normalise)
        return 0

    _, ok = synthetic_validation(br, tol_pct=args.tol)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
