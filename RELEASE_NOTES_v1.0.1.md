# BioRaman 1.0.1

*Raman Hyperspectral Map Analysis for Biophysics — Gibson Group, University of Manchester*

First public release under the MIT License, with standalone Windows and macOS
apps built automatically and attached below.

## Highlights

- **Now MIT-licensed** — free to use, modify and redistribute, including in
  commercial and proprietary work.
- **Standalone downloads** — no Python required. Grab `BioRaman-Windows.zip` or
  `BioRaman-macOS.zip` below, unzip, and run.
- **Smooth, publication-quality maps** by default, with an optional faithful
  pixel-exact view (set any map window's *Display σ* to 0).

## What's in this release

- Interactive 2-D Raman map exploration with band-intensity and band-ratio maps
- Baseline correction, Savitzky–Golay smoothing and reproducible JSON recipes
- Spectral unmixing: **MCR-ALS** and **N-FINDR** with abundance maps
- **K-means / hierarchical clustering** with per-cluster mean spectra and
  silhouette scores
- **Component analysis (DCLS / NNLS)** concentration maps and % lack-of-fit
- **Particle statistics** (Otsu binarisation, equivalent-circle diameter, counts)
- 3D confocal volume reconstruction and publication-quality rendering
- Multi-format input (Renishaw `.wdf`, WITec `.wip`, ASCII) and PNG/PDF/CSV export
- Headless command-line / batch mode and a one-click HTML analysis report

## Changes since 0.10.0

- Relicensed from GPL-3.0-or-later to **MIT**; updated headers, About dialog,
  `CITATION.cff` and `README`.
- Added Gibson Group affiliation (https://gibsongroupresearch.com/) and both
  maintainer addresses to the package metadata.
- About dialog now shows the live version string.
- Optional faithful pixel view (Display σ = 0); smooth rendering remains default.
- Minor plot/robustness fixes (axis units, exception handling).

## Install

**Windows** — download `BioRaman-Windows.zip`, unzip, run `BioRaman.exe`.
**macOS** — download `BioRaman-macOS.zip`, unzip; the first time, right-click
`BioRaman.app` → **Open** → **Open** (the app is unsigned). Apple Silicon build.
**From source** — `pip install -r requirements.txt` then `python bioraman.py`.

## Cite

If you use BioRaman in your work, please cite it — see `CITATION.cff`. A Zenodo
DOI for this release will be added here once minted.

---

Developed by Akalabya Bissoyi · Gibson Group, University of Manchester ·
akalabya.bissoyi@manchester.ac.uk · bissoyi.akalabya@gmail.com
