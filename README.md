# BioRaman

*Raman Hyperspectral Map Analysis for Biophysics*

<!-- DOI: after minting on Zenodo, replace XXXXXXX below (both places) with your
     Zenodo record ID and uncomment this badge line. -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

A desktop GUI (Tkinter + matplotlib) for exploring and analysing Raman
hyperspectral maps. It loads Renishaw WiRE / HORIBA LabSpec data and provides
baseline correction, peak fitting, spectral unmixing (MCR-ALS, N-FINDR),
clustering (K-means, hierarchical), and 3D confocal volume rendering.

## Features

- Interactive 2D Raman map exploration with band-ratio and peak-intensity maps
- Baseline correction and Savitzky–Golay smoothing
- Cluster analysis: K-means and agglomerative clustering with per-cluster mean spectra
- MCR-ALS multivariate curve resolution (non-negative factorisation)
- N-FINDR endmember extraction with NNLS abundance maps
- Spectral tools: resampling, spatial cropping, rotation, background subtraction
- Peak identification against a Raman band library
- Side-by-side spectra comparison
- 3D confocal volume viewer (volume scatter, orthogonal slices, surface, multi-band RGB)
- PNG / PDF / CSV export throughout
- Save preprocessed data (full spectral cube + wavenumber axis) to NPZ / HDF5 / CSV / TXT / MAT
- Multi-format input: Renishaw `.wdf`, WITec `.wip`, and ASCII `.txt`/`.csv`/`.dpt`/`.jdx`
- Reproducible preprocessing recipes (save/load as JSON) and non-destructive reprocessing
- Batch-process a whole folder, with a headless command-line mode for pipelines
- Per-pixel quality-control maps (SNR, intensity, saturation)
- One-click HTML analysis report and session save/restore
- Cluster validation via mean silhouette score

## Installation

Requires Python 3.9+.

```bash
pip install numpy scipy matplotlib pybaselines renishawWiRE pillow
pip install scikit-learn pandas seaborn openpyxl
```

Optional: install `ramanspy` to open WITec `.wip` files, and `h5py` for HDF5
export.

```bash
pip install ramanspy h5py
```

## Usage

Launch the graphical interface:

```bash
python bioraman.py
```

### Headless batch processing

Process a file or a whole folder from the command line — no GUI — using an
optional JSON preprocessing recipe:

```bash
# write the default recipe, edit it, then batch-process a folder
python bioraman.py --save-recipe recipe.json
python bioraman.py --input data/ --out processed/ --recipe recipe.json --format .npz
```

Recipes are also created and re-used inside the app via
**Preprocessing → Save/Load Recipe**.

## Versioning

This project follows [Semantic Versioning](https://semver.org/). Any change
that affects analytical output (algorithms, defaults, numerical results) is
released as at least a **MINOR** bump; breaking changes to file formats or the
public workflow are **MAJOR**. See [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use this tool in a publication, please cite it. Release-linked DOIs are
minted via Zenodo — see [CITATION.cff](CITATION.cff) and the Zenodo badge once
the repository is connected.

## License

Released under the GNU General Public License v3.0 — see [LICENSE](LICENSE).

This is copyleft: you are free to use, study, modify and redistribute
BioRaman, but any distributed modified version (including under a new name)
must also be released as open source under the GPL and must preserve the
original authorship and copyright notices.
