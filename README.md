# BioRaman

*Raman Hyperspectral Map Analysis for Biophysics*

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Version](https://img.shields.io/badge/version-1.0.2-blue.svg)

Developed in the [Gibson Group, University of Manchester](https://gibsongroupresearch.com/).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20562222.svg)](https://doi.org/10.5281/zenodo.20562222)

A desktop GUI (Tkinter + matplotlib) for exploring and analysing Raman
hyperspectral maps. It loads Renishaw WiRE / HORIBA LabSpec data and provides
baseline correction, peak fitting, spectral unmixing (MCR-ALS, N-FINDR),
clustering (K-means, hierarchical), and 3D confocal volume rendering.

## Features

- Interactive 2D Raman map exploration with band-ratio and peak-intensity maps
- Baseline correction and Savitzky–Golay smoothing
- Cluster analysis: K-means and agglomerative clustering with per-cluster mean spectra
- Component analysis (DCLS / NNLS) with reference spectra: concentration maps, % lack-of-fit, and concentration estimates (WiRE-style)
- Particle statistics: Otsu binarisation, equivalent circle diameter, area %, counts, histogram and CSV export
- MCR-ALS multivariate curve resolution (non-negative factorisation)
- N-FINDR endmember extraction with NNLS abundance maps
- Spectral tools: resampling, spatial cropping, rotation, background subtraction
- Peak identification against a Raman band library
- Full-spectrum library search against user-supplied reference libraries (RRUFF / Raman Open Database / SLoPP)
- Side-by-side spectra comparison
- 3D confocal volume reconstruction with a publication-quality renderer (solid / surface / cloud) and reference-spectra concentration (CLS) analysis
- 3D confocal volume viewer (volume scatter, orthogonal slices, surface, multi-band RGB)
- PNG / PDF / CSV export throughout
- Save preprocessed data (full spectral cube + wavenumber axis) to NPZ / HDF5 / CSV / TXT / MAT
- Multi-format input: Renishaw `.wdf`, WITec `.wip`, and ASCII `.txt`/`.csv`/`.dpt`/`.jdx`
- Reproducible preprocessing recipes (save/load as JSON) and non-destructive reprocessing
- Batch-process a whole folder, with a headless command-line mode for pipelines
- Per-pixel quality-control maps (SNR, intensity, saturation)
- One-click HTML analysis report and session save/restore
- Cluster validation via mean silhouette score

## Download (no Python needed)

Pre-built, double-click applications are attached to each
[release](../../releases/latest):

- **Windows** — download `BioRaman-Windows.zip`, unzip, open the `BioRaman`
  folder and run `BioRaman.exe`.
- **macOS** — download `BioRaman-macOS.zip`, unzip, and open `BioRaman.app`.
  Because the app is not code-signed, the first time you open it macOS will
  warn you. **Right-click the app → Open → Open**, or allow it under
  *System Settings → Privacy & Security*.

These bundles include everything BioRaman needs — no separate Python install.
The macOS build targets Apple Silicon; Intel-Mac users should run from source
(below).

## Install from source

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

## Reference spectral libraries

BioRaman does **not** bundle any spectral database, so there are no licensing
constraints on the software. For full-spectrum **Library Search**
(`Analysis → Library Search`), download a free library yourself and point the
tool at the folder:

- **RRUFF** — minerals, 14k+ spectra — https://rruff.info/zipped_data_files/raman/
- **Raman Open Database (ROD)** — open-access (CC0) — http://solsa.crystallography.net/rod/
- **SLoPP / SLoPP-E** — microplastics — https://rochmanlab.com/slopp-and-slopp-e-raman-libraries/

Please observe each database's own licence and cite it in your work.

## Testing & validation

Core analysis routines are factored into pure functions (`component_fit`,
`particle_stats`, `prep_spectra`, cube reconstruction, recipes, export) that are
covered by an automated test suite and a quantitative validation harness.

```bash
pip install -r requirements-dev.txt
pytest -q                                   # unit tests
python validation/validate_concentration.py # synthetic concentration validation
```

The validation harness checks that component-analysis concentration estimates
recover **known** synthetic concentrations within tolerance, and includes a
protocol (`validation/README.md`) for cross-checking against Renishaw WiRE's
Concentration Estimate on real data (the Tablet/aspirin/caffeine/paracetamol
exercise). Continuous integration (GitHub Actions) byte-compiles the app and
runs the tests + validation on every push.

> For quantitative work, run the WiRE cross-check and record the agreement
> before reporting concentrations in a publication.

## Versioning

This project follows [Semantic Versioning](https://semver.org/). Any change
that affects analytical output (algorithms, defaults, numerical results) is
released as at least a **MINOR** bump; breaking changes to file formats or the
public workflow are **MAJOR**. See [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use this tool in a publication, please cite it via its archived Zenodo
record. The concept DOI [10.5281/zenodo.20562222](https://doi.org/10.5281/zenodo.20562222)
always resolves to the latest release; see [CITATION.cff](CITATION.cff) for the
full metadata.

## License

Released under the **MIT License** — see [LICENSE](LICENSE).

You are free to use, study, modify and redistribute BioRaman, including in
proprietary and commercial work, provided the copyright notice and permission
notice are preserved.

## Authors & contact

Developed by **Akalabya Bissoyi** in the
[Gibson Group, University of Manchester](https://gibsongroupresearch.com/).

- akalabya.bissoyi@manchester.ac.uk
- bissoyi.akalabya@gmail.com
