# Raman Map Explorer

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
- 3D confocal volume viewer (volume scatter, orthogonal slices, surface, multi-band RGB)
- PNG / PDF / CSV export throughout

## Installation

Requires Python 3.9+.

```bash
pip install numpy scipy matplotlib pybaselines renishawWiRE pillow
pip install scikit-learn pandas seaborn openpyxl
```

## Usage

```bash
python Raman_map_explorer_v7.py
```

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

Released under the MIT License — see [LICENSE](LICENSE).
