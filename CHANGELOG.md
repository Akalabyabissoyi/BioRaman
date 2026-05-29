# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

> **Output-affecting changes** (anything that alters analytical results —
> algorithms, numerical defaults, preprocessing) are flagged with **[OUTPUT]**
> and always trigger at least a MINOR version bump.

## [Unreleased]

## [0.8.0] - 2026-05-29

### Added
- **[OUTPUT]** Peak identification window: detects spectral peaks and matches
  them against a Raman band library (with custom/CSV band-list import).
- **[OUTPUT]** Spectra comparison window: side-by-side comparison of selected
  spectra, components, and endmembers.

### Changed
- Renamed the application to **BioRaman** and the entry-point file to
  `bioraman.py` (previously the versioned `Raman_map_explorer_v11.py`) ahead of
  the first public release.
- Relicensed from MIT to **GNU GPL v3.0-or-later** (copyleft) so that any
  distributed modified or renamed version must remain open source and retain
  the original authorship. Added a GPL header to `bioraman.py` and a license
  notice to the About dialog.

## [0.7.0] - 2026-05-29

First version under semantic versioning. Prior builds (v1–v6) were tracked
informally and are summarised below.

### Added
- **[OUTPUT]** Cluster analysis: K-means and agglomerative clustering with
  selectable component count, colour-coded cluster map, and per-cluster mean spectra.
- **[OUTPUT]** MCR-ALS multivariate curve resolution (non-negative factorisation)
  with abundance maps and recovered pure spectra.
- **[OUTPUT]** N-FINDR endmember extraction with NNLS-derived abundance maps.
- Spectral tools: resampling to an equidistant grid, spatial crop, rotation,
  and background/substrate subtraction.

### Notes on earlier (untracked) versions
- v6 — 3D confocal volume viewer (volume scatter, orthogonal slices, surface,
  multi-band RGB).
- v1–v5 — Core map exploration, baseline correction, peak fitting, smoothing,
  and export functionality.

[Unreleased]: https://example.com/compare/v0.8.0...HEAD
[0.8.0]: https://example.com/releases/tag/v0.8.0
[0.7.0]: https://example.com/releases/tag/v0.7.0
