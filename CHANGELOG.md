# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

> **Output-affecting changes** (anything that alters analytical results —
> algorithms, numerical defaults, preprocessing) are flagged with **[OUTPUT]**
> and always trigger at least a MINOR version bump.

## [0.10.0] - 2026-06-02

### Added
- **Preprocessing recipes** — save/load all preprocessing parameters as a JSON
  recipe (`Preprocessing → Save/Load Recipe`) for reproducible, shareable
  analyses.
- **Non-destructive reprocessing** (`Preprocessing → Reprocess`) — the raw cube
  is now retained on load, so a changed recipe can be re-applied without
  re-reading the file from disk.
- **Batch processing** (`File → Batch Process Folder…`, and the `run_batch()`
  API) — apply one recipe to every supported file in a folder and export each
  processed cube plus a `batch_summary.csv`.
- **Headless command-line mode** — `python bioraman.py --input <file|folder>
  --out <dir> [--recipe r.json] [--format .npz] [--recursive]` runs batch
  preprocessing with no GUI, for pipelines and servers. `--save-recipe` writes
  the default recipe; `--version` prints the version.
- **Quality-control maps** (`Preprocessing → Quality Control Maps…`) — per-pixel
  signal-to-noise ratio, total/maximum intensity and detector-saturation maps,
  exportable to CSV/NPY.
- **Analysis report** (`File → Save Analysis Report (HTML)…`) — a
  self-contained HTML report bundling the mean-intensity map, mean spectrum,
  preprocessing recipe and processing log.
- **Session save/restore** (`File → Save/Load Session…`) — persist the source
  path, recipe and view settings to a `.bioses` file and restore later.
- **HDF5 export** (`.h5`) added to Save Processed Data.
- **[OUTPUT]** Cluster analysis now reports the mean **silhouette score** as a
  cluster-separation quality metric.

- **Confocal volume reconstruction** — depth-resolved `.wdf` files (a Z-stack
  encoded as `count = nx·ny·nz`) are now rebuilt into the true Z×Y×X×W volume
  from the stage xpos/ypos/zpos arrays; the 2-D map view averages over depth.
- **Publication-quality volume renderer** (`Analysis → Publication Volume
  (Plotly)`, and the "HQ Volume" toolbar button) with two modes: a **solid
  multi-band iso-surface** view that shows several bands together as different
  coloured chemical surfaces (the WiRE/paper "terrain" look), and a translucent
  **cloud** view of a single band. Controls for band ranges/colours, iso level,
  Gaussian smoothing/upsampling, Z-stretch, opacity and background; robust
  percentile contrast; interactive HTML + high-resolution PNG export. Plotly
  (and kaleido for PNG) are auto-installed on first use.
- **Concentration estimate analysis (reference CLS/NNLS)** in the volume
  renderer — load measured reference component spectra, fit every voxel by
  non-negative least squares (no normalisation, per the Renishaw "concentration
  estimate" method), colour the 3-D volume by the dominant component with a
  purple lack-of-fit (LoF) channel, and report the overall % of each component
  in the title/legend.
- **Automatic dependency bootstrap** — on first run the script installs the
  native `renishawWiRE` reader into the current interpreter if it is missing
  (so `.wdf` map geometry is recovered without manual setup). Set
  `BIORAMAN_NO_AUTOINSTALL=1` to disable for offline/frozen builds.

### Fixed
- Loading some `.wdf` maps crashed with `ValueError: not enough values to unpack
  (expected 3, got 1)` when the backend returned a 1-D/2-D array instead of a
  Y×X×W cube. `.wdf` files now prefer the native renishawWiRE reader, and every
  backend's output is normalised to a 3-D cube (using the file's map geometry
  where available). `preprocess_map()` also defensively promotes 1-D/2-D input.
- Large maps returned as a flattened 1-D/2-D array (e.g. an `IndexError` where
  the spectral axis was pixels×points long) are now reshaped back to a cube
  using the wavenumber count (len(xdata)); transposed (W, N) layouts are
  detected and corrected.

## [0.9.0] - 2026-06-02

### Added
- **Save Processed Data** (`save_processed()`, File menu / toolbar /
  `Ctrl+Shift+S`). Exports the full preprocessed spectral cube
  (baseline-corrected, smoothed, normalised, cosmic-ray-cleaned) together with
  the wavenumber axis to open, re-loadable formats: `.npz` (lossless NumPy
  archive), `.csv` / `.txt` / `.dpt` (Renishaw `#X #Y #Wave #Intensity` long
  format for maps, which round-trips through the built-in text reader; single
  spectra as two columns), and `.mat` (MATLAB, via SciPy). Proprietary
  instrument containers (`.wdf` / `.wip` / `.spc`) are binary and cannot be
  rewritten, so processed data is exported to these open formats instead.
- Consolidated the codebase to a single canonical entry point, `bioraman.py`
  (merged the `bioraman_universal_WITEC` development build; removed duplicate
  and superseded script copies).

## [Unreleased]

### Added
- Native ASCII reader (`_TextReader`) for **`.txt` / `.csv` / `.dpt` / `.dat`
  / `.jdx` / `.asc`** files. Auto-detects the delimiter (tab / semicolon /
  comma / whitespace), skips `%` comment and text-header lines, handles
  European decimal commas, and accepts single 2-column spectra or multi-column
  exports (wavenumber in the first column or first row). It also understands
  the **Renishaw "long" named-column export** (`#X #Y #Wave #Intensity` scans,
  `#Time #Wave #Intensity` series, or `#Wave #Intensity` single spectra),
  reshaping the long table back into a proper spectrum / line / map cube and
  ordering wavenumbers ascending. This removes the previous dependence on an
  external `raman_io` module for text data.

### Fixed
- Opening `.txt`/`.csv` spectra failed in builds without the optional
  `raman_io` package (the loader fell back to the WDF reader). These formats
  now load through the built-in `_TextReader`.

### Added
- WITec **`.wip`** file support alongside Renishaw `.wdf`. A `_WITecReader`
  adapter reads `.wip` via RamanSPy (`ramanspy.load.witec`, with a
  photonicdata `wip_loader` fallback) and promotes single spectra / line scans
  to a map cube for the existing pipeline. `.wip` is accepted in the Open
  dialog and inside RAMANMETRIX dataset ZIPs. Requires `pip install ramanspy`.
- RAMANMETRIX-compatible data input (`raman_metadata.py` + **File → 📦
  RAMANMETRIX Dataset**). Import a ZIP of spectra plus a metadata table
  (CSV/XLS/XLSX whose name contains "metadata"), with labels resolved using
  the documented longest-pattern-wins rule, `include` filtering, and `*`/`.`
  wildcards. Labels not present in the table are inferred from the
  `.../$type/$batch/` folder structure and embedded `YYMMDD_hhmmss`
  timestamps. Generate metadata templates (short / long / auto layouts; full
  or core column sets) and load any single spectrum from the archive straight
  into the map viewer. See
  https://docs.ramanmetrix.eu/documentation/Data.html

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

[Unreleased]: https://example.com/compare/v0.10.0...HEAD
[0.10.0]: https://example.com/releases/tag/v0.10.0
[0.9.0]: https://example.com/releases/tag/v0.9.0
[0.8.0]: https://example.com/releases/tag/v0.8.0
[0.7.0]: https://example.com/releases/tag/v0.7.0
