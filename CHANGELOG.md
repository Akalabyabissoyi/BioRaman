# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

> **Output-affecting changes** (anything that alters analytical results —
> algorithms, numerical defaults, preprocessing) are flagged with **[OUTPUT]**
> and always trigger at least a MINOR version bump.

## [1.1.0] - 2026-06-20

New feature release: a scalable PCA workflow and an expanded peak-assignment
library. No changes to existing analytical defaults.

### Added
- **PCA Studio — scalable, publication-grade PCA, built into BioRaman.** A new
  *Analysis → PCA Studio* window backed by an out-of-core engine
  (`pca_core.py`, batched `IncrementalPCA`) that streams through 10⁵–10⁶
  spectra instead of loading a single dense matrix, removing the practical
  dataset-size ceiling of the original PCA window. It adds standard / robust /
  sparse PCA variants, Hotelling's T² and Q-residual (SPE) diagnostics with 95%
  control limits, PLS-DA / LDA classification and k-means / agglomerative /
  HDBSCAN clustering on the scores, and colour-blind-safe (Okabe–Ito) 600 dpi
  PNG / vector PDF/SVG export with density rendering for large point clouds.
  PCA Studio runs both standalone (`python pca_studio.py`) and embedded, and
  can ingest the map currently loaded in BioRaman or load its own files
  (`.wdf`, `.csv`, `.txt`, `.dpt`, `.xlsx`).

### Changed
- `bioraman.py` is now the single canonical application file and includes the
  PCA Studio integration.
- `.gitignore` updated to exclude large / licence-restricted raw data and local
  working copies from the repository.

## [1.0.6] - 2026-06-11

Usability improvements to the PCA window. No analytical results are affected.

### Added
- **Adjustable marker size on the PC1 vs PC2 scores plot.** A new *Point size*
  control lets dense score clouds resolve instead of overlapping into a single
  blob; the default marker size is reduced accordingly. The setting also applies
  to the publication-quality panel export.
- **Manual aspect ratio for the scores plot.** A new *Aspect (H/W)* control sets
  the panel's height-to-width ratio (`0` = auto). Stretching the plot helps
  separate clusters that are compressed along one principal component.

### Changed
- **The PCA window's left control panel is now scrollable.** Previously the
  lower controls could be clipped off the bottom edge on shorter windows; all
  options are now reachable via scrollbar or mouse wheel.

## [1.0.5] - 2026-06-10

Robustness & scientific-correctness hardening from a static assessment of the
analysis core. **The version string is now unified at 1.0.5** across
`__version__`, the `VERSION` file, and the filename; the test and validation
harnesses now import the shipped build by name.

### Fixed
- **[OUTPUT] Cosmic-ray removal no longer erases sharp Raman bands (F1).** The
  detector previously thresholded the modified Z-score of the first derivative,
  which fires on the apex of any sharp band and could clip it to baseline. It is
  replaced by width-aware median-residual rejection: only narrow (≤ *max spike
  width* px), positive-going outliers are interpolated. Default Z threshold
  raised 8 → 12 (more conservative). A run summary now reports how many points
  were interpolated.
- **[OUTPUT] Area normalisation is no longer noise-biased (F3).** Baseline
  subtraction and the area used for normalisation no longer half-wave-rectify
  the signal (`clip(…,0)`), which inflated the area in proportion to noise and
  coupled the normalisation scale to SNR. The un-rectified, ~zero-mean signal is
  integrated, giving a noise-independent scale (verified constant across 0.5–4×
  noise).
- **Baseline-correction failures are surfaced, not swallowed (F5).** A failed
  baseline fit now leaves the spectrum uncorrected *and* is counted in the
  processing report (e.g. "ASLS — FAILED on N/M spectra") instead of silently
  claiming a correction was applied.
- **Non-finite inputs are sanitised (F6).** NaN/Inf samples are interpolated
  away before processing so they cannot poison baseline fitting, normalisation,
  PCA or least-squares solvers.
- **Savitzky–Golay window is always odd (F7).** An even polynomial order with a
  small window could produce an even window length and raise `ValueError`;
  the window is now forced odd and the polyorder clamped below it.
- **ASCII reader keeps the modal row width, not the maximum (F2).** A single
  malformed line with extra columns no longer discards every well-formed row;
  skipped rows are reported.
- **[OUTPUT] Cross-validation leakage reduced (F4).** The PLS-DA/LDA classifier
  now fits `StandardScaler` inside each CV fold via a `Pipeline` on the unscaled
  features, instead of scaling the whole matrix beforehand. (Global Hotelling-T²
  outlier screening remains a user-toggled pre-filter, now documented.)

### Added
- **Objective number-of-PC guidance on the scree plot (PCA).** The scree panel
  now overlays cumulative variance and marks three standard criteria — 95%
  cumulative variance, Kaiser (eigenvalue > 1 on autoscaled data, or the
  average-eigenvalue rule on mean-centred data) and the broken-stick model —
  and the status line reports the suggested component counts. Addresses the
  common, result-invalidating mistake of choosing the number of PCs by eye
  (Hanson 2017; Khristoforova 2022; Vajna 2011).
- **[OUTPUT] 95% Hotelling-T² confidence ellipses on PCA score plots.** The
  per-group ellipses are now proper 95% T² confidence regions (F-distribution,
  finite-sample, χ² fallback) instead of the previous fixed 2·√λ (~86%) ellipse,
  matching standard chemometrics practice (Gurian 2020).
- **User-initiated preprocessing.** Preprocessing → Settings now offers
  **"Apply & Reprocess Now"**, a **🔁 Reprocess** toolbar button, and the
  existing menu action re-run the recipe on the in-memory raw data without a
  reload, followed by a summary of cosmic points removed and any baseline
  failures.

## [1.0.2] - 2026-06-05

### Fixed
- **Standalone app started very slowly and could not open map files.** Two
  causes addressed:
  - The PyInstaller build is now **one-folder** instead of one-file, so the app
    no longer re-extracts its ~100 MB scientific stack to a temp directory on
    every launch. Startup drops from minutes to seconds.
  - The automatic dependency installer (`_ensure_package`) no longer runs inside
    a frozen/standalone build (`sys.frozen`). Previously it tried to `pip
    install renishawWiRE` using the app itself as the interpreter, which hung
    and left the `.wdf` reader unavailable. All readers are now bundled at build
    time, so map files open immediately.
- Bundled `renishawWiRE` and its submodules explicitly in the build spec.
- Updated the release workflow to package the one-folder Windows build
  (the whole `BioRaman` folder rather than a single `.exe`).

## [1.0.1] - 2026-06-05

### Changed
- **Relicensed from GNU GPL v3.0-or-later to the MIT License.** Updated
  `LICENSE`, the source-file headers, the in-app About dialog, `CITATION.cff`
  and `README.md` accordingly. BioRaman may now be used, modified and
  redistributed (including in proprietary work) under MIT terms.
- **Metadata / attribution** — added the group affiliation
  *Gibson Group, University of Manchester* (https://gibsongroupresearch.com/)
  and both maintainer addresses (`akalabya.bissoyi@manchester.ac.uk`,
  `bissoyi.akalabya@gmail.com`) to the package metadata, About dialog,
  `CITATION.cff` and `README`.
- The About dialog now shows the live `__version__` instead of a hard-coded
  string.

### Added
- **Faithful pixel view (optional)** — every map window's *Display σ* control
  can be set to `0` to disable display smoothing and show each measured pixel
  as a discrete cell (`interpolation="nearest"`). The default remains the
  smooth, presentation-quality view (bilinear interpolation with light
  Gaussian smoothing, `sigma=0.8`).

### Fixed
- Minor plot/robustness fixes: added axis units to the report mean-intensity
  map (`X (px)`/`Y (px)`); replaced bare `except:` clauses with
  `except Exception:`.

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
- **Component Analysis (DCLS / NNLS)** (`Analysis → Component Analysis`) —
  supervised least-squares fitting of reference spectra to a 2-D map, producing
  per-component **concentration maps**, a **% lack-of-fit** map and overall
  **concentration estimates**, matching the Renishaw WiRE workflow. Options:
  NNLS or DCLS, Spectrum / 1st / 2nd derivative, normalisation
  (none / vector / mean-centre+unit-variance), and a modelled polynomial
  background. Export maps + estimates to CSV.
- **Particle Statistics** (`Analysis → Particle Statistics`) — Otsu (or manual)
  binarisation of any single-band / concentration map, connected-component
  labelling, per-particle area and **equivalent circle diameter**, area %,
  counts, a size-distribution histogram, edge/size filtering and CSV export.
- **Full-spectrum library search** (`Analysis → Library Search`) — identify a
  pixel / ROI / whole-map spectrum by Pearson-correlation matching against a
  user-supplied reference library (RRUFF, Raman Open Database, SLoPP, or any
  folder of spectra), with raw / SNV / 1st-derivative preprocessing, overlap-
  aware scoring, a ranked match table, query-vs-match overlay, and CSV export.
  No library is bundled with BioRaman (the user loads their own downloaded
  one), so there are no licensing constraints on the software.
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

- **Higher-quality map rendering** — abundance (MCR-ALS, N-FINDR) and
  concentration (Component Analysis) maps now use NaN-aware Gaussian smoothing,
  bilinear interpolation and robust 2–98 percentile contrast (shared `show_map`
  helper), giving smooth, publication-quality images instead of noisy/blocky
  pixels.

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

[Unreleased]: https://example.com/compare/v1.0.2...HEAD
[1.0.2]: https://example.com/releases/tag/v1.0.2
[1.0.1]: https://example.com/releases/tag/v1.0.1
[0.10.0]: https://example.com/releases/tag/v0.10.0
[0.9.0]: https://example.com/releases/tag/v0.9.0
[0.8.0]: https://example.com/releases/tag/v0.8.0
[0.7.0]: https://example.com/releases/tag/v0.7.0
