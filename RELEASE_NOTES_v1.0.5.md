# BioRaman v1.0.5-1

Scientific-robustness hardening of the analysis core (from a static assessment),
plus PCA usability improvements and a user-initiated preprocessing workflow.
The fixed build is now the canonical `bioraman.py`, and the version string is
unified at **1.0.5** across `__version__`, `VERSION`, and the macOS bundle spec.

## Highlights

- **Cosmic-ray removal no longer erases sharp Raman bands.** Detection switched
  from a first-derivative Z-score (which fired on the apex of any sharp band) to
  width-aware median-residual rejection of narrow, positive-going spikes. Default
  threshold raised 8 → 12, and the number of interpolated points is now reported.
- **Quantitatively faithful area normalisation.** Baseline subtraction and the
  normalisation area no longer half-wave-rectify the signal, removing a noise-
  dependent (SNR) scaling bias. Verified noise-independent across a 0.5–4× sweep.
- **Less optimistic classification metrics.** `StandardScaler` is now fit inside
  each cross-validation fold (scikit-learn `Pipeline`) instead of on the whole
  matrix beforehand, reducing leakage.
- **Robust file loading.** The ASCII reader keeps the *modal* row width rather
  than the maximum, so a single malformed line can no longer discard every valid
  row; NaN/Inf samples are sanitised before processing.
- **No silent failures.** Baseline-correction failures are counted and surfaced
  in the processing report instead of being swallowed.

## New features

- **User-initiated preprocessing** — "Apply & Reprocess Now" in Preprocessing →
  Settings, a 🔁 Reprocess toolbar button, and a post-run summary (cosmic points
  removed, baseline failures).
- **PCA scree guidance** — objective number-of-component suggestions (95%
  cumulative variance, Kaiser, broken-stick) marked on the scree plot and shown
  in the status line.
- **95% Hotelling-T² confidence ellipses** on PCA score plots (F-distribution,
  finite-sample), replacing the previous fixed ~86% (2σ) ellipse.

## Fixes (assessment findings)

| ID | Fix |
|----|-----|
| F1 | Width-aware cosmic-ray removal; preserves real bands |
| F2 | Modal-width ASCII parsing; corrupt lines can't discard valid data |
| F3 | Non-rectifying area normalisation (no SNR bias) |
| F4 | Per-fold scaling in cross-validation |
| F5 | Baseline failures reported, not swallowed |
| F6 | NaN/Inf input sanitisation |
| F7 | Savitzky–Golay window forced odd |
| F8 | Version unified at 1.0.5 |

## Notes

- Items marked output-affecting in the changelog (F1, F3, F4, area handling, the
  95% ellipse) can change numerical results versus 1.0.2/1.0.3 — re-run analyses
  for direct comparability.
- Recommended pre-release gate: run `pytest -q` and
  `python validation/validate_concentration.py` in an environment with SciPy and
  scikit-learn installed.

See `CHANGELOG.md` for the full, itemised list.
