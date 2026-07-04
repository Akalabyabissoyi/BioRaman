# BioRaman — Group Comparison module

Adds two capabilities that were previously not in the GUI:

1. **Group Comparison** — per-cell *custom* band ratio (any A/B windows) across
   multiple maps, grouped by condition, with **Mann–Whitney U**, **Hodges–Lehmann
   median difference + 95% CI**, and **Cliff's δ**, plus a publication-style box
   plot (the Figure-3a style).
2. **Group Mean-Spectra Overlay** — mean intracellular spectrum per group with a
   **±SD envelope** and shaded band-A / band-B windows (the Figure-3b style).

Preprocessing matches the GUI defaults (cosmic-ray mod-Z on 1st derivative thr 12 /
half-width 3 → per-spectrum minimum → AsLS λ=1e5, p=0.001 → Savitzky–Golay 11/3 →
area normalisation). Cells are segmented by Otsu on the CH-stretch signal.

## Add it to the GUI

In `bioraman.py`:

```python
import bioraman_group_compare as bgc
# in the Analysis menu setup:
am.add_command(label="⚖  Group Comparison…",
               command=lambda: bgc.open_group_compare(self))
```

That's the only change. The window lets you add groups of `.wdf` files, set the
band windows, **Run**, and **Export** PNG + PDF.

## Run standalone (no GUI wiring)

```bash
python bioraman_group_compare.py        # opens the window
```

## Headless / command line

```bash
python bioraman_group_compare.py \
  --group PBS   pbs-1.wdf pbs-2.wdf pbs-3.wdf pbs-4.wdf \
  --group JK100 jk100-1.wdf jk100-2.wdf jk100-3.wdf jk100-9.wdf \
  --banda 3085 3165 --bandb 3350 3550 --out ./group_out
```

Outputs: `group_boxplot.png/pdf`, `group_mean_spectra.png/pdf`,
`per_cell_ratios.csv`, `group_comparison.json`.

The per-cell values, group statistics and figures reproduce those produced by the
interactive Group Comparison window, so the command-line and GUI routes are
interchangeable for batch use.

Requires the packages BioRaman already uses: numpy, scipy, matplotlib,
pybaselines, renishawWiRE, scikit-image.
