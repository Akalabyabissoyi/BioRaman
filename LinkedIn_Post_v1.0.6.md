**BioRaman just got a big update.**

For the last few months I've been building BioRaman — a free, open-source desktop app for Raman hyperspectral map analysis, made for biophysics labs that want to go from raw spectra to publication-ready figures without writing code. Here's what's new in the latest release.

**A scalable PCA Studio, built into the app.**
The old PCA window worked, but it choked on big maps. The new PCA Studio runs an out-of-core engine (batched IncrementalPCA) that streams through 10⁵–10⁶ spectra instead of loading everything into memory at once — so there's effectively no dataset size limit. It also adds robust/sparse PCA variants, T² and Q-residual diagnostics, PLS-DA/LDA and clustering on the scores, and colour-blind-safe 600 dpi / vector figure export. You can analyse a map you already have open, or load files straight into it.

**A bigger peak finder.**
The band-assignment library has been expanded with a lot more reference data, so peaks get matched to biomolecules more reliably across proteins, lipids, nucleic acids and more.

**FlyHash — anomaly & region detection.**
A fly-brain-inspired method for spotting unusual or distinct regions in a hyperspectral cube. Useful when you don't know in advance what you're looking for.

On top of that: MCR-ALS, N-FINDR and DCLS/NNLS unmixing, k-means & hierarchical clustering, per-pixel QC maps, full-spectrum library search (RRUFF/ROD/SLoPP), 3D confocal volume rendering, headless batch mode with reproducible JSON recipes, and one-click HTML reports.

No Python install needed — double-click apps for Windows and macOS.

Free and open-source (MIT). Feedback and bug reports very welcome.

Download: github.com/Akalabyabissoyi/BioRaman/releases
Cite: doi.org/10.5281/zenodo.20562222

Gibson Group · University of Manchester

#RamanSpectroscopy #Biophysics #OpenSource #Microscopy #PCA #DataAnalysis #ScientificSoftware
