# Example data

Sample datasets for trying out BioRaman.

| File | Size | Notes |
|------|------|-------|
| `caffeine.wdf` | ~8 KB | Single-point spectrum — fast, good first test |
| `aspirin.wdf` | ~72 KB | Small map |
| `Tablet.wdf` | ~11 MB | **Large hyperspectral map** |
| `Advanced map analysis 1 - slides.pdf` | ~1 MB | Walkthrough slides |
| `my results.png` | ~1 MB | Example output |

> ⚠️ **Caution — `Tablet.wdf` is large.**
> It is a full hyperspectral map (~11 MB) and is included so you can test the
> heavier analyses (MCR-ALS, N-FINDR, clustering, 3D rendering) on realistic
> data. Loading and processing it can be **slow and memory-hungry**, especially
> the multivariate and 3D steps, and on older or low-RAM machines the GUI may
> appear to hang while it computes. This is expected. If you just want a quick
> look at the interface, start with `caffeine.wdf` or `aspirin.wdf` instead.
