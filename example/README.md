# Example data

No sample datasets are bundled with this repository at the moment. The original
Renishaw `.wdf` example files have been removed pending permission from Renishaw
to redistribute them.

## Using your own data

BioRaman works with your own files in any supported format:

- Renishaw `.wdf`
- WITec `.wip`
- ASCII `.txt`, `.csv`, `.dpt`, `.jdx`, `.dat`

Open one via **File → Open** in the GUI, or pass it on the command line.

> ⚠️ **Caution — large hyperspectral maps may be slow.**
> Full hyperspectral maps are fully supported, but loading and the heavier
> analyses (MCR-ALS, N-FINDR, clustering, 3D rendering) can be slow and
> memory-hungry on large files. For very large files the GUI shows a warning
> before loading; the window may appear to freeze while it computes — this is
> expected.
