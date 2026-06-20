#!/usr/bin/env python3
"""Generate the BioRaman PCRS starter reference library.

This builds a small, license-clean Raman reference library for personal use by
synthesising spectra from *published characteristic band positions* (peak
centre + relative intensity) for common materials.  No copyrighted spectra are
copied — each reference is a sum of Gaussian peaks placed at literature band
positions, which is enough for IDFinder's correlation-based matching to flag a
plausible identity.  To extend the library with real measured spectra
(RRUFF / SLoPP / Raman Open Database), just drop their .txt/.csv files into the
same folder on a machine with internet access.

Output: pcrs_library/<category>__<name>.csv  (two columns: wavenumber, intensity)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "pcrs_library")

# Common Raman axis (cm-1) used for every synthetic reference.
X = np.arange(150.0, 3300.0, 2.0)

# (category, name, [(centre_cm-1, rel_intensity, fwhm_cm-1), ...])
# Band positions taken from widely published Raman reference values.
MATERIALS = [
    # ── minerals (RRUFF-style) ────────────────────────────────────────────────
    ("mineral", "Quartz",
     [(128, .3, 10), (207, .45, 12), (354, .25, 12), (465, 1.0, 9),
      (696, .12, 12), (808, .15, 14), (1085, .18, 16)]),
    ("mineral", "Calcite",
     [(156, .25, 10), (282, .45, 12), (712, .35, 10), (1086, 1.0, 9),
      (1435, .12, 18)]),
    ("mineral", "Aragonite",
     [(153, .3, 10), (206, .35, 11), (705, .4, 10), (1085, 1.0, 9)]),
    ("mineral", "Anatase_TiO2",
     [(144, 1.0, 8), (197, .15, 12), (399, .35, 14), (513, .3, 14),
      (639, .4, 16)]),
    ("mineral", "Rutile_TiO2",
     [(143, .5, 12), (235, .3, 30), (447, .8, 18), (612, 1.0, 16)]),
    ("mineral", "Gypsum",
     [(414, .25, 12), (493, .3, 12), (619, .25, 12), (670, .2, 12),
      (1008, 1.0, 9), (1135, .25, 14)]),
    ("mineral", "Hematite_Fe2O3",
     [(225, .8, 14), (245, .35, 14), (292, 1.0, 16), (411, .55, 18),
      (612, .35, 20), (1320, .5, 60)]),
    ("mineral", "Apatite",
     [(432, .25, 12), (591, .3, 14), (961, 1.0, 9), (1046, .3, 16)]),
    ("mineral", "Pyrite_FeS2",
     [(343, .7, 12), (379, 1.0, 11), (430, .35, 14)]),
    ("mineral", "Albite_feldspar",
     [(290, .4, 16), (478, 1.0, 14), (507, .8, 14), (815, .2, 18)]),
    ("mineral", "Magnetite_Fe3O4",
     [(308, .3, 30), (540, .4, 40), (668, 1.0, 30)]),
    ("mineral", "Dolomite",
     [(176, .3, 12), (300, .4, 12), (725, .35, 10), (1098, 1.0, 9)]),

    # ── microplastics / polymers (SLoPP-style) ────────────────────────────────
    ("polymer", "Polyethylene_PE",
     [(1063, .5, 14), (1130, .55, 12), (1296, .7, 12), (1440, .8, 16),
      (2848, 1.0, 22), (2883, .95, 22)]),
    ("polymer", "Polypropylene_PP",
     [(809, .55, 12), (841, .6, 12), (973, .5, 12), (998, .6, 12),
      (1152, .5, 14), (1330, .4, 14), (1458, .7, 16), (2882, 1.0, 24)]),
    ("polymer", "Polystyrene_PS",
     [(620, .4, 12), (1001, 1.0, 8), (1031, .6, 10), (1583, .4, 14),
      (1602, .7, 12), (3054, .6, 22)]),
    ("polymer", "PET_polyester",
     [(1096, .4, 14), (1116, .45, 12), (1290, .5, 14), (1614, .7, 12),
      (1726, 1.0, 14), (3082, .4, 22)]),
    ("polymer", "PVC",
     [(638, .6, 14), (695, .55, 14), (1430, .7, 18), (2910, 1.0, 24)]),
    ("polymer", "Nylon_PA6",
     [(1064, .5, 14), (1128, .55, 12), (1296, .6, 12), (1440, .7, 16),
      (1635, .6, 14), (2900, 1.0, 24)]),
    ("polymer", "PMMA_acrylic",
     [(812, .5, 14), (1450, .6, 16), (1730, .8, 14), (2950, 1.0, 24)]),
    ("polymer", "PTFE_teflon",
     [(733, 1.0, 12), (1216, .4, 16), (1300, .45, 16), (1380, .3, 16)]),
    ("polymer", "Polycarbonate_PC",
     [(704, .5, 12), (886, .4, 14), (1111, .5, 14), (1235, .6, 14),
      (1602, 1.0, 12)]),

    # ── biomolecules (bio Raman) ──────────────────────────────────────────────
    ("bio", "Phenylalanine",
     [(622, .4, 12), (1004, 1.0, 8), (1032, .5, 10), (1208, .3, 14),
      (1606, .5, 12)]),
    ("bio", "Protein_amideI",
     [(1004, .5, 9), (1240, .5, 30), (1340, .4, 18), (1450, .7, 18),
      (1655, 1.0, 22)]),
    ("bio", "Lipid",
     [(1064, .4, 14), (1301, .7, 14), (1440, 1.0, 16), (1660, .6, 16),
      (2850, .95, 22), (2880, .9, 22)]),
    ("bio", "DNA_nucleicacid",
     [(668, .4, 12), (785, 1.0, 12), (1093, .8, 12), (1340, .6, 16),
      (1578, .7, 14)]),
    ("bio", "Cholesterol",
     [(548, .4, 12), (700, .6, 12), (1440, .9, 16), (1670, 1.0, 16)]),
    ("bio", "Beta_carotene",
     [(1008, .5, 10), (1157, .85, 10), (1520, 1.0, 12)]),
    ("bio", "Glucose",
     [(517, .5, 12), (911, .5, 12), (1060, .7, 14), (1126, 1.0, 12)]),
    ("bio", "Collagen",
     [(855, .6, 12), (938, .5, 12), (1248, .6, 18), (1450, .7, 16),
      (1668, 1.0, 18)]),
    ("bio", "Hydroxyapatite_bone",
     [(430, .3, 12), (590, .35, 14), (960, 1.0, 9), (1070, .35, 16)]),
    ("bio", "Hemoglobin_heme",
     [(750, 1.0, 12), (1128, .5, 12), (1370, .8, 12), (1582, .7, 14),
      (1620, .6, 14)]),
]


def gaussian(x, c, fwhm):
    sigma = fwhm / 2.3548
    return np.exp(-((x - c) ** 2) / (2.0 * sigma ** 2))


def synth(bands):
    y = np.zeros_like(X)
    for c, amp, fwhm in bands:
        y += amp * gaussian(X, c, fwhm)
    # gentle sloping baseline + tiny noise → more realistic for matching
    y = y + 0.02 * (1.0 - (X - X.min()) / (X.max() - X.min()))
    y = y + np.random.default_rng(abs(hash(tuple(bands))) % 2**32).normal(
        0, 0.004, size=y.shape)
    y = np.clip(y, 0, None)
    return y


def main():
    os.makedirs(OUT, exist_ok=True)
    n = 0
    for cat, name, bands in MATERIALS:
        y = synth(bands)
        path = os.path.join(OUT, f"{cat}__{name}.csv")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("wavenumber,intensity\n")
            for xi, yi in zip(X, y):
                fh.write(f"{xi:.1f},{yi:.5f}\n")
        n += 1
    # manifest
    with open(os.path.join(OUT, "LIBRARY_INFO.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "BioRaman PCRS starter reference library\n"
            "=======================================\n\n"
            f"{n} synthetic reference spectra generated from published\n"
            "characteristic Raman band positions (peak centre + relative\n"
            "intensity). These are NOT measured spectra and contain no\n"
            "copyrighted data — they are intended as a free, license-clean\n"
            "starter library for IDFinder so it works out of the box.\n\n"
            "Categories: mineral, polymer (microplastics), bio.\n\n"
            "To use real measured libraries (RRUFF, SLoPP/SLoPP-E, Raman Open\n"
            "Database), download them on an internet-connected machine and drop\n"
            "the .txt/.csv files into this folder, or load them via the\n"
            "'Load library folder' button in BioRaman.\n")
    print(f"Wrote {n} reference spectra to {OUT}")


if __name__ == "__main__":
    main()
