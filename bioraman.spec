# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for BioRaman.

Builds a ONE-FOLDER app (Windows: dist/BioRaman/BioRaman.exe; macOS:
dist/BioRaman.app). One-folder is used deliberately: a one-file build would
re-extract the entire ~100 MB scientific stack to a temp directory on every
launch, which takes minutes. One-folder starts in seconds.

Heavy scientific dependencies are pulled in wholesale with collect_all so their
data files, compiled extensions and dynamically imported submodules are not
missed.

    pyinstaller --noconfirm bioraman.spec
"""
import os, sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Packages that MUST be present for a functional build. A missing entry here
# means the standalone app would ship broken (e.g. without renishawWiRE the
# app cannot decode .wdf map files), so we fail the build loudly instead of
# silently skipping. Override by setting BIORAMAN_ALLOW_MISSING=1.
CRITICAL_PKGS = {"renishawWiRE"}

datas, binaries, hiddenimports = [], [], []
_missing = []
for _pkg in ("sklearn", "scipy", "pybaselines", "renishawWiRE",
             "matplotlib", "seaborn", "pandas", "openpyxl", "PIL"):
    try:
        d, b, h = collect_all(_pkg)
        datas += d
        binaries += b
        hiddenimports += h
        print(f"[bioraman.spec] collected '{_pkg}' "
              f"({len(d)} data, {len(b)} binaries, {len(h)} hidden imports)")
    except Exception as exc:
        _missing.append(_pkg)
        print(f"[bioraman.spec] WARNING: could not collect '{_pkg}': {exc}")

# Make sure the native Renishaw .wdf reader and its submodules are bundled so
# the standalone app can open map files without any runtime install.
try:
    _wdf_subs = collect_submodules("renishawWiRE")
    hiddenimports += _wdf_subs
    print(f"[bioraman.spec] collected {len(_wdf_subs)} renishawWiRE submodules")
except Exception as exc:
    if "renishawWiRE" not in _missing:
        _missing.append("renishawWiRE")
    print(f"[bioraman.spec] WARNING: could not collect renishawWiRE "
          f"submodules: {exc}")
hiddenimports += ["renishawWiRE", "multiprocessing", "multiprocessing.pool"]

# Fail the build if anything critical is missing, unless explicitly overridden.
_critical_missing = sorted(CRITICAL_PKGS.intersection(_missing))
if _critical_missing and not os.environ.get("BIORAMAN_ALLOW_MISSING"):
    raise SystemExit(
        "\n[bioraman.spec] BUILD ABORTED — critical package(s) not found in "
        "the build environment:\n    " + ", ".join(_critical_missing) + "\n"
        "These are required for the app to function (renishawWiRE decodes "
        ".wdf map files).\nInstall them into the SAME Python you build with:\n"
        f"    {sys.executable} -m pip install " + " ".join(_critical_missing) +
        "\nThen rebuild. To build anyway (NOT recommended) set "
        "BIORAMAN_ALLOW_MISSING=1.\n")

block_cipher = None

a = Analysis(
    ["bioraman.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["PyQt5", "PySide2", "PyQt6", "PySide6", "tkinter.test"],
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# One-folder: keep binaries OUT of the EXE and COLLECT them alongside it.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BioRaman",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # GUI app — no terminal window
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="BioRaman",
)

app = BUNDLE(
    coll,
    name="BioRaman.app",
    icon=None,
    bundle_identifier="com.gibsongroupresearch.bioraman",
    info_plist={
        "NSHighResolutionCapable": True,
        "CFBundleShortVersionString": "1.0.3",
        "CFBundleVersion": "1.0.3",
    },
)
