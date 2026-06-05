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
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []
for _pkg in ("sklearn", "scipy", "pybaselines", "renishawWiRE",
             "matplotlib", "seaborn", "pandas", "openpyxl", "PIL"):
    try:
        d, b, h = collect_all(_pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        # Optional dependency not installed in the build env — skip it.
        pass

# Make sure the native Renishaw .wdf reader and its submodules are bundled so
# the standalone app can open map files without any runtime install.
try:
    hiddenimports += collect_submodules("renishawWiRE")
except Exception:
    pass
hiddenimports += ["renishawWiRE", "multiprocessing", "multiprocessing.pool"]

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
        "CFBundleShortVersionString": "1.0.2",
        "CFBundleVersion": "1.0.2",
    },
)
