# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller build spec for BioRaman.

Builds a single self-contained executable (Windows .exe / Linux binary) and a
macOS .app bundle. Heavy scientific dependencies are pulled in wholesale with
collect_all so their data files, compiled extensions and dynamically imported
submodules are not missed.

    pyinstaller --noconfirm bioraman.spec
"""
from PyInstaller.utils.hooks import collect_all

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

hiddenimports += ["multiprocessing", "multiprocessing.pool"]

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

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
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

app = BUNDLE(
    exe,
    name="BioRaman.app",
    icon=None,
    bundle_identifier="com.gibsongroupresearch.bioraman",
    info_plist={
        "NSHighResolutionCapable": True,
        "CFBundleShortVersionString": "1.0.1",
        "CFBundleVersion": "1.0.1",
    },
)
