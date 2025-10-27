# -*- mode: python ; coding: utf-8 -*-
import sys

# '.' ensures it finds both wavbeat_gui.py and wavbeat.py
sys.path.append('.')

a = Analysis(
    ['wavbeat_gui.py'],
    pathex=['.'], # Look for modules (like wavbeat.py) in the current directory
    binaries=[],
    datas=[],
    # Hidden imports for libraries that PyInstaller often misses parts of
    # tkinterdnd2 is custom and soundfile/scipy sometimes need explicit help
    hiddenimports=['tkinterdnd2', 'soundfile', 'scipy.signal'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    name='wavbeat_gui' # Internal name for the analysis step
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    a.zipfiles,
    name='wavbeat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    # Set to False because this is a Tkinter/GUI application
    console=False,
    disable_window_close=True # Recommended for Windows GUIs
)
