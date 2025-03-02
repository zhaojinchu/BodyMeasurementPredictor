# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['body_measurement_app2.py'],
    pathex=[],
    binaries=[],
    datas=[('rf_models.pkl', '.'), ('ridge_models.pkl', '.'), ('features.pkl', '.'), ('target.pkl', '.')],
    hiddenimports=['sklearn.ensemble', 'sklearn.linear_model', 'sklearn.model_selection'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BodyMeasurementApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
