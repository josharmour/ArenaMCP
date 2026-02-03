# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for ArenaMCP standalone app.

This creates a lightweight build without bundled TTS/voice ML models.
TTS will need to download models on first run.
"""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['src/arenamcp/standalone.py'],
    pathex=['src'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # MCP and FastMCP
        'mcp',
        'mcp.server',
        'mcp.server.fastmcp',
        # Core arenamcp modules
        'arenamcp',
        'arenamcp.server',
        'arenamcp.gamestate',
        'arenamcp.parser',
        'arenamcp.watcher',
        'arenamcp.scryfall',
        'arenamcp.draftstats',
        'arenamcp.draftstate',
        'arenamcp.draft_eval',
        'arenamcp.mtgadb',
        'arenamcp.coach',
        'arenamcp.voice',
        'arenamcp.tts',
        # Dependencies
        'watchdog',
        'watchdog.observers',
        'watchdog.events',
        'requests',
        'keyboard',
        'sounddevice',
        'numpy',
        'sqlite3',
        # LLM backends
        'google.generativeai',
        'anthropic',
        'ollama',
        # SSL for HTTPS requests
        'ssl',
        'certifi',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude heavy ML frameworks not needed for core functionality
        'torch',
        'torchvision',
        'torchaudio',
        'tensorflow',
        'keras',
        'sklearn',
        'scikit-learn',
        'scipy',  # Large, only used for some audio processing
        'pandas',  # Not needed for core functionality
        'matplotlib',
        'PIL',
        'cv2',
        'tkinter',
        # Exclude test frameworks
        'pytest',
        'py',
        # Exclude unnecessary ML stuff
        'transformers',
        'huggingface_hub',
        'onnx',
        'onnxruntime',
        'h5py',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
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
    name='ArenaMCP',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico' if Path('assets/icon.ico').exists() else None,
)
