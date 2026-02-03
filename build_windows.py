#!/usr/bin/env python3
"""Build script for ArenaMCP Windows distribution.

Creates a standalone Windows executable using PyInstaller.

Usage:
    python build_windows.py          # Build executable
    python build_windows.py --clean  # Clean and rebuild
    python build_windows.py --zip    # Build and create zip archive
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Build configuration
PROJECT_NAME = "ArenaMCP"
VERSION = "0.1.0"
SPEC_FILE = "arenamcp.spec"
DIST_DIR = Path("dist")
BUILD_DIR = Path("build")


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and print output."""
    print(f">> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def clean():
    """Remove build artifacts."""
    print("Cleaning build directories...")
    for dir_path in [DIST_DIR, BUILD_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  Removed {dir_path}")


def ensure_pyinstaller():
    """Ensure PyInstaller is installed."""
    try:
        import PyInstaller
        print(f"PyInstaller {PyInstaller.__version__} found")
    except ImportError:
        print("Installing PyInstaller...")
        run([sys.executable, "-m", "pip", "install", "pyinstaller"])


def build_exe():
    """Build the executable using PyInstaller."""
    print(f"\nBuilding {PROJECT_NAME}...")

    # Run PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        SPEC_FILE,
    ]
    run(cmd)

    exe_path = DIST_DIR / f"{PROJECT_NAME}.exe"
    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"\nBuild complete: {exe_path}")
        print(f"Size: {size_mb:.1f} MB")
        return exe_path
    else:
        print("ERROR: Build failed - executable not found")
        sys.exit(1)


def create_zip(exe_path: Path):
    """Create a zip archive of the distribution."""
    zip_name = f"{PROJECT_NAME}-{VERSION}-windows"
    zip_path = DIST_DIR / zip_name

    print(f"\nCreating archive: {zip_path}.zip")

    # Create a directory with all files
    staging_dir = DIST_DIR / zip_name
    staging_dir.mkdir(exist_ok=True)

    # Copy executable
    shutil.copy(exe_path, staging_dir / exe_path.name)

    # Copy README and other docs
    for doc in ["README.md", "LICENSE"]:
        if Path(doc).exists():
            shutil.copy(doc, staging_dir / doc)

    # Create example .env file
    env_example = staging_dir / ".env.example"
    env_example.write_text("""# ArenaMCP Configuration
# Copy this to .env and fill in your API keys

# For Gemini backend (recommended - fast and free tier available)
GOOGLE_API_KEY=your_google_api_key_here

# For Claude backend
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# For Ollama backend (no key needed - runs locally)
# Just install Ollama and run: ollama pull llama3.2
""")

    # Create quick start guide
    quickstart = staging_dir / "QUICKSTART.txt"
    quickstart.write_text(f"""ArenaMCP {VERSION} - Quick Start Guide
=====================================

1. SETUP
   - Copy .env.example to .env
   - Add your API key (GOOGLE_API_KEY for Gemini is recommended)

2. RUN COACHING MODE
   Double-click ArenaMCP.exe or run from command line:

   ArenaMCP.exe --backend gemini
   ArenaMCP.exe --backend ollama --model llama3.2

3. RUN DRAFT HELPER MODE (no API key needed)
   ArenaMCP.exe --draft --set MH3

4. HOTKEYS
   F4 = Push-to-talk (hold to speak, tap for quick advice)
   F5 = Mute/unmute TTS
   F6 = Change TTS voice
   F7 = Save bug report

5. TROUBLESHOOTING
   - Logs are saved to: %USERPROFILE%\\.arenamcp\\standalone.log
   - Run with --show-log to see recent log entries

For full documentation, visit:
https://github.com/yourusername/ArenaMCP
""")

    # Create the zip
    shutil.make_archive(str(zip_path), 'zip', DIST_DIR, zip_name)

    # Clean up staging
    shutil.rmtree(staging_dir)

    final_zip = Path(f"{zip_path}.zip")
    size_mb = final_zip.stat().st_size / (1024 * 1024)
    print(f"Archive created: {final_zip}")
    print(f"Size: {size_mb:.1f} MB")

    return final_zip


def main():
    parser = argparse.ArgumentParser(description="Build ArenaMCP for Windows")
    parser.add_argument("--clean", action="store_true",
                        help="Clean build directories before building")
    parser.add_argument("--zip", action="store_true",
                        help="Create zip archive after building")
    args = parser.parse_args()

    print(f"Building {PROJECT_NAME} v{VERSION}")
    print("="*50)

    if args.clean:
        clean()

    ensure_pyinstaller()
    exe_path = build_exe()

    if args.zip:
        create_zip(exe_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
