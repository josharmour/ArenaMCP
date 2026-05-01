# GEMINI.md

## Project Summary

`mtgacoach` (`arenamcp`) is an MTGA coaching project with:
- Python core in `src/arenamcp/`
- PySide6 desktop frontend in `src/arenamcp/desktop/`
- BepInEx plugin in `bepinex-plugin/MtgaCoachBridge/`

## Core Direction

- GRE bridge is the primary in-match source of truth.
- `Player.log` is fallback and diagnostics.
- App install is read-only under `Program Files`.
- Mutable runtime state belongs under `%LOCALAPPDATA%\mtgacoach`.

## Installer Direction

Prefer a **small installer**.

That means:
- ship launcher/app files and setup assets
- do not bundle a heavyweight full Python venv into the installer by default
- create the runtime venv after install under `%LOCALAPPDATA%\mtgacoach`
- install dependencies during setup/repair

If a release build starts packaging a huge runtime payload, treat that as a bug unless the user explicitly asked for an offline/fat installer.

## Repair / Setup UX

Repair/setup surfaces should expose explicit actions for:
- `Create venv`
- `Setup environment`
- `Install BepInEx`
- `Install Plugin`
- bridge repair / refresh actions

`Provision Runtime` alone is too vague; prefer the explicit split.

## Release Rules

When revving a release:
- bump `pyproject.toml`
- bump `src/arenamcp/__init__.py`
- bump `installer/mtgacoach.iss`
- commit
- tag `vX.Y.Z`
- push commit and tag
- create/update the GitHub release
- upload `mtgacoach-Setup.exe`

Do not publish a release unless the GitHub installer asset matches the tag.

## Common Commands

```bash
python -m pip install -e .[dev,full]
pytest tests -q
python -m arenamcp.desktop
python -m arenamcp.standalone --backend online
cd bepinex-plugin/MtgaCoachBridge && dotnet build -c Release
p=$(wslpath -w /home/joshu/repos/ArenaMCP/installer) && powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "\$p='${p}'; Set-Location -LiteralPath \$p; .\build-installer.ps1"
```

## Hygiene

Do not commit:
- `.venv/`
- `.tools/`
- `bin/`
- `obj/`
- scratch files

`tests/` may be ignored in git here; use `git add -f tests/...` when needed.
