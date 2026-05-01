# Repository Guidelines

## Project Structure & Module Organization

### Desktop Frontend
The desktop frontend is the PySide6 app at `src/arenamcp/desktop/`. It spawns the headless coach as a subprocess and communicates via JSON lines over stdin/stdout (pipe protocol).

### Python Coaching Engine
Core Python code lives in `src/arenamcp/`. Key runtime modules:
- `standalone.py` — Main entry point, coaching loop, `--pipe` mode for native GUI
- `pipe_adapter.py` — `PipeAdapter(UIAdapter)` writes JSON lines to stdout, reads commands from stdin
- `server.py` — MCP server exposing game state tools, bridge overlay
- `coach.py` — LLM prompt building, advice post-processing, fallback logic
- `gamestate.py` — Real-time game state tracking from MTGA log + GRE bridge
- `gre_bridge.py` — Named pipe server for direct GRE bridge communication
- `autopilot.py` — Autonomous play via GRE bridge + screen interaction
- `tts.py` — Kokoro ONNX TTS with lazy numpy import, winsound fallback on Windows
- `action_planner.py` — Autopilot action planning via LLM

### Installer / Runtime Model
Prefer a **small installer**.

That means:
- the installer ships launcher/app files and setup assets
- the runtime venv is created after install under `%LOCALAPPDATA%\mtgacoach`
- dependency installation happens in setup/repair, not by bundling a giant venv into the installer payload

Repair/setup surfaces should expose explicit actions for:
- `Create venv`
- `Setup environment`
- bridge/BepInEx/plugin repair actions

### BepInEx Plugin
`bepinex-plugin/MtgaCoachBridge/` — C# BepInEx 5 plugin injected into MTGA's Unity runtime for direct GRE state access and action submission.

### Installer
`installer/mtgacoach.iss` — Inno Setup script. Builds the small `mtgacoach-Setup.exe` installer.

### Tests
`tests/` — pytest regression tests for bridge serialization, game state normalization, and server overlay.

## Architecture: Pipe Protocol

The PySide6 app launches Python as a headless coach subprocess with `--pipe`:
```
Desktop App (PySide6)  ←→  Coach (standalone.py --pipe)
                       stdout: {"type":"log|advice|status|error|game_state", ...}
                       stdin:  {"cmd":"toggle_autopilot|cycle_voice|chat", ...}
```

`PipeAdapter` implements the `UIAdapter` interface. All coach→UI communication (log, advice, status, game_state) flows as JSON lines. GUI→coach commands (button clicks, chat) flow as JSON lines on stdin.

## Build, Test, and Development Commands

- `python -m pip install -e .[dev,full]`: install the Python package with test and full runtime extras.
- `pytest tests -q`: run the Python regression suite.
- `python -m arenamcp.standalone --pipe`: start the coach in headless pipe mode (for native GUI).
- `python -m arenamcp.diagnose`: run local environment diagnostics.
- `iscc installer/mtgacoach.iss`: build the Windows installer (requires Inno Setup 6).
- `cd bepinex-plugin/MtgaCoachBridge && dotnet build -c Release`: build the BepInEx plugin DLL.
- `p=$(wslpath -w /home/joshu/repos/ArenaMCP/installer) && powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "\$p='${p}'; Set-Location -LiteralPath \$p; .\build-installer.ps1"`: build the Windows installer from WSL.

### Dev Workflow
For iterating on changes without rebuilding the installer:
- **Python changes**: Just restart the app — the launch script imports from `src/` in the repo.

## Coding Style & Naming Conventions

Use 4-space indentation in Python and keep type hints on public functions where practical. Follow existing Python naming: `snake_case` for functions/modules, `PascalCase` for classes, and concise internal helper names prefixed with `_`. Keep JSON-like state keys stable; downstream code depends on exact names such as `pending_decision`, `decision_context`, and `local_seat_id`. In C# (BepInEx plugin), follow the existing plugin style: `PascalCase` methods, private `_camelCase` fields.

## Testing Guidelines

Use `pytest` for Python changes. Add or update focused regression tests in `tests/test_*.py` whenever you touch bridge serialization, game-state normalization, autopilot planning, or launcher behavior. There is no formal coverage gate, but state-pipeline fixes should include a reproducer-oriented test.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects with prefixes like `fix:`, `feat:`, `debug:`, plus versioned release commits such as `v2.2.3: ...`. Keep commits scoped and explain the subsystem touched. Do not add Co-Authored-By lines.

When revving a release, bump:
- `pyproject.toml`
- `src/arenamcp/__init__.py`
- `installer/mtgacoach.iss`

Do not publish a release unless the GitHub `mtgacoach-Setup.exe` asset matches the tag/version being published.

## Security & Configuration Tips

Do not commit API keys, `%LOCALAPPDATA%\mtgacoach` runtime data, `.arenamcp` user settings, MTGA logs, or copied game binaries. Treat `bin/`, `obj/`, `dist/`, and `__pycache__/` as generated artifacts unless a release task explicitly requires them.

Also avoid committing:
- `.venv/`
- `.tools/`
- scratch files like `test_write.txt`

Note: `tests/` may be gitignored in this repo, so use `git add -f tests/...` when a test file should ship.
