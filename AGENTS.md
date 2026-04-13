# Repository Guidelines

## Project Structure & Module Organization

### Desktop Frontends
There are two live desktop frontends:
- `installer/MtgaCoachLauncher/` — WinUI 3 native Windows app
- `src/arenamcp/desktop/` — PySide6 desktop app

Both are active. Do not remove one as cleanup unless the task explicitly requires it.

### Native Desktop App (WinUI 3)
The WinUI app at `installer/MtgaCoachLauncher/` communicates with a headless Python coaching subprocess via JSON lines over stdin/stdout (pipe protocol). Key C# files:
- `App.xaml.cs` — App entry, dark theme, hidden console allocation for PortAudio
- `Views/CoachPage.xaml + .cs` — Main coaching surface: game state, advice log, control buttons, chat
- `Views/RepairPage.xaml + .cs` — Runtime status, MTGA/BepInEx repair, log tails
- `Views/MainPage.xaml + .cs` — NavigationView shell (Coach + Repair tabs), auto-starts coach
- `Services/CoachProcess.cs` — Manages Python subprocess, reads stdout JSON events, sends stdin commands
- `Services/RuntimeDetector.cs` — C# port of windows_integration.py detection logic
- `Services/ProcessLauncher.cs` — Launch/repair subprocess helpers
- `Models/RuntimeState.cs` — Runtime state model (mirrors Python dataclass)

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

The WinUI app launches Python as a headless subprocess with `--pipe`:
```
WinUI App (C#)  ←→  Python (standalone.py --pipe)
                stdout: {"type":"log|advice|status|error|game_state", ...}
                stdin:  {"cmd":"toggle_autopilot|cycle_voice|chat", ...}
```

`PipeAdapter` implements the `UIAdapter` interface, replacing the Textual TUI's `TUIAdapter`. All coach→UI communication (log, advice, status, game_state) flows as JSON lines. GUI→coach commands (button clicks, chat) flow as JSON lines on stdin.

## Build, Test, and Development Commands

- `python -m pip install -e .[dev,full]`: install the Python package with test and full runtime extras.
- `pytest tests -q`: run the Python regression suite.
- `python -m arenamcp.standalone --pipe`: start the coach in headless pipe mode (for native GUI).
- `python -m arenamcp.diagnose`: run local environment diagnostics.
- `cd installer/MtgaCoachLauncher && dotnet build -c Debug -p:Platform=x64`: build the WinUI debug exe.
- `cd installer/MtgaCoachLauncher && dotnet publish -c Release -p:Platform=x64 --self-contained`: publish WinUI launcher binaries.
- `iscc installer/mtgacoach.iss`: build the Windows installer (requires Inno Setup 6).
- `cd bepinex-plugin/MtgaCoachBridge && dotnet build -c Release`: build the BepInEx plugin DLL.
- `p=$(wslpath -w /home/joshu/repos/ArenaMCP/installer) && powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "\$p='${p}'; Set-Location -LiteralPath \$p; .\build-installer.ps1"`: build the Windows installer from WSL.

### Dev Workflow
For iterating on changes without rebuilding the installer:
- **Python changes**: Just restart the app — the dev exe reads from `src/` in the repo.
- **C# changes**: `dotnet build -c Debug -p:Platform=x64` then relaunch the debug exe.
- **Debug exe path**: `installer/MtgaCoachLauncher/bin/x64/Debug/net8.0-windows10.0.19041.0/win-x64/MtgaCoachLauncher.exe`

## Coding Style & Naming Conventions

Use 4-space indentation in Python and keep type hints on public functions where practical. Follow existing Python naming: `snake_case` for functions/modules, `PascalCase` for classes, and concise internal helper names prefixed with `_`. Keep JSON-like state keys stable; downstream code depends on exact names such as `pending_decision`, `decision_context`, and `local_seat_id`. In C#, follow the existing plugin style: `PascalCase` methods, private `_camelCase` fields.

## Testing Guidelines

Use `pytest` for Python changes. Add or update focused regression tests in `tests/test_*.py` whenever you touch bridge serialization, game-state normalization, autopilot planning, or launcher behavior. There is no formal coverage gate, but state-pipeline fixes should include a reproducer-oriented test.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects with prefixes like `fix:`, `feat:`, `debug:`, plus versioned release commits such as `v1.8.0: native WinUI coaching app`. Keep commits scoped and explain the subsystem touched. Do not add Co-Authored-By lines.

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
