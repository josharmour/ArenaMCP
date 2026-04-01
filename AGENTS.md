# Repository Guidelines

## Project Structure & Module Organization

### Native Desktop App (WinUI 3)
The primary user-facing surface is a **WinUI 3 native Windows app** at `installer/MtgaCoachLauncher/`. It communicates with a headless Python coaching subprocess via JSON lines over stdin/stdout (pipe protocol). Key C# files:
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
- `tui.py` — Legacy Textual TUI (still functional but not the primary surface)
- `action_planner.py` — Autopilot action planning via LLM

### BepInEx Plugin
`bepinex-plugin/MtgaCoachBridge/` — C# BepInEx 5 plugin injected into MTGA's Unity runtime for direct GRE state access and action submission.

### Installer
`installer/mtgacoach.iss` — Inno Setup script. Builds `mtgacoach-Setup.exe` with self-contained WinUI launcher + Python source + BepInEx bundle.

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
- `python -m arenamcp.standalone`: start the coach with TUI (legacy).
- `python -m arenamcp.diagnose`: run local environment diagnostics.
- `cd installer/MtgaCoachLauncher && dotnet build -c Debug -p:Platform=x64`: build the WinUI debug exe.
- `cd installer/MtgaCoachLauncher && dotnet publish -c Release -p:Platform=x64 --self-contained`: publish self-contained release.
- `iscc installer/mtgacoach.iss`: build the Windows installer (requires Inno Setup 6).
- `cd bepinex-plugin/MtgaCoachBridge && dotnet build -c Release`: build the BepInEx plugin DLL.

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

## Security & Configuration Tips

Do not commit API keys, `%LOCALAPPDATA%\mtgacoach` runtime data, `.arenamcp` user settings, MTGA logs, or copied game binaries. Treat `bin/`, `obj/`, `dist/`, and `__pycache__/` as generated artifacts unless a release task explicitly requires them.
