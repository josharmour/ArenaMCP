# Repository Guidelines

## Project Structure & Module Organization

- `src/arenamcp/`: main Python package. Key modules: `server.py` (FastMCP STDIO server), `standalone.py` (local coach CLI), `watcher.py` (tails MTGA `Player.log`), `parser.py` (event parsing), `gamestate.py` (state model), `coach.py` (LLM backends + triggers).
- `tests/`: pytest suite (`tests/test_*.py`), focused on watcher/parser and state handling.
- `assets/`: optional runtime assets (may be empty in dev checkouts).
- `build/`, `dist/`: generated artifacts (PyInstaller / packaging).

## Build, Test, and Development Commands

- `.\install.bat`: create `venv\` and install an editable dev setup (best on Windows).
- `pip install -e ".[full,dev]"`: editable install with voice/LLM extras + pytest.
- `pip install -r requirements.txt`: non-editable dependency install (CI / quick checks).
- `python -m arenamcp.server`: run the MCP server (STDIO transport; configure via your MCP client).
- `.\run.bat --backend gemini|claude|ollama`: run standalone coaching mode.
- `pytest`: run tests.
- `python build_windows.py --clean --zip`: build Windows executable + zip (PyInstaller).

## Coding Style & Naming Conventions

- Python 3.10+; 4-space indentation; keep type hints on public APIs.
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Keep modules small and testable under `src/arenamcp/`; avoid side effects at import time (prefer explicit entrypoints/CLI).

## Testing Guidelines

- Use pytest; name tests `tests/test_*.py`.
- Prefer deterministic tests: avoid real MTGA logs, network calls, and audio devices; use sample log chunks and temp files.
- When changing parsing/state logic, add a test for the new log shape (even if it is a minimal snippet).

## Commit & Pull Request Guidelines

- Follow Conventional Commits used in history: `feat:`, `fix:`, `docs:` (optional scope/date like `docs(09-01): ...`).
- PRs should include: a short behavior description, repro steps (or a log snippet), and any new/changed env vars. Add screenshots for TUI changes.

## Security & Configuration Tips

- Never commit secrets; keep keys in `.env` (gitignored). If you add configuration, update `.env.example` (e.g., `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `MTGA_LOG_PATH`).
- Agent note: `CLAUDE.md` documents the intended layering (watcher -> parser -> state -> MCP tools).
