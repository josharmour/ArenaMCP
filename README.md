# MTGA Coach - Real-Time AI Game Coaching

An AI-powered coach for Magic: The Gathering Arena that watches your live games and provides strategic advice with voice output.

## What It Does

- **Real-time game tracking** - Reads the MTGA log to track battlefield, hands, life totals, stack, graveyard
- **Proactive coaching** - Advice triggers automatically on new turns, combat, low life, decisions
- **Voice output** - Advice spoken aloud via Kokoro TTS (toggle with F5)
- **Voice input** - Push-to-talk (F4) to ask questions mid-game
- **Win plan detection** - Background analysis finds lethal lines and alerts you with a sound
- **Draft helper** - 17lands stats + composite scoring for draft picks
- **Multiple LLM backends** - Claude Code CLI, Gemini CLI, cli-api-proxy, or local Ollama
- **Card lookup** - Scryfall integration with oracle text and rulings
- **Autopilot mode** - Experimental auto-play using screen coordinates

## Quick Start (Windows)

1. Install [Python 3.10+](https://python.org) (check "Add Python to PATH")
2. Double-click **`install.bat`** and follow the setup wizard
3. Double-click **`coach.bat`** to launch

See [INSTALL.md](INSTALL.md) for detailed instructions, including sending to someone else.

## Hotkeys

| Key | Action |
|-----|--------|
| F2 | Toggle coaching style (concise/detailed) |
| F3 | Analyze screenshot |
| F4 | Push-to-talk voice input |
| F5 | Mute/unmute TTS |
| F6 | Cycle TTS voice |
| F8 | Cycle TTS speed |
| F9 | Toggle AFK mode |
| F11 | Toggle autopilot |
| F12 | Cycle LLM model |
| Numpad 0 | Read win plan aloud |
| Ctrl+Q | Quit |

## Standalone Mode

The primary way to use the coach. No MCP client needed:

```bash
# Activate venv first
venv\Scripts\activate

# With cli-api-proxy (recommended)
python -m arenamcp.standalone --backend proxy

# With local Ollama (free, offline)
python -m arenamcp.standalone --backend ollama --model llama3.2

# With Claude Code CLI (uses subscription, no API key)
python -m arenamcp.standalone --backend claude-code

# With Gemini CLI
python -m arenamcp.standalone --backend gemini-cli

# With a specific language (Dutch STT + English TTS)
python -m arenamcp.standalone --backend ollama --model llama3.2 --language nl

# Draft mode
python -m arenamcp.standalone --draft --set MH3
```

The setup wizard (`install.bat`) auto-detects available backends and saves your choice to `~/.arenamcp/settings.json`, so after initial setup you can just run `coach.bat` without any flags.

## MCP Server Mode

Can also run as an MCP server for Claude Code, OpenCode, or any MCP client:

```json
{
  "mcpServers": {
    "mtga": {
      "command": "python",
      "args": ["-m", "arenamcp.server"],
      "cwd": "/path/to/ArenaMCP"
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| `get_game_state()` | Complete board state, hands, life totals, stack |
| `get_card_info(arena_id)` | Card oracle text via Scryfall |
| `get_draft_rating(card_name, set_code)` | 17lands draft statistics |
| `listen_for_voice(mode)` | Voice input (PTT or VOX) |
| `speak_advice(text)` | Text-to-speech output |
| `start_coaching(backend, model)` | Start proactive coaching |
| `reset_game_state()` | Fix wrong-seat detection |

## Configuration

Run `install.bat` (setup wizard) for guided configuration, or edit `~/.arenamcp/settings.json` directly.

### Settings (`~/.arenamcp/settings.json`)

The primary configuration file. Created by the setup wizard or on first launch.

| Setting | Description | Default |
|---------|-------------|---------|
| `backend` | LLM backend: `proxy`, `ollama`, `claude-code`, `gemini-cli` | `proxy` |
| `model` | Model name override (null = backend default) | `null` |
| `language` | Language for TTS + STT (`en`, `nl`, `es`, `fr`, `de`, `ja`, etc.) | `en` |
| `ollama_url` | Ollama API endpoint | `http://localhost:11434/v1` |
| `proxy_url` | Proxy API endpoint | (uses env var or default) |
| `proxy_api_key` | Proxy API key | (uses env var or default) |
| `voice` | Kokoro voice ID (e.g., `af_sarah`, `bm_george`) | `am_adam` |
| `voice_speed` | TTS speed multiplier | `1.0` |
| `voice_mode` | Voice input: `ptt`, `vox`, `none` | `ptt` |
| `muted` | Mute TTS output | `false` |

### Environment Variables (`.env`)

Legacy configuration, still supported. Settings.json takes priority where both exist.

| Variable | Description | Default |
|----------|-------------|---------|
| `PROXY_BASE_URL` | cli-api-proxy endpoint | `http://127.0.0.1:8080/v1` |
| `PROXY_API_KEY` | Proxy API key | `your-api-key-1` |
| `CLAUDE_CODE_CMD` | Path to Claude CLI | `claude` |
| `GEMINI_CLI_CMD` | Path to Gemini CLI | `gemini` |
| `VOICE_MODE` | Voice input mode (`ptt`/`vox`/`none`) | `ptt` |
| `MTGA_LOG_PATH` | Custom MTGA log path | auto-detected |

### Language Support

Set via `--language` flag or in settings.json. Affects both voice output (TTS) and voice input (STT).

| Code | Language | TTS | STT |
|------|----------|-----|-----|
| `en` | English | Yes | Yes |
| `de` | German | Yes | Yes |
| `es` | Spanish | Yes | Yes |
| `fr` | French | Yes | Yes |
| `it` | Italian | Yes | Yes |
| `ja` | Japanese | Yes | Yes |
| `ko` | Korean | Yes | Yes |
| `pt` | Portuguese | Yes | Yes |
| `zh` | Chinese | Yes | Yes |
| `hi` | Hindi | Yes | Yes |
| `nl` | Dutch | No (falls back to English) | Yes |

Note: TTS uses Kokoro which supports the languages above. STT uses Whisper which supports 99+ languages.

## Architecture

```
coach.bat / run.bat
    |
    v
launcher.py  (auto-restart wrapper)
    |
    v
arenamcp.standalone  (main app)
    |
    +-- MCPClient (in-process)
    |       +-- Log Watcher (watchdog -> Player.log)
    |       +-- Log Parser (JSON event routing)
    |       +-- Game State Manager (board tracking)
    |       +-- Scryfall Cache (card data)
    |       +-- 17lands Stats (draft ratings)
    |
    +-- CoachEngine (LLM advice generation)
    |       +-- Claude Code / Gemini CLI / Proxy / Ollama backends
    |       +-- GameStateTrigger (event -> advice routing)
    |       +-- Win Plan Worker (background lethal detection)
    |
    +-- Voice I/O
    |       +-- VoiceInput (faster-whisper STT)
    |       +-- VoiceOutput (Kokoro TTS)
    |
    +-- TUI (Textual terminal UI)
    |
    +-- Autopilot (experimental)
            +-- ActionPlanner (LLM -> game actions)
            +-- ScreenMapper (coordinate mapping)
            +-- InputController (mouse/keyboard simulation)
```

## Troubleshooting

### "Wrong player" / Advice is backwards
The coach infers which player you are from visible hand cards. Run `reset_game_state()` or restart the coach.

### No voice output
Download Kokoro TTS models (~340MB):
```bash
mkdir -p ~/.cache/kokoro
cd ~/.cache/kokoro
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### Ollama connection refused
Make sure Ollama is running: `ollama serve`

### MCP not connecting
1. Check the server runs: `python -m arenamcp.server`
2. Verify path in MCP config
3. Restart your MCP client after config changes

## Development

```bash
pip install -e ".[full,dev]"
pytest
```

## License

MIT
