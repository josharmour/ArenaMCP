# MTGA MCP Server - Real-Time Game Coaching

A Model Context Protocol (MCP) server that connects to live Magic: The Gathering Arena games, providing real-time game state and AI coaching through any MCP-compatible client.

## Features

- **Real-time game state** - Tracks battlefield, hands, life totals, stack, graveyard
- **Card lookup** - Scryfall integration with oracle text
- **Draft ratings** - 17lands statistics for limited play
- **Voice coaching** - Push-to-talk (F4) and voice-activated input
- **Text-to-speech** - Spoken advice via Kokoro TTS
- **Background coaching** - Proactive advice on game events (new turn, combat, low life)
- **Multiple LLM backends** - Claude Code CLI, Gemini CLI, or local Ollama

## Prerequisites

1. **MTGA** with "Detailed Logs (Plugin Support)" enabled in Settings
2. **Python 3.11+**
3. **For voice features**: Audio input/output devices
4. **For local LLM**: [Ollama](https://ollama.com) installed

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ArenaMCP.git
cd ArenaMCP

# Install dependencies
pip install -e .

# Or install manually
pip install fastmcp scrython watchdog requests sounddevice numpy
pip install faster-whisper  # For voice input
pip install kokoro-onnx     # For TTS output
```

### TTS Model Setup (Optional - for voice output)

```bash
# Create model directory
mkdir -p ~/.cache/kokoro

# Download Kokoro TTS models (~340MB total)
cd ~/.cache/kokoro
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### Ollama Setup (Optional - for local LLM)

```bash
# Install Ollama from https://ollama.com
# Then pull a model
ollama pull llama3.2      # Fast, 2GB
ollama pull gemma3:12b    # Better quality, 8GB
```

## Configuration

### Claude Code

Add to your Claude Code MCP settings (`~/.claude.json` or via `/mcp add`):

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

Or use the CLI:
```bash
claude mcp add mtga "python -m arenamcp.server" --cwd /path/to/ArenaMCP
```

### Other MCP Clients (OpenCode, etc.)

The server uses **STDIO transport**. Configure your client to run:

```bash
python -m arenamcp.server
```

Example for OpenCode (`~/.opencode/config.json`):
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

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `CLAUDE_CODE_CMD` | Path to Claude Code CLI (e.g., `claude`) | No |
| `GEMINI_CLI_CMD` | Path to Gemini CLI (e.g., `gemini`) | No |
| `MTGA_LOG_PATH` | Custom log path | No (auto-detected) |

## Standalone Mode (No MCP Client Required)

Run the coach directly without Claude Code or any MCP client:

```bash
# With Claude Code CLI (uses your subscription session, no API key)
python -m arenamcp.standalone --backend claude-code

# With Gemini CLI (uses your subscription session, no API key)
python -m arenamcp.standalone --backend gemini-cli

# With local Ollama (free, runs offline)
python -m arenamcp.standalone --backend ollama --model gemma3:12b

# Disable auto-speak (text only)
python -m arenamcp.standalone --backend claude-code --no-auto-speak
```

### Standalone Features

- **Proactive coaching** - Advice triggers automatically on game events
- **Voice input** - Press F4 to ask questions (or use `--voice vox` for voice activation)
- **TTS output** - Advice is spoken aloud (disable with `--no-auto-speak`)
- **Debug logging** - Full logs saved to `~/.arenamcp/debug.log` for bug reports

### Filing Bug Reports

If something goes wrong, the debug log contains everything needed to diagnose issues:

```bash
# Show log location and recent entries
python -m arenamcp.standalone --show-log

# Copy log for bug report
cat ~/.arenamcp/debug.log
```

The log includes timestamps, game state snapshots, triggers, LLM responses, and error traces.

## Usage (MCP Mode)

### 1. Start MTGA and Enter a Game

Make sure "Detailed Logs (Plugin Support)" is enabled in MTGA settings.

### 2. Connect Your MCP Client

**Claude Code:**
```
/mcp
```
Should show "Connected to mtga"

### 3. Available MCP Tools

#### Game State
```
get_game_state()
```
Returns complete board state: battlefield, hand, graveyard, stack, exile, life totals, turn info.

#### Card Lookup
```
get_card_info(arena_id=12345)
```
Get oracle text and details for any card by its MTGA ID.

#### Draft Ratings
```
get_draft_rating(card_name="Lightning Bolt", set_code="FDN")
```
Get 17lands win rates and pick order for limited.

#### Voice Input
```
listen_for_voice(mode="ptt", timeout=30)
```
- `mode="ptt"`: Hold **F4** to speak
- `mode="vox"`: Voice-activated (starts on sound)

#### Text-to-Speech
```
speak_advice(text="Attack with everything!")
```
Speaks text through your speakers.

#### Background Coaching
```
start_coaching(backend="ollama", model="gemma3:12b", auto_speak=true)
```
Starts proactive coaching that triggers on:
- New turn
- Combat phase (attackers/blockers)
- Low life (<5)
- Opponent low life
- Spells on stack

```
stop_coaching()
get_coaching_status()
get_pending_advice()  # Get queued advice without auto-speak
```

#### Reset Game State
```
reset_game_state()
```
Use if player detection is wrong (fixes "wrong seat" issues).

### 4. Example Session

```
You: /mcp
> Connected to mtga

You: What's the game state?
Claude: [Calls get_game_state()]
> You're at 15 life, opponent at 12. Turn 6, Main Phase 1.
> Your board: Tireless Tracker (3/2), 2 Clue tokens
> Opponent: Monastery Swiftspear (1/2), tapped

You: Start coaching with Ollama
Claude: [Calls start_coaching(backend="ollama", model="gemma3:12b", auto_speak=true)]
> Coaching started! You'll hear advice on game events.

[Turn changes]
Speaker: "New turn. You have priority with 4 mana available..."

You: Stop coaching
Claude: [Calls stop_coaching()]
> Coaching stopped.
```

## Troubleshooting

### "Wrong player" / Advice is backwards

The server infers which player you are from visible hand cards. If your hand is empty, it may guess wrong.

**Fix:**
```
reset_game_state()
```
Then make sure you have cards in hand when game state updates.

### "Ollama connection refused"

Make sure Ollama is running:
```bash
ollama serve
```

### "TTS model not found"

Download the Kokoro models (see Installation above).

### MCP not connecting

1. Check the server runs standalone: `python -m arenamcp.server`
2. Verify path in config is correct
3. Restart Claude Code after config changes

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client                               │
│            (Claude Code, OpenCode, etc.)                     │
└─────────────────────────┬───────────────────────────────────┘
                          │ STDIO
┌─────────────────────────▼───────────────────────────────────┐
│                    MCP Server                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Game Tools  │  │ Voice Tools │  │ Coaching Tools      │  │
│  │             │  │             │  │                     │  │
│  │ get_state   │  │ listen_ptt  │  │ start_coaching      │  │
│  │ get_card    │  │ speak       │  │ stop_coaching       │  │
│  │ get_draft   │  │             │  │ get_status          │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────────▼──────────┐  │
│  │ GameState   │  │ VoiceInput  │  │ CoachEngine         │  │
│  │ Manager     │  │ VoiceOutput │  │ (Claude Code/       │  │
│  │             │  │ (Kokoro)    │  │  Gemini CLI/        │  │
│  │             │  │            │  │  Ollama backends)   │  │
│  └──────┬──────┘  └─────────────┘  └─────────────────────┘  │
│         │                                                    │
│  ┌──────▼──────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Log Parser  │  │ Scryfall    │  │ 17lands             │  │
│  │ (watchdog)  │  │ Cache       │  │ Draft Stats         │  │
│  └──────┬──────┘  └─────────────┘  └─────────────────────┘  │
└─────────┼───────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│  MTGA Player.log                                             │
│  %AppData%\LocalLow\Wizards Of The Coast\MTGA\Player.log    │
└─────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Run tests
pytest

# Run server directly
python -m arenamcp.server

# Check syntax
python -m py_compile src/arenamcp/server.py
```

## License

MIT
