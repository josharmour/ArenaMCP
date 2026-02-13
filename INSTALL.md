# Installation Guide

## Sending to Someone Else

### Create a zip

From the project root:

```bash
git archive --format=zip -o ArenaMCP.zip HEAD
```

Or just zip the folder manually. Delete `venv/`, `__pycache__/`, and `.env` before zipping (they're machine-specific).

### What's in the zip

| File | Purpose |
|------|---------|
| `install.bat` | Double-click to install (runs setup wizard) |
| `coach.bat` | Double-click to launch the coach |
| `run.bat` | Alternative launcher (simpler, no auto-restart) |
| `setup_wizard.py` | Interactive Python installer (called by install.bat) |
| `.env.example` | Template for configuration |

---

## Prerequisites

### Required

- **Windows 10 or 11**
- **Python 3.10+** from [python.org](https://python.org)
  - During installation, **check "Add Python to PATH"**
- **MTGA** installed and launched at least once
  - In MTGA Settings, enable **"Detailed Logs (Plugin Support)"**

### LLM Backend (pick one)

The coach needs an LLM to generate advice. Choose one:

| Backend | Setup | Cost | Quality |
|---------|-------|------|---------|
| **cli-api-proxy** (recommended) | Install [cli-api-proxy](https://github.com/anthropics/cli-proxy-api), run on port 8080 | Uses your existing API keys | Best (routes to Claude, GPT, Gemini) |
| **Claude Code CLI** | Install [Claude Code](https://claude.ai/code), have active subscription | Subscription | Great |
| **Gemini CLI** | Install Gemini CLI, have active subscription | Subscription | Good |
| **Ollama** (local, free) | Install [Ollama](https://ollama.com), pull a model | Free | Depends on model/GPU |

---

## Step-by-Step Install

### 1. Install Python

Download from [python.org](https://python.org/downloads/).

When the installer opens:
- Check **"Add Python to PATH"** (bottom of first screen)
- Click "Install Now"

Verify it works by opening Command Prompt and typing:
```
python --version
```
You should see `Python 3.10.x` or higher.

### 2. Extract the zip

Unzip `ArenaMCP.zip` to any folder, e.g. `C:\ArenaMCP`.

### 3. Run the installer

Double-click **`install.bat`**.

The setup wizard will:
1. Check your Python version
2. Create a virtual environment (`venv/`)
3. Install all Python packages
4. **Auto-detect available backends** (Ollama, Proxy, Claude CLI, Gemini CLI) and show which are found
5. Let you **pick a model** from the detected list
6. **Language selection** for voice input/output (English, Dutch, Spanish, French, etc.)
7. Voice input mode (push-to-talk, voice activation, or disabled)
8. Save everything to `~/.arenamcp/settings.json`

This takes 2-5 minutes depending on your internet speed.

### 4. Set up your LLM backend

#### Option A: cli-api-proxy

```bash
npm install -g cli-api-proxy
cli-api-proxy --port 8080
```

Leave this running in a separate terminal. The setup wizard will auto-detect it.

#### Option B: Ollama (free, local)

1. Install from [ollama.com](https://ollama.com) (runs as a system service automatically)
2. Pull a model:
```bash
ollama pull llama3.2      # Fast, 2GB VRAM
ollama pull gemma3:12b    # Better quality, 8GB VRAM
```
3. The setup wizard will detect Ollama automatically and list your available models

Ollama serves an OpenAI-compatible API at `http://localhost:11434`. The coach connects to it natively - no proxy needed.

### 5. Launch the coach

Double-click **`coach.bat`**.

The TUI (terminal UI) will open showing:
- Game state panel (left)
- Advice panel (right)
- Status bar (bottom)

Start an MTGA game and the coach will begin tracking automatically.

---

## Folder Structure After Install

```
ArenaMCP/
  install.bat          # Installer (run once)
  coach.bat            # Main launcher (run this)
  run.bat              # Alternative launcher
  venv/                # Python virtual environment (created by installer)
  .env                 # Your configuration (created by installer)
  .env.example         # Configuration template
  src/arenamcp/        # Source code
  requirements.txt     # Python dependencies
  pyproject.toml       # Package metadata
```

---

## Manual Install (without install.bat)

If you prefer doing it yourself:

```bash
cd ArenaMCP

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install everything
pip install -e ".[full]"

# Install additional packages
pip install textual openai websocket-client scipy Pillow
pip install networkx beautifulsoup4 pyedhrec lxml
pip install pyautogui pydirectinput-rgx

# Run with Ollama (if installed)
python -m arenamcp.standalone --backend ollama --model llama3.2

# Run with proxy
python -m arenamcp.standalone --backend proxy

# Run with specific language
python -m arenamcp.standalone --backend ollama --model llama3.2 --language nl
```

---

## Language Configuration

The coach supports multiple languages for voice input (STT) and output (TTS).

Set during initial setup, or change anytime:

```bash
# Via command line flag (saved to settings)
python -m arenamcp.standalone --language nl    # Dutch
python -m arenamcp.standalone --language es    # Spanish
python -m arenamcp.standalone --language de    # German

# Or edit ~/.arenamcp/settings.json directly
{
  "language": "nl"
}

# Or re-run the setup wizard
python setup_wizard.py
```

Supported: English, German, Spanish, French, Italian, Japanese, Korean, Portuguese, Chinese, Hindi, Dutch.

Note: Dutch (`nl`) is supported for voice input (Whisper STT) but TTS output will fall back to English since Kokoro doesn't have Dutch voices yet.

---

## Voice Setup (Optional)

### Text-to-Speech (Kokoro)

TTS models download automatically on first use. If that fails, download manually:

**Windows (PowerShell):**
```powershell
mkdir "$env:USERPROFILE\.cache\kokoro" -Force
cd "$env:USERPROFILE\.cache\kokoro"
Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" -OutFile "kokoro-v1.0.onnx"
Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" -OutFile "voices-v1.0.bin"
```

### Voice Input (Whisper)

Requires a microphone. The `faster-whisper` model downloads automatically on first use (~150MB).

Set voice mode in `.env` or `settings.json`:
```
VOICE_MODE=ptt    # Push-to-talk (hold F4)
VOICE_MODE=vox    # Voice-activated
VOICE_MODE=none   # Disabled
```

---

## Configuration Files

The coach uses two config files:

### `~/.arenamcp/settings.json` (primary)

Created by the setup wizard. This is the main config file:
```json
{
  "backend": "ollama",
  "model": "llama3.2",
  "language": "en",
  "ollama_url": "http://localhost:11434/v1",
  "voice": "af_sarah",
  "voice_speed": 1.2,
  "voice_mode": "ptt",
  "muted": false
}
```

To reconfigure, either:
- Edit this file directly
- Re-run `python setup_wizard.py` (skips venv/deps if already installed)
- Use CLI flags like `--backend ollama --model llama3.2 --language nl`

### `.env` (legacy, still read)

Environment variables in the project root. Used for `PROXY_BASE_URL`, `PROXY_API_KEY`, `VOICE_MODE`. Settings.json takes priority where both exist.

---

## Updating

To update to a new version:

1. Get the new zip or `git pull`
2. Activate venv: `venv\Scripts\activate`
3. Reinstall: `pip install -e ".[full]"`
4. Relaunch: `coach.bat`

Your `.env` config is preserved across updates.

---

## Troubleshooting

### install.bat says "Python is not installed"
- Make sure you checked "Add Python to PATH" during Python install
- Try closing and reopening Command Prompt
- Or add Python to PATH manually: System Settings > Environment Variables > PATH > add `C:\Users\YOU\AppData\Local\Programs\Python\Python3XX\`

### pip install fails with build errors
Some packages need C++ build tools. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), check "Desktop development with C++".

### coach.bat says "Virtual environment not found"
Run `install.bat` first.

### Coach starts but no game state appears
- Make sure MTGA is running and you're in a game
- Check that "Detailed Logs" is enabled in MTGA settings
- The coach reads `%AppData%\LocalLow\Wizards Of The Coast\MTGA\Player.log`

### Voice not working
- Check your microphone is set as default recording device in Windows Sound settings
- Try `VOICE_MODE=none` in `.env` to disable voice and use text-only

### Win plan beep sounds but nothing happens
Press **Numpad 0** to hear the win plan read aloud. The sound means a viable win-in-2 or win-in-3 was found.
