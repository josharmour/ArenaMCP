"""Standalone MTGA Coach - Lightweight MCP client with voice I/O.

This app runs the ArenaMCP server and connects to it as an MCP client,
using a local LLM (Gemini/Ollama) for coaching advice with voice support.

Usage:
    python -m arenamcp.standalone --backend gemini
    python -m arenamcp.standalone --backend ollama --model llama3.2
    python -m arenamcp.standalone --draft --set MH3

The MCP server handles all game state tracking; this client just:
- Polls MCP tools for state changes
- Passes state to local LLM for advice
- Handles voice I/O (PTT/VOX input, TTS output)
"""

# Load .env before other imports
def _load_dotenv():
    """Load environment variables from .env file if it exists."""
    import os
    from pathlib import Path
    for env_path in [Path(".env"), Path(__file__).parent.parent.parent / ".env"]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        os.environ.setdefault(key.strip(), value.strip())
            break

_load_dotenv()

import argparse
import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from arenamcp.settings import get_settings

# Configure logging
LOG_DIR = Path.home() / ".arenamcp"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "standalone.log"

file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
root_logger.addHandler(file_handler)

# Suppress noisy third-party loggers
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


def copy_to_clipboard(text: str) -> bool:
    """Copy text to the Windows clipboard.

    Tries pyperclip first, falls back to Windows clip command.
    Returns True if successful, False otherwise.
    """
    # Try pyperclip first (if installed)
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"pyperclip failed: {e}")

    # Fallback: Windows clip command
    try:
        process = subprocess.Popen(
            ['clip'],
            stdin=subprocess.PIPE,
            shell=True
        )
        process.communicate(input=text.encode('utf-8'))
        return process.returncode == 0
    except Exception as e:
        logger.debug(f"clip command failed: {e}")
        return False


# Import dependencies
try:
    import keyboard
except ImportError:
    keyboard = None
    logger.warning("keyboard module not available - hotkeys disabled")


class UIAdapter:
    """Interface for UI feedback (CLI or TUI)."""
    def log(self, message: str) -> None: pass
    def advice(self, text: str, seat_info: str) -> None: pass
    def status(self, key: str, value: str) -> None: pass
    def error(self, message: str) -> None: pass

class CLIAdapter(UIAdapter):
    """Default adapter for CLI output."""
    def log(self, message: str) -> None:
        print(message)
    def advice(self, text: str, seat_info: str) -> None:
        print(f"\n[COACH|{seat_info}] {text}\n")
    def status(self, key: str, value: str) -> None:
        print(f"[{key}] {value}")
    def error(self, message: str) -> None:
        print(f"ERROR: {message}")

class MCPClient:
    """Simple in-process MCP client that calls server tools directly.

    Since the MCP server runs in-process, we import and call tools directly
    rather than going through STDIO transport.
    """

    def __init__(self):
        """Initialize MCP client by importing server module."""
        # Import server module - this starts the log watcher
        from arenamcp import server
        self._server = server

        # Ensure watcher is running
        server.start_watching()
        logger.info("MCP server initialized")

    def get_game_state(self) -> dict[str, Any]:
        """Call get_game_state MCP tool."""
        return self._server.get_game_state()

    def clear_pending_combat_steps(self) -> None:
        """Clear pending combat steps after trigger processing."""
        self._server.clear_pending_combat_steps()

    def poll_log(self) -> None:
        """Manually poll for new log content (backup for missed watchdog events)."""
        self._server.poll_log()

    def get_draft_pack(self) -> dict[str, Any]:
        """Call get_draft_pack MCP tool."""
        return self._server.get_draft_pack()

    def get_card_info(self, arena_id: int) -> dict[str, Any]:
        """Call get_card_info MCP tool."""
        return self._server.get_card_info(arena_id)

    def start_draft_helper(self, set_code: Optional[str] = None) -> dict[str, Any]:
        """Start the built-in draft helper."""
        return self._server.start_draft_helper_tool(set_code)

    def stop_draft_helper(self) -> dict[str, Any]:
        """Stop the draft helper."""
        return self._server.stop_draft_helper_tool()

    def get_draft_helper_status(self) -> dict[str, Any]:
        """Get draft helper status."""
        return self._server.get_draft_helper_status()

    def get_sealed_pool(self) -> dict[str, Any]:
        """Get sealed pool analysis."""
        return self._server.get_sealed_pool()


class UIAdapter:
    """Interface for UI feedback."""
    def log(self, message: str) -> None: ...
    def advice(self, text: str, seat_info: str) -> None: ...
    def status(self, key: str, value: str) -> None: ...
    def error(self, message: str) -> None: ...
    def speak(self, text: str) -> None: ...

class ConsoleAdapter(UIAdapter):
    """Fallback for CLI mode."""
    def log(self, message: str) -> None: print(message, end='')
    def advice(self, text: str, seat_info: str) -> None: print(f"\n[COACH|{seat_info}] {text}\n")
    def status(self, key: str, value: str) -> None: pass
    def error(self, message: str) -> None: print(f"ERROR: {message}")
    def speak(self, text: str) -> None: pass 


class StandaloneCoach:
    """Standalone coaching app using MCP client + local LLM."""

    def __init__(
        self,
        backend: str = "gemini",
        model: Optional[str] = None,
        voice_mode: str = "ptt",
        draft_mode: bool = False,
        set_code: Optional[str] = None,
        ui_adapter: Optional[UIAdapter] = None,
        register_hotkeys: bool = True,
    ):
        self._register_keyboard = register_hotkeys

        # Load settings
        self.settings = get_settings()

        # Resolve configuration (Args > Settings > Defaults)
        self._backend_name = backend or self.settings.get("backend", "gemini")
        self._model_name = model or self.settings.get("model")
        self._voice_mode = voice_mode or self.settings.get("voice_mode", "ptt")

        self.draft_mode = draft_mode
        self.set_code = set_code.upper() if set_code else None

        # State
        self.advice_style = "concise"
        self._advice_frequency = self.settings.get("advice_frequency", "start_of_turn")

        # TTS always enabled for non-realtime backends (GPT-realtime has native audio)
        self._auto_speak = True

        self.ui = ui_adapter or CLIAdapter()

        # Save validated configuration back to settings (ensure consistency)
        self.settings.set("backend", self._backend_name, save=False)
        if self._model_name:
            self.settings.set("model", self._model_name, save=False)
        self.settings.set("voice_mode", self._voice_mode, save=False)
        self.settings.set("advice_frequency", self._advice_frequency, save=True)

        self._running = False
        self._restart_requested = False
        self._mcp: Optional[MCPClient] = None

        # Voice components
        self._voice_input = None
        self._voice_output = None

        # GPT-realtime client (for realtime voice mode)
        self._realtime_client = None

        # LLM backend
        self._coach = None
        self._trigger = None

        # Threads
        self._coaching_thread: Optional[threading.Thread] = None
        self._voice_thread: Optional[threading.Thread] = None

    def speak_advice(self, text: str, blocking: bool = True) -> None:
        """Speak advice using the appropriate TTS method.

        For GPT-realtime: sends text to realtime API and plays audio response
        For other backends: uses local Kokoro TTS
        """
        if not text:
            return

        # Filter out passive calls from TTS (User Request)
        # We silence: Wait, Pass, No actions
        clean_text = text.lower().strip(" .!")
        silence_triggers = [
            "wait", 
            "pass", 
            "pass priority", 
            "no actions", 
            "wait for opponent",
            "opponent has priority"
        ]
        
        # Check if text starts with or is substantially just these phrases
        # We use a simple heuristic: if it contains no active verbs (Cast, Attack, Block, Play),
        # and matches a silence trigger, we skip it.
        is_passive = any(trigger in clean_text for trigger in silence_triggers)
        has_action = any(verb in clean_text for verb in ["cast", "play", "attack", "block", "activate", "kill", "destroy"])
        
        if is_passive and not has_action and len(text) < 60:
            return

        if self.backend_name == "gpt-realtime":
            # Use GPT-realtime's TTS
            try:
                from arenamcp.realtime import GPTRealtimeClient, RealtimeConfig, play_audio_pcm16

                # Lazy init realtime client for TTS
                if self._realtime_client is None:
                    config = RealtimeConfig(
                        turn_detection_type="none",  # Manual mode for TTS-only
                    )
                    config.instructions = "You are an English-speaking voice assistant. Read the following text exactly as provided in English, with natural intonation. Always respond in English only."
                    self._realtime_client = GPTRealtimeClient(config)
                    if not self._realtime_client.connect():
                        logger.error("Failed to connect GPT-realtime for TTS")
                        self._realtime_client = None
                        return

                # Send text and get audio response
                self._realtime_client.send_text(text)
                response = self._realtime_client.get_response(timeout=15.0)

                if response and response.get("audio"):
                    play_audio_pcm16(response["audio"])

            except Exception as e:
                logger.error(f"GPT-realtime TTS error: {e}")
        else:
            # Use local Kokoro TTS
            if self._voice_output:
                try:
                    self._voice_output.speak(text, blocking=blocking)
                except Exception as e:
                    logger.error(f"Kokoro TTS error: {e}")

    @property
    def backend_name(self) -> str:
        return self._backend_name
    
    @backend_name.setter
    def backend_name(self, value: str):
        self._backend_name = value
        self.settings.set("backend", value)
        
    @property
    def model_name(self) -> Optional[str]:
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: Optional[str]):
        self._model_name = value
        if value:
            self.settings.set("model", value)
            
    @property
    def voice_mode(self) -> str:
        return self._voice_mode
    
    @voice_mode.setter
    def voice_mode(self, value: str):
        self._voice_mode = value
        self.settings.set("voice_mode", value)
        if hasattr(self, "_voice_input") and self._voice_input:
            # Propagate to input handler if running
            # (Note: VoiceInput might need restart to change mode fully, preventing hot-swap here)
            pass

    @property
    def advice_frequency(self) -> str:
        return self._advice_frequency

    @advice_frequency.setter
    def advice_frequency(self, value: str):
        self._advice_frequency = value
        self.settings.set("advice_frequency", value)

    def _init_mcp(self) -> None:
        """Initialize MCP client connection."""
        logger.info("Initializing MCP server...")
        self._mcp = MCPClient()

        # Pre-initialize card databases to avoid lazy-load delays during gameplay
        logger.info("Pre-loading card databases...")
        try:
            from arenamcp.mtgjson import get_mtgjson
            from arenamcp.server import _get_scryfall, _get_mtgadb
            get_mtgjson()  # Load MTGJSON (primary source, updated daily)
            _get_mtgadb()  # Load MTGA local database
            _get_scryfall()  # Load Scryfall cache (fallback)
        except Exception as e:
            logger.warning(f"Failed to pre-load databases: {e}")

    def _init_llm(self) -> None:
        """Initialize LLM backend for coaching."""
        if self.draft_mode:
            return  # Draft mode uses MCP's built-in draft helper

        from arenamcp.coach import CoachEngine, GameStateTrigger, create_backend

        llm_backend = create_backend(self.backend_name, model=self.model_name)
        actual_model = getattr(llm_backend, 'model', 'unknown')
        logger.info(f"Created {self.backend_name} backend with model: {actual_model}")
        self._coach = CoachEngine(backend=llm_backend)
        self._trigger = GameStateTrigger()

    def _init_voice(self) -> None:
        """Initialize voice I/O components."""
        logger.info(f"_init_voice called, backend_name={self.backend_name}")
        # GPT-realtime handles its own STT/TTS - skip local initialization
        if self.backend_name == "gpt-realtime":
            logger.info("Using GPT-Realtime for voice I/O (no local TTS/Whisper)")
            self._voice_output = None
            self._voice_input = None
            return

        from arenamcp.tts import VoiceOutput
        from arenamcp.voice import VoiceInput

        # Initialize local TTS (Kokoro)
        logger.info("Initializing TTS (Kokoro)...")
        self._voice_output = VoiceOutput()
        voice_id, voice_desc = self._voice_output.current_voice
        logger.info(f"TTS voice: {voice_desc}")
        self.ui.status("VOICE", f"TTS Voice: {voice_desc}")

        # Initialize local STT (Whisper via VoiceInput)
        logger.info(f"Initializing voice input ({self.voice_mode})...")
        self._voice_input = VoiceInput(mode=self.voice_mode)

    def _coaching_loop(self) -> None:
        """Poll MCP for game state and provide coaching, with auto-draft detection."""
        logger.info("Coaching loop started")
        prev_state: dict[str, Any] = {}
        seat_announced = False

        last_advice_turn = 0
        last_advice_phase = ""
        # Critical triggers that always fire regardless of frequency setting
        # Combat triggers removed - too noisy for "start_of_turn" mode
        CRITICAL_PRIORITY = {"stack_spell", "low_life", "opponent_low_life"}

        # Draft/Sealed detection state
        in_draft_mode = False
        in_sealed_mode = False
        sealed_analyzed = False
        last_draft_pack = 0
        last_draft_pick = 0
        last_inactive_log = 0

        while self._running:
            try:
                # Check for active draft/sealed first
                draft_pack = self._mcp.get_draft_pack()

                if draft_pack.get("is_active"):
                    is_sealed = draft_pack.get("is_sealed", False)

                    if is_sealed:
                        # SEALED MODE
                        if not in_sealed_mode:
                            in_sealed_mode = True
                            in_draft_mode = False
                            set_code = draft_pack.get("set_code", "???")
                            self.ui.status("SEALED", f"Detected sealed event: {set_code}")
                            self.ui.log("[SEALED] Waiting for pool to be opened...\n")
                            logger.info(f"Auto-detected sealed: {set_code}")

                        # Check if pool is ready for analysis
                        if not sealed_analyzed:
                            sealed_result = self._mcp.get_sealed_pool()
                            pool_size = sealed_result.get("pool_size", 0)

                            if pool_size > 0:
                                sealed_analyzed = True
                                self.ui.log(f"\n[SEALED] Pool opened ({pool_size} cards)")
                                self.ui.log(sealed_result.get("detailed_text", ""))
                                self.ui.log("")

                                # Speak the recommendation
                                advice = sealed_result.get("spoken_advice", "")
                                if advice:
                                    logger.info(f"SEALED ADVICE: {advice}")
                                    self.speak_advice(advice)

                        time.sleep(2.0)  # Slower polling for sealed
                        continue

                    else:
                        # DRAFT MODE
                        pack_num = draft_pack.get("pack_number", 0)
                        pick_num = draft_pack.get("pick_number", 0)
                        cards = draft_pack.get("cards", [])
                        
                        # New pack detected
                        if cards and (pack_num != last_draft_pack or pick_num != last_draft_pick):
                            if not in_draft_mode:
                                in_draft_mode = True
                                in_sealed_mode = False
                                set_code = draft_pack.get("set_code", "???")
                                self.ui.status("DRAFT", f"Detected draft: {set_code}")
                                self.ui.log("[DRAFT] Auto-switching to draft advice mode\n")
                                logger.info(f"Auto-detected draft: {set_code}")

                            last_draft_pack = pack_num
                            last_draft_pick = pick_num

                            # Find best pick by GIH win rate
                            best_card = None
                            best_wr = 0.0
                            for card in cards:
                                gih_wr = card.get("gih_wr", 0) or 0
                                if gih_wr > best_wr:
                                    best_wr = gih_wr
                                    best_card = card

                            if best_card:
                                name = best_card.get("name", "Unknown")
                                wr_pct = f"{best_wr * 100:.1f}%" if best_wr else "N/A"
                                advice = f"Pick {name}, {wr_pct} win rate"

                                self.ui.log(f"\n[DRAFT P{pack_num}P{pick_num}] {advice}\n")
                                logger.info(f"DRAFT: P{pack_num}P{pick_num} - {advice}")
                                self.speak_advice(advice)
                            else:
                                msg = f"No recommended pick found (Cards: {len(cards)}, Best WR: {best_wr})"
                                self.ui.log(f"\n[DRAFT] {msg}\n")
                                logger.warning(msg)

                        time.sleep(1.0)  # Faster polling during draft
                        continue

                else:
                    # Inactive draft/sealed
                    if time.time() - last_inactive_log > 10.0:
                        logger.info("Draft inactive (waiting for event entry...)")
                        last_inactive_log = time.time()

                # Not in draft/sealed - regular game coaching
                if in_draft_mode or in_sealed_mode:
                    mode_name = "Sealed" if in_sealed_mode else "Draft"
                    in_draft_mode = False
                    in_sealed_mode = False
                    sealed_analyzed = False
                    self.ui.log(f"\n[{mode_name.upper()}] {mode_name} complete, switching to game coaching\n")
                    logger.info(f"{mode_name} ended, resuming game coaching")
                    last_draft_pack = 0
                    last_draft_pick = 0

                curr_state = self._mcp.get_game_state()
                turn = curr_state.get("turn", {})
                turn_num = turn.get("turn_number", 0)
                phase = turn.get("phase", "")

                # Detect new game (turn number decreased) and reset advice tracking
                if turn_num > 0 and turn_num < last_advice_turn:
                    logger.info(f"New game detected in coaching loop (turn {last_advice_turn} -> {turn_num}), resetting advice tracking")
                    last_advice_turn = 0
                    last_advice_phase = ""
                    seat_announced = False  # Re-announce seat for new game

                # Announce seat detection when game starts
                if not seat_announced and turn_num > 0:
                    players = curr_state.get("players", [])
                    for p in players:
                        if p.get("is_local"):
                            seat_id = p.get("seat_id")
                            self.ui.status("GAME", f"Detected as Seat {seat_id} - press F8 if this is wrong")
                            logger.info(f"Game detected, local seat = {seat_id}")
                            seat_announced = True
                            break

                if prev_state and self._trigger and turn_num > 0:
                    # Auto-detect draft mode
                    try:
                        draft_state = self._mcp.get_draft_pack()
                        is_draft_active = draft_state.get("is_active", False)
                        
                        if is_draft_active and not self.draft_mode:
                            logger.info("Auto-detected draft - enabling draft mode")
                            self.draft_mode = True
                            self.ui.status("MODE", "Draft")
                        elif not is_draft_active and self.draft_mode:
                            logger.info("Draft ended - disabling draft mode")
                            self.draft_mode = False
                            self.ui.status("MODE", "Game")
                    except Exception as e:
                        logger.debug(f"Draft detection error: {e}")

                    triggers = self._trigger.check_triggers(prev_state, curr_state)

                    # Clear pending combat steps after checking (they're now processed)
                    self._mcp.clear_pending_combat_steps()

                    # Sort triggers by priority to ensure we handle the most critical one only
                    # Priority order: Critical > Combat > Turn > Priority
                    trigger_priorities = {
                        "stack_spell": 10,
                        "low_life": 9,
                        "opponent_low_life": 8,
                        "combat_attackers": 7,
                        "combat_blockers": 6,
                        "new_turn": 5,
                        "priority_gained": 1
                    }

                    triggers.sort(key=lambda x: trigger_priorities.get(x, 0), reverse=True)

                    for trigger in triggers:
                        # Critical triggers always fire (stack spells, low life)
                        is_critical = trigger in CRITICAL_PRIORITY

                        # New turn triggers once per turn
                        is_new_turn = trigger == "new_turn" and turn_num > last_advice_turn

                        # Combat and priority triggers only in "every_priority" mode
                        is_frequent = (
                            self.advice_frequency == "every_priority" and
                            trigger in ("priority_gained", "combat_attackers", "combat_blockers") and
                            (turn_num > last_advice_turn or phase != last_advice_phase)
                        )
                        
                        # Additional check: Don't spam priority triggers if we just advised on new_turn
                        # unless distinct phase
                        if trigger == "priority_gained" and is_new_turn:
                            continue

                        should_advise = is_critical or is_new_turn or is_frequent

                        if not should_advise:
                            continue

                        logger.info(f"TRIGGER: {trigger}")
                        
                        # CRITICAL FIX: Only process ONE trigger per poll cycle for Realtime
                        # To prevent "active response in progress" errors from rapid-fire triggers
                        if self.backend_name in ("gemini-live", "gpt-realtime"):
                            # If we decided to advise, we stop processing other triggers
                            # We already prioritized the list above
                            # Note: filtering logic is done, we just proceed to advise generation below
                            pass # logic continues to existing advice block

                        # We will break AFTER advice is generated (lines 609+)
                        
                        if self._coach:
                            advice = self._coach.get_advice(
                                curr_state,
                                trigger=trigger,
                                style=self.advice_style
                            )
                            logger.info(f"ADVICE: {advice}")

                            # Record for debug history
                            self._record_advice(advice, trigger)

                            # CRITICAL: Break loop for Realtime backends
                            if self.backend_name in ("gemini-live", "gpt-realtime"):
                                break

                            last_advice_turn = turn_num
                            last_advice_phase = phase

                            # Show seat info with advice
                            local_seat = None
                            for p in curr_state.get("players", []):
                                if p.get("is_local"):
                                    local_seat = p.get("seat_id")
                                    break

                            # Count untapped lands for mana display
                            battlefield = curr_state.get("battlefield", [])
                            your_cards = [c for c in battlefield if c.get("owner_seat_id") == local_seat]
                            untapped_lands = sum(1 for c in your_cards
                                                 if "land" in c.get("type_line", "").lower()
                                                 and not c.get("is_tapped"))

                            seat_info = f"Seat {local_seat}|{untapped_lands} mana|{self.backend_name}" if local_seat else "Seat ?"
                            self.ui.advice(advice, seat_info)
                            self.speak_advice(advice)

                prev_state = curr_state

            except Exception as e:
                logger.error(f"Coaching loop error: {e}")
                logger.debug(traceback.format_exc())
                self._record_error(str(e), "coaching_loop")

            time.sleep(1.5)

        logger.info("Coaching loop stopped")

    def _voice_loop(self) -> None:
        """Handle voice input for questions (PTT mode with Whisper + Kokoro)."""
        if not self._voice_input:
            return

        logger.info(f"Voice loop started ({self.voice_mode})")
        if self.voice_mode == "ptt":
            self.ui.log("\n[MIC] Press F4 to ask (tap for quick advice)\n")
        else:
            self.ui.log("\n[MIC] Voice activation enabled\n")

        self._voice_input.start()

        while self._running:
            try:
                text = self._voice_input.wait_for_speech(timeout=2.0)

                if not self._voice_input._result_ready.is_set():
                    continue

                if self._coach and self._mcp:
                    game_state = self._mcp.get_game_state()

                    # Get current seat and mana for display
                    local_seat = None
                    for p in game_state.get("players", []):
                        if p.get("is_local"):
                            local_seat = p.get("seat_id")
                            break

                    # Count untapped lands for mana display
                    battlefield = game_state.get("battlefield", [])
                    your_cards = [c for c in battlefield if c.get("owner_seat_id") == local_seat]
                    untapped_lands = sum(1 for c in your_cards
                                         if "land" in c.get("type_line", "").lower()
                                         and not c.get("is_tapped"))

                    seat_info = f"Seat {local_seat}|{untapped_lands} mana|{self.backend_name}" if local_seat else "Seat ?"

                    # Check if we can use direct audio with Gemini
                    audio_data = self._voice_input.get_last_audio()
                    use_direct_audio = (
                        self.backend_name == "gemini" and
                        audio_data is not None and
                        len(audio_data) > 0 and
                        hasattr(self._coach._backend, 'complete_with_audio')
                    )

                    if use_direct_audio:
                        # Direct audio to Gemini - skip local transcription
                        logger.info(f"AUDIO INPUT: {len(audio_data)} samples -> Gemini")
                        self.ui.log("\n[AUDIO] Sending to Gemini...")
                        context = self._coach._format_game_context(game_state)

                        # FORCE specific answer mode
                        user_message = (
                            f"{context}\n\n"
                            "IMPORTANT: The user just asked a specific question via audio (attached). "
                            "Do NOT give generic gameplay advice. "
                            "Listen to the audio and answer EXACTLY what they asked. "
                            "If they asked about a specific card, interaction, or rule, explain it in detail. "
                            "Ignore your usual brevity constraints if needed to answer fully."
                        )
                        advice = self._coach._backend.complete_with_audio(
                            self._coach._system_prompt,
                            user_message,
                            audio_data
                        )
                    elif text and text.strip():
                        logger.info(f"QUESTION: {text}")
                        self.ui.log(f"\n[YOU] {text}")
                        advice = self._coach.get_advice(game_state, question=text)
                    else:
                        logger.info("QUICK ADVICE (F4 tap)")
                        self.ui.log("\n[QUICK] Analyzing...")
                        advice = self._coach.get_advice(game_state, trigger="user_request")

                    logger.info(f"RESPONSE: {advice}")
                    self.ui.advice(advice, seat_info)
                    self.speak_advice(advice)

                    # Record for debug history
                    trigger = "voice_audio" if use_direct_audio else ("voice_question" if text else "voice_quick")
                    self._record_advice(advice, trigger)

            except Exception as e:
                if self._running:
                    logger.error(f"Voice loop error: {e}")
                    self._record_error(str(e), "voice_loop")

        self._voice_input.stop()
        logger.info("Voice loop stopped")

    def _realtime_voice_loop(self) -> None:
        """Handle voice I/O for GPT-realtime with proactive coaching.

        This creates a persistent connection that:
        - Streams user audio continuously (server VAD detects speech)
        - Sends proactive prompts on game triggers
        - Plays back audio responses
        - Allows natural interruption by either party
        """
        logger.info(">>> _realtime_voice_loop ENTERED <<<")
        try:
            import sounddevice as sd
            import numpy as np
            from arenamcp.realtime import GPTRealtimeClient, RealtimeConfig, play_audio_pcm16, stop_audio
            from arenamcp.gemini_live import GeminiLiveClient
            from arenamcp.coach import GameStateTrigger
        except ImportError as e:
            logger.error(f"Failed to import realtime dependencies: {e}")
            self.ui.error(f"Realtime import error: {e}")
            return

        logger.info(f"{self.backend_name} voice loop starting...")
        self.ui.log(f"\n[REALTIME] Connecting to {self.backend_name}...\n")

        # Configure for server-side VAD with interruption support
        config = RealtimeConfig(
            turn_detection_type="server_vad",
            vad_threshold=0.5,
            vad_prefix_padding_ms=300,
            vad_silence_duration_ms=600,  # Shorter silence = faster responses
        )
        
        if self.model_name:
            config.deployment = self.model_name

        # Build system instructions with game context
        def get_instructions():
            base = """LANGUAGE: You MUST respond in ENGLISH only. No other languages.

You are an expert Magic: The Gathering Arena coach providing real-time voice advice.

CRITICAL RULES:
- ALWAYS respond in ENGLISH. Never use any other language.
- ALWAYS use the EXACT card names from the game state
- Keep responses to 1-2 sentences max - you're speaking aloud
- Be decisive and specific - name the EXACT card to play
- If you have nothing useful to say, stay SILENT - don't fill time with chatter

CARD SYNERGIES - Read oracle text carefully:
- Lifegain triggers: Cards like "Ajani's Pridemate" or "Twinblade Paladin" say "Whenever you gain life" - these combo with lifelink creatures, lifegain lands (Scoured Barrens), and spells that gain life
- ETB triggers: "When this creature enters" effects - sequence plays to maximize value
- Death triggers: "When this creature dies" - consider sacrifice synergies
- +1/+1 counters: Stack with other counter effects for exponential growth

TIMING & PRIORITY:
- On YOUR turn: Recommend plays in order - land drop first, then creatures/spells
- On OPPONENT'S turn: Only speak if you have INSTANT-speed options (instants, flash, activated abilities) or something is on the stack to respond to. Otherwise stay SILENT.
- Stack responses: If opponent casts something threatening, suggest responses from hand
- Combat math: Calculate if attacks are profitable considering blocks

PRIORITIES:
1. Lethal detection - can you or opponent win THIS turn?
2. Stack interactions - should you respond to what's happening?
3. Combat math - attacks/blocks that are favorable trades
4. Curve plays - what's the best use of available mana?
5. Synergy setups - plays that enable future combos

Example: "Play Scoured Barrens, then cast Ajani's Pridemate - it'll get a counter from the land's lifegain."
NOT: "Play a land and a creature." """

            if self._mcp:
                try:
                    game_state = self._mcp.get_game_state()
                    if game_state.get("turn", {}).get("turn_number", 0) > 0:
                        context = self._coach._format_game_context(game_state) if self._coach else ""
                        return f"{base}\n\nCURRENT GAME STATE:\n{context}"
                except Exception:
                    pass
            return base

        config.instructions = get_instructions()

        config.instructions = get_instructions()

        # Instantiate appropriate client
        if self.backend_name == "gemini-live":
            self.ui.log("[yellow]Initializing Gemini Live client...[/]")
            client = GeminiLiveClient(config)
        else:
            client = GPTRealtimeClient(config)
            
        self._realtime_client = client  # Store reference for cleanup

        # Track if we're currently playing audio (to avoid overlapping)
        is_playing = threading.Event()
        playback_queue = queue.Queue()  # Queue for streaming audio chunks

        def stop_playback():
            """Stop current audio playback and clear client buffers."""
            # Clear internal queue
            while not playback_queue.empty():
                try:
                    playback_queue.get_nowait()
                    playback_queue.task_done()
                except queue.Empty:
                    break
            
            is_playing.clear()
            
            # Interrupt client generation if possible
            if hasattr(client, 'interrupt'):
                client.interrupt()
            else:
                client.clear_audio_buffer()

        def play_audio_callback(audio: bytes):
            """Queue audio chunk for playback."""
            if audio:
                playback_queue.put(audio)

        def on_response_done(transcript: str, audio: bytes):
            if transcript:
                self.ui.advice(transcript, self.backend_name)
                logger.info(f"REALTIME RESPONSE: {transcript}")
                self._record_advice(transcript, "gpt_realtime")
            # Note: Audio is handled by on_audio streaming callback

        def on_error(error: str):
            self.ui.log(f"[red]REALTIME ERROR: {error}[/]")
            logger.error(f"Realtime error: {error}")
            self._record_error(error, f"{self.backend_name}_callback")

        client.set_callbacks(
            on_response_done=on_response_done,
            on_error=on_error,
            on_audio=play_audio_callback,  # Use the queue-based callback
        )

        if self.backend_name == "gemini-live":
            self.ui.log(f"[yellow]Attempting Gemini Live connection ({self.model_name})...[/]")
        else:
            self.ui.log("[yellow]Attempting GPT-Realtime connection...[/]")
            
        if not client.connect():
            self.ui.log(f"[red]Failed to connect to {self.backend_name}[/]")
            if self.backend_name == "gemini-live":
                self.ui.log("[red]Check GOOGLE_API_KEY in .env[/]")
            else:
                self.ui.log("[red]Check AZURE_REALTIME_API_KEY and endpoint[/]")
            return

        self.ui.log(f"[green]{self.backend_name} connected![/]")
        self.ui.log("[green]I'm listening - speak anytime to ask questions.[/]")
        self.ui.log("[green]I'll give proactive advice as the game progresses.[/]\n")

        # Initialize trigger detector for proactive advice
        trigger = GameStateTrigger()
        prev_state: dict = {}
        last_advice_turn = 0
        last_advice_phase = ""

        # Critical triggers that warrant interruption
        URGENT_TRIGGERS = {"stack_spell", "low_life", "opponent_low_life", "decision_required"}
        NORMAL_TRIGGERS = {"new_turn", "combat_attackers", "combat_blockers", "priority_gained"}

        # Start continuous audio capture
        sample_rate = 16000

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio status: {status}")
            # Don't send audio while we're playing back (prevents feedback loop)
            if is_playing.is_set():
                return
            
            # Check PTT state
            if self._voice_mode == "ptt":
                try:
                    import keyboard
                    # Hardcoded F4 for now, consistent with other PTT usage
                    if not keyboard.is_pressed('f4'):
                        return
                except ImportError:
                    pass

            # Send audio to GPT-realtime / Gemini continuously
            audio_chunk = indata.copy().flatten()
            client.send_audio(audio_chunk, sample_rate)

        try:
            # Use system default device
            device_name = "System Default"

            self.ui.log(f"[yellow]Starting audio capture on: {device_name}[/]")
            logger.info(f"Using audio input device: {device_name}")

            # Initialize Persistent Output Stream (24kHz is standard for Gemini/GPT Realtime)
            # persistent stream avoids heap corruption from repeated sd.play() calls
            # FORCE SYSTEM DEFAULT ON STARTUP (Ignore saved index to prevent stale device mapping)
            device_idx = None 
            if self.settings.get("device_index") is not None:
                logger.info("Ignoring saved audio device index - forcing System Default as requested")

            # Log default devices
            try:
                defaults = sd.default.device
                device_info_in = sd.query_devices(defaults[0])
                device_info_out = sd.query_devices(defaults[1])
                logger.info(f"Default Audio Input: {device_info_in['name']} (idx {defaults[0]})")
                logger.info(f"Default Audio Output: {device_info_out['name']} (idx {defaults[1]})")
            except Exception as e:
                logger.error(f"Failed to query default devices: {e}")

            # Force default device explicit selection if none provided
            if device_idx is None:
                 try:
                     # Explicitly query default output device index
                     default_out_idx = sd.default.device[1]
                     device_info = sd.query_devices(default_out_idx)
                     logger.info(f"Forcing default output device: {device_info['name']} (idx {default_out_idx})")
                     device_idx = default_out_idx
                 except Exception as e:
                     logger.warning(f"Could not determine default device explicitly: {e}")

            output_stream = sd.OutputStream(
                samplerate=24000, 
                channels=1, 
                dtype='int16',
                blocksize=4096, # Increased buffer size
                latency=0.2,    # Higher latency (200ms) to prevent underrun/stuttering
                device=device_idx
            )
            output_stream.start()
            
            actual_out_idx = device_idx if device_idx is not None else sd.default.device[1]
            try:
                actual_device = sd.query_devices(actual_out_idx)
                self.ui.log(f"[dim]Audio Output: {actual_device['name']}[/]")
                logger.info(f"Using Audio Output Device: {actual_device['name']} (idx {actual_out_idx})")
            except:
                pass

            # --- THREAD-SAFE AUDIO PLAYBACK ---
            # Decouple network thread from audio hardware blocking

            def playback_worker():
                logger.info(f"Playback worker started on thread {threading.current_thread().name}")
                while not self._stop_event.is_set():
                    try:
                        # Blocking wait for first chunk
                        chunk = playback_queue.get(timeout=0.1)
                        if chunk is None:
                            break
                        
                        # Set flag to suppress mic input (Echo Cancellation)
                        is_playing.set()
                        
                        # Batching: Try to grab more chunks if available to smooth playback
                        buffer = [chunk]
                        try:
                            while True:
                                # Non-blocking get for subsequent chunks
                                extra = playback_queue.get_nowait()
                                if extra is None:
                                    break
                                buffer.append(extra)
                                playback_queue.task_done()
                                if len(buffer) >= 10: # Limit batch size
                                    break
                        except queue.Empty:
                            pass
                            
                        # Play batched audio
                        try:
                            # Verify we have data
                            total_bytes = sum(len(b) for b in buffer)
                            # logger.debug(f"Writing {total_bytes} bytes to audio stream") 
                            
                            full_data = b"".join(buffer)
                            audio_data = np.frombuffer(full_data, dtype=np.int16)
                            output_stream.write(audio_data)
                        except Exception as e:
                            logger.error(f"Playback write error: {e}")
                            self.ui.log(f"[red]Audio write error: {e}[/]")
                        finally:
                            # Mark the initial chunk as done
                            playback_queue.task_done()
                            
                            # If queue is empty, we are done playing for now
                            if playback_queue.empty():
                                is_playing.clear()
                                
                    except queue.Empty:
                        is_playing.clear()
                    except Exception as e:
                         logger.error(f"Playback worker fatal error: {e}")

            # Start playback thread
            threading.Thread(target=playback_worker, daemon=True, name="AudioPlayback").start()

            # (Removed redundant definitions: play_audio_async, stop_playback)
                

            with sd.InputStream(
                device=None,  # None = Windows system default input
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                blocksize=int(sample_rate * 0.1),  # 100ms chunks
            ):
                self.ui.log("[green]Audio capture started - speak anytime![/]")
                logger.info("Audio stream started - listening for speech")

                last_context_update = 0
                game_announced = False
                last_proactive_advice = 0  # Timer for periodic proactive advice
                PROACTIVE_INTERVAL = 60.0  # Give proactive advice every 60 seconds if no triggers

                # Debug: show we're in the loop
                loop_count = 0

                while self._running:
                    # Auto-reconnection for stability (e.g. handling 1011 keepalive timeouts)
                    if hasattr(client, "_connected") and not client._connected:
                        self.ui.log("[bold red]Connection lost! Attempting reconnect...[/]")
                        logger.warning("Realtime client disconnected. Reconnecting...")
                        time.sleep(1) # Brief cooldown
                        
                        if client.connect():
                            self.ui.log("[green]Reconnected successfully.[/]")
                            # Force immediate context refresh
                            last_context_update = 0
                            # Reset advice triggers so full context is resent
                            game_announced = False 
                            last_advice_turn = 0
                            prev_state = {}
                            
                            # Re-send instructions to ensure session has them
                            try:
                                client.update_instructions(get_instructions())
                            except Exception as e:
                                logger.error(f"Failed to restore instructions: {e}")
                        else:
                            self.ui.log("[red]Reconnect failed. Retrying in 5s...[/]")
                            time.sleep(5)
                            continue

                    loop_count += 1
                    if loop_count == 1:
                        self.ui.log(f"[dim]Loop verified (Connected: {getattr(client, '_connected', 'Unknown')})[/]")

                    try:
                        # Poll for new log content (backup for missed watchdog events)
                        if self._mcp:
                            self._mcp.poll_log()

                        # Get current game state
                        if self._mcp:
                            curr_state = self._mcp.get_game_state()
                            turn = curr_state.get("turn", {})
                            turn_num = turn.get("turn_number", 0)
                            phase = turn.get("phase", "")
                            step = turn.get("step", "")

                            # Debug: show current turn status occasionally
                            if loop_count % 20 == 0 and turn_num > 0:  # Every 10 seconds if in game
                                hand_count = len(curr_state.get("hand", []))
                                self.ui.log(f"[dim]Turn {turn_num}, Phase: {phase}, Hand: {hand_count} cards[/]")

                            # Detect active game and announce if not yet done
                            if turn_num > 0 and not game_announced:
                                logger.info("Game detected, sending initial state")
                                game_announced = True
                                last_advice_turn = turn_num
                                last_advice_phase = phase

                                # Build game context
                                hand = curr_state.get("hand", [])
                                hand_names = [c.get("name", "?") for c in hand]
                                logger.info(f"HAND CONTENTS: {hand_names}")
                                self.ui.log(f"[cyan]Your hand: {', '.join(hand_names) if hand_names else '(empty)'}[/]")

                                context = self._coach._format_game_context(curr_state) if self._coach else str(curr_state)
                                logger.info(f"CONTEXT BEING SENT:\n{context[:500]}...")  # Log first 500 chars
                                self.ui.log(f"[green]>>> Game detected! Sending state to GPT...[/]")
                                self.ui.log(f"[dim]Context: {len(context)} chars[/]")
                                client.update_instructions(get_instructions())
                                client.send_text(f"Game in progress at turn {turn_num}. Here's the current state:\n{context}\n\nGive me advice on what to do.")

                            # Detect new game (turn reset)
                            if turn_num > 0 and turn_num < last_advice_turn:
                                logger.info("New game detected, resetting")
                                last_advice_turn = 0
                                last_advice_phase = ""
                                prev_state = {}
                                game_announced = False

                                # Update instructions for new game
                                client.update_instructions(get_instructions())

                                # Announce new game with context
                                context = self._coach._format_game_context(curr_state) if self._coach else str(curr_state)
                                client.send_text(f"A new game has started!\n{context}\n\nAnalyze the opening hand and give initial advice.")

                            # Check for triggers
                            if prev_state and turn_num > 0:
                                triggers = trigger.check_triggers(prev_state, curr_state)
                                self._mcp.clear_pending_combat_steps()

                                # Debug: log detected triggers
                                if triggers:
                                    # Sort triggers by priority to ensure we handle the most critical one only
                                    trigger_priorities = {
                                        "decision_required": 20,
                                        "stack_spell": 10,
                                        "low_life": 9,
                                        "opponent_low_life": 8,
                                        "combat_attackers": 7,
                                        "combat_blockers": 6,
                                        "new_turn": 5,
                                        "priority_gained": 1
                                    }
                                    triggers.sort(key=lambda x: trigger_priorities.get(x, 0), reverse=True)
                                    
                                    logger.info(f"TRIGGERS DETECTED (Sorted): {triggers}")
                                    self.ui.log(f"[dim]Triggers: {triggers}[/]")

                                for trig in triggers:
                                    is_urgent = trig in URGENT_TRIGGERS
                                    # Allow new_turn advice if turn increased OR if we haven't given advice this turn yet
                                    is_new_turn = trig == "new_turn" and turn_num >= last_advice_turn
                                    is_phase_change = trig in NORMAL_TRIGGERS and phase != last_advice_phase
                                    is_priority = trig == "priority_gained"  # Always advise on priority

                                    should_advise = is_urgent or is_new_turn or is_phase_change or is_priority

                                    # Debug: log trigger evaluation
                                    logger.debug(f"Trigger {trig}: urgent={is_urgent}, new_turn={is_new_turn}, phase_change={is_phase_change}, priority={is_priority} -> advise={should_advise}")

                                    # Don't interrupt if already speaking (unless urgent)
                                    # Don't skip just because we're speaking. New phase/turn = new context.
                                    # We will call stop_playback() below if we proceed.
                                    
                                    if should_advise:
                                        # Get hand card names for explicit reference
                                        hand = curr_state.get("hand", [])
                                        hand_cards = [f"{c.get('name')} ({c.get('mana_cost', '')})" for c in hand]
                                        hand_str = ", ".join(hand_cards) if hand_cards else "empty"

                                        # Check for instant-speed options in hand
                                        instant_options = []
                                        for c in hand:
                                            type_line = c.get("type_line", "").lower()
                                            oracle = c.get("oracle_text", "").lower()
                                            if "instant" in type_line or "flash" in oracle:
                                                instant_options.append(c.get("name"))

                                        # Get battlefield summary
                                        battlefield = curr_state.get("battlefield", [])
                                        local_player = next((p for p in curr_state.get("players", []) if p.get("is_local")), {})
                                        my_seat = local_player.get("seat_id", 1)
                                        my_creatures = [c.get("name") for c in battlefield
                                                       if c.get("owner_seat_id") == my_seat and c.get("power") is not None]
                                        opp_creatures = [c.get("name") for c in battlefield
                                                        if c.get("owner_seat_id") != my_seat and c.get("power") is not None]

                                        # Determine if it's our turn
                                        is_my_turn = turn.get("active_player") == my_seat
                                        stack = curr_state.get("stack", [])

                                    # Smart Filtering: Reduce "weird times" advice
                                    if trig == "priority_gained":
                                        # 1. Opponent's Turn Silence
                                        if not is_my_turn:
                                            has_stack = len(stack) > 0
                                            is_combat = "Combat" in phase
                                            has_decision = curr_state.get("pending_decision")
                                            
                                            if not has_stack and not is_combat and not has_decision:
                                                logger.debug("Suppressing priority_gained (opponent's turn, no immediate threat)")
                                                should_advise = False

                                        # 2. My Turn Flow
                                        # Don't say "Priority gained" immediately after "New Turn" advice
                                        elif is_new_turn or (turn_num == last_advice_turn and phase in ("Beginning", "Untap", "Upkeep", "Draw")):
                                            logger.debug("Suppressing priority_gained (redundant with new_turn)")
                                            should_advise = False

                                    # CRITICAL: If filtering suppressed advice, stop now
                                    if not should_advise:
                                        continue

                                        # Build prompt based on trigger
                                        pending_decision = curr_state.get("pending_decision")

                                        if pending_decision:
                                             prompt = f"DECISION: Game is asking '{pending_decision}'.\nContext: {phase} phase.\nMy board: {', '.join(my_creatures) or 'none'}\nWhat do I choose?"
                                        elif trig == "new_turn":
                                            if is_my_turn:
                                                prompt = f"Turn {turn_num} - MY TURN ({phase}).\nMy hand: {hand_str}\nMy board: {', '.join(my_creatures) or 'empty'}\nTheir board: {', '.join(opp_creatures) or 'empty'}\nWhat should I play?"
                                            else:
                                                prompt = f"Turn {turn_num} - OPPONENT'S TURN ({phase}).\nMy instant options: {', '.join(instant_options)}\nAnything to watch for?"
                                        elif trig == "stack_spell" or (stack and trig == "priority_gained"):
                                            stack_cards = [c.get("name") for c in stack]
                                            prompt = f"RESPONSE: Stack has {', '.join(stack_cards)}.\nPhase: {phase}\nMy instants: {', '.join(instant_options) or 'none'}\nDo I resolve, respond, or pass?"
                                        elif trig == "low_life":
                                            my_life = local_player.get("life_total", 20)
                                            prompt = f"WARNING: I'm at {my_life} life!\nMy hand: {hand_str}\nHow do I survive?"
                                        elif trig == "opponent_low_life":
                                            opp = next((p for p in curr_state.get("players", []) if not p.get("is_local")), {})
                                            opp_life = opp.get("life_total", 20)
                                            prompt = f"Opponent at {opp_life} life!\nMy attackers: {', '.join(my_creatures) or 'none'}\nDo I have lethal?"
                                        elif trig == "combat_attackers":
                                            if is_my_turn:
                                                prompt = f"Combat - declaring attackers.\nMy creatures: {', '.join(my_creatures) or 'none'}\nTheir blockers: {', '.join(opp_creatures) or 'none'}\nWhat should attack?"
                                            else:
                                                prompt = f"Opponent entering combat.\nTheir creatures: {', '.join(opp_creatures) or 'none'}\nMy creatures: {', '.join(my_creatures) or 'none'}\nAny pre-combat responses?"
                                        elif trig == "combat_blockers":
                                            if is_my_turn:
                                                prompt = f"Combat - opponents blocking.\nMy attackers: {', '.join(my_creatures) or 'none'}\nTheir blockers: {', '.join(opp_creatures) or 'none'}\nAny combat tricks?"
                                            else:
                                                prompt = f"Combat - declaring blockers.\nMy creatures: {', '.join(my_creatures) or 'none'}\nAttackers: {', '.join(opp_creatures) or 'none'}\nHow should I block?"
                                        else:
                                            # Enriched fallback for priority_gained
                                            stack_ids = [c.get("name") for c in stack]
                                            stack_str = ", ".join(stack_ids) if stack_ids else "empty"
                                            prompt = f"Priority passed to me in {phase} phase ({step}).\nStack: {stack_str}\nMy hand: {hand_str}\nCorrect technical play?"

                                        logger.info(f"TRIGGER: {trig} -> interrupting and sending prompt")
                                        stop_playback() # Ensure we stop any outdated advice
                                        if self.backend_name == "gemini-live":
                                            prompt = f"[SYSTEM: INTERRUPT - NEW GAME STATE] {prompt}"

                                        self.ui.log(f"[yellow]>>> Sending {trig} prompt to {self.backend_name}...[/]")
                                        client.send_text(prompt)

                                        last_advice_turn = turn_num
                                        last_advice_phase = phase
                                        last_proactive_advice = time.time()
                                        
                                        # CRITICAL: Break loop to process only one trigger per cycle
                                        break

                            # Periodic proactive advice if no triggers fired
                            now = time.time()
                            if turn_num > 0 and not is_playing.is_set():
                                time_since_advice = now - last_proactive_advice
                                if time_since_advice >= PROACTIVE_INTERVAL:
                                    # Get current game context
                                    hand = curr_state.get("hand", [])
                                    hand_cards = [f"{c.get('name')}" for c in hand]
                                    hand_str = ", ".join(hand_cards) if hand_cards else "empty"

                                    local_player = next((p for p in curr_state.get("players", []) if p.get("is_local")), {})
                                    my_life = local_player.get("life_total", 20)
                                    opp_player = next((p for p in curr_state.get("players", []) if not p.get("is_local")), {})
                                    opp_life = opp_player.get("life_total", 20)

                                    prompt = f"[Periodic check] Turn {turn_num}, {phase}. My life: {my_life}, Opponent: {opp_life}. My hand: {hand_str}. What should I be thinking about?"

                                    logger.info(f"PROACTIVE: Periodic advice prompt")
                                    self.ui.log(f"[cyan]>>> Proactive advice...[/]")
                                    client.send_text(prompt)
                                    last_proactive_advice = now

                            # Update context if state changed significantly
                            state_changed = False
                            if prev_state:
                                old_hand = len(prev_state.get("hand", []))
                                new_hand = len(curr_state.get("hand", []))
                                old_turn = prev_state.get("turn", {}).get("turn_number", 0)
                                new_turn = curr_state.get("turn", {}).get("turn_number", 0)
                                old_phase = prev_state.get("turn", {}).get("phase", "")
                                new_phase = curr_state.get("turn", {}).get("phase", "")
                                state_changed = (old_hand != new_hand or old_turn != new_turn or old_phase != new_phase)

                            prev_state = curr_state

                            # Refresh context on state change or every 5 seconds
                            now = time.time()
                            if state_changed or now - last_context_update > 5.0:
                                client.update_instructions(get_instructions())
                                last_context_update = now
                                if state_changed:
                                    logger.info("State changed - updated GPT context")

                    except Exception as e:
                        logger.error(f"Realtime loop error: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        self.ui.log(f"[red]Loop error: {e}[/]")
                        self._record_error(str(e), "realtime_loop")

                    time.sleep(0.5)  # Check for triggers every 500ms

        except Exception as e:
            logger.error(f"Realtime audio error: {e}")
            self._record_error(str(e), "realtime_audio")
        finally:
            client.disconnect()
            try:
                output_stream.stop()
                output_stream.close()
            except:
                pass
            self._realtime_client = None
            self.ui.log("\n[REALTIME] Disconnected\n")
            logger.info("GPT-Realtime voice loop stopped")

    def _on_mute_hotkey(self) -> None:
        """F5 - Toggle TTS mute."""
        if self._voice_output:
            muted = self._voice_output.toggle_mute()
            self.ui.status("VOICE", f"{'MUTED' if muted else 'UNMUTED'} (saved)")
        else:
            self.ui.status("VOICE", "TTS not enabled")

    def _on_voice_hotkey(self) -> None:
        """F6 - Change TTS voice."""
        if self._voice_output:
            voice_id, desc = self._voice_output.next_voice()
            self.ui.status("VOICE", f"Changed to: {desc} (saved)")
            try:
                self._voice_output.speak("Voice changed.", blocking=False)
            except Exception:
                pass
        else:
            self.ui.status("VOICE", "TTS not enabled")


    def save_bug_report(self, reason: str = "User Request") -> Optional["Path"]:
        """Save comprehensive bug report and return path."""
        bug_dir = LOG_DIR / "bug_reports"
        bug_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bug_file = bug_dir / f"bug_{timestamp}.json"

        try:
            # Collect comprehensive debug info
            report = self._collect_debug_info()
            report["reason"] = reason

            with open(bug_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

            # Copy path to clipboard
            clipboard_success = copy_to_clipboard(str(bug_file))

            # Make path clickable
            file_url = f"file:///{str(bug_file).replace(chr(92), '/')}"
            clickable = f"\x1b]8;;{file_url}\x1b\\{bug_file}\x1b]8;;\x1b\\"

            if clipboard_success:
                self.ui.log(f"\n[BUG] Saved: {clickable}")
                self.ui.log("[BUG] Path copied to clipboard!\n")
            else:
                self.ui.log(f"\n[BUG] Saved: {clickable}\n")
                
            return bug_file
        except Exception as e:
            self.ui.error(f"\n[BUG] Failed: {e}\n")
            logger.exception("Bug report failed")
            return None

    def _on_bug_report_hotkey(self) -> None:
        """F7 - Save comprehensive bug report for debugging."""
        self.save_bug_report("Hotkey F7")

    def take_screenshot_analysis(self) -> None:
        """Capture screen and request visual analysis (e.g. Mulligan)."""
        if "gemini" not in self.backend_name.lower():
             self.ui.log("[red]Visual analysis requires Gemini backend.[/]")
             return
             
        try:
            from PIL import ImageGrab
            import io
            
            # Capture primary screen
            img = ImageGrab.grab()
            
            # Resize if huge to save bandwidth/latency
            img.thumbnail((1920, 1080)) 
            
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            png_bytes = buf.getvalue()

            self.ui.log("[yellow]Analyzing screenshot...[/]")
            
            # Context
            try:
                game_state = self.get_game_state()
            except:
                game_state = {}
                
            system_prompt = "You are an expert Magic: The Gathering coach. Analyze this screenshot. If this is a Mulligan decision, start with 'KEEP' or 'MULLIGAN' immediately. Give a confidence score (0-10). Do NOT list the cards in the hand. Focus on the decision: curve, land count, and color fixing. If mid-game, analyze valid attacks or blocks explicitly."
            
            # We can optionally format game state into the prompt if available
            ctx = ""
            if game_state:
                 ctx = f" Turn {game_state.get('turn',{}).get('turn_number','?')}."
            
            user_msg = f"Analyze this screen state.{ctx} What is the best move?"
            
            if hasattr(self._coach, 'complete_with_image'):
                advice = self._coach.complete_with_image(system_prompt, user_msg, png_bytes)
                self.ui.advice(advice, "Visual Analysis")
                if self._auto_speak:
                    self.speak_advice(advice, blocking=False)
            else:
                self.ui.log("[red]Current backend does not support image analysis.[/]")

        except ImportError:
            self.ui.log("[red]Missing 'Pillow' library. Install with: pip install Pillow[/]")
        except Exception as e:
            self.ui.log(f"[red]Screenshot error: {e}[/]")
            import traceback
            print(f"Screenshot Error: {e}")

    def _collect_debug_info(self) -> dict:
        """Collect comprehensive debug information for bug reports."""
        import platform

        report = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",  # TODO: pull from package

            # System info
            "system": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "machine": platform.machine(),
            },

            # Coach configuration
            "config": {
                "backend": self.backend_name,
                "model": self.model_name,
                "voice_mode": self.voice_mode,
                "advice_style": self.advice_style,
                "advice_frequency": self.advice_frequency,
                "draft_mode": self.draft_mode,
                "set_code": self.set_code,
                "auto_speak": True if self.backend_name == "gemini-live" else self._auto_speak,
            },

            # Settings from disk
            "settings": dict(self.settings._data) if self.settings else {},

            # Current game state
            "game_state": self._mcp.get_game_state() if self._mcp else {},

            # Draft state if active
            "draft_state": self._mcp.get_draft_pack() if self._mcp else {},

            # Voice state
            "voice": {
                "tts_enabled": self._voice_output is not None,
                "tts_muted": self._voice_output._muted if self._voice_output else None,
                "tts_voice": self._voice_output.current_voice if self._voice_output else None,
                "stt_enabled": self._voice_input is not None,
                "realtime_connected": self._realtime_client is not None,
            },

            # Recent advice history
            "advice_history": list(self._advice_history) if hasattr(self, '_advice_history') else [],

            # LLM context (what the coach sees)
            "llm_context": self._get_llm_context(),

            # Recent log entries (last 100 lines)
            "recent_logs": self._get_recent_logs(100),

            # Error state
            "errors": list(self._recent_errors) if hasattr(self, '_recent_errors') else [],
        }

        return report

    def _get_llm_context(self) -> dict:
        """Get the current LLM context/prompt for debugging."""
        context = {
            "system_prompt": None,
            "formatted_game_state": None,
        }

        try:
            if self._coach:
                context["system_prompt"] = getattr(self._coach, '_system_prompt', None)
                if self._mcp:
                    game_state = self._mcp.get_game_state()
                    if hasattr(self._coach, '_format_game_context'):
                        context["formatted_game_state"] = self._coach._format_game_context(game_state)
        except Exception as e:
            context["error"] = str(e)

        return context

    def _get_recent_logs(self, num_lines: int = 100) -> list:
        """Get recent log entries from standalone.log."""
        try:
            if LOG_FILE.exists():
                with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                    # Read last N lines efficiently
                    lines = f.readlines()
                    return lines[-num_lines:]
        except Exception as e:
            return [f"Error reading logs: {e}"]
        return []

    def _record_advice(self, advice: str, trigger: str, game_context: str = None) -> None:
        """Record advice for debug history."""
        if not hasattr(self, '_advice_history'):
            self._advice_history = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger,
            "advice": advice,
            "game_context_snippet": game_context[:500] if game_context else None,
        }
        self._advice_history.append(entry)

        # Keep only last 20 entries
        if len(self._advice_history) > 20:
            self._advice_history = self._advice_history[-20:]

    def _record_error(self, error: str, context: str = None) -> None:
        """Record error for debug history."""
        if not hasattr(self, '_recent_errors'):
            self._recent_errors = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context,
        }
        self._recent_errors.append(entry)

        # Keep only last 10 errors
        if len(self._recent_errors) > 10:
            self._recent_errors = self._recent_errors[-10:]

    def _on_swap_seat_hotkey(self) -> None:
        """F8 - Swap local seat (fix wrong player detection)."""
        if not self._mcp:
            return

        try:
            from arenamcp.server import game_state
            # Get current state
            players = list(game_state.players.keys())
            current = game_state.local_seat_id

            if len(players) >= 2:
                # Swap to the other seat
                new_seat = [s for s in players if s != current][0] if current else players[0]
                # Use source=3 (User) to lock it
                game_state.set_local_seat_id(new_seat, source=3)
                self.ui.status("SEAT", f"Swapped to Seat {new_seat} (LOCKED - won't auto-change)")
                logger.info(f"Manual seat swap: {current} -> {new_seat} (locked by User)")
            else:
                self.ui.status("SEAT", f"Only {len(players)} player(s) detected, cannot swap")
        except Exception as e:
            self.ui.error(f"Seat swap failed: {e}")
            logger.error(f"Seat swap error: {e}")

    def _on_restart_hotkey(self) -> None:
        """F9 - Restart the coach."""
        self.ui.status("RESTART", "Restarting coach...")
        logger.info("F9 restart requested")
        self._restart_requested = True
        self._running = False

    def set_backend(self, provider: str, model: Optional[str] = None) -> None:
        """Explicitly set the backend provider."""
        if self.draft_mode:
            self.ui.status("PROVIDER", "Not available in draft mode")
            return

        old_backend = self.backend_name
        switching_realtime = (old_backend == "gpt-realtime") != (provider == "gpt-realtime")

        try:
            from arenamcp.coach import CoachEngine, create_backend

            # If switching to/from gpt-realtime, need to restart voice threads
            if switching_realtime and self._running:
                logger.info(f"Switching realtime mode: {old_backend} -> {provider}")
                self.ui.log(f"\n[yellow]Restarting voice system for {provider}...[/]\n")

                # Stop current threads
                self._running = False

                if self._voice_thread and self._voice_thread.is_alive():
                    self._voice_thread.join(timeout=3.0)
                if self._coaching_thread and self._coaching_thread.is_alive():
                    self._coaching_thread.join(timeout=3.0)

                # Cleanup old resources
                if self._voice_input:
                    try:
                        self._voice_input.stop()
                    except Exception:
                        pass
                    self._voice_input = None

                if self._realtime_client:
                    try:
                        self._realtime_client.disconnect()
                    except Exception:
                        pass
                    self._realtime_client = None

                self._running = True

            # Update backend
            self.backend_name = provider
            self.model_name = model
            llm_backend = create_backend(provider, model=model)
            self._coach = CoachEngine(backend=llm_backend)

            actual_model = getattr(llm_backend, 'model', 'default')

            # Reinitialize voice system if we switched realtime modes
            if switching_realtime and self._running:
                self._init_voice()

                # Start appropriate threads
                # Start appropriate threads
                if provider in ("gpt-realtime", "gemini-live"):
                    # Realtime: single thread handles proactive + voice
                    self._voice_thread = threading.Thread(
                        target=self._realtime_voice_loop, daemon=True, name="voice-realtime"
                    )
                    self._voice_thread.start()
                    self.ui.log(f"[green]{provider} mode active - speak naturally![/]\n")
                else:
                    # Other backends: separate coaching and voice threads
                    self._coaching_thread = threading.Thread(
                        target=self._coaching_loop, daemon=True, name="coaching"
                    )
                    self._coaching_thread.start()

                    self._voice_thread = threading.Thread(
                        target=self._voice_loop, daemon=True, name="voice"
                    )
                    self._voice_thread.start()
                    self.ui.log("[green]PTT mode active - press F4 to speak[/]\n")
            else:
                # Just reconfigure voice input if needed
                if self._voice_input:
                    enable_transcription = (provider != "gemini")
                    self._voice_input.transcription_enabled = enable_transcription
                    logger.info(f"Voice transcription enabled: {enable_transcription}")

            self.ui.status("PROVIDER", f"Switched to {provider.upper()} ({actual_model})")
            logger.info(f"Switched to {provider} backend, model: {actual_model}")
        except Exception as e:
            self.ui.error(f"Failed to set provider {provider}: {e}")
            logger.error(f"Set provider error: {e}")
            import traceback
            logger.debug(traceback.format_exc())


    def _on_model_cycle_hotkey(self) -> None:
        """F12 - Cycle through all available provider/model combinations."""
        if self.draft_mode:
            return

        # Unified logic: Cycle through the same list as the TUI dropdown
        # Format: (provider, model_name)
        combo_list = [
            ("gemini", "gemini-3-flash-preview"),
            ("gemini", "gemini-3-pro-preview"),
            ("ollama", "llama3.2:latest"),
            ("ollama", "gemma3n:latest"),
            ("ollama", "deepseek-r1:14b"),
            ("claude", "claude-sonnet-4-20250514"),
            ("claude", "claude-haiku-4-5-20251001"),
            ("gpt-realtime", "gpt-realtime"),
        ]

        # Find current index
        current_idx = -1
        for i, (p, m) in enumerate(combo_list):
            if p == self.backend_name and m == self.model_name:
                current_idx = i
                break
        
        # Next index
        next_idx = (current_idx + 1) % len(combo_list)
        new_provider, new_model = combo_list[next_idx]
        
        # Switch
        self.set_backend(new_provider, new_model)
        
        # Update TUI manually if needed (though set_backend logs it)
        # self.ui.status("MODEL", f"{new_provider}/{new_model}")
        try:
            # If current model is None or not in list, start at -1 so next is 0
            current_idx = current_list.index(self.model_name) if self.model_name in current_list else -1
        except ValueError:
            current_idx = -1
        
        next_idx = (current_idx + 1) % len(current_list)
        new_model = current_list[next_idx]
        
        self.model_name = new_model
        
        # Recreate coach
        self._reinit_coach()
        self.ui.status("MODEL", new_model)
        self.ui.log(f"\n[MODEL] Switched to {new_model}\n")


    def _on_style_toggle_hotkey(self) -> None:
        """F2 - Toggle advice style (Concise/Verbose)."""
        self.advice_style = "verbose" if self.advice_style == "concise" else "concise"
        self.ui.status("STYLE", self.advice_style.upper())
        self.ui.log(f"\n[STYLE] Changed to {self.advice_style.upper()}\n")

    def _on_frequency_toggle_hotkey(self) -> None:
        """F3 - Toggle advice frequency."""
        self.advice_frequency = "every_priority" if self.advice_frequency == "start_of_turn" else "start_of_turn"
        label = "EVERY PRIORITY" if self.advice_frequency == "every_priority" else "START OF TURN"
        self.ui.status("FREQ", label)
        self.ui.log(f"\n[FREQ] Changed to {label}\n")

    def _on_voice_cycle_hotkey(self) -> None:
        """F6 - Cycle TTS voice."""
        if self._voice_output:
            try:
                voice_id, desc = self._voice_output.next_voice(step=2)
                self.ui.status("VOICE_ID", desc)
                self.ui.log(f"\n[VOICE] Changed to: {desc}\n")
                self.ui.speak("Voice changed.")
            except Exception as e:
                self.ui.log(f"Error changing voice: {e}")

    def _reinit_coach(self):
        """Reinitialize the coach backend with current settings."""
        try:
            from arenamcp.coach import CoachEngine, create_backend
            llm_backend = create_backend(self.backend_name, model=self.model_name)
            self._coach = CoachEngine(backend=llm_backend)
            
            # Get actual model name for display if it was auto-selected
            actual_model = getattr(llm_backend, 'model', self.model_name or 'default')
            self.model_name = actual_model # Sync back
            
            # Configure voice input based on backend
            if self._voice_input:
                enable_transcription = (self.backend_name != "gemini")
                self._voice_input.transcription_enabled = enable_transcription
                logger.info(f"Voice transcription enabled: {enable_transcription} (Backend: {self.backend_name})")
            
            logger.info(f"Re-initialized {self.backend_name} backend, model: {actual_model}")
        except Exception as e:
            self.ui.log(f"\nbackend init failed: {e}\n")
            logger.error(f"Backend init error: {e}")

    def _register_hotkeys(self) -> None:
        """Register hotkeys."""
        if not self._register_keyboard:
            logger.info("Skipping global keyboard hotkey registration (TUI/Active Mode)")
            return

        if not keyboard:
            return
        try:
            keyboard.on_press_key("f2", lambda _: self._on_style_toggle_hotkey(), suppress=False)
            keyboard.on_press_key("f3", lambda _: self._on_frequency_toggle_hotkey(), suppress=False)
            keyboard.on_press_key("f5", lambda _: self._on_mute_hotkey(), suppress=False)
            keyboard.on_press_key("f6", lambda _: self._on_voice_cycle_hotkey(), suppress=False)
            keyboard.on_press_key("f7", lambda _: self._on_bug_report_hotkey(), suppress=False)
            keyboard.on_press_key("f8", lambda _: self._on_swap_seat_hotkey(), suppress=False)
            keyboard.on_press_key("f9", lambda _: self._on_restart_hotkey(), suppress=False)
            keyboard.on_press_key("f10", lambda _: self.run_speed_test(), suppress=False)
            keyboard.on_press_key("f12", lambda _: self._on_model_cycle_hotkey(), suppress=False)
            logger.info("Hotkeys registered")
        except Exception as e:
            logger.warning(f"Hotkey registration failed: {e}")

    def _unregister_hotkeys(self) -> None:
        """Unregister hotkeys."""
        if keyboard:
            try:
                keyboard.unhook_all()
            except (ValueError, KeyError, Exception):
                pass  # Already unhooked or error

    def start(self) -> None:
        """Start the standalone coach."""
        logger.info(f"start() called: backend_name={self.backend_name}, model={self.model_name}, draft={self.draft_mode}")
        if self._running:
            logger.info("Already running, returning early")
            return

        # Check API key early for non-draft mode
        if not self.draft_mode:
            if self.backend_name == "gemini":
                if not os.environ.get("GOOGLE_API_KEY"):
                    self.ui.error("GOOGLE_API_KEY not set")
                    self.ui.log("Set it in .env file or: set GOOGLE_API_KEY=your_key")
                    sys.exit(1)
            elif self.backend_name == "claude":
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    self.ui.error("ANTHROPIC_API_KEY not set")
                    self.ui.log("Set it in .env file or: set ANTHROPIC_API_KEY=your_key")
                    sys.exit(1)
            elif self.backend_name == "gemini-live":
                if not os.environ.get("GOOGLE_API_KEY"):
                    self.ui.error("GOOGLE_API_KEY not set")
                    self.ui.log("Set it in .env file or: set GOOGLE_API_KEY=your_key")
                    sys.exit(1)
            elif self.backend_name == "gpt-realtime":
                if not os.environ.get("AZURE_REALTIME_API_KEY"):
                    self.ui.error("AZURE_REALTIME_API_KEY not set")
                    self.ui.log("Set it in .env file or: set AZURE_REALTIME_API_KEY=your_key")
                    sys.exit(1)

        self._running = True

        # Initialize components
        self._init_mcp()
        self._init_voice()

        # Track actual model name for display
        actual_model = self.model_name

        if self.draft_mode:
            # Use MCP's built-in draft helper
            logger.info("Starting MCP draft helper...")
            result = self._mcp.start_draft_helper(self.set_code)
            logger.info(f"Draft helper: {result}")
        else:
            # Initialize LLM for coaching
            self._init_llm()
            # Get actual model name from backend
            if self._coach and hasattr(self._coach, '_backend'):
                actual_model = getattr(self._coach._backend, 'model', self.model_name)

            # Start threads based on backend
            logger.info(f"Starting threads for backend: {self.backend_name}")
            if self.backend_name in ("gpt-realtime", "gemini-live"):
                # Realtime streaming backends: single thread handles both proactive advice and voice I/O
                logger.info(f"Starting {self.backend_name} voice loop (no separate coaching thread)")
                self._voice_thread = threading.Thread(
                    target=self._realtime_voice_loop, daemon=True, name="voice-realtime"
                )
                self._voice_thread.start()
                logger.info(f"{self.backend_name} thread started")
                # No separate coaching thread - realtime loop handles triggers
            else:
                # Other backends: separate coaching and voice threads
                logger.info("Starting PTT voice loop + coaching loop")
                self._coaching_thread = threading.Thread(
                    target=self._coaching_loop, daemon=True, name="coaching"
                )
                self._coaching_thread.start()

                self._voice_thread = threading.Thread(
                    target=self._voice_loop, daemon=True, name="voice"
                )
                self._voice_thread.start()

        self._register_hotkeys()

        # Print status
        self.ui.log("\n" + "="*50)
        if self.draft_mode:
            self.ui.log("MTGA DRAFT HELPER")
            self.ui.log("="*50)
            self.ui.log(f"Set: {self.set_code or 'auto-detect'}")
            self.ui.log("Using MCP server's draft evaluation")
        else:
            self.ui.log("MTGA STANDALONE COACH")
            self.ui.log("="*50)
            self.ui.status("BACKEND", f"{self.backend_name} ({actual_model or 'default'})")
            if self.backend_name in ("gpt-realtime", "gemini-live"):
                self.ui.status("VOICE", "REALTIME (continuous)")
            else:
                self.ui.status("VOICE", f"PTT (F4) + Kokoro")
        self.ui.log("-"*50)
        self.ui.log("F5=mute F6=voice F7=bug F8=seat F9=restart F10=speed F12=model")
        self.ui.log("="*50)
        self.ui.log("\nWaiting for MTGA...")
        self.ui.log("F8=swap seat if wrong | F9=restart coach\n")

    def stop(self) -> None:
        """Stop the coach and clean up all resources.

        This method ensures proper termination of all threads and resources:
        1. Signals threads to stop via _running flag
        2. Stops voice input/output
        3. Disconnects realtime client
        4. Stops MCP server watcher
        5. Waits for threads to terminate
        """
        if not self._running:
            return

        logger.info("Stopping coach - beginning cleanup...")
        self._running = False

        # 1. Unregister hotkeys first to prevent new events
        self._unregister_hotkeys()

        # 2. Stop voice input immediately (releases PTT hotkey, stops VOX stream)
        if self._voice_input:
            try:
                logger.debug("Stopping voice input...")
                self._voice_input.stop()
            except Exception as e:
                logger.debug(f"Voice input stop error (non-fatal): {e}")
            self._voice_input = None

        # 3. Stop voice output (TTS) - interrupts any playing audio
        if self._voice_output:
            try:
                logger.debug("Stopping voice output...")
                self._voice_output.stop()
            except Exception as e:
                logger.debug(f"Voice output stop error (non-fatal): {e}")
            self._voice_output = None

        # 4. Disconnect GPT-realtime client (WebSocket cleanup)
        if self._realtime_client:
            try:
                logger.debug("Disconnecting realtime client...")
                self._realtime_client.disconnect()
            except Exception as e:
                logger.debug(f"Realtime disconnect error (non-fatal): {e}")
            self._realtime_client = None

        # 5. Stop draft helper if active
        if self.draft_mode and self._mcp:
            try:
                logger.debug("Stopping draft helper...")
                self._mcp.stop_draft_helper()
            except Exception as e:
                logger.debug(f"Draft helper stop error (non-fatal): {e}")

        # 6. Stop MCP server's log watcher
        if self._mcp:
            try:
                logger.debug("Stopping MCP watcher...")
                from arenamcp.server import stop_watching
                stop_watching()
            except Exception as e:
                logger.debug(f"Watcher stop error (non-fatal): {e}")

        # 7. Wait for daemon threads to finish (with timeout)
        # These should exit quickly since _running is False
        if self._coaching_thread and self._coaching_thread.is_alive():
            logger.debug("Waiting for coaching thread...")
            self._coaching_thread.join(timeout=2.0)
            if self._coaching_thread.is_alive():
                logger.warning("Coaching thread did not terminate cleanly")
        self._coaching_thread = None

        if self._voice_thread and self._voice_thread.is_alive():
            logger.debug("Waiting for voice thread...")
            self._voice_thread.join(timeout=2.0)
            if self._voice_thread.is_alive():
                logger.warning("Voice thread did not terminate cleanly")
        self._voice_thread = None

        # 8. Clear references to allow garbage collection
        self._mcp = None
        self._coach = None
        self._trigger = None

        logger.info("Coach stopped - cleanup complete")
        self.ui.log(f"\nStopped. Log: {LOG_FILE}")

    def run_speed_test(self):
        """Run latency test against all providers."""
        if not self.ui:
            return

        self.ui.log("\n[bold yellow]Running API Speed Test (3 passes)...[/]")
        
        # Define test cases: (Provider, Model Class, Model Name)
        from arenamcp.coach import GeminiBackend, ClaudeBackend, OllamaBackend
        
        tests = [
            ("Gemini Flash", GeminiBackend, "gemini-3-flash-preview"),
            ("Gemini Pro 3.0", GeminiBackend, "gemini-3-pro-preview"),
            ("Claude Haiku", ClaudeBackend, "claude-haiku-4-5-20251001"),
            ("Ollama Local", OllamaBackend, "llama3.2:latest"),
        ]
        
        import time

        for name, backend_cls, model_id in tests:
            try:
                self.ui.log(f"Testing {name}...")
                latencies = []
                
                # Init backend once
                backend = backend_cls(model=model_id)

                # Warmup / 3 passes
                for i in range(3):
                    start_req = time.perf_counter()
                    response = backend.complete("You are a helpful assistant.", "Say 'ok' and nothing else.")
                    req_ms = (time.perf_counter() - start_req) * 1000
                    
                    if response.startswith("Error"):
                        raise Exception(response)
                        
                    latencies.append(req_ms)
                    # Small delay between requests
                    time.sleep(0.1)

                avg_ms = sum(latencies) / len(latencies)
                min_ms = min(latencies)
                max_ms = max(latencies)
                
                self.ui.log(f"[green]PASS {name}: Avg {avg_ms:.0f}ms (Range: {min_ms:.0f}-{max_ms:.0f}ms)[/]")
                    
            except Exception as e:
                self.ui.log(f"[red]FAIL {name}: {e}[/]")

        self.ui.log("[bold yellow]Speed Test Complete.[/]\n")

    def run_forever(self) -> None:
        """Run until interrupted."""
        self.start()

        def signal_handler(sig, frame):
            print("\n\nShutting down...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while self._running:
            time.sleep(1)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="MTGA Standalone Coach - MCP client with voice I/O",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m arenamcp.standalone --backend gemini
  python -m arenamcp.standalone --backend ollama --model llama3.2
  python -m arenamcp.standalone --draft --set MH3

Environment variables:
  GOOGLE_API_KEY     Required for Gemini
  ANTHROPIC_API_KEY  Required for Claude
        """
    )

    parser.add_argument("--backend", "-b", choices=["claude", "gemini", "ollama", "gpt-realtime", "gemini-live"],
                        default=None, help="LLM backend (default: gemini)")
    parser.add_argument("--model", "-m", help="Model name override")
    parser.add_argument("--voice", "-v", choices=["ptt", "vox"], default=None,
                        help="Voice input: ptt (F4) or vox (auto)")
    parser.add_argument("--draft", action="store_true",
                        help="Draft helper mode (no LLM needed)")
    parser.add_argument("--set", "-s", dest="set_code",
                        help="Set code for draft (e.g., MH3, BLB)")
    parser.add_argument("--show-log", action="store_true",
                        help="Show log file and exit")
    parser.add_argument("--cli", action="store_true",
                        help="Run in legacy CLI mode (default is TUI)")

    args = parser.parse_args()

    # Launch TUI unless CLI mode requested or show-log
    if not args.cli and not args.show_log:
        try:
            from arenamcp.tui import run_tui
            run_tui(args)
            return
        except ImportError as e:
            print(f"Failed to load TUI (install 'textual'): {e}")
            print("Falling back to CLI mode...")


    if args.show_log:
        print(f"Log: {LOG_FILE}")
        if LOG_FILE.exists():
            with open(LOG_FILE) as f:
                for line in f.readlines()[-30:]:
                    print(line, end='')
        return

    # Check API keys
    if not args.draft:
        if args.backend == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
            print("Error: GOOGLE_API_KEY not set")
            sys.exit(1)
        if args.backend == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: ANTHROPIC_API_KEY not set")
            sys.exit(1)
        if args.backend == "gpt-realtime" and not os.environ.get("AZURE_REALTIME_API_KEY"):
            print("Error: AZURE_REALTIME_API_KEY not set")
            sys.exit(1)

    logger.info(f"Starting: backend={args.backend}, draft={args.draft}")

    while True:
        coach = StandaloneCoach(
            backend=args.backend,
            model=args.model,
            voice_mode=args.voice,
            draft_mode=args.draft,
            set_code=args.set_code,
        )

        try:
            coach.run_forever()
        except KeyboardInterrupt:
            coach.stop()
            break  # Exit on Ctrl+C
        except Exception as e:
            logger.error(f"Fatal: {e}")
            logger.debug(traceback.format_exc())
            print(f"\nError: {e}\nSee: {LOG_FILE}")
            sys.exit(1)

        # Check if restart was requested (F9)
        if coach._restart_requested:
            print("\n" + "="*50)
            print("RESTARTING...")
            print("="*50 + "\n")
            logger.info("Restarting coach...")
            continue
        else:
            break  # Normal exit


if __name__ == "__main__":
    main()
