"""Standalone MTGA coach - runs without MCP client.

This module provides a standalone coaching experience that runs independently
of Claude Code or any MCP client. It monitors MTGA games, provides proactive
advice on game events, and accepts voice questions via PTT (F4).

Usage:
    python -m arenamcp.standalone --backend gemini --auto-speak
    python -m arenamcp.standalone --backend claude --model claude-sonnet-4-20250514
    python -m arenamcp.standalone --backend ollama --model gemma3:12b

Logs are written to ~/.arenamcp/debug.log for bug reports.
"""

# Load .env file before anything else
def _load_dotenv():
    """Load environment variables from .env file if it exists."""
    import os
    from pathlib import Path

    # Look for .env in current dir and parent dirs
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
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Configure logging before imports that use it
LOG_DIR = Path.home() / ".arenamcp"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "debug.log"

# Set up file handler with detailed format
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Configure root logger directly (basicConfig is a no-op if already configured)
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
# Remove any existing handlers to avoid duplicates
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
root_logger.addHandler(file_handler)

# Also add a simple console handler for INFO level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Now import our modules
import keyboard

from arenamcp.coach import CoachEngine, GameStateTrigger, create_backend
from arenamcp.gamestate import GameState, create_game_state_handler
from arenamcp.parser import LogParser
from arenamcp.tts import VoiceOutput
from arenamcp.voice import VoiceInput
from arenamcp.watcher import MTGALogWatcher

# Try to import Scryfall for card names
try:
    from arenamcp.scryfall import ScryfallCache
    _scryfall = ScryfallCache()
except Exception:
    _scryfall = None


class StandaloneCoach:
    """Standalone MTGA coaching application.

    Combines log watching, game state tracking, LLM coaching, and voice I/O
    into a single self-contained application.
    """

    def __init__(
        self,
        backend: str = "gemini",
        model: Optional[str] = None,
        auto_speak: bool = True,
        voice_mode: str = "ptt",
    ):
        """Initialize standalone coach.

        Args:
            backend: LLM backend ("claude", "gemini", "ollama")
            model: Optional model override
            auto_speak: Whether to speak advice automatically
            voice_mode: Voice input mode ("ptt" or "vox")
        """
        self.backend_name = backend
        self.model_name = model
        self.auto_speak = auto_speak
        self.voice_mode = voice_mode

        # State
        self._running = False
        self._game_state = GameState()
        self._parser = LogParser()
        self._watcher: Optional[MTGALogWatcher] = None

        # Threads
        self._coaching_thread: Optional[threading.Thread] = None
        self._voice_thread: Optional[threading.Thread] = None

        # Components (lazy init)
        self._coach: Optional[CoachEngine] = None
        self._trigger: Optional[GameStateTrigger] = None
        self._voice_input: Optional[VoiceInput] = None
        self._voice_output: Optional[VoiceOutput] = None

        # Register game state handler
        handler = create_game_state_handler(self._game_state)
        self._parser.register_handler('GreToClientEvent', handler)

        logger.info("="*60)
        logger.info("STANDALONE COACH INITIALIZED")
        logger.info(f"Backend: {backend}" + (f" (model: {model})" if model else ""))
        logger.info(f"Auto-speak: {auto_speak}")
        logger.info(f"Voice mode: {voice_mode}")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info("="*60)

    def _get_game_state_dict(self) -> dict[str, Any]:
        """Convert GameState to dict format matching MCP tool output."""
        # Ensure local player is detected
        self._game_state.ensure_local_seat_id()

        # Build turn info
        turn = {
            "turn_number": self._game_state.turn_info.turn_number,
            "active_player": self._game_state.turn_info.active_player,
            "priority_player": self._game_state.turn_info.priority_player,
            "phase": self._game_state.turn_info.phase,
            "step": self._game_state.turn_info.step,
        }

        # Build players list with land info
        players = []
        for p in self._game_state.players.values():
            is_local = p.seat_id == self._game_state.local_seat_id

            # Count lands controlled by this player
            lands = []
            for obj in self._game_state.battlefield:
                controller = obj.controller_seat_id or obj.owner_seat_id
                if controller == p.seat_id and "CardType_Land" in obj.card_types:
                    land_name = None
                    if _scryfall and obj.grp_id:
                        try:
                            card = _scryfall.get_card_by_arena_id(obj.grp_id)
                            if card:
                                land_name = card.name
                        except Exception:
                            pass
                    if not land_name:
                        # Use subtypes for unknown lands (e.g., "Forest", "Island")
                        if hasattr(obj, 'subtypes') and obj.subtypes:
                            land_name = " ".join(obj.subtypes)
                        else:
                            land_name = f"Unknown Land (ID: {obj.grp_id})"
                    lands.append({"name": land_name, "tapped": obj.is_tapped})

            players.append({
                "seat_id": p.seat_id,
                "life_total": p.life_total,
                "mana_pool": p.mana_pool,  # Floating mana only
                "lands": lands,  # Lands this player controls
                "is_local": is_local,
            })

        def serialize_card(obj):
            """Serialize a game object with card name lookup."""
            card_data = {
                "instance_id": obj.instance_id,
                "grp_id": obj.grp_id,
                "owner_seat_id": obj.owner_seat_id,
                "controller_seat_id": obj.controller_seat_id,
                "power": obj.power,
                "toughness": obj.toughness,
                "is_tapped": obj.is_tapped,
                "card_types": obj.card_types,
            }
            # Try to get card name from Scryfall
            card_found = False
            if _scryfall and obj.grp_id:
                try:
                    card = _scryfall.get_card_by_arena_id(obj.grp_id)
                    if card:
                        card_data["name"] = card.name
                        card_data["mana_cost"] = card.mana_cost
                        card_data["oracle_text"] = card.oracle_text
                        card_found = True
                except Exception:
                    pass

            # Generate descriptive name for unknown cards using subtypes from logs
            if not card_found:
                type_names = [t.replace("CardType_", "") for t in obj.card_types] if obj.card_types else []
                # Use subtypes if available (e.g., "Badger Mole" instead of "Creature")
                subtype_str = " ".join(obj.subtypes) if hasattr(obj, 'subtypes') and obj.subtypes else ""

                if obj.power is not None and obj.toughness is not None:
                    # It's a creature - show P/T and subtypes
                    if subtype_str:
                        card_data["name"] = f"Unknown {obj.power}/{obj.toughness} {subtype_str}"
                    else:
                        type_str = "/".join(type_names) if type_names else "Creature"
                        card_data["name"] = f"Unknown {obj.power}/{obj.toughness} {type_str}"
                elif subtype_str:
                    card_data["name"] = f"Unknown {subtype_str}"
                elif type_names:
                    card_data["name"] = f"Unknown {'/'.join(type_names)}"
                else:
                    card_data["name"] = f"Unknown Card (ID: {obj.grp_id})"
                card_data["mana_cost"] = ""
                card_data["oracle_text"] = ""
                # Add subtypes to card_data for LLM context
                if hasattr(obj, 'subtypes') and obj.subtypes:
                    card_data["subtypes"] = obj.subtypes

            return card_data

        return {
            "turn": turn,
            "players": players,
            "battlefield": [serialize_card(o) for o in self._game_state.battlefield],
            "hand": [serialize_card(o) for o in self._game_state.hand],
            "graveyard": [serialize_card(o) for o in self._game_state.graveyard],
            "stack": [serialize_card(o) for o in self._game_state.stack],
        }

    def _init_components(self) -> None:
        """Initialize coach and voice components."""
        # Create LLM backend
        logger.info(f"Creating {self.backend_name} backend...")
        llm_backend = create_backend(self.backend_name, model=self.model_name)
        self._coach = CoachEngine(backend=llm_backend)
        self._trigger = GameStateTrigger()

        # Create voice components
        if self.auto_speak:
            logger.info("Initializing TTS (first call loads model)...")
            self._voice_output = VoiceOutput()

        logger.info(f"Initializing voice input ({self.voice_mode} mode)...")
        self._voice_input = VoiceInput(mode=self.voice_mode)

    def _coaching_loop(self) -> None:
        """Background loop that monitors game and provides proactive advice."""
        logger.info("Coaching loop started")
        prev_state: dict[str, Any] = {}

        # Debouncing state - prevent repeated advice
        last_advice_turn: int = 0
        last_advice_phase: str = ""
        last_advice_trigger: str = ""

        # High-priority triggers that always warrant advice
        HIGH_PRIORITY_TRIGGERS = {"stack_spell", "low_life", "opponent_low_life", "combat_blockers"}

        while self._running:
            try:
                curr_state = self._get_game_state_dict()
                curr_turn = curr_state.get("turn", {})
                turn_num = curr_turn.get("turn_number", 0)
                phase = curr_turn.get("phase", "")

                # Log state periodically for debugging
                if turn_num > 0:
                    logger.debug(f"Game state: turn={turn_num}, "
                                f"phase={phase}, "
                                f"players={len(curr_state['players'])}, "
                                f"battlefield={len(curr_state['battlefield'])}")

                if prev_state and self._trigger:
                    triggers = self._trigger.check_triggers(prev_state, curr_state)

                    for trigger in triggers:
                        logger.debug(f"TRIGGER detected: {trigger}")

                        # Debouncing logic - skip if we already advised this turn/phase
                        should_advise = False

                        if trigger in HIGH_PRIORITY_TRIGGERS:
                            # Always advise on high-priority events
                            should_advise = True
                            logger.debug(f"High priority trigger: {trigger}")
                        elif trigger == "new_turn" and turn_num > last_advice_turn:
                            # New turn - advise once per turn
                            should_advise = True
                            logger.debug(f"New turn {turn_num} > last {last_advice_turn}")
                        elif trigger == "combat_attackers" and last_advice_trigger != "combat_attackers":
                            # Combat attackers - advise once per combat
                            should_advise = True
                        elif trigger == "priority_gained":
                            # Only advise on priority if it's a new turn or new phase
                            if turn_num > last_advice_turn or phase != last_advice_phase:
                                should_advise = True
                                logger.debug(f"Priority in new context: turn {turn_num}, phase {phase}")
                            else:
                                logger.debug(f"Skipping priority_gained (same turn/phase)")

                        if not should_advise:
                            logger.debug(f"Skipping {trigger} (debounced)")
                            continue

                        logger.info(f"TRIGGER: {trigger}")
                        self._log_game_state(curr_state, trigger)

                        # Get advice
                        if self._coach:
                            advice = self._coach.get_advice(curr_state, trigger=trigger)
                            logger.info(f"ADVICE: {advice}")

                            # Update debounce state
                            last_advice_turn = turn_num
                            last_advice_phase = phase
                            last_advice_trigger = trigger

                            # Speak if enabled
                            if self.auto_speak and self._voice_output:
                                print(f"\n[COACH] [{trigger}] {advice}\n")
                                try:
                                    self._voice_output.speak(advice, blocking=True)
                                except Exception as e:
                                    logger.error(f"TTS error: {e}")
                            else:
                                print(f"\n[COACH] [{trigger}] {advice}\n")

                prev_state = curr_state

            except Exception as e:
                logger.error(f"Error in coaching loop: {e}")
                logger.debug(traceback.format_exc())

            time.sleep(1.5)

        logger.info("Coaching loop stopped")

    def _voice_loop(self) -> None:
        """Background loop for handling voice questions."""
        if not self._voice_input:
            return

        logger.info(f"Voice loop started ({self.voice_mode} mode)")
        if self.voice_mode == "ptt":
            print("\n[MIC] Press F4 to ask a question (tap for quick advice)\n")
        else:
            print("\n[MIC] Voice activation enabled - just speak\n")

        self._voice_input.start()

        while self._running:
            try:
                # Wait for voice input
                text = self._voice_input.wait_for_speech(timeout=2.0)

                # Check if event was triggered (result_ready was set)
                if not self._voice_input._result_ready.is_set():
                    continue  # Timeout, no input

                if self._coach:
                    game_state = self._get_game_state_dict()

                    if text and text.strip():
                        # User asked a specific question
                        logger.info(f"VOICE INPUT: {text}")
                        print(f"\n[YOU] {text}")

                        self._log_game_state(game_state, f"question: {text}")
                        advice = self._coach.get_advice(game_state, question=text)
                        logger.info(f"RESPONSE: {advice}")
                        print(f"\n[ANSWER] {advice}\n")
                    else:
                        # Empty speech (F4 tap) - give quick advice on current state
                        logger.info("QUICK ADVICE REQUEST (F4 tap)")
                        print("\n[QUICK] Analyzing current situation...")

                        self._log_game_state(game_state, "quick_advice")
                        advice = self._coach.get_advice(game_state, trigger="user_request")
                        logger.info(f"QUICK ADVICE: {advice}")
                        print(f"\n[COACH] {advice}\n")

                    if self.auto_speak and self._voice_output:
                        try:
                            self._voice_output.speak(advice, blocking=True)
                        except Exception as e:
                            logger.error(f"TTS error: {e}")

            except Exception as e:
                if self._running:  # Only log if not shutting down
                    logger.error(f"Error in voice loop: {e}")
                    logger.debug(traceback.format_exc())

        self._voice_input.stop()
        logger.info("Voice loop stopped")

    def _log_game_state(self, state: dict[str, Any], context: str) -> None:
        """Log full game state for debugging."""
        logger.debug(f"=== GAME STATE ({context}) ===")
        logger.debug(json.dumps(state, indent=2, default=str))
        logger.debug("=== END GAME STATE ===")

    def _on_mute_hotkey(self) -> None:
        """Handle F5 mute toggle hotkey."""
        if self._voice_output:
            muted = self._voice_output.toggle_mute()
            status = "MUTED" if muted else "UNMUTED"
            logger.info(f"Voice output {status}")
            print(f"\n[VOICE] Coach {status}\n")
        else:
            print("\n[VOICE] TTS not enabled\n")

    def _on_voice_hotkey(self) -> None:
        """Handle F6 voice change hotkey."""
        if self._voice_output:
            voice_id, description = self._voice_output.next_voice()
            logger.info(f"Changed voice to: {voice_id} ({description})")
            print(f"\n[VOICE] Changed to: {description}\n")
            # Speak a sample with new voice
            try:
                self._voice_output.speak("Voice changed.", blocking=False)
            except Exception as e:
                logger.debug(f"Could not play voice sample: {e}")
        else:
            print("\n[VOICE] TTS not enabled\n")

    def _on_bug_report_hotkey(self) -> None:
        """Handle F7 bug report hotkey."""
        # Save current game state snapshot for later analysis
        bug_dir = LOG_DIR / "bug_reports"
        bug_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bug_file = bug_dir / f"bug_{timestamp}.json"

        try:
            game_state = self._get_game_state_dict()
            report = {
                "timestamp": datetime.now().isoformat(),
                "backend": self.backend_name,
                "model": self.model_name,
                "game_state": game_state,
            }

            with open(bug_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Bug report saved: {bug_file}")
            print(f"\n[BUG] Report saved: {bug_file}\n")
        except Exception as e:
            logger.error(f"Failed to save bug report: {e}")
            print(f"\n[BUG] Failed to save report: {e}\n")

    def _register_hotkeys(self) -> None:
        """Register global hotkeys for coach control."""
        try:
            keyboard.on_press_key("f5", lambda _: self._on_mute_hotkey(), suppress=False)
            keyboard.on_press_key("f6", lambda _: self._on_voice_hotkey(), suppress=False)
            keyboard.on_press_key("f7", lambda _: self._on_bug_report_hotkey(), suppress=False)
            logger.info("Hotkeys registered: F5=mute, F6=voice, F7=bug report")
        except Exception as e:
            logger.warning(f"Failed to register hotkeys: {e}")

    def _unregister_hotkeys(self) -> None:
        """Unregister global hotkeys."""
        try:
            keyboard.unhook_all()
            logger.info("Hotkeys unregistered")
        except Exception:
            pass

    def _on_log_chunk(self, chunk: str) -> None:
        """Handle new log chunks from MTGA."""
        try:
            self._parser.process_chunk(chunk)
        except Exception as e:
            logger.debug(f"Parse error: {e}")

    def start(self) -> None:
        """Start the standalone coach."""
        if self._running:
            return

        logger.info("Starting standalone coach...")
        self._running = True

        # Initialize components
        self._init_components()

        # Start log watcher
        logger.info("Starting MTGA log watcher...")
        self._watcher = MTGALogWatcher(callback=self._on_log_chunk)
        self._watcher.start()

        # Start coaching thread
        self._coaching_thread = threading.Thread(
            target=self._coaching_loop,
            daemon=True,
            name="coaching-loop"
        )
        self._coaching_thread.start()

        # Start voice thread
        self._voice_thread = threading.Thread(
            target=self._voice_loop,
            daemon=True,
            name="voice-loop"
        )
        self._voice_thread.start()

        # Register hotkeys
        self._register_hotkeys()

        print("\n" + "="*50)
        print("MTGA Standalone Coach Running")
        print("="*50)
        print(f"Backend: {self.backend_name}" + (f" ({self.model_name})" if self.model_name else ""))
        print(f"Auto-speak: {self.auto_speak}")
        print(f"Voice: {self.voice_mode.upper()} mode" + (" - Hold F4 to talk, tap for quick advice" if self.voice_mode == "ptt" else ""))
        print(f"Debug log: {LOG_FILE}")
        print("-"*50)
        print("Hotkeys: F4=talk/advice  F5=mute  F6=voice  F7=bug")
        print("="*50)
        print("\nWaiting for MTGA game... (Ctrl+C to quit)\n")

        logger.info("Standalone coach started successfully")

    def stop(self) -> None:
        """Stop the standalone coach."""
        if not self._running:
            return

        logger.info("Stopping standalone coach...")
        self._running = False

        # Unregister hotkeys
        self._unregister_hotkeys()

        # Stop watcher
        if self._watcher:
            self._watcher.stop()

        # Wait for threads
        if self._coaching_thread:
            self._coaching_thread.join(timeout=2.0)
        if self._voice_thread:
            self._voice_thread.join(timeout=2.0)

        # Stop voice output
        if self._voice_output:
            self._voice_output.stop()

        logger.info("Standalone coach stopped")
        print("\nCoach stopped. Debug log saved to:", LOG_FILE)

    def run_forever(self) -> None:
        """Run until interrupted."""
        self.start()

        # Set up signal handlers
        def signal_handler(sig, frame):
            print("\n\nShutting down...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Keep main thread alive
        while self._running:
            time.sleep(1)


def main():
    """Entry point for standalone coach."""
    parser = argparse.ArgumentParser(
        description="MTGA Standalone Coach - Real-time game coaching without MCP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m arenamcp.standalone --backend gemini
  python -m arenamcp.standalone --backend claude --no-auto-speak
  python -m arenamcp.standalone --backend ollama --model gemma3:12b
  python -m arenamcp.standalone --backend gemini --voice vox

Environment variables:
  GOOGLE_API_KEY     Required for Gemini backend
  ANTHROPIC_API_KEY  Required for Claude backend

Debug logs are written to ~/.arenamcp/debug.log
        """
    )

    parser.add_argument(
        "--backend", "-b",
        choices=["claude", "gemini", "ollama"],
        default="gemini",
        help="LLM backend to use (default: gemini)"
    )

    parser.add_argument(
        "--model", "-m",
        help="Model name override (uses backend default if not specified)"
    )

    parser.add_argument(
        "--auto-speak",
        dest="auto_speak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically speak advice via TTS (default: enabled)"
    )

    parser.add_argument(
        "--voice", "-v",
        choices=["ptt", "vox"],
        default="ptt",
        help="Voice input mode: ptt (push-to-talk F4) or vox (voice activation)"
    )

    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Show the debug log file path and exit"
    )

    args = parser.parse_args()

    if args.show_log:
        print(f"Debug log: {LOG_FILE}")
        if LOG_FILE.exists():
            print(f"Size: {LOG_FILE.stat().st_size:,} bytes")
            print(f"\nLast 20 lines:")
            with open(LOG_FILE) as f:
                lines = f.readlines()
                for line in lines[-20:]:
                    print(line, end='')
        return

    # Log startup
    logger.info("="*60)
    logger.info(f"STANDALONE COACH STARTING")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Args: backend={args.backend}, model={args.model}, "
                f"auto_speak={args.auto_speak}, voice={args.voice}")
    logger.info("="*60)

    # Check API keys
    if args.backend == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        msg = "GOOGLE_API_KEY environment variable not set. Get a key at: https://makersuite.google.com/app/apikey"
        logger.error(msg)
        print(f"Error: {msg}")
        sys.exit(1)

    if args.backend == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        msg = "ANTHROPIC_API_KEY environment variable not set. Get a key at: https://console.anthropic.com/"
        logger.error(msg)
        print(f"Error: {msg}")
        sys.exit(1)

    # Run coach
    coach = StandaloneCoach(
        backend=args.backend,
        model=args.model,
        auto_speak=args.auto_speak,
        voice_mode=args.voice,
    )

    try:
        coach.run_forever()
    except KeyboardInterrupt:
        coach.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.debug(traceback.format_exc())
        print(f"\nFatal error: {e}")
        print(f"See debug log for details: {LOG_FILE}")
        sys.exit(1)


if __name__ == "__main__":
    main()
