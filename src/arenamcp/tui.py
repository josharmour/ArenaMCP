
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Header, Footer, Log, RichLog, Input, Button, Label, Select, Switch, Checkbox, Static, Markdown
from textual.binding import Binding
from textual.message import Message

import threading
import time
import os
import sys
import logging

# Import core logic
from arenamcp.standalone import StandaloneCoach, UIAdapter, LOG_FILE, LOG_DIR, copy_to_clipboard
from arenamcp.tts import VoiceOutput
from arenamcp.match_validator import MatchRecording, MatchValidator, start_recording, stop_recording, get_current_recording


class TextualLogHandler(logging.Handler):
    """Custom logging handler that writes to a Textual Log widget."""
    
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        
    def emit(self, record):
        try:
            msg = self.format(record)
            # We must schedule the write on the main thread
            # .write() on RichLog is thread-safe in Textual (mostly), but better explicit
            if hasattr(self.widget, 'app') and self.widget.app:
                self.widget.app.call_from_thread(self.widget.write, msg)
        except Exception:
            self.handleError(record)

class TUIAdapter(UIAdapter):
    """Adapter to route coach output to Textual widgets."""
    def __init__(self, app: "ArenaApp"):
        self.app = app

    def _safe_call(self, method, *args, **kwargs):
        """Invoke method on main thread, safely handling if we are already there."""
        # Textual apps primarily run on the main thread (usually)
        # We can check if we are in the same thread as the app loop
        try:
             # accessing private _thread_id is risky but standard in Textual hacking
             # better to just try/except or check threading
             if threading.get_ident() == self.app._thread_id:
                 method(*args, **kwargs)
             else:
                 self.app.call_from_thread(method, *args, **kwargs)
        except Exception:
             # Fallback if _thread_id missing or other error
             self.app.call_from_thread(method, *args, **kwargs)

    def log(self, message: str) -> None:
        self._safe_call(self.app.write_log, message)

    def advice(self, text: str, seat_info: str) -> None:
        self._safe_call(self.app.write_advice, text, seat_info)

    def status(self, key: str, value: str) -> None:
        self._safe_call(self.app.update_status, key, value)

    def error(self, message: str) -> None:
        self._safe_call(self.app.write_log, f"[bold red]ERROR: {message}[/]")

    def speak(self, text: str) -> None:
        """Speak text using the coach's voice output."""
        if self.app.coach and self.app.coach._voice_output:
            try:
                # Use call_from_thread to ensure thread safety if this triggers UI updates
                # but speak() is blocking by default in some contexts, so be careful.
                # Actually, StandaloneCoach calls this. We should just pass it through.
                # However, standalone.py calls self.ui.speak("Voice changed.")
                # We want to use the actual engine.
                self.app.coach._voice_output.speak(text, blocking=False)
            except Exception as e:
                self.error(f"TTS Error: {e}")


class GameStateDisplay(Static):
    """Live game state visualization widget."""
    
    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._last_state = None
    
    def update_state(self, game_state_snapshot: dict):
        """Update the display with current game state."""
        self._last_state = game_state_snapshot
        self.refresh_display()
    
    def refresh_display(self):
        """Render the current game state to markup."""
        if not self._last_state:
            self.update("[dim]Waiting for game state...[/]")
            return
        
        try:
            lines = []
            turn_info = self._last_state.get("turn_info", {})
            players = self._last_state.get("players", [])
            zones = self._last_state.get("zones", {})
            battlefield = zones.get("battlefield", [])
            hand = zones.get("my_hand", [])
            pending_decision = self._last_state.get("pending_decision")
            
            # Turn and Phase
            turn_num = turn_info.get("turn_number", "?")
            phase = turn_info.get("phase", "").replace("Phase_", "")
            step = turn_info.get("step", "").replace("Step_", "")
            active_player = turn_info.get("active_player", 0)
            priority_player = turn_info.get("priority_player", 0)
            
            # Identify local player
            local_player = next((p for p in players if p.get("is_local")), None)
            local_seat = local_player.get("seat_id") if local_player else None
            
            # FIX: During DeclareBlock, defending player (non-active) has priority
            # Arena logs may report active player has priority, but that's incorrect
            if step == "DeclareBlock" and active_player != local_seat:
                priority_player = local_seat  # Defending player chooses blockers
            
            # Header line
            active_label = f"[green]YOUR[/]" if active_player == local_seat else f"[yellow]OPP[/]"
            priority_label = f"[green]YOU[/]" if priority_player == local_seat else f"[yellow]OPP[/]"
            lines.append(f"[bold]T{turn_num}[/] {phase}{f':{step}' if step else ''} | Active:{active_label} Priority:{priority_label}")
            
            # Life totals
            if players:
                life_parts = []
                for p in players:
                    seat = p.get("seat_id")
                    life = p.get("life_total", 20)
                    is_you = p.get("is_local", False)
                    label = f"[bold green]YOU[/]" if is_you else f"[yellow]OPP[/]"
                    life_parts.append(f"{label}:{life}")
                lines.append(" | ".join(life_parts))
            
            # Pending decision
            if pending_decision:
                lines.append(f"[bold red]⚠ {pending_decision}[/]")
            
            # Battlefield (compact view) - count all permanents, not just creatures
            your_permanents = [c for c in battlefield if c.get("controller_seat_id") == local_seat]
            opp_permanents = [c for c in battlefield if c.get("controller_seat_id") != local_seat]
            your_creatures = [c for c in your_permanents if c.get("power") is not None]
            opp_creatures = [c for c in opp_permanents if c.get("power") is not None]
            
            if your_permanents or opp_permanents:
                lines.append(f"[bold]Board:[/] You:{len(your_permanents)} ({len(your_creatures)}⚔) Opp:{len(opp_permanents)} ({len(opp_creatures)}⚔)")
                
                # List your permanents
                if your_permanents:
                    your_names = [c.get("name", "?") for c in your_permanents]
                    lines.append(f"  [green]You:[/] {', '.join(your_names[:6])}" + ("..." if len(your_names) > 6 else ""))
                
                # List opponent permanents
                if opp_permanents:
                    opp_names = [c.get("name", "?") for c in opp_permanents]
                    lines.append(f"  [yellow]Opp:[/] {', '.join(opp_names[:6])}" + ("..." if len(opp_names) > 6 else ""))
                
                # Show attacking creatures prominently
                attackers = [c for c in battlefield if c.get("is_attacking")]
                blockers = [c for c in battlefield if c.get("is_blocking")]
                if attackers:
                    atk_names = [c.get("name", "?") for c in attackers]
                    lines.append(f"[bold red]⚔ Attacking:[/] {', '.join(atk_names[:3])}" + ("..." if len(atk_names) > 3 else ""))
                if blockers:
                    blk_names = [c.get("name", "?") for c in blockers]
                    lines.append(f"[bold blue]🛡 Blocking:[/] {', '.join(blk_names[:3])}" + ("..." if len(blk_names) > 3 else ""))
            else:
                lines.append("[dim]Board: Empty[/]")
            
            # Hand - show more cards (up to 7 visible)
            if hand:
                hand_names = [c.get("name", "?") for c in hand]
                lines.append(f"[bold]Hand ({len(hand)}):[/] {', '.join(hand_names[:7])}" + ("..." if len(hand) > 7 else ""))
            else:
                lines.append("[dim]Hand: Empty[/]")
            
            # Graveyard, Exile, Library counts
            graveyard = zones.get("graveyard", [])
            exile = zones.get("exile", [])
            your_gy = len([c for c in graveyard if c.get("owner_seat_id") == local_seat])
            opp_gy = len([c for c in graveyard if c.get("owner_seat_id") != local_seat])
            your_exile = len([c for c in exile if c.get("owner_seat_id") == local_seat])
            opp_exile = len([c for c in exile if c.get("owner_seat_id") != local_seat])
            library_count = zones.get("library_count", "?")
            
            lines.append(f"[dim]GY: You={your_gy} Opp={opp_gy} | Exile: You={your_exile} Opp={opp_exile} | Lib: {library_count}[/]")
            
            self.update("\n".join(lines))
        except Exception as e:
            self.update(f"[red]State Error: {e}[/]")


class Sidebar(Vertical):
    """Sidebar for settings and actions."""

    # Model options for different modes
    # NOTE: Azure requires you to CREATE deployments in Azure Portal first!
    # The deployment NAME you create is what you use here, not the base model name.
    STANDARD_OPTIONS = [
        ("Gemini 2.5 Flash", "gemini/gemini-2.5-flash"),
        ("Azure GPT-5.2", "azure/gpt-5.2"),
        ("Llama 3.2", "ollama/llama3.2:latest"),
        ("Gemma 3N", "ollama/gemma3n:latest"),
        ("DeepSeek R1", "ollama/deepseek-r1:14b"),
        ("GLM-4 Flash", "ollama/glm-4.7-flash:latest"),
    ]

    REALTIME_OPTIONS = [
        ("Gemini Live", "gemini-live/gemini-2.0-flash-exp-image-generation"),
        ("GPT-Realtime", "gpt-realtime/gpt-realtime"),
    ]

    def compose(self) -> ComposeResult:
        with Vertical(id="status-panel"):
            yield Static("Seat: Searching...", id="status-seat", classes="status-line")
            yield Static("Model: Default", id="status-model", classes="status-line")
            yield Static("Style: CONCISE", id="status-style", classes="status-line")
            yield Static("Voice: Initializing...", id="status-voice", classes="status-line")

        with Vertical(id="controls-panel"):
            yield Label("Standard", id="lbl-local", classes="toggle-label toggle-label-active")
            yield Switch(value=False, id="switch-realtime")
            yield Label("Realtime", id="lbl-realtime", classes="toggle-label")
            yield Select(self.STANDARD_OPTIONS, id="select-provider", allow_blank=False)
            yield Select([(desc, vid) for vid, desc in VoiceOutput.VOICES], id="select-voice", prompt="Voice")

        with Vertical(id="actions-panel"):
            yield Button("Mute (F5)", id="btn-mute", variant="default")
            yield Button("Speed Test", id="btn-speed", variant="default")
            yield Button("Debug Report", id="btn-debug-report", variant="default")
            yield Button("Screenshot (F3)", id="btn-screenshot", variant="primary")
            yield Button("Restart (F9)", id="btn-restart", variant="error")
            # Match validation buttons
            yield Button("🔴 Record Match", id="btn-record-start", variant="warning")
            yield Button("⏹ Stop & Analyze", id="btn-record-stop", variant="default")


class ArenaApp(App):
    """MTGA Coach TUI Application."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    Sidebar {
        width: 32;
        dock: left;
        height: 100%;
        background: $surface-darken-1;
        border-right: solid $primary;
        padding: 0 1;
    }

    #status-panel {
        height: auto;
        border-bottom: solid $primary;
        padding: 0;
    }

    .status-line {
        height: 1;
        color: $text-muted;
    }

    #controls-panel {
        height: auto;
        background: $surface-darken-2;
        border: solid $primary;
        padding: 0 1;
        margin-top: 1;
    }

    .mode-row {
        height: 1;
        align-vertical: middle;
        layout: horizontal;
    }

    .toggle-label {
        width: 1fr;
        content-align: center middle;
        color: $text-muted;
    }

    .toggle-label-active {
        color: $success;
        text-style: bold;
    }

    Select {
        height: 3;
        margin: 0;
    }

    #actions-panel {
        height: auto;
        margin-top: 1;
    }

    Button {
        width: 100%;
        margin: 0;
    }

    #main-area {
        width: 1fr;
        height: 100%;
        layout: vertical;
    }

    #game-state-display {
        height: 10;
        border: solid $success;
        background: $surface-darken-1;
        padding: 1;
        overflow-y: auto;
    }

    #log-view {
        height: 1fr;
        border: solid $accent;
        background: $surface;
        padding: 0 1;
    }

    Input {
        dock: bottom;
        height: 3;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("f2", "toggle_style", "Style"),
        ("f3", "screenshot", "Screenshot"),
        ("f5", "toggle_mute", "Mute"),
        ("f6", "cycle_voice", "Voice"),
        ("f7", "bug_report", "Bug"),
        ("f9", "restart", "Restart"),
        ("f10", "speed_test", "Speed"),
        ("f11", "toggle_realtime", "Mode"),
        ("f12", "cycle_model", "Model"),
    ]

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.coach = None
        self.log_widget = None
        self.game_state_widget = None
        self._restart_requested = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Sidebar()
        with Vertical(id="main-area"):
            # Game State Display (Top Right)
            yield GameStateDisplay(id="game-state-display")
            # Advice Log (Bottom Right)
            yield RichLog(id="log-view", markup=True, wrap=True)
            yield Input(placeholder="Ask the coach...", id="chat-input")
        yield Footer()

    def on_mount(self) -> None:
        """Start the coach thread when the app mounts."""
        self.log_widget = self.query_one("#log-view", RichLog)
        self.game_state_widget = self.query_one("#game-state-display", GameStateDisplay)
        
        self.write_log("[bold green]Initializing MTGA Coach...[/]")
        
        # Start initial coach logic in a thread
        threading.Thread(target=self.start_coach, daemon=True).start()
        
        # Start game state polling
        threading.Thread(target=self._poll_game_state, daemon=True).start()

    def _poll_game_state(self):
        """Poll game state and update display every 500ms."""
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        while True:
            if self.coach and self.coach._mcp:
                try:
                    from arenamcp.server import game_state
                    snapshot = game_state.get_snapshot()
                    self.call_from_thread(self.game_state_widget.update_state, snapshot)
                except Exception as e:
                    logger.error(f"Game state polling error: {e}", exc_info=True)
            time.sleep(0.5)

    def start_coach(self):
        """Initialize the StandaloneCoach with our TUI Adapter."""
        try:
            self.adapter = TUIAdapter(self)
            self.coach = StandaloneCoach(
                backend=self.args.backend,
                model=self.args.model,
                voice_mode=self.args.voice,
                draft_mode=self.args.draft,
                set_code=self.args.set_code,
                ui_adapter=self.adapter,
                register_hotkeys=False  # Textual handles keys
            )
            
            # Sync initial state to UI
            self.call_from_thread(self.sync_ui_state)
            
            # Run the main loop
            self.coach.start()
        except Exception as e:
            import traceback
            self.call_from_thread(self.write_log, f"[bold red]Fatal Error: {e}[/]")
            self.call_from_thread(self.write_log, f"[red]{traceback.format_exc()}[/]")

    def sync_ui_state(self):
        """Update TUI widgets to match Coach state."""
        if not self.coach:
            return

        # Build composite provider string
        current_backend = self.coach.backend_name
        current_model = self.coach.model_name

        # Determine if we're in realtime mode based on current backend
        is_realtime = current_backend in ("gpt-realtime", "gemini-live")

        # Update the realtime switch and mode labels
        try:
            realtime_switch = self.query_one("#switch-realtime", Switch)
            if realtime_switch.value != is_realtime:
                with self.prevent(Switch.Changed):
                    realtime_switch.value = is_realtime
            self.update_mode_labels(is_realtime)
        except Exception:
            pass

        # Update provider select options based on mode
        self._update_provider_options(is_realtime)

        # Try to match current state to dropdown options
        combo = f"{current_backend}/{current_model}" if current_model else current_backend

        try:
            select_widget = self.query_one("#select-provider", Select)
            found = False
            for opt in select_widget._options:
                if str(opt[1]) == combo:
                    select_widget.value = combo
                    found = True
                    break
            if not found:
                for opt in select_widget._options:
                    if str(opt[1]).startswith(current_backend):
                        select_widget.value = opt[1]
                        break
        except Exception:
            pass

        # Sync states
        self.update_status("MODEL", self.coach.model_name or "Default")
        self.update_status("STYLE", self.coach.advice_style.upper())

        # Sync voice output
            # Sync voice output
        if self.coach._voice_output:
            curr_id, desc = self.coach._voice_output.current_voice
            self.update_status("VOICE_ID", desc)
            try:
                v_select = self.query_one("#select-voice", Select)
                if v_select.value != curr_id:
                    v_select.value = curr_id
            except Exception as e:
                self.write_log(f"[yellow]Voice Sync Warning: {e}[/]")
            self.sub_title = desc

        # Start seat info polling
        threading.Thread(target=self._poll_seat_info, daemon=True).start()

    def _poll_seat_info(self):
        """Poll game state for seat info periodically."""
        import time
        while True:
            if self.coach and self.coach._mcp:
                try:
                    # Access internal game state directly for debug info
                    # Note: accessing private member _game_state via mcp server might be tricky
                    # But StandaloneCoach has ._mcp.mcp -> which is the FastMCP object
                    # We need the global game_state object from server.py
                    # Easier way: call a method on coach that gets it locally if possible,
                    # or just import it if running in same process (which we are)
                    from arenamcp.server import game_state
                    
                    seat_id = game_state.local_seat_id
                    source = game_state.get_seat_source_name()
                    
                    # Heuristic translation for 1v1:
                    # Seat 1 is usually "Bottom" (You), Seat 2 is "Top" (Opponent)
                    # This might vary, but it's better than "Seat 1"
                    seat_label = f"Seat {seat_id}"
                    if seat_id == 1:
                        seat_label = "Bottom (1)"
                    elif seat_id == 2:
                        seat_label = "Top (2)"
                    
                    val = f"{seat_label} [{source}]" if seat_id is not None else "Searching..."
                    self.call_from_thread(self.update_status, "SEAT_INFO", val)
                except Exception:
                    pass
            time.sleep(1.0)
            
    # --- UI Update Methods (Called via call_from_thread) ---

    def write_log(self, message: str, highlight: bool = False) -> None:
        """Write to the log widget."""
        if self.log_widget:
            self.log_widget.write(message)



    def write_advice(self, text: str, seat_info: str) -> None:
        """Write specialized advice block."""
        if self.log_widget:
            self.log_widget.write(f"\n[bold magenta]--- COACH ({seat_info}) ---[/]")
            self.log_widget.write(f"[bold white]{text}[/]")
            self.log_widget.write("[magenta]-----------------------[/]\n")

    def update_status(self, key: str, value: str) -> None:
        """Update status labels."""
        key = key.upper()
        if key == "STYLE":
            self.query_one("#status-style", Static).update(f"Style: {value}")
        elif key == "MODEL":
            self.query_one("#status-model", Static).update(f"Model: {value}")
        elif key == "VOICE_ID" or key == "VOICE":
            self.query_one("#status-voice", Static).update(f"Voice: {value}")
            self.sub_title = value
        elif key == "SEAT_INFO":
            self.query_one("#status-seat", Static).update(f"Seat: {value}")

    # --- Actions ---

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self.coach:
            return
            
        btn_id = event.button.id
        if btn_id == "btn-mute":
            self.action_toggle_mute()
        elif btn_id == "btn-speed":
            self.action_speed_test()
        elif btn_id == "btn-bug":
            self.action_bug_report()
        elif btn_id == "btn-debug-report":
            self.action_take_debug_report()
        elif btn_id == "btn-screenshot":
            self.action_screenshot()
        elif btn_id == "btn-restart":
            self.action_restart()
        elif btn_id == "btn-record-start":
            self.action_start_recording()
        elif btn_id == "btn-record-stop":
            self.action_stop_recording()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle chat input."""
        text = event.value.strip()
        if not text:
            return
            
        event.input.value = ""
        self.write_log(f"\n[bold cyan]YOU: {text}[/]")
        
        if self.coach and self.coach._mcp:
            # Run in thread to avoid blocking UI
            threading.Thread(
                target=self._process_chat, 
                args=(text,), 
                daemon=True
            ).start()
            
    def _process_chat(self, text: str):
        try:
            game_state = self.coach._mcp.get_game_state()
            advice = self.coach._coach.get_advice(game_state, question=text)
            self.write_advice(advice, "Chat Response")
        except Exception as e:
            self.write_log(f"[red]Chat error: {e}[/]")

    # --- Settings Changes ---
    
    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "select-provider" and self.coach:
            if event.value == Select.BLANK or not event.value:
                return

            selection = str(event.value)
            
            # Parse provider/model
            if "/" in selection:
                new_provider, new_model = selection.split("/", 1)
            else:
                new_provider = selection
                new_model = None

            # Check if actually changed
            if new_provider != self.coach.backend_name or new_model != self.coach.model_name:
                self.write_log(f"[yellow]Switching to {new_provider} ({new_model})...[/]")
                
                # Verify availability (threaded to not block UI)
                threading.Thread(
                    target=self._verify_and_switch,
                    args=(new_provider, new_model),
                    daemon=True
                ).start()
                
        elif event.select.id == "select-voice" and self.coach:
            if event.value == Select.BLANK or not event.value:
                return
                
            voice_id = str(event.value)
            if self.coach._voice_output:
                # Check if actually changed to avoid cycles
                curr, _ = self.coach._voice_output.current_voice
                if curr != voice_id:
                    self.coach._voice_output.set_voice(voice_id)
                    _, desc = self.coach._voice_output.current_voice
                    self.write_log(f"[yellow]Voice changed to: {desc}[/]")
                    # Update status line
                    self.update_status("VOICE_ID", desc)
                    # Test speak
                    threading.Thread(target=lambda: self.coach._voice_output.speak("Voice checked.", blocking=False), daemon=True).start()

    def _verify_and_switch(self, provider, model):
        """Verify model exists (if ollama) then switch."""
        if provider == "ollama":
            self.write_log(f"Verifying {model} in ollama...")
            try:
                import subprocess
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
                if model not in result.stdout:
                    self.write_log(f"[bold red]WARNING: Model '{model}' not found in Ollama![/]")
                    self.write_log(f"[red]Please run: ollama pull {model}[/]")
                else:
                    self.write_log(f"[green]Verified {model} exists.[/]")
            except Exception as e:
                self.write_log(f"[red]Failed to verify ollama models: {e}[/]")

        # Proceed with switch
        self.coach.set_backend(provider, model)


    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "switch-realtime":
            is_realtime = event.value
            self._update_realtime_mode(is_realtime)

    def _update_realtime_mode(self, is_realtime: bool) -> None:
        """Update mode labels and provider options when toggling Local/Realtime."""
        try:
            self.update_mode_labels(is_realtime)
            self._update_provider_options(is_realtime)

            # Auto-select first option and switch backend
            options = Sidebar.REALTIME_OPTIONS if is_realtime else Sidebar.STANDARD_OPTIONS
            if options:
                select = self.query_one("#select-provider", Select)
                select.value = options[0][1]

            mode_name = "Realtime" if is_realtime else "Local"
            self.write_log(f"[yellow]Switched to {mode_name} mode[/]")
        except Exception as e:
            self.write_log(f"[red]Mode switch error: {e}[/]")

    def update_mode_labels(self, is_realtime: bool) -> None:
        """Update the active class on Local/Realtime labels."""
        try:
            lbl_local = self.query_one("#lbl-local", Label)
            lbl_realtime = self.query_one("#lbl-realtime", Label)

            if is_realtime:
                lbl_realtime.add_class("toggle-label-active")
                lbl_local.remove_class("toggle-label-active")
            else:
                lbl_local.add_class("toggle-label-active")
                lbl_realtime.remove_class("toggle-label-active")
        except Exception:
            pass

    def _update_provider_options(self, is_realtime: bool) -> None:
        """Update provider select options based on current mode."""
        try:
            select = self.query_one("#select-provider", Select)
            options = Sidebar.REALTIME_OPTIONS if is_realtime else Sidebar.STANDARD_OPTIONS
            select.set_options(options)
        except Exception:
            pass


    # --- Hotkey Actions ---
    
    def action_screenshot(self) -> None:
        """Take screenshot for mulligan analysis."""
        if self.coach:
            threading.Thread(target=self.coach.take_screenshot_analysis, daemon=True).start()

    def action_bug_report(self) -> None:
        """Alias for F7 binding."""
        self.action_take_debug_report()

    def action_take_debug_report(self) -> None:
        """Generate bug report and copy path."""
        if self.coach:
            # We wrap the existing logic but ensure we get the path
            threading.Thread(target=self._do_debug_report_copy, daemon=True).start()

    def _do_debug_report_copy(self):
        """Generate bug report, take screenshots, and copy path to clipboard."""
        import datetime
        import os
        from pathlib import Path
        
        # Trigger report generation
        report_path = self.coach.save_bug_report(reason="Manual Debug Report")
        
        if report_path:
            # Copy report path to clipboard
            from arenamcp.clipboard_utils import copy_to_clipboard
            copy_to_clipboard(str(report_path))
            
            # Take screenshots
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            desktop = Path.home() / "Desktop"
            desktop.mkdir(exist_ok=True)
            
            tui_screenshot = desktop / f"tui_screenshot_{timestamp}.png"
            arena_screenshot = desktop / f"arena_screenshot_{timestamp}.png"
            
            screenshots_taken = []
            
            # Screenshot the TUI window
            try:
                import pygetwindow as gw
                from PIL import ImageGrab
                
                # Find TUI window (title contains "MTGA Coach")
                tui_windows = [w for w in gw.getWindowsWithTitle("MTGA Coach")]
                if tui_windows:
                    tui_win = tui_windows[0]
                    tui_win.activate()
                    import time
                    time.sleep(0.1)  # Let window come to front
                    
                    bbox = (tui_win.left, tui_win.top, tui_win.left + tui_win.width, tui_win.top + tui_win.height)
                    img = ImageGrab.grab(bbox)
                    img.save(tui_screenshot)
                    screenshots_taken.append(f"TUI: {tui_screenshot.name}")
            except Exception as e:
                self.call_from_thread(self.write_log, f"[yellow]TUI screenshot failed: {e}[/]")
            
            # Screenshot the Arena window
            try:
                import pygetwindow as gw
                from PIL import ImageGrab
                
                # Find Arena window (title contains "MTGA" or "Magic: The Gathering Arena")
                arena_windows = [w for w in gw.getAllWindows() if "MTGA" in w.title or "Magic" in w.title]
                if arena_windows:
                    arena_win = arena_windows[0]
                    arena_win.activate()
                    import time
                    time.sleep(0.1)
                    
                    bbox = (arena_win.left, arena_win.top, arena_win.left + arena_win.width, arena_win.top + arena_win.height)
                    img = ImageGrab.grab(bbox)
                    img.save(arena_screenshot)
                    screenshots_taken.append(f"Arena: {arena_screenshot.name}")
            except Exception as e:
                self.call_from_thread(self.write_log, f"[yellow]Arena screenshot failed: {e}[/]")
            
            # Generate validation report if recording is in progress
            validation_summary = None
            try:
                current = get_current_recording()
                if current and len(current.frames) > 0:
                    from arenamcp.match_validator import MatchValidator
                    validator = MatchValidator()
                    
                    # Build a recording object for validation
                    from arenamcp.match_validator import MatchRecording
                    recording = MatchRecording(
                        match_id=current.match_id,
                        start_time=current.start_time,
                        frames=current.frames
                    )
                    
                    results = validator.validate_recording(recording)
                    
                    # Summarize results
                    errors = len([r for r in results if r.severity == "error"])
                    warnings = len([r for r in results if r.severity == "warning"])
                    infos = len([r for r in results if r.severity == "info"])
                    
                    validation_summary = f"Validation: {errors} errors, {warnings} warnings, {infos} timing ({len(current.frames)} frames)"
                    
                    # Save a quick validation report alongside the bug report
                    val_report = validator.generate_report(recording)
                    val_path = report_path.parent / f"validation_{report_path.stem}.txt"
                    val_path.write_text(val_report)
                    validation_summary += f"\n[dim]Saved: {val_path.name}[/]"
            except Exception as e:
                self.call_from_thread(self.write_log, f"[yellow]Validation failed: {e}[/]")
            
            # Report success
            msg_parts = [f"[bold green]✓ Debug Report:[/] {report_path.name}", "[green]Path copied to clipboard[/]"]
            if screenshots_taken:
                msg_parts.append(f"[green]Screenshots: {', '.join(screenshots_taken)}[/]")
            if validation_summary:
                msg_parts.append(f"[cyan]{validation_summary}[/]")
            
            self.call_from_thread(self.write_log, "\n".join(msg_parts))
        else:
            self.call_from_thread(self.write_log, "[red]Failed to generate debug report[/]")

    def action_restart(self) -> None:
        """Handle restart request - cleanly stops coach and exits app for restart."""
        if self.coach:
            self.write_log("[yellow]Restarting coach...[/]")
            # Stop the coach cleanly in a thread to avoid blocking
            def do_restart():
                self.coach.stop()
                # Signal restart via exit with special result
                self._restart_requested = True
                self.call_from_thread(self.exit, "restart")
            threading.Thread(target=do_restart, daemon=True).start()

    def action_start_recording(self) -> None:
        """Start recording match for post-game validation."""
        from datetime import datetime
        
        current = get_current_recording()
        if current:
            self.write_log("[yellow]⚠ Recording already in progress. Stop it first.[/]")
            return
        
        # Generate a match ID based on timestamp if we don't have one from Arena
        match_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.coach and hasattr(self.coach, '_mcp') and self.coach._mcp:
            try:
                gs = self.coach._mcp.get_game_state()
                if gs and gs.get("match_id"):
                    match_id = gs["match_id"]
            except:
                pass
        
        recording = start_recording(match_id)
        
        # Update button appearance
        try:
            btn = self.query_one("#btn-record-start", Button)
            btn.label = "🔴 RECORDING..."
            btn.variant = "error"
        except:
            pass
        
        self.write_log(f"[bold red]🔴 MATCH RECORDING STARTED[/]")
        self.write_log(f"[dim]Match ID: {match_id}[/]")
        self.write_log("[dim]Click 'Stop & Analyze' when match ends.[/]")

    def action_stop_recording(self) -> None:
        """Stop recording and run validation analysis."""
        recording = get_current_recording()
        if not recording:
            self.write_log("[yellow]⚠ No recording in progress.[/]")
            return
        
        # Stop recording
        recording = stop_recording()
        
        # Reset button
        try:
            btn = self.query_one("#btn-record-start", Button)
            btn.label = "🔴 Record Match"
            btn.variant = "warning"
        except:
            pass
        
        self.write_log(f"[bold green]⏹ RECORDING STOPPED[/]")
        self.write_log(f"[dim]Recorded {len(recording.frames)} frames[/]")
        
        # Run validation in background thread
        threading.Thread(
            target=self._run_validation_analysis,
            args=(recording,),
            daemon=True
        ).start()

    def _run_validation_analysis(self, recording: MatchRecording) -> None:
        """Run validation analysis and output results."""
        from datetime import datetime
        
        self.call_from_thread(self.write_log, "[yellow]Analyzing match data...[/]")
        
        try:
            validator = MatchValidator()
            discrepancies = validator.validate_recording(recording)
            
            # Save recording to disk
            validation_dir = LOG_DIR / "match_validations"
            validation_dir.mkdir(exist_ok=True)
            
            recording_path = recording.save(validation_dir)
            
            # Generate report
            report = validator.generate_report(recording)
            
            # Save report to file
            report_path = validation_dir / f"report_{recording.match_id}.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Output summary to TUI
            self.call_from_thread(self.write_log, "")
            self.call_from_thread(self.write_log, "[bold cyan]═══ VALIDATION REPORT ═══[/]")
            self.call_from_thread(self.write_log, f"Frames analyzed: {len(recording.frames)}")
            
            if not discrepancies:
                self.call_from_thread(self.write_log, "[bold green]✅ No discrepancies found![/]")
                self.call_from_thread(self.write_log, "[green]Our parsing matches Arena's data.[/]")
            else:
                errors = [d for d in discrepancies if d.severity == "error"]
                warnings = [d for d in discrepancies if d.severity == "warning"]
                infos = [d for d in discrepancies if d.severity == "info"]
                
                problem_count = len(errors) + len(warnings)
                
                if problem_count == 0:
                    self.call_from_thread(self.write_log, "[bold green]✅ No significant issues![/]")
                    self.call_from_thread(self.write_log, f"[dim]  (Timing variations: {len(infos)})[/]")
                else:
                    self.call_from_thread(self.write_log, f"[yellow]Found {problem_count} issues:[/]")
                    if errors:
                        self.call_from_thread(self.write_log, f"  [red]Errors: {len(errors)}[/]")
                    if warnings:
                        self.call_from_thread(self.write_log, f"  [yellow]Warnings: {len(warnings)}[/]")
                    if infos:
                        self.call_from_thread(self.write_log, f"  [dim]Timing: {len(infos)}[/]")
                    
                    # Show sample discrepancies
                    if errors:
                        self.call_from_thread(self.write_log, "\n[red]Sample Errors:[/]")
                        for d in errors[:5]:
                            self.call_from_thread(
                                self.write_log,
                                f"  Turn {d.turn}: {d.field} - Arena={d.arena_value}, Parsed={d.parsed_value}"
                            )
            
            # Output file paths
            self.call_from_thread(self.write_log, "")
            self.call_from_thread(self.write_log, f"[bold]Recording saved:[/] {recording_path.name}")
            self.call_from_thread(self.write_log, f"[bold]Report saved:[/] {report_path.name}")
            
            # Copy recording path to clipboard
            copy_to_clipboard(str(recording_path))
            self.call_from_thread(self.write_log, "[green]Path copied to clipboard[/]")
            self.call_from_thread(self.write_log, "[bold cyan]═══════════════════════════[/]")
            
        except Exception as e:
            self.call_from_thread(self.write_log, f"[red]Validation error: {e}[/]")
            import traceback
            traceback.print_exc()


    def action_toggle_style(self) -> None:
        if self.coach:
            threading.Thread(target=self.coach._on_style_toggle_hotkey, daemon=True).start()

    def action_toggle_freq(self) -> None:
        if self.coach:
            threading.Thread(target=self.coach._on_frequency_toggle_hotkey, daemon=True).start()

    def action_toggle_mute(self) -> None:
        """Toggle TTS mute."""
        if self.coach:
            threading.Thread(target=self.coach._on_mute_hotkey, daemon=True).start()

    def action_speed_test(self) -> None:
        """Run API speed test."""
        if self.coach:
            threading.Thread(target=self.coach.run_speed_test, daemon=True).start()

        if self.coach:
            threading.Thread(target=self.coach.run_speed_test, daemon=True).start()

    def action_cycle_voice(self) -> None:
        if self.coach:
            threading.Thread(target=self.coach._on_voice_cycle_hotkey, daemon=True).start()

    def action_cycle_model(self) -> None:
        """Cycle through available models for the current mode."""
        if not self.coach:
            return
            
        current_backend = self.coach.backend_name
        current_model = self.coach.model_name
        is_realtime = current_backend in ("gpt-realtime", "gemini-live")
        
        options = Sidebar.REALTIME_OPTIONS if is_realtime else Sidebar.STANDARD_OPTIONS
        if not options:
            return
            
        # Find current index
        current_combo = f"{current_backend}/{current_model}" if current_model else current_backend
        
        try:
            # 1. Try exact match
            idx = -1
            for i, (_, val) in enumerate(options):
                if str(val) == current_combo:
                    idx = i
                    break
            
            # 2. If no exact match (custom model?), try prefix match strictly for fallback at index 0
            if idx == -1:
                for i, (_, val) in enumerate(options):
                    if str(val).startswith(current_backend):
                        idx = i
                        break
            
            # 3. Default to 0 if totally lost
            if idx == -1:
                idx = 0
            
            # Next index
            next_idx = (idx + 1) % len(options)
            _, next_val = options[next_idx]
            
            # Switch via the UI method to ensure everything updates cleanly
            select = self.query_one("#select-provider", Select)
            select.value = next_val
            
        except Exception as e:
            self.write_log(f"[red]Model cycle error: {e}[/]")

    def action_toggle_realtime(self) -> None:
        """Toggle between Local and Realtime mode (F11)."""
        try:
            switch = self.query_one("#switch-realtime", Switch)
            switch.value = not switch.value
            # The switch change handler will update the model list
        except Exception as e:
            self.write_log(f"[red]Toggle error: {e}[/]")

    def action_quit(self) -> None:
        if self.coach:
            self.coach.stop()
        self.exit()

def run_tui(args):
    """Run the TUI application with restart support.

    Args:
        args: Command line arguments from argparse.

    The TUI will restart if the user presses F9 (Restart).
    When running under the launcher, it exits with code 42 to signal restart.
    """
    import os

    # Check if running under the launcher
    under_launcher = os.environ.get("ARENAMCP_LAUNCHER") == "1"

    app = ArenaApp(args)
    result = app.run()

    # Check if restart was requested
    if result == "restart" or app._restart_requested:
        if under_launcher:
            # Signal the launcher to restart us
            sys.exit(42)
        else:
            # Not under launcher - just exit normally
            print("\nRestart requested. Run coach.bat to restart.")
            sys.exit(0)
    else:
        # Normal exit
        sys.exit(0)
