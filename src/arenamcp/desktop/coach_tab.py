from __future__ import annotations

import html
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .audio import AudioPlayback
from .coach_process import CoachProcess
from .tts_manager import TtsManager


class CoachTab(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._process: Optional[CoachProcess] = None
        self._tts = TtsManager(self)
        self._all_log_lines: list[str] = []
        self._last_game_state = "Turn 0 waiting for MTGA..."
        self._status_labels: dict[str, QLabel] = {}
        self._build_ui()
        self._tts.log_line.connect(lambda text: self.append_log(text, role="dim"))
        self._tts.status_line.connect(lambda text: self.append_log(text, role="status"))
        self._tts.error_line.connect(lambda text: self.append_log(f"TTS: {text}", role="error"))
        try:
            self._tts.start()
        except Exception as exc:
            self.append_log(f"TTS worker failed to start: {exc}", role="error")

    def attach_process(self, process: CoachProcess) -> None:
        if self._process is process:
            return

        self.detach_process()
        self._process = process
        process.event_received.connect(self._handle_event)
        process.stderr_line.connect(self._handle_stderr)
        process.exited.connect(self._handle_process_exit)
        self.append_log("Coach process started.", role="status")

    def detach_process(self) -> None:
        if self._process is None:
            return

        try:
            self._process.event_received.disconnect(self._handle_event)
            self._process.stderr_line.disconnect(self._handle_stderr)
            self._process.exited.disconnect(self._handle_process_exit)
        except (RuntimeError, TypeError):
            pass
        self._process = None

    def shutdown(self) -> None:
        self._tts.shutdown()

    def append_log(self, text: str, role: str = "default") -> None:
        self._all_log_lines.append(text)
        if len(self._all_log_lines) > 500:
            self._all_log_lines = self._all_log_lines[-500:]

        colors = {
            "advice": "#69d46c",
            "header": "#b48cff",
            "error": "#ff6666",
            "status": "#64c8dc",
            "dim": "#8a8a8a",
            "default": "#d7d7d7",
        }
        color = colors.get(role, colors["default"])
        escaped = html.escape(text).replace("\n", "<br>")
        self.log_view.append(f"<span style='color:{color}; font-family:Consolas;'>{escaped}</span>")
        scroll_bar = self.log_view.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        status_box = QGroupBox("Status")
        status_layout = QGridLayout(status_box)
        keys = [
            ("seat", "Seat"),
            ("backend", "Backend"),
            ("model", "Model"),
            ("bridge", "Bridge"),
            ("analysis", "Analysis"),
        ]
        for row, (key, label) in enumerate(keys):
            status_layout.addWidget(QLabel(f"{label}:"), row, 0)
            value = QLabel("-")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            status_layout.addWidget(value, row, 1)
            self._status_labels[key] = value
        root.addWidget(status_box)

        button_row = QHBoxLayout()
        commands = [
            ("Mode", "cycle_mode"),
            ("Model", "cycle_model"),
            ("Style", "toggle_style"),
            ("Voice", "cycle_voice"),
            ("Speed", "cycle_speed"),
            ("Mute", "toggle_mute"),
            ("AP", "toggle_autopilot"),
            ("AFK", "toggle_afk"),
            ("Land Only", "toggle_land_only"),
            ("Analyze Match", "analyze_match"),
            ("Screen", "analyze_screen"),
            ("Win Plan", "read_win_plan"),
            ("Restart", "restart"),
        ]
        for label, command in commands:
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, cmd=command: self._send_command(cmd))
            button_row.addWidget(button)
        debug_button = QPushButton("Debug Report")
        debug_button.clicked.connect(self._submit_debug_report)
        button_row.addWidget(debug_button)
        root.addLayout(button_row)

        game_box = QGroupBox("Game State")
        game_layout = QVBoxLayout(game_box)
        self.game_state_view = QPlainTextEdit()
        self.game_state_view.setReadOnly(True)
        self.game_state_view.setPlainText(self._last_game_state)
        game_layout.addWidget(self.game_state_view)
        root.addWidget(game_box, stretch=2)

        log_box = QGroupBox("Coach Log")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        log_layout.addWidget(self.log_view)
        root.addWidget(log_box, stretch=3)

        chat_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask the coach or use slash commands like /deck or /analyze")
        self.chat_input.returnPressed.connect(self.send_chat)
        chat_row.addWidget(self.chat_input, stretch=1)
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_chat)
        chat_row.addWidget(send_button)
        root.addLayout(chat_row)

    def send_chat(self) -> None:
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_input.clear()
        self.append_log(f"> {text}", role="status")
        if self._process is not None:
            self._process.send_command("chat", text)

    def _send_command(self, command: str) -> None:
        if self._process is not None:
            self._process.send_command(command)

    def _submit_debug_report(self) -> None:
        if self._process is None:
            self.append_log("Coach process is not running.", role="error")
            return

        note, ok = QInputDialog.getText(
            self,
            "Debug Report",
            "What went wrong? This will save a local report and try to create a GitHub issue.",
        )
        if not ok:
            return
        self.append_log("Submitting debug report...", role="status")
        self._process.send_command("bugreport", note.strip())

    def _handle_event(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return

        event_type = payload.get("type")
        if event_type == "log":
            self.append_log(str(payload.get("message", "")), role="dim")
        elif event_type == "advice":
            seat_info = str(payload.get("seat_info", ""))
            text = str(payload.get("text", ""))
            self.append_log(f"COACH ({seat_info})", role="header")
            self.append_log(text, role="advice")
        elif event_type == "status":
            self._update_status(str(payload.get("key", "")), str(payload.get("value", "")))
        elif event_type == "error":
            self.append_log(f"ERROR: {payload.get('message', '')}", role="error")
        elif event_type == "subtask":
            status = str(payload.get("status", "")).strip()
            if status:
                self.append_log(f"  > {status}", role="status")
        elif event_type == "game_state":
            data = payload.get("data")
            if isinstance(data, dict):
                self._update_game_state(data)
        elif event_type == "speak_request":
            self._tts.request_speech(
                text=str(payload.get("text", "")),
                voice_id=str(payload.get("voice_id", "")),
                voice_name=str(payload.get("voice_name", "")),
                speed=float(payload.get("speed", 1.0) or 1.0),
            )
        elif event_type == "speak_audio":
            path = str(payload.get("path", ""))
            AudioPlayback.play_file(path)
        elif event_type == "speak_stop":
            self._tts.stop_speech()

    def _handle_stderr(self, line: str) -> None:
        self.append_log(f"[stderr] {line}", role="error")

    def _handle_process_exit(self, exit_code: int) -> None:
        self.append_log(f"Coach process exited ({exit_code}).", role="error")

    def _update_status(self, key: str, value: str) -> None:
        normalized = key.upper()
        if normalized == "SEAT_INFO":
            self._status_labels["seat"].setText(value or "-")
        elif normalized == "BACKEND":
            self._status_labels["backend"].setText(value or "-")
        elif normalized == "MODEL":
            self._status_labels["model"].setText(value or "-")
        elif normalized in {"BRIDGE", "GRE"}:
            self._status_labels["bridge"].setText(value or "-")
        elif normalized == "ANALYSIS":
            self._status_labels["analysis"].setText(value or "-")

    def _update_game_state(self, data: dict[str, Any]) -> None:
        lines: list[str] = []

        turn = data.get("turn", {})
        if isinstance(turn, dict):
            turn_num = _int_value(turn.get("turn_number"))
            phase = _str_value(turn.get("phase"))
            step = _str_value(turn.get("step"))
            active_player = _int_value(turn.get("active_player"))
            local_seat = _int_value(data.get("local_seat_id"))

            header = f"Turn {turn_num}  {phase}"
            if step:
                header += f" / {step}"
            if active_player and local_seat:
                header += "  (your turn)" if active_player == local_seat else "  (opp turn)"
            lines.append(header)

        players = data.get("players", [])
        if isinstance(players, list) and players:
            parts: list[str] = []
            for player in players:
                if not isinstance(player, dict):
                    continue
                label = "YOU" if _bool_value(player.get("is_local")) else "OPP"
                parts.append(f"{label}: {_int_value(player.get('life_total'))}")
            if parts:
                lines.append(" | ".join(parts))

        pending_decision = _str_value(data.get("pending_decision"))
        if pending_decision:
            lines.append(f">>> {pending_decision}")

        zones = data.get("zones")
        if not isinstance(zones, dict):
            zones = data

        local_seat = _int_value(data.get("local_seat_id"))
        hand = zones.get("my_hand") or zones.get("hand") or []
        if isinstance(hand, list) and hand:
            cards = []
            for card in hand:
                if not isinstance(card, dict):
                    continue
                name = _str_value(card.get("name"), "?")
                cost = _str_value(card.get("mana_cost"))
                cards.append(f"{name} ({cost})" if cost else name)
            lines.append(f"Hand ({len(cards)}): {', '.join(cards)}")

        battlefield = zones.get("battlefield", [])
        if isinstance(battlefield, list):
            yours: list[str] = []
            opps: list[str] = []
            for card in battlefield:
                if not isinstance(card, dict):
                    continue
                owner = _int_value(card.get("owner_seat_id"))
                type_line = _str_value(card.get("type_line"))
                parts = [_str_value(card.get("name"), "?")]
                if "Creature" in type_line:
                    parts.append(f"{_int_value(card.get('power'))}/{_int_value(card.get('toughness'))}")
                if _bool_value(card.get("is_tapped")):
                    parts.append("T")
                counters = card.get("counters", {})
                if isinstance(counters, dict):
                    for counter_name, counter_value in counters.items():
                        if counter_name == "Loyalty":
                            continue
                        parts.append(f"+{_int_value(counter_value)} {counter_name}")
                if _bool_value(card.get("is_attacking")):
                    parts.append("ATK")
                if _bool_value(card.get("is_blocking")):
                    parts.append("BLK")
                item = " ".join(parts)
                (yours if owner == local_seat else opps).append(item)
            lines.append(f"Your Board ({len(yours)}): {', '.join(yours)}")
            if opps:
                lines.append(f"Opp Board ({len(opps)}): {', '.join(opps)}")

        stack = zones.get("stack", [])
        if isinstance(stack, list) and stack:
            items = [_str_value(item.get("name"), "?") for item in stack if isinstance(item, dict)]
            if items:
                lines.append(f"Stack: {' -> '.join(items)}")

        graveyard = zones.get("graveyard", [])
        if isinstance(graveyard, list) and graveyard:
            yours: list[str] = []
            opps: list[str] = []
            for card in graveyard:
                if not isinstance(card, dict):
                    continue
                name = _str_value(card.get("name"), "?")
                owner = _int_value(card.get("owner_seat_id"))
                (yours if owner == local_seat else opps).append(name)
            if yours:
                lines.append(f"Your GY ({len(yours)}): {', '.join(yours)}")
            if opps:
                lines.append(f"Opp GY ({len(opps)}): {', '.join(opps)}")

        exile = zones.get("exile", [])
        if isinstance(exile, list) and exile:
            items = [_str_value(item.get("name"), "?") for item in exile if isinstance(item, dict)]
            if items:
                lines.append(f"Exile ({len(items)}): {', '.join(items)}")

        command_zone = zones.get("command", [])
        if isinstance(command_zone, list) and command_zone:
            items = [_str_value(item.get("name"), "?") for item in command_zone if isinstance(item, dict)]
            if items:
                lines.append(f"Command: {', '.join(items)}")

        library_count = zones.get("library_count")
        if library_count is not None:
            lines.append(f"Library: {_int_value(library_count)} cards")

        legal_actions = data.get("legal_actions", [])
        if isinstance(legal_actions, list) and legal_actions:
            actions = []
            for action in legal_actions:
                action_text = _str_value(action)
                if (
                    action_text
                    and action_text != "Pass"
                    and not action_text.startswith("Action: Activate_Mana")
                    and not action_text.startswith("Action: FloatMana")
                ):
                    actions.append(action_text)
            if actions:
                lines.append(f"Legal ({len(actions)}): {', '.join(actions)}")

        text = "\n".join(line for line in lines if line).strip() or "Turn 0 waiting for MTGA..."
        self._last_game_state = text
        self.game_state_view.setPlainText(text)


def _str_value(value: Any, fallback: str = "") -> str:
    return value if isinstance(value, str) else fallback


def _int_value(value: Any, fallback: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, dict):
        nested = value.get("value")
        return _int_value(nested, fallback)
    return fallback


def _bool_value(value: Any, fallback: bool = False) -> bool:
    return value if isinstance(value, bool) else fallback
