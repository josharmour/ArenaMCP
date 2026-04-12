from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .runtime import (
    RuntimeState,
    detect_runtime_state,
    install_bepinex,
    install_plugin,
    is_mtga_running,
    open_path,
    open_url,
    read_version,
    repair_bridge_stack,
    run_setup_wizard,
    set_saved_mtga_dir,
    tail_text,
)


class RepairTab(QWidget):
    restart_requested = Signal(bool, bool, bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state: Optional[RuntimeState] = None
        self._status_labels: dict[str, QLabel] = {}
        self._build_ui()
        self.refresh_state()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        launch_box = QGroupBox("Launch Options")
        launch_layout = QHBoxLayout(launch_box)
        restart_button = QPushButton("Restart Coach")
        restart_button.clicked.connect(lambda: self.restart_requested.emit(False, self.dry_run_check.isChecked(), self.afk_check.isChecked()))
        launch_layout.addWidget(restart_button)
        autopilot_button = QPushButton("Restart as Autopilot")
        autopilot_button.clicked.connect(lambda: self.restart_requested.emit(True, self.dry_run_check.isChecked(), self.afk_check.isChecked()))
        launch_layout.addWidget(autopilot_button)
        self.dry_run_check = QCheckBox("Dry-run")
        launch_layout.addWidget(self.dry_run_check)
        self.afk_check = QCheckBox("AFK")
        launch_layout.addWidget(self.afk_check)
        launch_layout.addStretch(1)
        layout.addWidget(launch_box)

        status_box = QGroupBox("Runtime Status")
        status_layout = QFormLayout(status_box)
        for key in (
            "Runtime Root",
            "Python Runtime",
            "MTGA Install",
            "MTGA Process",
            "BepInEx",
            "Bridge Plugin",
            "BepInEx Bundle",
            "Player.log",
            "Bridge Readiness",
        ):
            label = QLabel("...")
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._status_labels[key] = label
            status_layout.addRow(f"{key}:", label)
        layout.addWidget(status_box)

        refresh_row = QHBoxLayout()
        refresh_status = QPushButton("Refresh Status")
        refresh_status.clicked.connect(self.refresh_state)
        refresh_row.addWidget(refresh_status)
        refresh_logs = QPushButton("Refresh Logs")
        refresh_logs.clicked.connect(self.refresh_log_tails)
        refresh_row.addWidget(refresh_logs)
        refresh_row.addStretch(1)
        layout.addLayout(refresh_row)

        mtga_box = QGroupBox("MTGA Location")
        mtga_layout = QHBoxLayout(mtga_box)
        self.mtga_path_box = QLineEdit()
        self.mtga_path_box.setPlaceholderText(r"C:\Program Files\Wizards of the Coast\MTGA")
        mtga_layout.addWidget(self.mtga_path_box, stretch=1)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_mtga)
        mtga_layout.addWidget(browse_button)
        save_button = QPushButton("Save")
        save_button.clicked.connect(self._save_mtga_path)
        mtga_layout.addWidget(save_button)
        layout.addWidget(mtga_box)

        fix_box = QGroupBox("Repair")
        fix_layout = QVBoxLayout(fix_box)
        action_row = QHBoxLayout()
        self.fix_all_button = QPushButton("Fix Everything")
        self.fix_all_button.clicked.connect(self.fix_everything)
        action_row.addWidget(self.fix_all_button)
        provision_button = QPushButton("Provision Runtime")
        provision_button.clicked.connect(self._provision_runtime)
        action_row.addWidget(provision_button)
        repair_button = QPushButton("Repair MTGA Bridge")
        repair_button.clicked.connect(self._repair_bridge)
        action_row.addWidget(repair_button)
        install_bepinex_button = QPushButton("Install BepInEx")
        install_bepinex_button.clicked.connect(self._install_bepinex)
        action_row.addWidget(install_bepinex_button)
        install_plugin_button = QPushButton("Install Plugin")
        install_plugin_button.clicked.connect(self._install_plugin)
        action_row.addWidget(install_plugin_button)
        action_row.addStretch(1)
        fix_layout.addLayout(action_row)

        open_row = QHBoxLayout()
        open_mtga = QPushButton("Open MTGA Folder")
        open_mtga.clicked.connect(self._open_mtga)
        open_row.addWidget(open_mtga)
        open_player_log = QPushButton("Open Player.log")
        open_player_log.clicked.connect(self._open_player_log)
        open_row.addWidget(open_player_log)
        open_bepinex_log = QPushButton("Open BepInEx Log")
        open_bepinex_log.clicked.connect(self._open_bepinex_log)
        open_row.addWidget(open_bepinex_log)
        open_releases = QPushButton("GitHub Releases")
        open_releases.clicked.connect(lambda: open_url())
        open_row.addWidget(open_releases)
        open_row.addStretch(1)
        fix_layout.addLayout(open_row)

        self.fix_status = QLabel(f"mtgacoach v{read_version()}")
        fix_layout.addWidget(self.fix_status)
        self.fix_log = QPlainTextEdit()
        self.fix_log.setReadOnly(True)
        self.fix_log.setMaximumBlockCount(500)
        self.fix_log.setMinimumHeight(120)
        fix_layout.addWidget(self.fix_log)
        layout.addWidget(fix_box)

        log_box = QGroupBox("Log Tails")
        log_layout = QVBoxLayout(log_box)
        self.log_tail_view = QPlainTextEdit()
        self.log_tail_view.setReadOnly(True)
        self.log_tail_view.setMinimumHeight(260)
        log_layout.addWidget(self.log_tail_view)
        layout.addWidget(log_box)

    def refresh_state(self) -> None:
        state = detect_runtime_state()
        self._state = state
        if not self.mtga_path_box.text().strip() and state.mtga_dir:
            self.mtga_path_box.setText(state.mtga_dir)

        self._set_status(
            "Runtime Root",
            f"{state.runtime_venv_dir} [{state.python_source}]"
            if state.runtime_venv_exists
            else f"{state.runtime_root} (setup required)",
            "ok" if state.runtime_venv_exists else "warn",
        )
        self._set_status(
            "Python Runtime",
            f"{state.python_exe} [{state.python_source}]" if state.python_exe else "Missing",
            "ok" if state.python_exe else "error",
        )
        self._set_status(
            "MTGA Install",
            f"{state.mtga_dir} ({state.mtga_dir_source})" if state.mtga_dir else "Not detected",
            "ok" if state.mtga_dir else "error",
        )
        self._set_status("MTGA Process", "Running" if state.mtga_running else "Not running", "warn" if state.mtga_running else "ok")
        self._set_status("BepInEx", state.bepinex_dir or "Missing", "ok" if state.bepinex_installed else "error")
        plugin_text = (
            state.plugin_install_path
            or (f"Built at {state.plugin_build_path}" if state.plugin_built and state.plugin_build_path else "Missing")
        )
        self._set_status(
            "Bridge Plugin",
            plugin_text,
            "ok" if state.plugin_installed else "warn" if state.plugin_built else "error",
        )
        bundle_text = state.bepinex_bundle or ("Already installed in MTGA" if state.bepinex_installed else "No bundle found")
        self._set_status(
            "BepInEx Bundle",
            bundle_text,
            "ok" if state.bepinex_bundle or state.bepinex_installed else "warn",
        )
        player_log_path = state.player_log
        self._set_status(
            "Player.log",
            player_log_path if Path(player_log_path).exists() else f"Missing ({player_log_path})",
            "ok" if Path(player_log_path).exists() else "warn",
        )
        readiness = "Ready" if state.is_fully_provisioned else "Ready (fallback)" if state.is_launchable else "Incomplete"
        self._set_status(
            "Bridge Readiness",
            readiness,
            "ok" if state.is_fully_provisioned or state.is_launchable else "warn",
        )
        self.refresh_log_tails()

    def refresh_log_tails(self) -> None:
        standalone_log = Path.home() / ".arenamcp" / "standalone.log"
        lines = [
            "[standalone.log tail]",
            tail_text(str(standalone_log), 3000),
            "",
            "[Player.log tail]",
            tail_text(self._state.player_log if self._state else None, 3000),
            "",
            "[BepInEx log tail]",
            tail_text(self._state.bepinex_log if self._state else None, 3000),
        ]
        self.log_tail_view.setPlainText("\n".join(lines))

    def fix_everything(self) -> None:
        self.fix_all_button.setEnabled(False)
        self.fix_log.clear()
        try:
            self._set_fix_step("Scanning...")
            state = detect_runtime_state()
            mtga_dir = state.mtga_dir or self._selected_mtga_dir(required=False)

            if state.is_fully_provisioned and not state.issues:
                self._append_fix_log("[ok] System is fully provisioned")
                self._set_fix_step("All good")
                return

            self._append_fix_log(f"Found {len(state.issues)} issue(s): {', '.join(state.issues) or 'none'}")

            if state.python_exe is None:
                self._append_fix_log("[!!] No Python found")
                self._set_fix_step("Blocked: no Python")
                return

            if not state.runtime_venv_exists:
                self._append_fix_log("[..] Launching setup wizard to provision the runtime")
                run_setup_wizard()
                self._append_fix_log("[ok] Setup wizard launched. Finish it, then click Refresh Status.")
                self._set_fix_step("Runtime setup started")
                return

            if not mtga_dir:
                self._append_fix_log("[!!] MTGA install not detected")
                self._set_fix_step("Blocked: no MTGA path")
                return

            if is_mtga_running():
                self._append_fix_log("[!!] Close MTGA before repairing the bridge stack")
                self._set_fix_step("Blocked: MTGA running")
                return

            if not state.bepinex_installed:
                self._set_fix_step("Installing BepInEx...")
                target = install_bepinex(mtga_dir)
                self._append_fix_log(f"[ok] BepInEx installed at {target}")
            else:
                self._append_fix_log("[ok] BepInEx already installed")

            state = detect_runtime_state()
            plugin_needs_update = False
            if (
                state.plugin_built
                and state.plugin_build_path
                and state.plugin_install_path
                and Path(state.plugin_build_path).stat().st_mtime > Path(state.plugin_install_path).stat().st_mtime
            ):
                plugin_needs_update = True

            if not state.plugin_installed or plugin_needs_update:
                self._set_fix_step("Installing bridge plugin...")
                target = install_plugin(mtga_dir)
                action = "updated" if plugin_needs_update else "installed"
                self._append_fix_log(f"[ok] Plugin {action} at {target}")
            else:
                self._append_fix_log("[ok] Bridge plugin already installed")

            self.refresh_state()
            state = self._state
            if state and state.is_fully_provisioned:
                self._set_fix_step("All fixed")
                self._append_fix_log("[ok] System is fully provisioned and ready")
            elif state and state.is_launchable:
                self._set_fix_step("Launchable")
                self._append_fix_log(f"[ok] Launchable with caveats: {', '.join(state.issues)}")
            else:
                self._set_fix_step("Some issues remain")
        except Exception as exc:
            self._set_fix_step("Error")
            self._append_fix_log(f"[!!] Fix failed: {exc}")
            QMessageBox.critical(self, "Repair Failed", str(exc))
        finally:
            self.fix_all_button.setEnabled(True)

    def _browse_mtga(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select MTGA Install Folder")
        if folder:
            self.mtga_path_box.setText(folder)

    def _save_mtga_path(self) -> None:
        path = self.mtga_path_box.text().strip()
        if not path:
            QMessageBox.information(self, "mtgacoach", "Choose an MTGA folder first.")
            return
        set_saved_mtga_dir(path)
        self.refresh_state()
        QMessageBox.information(self, "mtgacoach", f"Saved MTGA folder:\n{path}")

    def _provision_runtime(self) -> None:
        if self._state and self._state.python_source == "app_runtime":
            QMessageBox.information(self, "mtgacoach", "Bundled runtime is already installed.")
            return
        try:
            run_setup_wizard()
            QMessageBox.information(self, "mtgacoach", "Setup wizard launched.")
        except Exception as exc:
            QMessageBox.critical(self, "Provision Runtime Failed", str(exc))

    def _repair_bridge(self) -> None:
        try:
            changed = repair_bridge_stack(self._selected_mtga_dir())
            self.refresh_state()
            QMessageBox.information(self, "mtgacoach", "\n".join(changed) if changed else "No changes needed.")
        except Exception as exc:
            QMessageBox.critical(self, "Repair Bridge Failed", str(exc))

    def _install_bepinex(self) -> None:
        try:
            target = install_bepinex(self._selected_mtga_dir())
            self.refresh_state()
            QMessageBox.information(self, "mtgacoach", f"BepInEx installed at:\n{target}")
        except Exception as exc:
            QMessageBox.critical(self, "Install BepInEx Failed", str(exc))

    def _install_plugin(self) -> None:
        try:
            target = install_plugin(self._selected_mtga_dir())
            self.refresh_state()
            QMessageBox.information(self, "mtgacoach", f"Plugin installed at:\n{target}")
        except Exception as exc:
            QMessageBox.critical(self, "Install Plugin Failed", str(exc))

    def _open_mtga(self) -> None:
        try:
            open_path(self._selected_mtga_dir())
        except Exception:
            pass

    def _open_player_log(self) -> None:
        if self._state:
            open_path(self._state.player_log)

    def _open_bepinex_log(self) -> None:
        if self._state and self._state.bepinex_log:
            open_path(self._state.bepinex_log)

    def _selected_mtga_dir(self, required: bool = True) -> Optional[str]:
        text = self.mtga_path_box.text().strip()
        if text:
            return text
        if self._state and self._state.mtga_dir:
            return self._state.mtga_dir
        if required:
            raise RuntimeError("MTGA install folder is not set")
        return None

    def _append_fix_log(self, message: str) -> None:
        self.fix_log.appendPlainText(message)
        QApplication.processEvents()

    def _set_fix_step(self, message: str) -> None:
        self.fix_status.setText(message)
        QApplication.processEvents()

    def _set_status(self, key: str, text: str, level: str) -> None:
        label = self._status_labels.get(key)
        if label is None:
            return
        colors = {
            "ok": "#245c3c",
            "warn": "#8a5a00",
            "error": "#8d1f1f",
            "default": "#334e68",
        }
        label.setText(text)
        label.setStyleSheet(f"color: {colors.get(level, colors['default'])};")
