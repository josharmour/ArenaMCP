from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QStatusBar, QTabWidget

from .coach_process import CoachProcess
from .coach_tab import CoachTab
from .repair_tab import RepairTab
from .runtime import detect_runtime_state, read_version


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._closing = False
        self._launch_flags = (False, False, False)
        self._process: Optional[CoachProcess] = None

        self.setWindowTitle(f"mtgacoach v{read_version()}")
        self.resize(1400, 980)

        tabs = QTabWidget()
        self.coach_tab = CoachTab()
        self.repair_tab = RepairTab()
        self.repair_tab.restart_requested.connect(self.restart_coach)
        tabs.addTab(self.coach_tab, "Coach")
        tabs.addTab(self.repair_tab, "Repair")
        self.tabs = tabs
        self.setCentralWidget(tabs)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self._status_bar = status_bar

        refresh_action = QAction("Refresh Status", self)
        refresh_action.triggered.connect(self.refresh_state)
        self.menuBar().addAction(refresh_action)

        self.refresh_state()
        self._auto_start()

    def refresh_state(self) -> None:
        self.repair_tab.refresh_state()

    def restart_coach(self, autopilot: bool, dry_run: bool, afk: bool) -> None:
        self._launch_flags = (autopilot, dry_run, afk)
        if self._process is not None:
            self.coach_tab.detach_process()
            self._process.stop()
            self._process.deleteLater()
            self._process = None
        self._start_coach(*self._launch_flags)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._closing = True
        if self._process is not None:
            self.coach_tab.detach_process()
            self._process.stop()
            self._process.deleteLater()
            self._process = None
        self.coach_tab.shutdown()
        super().closeEvent(event)

    def _auto_start(self) -> None:
        state = detect_runtime_state()
        if state.python_exe is None:
            self._status_bar.showMessage("Python not found. Go to Repair to set up.")
            self.tabs.setCurrentIndex(1)
            return
        self._start_coach(False, False, False)

    def _start_coach(self, autopilot: bool, dry_run: bool, afk: bool) -> None:
        process = CoachProcess(self)
        process.exited.connect(self._on_process_exited)
        self._process = process
        self.coach_tab.attach_process(process)

        try:
            process.start(autopilot=autopilot, dry_run=dry_run, afk=afk)
            self._status_bar.showMessage("Coach is running.")
            self.tabs.setCurrentIndex(0)
        except Exception as exc:
            self.coach_tab.detach_process()
            self._process = None
            self._status_bar.showMessage(f"Coach failed to start: {exc}")
            self.tabs.setCurrentIndex(1)
            QMessageBox.critical(
                self,
                "Coach Launch Failed",
                f"{exc}\n\n{process.last_error}",
            )

    def _on_process_exited(self, exit_code: int) -> None:
        if self._closing:
            return

        self._status_bar.showMessage(f"Coach exited ({exit_code}). Restarting...")
        if self._process is not None:
            self.coach_tab.detach_process()
            self._process.deleteLater()
            self._process = None
        QTimer.singleShot(250, lambda: self._start_coach(*self._launch_flags))
