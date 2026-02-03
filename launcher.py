"""Lightweight launcher for MTGA Coach with auto-restart support.

This launcher:
- Runs the TUI in a subprocess so it can be cleanly restarted
- Detects restart requests and relaunches automatically
- Can optionally watch for code changes during development
- Handles zombie process cleanup properly

Usage from command line:
    python launcher.py
    python launcher.py --watch  # Auto-restart on .py file changes

The desktop shortcut should point to coach.bat which calls this.
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

# Constants
REPO_DIR = Path(__file__).parent
SRC_DIR = REPO_DIR / "src" / "arenamcp"
RESTART_EXIT_CODE = 42  # Special exit code meaning "please restart"
WATCH_EXTENSIONS = {".py"}
WATCH_DEBOUNCE_MS = 500


def get_python_executable():
    """Get the Python executable path."""
    return sys.executable


def run_coach(extra_args=None):
    """Run the coach as a subprocess and return exit code."""
    cmd = [get_python_executable(), "-m", "arenamcp.standalone"]
    if extra_args:
        cmd.extend(extra_args)

    # Set environment to signal we're in launcher mode
    env = os.environ.copy()
    env["ARENAMCP_LAUNCHER"] = "1"

    try:
        # Run as subprocess - this allows clean termination
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_DIR),
            env=env,
            # Don't capture output - let it go to the terminal
            stdout=None,
            stderr=None,
        )

        # Wait for process to complete
        return proc.wait()

    except KeyboardInterrupt:
        # User pressed Ctrl+C - terminate gracefully
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


def watch_for_changes():
    """Simple file watcher that returns True when .py files change."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        changed = [False]

        class Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.src_path.endswith('.py'):
                    changed[0] = True

        observer = Observer()
        observer.schedule(Handler(), str(SRC_DIR), recursive=True)
        observer.start()

        return observer, lambda: changed[0]
    except ImportError:
        return None, lambda: False


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_banner(restart_count=0):
    """Print startup banner."""
    print("=" * 50)
    print("  MTGA Coach Launcher")
    print("=" * 50)
    if restart_count > 0:
        print(f"  Restart #{restart_count}")
    print()
    print("  Ctrl+C = Exit | F9 in app = Restart")
    print("=" * 50)
    print()


def main():
    """Main launcher loop."""
    import argparse

    parser = argparse.ArgumentParser(description="MTGA Coach Launcher")
    parser.add_argument("--watch", "-w", action="store_true",
                        help="Auto-restart when .py files change")
    parser.add_argument("args", nargs="*", help="Arguments to pass to coach")

    args = parser.parse_args()

    # Setup file watcher if requested
    observer = None
    check_changed = lambda: False
    if args.watch:
        observer, check_changed = watch_for_changes()
        if observer:
            print("[Launcher] Watching for code changes...")
        else:
            print("[Launcher] Warning: watchdog not installed, --watch disabled")

    restart_count = 0

    try:
        while True:
            clear_screen()
            print_banner(restart_count)

            # Run the coach
            exit_code = run_coach(args.args)

            # Check why it exited
            if exit_code == RESTART_EXIT_CODE:
                # Explicit restart request from F9
                print("\n[Launcher] Restart requested...")
                restart_count += 1
                time.sleep(0.5)
                continue

            elif check_changed():
                # Code changed during execution
                print("\n[Launcher] Code changed, restarting...")
                restart_count += 1
                time.sleep(0.5)
                continue

            elif exit_code != 0:
                # Error exit
                print(f"\n[Launcher] Coach exited with error code {exit_code}")
                print("\nPress Enter to restart, or Ctrl+C to exit...")
                try:
                    input()
                    restart_count += 1
                    continue
                except KeyboardInterrupt:
                    break
            else:
                # Normal exit (Ctrl+Q from app)
                print("\n[Launcher] Coach exited normally.")
                break

    except KeyboardInterrupt:
        print("\n[Launcher] Interrupted, exiting...")
    finally:
        if observer:
            observer.stop()
            observer.join()

    print("\nGoodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
