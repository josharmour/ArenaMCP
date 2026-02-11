"""Quick test: verify the input backend can find MTGA and click.

Usage (run from Windows Python, not WSL):
    python tools/test_click.py              # find window + click pass button
    python tools/test_click.py --dry-run    # just log, don't actually click
    python tools/test_click.py --notepad    # test on Notepad instead of MTGA
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test autopilot input backend")
    parser.add_argument("--dry-run", action="store_true", help="Log only, no clicks")
    parser.add_argument("--notepad", action="store_true", help="Test on Notepad instead of MTGA")
    args = parser.parse_args()

    # Set up logging to console
    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    from arenamcp.input_controller import (
        find_mtga_hwnd, get_client_rect, force_foreground,
        InputController, _create_backend,
    )

    # --- 1. Test backend creation ---
    print("\n=== Backend Selection ===")
    backend = _create_backend()
    if backend:
        print(f"  Active backend: {backend.name}")
    else:
        print("  ERROR: No backend available!")
        return

    # --- 2. Test window detection ---
    print("\n=== Window Detection ===")
    if args.notepad:
        import ctypes
        # Try common Notepad window titles (varies by Windows version)
        notepad_titles = [
            "Untitled - Notepad",   # Win10 classic
            "Notepad",              # Generic
            "*Untitled - Notepad",  # Unsaved changes
        ]
        hwnd = None
        for title in notepad_titles:
            hwnd = ctypes.windll.user32.FindWindowW(None, title)
            if hwnd:
                break
        # Win11 Notepad: try by class name instead
        if not hwnd:
            hwnd = ctypes.windll.user32.FindWindowW("Notepad", None)
        if not hwnd:
            hwnd = ctypes.windll.user32.FindWindowW("NOTEPAD", None)
        # Last resort: enumerate windows for anything with "notepad" in title
        if not hwnd:
            from arenamcp.input_controller import _IS_WINDOWS
            if _IS_WINDOWS:
                WNDENUMPROC = ctypes.WINFUNCTYPE(
                    ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM,
                )
                found = [None]
                def _cb(h, _):
                    length = ctypes.windll.user32.GetWindowTextLengthW(h)
                    if length > 0:
                        buf = ctypes.create_unicode_buffer(length + 1)
                        ctypes.windll.user32.GetWindowTextW(h, buf, length + 1)
                        if "notepad" in buf.value.lower():
                            found[0] = h
                            return False
                    return True
                ctypes.windll.user32.EnumWindows(WNDENUMPROC(_cb), 0)
                hwnd = found[0]
        if not hwnd:
            print("  Notepad not found — open Notepad first!")
            return
        print(f"  Found Notepad: hwnd={hwnd}")
    else:
        hwnd = find_mtga_hwnd()
        if not hwnd:
            print("  MTGA window not found — is it running?")
            print("  Tip: use --notepad to test on Notepad instead")
            return
        print(f"  Found MTGA: hwnd={hwnd}")

    rect = get_client_rect(hwnd)
    if not rect:
        print("  ERROR: Could not get client rect")
        return
    left, top, w, h = rect
    print(f"  Client rect: left={left}, top={top}, {w}x{h}")

    # --- 3. Test focus ---
    print("\n=== Focus Test ===")
    if not args.dry_run:
        ok = force_foreground(hwnd)
        print(f"  force_foreground: {'OK' if ok else 'FAILED'}")
        time.sleep(0.3)
    else:
        print("  [DRY RUN] Skipping focus")

    # --- 4. Test click ---
    print("\n=== Click Test ===")
    controller = InputController(dry_run=args.dry_run)
    print(f"  Controller backend: {controller.backend_name}")

    if args.notepad:
        # Click center of notepad
        cx, cy = left + w // 2, top + h // 2
        print(f"  Clicking center of Notepad: ({cx}, {cy})")
        result = controller.click(cx, cy, "Notepad center", rect)
    else:
        # Click the pass/resolve button area (normalized 0.78, 0.85)
        pass_x = int(left + 0.78 * w)
        pass_y = int(top + 0.85 * h)
        print(f"  Clicking pass button area: ({pass_x}, {pass_y})")
        print(f"    (normalized: 0.78, 0.85 within {w}x{h} client area)")
        result = controller.click(pass_x, pass_y, "Pass/Resolve button", rect)

    print(f"  Result: {result}")

    # --- 5. Summary ---
    print(f"\n=== Summary ===")
    print(f"  Backend:  {controller.backend_name}")
    print(f"  Window:   {'Notepad' if args.notepad else 'MTGA'} ({w}x{h})")
    print(f"  Click:    {'OK' if result.success else 'FAILED'}")
    print(f"  Dry run:  {args.dry_run}")

if __name__ == "__main__":
    main()
