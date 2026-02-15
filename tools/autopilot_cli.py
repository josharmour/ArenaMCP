"""Headless MTGA Autopilot CLI.

Runs the autopilot loop without a TUI, focusing on autonomous play,
verification, and vision-assisted monitoring.
"""

import sys, os
import threading
import time
import logging
import argparse
from datetime import datetime
from PIL import ImageGrab
import io

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arenamcp.standalone import MCPClient, UIAdapter
from arenamcp.coach import CoachEngine, create_backend, GameStateTrigger
from arenamcp.action_planner import ActionPlanner
from arenamcp.autopilot import AutopilotEngine, AutopilotConfig, AutopilotState
from arenamcp.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AutopilotCLI")

class SimpleCLIAdapter(UIAdapter):
    """Clean CLI output for the autopilot."""
    def log(self, message: str) -> None:
        # Strip rich tags if any
        msg = message.replace("[bold]", "").replace("[/]", "").replace("[green]", "").replace("[red]", "")
        print(f"[*] {msg}")
        
    def advice(self, text: str, seat_info: str) -> None:
        print(f"\n[AI|{seat_info}] {text}\n")
        
    def status(self, key: str, value: str) -> None:
        print(f"STATUS | {key}: {value}")
        
    def error(self, message: str) -> None:
        print(f"!!! ERROR: {message}")

class HeadlessAutopilot:
    def __init__(self, args):
        self.args = args
        self.settings = get_settings()
        self.mcp = MCPClient()
        
        # 1. Initialize LLM
        logger.info(f"Initializing LLM ({args.backend})...")
        self.backend = create_backend(args.backend, model=args.model)
        self.coach = CoachEngine(backend=self.backend)
        self.planner = ActionPlanner(backend=self.backend)
        self.trigger_detector = GameStateTrigger()
        
        # 2. Initialize Autopilot Hands
        from arenamcp.screen_mapper import ScreenMapper
        from arenamcp.input_controller import InputController
        self.mapper = ScreenMapper()
        self.controller = InputController(dry_run=args.dry_run)
        
        config = AutopilotConfig(
            dry_run=args.dry_run,
            auto_execute_delay=0.5, # Faster response for CLI
            verify_after_action=True
        )
        
        self.engine = AutopilotEngine(
            planner=self.planner,
            mapper=self.mapper,
            controller=self.controller,
            get_game_state=self.mcp.get_game_state,
            config=config,
            ui_advice_fn=SimpleCLIAdapter().advice
        )
        
        self.running = True
        self.last_state = {}
        self.last_action_time = time.time()
        self.idle_threshold = getattr(args, 'idle_threshold', 8)

    def run(self):
        print("\n" + "="*50)
        print(" MTGA HEADLESS AUTOPILOT STARTING")
        print(f" Backend: {self.args.backend} | Model: {self.args.model}")
        print(f" Mode: {'DRY RUN' if self.args.dry_run else 'LIVE'}")
        print("="*50 + "\n")

        # Show watcher status
        from arenamcp import server as _srv
        if _srv.watcher:
            print(f"  Log: {_srv.watcher.log_path}")
            print(f"  Turn: {_srv.game_state.turn_info.turn_number} | Players: {list(_srv.game_state.players.keys())}")
        else:
            print("  WARNING: No watcher created!")

        while self.running:
            try:
                # 1. Poll Game State
                self.mcp.poll_log()
                gs = self.mcp.get_game_state()
                
                if not gs or not gs.get("turn"):
                    time.sleep(2)
                    continue
                
                turn = gs.get("turn", {})
                turn_num = turn.get("turn_number", 0)
                phase = turn.get("phase", "")
                
                if turn_num == 0:
                    print(f"Waiting for match... (Time: {datetime.now().strftime('%H:%M:%S')})", end="\r")
                    time.sleep(2)
                    continue

                # Determine priority before trigger processing
                local_seat = 1
                for p in gs.get("players", []):
                    if p.get("is_local"): local_seat = p.get("seat_id")
                is_my_priority = turn.get("priority_player") == local_seat

                # 2. Check for Triggers
                triggers = self.trigger_detector.check_triggers(self.last_state, gs)

                if triggers:
                    logger.info(f"Triggers detected: {triggers}")
                    # Prioritize critical triggers
                    for trigger in sorted(triggers, key=lambda x: 1 if x == "decision_required" else 0):
                        # Process through autopilot engine
                        if self.engine.process_trigger(gs, trigger):
                            # CRITICAL: If we took action, the game state is now STALE.
                            # We must poll again to get the result of our action.
                            logger.info("Action taken, refreshing game state...")
                            self.mcp.poll_log()
                            gs = self.mcp.get_game_state()

                        self.last_action_time = time.time()

                # 3. Proactive check: if we have priority and a pending decision,
                # ensure autopilot acts even if triggers missed the transition
                if not triggers and is_my_priority:
                    pending = gs.get("pending_decision")
                    if pending and pending != "Action Required":
                        logger.info(f"Proactive: pending decision '{pending}' with our priority, no trigger fired")
                        if self.engine.process_trigger(gs, "decision_required"):
                            self.mcp.poll_log()
                            gs = self.mcp.get_game_state()
                        self.last_action_time = time.time()
                    elif pending == "Action Required" or not pending:
                        # We have priority but no specific decision - check if idle too long
                        idle_time = time.time() - self.last_action_time
                        if idle_time > self.idle_threshold:
                            logger.info(f"Proactive: priority held for {idle_time:.1f}s, forcing trigger")
                            if self.engine.process_trigger(gs, "priority_gained"):
                                self.mcp.poll_log()
                                gs = self.mcp.get_game_state()
                            self.last_action_time = time.time()

                # 4. Vision Watchdog
                if is_my_priority and (time.time() - self.last_action_time > self.idle_threshold):
                    logger.info("Watchdog: Idle detected on player turn. Checking vision...")
                    self._vision_check(gs)
                    self.last_action_time = time.time()

                self.last_state = gs

                # Adaptive polling: faster when we have priority (game waiting for us)
                if is_my_priority:
                    time.sleep(0.5)
                elif turn_num > 0:
                    time.sleep(1.0)
                else:
                    time.sleep(2.0)
                
            except KeyboardInterrupt:
                logger.info("Stopping autopilot...")
                self.running = False
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(5)

    def _vision_check(self, gs):
        """Take a screenshot and ask the LLM if we are stuck on a UI prompt."""
        logger.info("Capturing screenshot for watchdog...")
        window_rect = self.mapper.refresh_window()
        if not window_rect: return
        
        left, top, w, h = window_rect
        screenshot = ImageGrab.grab(bbox=(left, top, left+w, top+h))
        buf = io.BytesIO()
        screenshot.save(buf, format='PNG')
        
        system_prompt = "You are an MTG Arena watchdog. Check this screenshot for any blocking UI elements (prompts, targets, scry, etc.) that the player needs to interact with."
        user_msg = "Am I stuck on a choice? If so, what should I click? If no prompt is visible, say 'CLEAR'."
        
        try:
            if hasattr(self.backend, 'complete_with_image'):
                response = self.backend.complete_with_image(system_prompt, user_msg, buf.getvalue())
                logger.info(f"Watchdog response: {response}")
                if "CLEAR" not in response.upper():
                    logger.warning(f"Vision watchdog: stuck detected - {response}")
                    self.controller.focus_mtga_window()
                    time.sleep(0.2)

                    # Check for known button keywords in the LLM response
                    response_lower = response.lower()
                    if any(kw in response_lower for kw in ("done", "ok", "confirm", "submit", "accept")):
                        self.engine._click_fixed("done")
                    elif any(kw in response_lower for kw in ("cancel", "decline", "no")):
                        self.controller.press_key("escape", "Vision: dismiss")
                    elif "keep" in response_lower:
                        self.engine._click_fixed("keep")
                    elif "mulligan" in response_lower:
                        self.engine._click_fixed("mulligan")
                    else:
                        # Fallback: try spacebar (pass/resolve) then escape
                        self.controller.press_key("space", "Vision: spacebar fallback")
                        time.sleep(0.5)
                        self.controller.press_key("escape", "Vision: escape fallback")
        except Exception as e:
            logger.error(f"Vision watchdog failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="proxy")
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--idle-threshold", type=float, default=8,
                        help="Seconds of idle priority before vision watchdog fires (default: 8)")
    args = parser.parse_args()
    
    app = HeadlessAutopilot(args)
    app.run()
