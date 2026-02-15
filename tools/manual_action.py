"""Manual Autopilot Action Executor.

Allows specifying an action (like "play land", "cast spell", or a button name)
to execute immediately on the current game state.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import argparse
import json
import logging

def main():
    parser = argparse.ArgumentParser(description="Execute a manual autopilot action")
    parser.add_argument("action_type", help="Type: play_land, cast_spell, click_button, resolve, etc.")
    parser.add_argument("--name", help="Card or button name", default="")
    parser.add_argument("--dry-run", action="store_true", help="Log only, no clicks")
    parser.add_argument("--x", type=float, help="Override normalized X coordinate (0.0-1.0)")
    parser.add_argument("--y", type=float, help="Override normalized Y coordinate (0.0-1.0)")
    parser.add_argument("--click", action="store_true", help="Do a simple click instead of drag/cast (for testing coordinates)")
    parser.add_argument("--hover", action="store_true", help="Just move the mouse to the calculated position without clicking")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logger = logging.getLogger("ManualAction")

    from arenamcp.input_controller import InputController
    from arenamcp.screen_mapper import ScreenMapper, ScreenCoord
    from arenamcp.autopilot import AutopilotEngine, AutopilotConfig, GameAction, ActionType
    from arenamcp.server import get_game_state

    # 1. Get current game state
    logger.info("Fetching game state...")
    game_state = get_game_state()
    if not game_state:
        logger.error("Could not fetch game state. Is the server running?")
        return

    # 2. Setup Engine components
    mapper = ScreenMapper()
    window_rect = mapper.refresh_window()
    if window_rect:
        logger.info(f"MTGA Window Rect: {window_rect}")
    else:
        logger.error("MTGA Window not found!")
        return

    controller = InputController(dry_run=args.dry_run)
    config = AutopilotConfig(dry_run=args.dry_run)
    
    # We create a dummy planner since we are providing the action manually
    engine = AutopilotEngine(
        planner=None, 
        mapper=mapper, 
        controller=controller, 
        get_game_state=get_game_state,
        config=config
    )

    # 3. Construct the action
    try:
        atype = ActionType(args.action_type.lower())
    except ValueError:
        logger.error(f"Invalid action type: {args.action_type}")
        logger.info(f"Valid types: {[t.value for t in ActionType]}")
        return

    action = GameAction(action_type=atype, card_name=args.name)
    logger.info(f"Executing: {action}")

    # 4. Focus and Execute
    if not args.dry_run:
        logger.info("Focusing MTGA... (You have 3 seconds to click the MTGA window!)")
        controller.focus_mtga_window()
        time.sleep(3.0)

    # Override coordinates if provided
    if args.x is not None or args.y is not None:
        def get_custom_coord(*args_inner, **kwargs_inner):
            cx = args.x if args.x is not None else 0.5
            cy = args.y if args.y is not None else 0.95
            return ScreenCoord(cx, cy, "Manual Override")
        
        # Monkey patch the mapper for this run
        mapper.get_card_in_hand_coord = get_custom_coord
        mapper.get_button_coord = get_custom_coord
        logger.info(f"OVERRIDE enabled: x={args.x}, y={args.y}")

    if args.click or args.hover:
        # Simple click/hover test
        hand = game_state.get("hand", [])
        battlefield = game_state.get("battlefield", [])
        
        # Try finding as permanent first, then hand card
        local_seat = 1
        for p in game_state.get("players", []):
            if p.get("is_local"):
                local_seat = p.get("seat_id")

        coord = mapper.get_permanent_coord(args.name, None, battlefield, local_seat, local_seat)
        if not coord:
            coord = mapper.get_card_in_hand_coord(args.name, hand, game_state)
            
        if coord:
            abs_x, abs_y = coord.to_absolute(window_rect)
            if args.hover:
                logger.info(f"TEST HOVER at {abs_x}, {abs_y}")
                controller._backend.move_to(abs_x, abs_y, duration=0.5)
                return # Hover test finished
            else:
                logger.info(f"TEST CLICK at {abs_x}, {abs_y}")
                result = controller.click(abs_x, abs_y, "Test Click", window_rect)
        else:
            logger.error("Could not determine coordinates")
            return
    else:
        # Full action execution
        result = engine._execute_action(action, game_state)
    
    if result.success:
        logger.info(f"SUCCESS: {result}")
    else:
        logger.error(f"FAILED: {result.error}")

if __name__ == "__main__":
    main()
