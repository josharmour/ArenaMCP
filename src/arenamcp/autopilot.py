"""Autopilot Mode - Core Orchestration Engine.

Ties ActionPlanner + ScreenMapper + InputController together with
human-in-the-loop confirmation gates (spacebar to confirm, escape to skip).

The autopilot layers onto the existing coaching loop without replacing it:

    GameState polling → Triggers → ActionPlanner.plan_actions() → Preview
    → [SPACEBAR confirm] → InputController.execute() → Verify state → Loop
"""

import logging
import threading
import time
import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from PIL import ImageGrab

from arenamcp.action_planner import ActionPlan, ActionPlanner, ActionType, GameAction
from arenamcp.input_controller import ClickResult, InputController
from arenamcp.screen_mapper import FixedCoordinates, ScreenCoord, ScreenMapper

logger = logging.getLogger(__name__)


class AutopilotState(Enum):
    """Current state of the autopilot engine."""
    IDLE = "idle"
    PLANNING = "planning"
    PREVIEWING = "previewing"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class AutopilotConfig:
    """Configuration for autopilot behavior."""
    confirm_each_action: bool = False  # Per-action confirmation (legacy, slow)
    confirm_plan: bool = False  # Plan-level confirmation (legacy, slow)
    auto_execute_delay: float = 1.0  # Seconds before auto-executing (F1/F4 cancels)
    auto_pass_priority: bool = True
    auto_resolve: bool = True
    verify_after_action: bool = True
    verification_timeout: float = 4.0
    action_delay: float = 1.0
    post_action_delay: float = 1.5  # Delay after action to allow GRE to update
    planning_timeout: float = 15.0
    enable_vision_fallback: bool = True
    enable_tts_preview: bool = True
    dry_run: bool = False
    afk_mode: bool = False  # When True, auto-pass everything without LLM
    land_drop_mode: bool = False  # When True, auto-play one land per turn (no LLM)


class AutopilotEngine:
    """Core autopilot orchestration engine.

    Coordinates action planning, screen mapping, input control, and
    human confirmation to execute MTGA actions automatically.
    """

    def __init__(
        self,
        planner: ActionPlanner,
        mapper: ScreenMapper,
        controller: InputController,
        get_game_state: Callable[[], dict[str, Any]],
        config: Optional[AutopilotConfig] = None,
        speak_fn: Optional[Callable[[str, bool], None]] = None,
        ui_advice_fn: Optional[Callable[[str, str], None]] = None,
    ):
        """Initialize the autopilot engine.

        Args:
            planner: ActionPlanner for LLM-based action planning.
            mapper: ScreenMapper for coordinate calculations.
            controller: InputController for mouse/keyboard input.
            get_game_state: Callable that returns current game state dict.
            config: Optional autopilot configuration.
            speak_fn: Optional TTS function (text, blocking) for previewing actions.
            ui_advice_fn: Optional UI callback (text, label) for displaying actions.
        """
        self._planner = planner
        self._mapper = mapper
        self._controller = controller
        self._get_game_state = get_game_state
        self._config = config or AutopilotConfig()
        self._speak_fn = speak_fn
        self._ui_advice_fn = ui_advice_fn

        # State
        self._state = AutopilotState.IDLE
        self._current_plan = None
        self._current_action_idx = 0
        self._lock = threading.Lock()

        # Confirmation events
        self._confirm_event = threading.Event()
        self._skip_event = threading.Event()
        self._abort_event = threading.Event()

        # Statistics
        self._actions_executed = 0
        self._actions_skipped = 0
        self._plans_completed = 0
        self._consecutive_failed_verifications = 0

        # Land-drop dedup: track last turn we played a land to prevent
        # double-triggers when game state hasn't updated yet
        self._land_drop_last_turn: int = -1

    @property
    def afk_mode(self) -> bool:
        """Whether AFK mode is active."""
        return self._config.afk_mode

    def toggle_afk(self) -> bool:
        """Toggle AFK mode on/off. Returns new state."""
        self._config.afk_mode = not self._config.afk_mode
        status = "ON" if self._config.afk_mode else "OFF"
        logger.info(f"AFK mode toggled: {status}")
        self._notify("AFK", status)
        return self._config.afk_mode

    @property
    def land_drop_mode(self) -> bool:
        """Whether land-drop-only mode is active."""
        return self._config.land_drop_mode

    def toggle_land_drop(self) -> bool:
        """Toggle land-drop-only mode on/off. Returns new state."""
        self._config.land_drop_mode = not self._config.land_drop_mode
        status = "ON" if self._config.land_drop_mode else "OFF"
        logger.info(f"Land-drop mode toggled: {status}")
        self._notify("LAND_DROP", status)
        return self._config.land_drop_mode

    @property
    def state(self) -> AutopilotState:
        """Current autopilot state."""
        return self._state

    @property
    def current_plan(self) -> Optional[ActionPlan]:
        """Currently active action plan."""
        return self._current_plan

    @property
    def stats(self) -> dict[str, int]:
        """Execution statistics."""
        return {
            "executed": self._actions_executed,
            "skipped": self._actions_skipped,
            "plans": self._plans_completed,
        }

    def on_spacebar(self) -> None:
        """Handle spacebar press (confirm current action/plan)."""
        logger.info("Autopilot: spacebar pressed (confirm)")
        self._confirm_event.set()

    def on_escape(self) -> None:
        """Handle escape press (skip current action)."""
        logger.info("Autopilot: escape pressed (skip)")
        self._skip_event.set()

    def on_abort(self) -> None:
        """Handle abort (double-escape or F11 toggle off)."""
        logger.info("Autopilot: abort requested")
        self._abort_event.set()
        self._confirm_event.set()  # Unblock any waiting
        self._skip_event.set()

    def _clear_events(self) -> None:
        """Clear all confirmation events."""
        self._confirm_event.clear()
        self._skip_event.clear()
        self._abort_event.clear()

    def _wait_for_cancel(self, timeout: Optional[float] = None) -> str:
        """Countdown timer that auto-executes unless user cancels.

        The autopilot previews its plan, then auto-executes after a brief
        countdown. Pressing F1 or F4 during the countdown cancels execution.

        Args:
            timeout: Seconds to wait. Defaults to config.auto_execute_delay.

        Returns:
            "execute" if countdown expires (no user input),
            "cancel" if user presses F1 or F4,
            "abort" if abort event is set.
        """
        if timeout is None:
            timeout = self._config.auto_execute_delay

        self._confirm_event.clear()
        self._skip_event.clear()

        remaining = timeout
        while remaining > 0:
            if self._abort_event.is_set():
                return "abort"
            # F1 (confirm_event) or F4 (skip_event) = cancel
            if self._confirm_event.wait(timeout=0.05):
                logger.info("User cancelled auto-execute (F1)")
                return "cancel"
            if self._skip_event.is_set():
                logger.info("User cancelled auto-execute (F4)")
                return "cancel"
            remaining -= 0.05

        # Timeout expired with no user input → auto-execute
        return "execute"

    def _wait_for_confirmation(self, timeout: float = 60.0) -> str:
        """Legacy: wait for explicit user confirmation.

        Only used when confirm_plan or confirm_each_action is True.

        Returns:
            "confirm" if F1, "skip" if F4, "abort" if abort.
        """
        self._confirm_event.clear()
        self._skip_event.clear()

        while True:
            if self._abort_event.is_set():
                return "abort"
            if self._confirm_event.wait(timeout=0.1):
                return "confirm"
            if self._skip_event.is_set():
                return "skip"
            timeout -= 0.1
            if timeout <= 0:
                return "skip"

    def process_trigger(
        self,
        game_state: dict[str, Any],
        trigger: str,
    ) -> bool:
        """Main entry point from the coaching loop.

        Processes a game state trigger through the full autopilot pipeline:
        1. PLANNING: Generate action plan via LLM
        2. PREVIEWING: Display plan, wait for confirmation
        3. EXECUTING: Execute each action with per-action confirmation
        4. VERIFYING: Verify state changes after each action

        Args:
            game_state: Current game state dict.
            trigger: Trigger name (e.g., "new_turn", "combat_attackers").

        Returns:
            True if plan was fully executed, False otherwise.
        """
        if not self._lock.acquire(blocking=False):
            logger.debug(f"Autopilot: already processing a trigger, skipping {trigger}")
            return False

        try:
            if self._abort_event.is_set():
                self._state = AutopilotState.IDLE
                return False

            self._clear_events()

            # --- AFK MODE: auto-pass everything without LLM ---
            if self._config.afk_mode:
                return self._handle_afk(game_state, trigger)

            # --- LAND DROP MODE: auto-play one land per turn without LLM ---
            if self._config.land_drop_mode:
                return self._handle_land_drop(game_state, trigger)

            # --- Quick shortcuts: auto-pass/resolve without LLM ---
            # These save 5-15s by not calling the LLM for obvious actions.
            pending = game_state.get("pending_decision")
            has_decision = pending is not None and pending != "Action Required"
            turn = game_state.get("turn", {})
            local_seat = None
            for p in game_state.get("players", []):
                if p.get("is_local"):
                    local_seat = p.get("seat_id")
            is_my_turn = turn.get("active_player") == local_seat if local_seat else False

            # NEVER auto-pass when there's a pending decision (scry, discard, target, etc.)
            if not has_decision:
                # Get legal actions once to check if we can actually do anything
                legal = self._get_legal_actions(game_state)
                can_do_anything = legal and legal != ["Pass"] and not all("Wait" in a for a in legal)

                if self._config.auto_pass_priority and trigger == "priority_gained":
                    if not can_do_anything:
                        logger.info("Autopilot: auto-passing priority (no actions)")
                        if not self._config.dry_run:
                            self._controller.focus_mtga_window()
                            time.sleep(0.15)
                        self._exec_pass_priority()
                        return True

                if self._config.auto_resolve and trigger == "spell_resolved":
                    if not is_my_turn and not can_do_anything:
                        logger.info("Autopilot: auto-resolving (opponent's spell, no responses)")
                        if not self._config.dry_run:
                            self._controller.focus_mtga_window()
                            time.sleep(0.15)
                        self._exec_resolve()
                        return True

                # Auto-pass stack triggers with no instant-speed responses
                if trigger in ("stack_spell_yours", "stack_spell_opponent"):
                    if not can_do_anything:
                        logger.info(f"Autopilot: auto-passing {trigger} (no instant responses)")
                        if not self._config.dry_run:
                            self._controller.focus_mtga_window()
                            time.sleep(0.15)
                        self._exec_pass_priority()
                        return True

                # Auto-pass opponent's turn with no responses
                if trigger == "opponent_turn":
                    if not can_do_anything:
                        logger.info("Autopilot: auto-passing opponent turn (no responses)")
                        return True  # Just skip, don't click anything

            # --- 1. PLANNING ---
            self._state = AutopilotState.PLANNING
            self._notify("AUTOPILOT", f"Planning: {trigger}...")

            # Snapshot state before planning (for staleness check)
            pre_plan_turn = game_state.get("turn", {})
            pre_turn_num = pre_plan_turn.get("turn_number", 0)
            pre_phase = pre_plan_turn.get("phase", "")
            pre_active = pre_plan_turn.get("active_player", 0)

            legal_actions = self._get_legal_actions(game_state)
            decision_context = game_state.get("decision_context")

            plan = self._planner.plan_actions(
                game_state, trigger, legal_actions, decision_context
            )

            if not plan.actions:
                logger.info("Autopilot: planner returned no actions")
                self._state = AutopilotState.IDLE
                return False

            # --- STALENESS CHECK ---
            # Re-poll game state after planning (LLM call may take 5-15s).
            # If the game has moved on (different turn, phase, or active player),
            # discard the stale plan instead of executing outdated actions.
            try:
                fresh_state = self._get_game_state()
                fresh_turn = fresh_state.get("turn", {})
                stale = False

                if fresh_turn.get("turn_number", 0) != pre_turn_num:
                    logger.warning(f"STALE: turn advanced {pre_turn_num} → {fresh_turn.get('turn_number')}")
                    stale = True
                elif fresh_turn.get("active_player", 0) != pre_active:
                    logger.warning(f"STALE: active player changed {pre_active} → {fresh_turn.get('active_player')}")
                    stale = True
                elif fresh_turn.get("phase", "") != pre_phase:
                    # Lenient phase check: allow Main1 -> Main2 or Combat steps as long as it's still
                    # the same turn and player. BUT if a sorcery/land action was planned and we are
                    # now in Combat, it's stale.
                    is_sorcery_play = any(a.action_type in (ActionType.PLAY_LAND, ActionType.CAST_SPELL) for a in plan.actions)
                    now_combat = "Combat" in fresh_turn.get("phase", "")

                    if is_sorcery_play and now_combat:
                        logger.warning(f"STALE: phase changed {pre_phase} → {fresh_turn.get('phase')} (sorcery plan in combat)")
                        stale = True
                    else:
                        # For other changes (Main1->Main2, or combat step changes), we can try to proceed
                        # but we should update the game_state so coordinates are fresh.
                        logger.info(f"Phase changed {pre_phase} → {fresh_turn.get('phase')}, proceeding with caution")

                if stale:
                    self._notify("AUTOPILOT", "Plan discarded (game moved on)")
                    self._state = AutopilotState.IDLE
                    return False

                # Use the fresh state for execution (more accurate coordinates)
                game_state = fresh_state
            except Exception as e:
                logger.error(f"Staleness check failed: {e}")
                # Continue with original state if re-poll fails

            self._current_plan = plan
            self._current_action_idx = 0

            # --- 2. PREVIEWING (auto-execute countdown) ---
            self._state = AutopilotState.PREVIEWING
            plan_text = self._format_plan_preview(plan)

            self._notify("AUTOPILOT", plan_text)
            if self._config.enable_tts_preview and self._speak_fn:
                self._speak_fn(f"Plan: {plan.overall_strategy}", False)

            # Auto-execute countdown: executes after delay unless user cancels
            if self._config.confirm_plan:
                # Legacy mode: wait for explicit F1 confirm
                logger.info("Waiting for plan confirmation (F1)...")
                result = self._wait_for_confirmation()
                if result == "abort":
                    self._state = AutopilotState.IDLE
                    self._notify("AUTOPILOT", "Aborted")
                    return False
                if result == "skip":
                    self._state = AutopilotState.IDLE
                    self._actions_skipped += len(plan.actions)
                    self._notify("AUTOPILOT", "Plan skipped")
                    return False
            elif self._config.auto_execute_delay > 0:
                # New default: auto-execute after countdown, F1/F4 cancels
                delay = self._config.auto_execute_delay
                self._notify("AUTOPILOT", f"Executing in {delay:.1f}s... [F1/F4 to cancel]")
                result = self._wait_for_cancel(delay)
                if result == "abort":
                    self._state = AutopilotState.IDLE
                    self._notify("AUTOPILOT", "Aborted")
                    return False
                if result == "cancel":
                    self._state = AutopilotState.IDLE
                    self._actions_skipped += len(plan.actions)
                    self._notify("AUTOPILOT", "Plan cancelled by user")
                    return False
                # result == "execute" → proceed

            # --- PRE-EXECUTION STALENESS RECHECK ---
            # The countdown may have consumed up to 1s. Re-poll game state to
            # make sure the game hasn't moved on during that window.
            try:
                exec_state = self._get_game_state()
                exec_turn = exec_state.get("turn", {})
                if (exec_turn.get("turn_number", 0) != pre_turn_num
                        or exec_turn.get("active_player", 0) != pre_active):
                    logger.warning("STALE at execution time — game moved on during countdown")
                    self._notify("AUTOPILOT", "Plan discarded (game moved during countdown)")
                    self._state = AutopilotState.IDLE
                    return False
                game_state = exec_state  # Use freshest state
            except Exception as e:
                logger.error(f"Pre-execution recheck failed: {e}")

            # --- 3. EXECUTING ---
            self._state = AutopilotState.EXECUTING

            # Focus MTGA now — no more user input expected
            if not self._config.dry_run:
                self._controller.focus_mtga_window()
                time.sleep(0.15)

            for i, action in enumerate(plan.actions):
                if self._abort_event.is_set():
                    self._state = AutopilotState.IDLE
                    return False

                self._current_action_idx = i

                action_text = f"[{i+1}/{len(plan.actions)}] {action}"
                self._notify("AUTOPILOT", action_text)

                # Per-action staleness check: verify game hasn't advanced
                # between multi-step actions (e.g., declare attackers then done)
                if i > 0:
                    try:
                        step_state = self._get_game_state()
                        step_turn = step_state.get("turn", {})
                        if step_turn.get("turn_number", 0) != pre_turn_num:
                            logger.warning(f"STALE mid-execution: turn advanced at action {i+1}")
                            self._notify("AUTOPILOT", "Stopping: game advanced mid-plan")
                            self._state = AutopilotState.IDLE
                            return False
                        game_state = step_state
                    except Exception:
                        pass

                # Legacy per-action confirmation (only if explicitly enabled)
                if self._config.confirm_each_action:
                    result = self._wait_for_confirmation()
                    if result == "abort":
                        self._state = AutopilotState.IDLE
                        return False
                    if result == "skip":
                        self._actions_skipped += 1
                        continue

                # Snapshot state before action (for verification)
                pre_state = self._get_game_state() if self._config.verify_after_action else None

                # Execute
                click_result = self._execute_action(action, game_state)
                if not click_result.success:
                    logger.warning(f"Action failed: {click_result}")
                    self._notify("AUTOPILOT", f"FAILED: {click_result.error}")
                    continue

                self._actions_executed += 1

                # --- 4. VERIFYING ---
                if self._config.verify_after_action and pre_state:
                    self._state = AutopilotState.VERIFYING
                    verified = self._verify_action(action, pre_state)
                    if not verified:
                        logger.warning(f"Action verification failed for: {action}")
                        self._notify("AUTOPILOT", "Verification: state unchanged (may be OK)")
                        self._consecutive_failed_verifications += 1
                        
                        if self._consecutive_failed_verifications >= 3:
                            self._recover_stuck()
                    else:
                        self._consecutive_failed_verifications = 0

                # Delay between actions
                if i < len(plan.actions) - 1:
                    self._controller.wait(self._config.action_delay, "between actions")

            self._state = AutopilotState.IDLE
            self._plans_completed += 1
            self._notify("AUTOPILOT", f"Plan complete ({len(plan.actions)} actions)")
            return True
        finally:
            self._lock.release()

    def _handle_afk(self, game_state: dict[str, Any], trigger: str) -> bool:
        """Handle a trigger in AFK mode — auto-pass without LLM.

        AFK mode clicks pass/resolve/done for all priority decisions.
        For mandatory choices (mulligan, scry), picks the "safe default":
        - Mulligan: keep hand
        - Scry: scry to bottom
        - Declare Attackers/Blockers: skip (don't attack/block)
        - Choose Play/Draw: choose play
        - All other decisions: click Done/spacebar
        """
        pending = game_state.get("pending_decision")
        decision_context = game_state.get("decision_context") or {}
        dec_type = decision_context.get("type", "")

        # Mandatory decisions that need a specific click
        if pending:
            pending_lower = pending.lower() if isinstance(pending, str) else ""

            if "mulligan" in pending_lower:
                logger.info("AFK: keeping hand (mulligan)")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                return self._click_fixed("keep").success

            if "scry" in pending_lower:
                logger.info("AFK: scry to bottom")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                return self._click_fixed("scry_bottom").success

            # New decision types from expanded GRE handling
            if dec_type == "declare_attackers":
                logger.info("AFK: skipping attackers (click Done)")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                return self._click_fixed("done").success

            if dec_type == "declare_blockers":
                logger.info("AFK: skipping blockers (click Done)")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                return self._click_fixed("done").success

            if dec_type == "choose_starting_player":
                logger.info("AFK: choosing to play")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                # "Play" is typically the first option
                return self._click_fixed("pass").success

            if dec_type in (
                "assign_damage", "order_combat_damage", "pay_costs",
                "search", "distribution", "numeric_input",
                "select_replacement", "casting_time_options",
                "select_counters", "order_triggers",
                "select_n_group", "select_from_groups",
                "search_from_groups", "gather",
            ):
                logger.info(f"AFK: auto-accepting decision '{dec_type}' (click Done)")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                result = self._click_fixed("done")
                if result.success:
                    return True
                # Done didn't work, try spacebar
                self._controller.press_key("space", f"AFK: {dec_type} spacebar")
                return True

            # Unknown decision: try Done button, then spacebar
            if pending_lower and "mulligan" not in pending_lower and "scry" not in pending_lower:
                logger.warning(f"AFK: unknown decision '{pending}' - trying Done button")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                result = self._click_fixed("done")
                if result.success:
                    return True
                # Done didn't work, try spacebar
                self._controller.press_key("space", "AFK: unknown decision spacebar")
                return True

        # Everything else: click pass/resolve/done
        logger.info(f"AFK: passing ({trigger})")
        if not self._config.dry_run:
            self._controller.focus_mtga_window()
            time.sleep(0.15)
        return self._exec_pass_priority().success

    def _handle_land_drop(self, game_state: dict[str, Any], trigger: str) -> bool:
        """Handle a trigger in land-drop-only mode.

        Automatically plays one land per turn by dragging it from hand to
        the battlefield. No LLM is used. All other priority passes are
        auto-resolved so the game keeps moving.
        """
        turn = game_state.get("turn", {})
        phase = turn.get("phase", "")
        local_seat = None
        local_player = None
        for p in game_state.get("players", []):
            if p.get("is_local"):
                local_seat = p.get("seat_id")
                local_player = p

        is_my_turn = turn.get("active_player") == local_seat if local_seat else False
        is_main_phase = "Main" in phase
        stack = game_state.get("stack", [])
        is_stack_empty = len(stack) == 0
        lands_played = local_player.get("lands_played", 0) if local_player else 1
        turn_number = turn.get("turn_number", 0)

        # Check if we can play a land right now
        # Also guard against double-triggers: if we already dragged a land
        # this turn, skip (the server may not have confirmed lands_played yet)
        already_played_this_turn = self._land_drop_last_turn == turn_number
        if is_my_turn and is_main_phase and is_stack_empty and lands_played < 1 and not already_played_this_turn:
            hand = game_state.get("hand", [])
            land_card = None
            for card in hand:
                card_types = card.get("card_types", [])
                type_line = card.get("type_line", "")
                if any("Land" in ct for ct in card_types) or "Land" in type_line:
                    land_card = card
                    break

            if land_card:
                land_name = land_card.get("name", "Land")
                logger.info(f"LAND DROP: playing {land_name}")
                self._notify("LAND_DROP", f"Playing {land_name}")

                coord = self._mapper.get_card_in_hand_coord(
                    land_name, hand, game_state
                )
                if coord:
                    window_rect = self._mapper.window_rect
                    if not window_rect:
                        window_rect = self._mapper.refresh_window()
                    if window_rect:
                        if not self._config.dry_run:
                            self._controller.focus_mtga_window()
                            time.sleep(0.15)

                        from_x, from_y = coord.to_absolute(window_rect)
                        # Drag to center of battlefield (y ≈ 0.50)
                        target = ScreenCoord(0.50, 0.50, f"Battlefield: {land_name}")
                        to_x, to_y = target.to_absolute(window_rect)

                        result = self._controller.drag_card_from_hand(
                            from_x, from_y, to_x, to_y, land_name, window_rect
                        )
                        if result.success:
                            self._actions_executed += 1
                            self._land_drop_last_turn = turn_number
                            logger.info(f"LAND DROP: {land_name} played successfully")
                            return True
                        else:
                            logger.warning(f"LAND DROP: drag failed: {result.error}")
                else:
                    logger.warning(f"LAND DROP: could not map {land_name} in hand")

        # For everything else, auto-pass to keep the game moving
        pending = game_state.get("pending_decision")
        decision_context = game_state.get("decision_context") or {}
        dec_type = decision_context.get("type", "")

        if pending:
            pending_lower = pending.lower() if isinstance(pending, str) else ""
            if "mulligan" in pending_lower:
                logger.info("LAND DROP: keeping hand (mulligan)")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                return self._click_fixed("keep").success
            if "scry" in pending_lower:
                logger.info("LAND DROP: scry to bottom")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                return self._click_fixed("scry_bottom").success

            # New decision types: auto-pass combat, auto-accept others
            if dec_type in ("declare_attackers", "declare_blockers"):
                logger.info(f"LAND DROP: skipping {dec_type} (click Done)")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                return self._click_fixed("done").success

            if dec_type == "choose_starting_player":
                logger.info("LAND DROP: choosing to play")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                return self._click_fixed("pass").success

            if dec_type in (
                "assign_damage", "order_combat_damage", "pay_costs",
                "search", "distribution", "numeric_input",
                "select_replacement", "casting_time_options",
                "select_counters", "order_triggers",
                "select_n_group", "select_from_groups",
                "search_from_groups", "gather",
            ):
                logger.info(f"LAND DROP: auto-accepting decision '{dec_type}' (click Done)")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                result = self._click_fixed("done")
                if result.success:
                    return True
                self._controller.press_key("space", f"LAND DROP: {dec_type} spacebar")
                return True

            # Unknown decision: try Done button, then spacebar
            if pending_lower and "mulligan" not in pending_lower and "scry" not in pending_lower:
                logger.warning(f"LAND DROP: unknown decision '{pending}' - trying Done button")
                if not self._config.dry_run:
                    self._controller.focus_mtga_window()
                    time.sleep(0.15)
                result = self._click_fixed("done")
                if result.success:
                    return True
                self._controller.press_key("space", "LAND DROP: unknown decision spacebar")
                return True

        logger.info(f"LAND DROP: passing ({trigger})")
        if not self._config.dry_run:
            self._controller.focus_mtga_window()
            time.sleep(0.15)
        return self._exec_pass_priority().success

    def _get_legal_actions(self, game_state: dict[str, Any]) -> list[str]:
        """Get legal actions from the rules engine."""
        try:
            from arenamcp.rules_engine import RulesEngine
            return RulesEngine.get_legal_actions(game_state)
        except Exception as e:
            logger.error(f"Failed to get legal actions: {e}")
            return []

    def _format_plan_preview(self, plan: ActionPlan) -> str:
        """Format a plan for human-readable preview."""
        lines = [f"PLAN: {plan.overall_strategy}"]
        for i, action in enumerate(plan.actions, 1):
            lines.append(f"  {i}. {action}")
            if action.reasoning:
                lines.append(f"     ({action.reasoning})")
        delay = self._config.auto_execute_delay
        if delay > 0 and not self._config.confirm_plan:
            lines.append(f"[Auto-executing in {delay:.0f}s | F1/F4=cancel | F11=abort]")
        else:
            lines.append("[F1=confirm | F4=skip | F11=abort]")
        return "\n".join(lines)

    def _notify(self, label: str, text: str) -> None:
        """Send notification to UI."""
        logger.info(f"[{label}] {text}")
        if self._ui_advice_fn:
            try:
                self._ui_advice_fn(text, label)
            except Exception:
                pass

    def _get_vision_coord(self, card_name: str) -> Optional[ScreenCoord]:
        """Capture screenshot and use vision to find a card."""
        try:
            # Capture MTGA window area
            window_rect = self._mapper.window_rect
            if not window_rect:
                window_rect = self._mapper.refresh_window()
            if not window_rect:
                return None

            left, top, width, height = window_rect
            # Grab slightly larger area or full screen if needed, 
            # but usually client area is best
            screenshot = ImageGrab.grab(bbox=(left, top, left+width, top+height))
            
            # Convert to PNG bytes
            buf = io.BytesIO()
            screenshot.save(buf, format='PNG')
            png_bytes = buf.getvalue()
            
            # Use the planner's backend for vision
            backend = self._planner._backend
            return self._mapper.get_card_coord_via_vision(card_name, png_bytes, backend)
        except Exception as e:
            logger.error(f"Failed to get vision coord: {e}")
            return None

    def _recover_stuck(self) -> None:
        """Attempt to recover from a stuck state (UI prompts, dialogs, etc)."""
        self._notify("AUTOPILOT", "STUCK DETECTED: Attempting recovery...")

        # 1. Re-poll state to see if there's a pending decision we can re-plan for
        try:
            fresh_state = self._get_game_state()
            pending = fresh_state.get("pending_decision")
            if pending and pending != "Action Required":
                logger.info(f"Stuck recovery: found pending decision '{pending}', re-planning")
                self._notify("AUTOPILOT", f"Re-planning for: {pending}")
                legal = self._get_legal_actions(fresh_state)
                plan = self._planner.plan_actions(fresh_state, "decision_required", legal)
                if plan.actions:
                    for action in plan.actions:
                        self._execute_action(action, fresh_state)
                        time.sleep(self._config.action_delay)
                    self._consecutive_failed_verifications = 0
                    return
        except Exception as e:
            logger.error(f"Stuck recovery re-plan failed: {e}")

        # 2. Try common dismissal keys
        logger.warning("Stuck recovery: sending Escape and Spacebar")
        self._controller.focus_mtga_window()
        time.sleep(0.2)
        self._controller.press_key("escape", "Dismissing dialog")
        time.sleep(0.5)
        self._controller.press_key("space", "Confirming priority")

        # 3. Vision analysis of the stuck state
        logger.info("Stuck recovery: Analyzing screen via vision")
        coord = self._get_vision_coord("Blocking UI Prompt")
        if coord:
            logger.info(f"Vision suggests stuck UI element at {coord}")
            abs_x, abs_y = coord.to_absolute(self._mapper.window_rect)
            self._controller.click(abs_x, abs_y, "Dismissing via vision")

        self._consecutive_failed_verifications = 0

    # --- Action Execution Handlers ---

    def _execute_action(
        self, action: GameAction, game_state: dict[str, Any]
    ) -> ClickResult:
        """Route an action to the appropriate execution handler.

        Args:
            action: The GameAction to execute.
            game_state: Current game state for context.

        Returns:
            ClickResult from the execution.
        """
        handlers = {
            ActionType.PASS_PRIORITY: self._exec_pass_priority,
            ActionType.RESOLVE: self._exec_resolve,
            ActionType.CLICK_BUTTON: lambda: self._exec_click_button(action),
            ActionType.PLAY_LAND: lambda: self._exec_play_card(action, game_state),
            ActionType.CAST_SPELL: lambda: self._exec_play_card(action, game_state),
            ActionType.ACTIVATE_ABILITY: lambda: self._exec_activate_ability(action, game_state),
            ActionType.DECLARE_ATTACKERS: lambda: self._exec_declare_attackers(action, game_state),
            ActionType.DECLARE_BLOCKERS: lambda: self._exec_declare_blockers(action, game_state),
            ActionType.SELECT_TARGET: lambda: self._exec_select_target(action, game_state),
            ActionType.SELECT_N: lambda: self._exec_select_n(action, game_state),
            ActionType.MODAL_CHOICE: lambda: self._exec_modal_choice(action, game_state),
            ActionType.MULLIGAN_KEEP: lambda: self._exec_mulligan(keep=True),
            ActionType.MULLIGAN_MULL: lambda: self._exec_mulligan(keep=False),
            ActionType.DRAFT_PICK: lambda: self._exec_draft_pick(action, game_state),
            ActionType.ORDER_BLOCKERS: lambda: self._exec_order_blockers(action, game_state),
            # New decision types — most resolve via Done/pass after LLM selection
            ActionType.ASSIGN_DAMAGE: lambda: self._exec_done_action("assign_damage"),
            ActionType.ORDER_COMBAT_DAMAGE: lambda: self._exec_done_action("order_combat_damage"),
            ActionType.PAY_COSTS: lambda: self._exec_done_action("pay_costs"),
            ActionType.SEARCH_LIBRARY: lambda: self._exec_select_n(action, game_state),
            ActionType.DISTRIBUTE: lambda: self._exec_done_action("distribute"),
            ActionType.NUMERIC_INPUT: lambda: self._exec_done_action("numeric_input"),
            ActionType.CHOOSE_STARTING_PLAYER: lambda: self._exec_choose_play_draw(action),
            ActionType.SELECT_REPLACEMENT: lambda: self._exec_done_action("select_replacement"),
            ActionType.SELECT_COUNTERS: lambda: self._exec_select_n(action, game_state),
            ActionType.CASTING_OPTIONS: lambda: self._exec_modal_choice(action, game_state),
            ActionType.ORDER_TRIGGERS: lambda: self._exec_done_action("order_triggers"),
        }

        handler = handlers.get(action.action_type)
        if not handler:
            return ClickResult(False, 0, 0, str(action), f"No handler for {action.action_type}")

        return handler()

    def _click_fixed(self, name: str) -> ClickResult:
        """Click a fixed-position button by name."""
        coord = self._mapper.get_button_coord(name)
        if not coord:
            return ClickResult(False, 0, 0, name, f"Unknown button: {name}")

        window_rect = self._mapper.window_rect
        if not window_rect:
            window_rect = self._mapper.refresh_window()
        if not window_rect:
            return ClickResult(False, 0, 0, name, "MTGA window not found")

        abs_x, abs_y = coord.to_absolute(window_rect)
        return self._controller.click(abs_x, abs_y, coord.description, window_rect)

    def _exec_pass_priority(self) -> ClickResult:
        """Click the pass/resolve button."""
        return self._click_fixed("pass")

    def _exec_resolve(self) -> ClickResult:
        """Click the resolve button."""
        return self._click_fixed("resolve")

    def _exec_click_button(self, action: GameAction) -> ClickResult:
        """Click a named button."""
        button_name = action.card_name.lower().replace(" ", "_")
        # Fallback for common MTGA action buttons that might be named differently by the LLM
        if button_name in ("next", "attack", "all_attack", "done", "no_attacks", "no_blocks"):
            return self._click_fixed("pass") # They all share the same spot
        return self._click_fixed(button_name)

    def _exec_play_card(
        self, action: GameAction, game_state: dict[str, Any]
    ) -> ClickResult:
        """Play a card from hand (land or spell).

        Lands are dragged from hand to the battlefield land row (y ≈ 0.75)
        because MTGA requires a drag gesture to play them. Spells are
        clicked normally (MTGA auto-casts on click).
        """
        hand = game_state.get("hand", [])
        hand_names = [c.get("name", "???") for c in hand]
        logger.info(
            f"_exec_play_card: looking for '{action.card_name}' in hand "
            f"({len(hand)} cards): {hand_names}"
        )
        coord = self._mapper.get_card_in_hand_coord(action.card_name, hand, game_state)

        if not coord:
            # Vision fallback
            if self._config.enable_vision_fallback:
                logger.info(f"Trying vision fallback for '{action.card_name}'")
                coord = self._get_vision_coord(action.card_name)
            
            if not coord:
                return ClickResult(False, 0, 0, action.card_name, "Card not found in hand (Heuristic & Vision failed)")

        window_rect = self._mapper.window_rect
        if not window_rect:
            window_rect = self._mapper.refresh_window()
        if not window_rect:
            return ClickResult(False, 0, 0, action.card_name, "MTGA window not found")

        abs_x, abs_y = coord.to_absolute(window_rect)

        # Lands and Spells: drag from hand to battlefield center
        if action.action_type in (ActionType.PLAY_LAND, ActionType.CAST_SPELL):
            target = ScreenCoord(0.50, 0.50, f"Battlefield: {action.card_name}")
            to_x, to_y = target.to_absolute(window_rect)
            return self._controller.drag_card_from_hand(
                abs_x, abs_y, to_x, to_y, action.card_name, window_rect
            )

        # Abilities/Other: click to cast
        return self._controller.click_card_in_hand(
            abs_x, abs_y, action.card_name, window_rect
        )

    def _exec_activate_ability(
        self, action: GameAction, game_state: dict[str, Any]
    ) -> ClickResult:
        """Click a permanent on the battlefield to activate its ability."""
        battlefield = game_state.get("battlefield", [])
        local_seat = None
        for p in game_state.get("players", []):
            if p.get("is_local"):
                local_seat = p.get("seat_id")

        if not local_seat:
            return ClickResult(False, 0, 0, action.card_name, "Local seat not found")

        coord = self._mapper.get_permanent_coord(
            action.card_name, None, battlefield, local_seat, local_seat
        )

        if not coord:
            # Vision fallback
            if self._config.enable_vision_fallback:
                logger.info(f"Trying vision fallback for board permanent '{action.card_name}'")
                coord = self._get_vision_coord(action.card_name)
            
            if not coord:
                return ClickResult(False, 0, 0, action.card_name, "Permanent not found on battlefield (Heuristic & Vision failed)")

        window_rect = self._mapper.window_rect
        if not window_rect:
            window_rect = self._mapper.refresh_window()
        if not window_rect:
            return ClickResult(False, 0, 0, action.card_name, "MTGA window not found")

        abs_x, abs_y = coord.to_absolute(window_rect)
        return self._controller.click(abs_x, abs_y, f"Activate: {action.card_name}", window_rect)

    def _exec_declare_attackers(
        self, action: GameAction, game_state: dict[str, Any]
    ) -> ClickResult:
        """Click each attacking creature, then click Done."""
        battlefield = game_state.get("battlefield", [])
        local_seat = None
        for p in game_state.get("players", []):
            if p.get("is_local"):
                local_seat = p.get("seat_id")

        if not local_seat:
            return ClickResult(False, 0, 0, "attackers", "Local seat not found")

        window_rect = self._mapper.window_rect
        if not window_rect:
            window_rect = self._mapper.refresh_window()
        if not window_rect:
            return ClickResult(False, 0, 0, "attackers", "MTGA window not found")

        last_result = ClickResult(True, 0, 0, "attackers")

        for attacker_name in action.attacker_names:
            coord = self._mapper.get_permanent_coord(
                attacker_name, None, battlefield, local_seat, local_seat
            )
            if coord:
                abs_x, abs_y = coord.to_absolute(window_rect)
                result = self._controller.click(
                    abs_x, abs_y, f"Attack: {attacker_name}", window_rect
                )
                if not result.success:
                    logger.warning(f"Failed to click attacker {attacker_name}")
                last_result = result
                self._controller.wait(self._config.action_delay, "between attacker clicks")

        # Click Done
        self._controller.wait(0.3, "before Done")
        done_result = self._click_fixed("done")
        return done_result if done_result.success else last_result

    def _exec_declare_blockers(
        self, action: GameAction, game_state: dict[str, Any]
    ) -> ClickResult:
        """Click blocker, then click attacker it should block, then Done."""
        battlefield = game_state.get("battlefield", [])
        local_seat = None
        opp_seat = None
        for p in game_state.get("players", []):
            if p.get("is_local"):
                local_seat = p.get("seat_id")
            else:
                opp_seat = p.get("seat_id")

        if not local_seat or not opp_seat:
            return ClickResult(False, 0, 0, "blockers", "Seat info not found")

        window_rect = self._mapper.window_rect
        if not window_rect:
            window_rect = self._mapper.refresh_window()
        if not window_rect:
            return ClickResult(False, 0, 0, "blockers", "MTGA window not found")

        last_result = ClickResult(True, 0, 0, "blockers")

        for blocker_name, attacker_name in action.blocker_assignments.items():
            # Click the blocker (our creature)
            blocker_coord = self._mapper.get_permanent_coord(
                blocker_name, None, battlefield, local_seat, local_seat
            )
            if blocker_coord:
                bx, by = blocker_coord.to_absolute(window_rect)
                self._controller.click(bx, by, f"Blocker: {blocker_name}", window_rect)
                self._controller.wait(0.2, "blocker selected")

            # Click the attacker (opponent's creature)
            attacker_coord = self._mapper.get_permanent_coord(
                attacker_name, None, battlefield, opp_seat, local_seat
            )
            if attacker_coord:
                ax, ay = attacker_coord.to_absolute(window_rect)
                result = self._controller.click(
                    ax, ay, f"Block {attacker_name} with {blocker_name}", window_rect
                )
                last_result = result
                self._controller.wait(self._config.action_delay, "between block assignments")

        # Click Done
        self._controller.wait(0.3, "before Done")
        done_result = self._click_fixed("done")
        return done_result if done_result.success else last_result

    def _exec_select_target(
        self, action: GameAction, game_state: dict[str, Any]
    ) -> ClickResult:
        """Click on a target permanent or player."""
        if not action.target_names:
            return ClickResult(False, 0, 0, "target", "No target specified")

        target_name = action.target_names[0]
        battlefield = game_state.get("battlefield", [])

        # Try to find target on battlefield
        local_seat = None
        opp_seat = None
        for p in game_state.get("players", []):
            if p.get("is_local"):
                local_seat = p.get("seat_id")
            else:
                opp_seat = p.get("seat_id")

        window_rect = self._mapper.window_rect
        if not window_rect:
            window_rect = self._mapper.refresh_window()
        if not window_rect:
            return ClickResult(False, 0, 0, target_name, "MTGA window not found")

        # Search both sides of the battlefield
        for owner in [opp_seat, local_seat]:
            if owner is None:
                continue
            coord = self._mapper.get_permanent_coord(
                target_name, None, battlefield, owner, local_seat
            )
            if coord:
                abs_x, abs_y = coord.to_absolute(window_rect)
                return self._controller.click(
                    abs_x, abs_y, f"Target: {target_name}", window_rect
                )

        return ClickResult(False, 0, 0, target_name, "Target not found on battlefield")

    def _exec_select_n(
        self, action: GameAction, game_state: dict[str, Any]
    ) -> ClickResult:
        """Handle scry or multi-select UI (select N cards)."""
        # Scry: top or bottom
        if action.scry_position:
            button = "scry_top" if action.scry_position == "top" else "scry_bottom"
            return self._click_fixed(button)

        # Multi-select: click each card then Done
        window_rect = self._mapper.window_rect
        if not window_rect:
            window_rect = self._mapper.refresh_window()
        if not window_rect:
            return ClickResult(False, 0, 0, "select_n", "MTGA window not found")

        last_result = ClickResult(True, 0, 0, "select_n")

        for i, card_name in enumerate(action.select_card_names):
            coord = self._mapper.get_option_coord(
                i, len(action.select_card_names), "select"
            )
            if coord:
                abs_x, abs_y = coord.to_absolute(window_rect)
                result = self._controller.click(
                    abs_x, abs_y, f"Select: {card_name}", window_rect
                )
                last_result = result
                self._controller.wait(0.2, "between selections")

        # Click Done
        self._controller.wait(0.3, "before Done")
        done_result = self._click_fixed("done")
        return done_result if done_result.success else last_result

    def _exec_modal_choice(
        self, action: GameAction, game_state: dict[str, Any]
    ) -> ClickResult:
        """Click a modal choice option."""
        # Determine total options from decision context
        decision = game_state.get("decision_context", {})
        total_options = decision.get("total_options", 2)

        coord = self._mapper.get_option_coord(
            action.modal_index, total_options, "modal"
        )
        if not coord:
            return ClickResult(False, 0, 0, "modal", "Cannot determine option position")

        window_rect = self._mapper.window_rect
        if not window_rect:
            window_rect = self._mapper.refresh_window()
        if not window_rect:
            return ClickResult(False, 0, 0, "modal", "MTGA window not found")

        abs_x, abs_y = coord.to_absolute(window_rect)
        return self._controller.click(
            abs_x, abs_y, f"Modal option {action.modal_index}", window_rect
        )

    def _exec_mulligan(self, keep: bool) -> ClickResult:
        """Click Keep or Mulligan button."""
        return self._click_fixed("keep" if keep else "mulligan")

    def _exec_draft_pick(
        self, action: GameAction, game_state: dict[str, Any]
    ) -> ClickResult:
        """Double-click a draft card to pick it."""
        # Try positional first, then vision fallback
        # For draft, we need pack info
        pack = game_state.get("draft_pack", {})
        cards = pack.get("cards", [])
        pack_size = len(cards)

        # Find card index
        card_idx = None
        for i, card in enumerate(cards):
            if card.get("name", "").lower() == action.card_name.lower():
                card_idx = i
                break

        if card_idx is None:
            return ClickResult(False, 0, 0, action.card_name, "Card not found in draft pack")

        coord = self._mapper.get_draft_card_coord(card_idx, pack_size)
        if not coord:
            return ClickResult(False, 0, 0, action.card_name, "Cannot calculate draft position")

        window_rect = self._mapper.window_rect
        if not window_rect:
            window_rect = self._mapper.refresh_window()
        if not window_rect:
            return ClickResult(False, 0, 0, action.card_name, "MTGA window not found")

        abs_x, abs_y = coord.to_absolute(window_rect)
        return self._controller.double_click(
            abs_x, abs_y, f"Draft pick: {action.card_name}", window_rect
        )

    def _exec_order_blockers(
        self, action: GameAction, game_state: dict[str, Any]
    ) -> ClickResult:
        """Order blockers by dragging (rarely needed)."""
        # Blocker ordering uses drag to reorder. For now, just click Done
        # since MTGA defaults to a reasonable order.
        logger.info("Blocker ordering: using default order (click Done)")
        return self._click_fixed("done")

    def _exec_done_action(self, decision_name: str) -> ClickResult:
        """Generic handler for decisions that just need a Done click after MTGA auto-selects."""
        logger.info(f"{decision_name}: accepting default / clicking Done")
        result = self._click_fixed("done")
        if not result.success:
            # Fallback: try spacebar
            self._controller.press_key("space", f"{decision_name}: spacebar fallback")
            return ClickResult(True, 0, 0, decision_name, "spacebar fallback")
        return result

    def _exec_choose_play_draw(self, action: GameAction) -> ClickResult:
        """Handle choose starting player (play or draw)."""
        choice = action.play_or_draw.lower() if action.play_or_draw else "play"
        logger.info(f"Choosing to {choice}")
        # In MTGA, "Play" is the first option button, "Draw" is second
        # Both typically resolve via the pass/done area or modal options
        if choice == "draw":
            # Try clicking the second option
            coord = self._mapper.get_option_coord(1, 2, "modal")
            if coord:
                window_rect = self._mapper.window_rect
                if not window_rect:
                    window_rect = self._mapper.refresh_window()
                if window_rect:
                    abs_x, abs_y = coord.to_absolute(window_rect)
                    return self._controller.click(abs_x, abs_y, "Choose: Draw", window_rect)
        # Default: "Play" = first option
        coord = self._mapper.get_option_coord(0, 2, "modal")
        if coord:
            window_rect = self._mapper.window_rect
            if not window_rect:
                window_rect = self._mapper.refresh_window()
            if window_rect:
                abs_x, abs_y = coord.to_absolute(window_rect)
                return self._controller.click(abs_x, abs_y, "Choose: Play", window_rect)
        # Last fallback
        return self._click_fixed("pass")

    # --- State Verification ---

    def _verify_action(
        self, action: GameAction, pre_state: dict[str, Any]
    ) -> bool:
        """Verify that an action caused the expected state change.

        Polls game state for up to verification_timeout seconds.

        Args:
            action: The action that was executed.
            pre_state: Game state snapshot from before the action.

        Returns:
            True if state changed as expected.
        """
        # Initial delay to give MTGA time to process the click and update logs
        time.sleep(self._config.post_action_delay)

        deadline = time.time() + self._config.verification_timeout
        poll_interval = 0.3

        card_name = action.card_name.lower() if action.card_name else ""

        while time.time() < deadline:
            try:
                post_state = self._get_game_state()

                # 1. Global state changes (Turn, Phase, Priority)
                pre_turn = pre_state.get("turn", {})
                post_turn = post_state.get("turn", {})

                if (
                    post_turn.get("phase") != pre_turn.get("phase")
                    or post_turn.get("step") != pre_turn.get("step")
                    or post_turn.get("priority_player") != pre_turn.get("priority_player")
                    or post_turn.get("turn_number") != pre_turn.get("turn_number")
                ):
                    logger.info(f"Action verified: global state changed ({pre_turn.get('phase')} -> {post_turn.get('phase')})")
                    return True

                # 2. Specific Action Verification
                if action.action_type in (ActionType.PLAY_LAND, ActionType.CAST_SPELL):
                    # Card should no longer be in hand, or should be on stack/battlefield/GY
                    pre_hand = [c.get("instance_id") for c in pre_state.get("hand", [])]
                    post_hand = [c.get("instance_id") for c in post_state.get("hand", [])]
                    
                    if len(post_hand) < len(pre_hand):
                        logger.info(f"Action verified: card '{action.card_name}' left hand")
                        return True
                    
                    # Check if card appeared on battlefield
                    post_bf = [c.get("name", "").lower() for c in post_state.get("battlefield", [])]
                    if any(card_name in name for name in post_bf):
                        # This is a bit weak if the card was already there, but better than nothing
                        # Ideally we'd track instance_id movement
                        pass

                if action.action_type == ActionType.DECLARE_ATTACKERS:
                    # Check if any creatures are now attacking that weren't before
                    pre_atk = sum(1 for c in pre_state.get("battlefield", []) if c.get("is_attacking"))
                    post_atk = sum(1 for c in post_state.get("battlefield", []) if c.get("is_attacking"))
                    if post_atk > pre_atk or (post_atk == 0 and pre_atk > 0): # attacking finished
                        logger.info("Action verified: attackers declared")
                        return True

                # 3. Generic fallback: did ANYTHING change?
                # Hand size changed
                if len(post_state.get("hand", [])) != len(pre_state.get("hand", [])):
                    logger.info("Action verified: hand size changed")
                    return True

                # Battlefield count changed
                if len(post_state.get("battlefield", [])) != len(pre_state.get("battlefield", [])):
                    logger.info("Action verified: battlefield count changed")
                    return True
                
                # Stack size changed
                if len(post_state.get("stack", [])) != len(pre_state.get("stack", [])):
                    logger.info("Action verified: stack changed")
                    return True

            except Exception as e:
                logger.error(f"Verification poll error: {e}")

            time.sleep(poll_interval)

        logger.warning(f"Action verification timed out after {self._config.verification_timeout}s")
        return False
