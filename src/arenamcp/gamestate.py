"""MTGA game state tracking from parsed log events.

This module provides the GameState class that maintains a complete
snapshot of the current game state from parsed MTGA log events.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ZoneType(Enum):
    """Zone types in MTGA."""
    BATTLEFIELD = "ZoneType_Battlefield"
    HAND = "ZoneType_Hand"
    GRAVEYARD = "ZoneType_Graveyard"
    EXILE = "ZoneType_Exile"
    LIBRARY = "ZoneType_Library"
    STACK = "ZoneType_Stack"
    LIMBO = "ZoneType_Limbo"
    COMMAND = "ZoneType_Command"
    PENDING = "ZoneType_Pending"
    REVEALED = "ZoneType_Revealed"
    UNKNOWN = "Unknown"


@dataclass
class GameObject:
    """A game object (card, token, ability, etc.) in the game."""
    instance_id: int
    grp_id: int
    zone_id: int
    owner_seat_id: int
    controller_seat_id: Optional[int] = None
    visibility: Optional[str] = None
    card_types: list[str] = field(default_factory=list)
    subtypes: list[str] = field(default_factory=list)
    power: Optional[int] = None
    toughness: Optional[int] = None
    is_tapped: bool = False
    # For abilities: instance_id of the source permanent
    # For abilities: instance_id of the source permanent
    parent_instance_id: Optional[int] = None
    # For summoning sickness tracking
    turn_entered_battlefield: int = -1
    # Combat status
    is_attacking: bool = False
    is_blocking: bool = False
    def to_dict(self) -> dict:
        """Convert to simple dict for snapshot serialization."""
        return {
            "instance_id": self.instance_id,
            "grp_id": self.grp_id,
            "zone_id": self.zone_id,
            "owner_seat_id": self.owner_seat_id,
            "controller_seat_id": self.controller_seat_id,
            "visibility": self.visibility,
            "card_types": self.card_types,
            "subtypes": self.subtypes,
            "power": self.power,
            "toughness": self.toughness,
            "is_tapped": self.is_tapped,
            "is_tapped": self.is_tapped,
            "turn_entered_battlefield": self.turn_entered_battlefield,
            "is_attacking": self.is_attacking,
            "is_blocking": self.is_blocking,
        }


@dataclass
class Zone:
    """A game zone (battlefield, hand, etc.)."""
    zone_id: int
    zone_type: ZoneType
    owner_seat_id: Optional[int] = None
    object_instance_ids: list[int] = field(default_factory=list)
    def to_dict(self) -> dict:
        """Convert to simple dict for snapshot."""
        return {
            "zone_id": self.zone_id,
            "zone_type": self.zone_type.name, # Enum to string
            "owner_seat_id": self.owner_seat_id,
            "object_instance_ids": self.object_instance_ids,
        }


@dataclass
class Player:
    """A player in the game."""
    seat_id: int
    life_total: int = 20
    lands_played: int = 0
    mana_pool: dict[str, int] = field(default_factory=dict)
    def to_dict(self) -> dict:
        return {
            "seat_id": self.seat_id,
            "life_total": self.life_total,
            "lands_played": self.lands_played,
            "mana_pool": self.mana_pool,
        }


@dataclass
class TurnInfo:
    """Current turn information."""
    turn_number: int = 0
    active_player: int = 0
    priority_player: int = 0
    phase: str = ""
    step: str = ""
    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "active_player": self.active_player,
            "priority_player": self.priority_player,
            "phase": self.phase,
            "step": self.step,
        }


class GameState:
    """Maintains complete game state from parsed MTGA log events.

    This class tracks zones, game objects, players, and turn information
    as they are updated via GameStateMessage events from the log parser.
    """

    def __init__(self) -> None:
        """Initialize empty game state."""
        # Core state dictionaries
        self.zones: dict[int, Zone] = {}
        self.game_objects: dict[int, GameObject] = {}
        self.players: dict[int, Player] = {}
        self.turn_info: TurnInfo = TurnInfo()

        # Local player tracking
        self.local_seat_id: Optional[int] = None
        # Source of the seat ID: 0=None, 1=Inferred, 2=System (MatchCreated), 3=User (F8)
        self._seat_source: int = 0
        
        # Opponent card history tracking
        self.played_cards: dict[int, list[int]] = {}  # seat_id -> list of grp_ids
        self._seen_instances: set[int] = set()  # Track instances to avoid double-counting

        # Combat step tracking - captures steps that happen between polls
        # This ensures fast combat phases aren't missed
        self._pending_combat_steps: list[str] = []

        # Decision tracking (for Mulligan advice, etc.)
        self.pending_decision: Optional[str] = None  # e.g. "Mulligan"
        self.decision_seat_id: Optional[int] = None
        # PHASE 1: Enhanced decision context
        self.decision_context: Optional[dict] = None  # Rich context: source_card, options, etc.
        self.decision_timestamp: float = 0  # Track when decision was set
        
        # Match tracking (to avoid stale state across matches)
        self.match_id: Optional[str] = None

    def reset(self) -> None:
        """Reset the game state to essentially empty (for new match)."""
        logger.info("Resetting GameState for new match")
        self.zones.clear()
        self.game_objects.clear()
        self.players.clear()
        self.turn_info = TurnInfo()
        
        # Reset local player seat logic (but keep User/System if we want persistence?)
        # Actually for a NEW match, System seat ID will change, so we should clear it.
        # But User override (F8) might be intended to persist? 
        # Usually seat ID changes every match, so User override might be wrong for next match.
        # Better to reset seat source to 0 or keep it if it's 3?
        # Let's trust reset_local_player(force=False) logic which keeps User/System.
        # But wait, if System ID is from OLD match, it's invalid.
        # So for a full match reset, we probably want to clear System source too.
        # Always reset seat ID for a new match. 
        # Even if user manually set it (Source 3) in previous match, 
        # it is likely invalid for the new match.
        self.local_seat_id = None
        self._seat_source = 0
            
        self.played_cards.clear()
        self._seen_instances.clear()
        self._pending_combat_steps.clear()
        self.pending_decision = None
        self.decision_seat_id = None
        self.decision_context = None
        self.decision_timestamp = 0

    # Backward compatibility for _seat_manually_set
    @property
    def _seat_manually_set(self) -> bool:
        return self._seat_source == 3

    @property
    def opponent_seat_id(self) -> Optional[int]:
        """Get the opponent's seat ID (the seat that isn't local player)."""
        if self.local_seat_id is None:
            return None
        # In 2-player games, opponent is the other seat
        for seat_id in self.players:
            if seat_id != self.local_seat_id:
                return seat_id
        return None

    def get_objects_in_zone(
        self,
        zone_type: ZoneType,
        owner: Optional[int] = None
    ) -> list[GameObject]:
        """Get all game objects in zones of a specific type.

        Args:
            zone_type: The type of zone to query.
            owner: Optional seat ID to filter by owner.

        Returns:
            List of GameObjects in matching zones.
        """
        result = []
        for zone in self.zones.values():
            if zone.zone_type != zone_type:
                continue
            if owner is not None and zone.owner_seat_id != owner:
                continue
            for instance_id in zone.object_instance_ids:
                if instance_id in self.game_objects:
                    result.append(self.game_objects[instance_id])
        return result

    def get_player_objects(self, seat_id: int) -> list[GameObject]:
        """Get all game objects owned by a specific player.

        Args:
            seat_id: The player's seat ID.

        Returns:
            List of GameObjects owned by the player.
        """
        return [
            obj for obj in self.game_objects.values()
            if obj.owner_seat_id == seat_id
        ]

    @property
    def battlefield(self) -> list[GameObject]:
        """Get all objects on the battlefield."""
        return self.get_objects_in_zone(ZoneType.BATTLEFIELD)

    @property
    def hand(self) -> list[GameObject]:
        """Get objects in all hands (filtered by local player if set)."""
        if self.local_seat_id is not None:
            # First try zone-based filter
            result = self.get_objects_in_zone(ZoneType.HAND, self.local_seat_id)
            if result:
                return result
            # Fallback: get all hand cards where the card's owner matches local player
            # (handles case where zone.owner_seat_id is None but card has owner set)
            all_hand = self.get_objects_in_zone(ZoneType.HAND)
            return [obj for obj in all_hand if obj.owner_seat_id == self.local_seat_id]
        return self.get_objects_in_zone(ZoneType.HAND)

    @property
    def graveyard(self) -> list[GameObject]:
        """Get all objects in graveyards."""
        return self.get_objects_in_zone(ZoneType.GRAVEYARD)

    @property
    def stack(self) -> list[GameObject]:
        """Get all objects on the stack."""
        return self.get_objects_in_zone(ZoneType.STACK)

    def get_opponent_played_cards(self) -> list[int]:
        """Get list of grp_ids of cards opponent has revealed.

        Returns:
            List of grp_ids (arena card IDs) that opponent has played.
        """
        if self.opponent_seat_id is None:
            return []
        return self.played_cards.get(self.opponent_seat_id, [])

    def get_pending_combat_steps(self) -> list[dict]:
        """Get combat steps that occurred since last check.

        Returns list of dicts with 'step', 'active_player', 'turn' keys.
        This allows catching fast combat phases that happen between polls.
        """
        return self._pending_combat_steps.copy()

    def get_snapshot(self) -> dict:
        """Get a complete, serializable snapshot of the current game state.
        
        This 'flattens' the state into a single JSON-ready dictionary, 
        serving as the 'State Anchor' for the middleware.
        """
        # Resolve owners
        local_player = self.players.get(self.local_seat_id) if self.local_seat_id else None
        opponent_seat = self.opponent_seat_id
        
        # Enrich objects with card names for TUI display
        def enrich_obj(obj):
            """Add card name to object dict for TUI."""
            data = obj.to_dict()
            # Lazy import to avoid circular dependency
            from arenamcp import server
            card_info = server.get_card_info(obj.grp_id)
            data["name"] = card_info.get("name", f"Unknown ({obj.grp_id})")
            data["type_line"] = card_info.get("type_line", "")
            return data
        
        # Serialize zones of interest
        # We group by meaningful logical zones rather than just raw zone IDs
        
        # Manually serialize players to include is_local flag for TUI
        players_list = []
        for p in self.players.values():
            p_dict = p.to_dict()
            p_dict["is_local"] = (p.seat_id == self.local_seat_id)
            players_list.append(p_dict)
            
        snapshot = {
            "match_id": self.match_id,
            "local_seat_id": self.local_seat_id,
            "opponent_seat_id": opponent_seat,
            "turn_info": self.turn_info.to_dict(),
            "players": players_list,
            "zones": {
                "battlefield": [enrich_obj(obj) for obj in self.battlefield],
                "my_hand": [enrich_obj(obj) for obj in self.hand] if self.local_seat_id else [],
                "opponent_hand_count": len(self.get_objects_in_zone(ZoneType.HAND, opponent_seat)) if opponent_seat else 0,
                "stack": [enrich_obj(obj) for obj in self.stack],
                "graveyard": [enrich_obj(obj) for obj in self.graveyard],
                "exile": [enrich_obj(obj) for obj in self.get_objects_in_zone(ZoneType.EXILE)],
            },
            "pending_decision": self.pending_decision,
            "decision_seat_id": self.decision_seat_id,
            "decision_context": self.decision_context,
        }
        return snapshot

    def clear_pending_combat_steps(self) -> None:
        """Clear the pending combat steps after processing."""
        self._pending_combat_steps.clear()
    
    def get_seat_source_name(self) -> str:
        """Get human-readable name of the seat ID source."""
        if self._seat_source == 0: return "None"
        if self._seat_source == 1: return "Inferred"
        if self._seat_source == 2: return "System"
        if self._seat_source == 3: return "User"
        return "Unknown"

    def set_local_seat_id(self, seat_id: int, source: int = 2) -> None:
        """Explicitly set the local player's seat ID if source priority allows.
        
        Source levels:
        1: Inferred (from hand visibility)
        2: System (from MatchCreated events)
        3: User (Manual override via F8)

        Args:
            seat_id: The local player's seat ID.
            source: Priority level (default 2=System).
        """
        # Only overwrite if new source is >= current source
        if source >= self._seat_source:
            self.local_seat_id = seat_id
            self._seat_source = source
            source_name = self.get_seat_source_name()
            logger.info(f"Set local_seat_id to {seat_id} (Source: {source_name})")
        else:
            logger.info(f"Ignored seat update to {seat_id} (Source {source} < Current {self._seat_source})")

    def reset_local_player(self, force: bool = False) -> None:
        """Reset local_seat_id logic.
        
        Args:
            force: If True, reset EVERYTHING (used for full restart).
                   If False, only reset INFERRED (1) sources. 
                   System (2) and User (3) are preserved across game resets (e.g. BO3).
        """
        if force or self._seat_source <= 1:
            self.local_seat_id = None
            self._seat_source = 0
            logger.info("Reset local_seat_id (cleared)")
        else:
            logger.info(f"Preserving local_seat_id={self.local_seat_id} (Source: {self.get_seat_source_name()})")

    def ensure_local_seat_id(self) -> None:
        """Ensure local_seat_id is set by inferring from existing data.

        Called by server before returning game state to ensure is_local
        is correctly determined. Uses hand zone with cards that have
        known grp_ids (you can see your own cards but not opponent's).
        """
        if self.local_seat_id is not None:
            return  # Already set

        # Try to infer from hand zones that have cards with VISIBLE grp_ids
        # Opponent's hand zone may have instance_ids but grp_id=0 (hidden)
        for zone in self.zones.values():
            if zone.zone_type != ZoneType.HAND:
                continue
            if zone.owner_seat_id is None:
                continue
            if not zone.object_instance_ids:
                continue

            # Check if ANY card in this hand has a known grp_id (not 0)
            has_visible_card = False
            for instance_id in zone.object_instance_ids:
                obj = self.game_objects.get(instance_id)
                if obj and obj.grp_id != 0:
                    has_visible_card = True
                    break

            if has_visible_card:
                # Use source=1 (Inferred)
                self.set_local_seat_id(zone.owner_seat_id, source=1)
                return

        # Fallback: log that we couldn't determine local player
        logger.debug("Could not infer local_seat_id - no hand zone with visible grp_ids found")

    def update_from_message(self, message: dict) -> None:
        """Update game state from a GameStateMessage payload.

        Handles both full and incremental (diff) updates from the game.
        All updates are treated as upserts (create or update).

        Args:
            message: The GameStateMessage dict from parsed log event.
        """
        # Extract type (full vs diff) - not currently used but logged
        msg_type = message.get("type", "Unknown")
        logger.debug(f"Processing GameStateMessage type: {msg_type}")

        # Update turn info FIRST so that zone updates use the correct turn number
        turn_info = message.get("turnInfo")
        if turn_info:
            self._update_turn_info(turn_info)

        # Update game objects
        game_objects = message.get("gameObjects", [])
        for obj_data in game_objects:
            self._update_game_object(obj_data)

        # Update zones
        zones = message.get("zones", [])
        for zone_data in zones:
            self._update_zone(zone_data)

        # Update players
        players = message.get("players", [])
        for player_data in players:
            self._update_player(player_data)

    def _update_game_object(self, obj_data: dict) -> None:
        """Update or create a game object from message data.

        Args:
            obj_data: Game object dict from GameStateMessage.
        """
        instance_id = obj_data.get("instanceId")
        if instance_id is None:
            return

        existing_obj = self.game_objects.get(instance_id)

        # Helper to get value from update or fallback to existing
        def get_val(key, default):
            if existing_obj:
                # If existing, prefer update if key exists, else existing attr
                # We need to map key string to attr name sometimes
                return obj_data.get(key, default)
                # Wait, this logic is tricky if I want "if key in obj_data".
            return obj_data.get(key, default)

        # Better merge logic:
        # 1. Start with defaults or existing values
        if existing_obj:
            grp_id = existing_obj.grp_id
            zone_id = existing_obj.zone_id
            owner_seat_id = existing_obj.owner_seat_id
            controller_seat_id = existing_obj.controller_seat_id
            visibility = existing_obj.visibility
            power = existing_obj.power
            toughness = existing_obj.toughness
            is_tapped = existing_obj.is_tapped
            card_types = existing_obj.card_types
            subtypes = existing_obj.subtypes
            parent_instance_id = existing_obj.parent_instance_id
            turn_entered_battlefield = existing_obj.turn_entered_battlefield
            is_attacking = existing_obj.is_attacking
            is_blocking = existing_obj.is_blocking
        else:
            grp_id = 0
            zone_id = 0
            owner_seat_id = 0
            controller_seat_id = None
            visibility = None
            power = None
            toughness = None
            is_tapped = False
            card_types = []
            subtypes = []
            parent_instance_id = None
            turn_entered_battlefield = -1
            is_attacking = False
            is_blocking = False

        # 2. Overwrite with present data
        if "grpId" in obj_data: grp_id = obj_data["grpId"]
        if "zoneId" in obj_data: zone_id = obj_data["zoneId"]
        if "ownerSeatId" in obj_data: owner_seat_id = obj_data["ownerSeatId"]
        if "controllerSeatId" in obj_data: controller_seat_id = obj_data["controllerSeatId"]
        if "visibility" in obj_data: visibility = obj_data["visibility"]
        
        if "power" in obj_data:
             p = obj_data["power"]
             power = p.get("value") if isinstance(p, dict) else p
        
        if "toughness" in obj_data:
             t = obj_data["toughness"]
             toughness = t.get("value") if isinstance(t, dict) else t

        if "isTapped" in obj_data: is_tapped = obj_data["isTapped"]
        if "parentId" in obj_data: parent_instance_id = obj_data["parentId"]

        if "isAttacking" in obj_data: is_attacking = bool(obj_data["isAttacking"])
        if "isBlocking" in obj_data: is_blocking = bool(obj_data["isBlocking"])

        if "cardTypes" in obj_data:
            card_types = list(obj_data["cardTypes"])
            
        if "subtypes" in obj_data:
            subtypes = []
            for st in obj_data["subtypes"]:
                clean_subtype = st.replace("SubType_", "") if isinstance(st, str) else str(st)
                subtypes.append(clean_subtype)

        game_object = GameObject(
            instance_id=instance_id,
            grp_id=grp_id,
            zone_id=zone_id,
            owner_seat_id=owner_seat_id,
            controller_seat_id=controller_seat_id,
            visibility=visibility,
            card_types=card_types,
            subtypes=subtypes,
            power=power,
            toughness=toughness,
            is_tapped=is_tapped,
            parent_instance_id=parent_instance_id,
            turn_entered_battlefield=turn_entered_battlefield,
            is_attacking=is_attacking,
            is_blocking=is_blocking,
        )


        self.game_objects[instance_id] = game_object
        logger.debug(f"Updated game object {instance_id} (grpId={grp_id})")

        # Track cards when first seen in non-library zones (reveals card identity)
        self._track_played_card(game_object)

    def _track_played_card(self, game_object: GameObject) -> None:
        """Track a card as played/revealed if first seen in non-library zone.

        This records the grp_id (card identity) when a card instance is
        first observed outside of the library, which indicates the card
        has been revealed to both players.

        Args:
            game_object: The game object to potentially track.
        """
        # Skip if already seen this instance
        if game_object.instance_id in self._seen_instances:
            return

        # Skip if grp_id is 0 (unknown/hidden card)
        if game_object.grp_id == 0:
            return

        # Determine zone type for this object
        zone = self.zones.get(game_object.zone_id)
        if zone is None:
            # Zone not yet known; can't determine if revealed
            return

        # Only track cards in non-library zones (where identity is revealed)
        # Library cards are hidden; once they move elsewhere, they're revealed
        non_library_zones = {
            ZoneType.BATTLEFIELD,
            ZoneType.HAND,
            ZoneType.GRAVEYARD,
            ZoneType.EXILE,
            ZoneType.STACK,
            ZoneType.COMMAND,
            ZoneType.REVEALED,
        }

        if zone.zone_type not in non_library_zones:
            return

        # Mark as seen
        self._seen_instances.add(game_object.instance_id)

        # Add to played cards for this owner
        owner = game_object.owner_seat_id
        if owner not in self.played_cards:
            self.played_cards[owner] = []
        self.played_cards[owner].append(game_object.grp_id)
        logger.debug(f"Tracked played card: owner={owner}, grpId={game_object.grp_id}")

    def _update_zone(self, zone_data: dict) -> None:
        """Update or create a zone from message data.

        Args:
            zone_data: Zone dict from GameStateMessage.
        """
        zone_id = zone_data.get("zoneId")
        if zone_id is None:
            return

        zone_type_str = zone_data.get("type", "Unknown")
        try:
            zone_type = ZoneType(zone_type_str)
        except ValueError:
            zone_type = ZoneType.UNKNOWN
            logger.debug(f"Unknown zone type: {zone_type_str}")

        owner_seat_id = zone_data.get("ownerSeatId")
        object_instance_ids = zone_data.get("objectInstanceIds", [])

        zone = Zone(
            zone_id=zone_id,
            zone_type=zone_type,
            owner_seat_id=owner_seat_id,
            object_instance_ids=object_instance_ids,
        )

        self.zones[zone_id] = zone
        logger.debug(f"Updated zone {zone_id} ({zone_type.name})")

        # Track battlefield entry for summoning sickness
        if zone_type == ZoneType.BATTLEFIELD:
            current_turn = self.turn_info.turn_number
            for instance_id in object_instance_ids:
                obj = self.game_objects.get(instance_id)
                if obj and obj.turn_entered_battlefield == -1:
                    obj.turn_entered_battlefield = current_turn
                    logger.debug(f"Object {instance_id} entered battlefield on turn {current_turn}")

        # Infer local player from hand visibility
        # Only infer if hand zone has cards with known grp_ids (not 0)
        # Opponent's hand has instance_ids but grp_id=0 (hidden cards)
        # NEVER override a manually set seat
        if zone_type == ZoneType.HAND and owner_seat_id is not None and object_instance_ids:
            if self.local_seat_id is None and not self._seat_manually_set:
                # Check if any card in this zone has a visible grp_id
                has_visible_card = False
                for instance_id in object_instance_ids:
                    obj = self.game_objects.get(instance_id)
                    if obj and obj.grp_id != 0:
                        has_visible_card = True
                        break

                if has_visible_card:
                    self.local_seat_id = owner_seat_id
                    logger.info(f"Inferred local player as seat {owner_seat_id} from hand zone with visible grp_ids")

    def _update_player(self, player_data: dict) -> None:
        """Update or create a player from message data.

        Args:
            player_data: Player dict from GameStateMessage.
        """
        seat_id = player_data.get("seatId") or player_data.get("systemSeatNumber")
        if seat_id is None:
            return

        life_total = player_data.get("lifeTotal", 20)
        lands_played = player_data.get("landsPlayedThisTurn", 0)

        # FALLBACK: Arena doesn't always track landsPlayedThisTurn correctly
        # Infer by counting lands that entered battlefield this turn
        if lands_played == 0 and self.turn_info.turn_number > 0:
            current_turn = self.turn_info.turn_number
            inferred_lands = 0
            for obj in self.battlefield:
                # Check card_types for "Land" (stored as "CardType_Land" from Arena)
                is_land_by_type = any("Land" in ct for ct in obj.card_types) if obj.card_types else False
                # Fallback: check type_line (e.g., "Basic Land — Forest")
                is_land_by_line = "land" in obj.type_line.lower() if obj.type_line else False
                is_land = is_land_by_type or is_land_by_line
                
                if (obj.owner_seat_id == seat_id and 
                    obj.controller_seat_id == seat_id and
                    is_land and
                    obj.turn_entered_battlefield == current_turn):
                    inferred_lands += 1
                    logger.debug(f"Inferred land: {obj.name} entered turn {current_turn} for seat {seat_id}")
            
            if inferred_lands > 0:
                lands_played = inferred_lands
                logger.info(f"Inferred lands_played={lands_played} for seat {seat_id} (Arena reported 0)")

        # Extract mana pool if present
        mana_pool = {}
        for mana_data in player_data.get("manaPool", []):
            mana_type = mana_data.get("type", "unknown")
            mana_count = mana_data.get("count", 0)
            mana_pool[mana_type] = mana_count

        player = Player(
            seat_id=seat_id,
            life_total=life_total,
            lands_played=lands_played,
            mana_pool=mana_pool,
        )

        self.players[seat_id] = player
        logger.debug(f"Updated player {seat_id} (life={life_total}, lands={lands_played})")

    def _update_turn_info(self, turn_data: dict) -> None:
        """Update turn info from message data.

        Args:
            turn_data: TurnInfo dict from GameStateMessage.
        """
        new_turn = turn_data.get("turnNumber", self.turn_info.turn_number)

        # Detect new game: turn number resets to 1 or decreases significantly
        # Detect new game: turn number resets to 1 or decreases significantly
        if self.turn_info.turn_number > 3 and new_turn <= 1:
            logger.info(f"New game detected (turn {self.turn_info.turn_number} -> {new_turn}) - Performing Search & Destroy on old state.")
            
            # FULL RESET of all zones, objects, players
            self.reset()
            
            # reset() makes turn_number 0, which is fine as we overwrite it below

        new_active = turn_data.get("activePlayer", self.turn_info.active_player)
        
        # Clear stale pending combat steps when turn or active player changes
        # Must check BEFORE updating turn_info
        if new_turn != self.turn_info.turn_number or new_active != self.turn_info.active_player:
            if self._pending_combat_steps:
                logger.debug(f"Clearing {len(self._pending_combat_steps)} stale pending combat steps (turn/active changed)")
                self._pending_combat_steps.clear()
        
        self.turn_info.turn_number = new_turn
        self.turn_info.active_player = new_active
        self.turn_info.priority_player = turn_data.get("priorityPlayer", self.turn_info.priority_player)
        new_phase = turn_data.get("phase", self.turn_info.phase)
        new_step = turn_data.get("step", self.turn_info.step)

        # Safety: auto-clear mulligan decision once game proceeds past turn 1
        # This handles cases where SubmitDeckResp message was missed
        if self.pending_decision == "Mulligan" and new_turn > 1:
            logger.info(f"Auto-clearing stale Mulligan decision (now turn {new_turn})")
            self.pending_decision = None
            self.decision_seat_id = None
            self.decision_context = None
            self.decision_timestamp = 0

        # Detect phase change
        if new_phase != self.turn_info.phase:
            logger.debug(f"Phase change: {self.turn_info.phase} -> {new_phase}")
            # If step is not explicitly in the update, reset it to avoid stale steps
            # (e.g., Step_Draw appearing in Phase_Main1)
            if "step" not in turn_data:
                new_step = ""
                logger.debug("Resetting step due to phase change")

        # Track combat steps as they happen (for event-driven triggers)
        if "Combat" in new_phase and new_step != self.turn_info.step:
            if "DeclareAttack" in new_step or "DeclareBlock" in new_step:
                # Store step with active player info for trigger generation
                self._pending_combat_steps.append({
                    "step": new_step,
                    "active_player": self.turn_info.active_player,
                    "turn": new_turn
                })
                logger.info(f"Queued combat step: {new_step} (active_player={self.turn_info.active_player})")

        self.turn_info.phase = new_phase
        self.turn_info.step = new_step
        logger.debug(f"Updated turn info: turn {self.turn_info.turn_number}, phase {self.turn_info.phase}, step {self.turn_info.step}")


def create_game_state_handler(game_state: GameState) -> Callable[[dict], None]:
    """Create an event handler that updates a GameState from GreToClientEvent.

    The handler extracts GameStateMessage from GreToClientEvent payloads
    and updates the provided GameState object.

    Args:
        game_state: The GameState instance to update.

    Returns:
        A handler function suitable for LogParser.register_handler().

    Example:
        game_state = GameState()
        handler = create_game_state_handler(game_state)
        parser.register_handler('GreToClientEvent', handler)
    """
    def handler(payload: dict) -> None:
        # GreToClientEvent contains greToClientMessages array
        gre_event = payload.get("greToClientEvent", {})
        messages = gre_event.get("greToClientMessages", [])

        # Process each message in the array
        for msg in messages:
            msg_type = msg.get("type", "")

            # Handle GameStateMessage
            if msg_type == "GREMessageType_GameStateMessage":
                game_state_msg = msg.get("gameStateMessage")
                if game_state_msg:
                    game_state.update_from_message(game_state_msg)
            
            # Handle Mulligan/Decision Requests
            elif msg_type == "GREMessageType_SubmitDeckReq" or msg_type == "GREMessageType_IntermissionReq":
                # This explicitly asks for a deck submission, usually mulligan phase
                logger.info(f"Captured Decision: Mulligan Check ({msg_type})")
                
                # If we get a Mulligan request while deeply into a game (Turn > 1), 
                # it implies a restart/rematch that happened before the 'Turn 1' update arrived.
                if game_state.turn_info.turn_number > 1:
                     logger.info(f"Mulligan Request detected at Turn {game_state.turn_info.turn_number} -> Resetting Ghost State.")
                     game_state.reset()
                
                game_state.pending_decision = "Mulligan"
                game_state.decision_seat_id = game_state.local_seat_id
                import time
                game_state.decision_timestamp = time.time()
                game_state.decision_context = {"type": "mulligan"}
                
            elif msg_type == "GREMessageType_PromptReq":
                import time
                prompt_text = msg.get("promptReq", {}).get("prompt", {}).get("text", "Action Required")
                logger.info(f"Captured Decision: Prompt ({prompt_text})")
                game_state.pending_decision = prompt_text
                game_state.decision_timestamp = time.time()
                game_state.decision_context = {"type": "prompt", "text": prompt_text}
                
            elif msg_type == "GREMessageType_SelectTargetsReq":
                # PHASE 1: Capture rich context for target selection
                import time
                req = msg.get("selectTargetsReq", {})
                source_id = req.get("sourceId")
                source_card = None
                if source_id:
                    # Try to find the source card on the stack
                    for obj in game_state.stack:
                        if obj.instance_id == source_id:
                            source_card = game_state._card_db.get_card_name(obj.grp_id)
                            break
                
                logger.info(f"Captured Decision: Select Targets (source: {source_card or 'unknown'})")
                game_state.pending_decision = "Select Targets"
                game_state.decision_timestamp = time.time()
                game_state.decision_context = {
                    "type": "target_selection",
                    "source_card": source_card,
                    "source_id": source_id,
                }
                
            elif msg_type == "GREMessageType_SelectNReq":
                # PHASE 1: Capture rich context for N-selection (discard, scry, etc.)
                import time
                req = msg.get("selectNReq", {})
                context_data = req.get("context", {})
                num_to_select = req.get("count", 1)
                min_select = req.get("minCount", num_to_select)
                max_select = req.get("maxCount", num_to_select)
                
                # Try to determine the type of selection (discard, scry, etc.)
                # This is heuristic-based on MTGA's context strings
                context_str = str(context_data).lower()
                if "discard" in context_str:
                    selection_type = "discard"
                elif "scry" in context_str:
                    selection_type = "scry"
                elif "surveil" in context_str:
                    selection_type = "surveil"
                else:
                    selection_type = "select_n"
                
                logger.info(f"Captured Decision: Select {num_to_select} Items ({selection_type})")
                game_state.pending_decision = "Select Items"
                game_state.decision_timestamp = time.time()
                game_state.decision_context = {
                    "type": selection_type,
                    "count": num_to_select,
                    "min": min_select,
                    "max": max_select,
                    "context_raw": context_data,
                }
                
            elif msg_type == "GREMessageType_GroupOptionReq":
                # PHASE 1: Capture modal spell options
                import time
                req = msg.get("groupOptionReq", {})
                options = req.get("options", [])
                
                logger.info(f"Captured Decision: Choose Mode ({len(options)} options)")
                game_state.pending_decision = "Choose Mode"
                game_state.decision_timestamp = time.time()
                game_state.decision_context = {
                    "type": "modal_choice",
                    "num_options": len(options),
                    "options": options,
                }

            elif "Resp" in msg_type:
                # PHASE 3: Only clear decisions on actual decision responses, not generic responses
                # Only clear on explicit decision submission responses
                decision_response_types = [
                    "GREMessageType_SelectTargetsResp",
                    "GREMessageType_SelectNResp",
                    "GREMessageType_GroupOptionResp",
                    "GREMessageType_SubmitDeckResp",  # Mulligan
                    "GREMessageType_PromptResp",
                ]
                
                if game_state.pending_decision and msg_type in decision_response_types:
                    # Don't auto-clear Mulligan (handled separately)
                    if game_state.pending_decision == "Mulligan" and msg_type != "GREMessageType_SubmitDeckResp":
                        pass
                    else:
                        logger.debug(f"Clearing decision '{game_state.pending_decision}' due to {msg_type}")
                        game_state.pending_decision = None
                        game_state.decision_seat_id = None
                        game_state.decision_context = None
                        game_state.decision_timestamp = 0

            elif msg_type == "GREMessageType_TimerStateMessage":
                pass
                
            elif msg_type == "GREMessageType_SubmitDeckResp":
                # We answered the mulligan, clear it
                if game_state.pending_decision == "Mulligan":
                    logger.info("Decision Cleared: Mulligan")
                    game_state.pending_decision = None
                    game_state.decision_seat_id = None
                    game_state.decision_context = None
                    game_state.decision_timestamp = 0

        # Also handle direct GameStateMessage events (legacy format)
        if "gameObjects" in payload or "zones" in payload or "players" in payload:
            game_state.update_from_message(payload)
        
        # Record frame if recording is active
        try:
            from arenamcp.match_validator import record_frame
            snapshot = game_state.get_snapshot()
            record_frame(payload, snapshot)
        except ImportError:
            pass  # match_validator not available
        except Exception as e:
            logger.debug(f"Frame recording skipped: {e}")

    return handler


def create_recording_handler(
    game_state: GameState,
    recording: "MatchRecording"  # Forward reference to avoid circular import
) -> Callable[[dict], None]:
    """Create a handler that updates GameState AND records frames for validation.
    
    This wraps the base handler to also capture raw messages alongside
    our parsed snapshots for post-match comparison.
    
    Args:
        game_state: The GameState instance to update.
        recording: The MatchRecording to add frames to.
        
    Returns:
        A handler function suitable for LogParser.register_handler().
    """
    base_handler = create_game_state_handler(game_state)
    
    def recording_handler(payload: dict) -> None:
        # First, apply the update via base handler
        base_handler(payload)
        
        # Then, record the frame with current snapshot
        try:
            snapshot = game_state.get_snapshot()
            recording.add_frame(payload, snapshot)
        except Exception as e:
            logger.warning(f"Failed to record frame: {e}")
    
    return recording_handler

