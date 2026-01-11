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
    """A game object (card, token, etc.) in the game."""
    instance_id: int
    grp_id: int
    zone_id: int
    owner_seat_id: int
    controller_seat_id: Optional[int] = None
    visibility: Optional[str] = None
    card_types: list[str] = field(default_factory=list)
    power: Optional[int] = None
    toughness: Optional[int] = None
    is_tapped: bool = False


@dataclass
class Zone:
    """A game zone (battlefield, hand, etc.)."""
    zone_id: int
    zone_type: ZoneType
    owner_seat_id: Optional[int] = None
    object_instance_ids: list[int] = field(default_factory=list)


@dataclass
class Player:
    """A player in the game."""
    seat_id: int
    life_total: int = 20
    mana_pool: dict[str, int] = field(default_factory=dict)


@dataclass
class TurnInfo:
    """Current turn information."""
    turn_number: int = 0
    active_player: int = 0
    priority_player: int = 0
    phase: str = ""
    step: str = ""


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

        # Local player tracking (set externally from MatchCreated event)
        self.local_seat_id: Optional[int] = None

        # Opponent card history tracking
        self.played_cards: dict[int, list[int]] = {}  # seat_id -> list of grp_ids
        self._seen_instances: set[int] = set()  # Track instances to avoid double-counting

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
            return self.get_objects_in_zone(ZoneType.HAND, self.local_seat_id)
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

        # Update turn info
        turn_info = message.get("turnInfo")
        if turn_info:
            self._update_turn_info(turn_info)

    def _update_game_object(self, obj_data: dict) -> None:
        """Update or create a game object from message data.

        Args:
            obj_data: Game object dict from GameStateMessage.
        """
        instance_id = obj_data.get("instanceId")
        if instance_id is None:
            return

        grp_id = obj_data.get("grpId", 0)
        zone_id = obj_data.get("zoneId", 0)
        owner_seat_id = obj_data.get("ownerSeatId", 0)
        controller_seat_id = obj_data.get("controllerSeatId")
        visibility = obj_data.get("visibility")

        # Extract card types
        card_types = []
        for ct in obj_data.get("cardTypes", []):
            card_types.append(ct)

        # Extract power/toughness if present
        power = obj_data.get("power", {}).get("value") if isinstance(obj_data.get("power"), dict) else obj_data.get("power")
        toughness = obj_data.get("toughness", {}).get("value") if isinstance(obj_data.get("toughness"), dict) else obj_data.get("toughness")

        # Check if tapped
        is_tapped = obj_data.get("isTapped", False)

        game_object = GameObject(
            instance_id=instance_id,
            grp_id=grp_id,
            zone_id=zone_id,
            owner_seat_id=owner_seat_id,
            controller_seat_id=controller_seat_id,
            visibility=visibility,
            card_types=card_types,
            power=power,
            toughness=toughness,
            is_tapped=is_tapped,
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

    def _update_player(self, player_data: dict) -> None:
        """Update or create a player from message data.

        Args:
            player_data: Player dict from GameStateMessage.
        """
        seat_id = player_data.get("seatId") or player_data.get("systemSeatNumber")
        if seat_id is None:
            return

        life_total = player_data.get("lifeTotal", 20)

        # Extract mana pool if present
        mana_pool = {}
        for mana_data in player_data.get("manaPool", []):
            mana_type = mana_data.get("type", "unknown")
            mana_count = mana_data.get("count", 0)
            mana_pool[mana_type] = mana_count

        player = Player(
            seat_id=seat_id,
            life_total=life_total,
            mana_pool=mana_pool,
        )

        self.players[seat_id] = player
        logger.debug(f"Updated player {seat_id} (life={life_total})")

    def _update_turn_info(self, turn_data: dict) -> None:
        """Update turn info from message data.

        Args:
            turn_data: TurnInfo dict from GameStateMessage.
        """
        self.turn_info.turn_number = turn_data.get("turnNumber", self.turn_info.turn_number)
        self.turn_info.active_player = turn_data.get("activePlayer", self.turn_info.active_player)
        self.turn_info.priority_player = turn_data.get("priorityPlayer", self.turn_info.priority_player)
        self.turn_info.phase = turn_data.get("phase", self.turn_info.phase)
        self.turn_info.step = turn_data.get("step", self.turn_info.step)
        logger.debug(f"Updated turn info: turn {self.turn_info.turn_number}, phase {self.turn_info.phase}")


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
        # GreToClientEvent wraps the actual event data
        gre_event = payload.get("greToClientEvent", payload)

        # Check for gameStateMessage in the event
        game_state_msg = gre_event.get("gameStateMessage")
        if game_state_msg:
            game_state.update_from_message(game_state_msg)
            return

        # Also handle direct GameStateMessage events
        if "gameObjects" in payload or "zones" in payload or "players" in payload:
            game_state.update_from_message(payload)

    return handler
