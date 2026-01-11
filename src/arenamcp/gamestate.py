"""MTGA game state tracking from parsed log events.

This module provides the GameState class that maintains a complete
snapshot of the current game state from parsed MTGA log events.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
