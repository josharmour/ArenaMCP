# Building a Real-Time MTGA MCP Server for LLM Game Analysis

**A real-time MCP server that bridges MTGA logs to Claude can transform static game tracking into dynamic conversational coaching.** The key insight from this research: while several MTG MCP servers exist for card lookup and deck building, **none handle real-time game state from MTGA** — creating a significant opportunity. This guide provides a complete technical blueprint combining MTGA log parsing, MCP server architecture, external API integration, and lessons from chess/Go AI assistants.

## MTGA log files reveal everything except your opponent's hidden cards

MTGA's **Player.log** file (located at `%AppData%\LocalLow\Wizards Of The Coast\MTGA\Player.log`) contains structured JSON blocks for every game event when "Detailed Logs (Plugin Support)" is enabled in settings. The log is a mix of plain text markers and JSON payloads, with `[UnityCrossThreadLogger]` prefixing game events.

The critical message type is **`GreToClientEvent`** (Game Rules Engine to Client), which contains `GameStateMessage` objects with complete board state:

```typescript
interface GameStateMessage {
  type: string;
  gameObjects: ArenaMatchGameObject[];  // Every card/permanent in play
  turnInfo: TurnInfo;                    // Phase, active player, priority
  players: PlayerState[];                // Life totals, mana pools
  zones: Zone[];                         // Hand, battlefield, graveyard, exile
  annotations: Annotation[];             // Counters, attachments
}

interface ArenaMatchGameObject {
  instanceId: number;      // Runtime unique ID
  grpId: number;           // Card definition ID → maps to Scryfall
  zoneId: number;          // Current zone location
  ownerSeatId: number;     // Player 1 or 2
  isTapped: boolean;
  power: number;
  toughness: number;
  damage: number;
}
```

**What's available**: your hand contents, all battlefield permanents (tapped state, counters, damage, attachments), graveyard, exile, stack, library sizes, life totals, mana pools, turn/phase info, and timer state. **What's NOT available**: opponent's hand contents, opponent's library contents, face-down card identities, and sideboard changes (a known WotC logging gap).

## Real-time log tailing requires handling Windows file locking

For live game analysis, the server must tail the log file while MTGA is actively writing to it. The **watchdog** library is the recommended approach for Python:

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class MTGALogHandler(FileSystemEventHandler):
    def __init__(self, parser):
        self.parser = parser
        self.file_position = 0
        
    def on_modified(self, event):
        if event.src_path.endswith('Player.log'):
            with open(event.src_path, 'r', encoding='utf-8') as f:
                f.seek(self.file_position)
                new_content = f.read()
                self.file_position = f.tell()
                if new_content:
                    self.parser.process_new_lines(new_content)

# Handle log truncation on MTGA restart
def on_created(self, event):
    if event.src_path.endswith('Player.log'):
        self.file_position = 0  # Reset to start
```

The log format requires buffering multi-line JSON blocks — entries start with `[UnityCrossThreadLogger]` or `[Client GRE]` markers, followed by JSON spanning multiple lines until a closing brace. Existing trackers like **mtga-pro-tracker** and **17lands mtga-log-client** provide battle-tested parsing patterns.

## MCP architecture should use tools for dynamic state, resources for static data

For game state that changes every second during active play, **tools** are the correct MCP primitive — the LLM decides when to poll for updates. Resources are better suited for static card definitions and rules.

**Recommended tool structure**:

```python
from fastmcp import FastMCP

mcp = FastMCP("MTGAGameServer")

@mcp.tool
def get_game_state() -> dict:
    """Get complete current game state including board, hands, life totals"""
    return {
        "turn": game.turn_number,
        "phase": game.current_phase,
        "active_player": game.active_player,
        "priority_player": game.priority_player,
        "players": [
            {
                "seat": p.seat_id,
                "life": p.life_total,
                "hand_size": len(p.hand),
                "library_size": p.library_size,
                "mana_pool": p.mana_pool
            } for p in game.players
        ],
        "battlefield": [serialize_permanent(p) for p in game.battlefield],
        "stack": [serialize_stack_item(s) for s in game.stack],
        "my_hand": [serialize_card(c) for c in game.my_hand],
        "my_graveyard": [serialize_card(c) for c in game.my_graveyard]
    }

@mcp.tool
def get_card_info(arena_id: int) -> dict:
    """Look up card oracle text and rulings by Arena ID (grpId)"""
    return scryfall_cache.get_by_arena_id(arena_id)

@mcp.tool  
def get_opponent_played_cards() -> list:
    """Get list of all cards opponent has played this game"""
    return [serialize_card(c) for c in game.opponent_played_history]

@mcp.tool
def get_draft_rating(card_name: str, set_code: str) -> dict:
    """Get 17lands draft statistics for a card"""
    return seventeen_lands_cache.get_card_stats(card_name, set_code)
```

For **Claude Code CLI integration**, configure via `~/.claude.json`:

```json
{
  "mcpServers": {
    "mtga": {
      "command": "python",
      "args": ["/path/to/mtga_mcp_server.py"],
      "env": {
        "MTGA_LOG_PATH": "%AppData%/LocalLow/Wizards Of The Coast/MTGA/Player.log"
      }
    }
  }
}
```

Use **STDIO transport** for local operation — it provides the lowest latency (~microseconds vs milliseconds for HTTP) and requires no network configuration.

## Scryfall's arena endpoint directly maps MTGA card IDs

The critical API endpoint for MTGA integration is `GET https://api.scryfall.com/cards/arena/{grpId}`, which directly converts MTGA's `grpId` values to full card data including oracle text, rulings, and images. However, with **10 requests/second rate limits**, you should download Scryfall's bulk data daily and build a local cache:

```python
import scrython
import json

class CardDataService:
    def __init__(self):
        self.arena_index = {}
        self._load_bulk_data()
    
    def _load_bulk_data(self):
        bulk = scrython.bulk_data.ByType(type='default_cards')
        cards = bulk.download()
        for card in cards:
            if card.get('arena_id'):
                self.arena_index[card['arena_id']] = card
    
    def get_by_arena_id(self, arena_id: int) -> dict:
        if arena_id in self.arena_index:
            return self.arena_index[arena_id]
        # Fallback to API for new cards
        return scrython.cards.Arena(id=arena_id).scryfallJson
```

For **17lands draft data**, there's no real-time API — download CSV bulk files from `https://17lands-public.s3.amazonaws.com/analysis_data/` and process locally. Key statistics include **GIH WR** (Games in Hand Win Rate), **ALSA** (Average Last Seen At), and **IWD** (Improvement When Drawn).

## Existing MTG MCP servers focus on deck building, not live games

Research revealed several existing MTG MCP servers, but **none handle real-time MTGA game state**:

- **artillect/mtg-mcp-servers** (Python): Deck management and Scryfall integration, but simulates games rather than reading from MTGA
- **pato/mtg-mcp** (Rust): Card lookup using Scryfall query syntax
- **ericraio/mtg-mcp** (Swift): Actor-based game simulation

The gap is clear: these servers let you build decks and look up cards, but don't connect to actual MTGA gameplay. Existing trackers like **MTGA Pro Tracker** and **MTGAHelper** parse logs for overlays but don't expose data via MCP for LLM consumption.

## Game state representation for LLMs should be explicit and contextual

Chess AI research shows that **providing legal actions alongside state improves LLM performance**. For MTG, structure game state with explicit card text included rather than just IDs:

```python
def serialize_permanent(perm) -> dict:
    card_data = card_service.get_by_arena_id(perm.grpId)
    return {
        "name": card_data["name"],
        "oracle_text": card_data["oracle_text"],
        "type_line": card_data["type_line"],
        "is_tapped": perm.is_tapped,
        "power": perm.power,
        "toughness": perm.toughness,
        "damage_marked": perm.damage,
        "counters": perm.counters,  # e.g., [{"type": "+1/+1", "count": 2}]
        "attached_to": perm.parent_name if perm.parent_id else None,
        "controller": "you" if perm.controller == my_seat else "opponent"
    }
```

The **"Calculator + Coach" pattern** from Go AI implementations is particularly relevant: use deterministic game state tracking (the calculator) for accuracy while the LLM provides strategic commentary and explanations (the coach).

## Complete architecture recommendation

```
┌─────────────────────────────────────────────────────────────────┐
│                    MTGA MCP Server                              │
├─────────────────────────────────────────────────────────────────┤
│  Log Watcher Layer (watchdog)                                   │
│  └── Player.log → FileSystemEventHandler → Line buffer          │
├─────────────────────────────────────────────────────────────────┤
│  Log Parser Layer                                               │
│  ├── Line accumulator (handles multi-line JSON)                 │
│  ├── Event router (GreToClientEvent, MatchCreated, etc.)       │
│  └── State updater (incremental game state updates)             │
├─────────────────────────────────────────────────────────────────┤
│  Game State Manager                                             │
│  ├── Current game snapshot                                      │
│  ├── Game history (cards played, actions taken)                 │
│  ├── Delta tracking (changes since last query)                  │
│  └── Event stream for notable occurrences                       │
├─────────────────────────────────────────────────────────────────┤
│  External Data Layer                                            │
│  ├── Scryfall cache (bulk data + API fallback)                 │
│  ├── 17lands cache (CSV files per set)                         │
│  └── MTGA card definitions (from game files)                    │
├─────────────────────────────────────────────────────────────────┤
│  MCP Server (FastMCP, STDIO transport)                          │
│  ├── Tools: get_game_state, get_card_info, get_draft_stats     │
│  ├── Resources: game://rules, game://card/{arena_id}           │
│  └── Prompts: analyze_board_state, suggest_play                 │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation steps for building the server

**Phase 1: Log parsing foundation**
1. Implement file watcher using watchdog with Windows file locking handling
2. Build JSON block accumulator that handles multi-line log entries
3. Create event router for GreToClientEvent, GameStateMessage, etc.
4. Reference **mtga-pro-tracker** and **17lands mtga-log-client** for proven patterns

**Phase 2: Game state management**
1. Define GameState class tracking all zones, players, turn info
2. Implement incremental updates from GameStateMessage
3. Map `grpId` to card names using Scryfall data
4. Handle log truncation on MTGA restart (reset state)

**Phase 3: MCP server with FastMCP**
1. Install: `pip install fastmcp scrython watchdog`
2. Implement core tools: `get_game_state()`, `get_card_info()`, `get_opponent_history()`
3. Configure Claude Code: `claude mcp add mtga python /path/to/server.py`
4. Test with `/mcp` command in Claude Code to verify connection

**Phase 4: External data integration**
1. Download Scryfall bulk data daily, build `arena_id` index
2. Download 17lands CSVs for current draft set weekly
3. Implement card lookup with cache-first, API-fallback pattern

**Phase 5: LLM-optimized representations**
1. Include oracle text in all card serializations
2. Add strategic context (opponent cards seen, cards in deck remaining)
3. Implement delta queries for efficient updates during long games

## Key technical decisions and trade-offs

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| **Language** | Python (FastMCP) | Best log parsing libraries, Scryfall SDK (scrython), rapid prototyping |
| **Transport** | STDIO | Lowest latency for local use, simplest configuration |
| **Card data** | Scryfall bulk + API | Free, comprehensive, has `arena_id` mapping |
| **Draft stats** | 17lands CSV | Only source; no API available |
| **State updates** | Polling via tools | MCP Resources don't support true server-push to Claude |
| **Caching** | SQLite or JSON files | Card data changes rarely; persist across sessions |

## Conclusion

Building a real-time MTGA MCP server fills a clear gap in the ecosystem — existing MTG MCP servers handle card lookup and deck building, but none connect to live games. The technical foundation is solid: MTGA's detailed logging exposes complete game state, Scryfall provides direct `arena_id` mapping, and FastMCP makes server implementation straightforward.

The most valuable insight from this research is the **Calculator + Coach pattern**: use deterministic game state tracking for accuracy while leveraging the LLM for strategic explanation and conversational coaching. This avoids the pitfalls seen in chess LLM research where models struggle with complex state tracking, while capitalizing on LLMs' strength in natural language explanation.

Starting with log parsing from proven trackers and exposing state via simple MCP tools creates an immediate foundation for conversational game analysis — far beyond what pre-programmed pattern matching can achieve.