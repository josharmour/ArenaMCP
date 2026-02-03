# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the design and implementation of a real-time MCP (Model Context Protocol) server that bridges MTGA (Magic: The Gathering Arena) game logs to Claude for conversational game analysis and coaching.

The key differentiator: existing MTG MCP servers handle card lookup and deck building, but this one connects to **live MTGA games** by parsing the Player.log file in real-time.

## Architecture

The server follows a layered architecture:

1. **Log Watcher Layer** - Uses watchdog to tail `%AppData%\LocalLow\Wizards Of The Coast\MTGA\Player.log`
2. **Log Parser Layer** - Accumulates multi-line JSON blocks, routes events (GreToClientEvent, MatchCreated, etc.)
3. **Game State Manager** - Maintains current game snapshot, history, and delta tracking
4. **External Data Layer** - Scryfall cache for card data, 17lands CSVs for draft statistics
5. **MCP Server** - FastMCP with STDIO transport exposing tools and resources

## Key Technical Decisions

- **Language**: Python with FastMCP
- **Transport**: STDIO (lowest latency for local use)
- **Card Data**: Scryfall bulk download + API fallback, using `arena_id` mapping
- **Draft Stats**: 17lands CSV files (no API available)
- **State Updates**: Polling via MCP tools (MCP Resources don't support server-push)

## Core MCP Tools to Implement

- `get_game_state()` - Complete board state, hands, life totals, stack
- `get_card_info(arena_id)` - Card oracle text and rulings via Scryfall
- `get_opponent_played_cards()` - Cards opponent has revealed this game
- `get_draft_rating(card_name, set_code)` - 17lands draft statistics

## MTGA Log Format

The Player.log contains `[UnityCrossThreadLogger]` prefixed lines with JSON payloads. The critical message type is `GreToClientEvent` containing `GameStateMessage` objects with:
- `gameObjects` - All cards/permanents with `grpId` (maps to Scryfall `arena_id`)
- `turnInfo` - Phase, active player, priority
- `players` - Life totals, mana pools
- `zones` - Hand, battlefield, graveyard, exile

## Design Pattern

Follow the **Calculator + Coach** pattern: use deterministic game state tracking for accuracy while the LLM provides strategic commentary. This avoids LLM state-tracking weaknesses while leveraging its explanation strengths.

## Dependencies

```
pip install fastmcp scrython watchdog
```

## Claude Code Configuration

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

## Reference Implementations

For log parsing patterns, reference:
- mtga-pro-tracker
- 17lands mtga-log-client
