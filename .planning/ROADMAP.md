# Roadmap: MTGA MCP Server

## Overview

Build a real-time MCP server that parses MTGA Player.log to expose live game state to Claude. Start with reliable log watching and parsing, build game state tracking, integrate external card/draft data, then wrap it all in FastMCP tools.

## Domain Expertise

None

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Foundation** - Log watcher and parser for real-time event extraction
- [ ] **Phase 2: Game State** - State manager tracking zones, turns, and game objects
- [ ] **Phase 3: External Data** - Scryfall card data and 17lands draft statistics
- [ ] **Phase 4: MCP Server** - FastMCP tools exposing game state and card info

## Phase Details

### Phase 1: Foundation
**Goal**: Reliable log watching with event extraction from Player.log
**Depends on**: Nothing (first phase)
**Research**: Unlikely (Python watchdog, JSON parsing - established patterns)
**Plans**: TBD

Key deliverables:
- Watchdog-based log watcher handling Windows file locking
- Log parser accumulating multi-line JSON blocks
- Event routing for GreToClientEvent, MatchCreated, GameStateMessage
- Basic event callback system

### Phase 2: Game State
**Goal**: Complete game state snapshot from parsed events
**Depends on**: Phase 1
**Research**: Unlikely (internal data structures)
**Plans**: TBD

Key deliverables:
- GameState class with zones (battlefield, hand, graveyard, exile, library, stack)
- Card/permanent tracking with grpId mapping
- Turn/phase tracking (active player, priority, phase)
- Player state (life totals, mana pools)
- History tracking for opponent played cards

### Phase 3: External Data
**Goal**: Card oracle text and draft statistics from external sources
**Depends on**: Phase 2
**Research**: Likely (external APIs and data formats)
**Research topics**: Scryfall bulk data format and arena_id mapping, 17lands CSV structure (GIH WR, ALSA, IWD columns)
**Plans**: TBD

Key deliverables:
- Scryfall bulk data download and indexing by arena_id
- API fallback for cards not in bulk data
- 17lands CSV parsing for draft statistics
- Caching layer for performance

### Phase 4: MCP Server
**Goal**: FastMCP server exposing all functionality via STDIO
**Depends on**: Phase 3
**Research**: Unlikely (FastMCP well-documented in constraints)
**Plans**: TBD

Key deliverables:
- get_game_state() tool with full board state serialization
- get_card_info(arena_id) tool for oracle text
- get_opponent_played_cards() tool
- get_draft_rating(card_name, set_code) tool
- Card serialization including oracle text (Calculator + Coach pattern)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Foundation | 0/TBD | Not started | - |
| 2. Game State | 0/TBD | Not started | - |
| 3. External Data | 0/TBD | Not started | - |
| 4. MCP Server | 0/TBD | Not started | - |
