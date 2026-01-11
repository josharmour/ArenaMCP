# Roadmap: MTGA MCP Server

## Overview

Build a real-time MCP server that parses MTGA Player.log to expose live game state to Claude. Start with reliable log watching and parsing, build game state tracking, integrate external card/draft data, then wrap it all in FastMCP tools.

## Domain Expertise

None

## Completed Milestones

- [v1.0 MVP](milestones/v1.0-ROADMAP.md) (Phases 1-4) — SHIPPED 2026-01-11

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

<details>
<summary>v1.0 MVP (Phases 1-4) — SHIPPED 2026-01-11</summary>

- [x] **Phase 1: Foundation** - Log watcher and parser for real-time event extraction
- [x] **Phase 2: Game State** - State manager tracking zones, turns, and game objects
- [x] **Phase 3: External Data** - Scryfall card data and 17lands draft statistics
- [x] **Phase 4: MCP Server** - FastMCP tools exposing game state and card info

</details>

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation | v1.0 | 2/2 | Complete | 2026-01-11 |
| 2. Game State | v1.0 | 1/1 | Complete | 2026-01-11 |
| 3. External Data | v1.0 | 2/2 | Complete | 2026-01-11 |
| 4. MCP Server | v1.0 | 1/1 | Complete | 2026-01-11 |

**v1.0 Complete:** All 4 phases delivered.
