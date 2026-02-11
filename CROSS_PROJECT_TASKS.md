# Cross-Project Integration Tasks

## ArenaMCP ↔ MTGA Voice Assistant

Feature integration task list for sharing strengths between both projects.

**Created**: 2026-02-10
**Projects**:
- **MTGA Voice Assistant**: `/mnt/c/Users/joshu/mtga-voice-assistant` (GitHub: josharmour/mtga-voice-assistant)
- **ArenaMCP**: `Z:/ArenaMCP`

---

## Priority 1: Claude Code CLI Backend
**Direction**: ArenaMCP → Voice Assistant
**Effort**: Small | **Impact**: High

**Problem**: Users are confused about API keys (see Reddit feedback). Many have a Claude Code subscription but no API key.

**Task**: Add a `ClaudeCodeAdvisor` to `src/core/llm/` that spawns a `claude` subprocess with stream-json I/O, using the user's existing subscription. No API key needed.

**Reference**: ArenaMCP's `coach.py` — `ClaudeCodeBackend` class (subprocess with persistent process, request queuing, 12s timeout).

**Steps**:
- [ ] Create `src/core/llm/claudecode_advisor.py` based on ArenaMCP's ClaudeCodeBackend
- [ ] Add "Claude Code (subscription)" to provider dropdown in `src/core/ui.py`
- [ ] Register in `src/core/ai.py` advisor_map
- [ ] Hide API key field when this provider is selected (same as Ollama)
- [ ] Test with `claude` CLI installed

**Status**: Not started

---

## Priority 2: Synergy Graph for Draft Evaluation
**Direction**: Voice Assistant → ArenaMCP
**Effort**: Medium | **Impact**: High

**Problem**: ArenaMCP's `draft_eval.py` scores cards with 17Lands stats + basic keyword matching. No real synergy analysis across picked cards.

**Task**: Port the NetworkX SynergyGraph from Voice Assistant's `src/cognitive/__init__.py` into ArenaMCP to enhance `evaluate_pack()` with weighted synergy scoring (tribal, mechanic, theme, keyword interactions).

**Reference**: Voice Assistant's `src/cognitive/__init__.py` — SynergyGraph class with inverted indexing and weighted edges.

**Steps**:
- [ ] Copy/adapt SynergyGraph into ArenaMCP's `src/arenamcp/` (or shared package)
- [ ] Build synergy graph from Scryfall bulk data on first run
- [ ] Integrate `find_synergies_for_card()` into `draft_eval.py:evaluate_pack()`
- [ ] Add synergy score as a weighted component alongside GIH WR
- [ ] Cache the graph to `~/.arenamcp/cache/synergy_graph.pkl`

**Status**: Not started

---

## Priority 3: Deck Builder MCP Tool
**Direction**: Voice Assistant → ArenaMCP
**Effort**: Medium | **Impact**: High

**Problem**: ArenaMCP helps draft but stops after the last pick. No deck building assistance.

**Task**: Port Voice Assistant's deck builder (`src/core/deck_builder_v2.py`) and expose it as a `build_deck` MCP tool. Takes draft pool from DraftState, recommends color pairs, mana base, and cuts.

**Reference**: Voice Assistant's `src/core/deck_builder_v2.py` — color pair analysis, synergy scoring, mana base calculation.

**Steps**:
- [ ] Adapt DeckBuilderV2 for ArenaMCP's data structures (DraftState → picked_cards)
- [ ] Add `build_deck()` tool to `server.py`
- [ ] Wire up Scryfall data for card type/cost analysis
- [ ] Return structured recommendation (main deck, sideboard, mana base)
- [ ] Add to standalone coach as a post-draft command

**Status**: Not started

---

## Priority 4: VOX Voice Activation Mode
**Direction**: ArenaMCP → Voice Assistant
**Effort**: Small | **Impact**: Medium

**Problem**: Voice Assistant only has push-to-talk. Hands-free voice activation is better for gameplay.

**Task**: Port ArenaMCP's VOXDetector (RMS-based voice detection with silence timeout) into Voice Assistant as an alternative to PTT.

**Reference**: ArenaMCP's `triggers.py` — VOXDetector class with configurable RMS threshold and silence timeout.

**Steps**:
- [ ] Port VOXDetector from ArenaMCP's `triggers.py` into Voice Assistant
- [ ] Add "Voice Mode: PTT / VOX" toggle in Settings panel (`src/core/ui.py`)
- [ ] Store preference in `config_manager.py`
- [ ] Wire into existing voice input pipeline
- [ ] Add sensitivity slider for RMS threshold

**Status**: Not started

---

## Priority 5: MCP Server Exposure
**Direction**: ArenaMCP → Voice Assistant
**Effort**: Large | **Impact**: High

**Problem**: Voice Assistant treats the LLM as a stateless API. With MCP, the LLM drives the interaction — querying game state, looking up cards, and checking stats on demand.

**Task**: Expose Voice Assistant's systems as MCP tools using FastMCP. The Tkinter GUI continues handling display while the MCP server handles LLM communication.

**Reference**: ArenaMCP's `server.py` — FastMCP STDIO server with lazy-loaded resources.

**Steps**:
- [ ] Add `fastmcp` dependency to requirements.txt
- [ ] Create `src/core/mcp_server.py` exposing tools:
  - `get_game_state()` — current board state from GameStateManager
  - `get_card_info(name_or_id)` — card data from ArenaCardDatabase
  - `get_draft_rating(card, set)` — 17Lands stats from CardStatsDB
  - `search_cards(query)` — semantic search via ChromaDB
  - `get_synergies(card)` — synergy graph lookup
  - `get_metagame(format)` — MTGGoldfish data
  - `get_commander_info(name)` — EDHREC data
- [ ] Support running alongside GUI (shared GameStateManager)
- [ ] Add Claude Code MCP config generation (`~/.claude.json` snippet)
- [ ] Document setup in README

**Status**: Not started

---

## Priority 6: Autopilot Engine
**Direction**: ArenaMCP → Voice Assistant
**Effort**: Large | **Impact**: Medium

**Problem**: Voice Assistant is observe-and-advise only. Autopilot lets the LLM actually play.

**Task**: Port ArenaMCP's autopilot pipeline (action planner → screen mapper → input controller) into Voice Assistant as an optional advanced mode.

**Reference**: ArenaMCP's `action_planner.py`, `screen_mapper.py`, `input_controller.py`, `autopilot.py`.

**Steps**:
- [ ] Port autopilot modules into `src/core/autopilot/`
- [ ] Wire action planner to GameStateManager output
- [ ] Add "Autopilot Mode" toggle in Settings with safety warnings
- [ ] Implement human-in-the-loop confirmation (SPACEBAR confirm, ESC skip)
- [ ] Add preview panel in GUI showing planned actions before execution
- [ ] Add PyAutoGUI and pygetwindow to optional dependencies

**Status**: Not started

---

## Priority 7: EDHREC + MTGGoldfish MCP Tools
**Direction**: Voice Assistant → ArenaMCP
**Effort**: Small | **Impact**: Low

**Problem**: ArenaMCP only knows about Arena Limited. Users ask about other formats between matches.

**Task**: Port Voice Assistant's EDHREC and MTGGoldfish clients and expose them as MCP tools.

**Reference**: Voice Assistant's `src/data/edhrec.py` and `src/data/mtggoldfish.py`.

**Steps**:
- [ ] Copy/adapt EDHREC client into ArenaMCP
- [ ] Copy/adapt MTGGoldfish client into ArenaMCP
- [ ] Add `get_metagame(format)` tool to `server.py`
- [ ] Add `get_commander_info(name)` tool to `server.py`
- [ ] Cache responses with same TTL strategy (EDHREC 24h, MTGGoldfish 1h)

**Status**: Not started

---

## Priority 8: Real-time Voice APIs
**Direction**: ArenaMCP → Voice Assistant
**Effort**: Medium | **Impact**: Low

**Problem**: Voice Assistant uses Whisper → LLM → Kokoro pipeline (multiple seconds latency). Real-time voice APIs offer sub-second conversational coaching.

**Task**: Add Gemini Live and/or GPT-Realtime as voice backends — bidirectional WebSocket audio streams with no transcription step.

**Reference**: ArenaMCP's `gemini_live.py` and `realtime.py`.

**Steps**:
- [ ] Port GeminiLiveClient from ArenaMCP
- [ ] Port GPTRealtimeClient from ArenaMCP
- [ ] Add "Real-time Voice" provider option in Settings
- [ ] Require Azure/Google API keys for these backends
- [ ] Handle WebSocket lifecycle (connect/disconnect/reconnect)
- [ ] Fallback to standard pipeline if WebSocket fails

**Status**: Not started

---

## Other Opportunities (Not Prioritized)

### MTGA Native Database Reader
**Direction**: ArenaMCP → Voice Assistant

ArenaMCP reads MTGA's installed CardDatabase SQLite directly (always current, no rebuild). Could replace or supplement Voice Assistant's pre-built `unified_cards.db`. Lower priority since the current approach works and not all dev environments have MTGA installed.

### Match State Recovery
**Direction**: Voice Assistant → ArenaMCP

Voice Assistant saves match state every 10 turns and resumes from file offset. ArenaMCP loses all context on restart. Port the offset-based resume and `last_match.json` persistence.

### ChromaDB Semantic Search
**Direction**: Voice Assistant → ArenaMCP

Add a `search_cards(query)` MCP tool backed by ChromaDB for natural language card lookups ("do I have removal for that?" instead of exact name matching).

### Shared Card Data Package
**Direction**: Both

Extract log parsing, card data, and 17Lands integration into a shared Python package that both projects depend on. Biggest long-term maintenance win but requires significant refactoring. Consider after individual feature ports stabilize.
