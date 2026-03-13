"""Tests for parser robustness against brace desync (Workstream 4)."""

import pytest

from arenamcp.parser import LogParser


@pytest.fixture
def parser():
    events = []
    p = LogParser(on_event=lambda t, d: events.append((t, d)))
    return p, events


class TestStringAwareBraceCounting:
    def test_braces_inside_string_ignored(self, parser):
        """Braces inside JSON string values must not affect depth tracking."""
        p, events = parser
        chunk = (
            '[UnityCrossThreadLogger]GreToClientEvent\n'
            '{"text": "Tap {T}: Add {G}", "value": 1}\n'
        )
        p.process_chunk(chunk)
        assert len(events) == 1
        assert events[0][1]["text"] == "Tap {T}: Add {G}"

    def test_escaped_quotes_in_string(self, parser):
        """Escaped quotes inside strings don't break string tracking."""
        p, events = parser
        chunk = (
            '[UnityCrossThreadLogger]GreToClientEvent\n'
            '{"msg": "He said \\"hello\\" to {them}", "ok": true}\n'
        )
        p.process_chunk(chunk)
        assert len(events) == 1
        assert events[0][1]["ok"] is True

    def test_nested_braces_in_string(self, parser):
        """Deeply nested braces in string values don't desync."""
        p, events = parser
        chunk = (
            '[UnityCrossThreadLogger]GreToClientEvent\n'
            '{"rules": "{{{nested}}}", "count": 42}\n'
        )
        p.process_chunk(chunk)
        assert len(events) == 1
        assert events[0][1]["count"] == 42

    def test_normal_nested_json(self, parser):
        """Normal nested JSON objects still work correctly."""
        p, events = parser
        chunk = (
            '[UnityCrossThreadLogger]GreToClientEvent\n'
            '{\n'
            '  "outer": {\n'
            '    "inner": {"deep": true}\n'
            '  }\n'
            '}\n'
        )
        p.process_chunk(chunk)
        assert len(events) == 1
        assert events[0][1]["outer"]["inner"]["deep"] is True


class TestChunkBoundaries:
    def test_split_across_chunks(self, parser):
        """JSON split across two chunks reassembles correctly."""
        p, events = parser
        p.process_chunk('[UnityCrossThreadLogger]GreToClientEvent\n{"ke')
        assert len(events) == 0
        p.process_chunk('y": "value"}\n')
        assert len(events) == 1
        assert events[0][1]["key"] == "value"

    def test_partial_line_buffering(self, parser):
        """Incomplete line at chunk boundary is buffered."""
        p, events = parser
        # First chunk ends mid-line (no trailing newline)
        p.process_chunk('[UnityCrossThreadLogger]GreTo')
        assert len(events) == 0
        # Second chunk completes the line
        p.process_chunk('ClientEvent\n{"a": 1}\n')
        assert len(events) == 1
        assert events[0][1]["a"] == 1

    def test_multiple_events_in_one_chunk(self, parser):
        """Multiple complete JSON blocks in a single chunk."""
        p, events = parser
        chunk = (
            '[UnityCrossThreadLogger]GreToClientEvent\n'
            '{"first": true}\n'
            '[UnityCrossThreadLogger]GreToClientEvent\n'
            '{"second": true}\n'
        )
        p.process_chunk(chunk)
        assert len(events) == 2


class TestMalformedRecovery:
    def test_malformed_then_valid(self, parser):
        """Parser recovers from malformed JSON and processes next valid block."""
        p, events = parser
        # Brace-balanced but JSON-invalid block (json.loads will reject it)
        chunk = (
            '[UnityCrossThreadLogger]GreToClientEvent\n'
            '{not valid json at all}\n'
        )
        p.process_chunk(chunk)
        # The malformed block has balanced braces so depth reaches 0,
        # json.loads fails, parser resets.

        # Now a valid block should parse successfully
        chunk2 = (
            '[UnityCrossThreadLogger]GreToClientEvent\n'
            '{"good": true}\n'
        )
        p.process_chunk(chunk2)
        good_events = [e for e in events if e[1].get("good")]
        assert len(good_events) == 1

    def test_unclosed_string_recovers_via_buffer_limit(self, parser):
        """Unclosed string in JSON block eventually resets via buffer overflow."""
        p, events = parser
        # Set a small buffer limit for this test
        p._MAX_BUFFER_LINES = 10

        # Start a block with an unclosed string — brace depth stays at 1
        p.process_chunk('[UnityCrossThreadLogger]GreToClientEvent\n')
        p.process_chunk('{"bad": "unclosed string\n')
        # Feed enough lines to trigger buffer overflow
        for i in range(15):
            p.process_chunk(f'more garbage line {i}\n')

        # Parser should have reset. Next valid event should work.
        p.process_chunk('[UnityCrossThreadLogger]GreToClientEvent\n{"recovered": true}\n')
        recovered = [e for e in events if e[1].get("recovered")]
        assert len(recovered) == 1

    def test_negative_brace_depth_resets(self, parser):
        """Negative brace depth resets state and allows recovery."""
        p, events = parser
        # Extra closing brace
        p.process_chunk('[UnityCrossThreadLogger]GreToClientEvent\n')
        p.process_chunk('{"a": 1}}\n')  # extra }
        # Parser should reset. Next valid event should work.
        p.process_chunk('[UnityCrossThreadLogger]GreToClientEvent\n{"b": 2}\n')
        good_events = [e for e in events if e[1].get("b") == 2]
        assert len(good_events) == 1

    def test_runaway_buffer_resets(self, parser):
        """Exceeding max buffer lines resets parser and allows recovery."""
        p, events = parser
        # Start a JSON block that never closes
        p.process_chunk('[UnityCrossThreadLogger]GreToClientEvent\n{\n')
        # Feed many lines without closing
        for i in range(p._MAX_BUFFER_LINES + 10):
            p.process_chunk(f'  "line_{i}": {i},\n')

        # Parser should have reset by now. Valid event should work.
        p.process_chunk('[UnityCrossThreadLogger]GreToClientEvent\n{"recovered": true}\n')
        recovered = [e for e in events if e[1].get("recovered")]
        assert len(recovered) == 1


class TestBraceDeltaDirectly:
    """Direct unit tests for _count_brace_delta (stateless mode)."""

    def test_simple_open(self):
        p = LogParser()
        assert p._count_brace_delta("{", stateful=False) == 1

    def test_simple_close(self):
        p = LogParser()
        assert p._count_brace_delta("}", stateful=False) == -1

    def test_balanced(self):
        p = LogParser()
        assert p._count_brace_delta('{"key": "val"}', stateful=False) == 0

    def test_braces_in_string(self):
        p = LogParser()
        assert p._count_brace_delta('"text with {braces}"', stateful=False) == 0

    def test_escaped_quote(self):
        p = LogParser()
        # The string is: "say \"hi\" {there}"  — braces are inside the string
        assert p._count_brace_delta(r'"say \"hi\" {there}"', stateful=False) == 0

    def test_nested(self):
        p = LogParser()
        assert p._count_brace_delta('{"a": {"b": 1}}', stateful=False) == 0

    def test_only_opens(self):
        p = LogParser()
        assert p._count_brace_delta("{{{", stateful=False) == 3

    def test_mixed_string_and_real(self):
        p = LogParser()
        # Real open brace, then string with braces, then real close
        result = p._count_brace_delta('{"text": "{T}: Add {G}"}', stateful=False)
        assert result == 0
