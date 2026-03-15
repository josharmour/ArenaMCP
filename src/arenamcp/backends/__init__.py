"""Pluggable LLM backends for MTG game coaching.

This package provides the LLMBackend protocol and concrete implementations
for Claude Code CLI, Gemini CLI, Codex CLI, and OpenAI-compatible proxy APIs.
"""

from arenamcp.backends.base import LLMBackend, _kill_proc_tree, _resolve_command
from arenamcp.backends.claude_code import ClaudeCodeBackend
from arenamcp.backends.gemini import GeminiCliBackend
from arenamcp.backends.codex import CodexCliBackend
from arenamcp.backends.proxy import ProxyBackend

__all__ = [
    "LLMBackend",
    "ClaudeCodeBackend",
    "GeminiCliBackend",
    "CodexCliBackend",
    "ProxyBackend",
    "_kill_proc_tree",
    "_resolve_command",
]
