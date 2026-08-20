from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from terminal_bench.agents.installed_agents.claude_code.claude_code_agent import (
    ClaudeCodeAgent,
)
from terminal_bench.agents.installed_agents.codex.codex_agent import CodexAgent
from terminal_bench.agents.installed_agents.gemini_cli.gemini_cli_agent import (
    GeminiCliAgent,
)
from terminal_bench.agents.installed_agents.opencode.opencode_agent import OpenCodeAgent


@pytest.mark.parametrize(
    ("agent", "key"),
    [
        (CodexAgent("openai/gpt-5.5"), "OPENAI_API_KEY"),
        (GeminiCliAgent("google/gemini-3.1-pro"), "GEMINI_API_KEY"),
        (ClaudeCodeAgent("anthropic/claude-opus-4"), "ANTHROPIC_API_KEY"),
        (OpenCodeAgent("openrouter/minimax/minimax-m2.7"), "OPENROUTER_API_KEY"),
    ],
)
def test_published_agents_forward_their_api_key(agent: object, key: str) -> None:
    with mock.patch.dict(os.environ, {key: "test-api-key"}, clear=True):
        env = agent._env  # type: ignore[attr-defined]

    assert env[key] == "test-api-key"


@pytest.mark.parametrize(
    ("agent", "key", "legacy_env"),
    [
        (
            CodexAgent("openai/gpt-5.5"),
            "OPENAI_API_KEY",
            {"CODEX_AUTH_JSON_BASE64": "legacy"},
        ),
        (
            GeminiCliAgent("google/gemini-3.1-pro"),
            "GEMINI_API_KEY",
            {"GEMINI_CLI_AUTH": "/tmp/legacy"},
        ),
        (
            ClaudeCodeAgent("anthropic/claude-opus-4"),
            "ANTHROPIC_API_KEY",
            {"CLAUDE_CODE_AUTH_DIR": "/tmp/legacy"},
        ),
        (
            OpenCodeAgent("openrouter/minimax/minimax-m2.7"),
            "OPENROUTER_API_KEY",
            {"OPENROUTER_API": "legacy"},
        ),
    ],
)
def test_published_agents_reject_legacy_only_auth(
    agent: object,
    key: str,
    legacy_env: dict[str, str],
) -> None:
    with mock.patch.dict(os.environ, legacy_env, clear=True):
        with pytest.raises(KeyError, match=key):
            _ = agent._env  # type: ignore[attr-defined]


def test_gemini_metadata_records_api_key_auth_only() -> None:
    agent = GeminiCliAgent("google/gemini-3.1-pro")

    assert agent._auth_metadata == {
        "agent_name": "gemini-cli",
        "mode_name": "api",
        "was_gemini_api_key_used": True,
        "was_gemini_api_auth_used": False,
    }


def test_codex_setup_logs_in_with_api_key() -> None:
    agent = CodexAgent("openai/gpt-5.5")
    script_path = agent._install_agent_script_path
    try:
        script = script_path.read_text(encoding="utf-8")
    finally:
        script_path.unlink()

    assert "codex login --with-api-key" in script
    assert "OPENAI_API_KEY" in script
    assert "CODEX_AUTH_JSON_BASE64" not in script
    assert "auth.json" not in script


def test_opencode_preserves_standard_openrouter_placeholder(tmp_path: Path) -> None:
    source_config = tmp_path / "opencode.json"
    source_config.write_text(
        '{"permission": {"bash": "deny"}, "key": "{env:OPENROUTER_API_KEY}"}',
        encoding="utf-8",
    )
    copied: dict[str, str] = {}
    session = mock.Mock()
    session.copy_to_container.side_effect = lambda path, **_kwargs: copied.update(
        text=Path(path).read_text(encoding="utf-8")
    )
    agent = OpenCodeAgent("openrouter/minimax/minimax-m2.7")

    with (
        mock.patch.object(OpenCodeAgent, "_HOST_CONFIG_PATH", source_config),
        mock.patch.object(agent, "_get_container_home", return_value="/root"),
    ):
        agent._copy_opencode_config(session)

    assert '"bash": "deny"' in source_config.read_text(encoding="utf-8")
    assert '"bash": "allow"' in copied["text"]
    assert "{env:OPENROUTER_API_KEY}" in copied["text"]
    assert "{env:OPENROUTER_API}" not in copied["text"]
