"""Focused regressions for the Copilot ACP shim safety layer."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.copilot_acp_client import CopilotACPClient


class _FakeProcess:
    def __init__(self) -> None:
        self.stdin = io.StringIO()


class CopilotACPClientSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = CopilotACPClient(acp_cwd="/tmp")

    def test_extracted_tool_calls_match_openai_sdk_shape(self) -> None:
        tool_response = (
            "I'll inspect that.\n"
            "<tool_call>"
            '{"id":"call_read","type":"function",'
            '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
            "</tool_call>"
        )

        with patch.object(self.client, "_run_prompt", return_value=(tool_response, "")):
            response = self.client._create_chat_completion(
                model="copilot-acp",
                messages=[{"role": "user", "content": "read README.md"}],
                tools=[
                    {
                        "type": "function",
                        "function": {"name": "read_file", "parameters": {}},
                    }
                ],
            )

        choice = response.choices[0]
        self.assertEqual(choice.finish_reason, "tool_calls")
        tool_call = choice.message.tool_calls[0]
        self.assertEqual(tool_call.id, "call_read")
        self.assertEqual(tool_call.function.name, "read_file")
        self.assertEqual(
            json.loads(tool_call.function.arguments),
            {"path": "README.md"},
        )
        self.assertEqual(dict(tool_call)["id"], "call_read")
        self.assertEqual(dict(tool_call.function)["name"], "read_file")
        self.assertEqual(choice.message.content, "I'll inspect that.")

    def test_stream_true_returns_iterable_text_chunks(self) -> None:
        with patch.object(self.client, "_run_prompt", return_value=("Hello from ACP", "")):
            stream = self.client._create_chat_completion(
                model="copilot-acp",
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )

        chunks = list(stream)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].choices[0].delta.content, "Hello from ACP")
        self.assertIsNone(chunks[0].choices[0].delta.tool_calls)
        self.assertEqual(chunks[0].choices[0].finish_reason, "stop")
        self.assertEqual(chunks[1].choices, [])
        self.assertEqual(chunks[1].usage.total_tokens, 0)

    def test_stream_true_preserves_tool_call_deltas(self) -> None:
        tool_response = (
            "<tool_call>"
            '{"id":"call_read","type":"function",'
            '"function":{"name":"read_file","arguments":"{\\"path\\":\\"README.md\\"}"}}'
            "</tool_call>"
        )

        with patch.object(self.client, "_run_prompt", return_value=(tool_response, "")):
            stream = self.client._create_chat_completion(
                model="copilot-acp",
                messages=[{"role": "user", "content": "read README.md"}],
                stream=True,
            )

        chunks = list(stream)
        delta = chunks[0].choices[0].delta
        self.assertIsNone(delta.content)
        self.assertEqual(chunks[0].choices[0].finish_reason, "tool_calls")
        self.assertEqual(len(delta.tool_calls), 1)
        tool_delta = delta.tool_calls[0]
        self.assertEqual(tool_delta.index, 0)
        self.assertEqual(tool_delta.id, "call_read")
        self.assertEqual(tool_delta.function.name, "read_file")
        self.assertEqual(
            json.loads(tool_delta.function.arguments),
            {"path": "README.md"},
        )
        self.assertEqual(chunks[1].choices, [])

    def test_timeout_object_is_coerced_for_streaming_requests(self) -> None:
        captured: dict[str, float] = {}

        def fake_run_prompt(
            prompt_text: str, *, timeout_seconds: float, model: str | None = None
        ) -> tuple[str, str]:
            captured["timeout"] = timeout_seconds
            return "ok", ""

        timeout = type(
            "TimeoutLike",
            (),
            {"read": 12.0, "write": 5.0, "connect": 3.0, "pool": 1.0},
        )()

        with patch.object(self.client, "_run_prompt", side_effect=fake_run_prompt):
            list(
                self.client._create_chat_completion(
                    model="copilot-acp",
                    messages=[{"role": "user", "content": "hello"}],
                    timeout=timeout,
                    stream=True,
                )
            )

        self.assertEqual(captured["timeout"], 12.0)

    def _dispatch(self, message: dict, *, cwd: str) -> dict:
        process = _FakeProcess()
        handled = self.client._handle_server_message(
            message,
            process=process,
            cwd=cwd,
            text_parts=[],
            reasoning_parts=[],
        )
        self.assertTrue(handled)
        payload = process.stdin.getvalue().strip()
        self.assertTrue(payload)
        return json.loads(payload)

    def test_request_permission_is_not_auto_allowed(self) -> None:
        response = self._dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/request_permission",
                "params": {},
            },
            cwd="/tmp",
        )

        outcome = (((response.get("result") or {}).get("outcome") or {}).get("outcome"))
        self.assertEqual(outcome, "cancelled")

    def test_read_text_file_blocks_internal_hermes_hub_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            blocked = home / ".hermes" / "skills" / ".hub" / "index-cache" / "entry.json"
            blocked.parent.mkdir(parents=True, exist_ok=True)
            blocked.write_text('{"token":"sk-test-secret-1234567890"}')

            with patch.dict(
                os.environ,
                {"HOME": str(home), "HERMES_HOME": str(home / ".hermes")},
                clear=False,
            ):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "fs/read_text_file",
                        "params": {"path": str(blocked)},
                    },
                    cwd=str(home),
                )

        self.assertIn("error", response)

    def test_read_text_file_redacts_sensitive_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            secret_file = root / "config.env"
            secret_file.write_text("OPENAI_API_KEY=sk-proj-abc123def456ghi789jkl012")

            # agent.redact snapshots HERMES_REDACT_SECRETS at import time into
            # _REDACT_ENABLED, so patching os.environ is a no-op. Flip the
            # module-level constant directly for the duration of the call.
            with patch("agent.redact._REDACT_ENABLED", True):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "fs/read_text_file",
                        "params": {"path": str(secret_file)},
                    },
                    cwd=str(root),
                )

        content = ((response.get("result") or {}).get("content") or "")
        self.assertNotIn("abc123def456", content)
        self.assertIn("OPENAI_API_KEY=", content)

    def test_write_text_file_reuses_write_denylist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            home = Path(tmpdir) / "home"
            target = home / ".ssh" / "id_rsa"
            target.parent.mkdir(parents=True, exist_ok=True)

            with patch(
                "agent.copilot_acp_client.get_write_denied_error",
                return_value="Write denied: protected",
                create=True,
            ):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "fs/write_text_file",
                        "params": {
                            "path": str(target),
                            "content": "fake-private-key",
                        },
                    },
                    cwd=str(home),
                )

        self.assertIn("error", response)
        self.assertFalse(target.exists())

    def test_write_text_file_respects_safe_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            safe_root = root / "workspace"
            safe_root.mkdir()
            outside = root / "outside.txt"

            with patch.dict(os.environ, {"HERMES_WRITE_SAFE_ROOT": str(safe_root)}, clear=False):
                response = self._dispatch(
                    {
                        "jsonrpc": "2.0",
                        "id": 5,
                        "method": "fs/write_text_file",
                        "params": {
                            "path": str(outside),
                            "content": "should-not-write",
                        },
                    },
                    cwd=str(root),
                )

        self.assertIn("error", response)
        self.assertIn("HERMES_WRITE_SAFE_ROOT", str(response["error"]))
        self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()


# ── HOME env propagation tests (from PR #11285) ─────────────────────

from unittest.mock import patch as _patch
import pytest


def _make_home_client(tmp_path):
    return CopilotACPClient(
        api_key="copilot-acp",
        base_url="acp://copilot",
        acp_command="copilot",
        acp_args=["--acp", "--stdio"],
        acp_cwd=str(tmp_path),
    )


def _fake_popen_capture(captured):
    def _fake(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        raise FileNotFoundError("copilot not found")
    return _fake


def test_run_prompt_preserves_real_home_when_profile_home_available(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    (hermes_home / "home").mkdir(parents=True)
    real_home = tmp_path / "real-home"
    real_home.mkdir()

    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_REAL_HOME", raising=False)

    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start copilot-acp command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert captured["kwargs"]["env"]["HOME"] == str(real_home)
    assert captured["kwargs"]["env"]["HERMES_REAL_HOME"] == str(real_home)


def test_run_prompt_passes_home_when_parent_env_is_clean(monkeypatch, tmp_path):
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)

    captured = {}
    client = _make_home_client(tmp_path)

    with _patch("agent.copilot_acp_client.subprocess.Popen", side_effect=_fake_popen_capture(captured)):
        with pytest.raises(RuntimeError, match="Could not start copilot-acp command"):
            client._run_prompt("hello", timeout_seconds=1)

    assert "env" in captured["kwargs"]
    assert captured["kwargs"]["env"]["HOME"]


# ---------------------------------------------------------------------------
# Provider-aware command resolution (hermes-claude-acp Phase 1 identity split)
#
# Live-deployment regression: a claude-acp agent whose construction path did
# not seed explicit command kwargs fell through to the module-level Copilot
# env-trio fallback and spawned `copilot`, failing with a GitHub-Copilot
# install hint. Every construction path must resolve claude-acp's command
# from ITS registry entry (HERMES_CLAUDE_ACP_COMMAND -> claude-agent-acp),
# and error messages must name the right provider.
# ---------------------------------------------------------------------------


def _clear_acp_env(monkeypatch):
    for var in (
        "HERMES_COPILOT_ACP_COMMAND",
        "COPILOT_CLI_PATH",
        "HERMES_COPILOT_ACP_ARGS",
        "HERMES_CLAUDE_ACP_COMMAND",
        "HERMES_CLAUDE_ACP_ARGS",
    ):
        monkeypatch.delenv(var, raising=False)


def _assert_claude_spawn_argv0(client, expected_argv0, expected_provider_in_error):
    """Drive a real ClaudeACPSession spawn attempt (Phase 2 persistent
    client) and assert argv[0] + error text."""
    captured = {}
    with _patch(
        "agent.claude_acp_client.subprocess.Popen",
        side_effect=_fake_popen_capture(captured),
    ):
        with pytest.raises(RuntimeError, match=expected_provider_in_error):
            client._get_or_create_session().ensure_started()
    assert captured["cmd"][0] == expected_argv0


def _assert_spawn_argv0(client, expected_argv0, expected_provider_in_error):
    """Drive a real _run_prompt spawn attempt and assert argv[0] + error text."""
    captured = {}
    with _patch(
        "agent.copilot_acp_client.subprocess.Popen",
        side_effect=_fake_popen_capture(captured),
    ):
        with pytest.raises(RuntimeError, match=expected_provider_in_error):
            client._run_prompt("hello", timeout_seconds=1)
    assert captured["cmd"][0] == expected_argv0


def test_claude_acp_client_spawns_claude_command_without_explicit_kwargs(monkeypatch, tmp_path):
    """The deployed-bug repro: base_url acp://claude, no command kwargs,
    HERMES_CLAUDE_ACP_COMMAND set — argv[0] must be the claude command,
    never `copilot`, and the failure hint must name claude-acp."""
    _clear_acp_env(monkeypatch)
    monkeypatch.setenv("HERMES_CLAUDE_ACP_COMMAND", "/bin/true")

    client = CopilotACPClient(base_url="acp://claude", acp_cwd=str(tmp_path))
    assert client._provider == "claude-acp"
    assert client._acp_command == "/bin/true"
    assert client._acp_args == []
    _assert_spawn_argv0(client, "/bin/true", "Could not start claude-acp command")


def test_claude_acp_client_agent_init_shape_kwargs(monkeypatch, tmp_path):
    """agent_init passes command=None/args=[] when the agent had no explicit
    acp_command — the client must still resolve claude-acp's registry command."""
    _clear_acp_env(monkeypatch)
    monkeypatch.setenv("HERMES_CLAUDE_ACP_COMMAND", "/bin/true")

    client = CopilotACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command=None,
        args=[],
        acp_cwd=str(tmp_path),
    )
    assert client._acp_command == "/bin/true"
    _assert_spawn_argv0(client, "/bin/true", "Could not start claude-acp command")


def test_claude_acp_client_defaults_to_claude_agent_acp(monkeypatch, tmp_path):
    """Without any env override, claude-acp resolves its registry default —
    claude-agent-acp with no args — not `copilot --acp --stdio`."""
    _clear_acp_env(monkeypatch)

    client = CopilotACPClient(base_url="acp://claude", acp_cwd=str(tmp_path))
    assert client._acp_command == "claude-agent-acp"
    assert client._acp_args == []


def test_copilot_acp_client_still_resolves_copilot(monkeypatch, tmp_path):
    """Mirror: copilot-acp keeps its stock resolution, unaffected by (and not
    reading) the claude-acp env var."""
    _clear_acp_env(monkeypatch)
    monkeypatch.setenv("HERMES_CLAUDE_ACP_COMMAND", "/should/not/leak")

    client = CopilotACPClient(base_url="acp://copilot", acp_cwd=str(tmp_path))
    assert client._provider == "copilot-acp"
    assert client._acp_command == "copilot"
    assert client._acp_args == ["--acp", "--stdio"]
    _assert_spawn_argv0(client, "copilot", "Could not start copilot-acp command")


def test_create_openai_client_path_resolves_claude_command(monkeypatch, tmp_path):
    """The per-request client recreation path (agent_runtime_helpers.
    create_openai_client) must produce a Phase-2 ClaudeACPClient (NOT
    CopilotACPClient — that dispatch was Phase 1's interim wiring, replaced
    once the persistent client landed) that spawns the claude command —
    including when the agent has no base_url at all."""
    from types import SimpleNamespace

    from agent.agent_runtime_helpers import create_openai_client
    from agent.claude_acp_client import ClaudeACPClient

    _clear_acp_env(monkeypatch)
    monkeypatch.setenv("HERMES_CLAUDE_ACP_COMMAND", "/bin/true")

    agent = SimpleNamespace(
        provider="claude-acp",
        base_url="acp://claude",
        _client_log_context=lambda: "test",
    )
    for kwargs in (
        {"api_key": "claude-acp", "base_url": "acp://claude"},
        {"api_key": "claude-acp", "base_url": ""},  # provider-only construction
    ):
        client = create_openai_client(agent, dict(kwargs), reason="test", shared=False)
        assert isinstance(client, ClaudeACPClient)
        assert client._acp_command == "/bin/true"
        _assert_claude_spawn_argv0(client, "/bin/true", "Could not start claude-acp command")


def test_create_openai_client_path_keeps_copilot_stock(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from agent.agent_runtime_helpers import create_openai_client

    _clear_acp_env(monkeypatch)

    agent = SimpleNamespace(
        provider="copilot-acp",
        base_url="acp://copilot",
        _client_log_context=lambda: "test",
    )
    client = create_openai_client(
        agent, {"api_key": "copilot-acp", "base_url": "acp://copilot"},
        reason="test", shared=False,
    )
    assert isinstance(client, CopilotACPClient)
    assert client._acp_command == "copilot"
    assert client._acp_args == ["--acp", "--stdio"]


def test_resolve_provider_client_path_resolves_claude_command(monkeypatch, tmp_path):
    """auxiliary_client.resolve_provider_client('claude-acp') must construct
    a Phase-2 ClaudeACPClient with the claude command from the registry
    credentials (superseding Phase 1's interim CopilotACPClient dispatch)."""
    from agent.auxiliary_client import resolve_provider_client
    from agent.claude_acp_client import ClaudeACPClient

    _clear_acp_env(monkeypatch)
    monkeypatch.setenv("HERMES_CLAUDE_ACP_COMMAND", "/bin/true")
    monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: command)

    client, model = resolve_provider_client(provider="claude-acp", model="claude-sonnet-5")
    assert isinstance(client, ClaudeACPClient)
    assert client._acp_command == "/bin/true"
    _assert_claude_spawn_argv0(client, "/bin/true", "Could not start claude-acp command")
