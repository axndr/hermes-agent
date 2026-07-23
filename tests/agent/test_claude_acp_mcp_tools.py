"""Tests for Phase 3 (native tools via MCP), PLAN.md / SPEC.md §2.4.

Covers, per the task's verification checklist:
  - the MCP server serves the turn's registry (real list_tools/call_tool
    round trip against `HermesACPMcpServer`, in-process, no fakes);
  - registry swap between turns (same URL/token, different tool set);
  - tool execution routes through the canonical Hermes executor
    (`model_tools.handle_function_call`), spied via monkeypatch;
  - session respawn recreates the MCP server (new instance, new
    port/token);
  - flag OFF => no server started, text bridge intact (byte-for-byte);
  - completion shape in MCP mode: `tool_calls=None`, `finish_reason="stop"`.

Uses the same `FakeTransport` fixture as
`tests/agent/test_claude_acp_client.py` (imported, not duplicated) so ACP
wire-level behavior (handshake, session/prompt, notifications) is driven
deterministically without a real claude-agent-acp subprocess. The MCP
server itself IS real (a real HermesACPMcpServer bound to 127.0.0.1 on an
ephemeral port, driven with a real `mcp` client session) — only the
claude-agent-acp side is faked, matching Phase 2's precedent of faking the
one component (the subprocess) that's expensive/flaky to spin up for every
test, while exercising everything else for real.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import patch

import pytest

from agent.claude_acp_client import (
    ClaudeACPClient,
    ClaudeACPSession,
    _mcp_tools_enabled,
)
from agent.transports.hermes_acp_mcp_server import HermesACPMcpServer
from tests.agent.test_claude_acp_client import FakeTransport, _factory, _make_session


ECHO_TOOL = {
    "type": "function",
    "function": {
        "name": "echo_tool",
        "description": "Echoes back its input.",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
}

OTHER_TOOL = {
    "type": "function",
    "function": {
        "name": "other_tool",
        "description": "A different tool.",
        "parameters": {"type": "object", "properties": {}},
    },
}


class FakeMcpServer:
    """Stand-in for HermesACPMcpServer, mirroring FakeTransport's pattern —
    records lifecycle/registry calls so tests can assert on them without a
    real HTTP server, for the tests that only care about ClaudeACPSession's
    wiring (start-before-session/new, set_tools-per-turn, stop-on-close)."""

    instances: list["FakeMcpServer"] = []

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.set_tools_calls: list[Optional[list[dict]]] = []
        # "before" / "after" / None — records whether the FIRST set_tools()
        # call happened before or after start(), for the round-1 review
        # finding #1 regression test (registry must be populated before
        # the server starts accepting connections / before session/new).
        self.set_tools_order_relative_to_start: Optional[str] = None
        self._token = f"fake-token-{len(FakeMcpServer.instances)}"
        self._port = 40000 + len(FakeMcpServer.instances)
        FakeMcpServer.instances.append(self)

    def start(self, timeout: float = 10.0) -> None:
        self.started = True

    def stop(self, timeout: float = 3.0) -> None:
        self.stopped = True

    def set_tools(self, tools) -> None:
        if self.set_tools_order_relative_to_start is None:
            self.set_tools_order_relative_to_start = "after" if self.started else "before"
        self.set_tools_calls.append(tools)

    @property
    def token(self) -> str:
        return self._token

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}/mcp"


def _fake_mcp_factory():
    def _make():
        return FakeMcpServer()

    return _make


def setup_function(_fn=None):
    FakeTransport.instances.clear()
    FakeMcpServer.instances.clear()


# ---------------------------------------------------------------------------
# Flag OFF: text bridge stays byte-for-byte unchanged.
# ---------------------------------------------------------------------------


def test_flag_off_by_default():
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("HERMES_CLAUDE_ACP_MCP_TOOLS", None)
        assert _mcp_tools_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on"])
def test_flag_truthy_values(value):
    with patch.dict("os.environ", {"HERMES_CLAUDE_ACP_MCP_TOOLS": value}):
        assert _mcp_tools_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "", "off", "no"])
def test_flag_falsy_values(value):
    with patch.dict("os.environ", {"HERMES_CLAUDE_ACP_MCP_TOOLS": value}):
        assert _mcp_tools_enabled() is False


def test_flag_off_no_mcp_server_started_and_text_bridge_intact():
    """With the flag off, ensure_started() must not touch the MCP-server
    factory at all, and send_turn must still inject the `<tool_call>` text
    protocol exactly like Phase 2."""
    session = _make_session(
        mcp_tools_enabled=False, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started()
    assert FakeMcpServer.instances == []  # never constructed

    transport = FakeTransport.instances[0]
    new_session_params = [p for (m, p) in transport.requests if m == "session/new"][0]
    assert new_session_params.get("mcpServers") in (None, [])

    transport.push_text_chunk("ok")
    session.send_turn(
        [{"role": "user", "content": "hi"}],
        tools=[ECHO_TOOL],
    )
    prompt_calls = [p for (m, p) in transport.requests if m == "session/prompt"]
    prompt_text = prompt_calls[0]["prompt"][0]["text"]
    assert "<tool_call>" in prompt_text
    assert "echo_tool" in prompt_text


# ---------------------------------------------------------------------------
# Flag ON: ClaudeACPSession wiring (server lifecycle + mcpServers plumbing).
# ---------------------------------------------------------------------------


def test_flag_on_starts_mcp_server_before_session_new():
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started()

    assert len(FakeMcpServer.instances) == 1
    fake_mcp = FakeMcpServer.instances[0]
    assert fake_mcp.started is True

    transport = FakeTransport.instances[0]
    new_session_params = [p for (m, p) in transport.requests if m == "session/new"][0]
    servers = new_session_params.get("mcpServers")
    assert isinstance(servers, list) and len(servers) == 1
    entry = servers[0]
    assert entry["name"] == "hermes-tools"
    assert entry["type"] == "http"
    assert entry["url"] == fake_mcp.url
    headers = {h["name"]: h["value"] for h in entry.get("headers", [])}
    assert headers.get("Authorization") == f"Bearer {fake_mcp.token}"


def test_flag_on_prompt_has_no_tool_call_injection():
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started()
    transport = FakeTransport.instances[0]
    transport.push_text_chunk("done")

    session.send_turn([{"role": "user", "content": "hi"}], tools=[ECHO_TOOL])

    prompt_calls = [p for (m, p) in transport.requests if m == "session/prompt"]
    prompt_text = prompt_calls[0]["prompt"][0]["text"]
    assert "<tool_call>" not in prompt_text
    # The tool schema is served over MCP, not inlined into the prompt text.
    assert "echo_tool" not in prompt_text


def test_flag_on_close_stops_mcp_server():
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started()
    fake_mcp = FakeMcpServer.instances[0]
    assert fake_mcp.stopped is False
    session.close()
    assert fake_mcp.stopped is True


def test_flag_on_populates_registry_before_session_new():
    """Round-1 review finding #1 (HIGH): claude-agent-acp's MCP client
    calls tools/list ONCE, at session/new time — not per turn. A registry
    that's still empty at that moment (populated only later, inside
    send_turn) means Claude never sees any Hermes tool for the session's
    whole life, live-probe-verified. `ensure_started(tools=...)` must set
    the server's registry BEFORE `session/new` is sent."""
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started(tools=[ECHO_TOOL])

    fake_mcp = FakeMcpServer.instances[0]
    # set_tools must have been called (with the first turn's tools) BEFORE
    # start() — never after, and never left empty.
    assert fake_mcp.set_tools_calls == [[ECHO_TOOL]]
    assert fake_mcp.set_tools_order_relative_to_start == "before"


def test_client_threads_tools_into_ensure_started():
    """End-to-end through ClaudeACPClient: the `tools` kwarg passed to
    `chat.completions.create` must reach `ensure_started`, not just
    `send_turn` — otherwise the first-turn session/new race reproduces."""
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
        _mcp_server_factory=_fake_mcp_factory(),
    )
    client._mcp_tools_enabled = True
    session = client._get_or_create_session()

    with patch.object(session, "ensure_started", wraps=session.ensure_started) as spy:
        transport_holder = []

        def _run():
            client._run_turn_with_recovery(
                [{"role": "user", "content": "hi"}],
                model=None, tools=[ECHO_TOOL], tool_choice=None,
                timeout_seconds=5, effort=None,
            )

        # send_turn needs a transport to exist with a queued response;
        # ensure_started() creates it, so just drive the real call.
        _run()
        spy.assert_called_once_with(tools=[ECHO_TOOL])

    fake_mcp = FakeMcpServer.instances[0]
    assert fake_mcp.set_tools_calls[0] == [ECHO_TOOL]


# ---------------------------------------------------------------------------
# Registry swap between turns.
# ---------------------------------------------------------------------------


def test_registry_swap_between_turns():
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started()
    fake_mcp = FakeMcpServer.instances[0]
    transport = FakeTransport.instances[0]

    transport.push_text_chunk("turn1 done")
    session.send_turn([{"role": "user", "content": "hi"}], tools=[ECHO_TOOL])
    assert fake_mcp.set_tools_calls[-1] == [ECHO_TOOL]

    transport.push_text_chunk("turn2 done")
    session.send_turn(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "turn1 done"},
            {"role": "user", "content": "again"},
        ],
        tools=[OTHER_TOOL],
    )
    assert fake_mcp.set_tools_calls[-1] == [OTHER_TOOL]
    # Same server instance (same URL/token) across both turns — the
    # registry changed, not the session/new-advertised endpoint.
    assert len(FakeMcpServer.instances) == 1


# ---------------------------------------------------------------------------
# Respawn recreates the MCP server.
# ---------------------------------------------------------------------------


def test_respawn_recreates_mcp_server_with_new_instance():
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
        _mcp_server_factory=_fake_mcp_factory(),
    )
    with patch.dict("os.environ", {"HERMES_CLAUDE_ACP_MCP_TOOLS": "1"}):
        client._mcp_tools_enabled = True  # deterministic regardless of env timing
        messages = [{"role": "user", "content": "hello"}]
        client._run_turn_with_recovery(
            messages, model=None, tools=[ECHO_TOOL], tool_choice=None,
            timeout_seconds=5, effort=None,
        )
        assert len(FakeMcpServer.instances) == 1
        first_mcp = FakeMcpServer.instances[0]
        assert first_mcp.started

        FakeTransport.instances[0].kill()
        messages2 = messages + [
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "again"},
        ]
        client._run_turn_with_recovery(
            messages2, model=None, tools=[ECHO_TOOL], tool_choice=None,
            timeout_seconds=5, effort=None,
        )
        assert len(FakeMcpServer.instances) == 2
        second_mcp = FakeMcpServer.instances[1]
        assert second_mcp is not first_mcp
        assert second_mcp.started
        # Old server was torn down when the dead session was dropped.
        assert first_mcp.stopped is True


# ---------------------------------------------------------------------------
# Completion shape in MCP mode: tool_calls=None, finish_reason="stop".
# ---------------------------------------------------------------------------


def test_mcp_mode_completion_has_no_tool_calls():
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
        _mcp_server_factory=_fake_mcp_factory(),
    )
    client._mcp_tools_enabled = True

    session = client._get_or_create_session()
    session.ensure_started()
    transport = FakeTransport.instances[0]
    # Even if the reply text happens to contain something <tool_call>-shaped
    # (e.g. Claude quoting it back, or a red-herring from a tool's own
    # output), MCP mode must never regex-extract it into tool_calls — tool
    # work already happened natively inside the ACP turn.
    transport.push_text_chunk("Sure, here's the answer. <tool_call>{\"looks\": \"like json\"}</tool_call>")

    completion = client.chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
        tools=[ECHO_TOOL],
    )
    choice = completion.choices[0]
    assert choice.message.tool_calls is None
    assert choice.finish_reason == "stop"
    assert "<tool_call>" in choice.message.content  # left untouched, not stripped


def test_non_mcp_mode_still_extracts_tool_calls():
    """Control: with the flag off, the exact same reply text is still
    regex-extracted as before (Phase 2 behavior unchanged)."""
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
    )
    client._mcp_tools_enabled = False

    session = client._get_or_create_session()
    session.ensure_started()
    transport = FakeTransport.instances[0]
    transport.push_text_chunk(
        '<tool_call>{"id": "1", "type": "function", '
        '"function": {"name": "echo_tool", "arguments": "{}"}}</tool_call>'
    )

    completion = client.chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
        tools=[ECHO_TOOL],
    )
    choice = completion.choices[0]
    assert choice.message.tool_calls is not None
    assert len(choice.message.tool_calls) == 1
    assert choice.finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Tool activity forwarded as reasoning text in MCP mode.
# ---------------------------------------------------------------------------


def _push_tool_call_notification(transport, *, title: str, status: str) -> None:
    """FakeTransport.push_text_chunk only builds agent_message_chunk /
    agent_thought_chunk-shaped updates (a bare text content block); a real
    `tool_call` / `tool_call_update` session/update carries `title` /
    `status` / `toolCallId` instead, so push it directly onto the same
    notification queue push_text_chunk uses."""
    transport._notifications.append(
        {
            "method": "session/update",
            "params": {
                "sessionId": "sess-fake-1",
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call-1",
                    "title": title,
                    "status": status,
                },
            },
        }
    )


def test_tool_call_notifications_forwarded_as_reasoning_in_mcp_mode():
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started()
    transport = FakeTransport.instances[0]
    _push_tool_call_notification(transport, title="echo_tool", status="in_progress")
    transport.push_text_chunk("final answer")

    outcome = session.send_turn([{"role": "user", "content": "hi"}], tools=[ECHO_TOOL])
    assert "echo_tool" in outcome.reasoning
    assert outcome.text == "final answer"


def test_tool_call_notifications_not_forwarded_when_mcp_disabled():
    session = _make_session(mcp_tools_enabled=False)
    session.ensure_started()
    transport = FakeTransport.instances[0]
    _push_tool_call_notification(transport, title="echo_tool", status="in_progress")
    transport.push_text_chunk("final answer")

    outcome = session.send_turn([{"role": "user", "content": "hi"}])
    assert outcome.reasoning == ""
    assert outcome.text == "final answer"


# ---------------------------------------------------------------------------
# Real HermesACPMcpServer: list_tools/call_tool round trip, dispatch seam.
# ---------------------------------------------------------------------------


def test_real_mcp_server_list_and_call_tool_round_trip():
    calls: list[tuple[str, dict]] = []

    def spy_dispatch(name: str, args: dict) -> str:
        calls.append((name, args))
        return f"result for {name}"

    server = HermesACPMcpServer(dispatch=spy_dispatch)
    server.set_tools([ECHO_TOOL])
    server.start()
    try:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async def _run():
            async with streamablehttp_client(
                server.url, headers={"Authorization": f"Bearer {server.token}"}
            ) as (read, write, _):
                async with ClientSession(read, write) as mcp_session:
                    await mcp_session.initialize()
                    listed = await mcp_session.list_tools()
                    names = [t.name for t in listed.tools]
                    result = await mcp_session.call_tool("echo_tool", {"text": "hi"})
                    return names, result

        names, result = asyncio.run(_run())
        assert names == ["echo_tool"]
        assert calls == [("echo_tool", {"text": "hi"})]
        text_blocks = [c.text for c in result.content if hasattr(c, "text")]
        assert any("result for echo_tool" in t for t in text_blocks)
    finally:
        server.stop()


def test_real_mcp_server_registry_swap_reflected_in_list_tools():
    server = HermesACPMcpServer(dispatch=lambda n, a: "x")
    server.set_tools([ECHO_TOOL])
    server.start()
    try:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async def _list_names():
            async with streamablehttp_client(
                server.url, headers={"Authorization": f"Bearer {server.token}"}
            ) as (read, write, _):
                async with ClientSession(read, write) as mcp_session:
                    await mcp_session.initialize()
                    listed = await mcp_session.list_tools()
                    return [t.name for t in listed.tools]

        assert asyncio.run(_list_names()) == ["echo_tool"]

        server.set_tools([OTHER_TOOL])
        assert asyncio.run(_list_names()) == ["other_tool"]
    finally:
        server.stop()


def test_real_mcp_server_rejects_missing_or_wrong_bearer_token():
    server = HermesACPMcpServer(dispatch=lambda n, a: "x")
    server.set_tools([ECHO_TOOL])
    server.start()
    try:
        import httpx

        r = httpx.post(
            server.url,
            json={},
            headers={"Accept": "application/json, text/event-stream"},
            follow_redirects=True,
            timeout=5,
        )
        assert r.status_code == 401

        r2 = httpx.post(
            server.url,
            json={},
            headers={
                "Accept": "application/json, text/event-stream",
                "Authorization": "Bearer wrong-token",
            },
            follow_redirects=True,
            timeout=5,
        )
        assert r2.status_code == 401
    finally:
        server.stop()


def test_real_mcp_server_bound_to_loopback_only():
    server = HermesACPMcpServer(dispatch=lambda n, a: "x")
    server.start()
    try:
        assert server.url.startswith("http://127.0.0.1:")
    finally:
        server.stop()


def test_dispatch_routes_through_canonical_hermes_executor(monkeypatch):
    """The default dispatch (used when ClaudeACPSession doesn't inject a
    test factory) must call `model_tools.handle_function_call` — the same
    function conversation_loop uses for every other provider's tool_calls
    and the Codex MCP bridge (`hermes_tools_mcp_server.py`) already reuses.
    Spied via monkeypatch rather than a real tool invocation."""
    import model_tools
    from agent.transports.hermes_acp_mcp_server import default_dispatch

    seen: list[tuple[str, dict]] = []

    def fake_handle_function_call(name, args, *a, **kw):
        seen.append((name, args))
        return "ok"

    monkeypatch.setattr(model_tools, "handle_function_call", fake_handle_function_call)
    result = default_dispatch("echo_tool", {"text": "hi"})
    assert result == "ok"
    assert seen == [("echo_tool", {"text": "hi"})]


# ---------------------------------------------------------------------------
# Permission gating: MCP-bridged tool calls approved, Claude's own
# built-ins still denied (round-1 review finding #2).
# ---------------------------------------------------------------------------


def _permission_request(title: str, *, req_id: int = 1) -> dict:
    return {
        "id": req_id,
        "method": "session/request_permission",
        "params": {
            "sessionId": "sess-fake-1",
            "toolCall": {"toolCallId": "call-1", "title": title},
            "options": [
                {"optionId": "allow-once-id", "kind": "allow_once", "name": "Allow once"},
                {"optionId": "reject-id", "kind": "reject_once", "name": "Reject"},
            ],
        },
    }


def test_mcp_bridged_tool_permission_request_is_approved():
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started()
    transport = FakeTransport.instances[0]

    req = _permission_request("mcp__hermes-tools__skills_list")
    session._handle_server_request(req)

    assert len(transport.responses) == 1
    rid, result = transport.responses[0]
    assert rid == 1
    assert result == {"outcome": {"outcome": "selected", "optionId": "allow-once-id"}}


def test_claude_own_tool_permission_request_still_cancelled_in_mcp_mode():
    """A permission request for one of Claude's own built-in tools (title
    has no `mcp__hermes-tools__` prefix) must stay denied even when MCP
    mode is on — only Hermes-bridged MCP tool calls are auto-approved."""
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started()
    transport = FakeTransport.instances[0]

    req = _permission_request("Bash")
    session._handle_server_request(req)

    assert transport.responses == [(1, {"outcome": {"outcome": "cancelled"}})]


def test_permission_request_cancelled_when_mcp_mode_off():
    """Flag off: identical behavior to Phase 2 — every permission request
    is cancelled, even one shaped like an MCP tool title (there IS no MCP
    server in this mode, so this shouldn't be reachable in practice, but
    the handler must not accidentally start approving things)."""
    session = _make_session(mcp_tools_enabled=False)
    session.ensure_started()
    transport = FakeTransport.instances[0]

    req = _permission_request("mcp__hermes-tools__skills_list")
    session._handle_server_request(req)

    assert transport.responses == [(1, {"outcome": {"outcome": "cancelled"}})]


def test_mcp_permission_request_falls_back_to_cancel_without_allow_option():
    """Defensive: if the agent's offered `options` somehow contain no
    allow-shaped option at all, fail closed (cancel) rather than crash or
    fabricate an option_id the agent never offered."""
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started()
    transport = FakeTransport.instances[0]

    req = {
        "id": 1,
        "method": "session/request_permission",
        "params": {
            "sessionId": "sess-fake-1",
            "toolCall": {"toolCallId": "call-1", "title": "mcp__hermes-tools__skills_list"},
            "options": [{"optionId": "reject-id", "kind": "reject_once", "name": "Reject"}],
        },
    }
    session._handle_server_request(req)

    assert transport.responses == [(1, {"outcome": {"outcome": "cancelled"}})]


def test_malformed_arguments_path_returns_error_not_exception():
    """A tool name not present in the current registry (e.g. Claude races
    a stale tools/list against a registry swap, or sends garbage) must
    produce a normal MCP tool-error content block, not crash the server or
    hang the turn."""
    server = HermesACPMcpServer(dispatch=lambda n, a: "should not be called")
    server.set_tools([ECHO_TOOL])
    server.start()
    try:
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async def _call_unknown():
            async with streamablehttp_client(
                server.url, headers={"Authorization": f"Bearer {server.token}"}
            ) as (read, write, _):
                async with ClientSession(read, write) as mcp_session:
                    await mcp_session.initialize()
                    return await mcp_session.call_tool("nonexistent_tool", {})

        result = asyncio.run(_call_unknown())
        text_blocks = [c.text for c in result.content if hasattr(c, "text")]
        assert any("unknown tool" in t for t in text_blocks)
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Hybrid loop-tool routing (found live: rook's memory tool errored via MCP —
# _AGENT_LOOP_TOOLS can only execute inside run_agent's loop)
# ---------------------------------------------------------------------------

MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "memory",
        "description": "store a memory",
        "parameters": {"type": "object", "properties": {}},
    },
}


def test_split_loop_tools():
    from agent.claude_acp_client import _split_loop_tools

    dispatcher, loop = _split_loop_tools([ECHO_TOOL, MEMORY_TOOL])
    assert [t["function"]["name"] for t in dispatcher] == ["echo_tool"]
    assert [t["function"]["name"] for t in loop] == ["memory"]
    assert _split_loop_tools(None) == ([], [])


def test_mcp_registry_excludes_loop_tools_and_prompt_carries_them():
    """MCP mode with a mixed tool list: the MCP registry must receive ONLY
    dispatcher tools, and the prompt must carry the <tool_call> protocol
    with ONLY the loop tools in its catalog."""
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started(tools=[ECHO_TOOL, MEMORY_TOOL])
    server = FakeMcpServer.instances[0]
    assert [t["function"]["name"] for t in server.set_tools_calls[0]] == ["echo_tool"]

    transport = FakeTransport.instances[0]
    transport.push_text_chunk("ok")
    session.send_turn(
        [{"role": "user", "content": "hi"}],
        tools=[ECHO_TOOL, MEMORY_TOOL],
    )
    # Per-turn registry swap also filtered.
    assert [t["function"]["name"] for t in server.set_tools_calls[-1]] == ["echo_tool"]
    prompt_params = [p for (m, p) in transport.requests if m == "session/prompt"][0]
    prompt_text = prompt_params["prompt"][0]["text"]
    assert "<tool_call>" in prompt_text  # protocol present for loop tools
    assert '"memory"' in prompt_text  # loop tool in catalog
    assert '"echo_tool"' not in prompt_text  # dispatcher tool NOT in catalog


def test_mcp_mode_pure_dispatcher_tools_keeps_plain_prompt():
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started(tools=[ECHO_TOOL])
    transport = FakeTransport.instances[0]
    transport.push_text_chunk("ok")
    session.send_turn([{"role": "user", "content": "hi"}], tools=[ECHO_TOOL])
    prompt_text = [p for (m, p) in transport.requests if m == "session/prompt"][0]["prompt"][0]["text"]
    assert "<tool_call>" not in prompt_text


def test_mcp_mode_completion_extracts_loop_tool_calls():
    """A loop-tool <tool_call> emitted in MCP mode must round-trip as real
    tool_calls on the completion so run_agent executes it in-loop."""
    from agent.claude_acp_client import ClaudeACPClient

    tool_call_text = (
        '<tool_call>{"id": "m1", "type": "function", "function": '
        '{"name": "memory", "arguments": "{}"}}</tool_call>'
    )
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(_notifications=[]),
        _mcp_server_factory=_fake_mcp_factory(),
    )
    client._mcp_tools_enabled = True
    FakeTransport_cls = FakeTransport  # after construction, seed the chunk
    resp = None
    # Build session first so we can seed the notification on ITS transport.
    session = client._get_or_create_session()
    session.ensure_started(tools=[MEMORY_TOOL])
    FakeTransport_cls.instances[0].push_text_chunk(tool_call_text)
    resp = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "remember"}],
        tools=[MEMORY_TOOL],
    )
    tc = resp.choices[0].message.tool_calls
    assert tc and tc[0].function.name == "memory"
    assert resp.choices[0].finish_reason == "tool_calls"


def test_mcp_mode_streaming_dispatch_respects_loop_tools():
    """stream=True in MCP mode: loop tools present -> post-hoc list (text
    protocol in play); dispatcher-only -> live generator."""
    from agent.claude_acp_client import ClaudeACPClient

    def _client():
        c = ClaudeACPClient(
            api_key="claude-acp",
            base_url="acp://claude",
            command="claude-agent-acp",
            args=["--acp"],
            _transport_factory=_factory(_notifications=[]),
            _mcp_server_factory=_fake_mcp_factory(),
        )
        c._mcp_tools_enabled = True
        return c

    c1 = _client()
    sentinel = iter(())
    c1._stream_turn = lambda *a, **k: sentinel  # type: ignore[assignment]
    assert (
        c1.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}],
            tools=[ECHO_TOOL], stream=True,
        )
        is sentinel
    )

    c2 = _client()
    result = c2.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}],
        tools=[ECHO_TOOL, MEMORY_TOOL], stream=True,
    )
    assert isinstance(result, list) and len(result) == 2


def test_loop_tool_fallback_set_matches_model_tools():
    """The ImportError fallback mirror in _split_loop_tools must not drift
    from the source of truth."""
    import inspect
    from model_tools import _AGENT_LOOP_TOOLS
    from agent import claude_acp_client as mod

    src = inspect.getsource(mod._split_loop_tools)
    # The fallback literal must contain exactly the canonical names.
    for name in _AGENT_LOOP_TOOLS:
        assert f'"{name}"' in src
    assert _AGENT_LOOP_TOOLS == {"todo", "memory", "session_search", "delegate_task"}


def test_hybrid_prompt_scopes_text_protocol_and_names_mcp():
    """The hybrid preamble must direct dispatcher tools to native MCP and
    scope <tool_call> to the loop catalog (both-channels hazard guard)."""
    session = _make_session(
        mcp_tools_enabled=True, mcp_server_factory=_fake_mcp_factory()
    )
    session.ensure_started(tools=[ECHO_TOOL, MEMORY_TOOL])
    transport = FakeTransport.instances[0]
    transport.push_text_chunk("ok")
    session.send_turn(
        [{"role": "user", "content": "hi"}], tools=[ECHO_TOOL, MEMORY_TOOL]
    )
    prompt_text = [p for (m, p) in transport.requests if m == "session/prompt"][0]["prompt"][0]["text"]
    assert "native MCP tools" in prompt_text
    assert "Never emit <tool_call> blocks for any other tool" in prompt_text
    assert '"memory"' in prompt_text and '"echo_tool"' not in prompt_text


def test_hybrid_extraction_drops_dispatcher_named_text_calls():
    """A <tool_call> naming a DISPATCHER tool in hybrid mode must be dropped
    (double-execution guard) while loop-tool calls survive."""
    from agent.claude_acp_client import ClaudeACPClient

    both = (
        '<tool_call>{"id": "d1", "type": "function", "function": '
        '{"name": "echo_tool", "arguments": "{}"}}</tool_call>'
        '<tool_call>{"id": "m1", "type": "function", "function": '
        '{"name": "memory", "arguments": "{}"}}</tool_call>'
    )
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
        _mcp_server_factory=_fake_mcp_factory(),
    )
    client._mcp_tools_enabled = True
    session = client._get_or_create_session()
    session.ensure_started(tools=[ECHO_TOOL, MEMORY_TOOL])
    FakeTransport.instances[0].push_text_chunk(both)
    resp = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "go"}],
        tools=[ECHO_TOOL, MEMORY_TOOL],
    )
    tc = resp.choices[0].message.tool_calls
    assert tc and len(tc) == 1 and tc[0].function.name == "memory"
