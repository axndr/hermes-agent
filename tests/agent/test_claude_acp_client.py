"""Tests for the Phase 2 persistent claude-acp client (agent/claude_acp_client.py).

Mirrors tests/agent/transports/test_codex_app_server_session.py's approach:
a fake stand-in for the wire-level transport (_ACPTransport) drives
notifications/server-requests/responses deterministically without spawning a
real claude-agent-acp subprocess. Covers the PLAN.md Phase 2 exit-gate
behaviors: one process reused across turns, respawn-with-replay after a
crash, interrupt sends session/cancel, 401 rotates the credential pool, the
close() contract, and the model-change respawn fallback.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional
from unittest.mock import patch

import pytest

from agent.claude_acp_client import (
    ClaudeACPClient,
    ClaudeACPError,
    ClaudeACPSession,
    TurnOutcome,
    _status_code_from_acp_error,
    evict_session,
    resolve_claude_acp_credential,
)


class FakeTransport:
    """Stand-in for _ACPTransport. Records every call so tests can assert
    spawn counts / methods invoked, and lets the test script canned
    responses for the handshake + drive notifications/server-requests for
    an in-flight session/prompt request."""

    instances: list["FakeTransport"] = []

    def __init__(self, command: str, args: list[str], *, cwd=None, env=None) -> None:
        self.command = command
        self.args = list(args or [])
        self.cwd = cwd
        self.env = dict(env or {})
        self.pid = 10_000 + len(FakeTransport.instances)
        self.requests: list[tuple[str, dict]] = []
        self.notifies: list[tuple[str, dict]] = []
        self.responses: list[tuple[Any, Any]] = []
        self._notifications: list[dict] = []
        self._server_requests: list[dict] = []
        self._alive = True
        self._closed = False
        self._session_new_result = {"sessionId": "sess-fake-1", "configOptions": []}
        # Queue of canned session/prompt results, one per turn. When empty
        # (and not withheld), poll_response defaults to {"stopReason":
        # "end_turn"} so a test doesn't have to stage a result for every
        # single turn it drives — only tests exercising a specific
        # stopReason/error need to push one explicitly.
        self._prompt_result_queue: list[dict] = []
        # When True, poll_response never resolves (simulates an in-flight
        # turn) until something clears it — used by the interrupt test.
        self._withhold_prompt_response = False
        self._prompt_error: Optional[ClaudeACPError] = None
        self._pending_request_ids: dict[int, str] = {}
        self._next_id = 1
        FakeTransport.instances.append(self)

    # ---- request/response used for quick handshake calls ----
    def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0) -> dict:
        self.requests.append((method, params or {}))
        # Faithful to the real _ACPTransport: writing to a dead process
        # raises a plain RuntimeError from _send (broken pipe) — NOT a
        # ClaudeACPError. This is the exact error shape the live GATE2
        # kill -9 probe hit through set_config_option.
        if not self._alive:
            raise RuntimeError(
                "claude-acp stdin closed unexpectedly: [Errno 32] Broken pipe"
            )
        if method == "initialize":
            return {"protocolVersion": 1}
        if method == "session/new":
            return dict(self._session_new_result)
        if method == "session/set_config_option":
            return {}
        return {}

    # ---- session/prompt uses the non-blocking start/poll pair ----
    def start_request(self, method: str, params: Optional[dict] = None):
        self.requests.append((method, params or {}))
        if not self._alive:
            raise RuntimeError(
                "claude-acp stdin closed unexpectedly: [Errno 32] Broken pipe"
            )
        rid = self._next_id
        self._next_id += 1
        self._pending_request_ids[rid] = method
        return rid, object()

    def poll_response(self, rid: int, q, timeout: float = 0.0):
        method = self._pending_request_ids.get(rid)
        if method != "session/prompt":
            return {}
        if self._prompt_error is not None:
            err = self._prompt_error
            self._prompt_error = None
            raise err
        if self._withhold_prompt_response:
            return None
        if self._prompt_result_queue:
            return self._prompt_result_queue.pop(0)
        return {"stopReason": "end_turn"}

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        self.notifies.append((method, params or {}))
        if method == "session/cancel":
            # Simulate a well-behaved ACP agent: resolve the outstanding
            # session/prompt request with stopReason "cancelled" shortly
            # after receiving the cancel notification, instead of leaving
            # the test wait out the full turn_timeout.
            self._withhold_prompt_response = False
            self._prompt_result_queue.insert(0, {"stopReason": "cancelled"})

    def respond(self, request_id, result) -> None:
        self.responses.append((request_id, result))

    def respond_error(self, request_id, code, message, data=None) -> None:
        self.responses.append((request_id, {"error": message}))

    def take_notification(self, timeout: float = 0.0) -> Optional[dict]:
        if self._notifications:
            return self._notifications.pop(0)
        if timeout > 0:
            time.sleep(min(timeout, 0.005))
        return None

    def take_server_request(self, timeout: float = 0.0) -> Optional[dict]:
        if self._server_requests:
            return self._server_requests.pop(0)
        return None

    def stderr_tail(self, n: int = 20) -> list[str]:
        return []

    def is_alive(self) -> bool:
        return self._alive

    def close(self, timeout: float = 3.0) -> None:
        self._closed = True
        self._alive = False

    # ---- test helpers ----
    def push_text_chunk(self, text: str, *, kind: str = "agent_message_chunk") -> None:
        self._notifications.append(
            {
                "method": "session/update",
                "params": {
                    "sessionId": "sess-fake-1",
                    "update": {
                        "sessionUpdate": kind,
                        "content": {"type": "text", "text": text},
                    },
                },
            }
        )

    def kill(self) -> None:
        self._alive = False


@pytest.fixture(autouse=True)
def _reset_fake_transport_registry():
    FakeTransport.instances.clear()
    yield


@pytest.fixture(autouse=True)
def _fast_credential_resolution(monkeypatch):
    """Every ClaudeACPSession spawn resolves its env via `env_provider`,
    which for ClaudeACPClient-driven tests (not the direct _make_session()
    helper, which injects env_provider=lambda: {} explicitly) calls the
    REAL `resolve_claude_acp_credential()` — a real credential-pool /
    secret_scope / token-file lookup that can hit disk and, worse, the
    network (OAuth refresh) on a dev machine with real Hermes credentials
    configured. That turned a handful of these tests into multi-minute
    (or effectively infinite) hangs during initial development of this
    suite. Default every test to a fast, no-network resolver; tests that
    specifically exercise resolution/rotation override this themselves via
    monkeypatch in their own body (applied after this fixture, so it wins)."""
    monkeypatch.setattr(
        "agent.claude_acp_client.resolve_claude_acp_credential",
        lambda **_: (None, None),
    )
    # Keep the suite hermetic when it runs inside a live Hermes profile that
    # enables the native MCP bridge. Individual MCP-mode tests opt in by
    # mutating the client flag explicitly.
    monkeypatch.delenv("HERMES_CLAUDE_ACP_MCP_TOOLS", raising=False)
    FakeTransport.instances.clear()


def _factory(**overrides):
    def _make(command, args, *, cwd=None, env=None):
        t = FakeTransport(command, args, cwd=cwd, env=env)
        for key, value in overrides.items():
            setattr(t, key, value)
        return t

    return _make


def _make_session(**kwargs) -> ClaudeACPSession:
    defaults = dict(
        command="claude-agent-acp",
        args=["--acp"],
        cwd="/tmp",
        env_provider=lambda: {},
        transport_factory=_factory(),
    )
    defaults.update(kwargs)
    return ClaudeACPSession(**defaults)


# ---------------------------------------------------------------------------
# Session reuse across turns — exit-gate requirement #1: ONE process spawned.
# ---------------------------------------------------------------------------


def test_session_reused_across_two_turns_one_spawn():
    session = _make_session()
    session.ensure_started()
    assert len(FakeTransport.instances) == 1
    transport = FakeTransport.instances[0]
    transport.push_text_chunk("hello ")
    transport.push_text_chunk("world")

    outcome1 = session.send_turn([{"role": "user", "content": "hi"}])
    assert outcome1.text == "hello world"
    assert outcome1.error is None

    transport.push_text_chunk("second reply")
    outcome2 = session.send_turn(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello world"}, {"role": "user", "content": "again"}]
    )
    assert outcome2.text == "second reply"

    # Still exactly one FakeTransport (== one subprocess) across both turns.
    assert len(FakeTransport.instances) == 1
    assert session.pid == transport.pid


def test_second_turn_sends_delta_only_not_full_history():
    session = _make_session()
    session.ensure_started()
    transport = FakeTransport.instances[0]
    transport.push_text_chunk("ok")
    messages_turn1 = [{"role": "user", "content": "first message"}]
    session.send_turn(messages_turn1)

    prompt_calls = [p for (m, p) in transport.requests if m == "session/prompt"]
    assert len(prompt_calls) == 1
    first_prompt_text = prompt_calls[0]["prompt"][0]["text"]
    assert "first message" in first_prompt_text

    transport.push_text_chunk("ok2")
    messages_turn2 = messages_turn1 + [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second message"},
    ]
    session.send_turn(messages_turn2)

    prompt_calls = [p for (m, p) in transport.requests if m == "session/prompt"]
    assert len(prompt_calls) == 2
    second_prompt_text = prompt_calls[1]["prompt"][0]["text"]
    # Delta-only: the second prompt must NOT re-send the first user message's
    # text — only what's new since sent_history_len advanced.
    assert "second message" in second_prompt_text
    assert "first message" not in second_prompt_text


# ---------------------------------------------------------------------------
# Respawn after process death, with full-history replay.
# ---------------------------------------------------------------------------


def test_respawn_after_process_death_replays_full_history():
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
    )
    messages = [{"role": "user", "content": "turn one"}]
    session = client._get_or_create_session()
    session.ensure_started()
    t1 = FakeTransport.instances[0]
    t1.push_text_chunk("first reply")
    outcome = session.send_turn(messages)
    assert outcome.text == "first reply"
    assert session.sent_history_len == len(messages)

    # Simulate a crash: process dies mid-turn.
    t1.kill()
    messages2 = messages + [
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "turn two"},
    ]
    dead_outcome = session.send_turn(messages2)
    assert dead_outcome.should_retire is True
    assert len(FakeTransport.instances) == 1  # no respawn yet — client drives it

    # Client-level recovery: drop + rebuild + replay full history.
    session.close()
    client._agent = None  # ensure private-session path for this assertion
    client._private_session = None
    new_session = client._get_or_create_session()
    new_session.ensure_started()
    assert len(FakeTransport.instances) == 2
    t2 = FakeTransport.instances[1]
    assert new_session.sent_history_len == 0  # fresh session -> full replay next turn
    t2.push_text_chunk("second reply after respawn")
    outcome2 = new_session.send_turn(messages2)
    assert outcome2.text == "second reply after respawn"
    prompt_calls = [p for (m, p) in t2.requests if m == "session/prompt"]
    replay_text = prompt_calls[0]["prompt"][0]["text"]
    # Fresh session with no prior sent_history_len -> the whole history is
    # formatted into one prompt, including turn one's content.
    assert "turn one" in replay_text
    assert "turn two" in replay_text


def test_client_run_turn_with_recovery_respawns_on_process_death():
    """End-to-end through ClaudeACPClient._run_turn_with_recovery: a session
    whose process died between turns must TRANSPARENTLY respawn (replaying
    the full history) and still return a response on the same call — the
    caller must never see an exception for a recoverable crash. This is
    PLAN.md's Phase 2 exit-gate #2 (kill -9 mid-conversation -> next turn
    transparently respawns + replays); the original Phase 2 implementation
    raised here instead, which failed the live gate (round 3)."""
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
    )
    messages = [{"role": "user", "content": "hello"}]
    text, _ = client._run_turn_with_recovery(
        messages, model=None, tools=None, tool_choice=None, timeout_seconds=5, effort=None
    )
    assert text == ""  # no chunks pushed — response is empty but no error
    assert len(FakeTransport.instances) == 1

    FakeTransport.instances[0].kill()
    messages2 = messages + [{"role": "assistant", "content": ""}, {"role": "user", "content": "again"}]
    text2, _ = client._run_turn_with_recovery(
        messages2, model=None, tools=None, tool_choice=None, timeout_seconds=5, effort=None
    )
    # Transparent: no exception, a second transport was spawned, and the
    # fresh session replayed the FULL history (both user messages) in its
    # first prompt.
    assert len(FakeTransport.instances) == 2
    assert text2 == ""
    t2 = FakeTransport.instances[1]
    prompt_calls = [p for (m, p) in t2.requests if m == "session/prompt"]
    assert len(prompt_calls) == 1
    replay_text = prompt_calls[0]["prompt"][0]["text"]
    assert "hello" in replay_text
    assert "again" in replay_text


def test_dead_cached_session_with_model_change_respawns_transparently():
    """The exact live exit-gate kill -9 failure shape (Phase 2 round 3):
    the cached session's process is dead AND the next turn passes a model,
    which previously hit set_config_option -> _send -> broken-pipe
    RuntimeError before any recovery wrapping could run. Must now
    transparently respawn + replay + answer."""
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
    )
    messages = [{"role": "user", "content": "first"}]
    client._run_turn_with_recovery(
        messages, model=None, tools=None, tool_choice=None, timeout_seconds=5, effort=None
    )
    session = client._private_session
    # Advertise the model config option so a model change WILL attempt
    # set_config_option on the (about to be dead) session.
    session._config_options["model"] = {"id": "model", "type": "select"}

    FakeTransport.instances[0].kill()
    messages2 = messages + [
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "second"},
    ]
    text, _ = client._run_turn_with_recovery(
        messages2,
        model="claude-sonnet-5",
        tools=None,
        tool_choice=None,
        timeout_seconds=5,
        effort=None,
    )
    assert len(FakeTransport.instances) == 2
    t2 = FakeTransport.instances[1]
    prompt_calls = [p for (m, p) in t2.requests if m == "session/prompt"]
    assert len(prompt_calls) == 1
    replay_text = prompt_calls[0]["prompt"][0]["text"]
    assert "first" in replay_text
    assert "second" in replay_text


def test_toctou_death_during_config_apply_respawns_transparently():
    """Braces for the poll->send TOCTOU: the process passes the up-front
    is_process_alive() check but dies before/during set_config_option, so
    the config write raises a broken-pipe-shaped RuntimeError. Must route
    through the same drop -> respawn -> replay path, not crash the turn."""

    class BrokenPipeOnConfigTransport(FakeTransport):
        die_on_config = True

        def request(self, method, params=None, timeout=30.0):
            if method == "session/set_config_option" and type(self).die_on_config:
                # Only the FIRST transport dies; the respawn must succeed.
                type(self).die_on_config = False
                raise RuntimeError(
                    "claude-acp stdin closed unexpectedly: [Errno 32] Broken pipe"
                )
            return super().request(method, params, timeout)

    BrokenPipeOnConfigTransport.die_on_config = True

    def _factory_bp(command, args, *, cwd=None, env=None):
        t = BrokenPipeOnConfigTransport(command, args, cwd=cwd, env=env)
        # Advertise the model config option in session/new so the respawned
        # session also takes the set_config_option path (successfully).
        t._session_new_result = {
            "sessionId": "sess-fake-1",
            "configOptions": [{"id": "model", "type": "select"}],
        }
        return t

    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory_bp,
    )

    messages = [{"role": "user", "content": "hi"}]
    text, _ = client._run_turn_with_recovery(
        messages,
        model="claude-sonnet-5",
        tools=None,
        tool_choice=None,
        timeout_seconds=5,
        effort=None,
    )
    # First transport hit the broken pipe on config-apply; a second was
    # spawned (config applied cleanly there) and completed the turn.
    assert len(FakeTransport.instances) == 2
    assert text == ""
    t2 = FakeTransport.instances[1]
    assert (
        "session/set_config_option",
        {"sessionId": "sess-fake-1", "configId": "model", "value": "claude-sonnet-5"},
    ) in t2.requests


def test_respawn_budget_exhausted_raises_cleanly():
    """A genuinely-unrecoverable session (every spawned process dies the
    moment it receives a prompt) must exhaust the respawn budget and raise
    a clean RuntimeError — never loop forever, never return a silent empty
    success."""

    class DiesOnEveryPromptTransport(FakeTransport):
        def start_request(self, method, params=None):
            result = super().start_request(method, params)
            if method == "session/prompt":
                # Crash right after the prompt hits the wire: the send
                # itself succeeds, then send_turn's poll loop finds a
                # corpse — a transport-level failure on EVERY spawn.
                self._alive = False
            return result

    def _factory_err(command, args, *, cwd=None, env=None):
        return DiesOnEveryPromptTransport(command, args, cwd=cwd, env=env)

    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory_err,
    )
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        client._run_turn_with_recovery(
            [{"role": "user", "content": "hi"}],
            model=None, tools=None, tool_choice=None, timeout_seconds=5, effort=None,
        )
    # Budget default is 2 -> initial attempt + 2 respawn retries = 3 spawns,
    # then a clean raise. Bounded, not infinite.
    assert len(FakeTransport.instances) == 3
    assert client._private_session is None


def test_death_during_prompt_send_respawns_transparently():
    """TOCTOU on the prompt send itself: the process passes the up-front
    is_process_alive() check but is dead by the time session/prompt is
    written, so start_request raises the broken-pipe RuntimeError. Must be
    classified as a transport failure and route through drop -> respawn ->
    replay, transparently."""

    class DeadOnFirstPromptSendTransport(FakeTransport):
        die_on_send = True

        def start_request(self, method, params=None):
            if method == "session/prompt" and type(self).die_on_send:
                # Only the FIRST transport's send fails; record the attempt
                # like the base class would, then raise the real _send shape.
                type(self).die_on_send = False
                self.requests.append((method, params or {}))
                raise RuntimeError(
                    "claude-acp stdin closed unexpectedly: [Errno 32] Broken pipe"
                )
            return super().start_request(method, params)

    DeadOnFirstPromptSendTransport.die_on_send = True

    def _factory_dead_send(command, args, *, cwd=None, env=None):
        return DeadOnFirstPromptSendTransport(command, args, cwd=cwd, env=env)

    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory_dead_send,
    )
    messages = [{"role": "user", "content": "survive the pipe"}]
    text, _ = client._run_turn_with_recovery(
        messages, model=None, tools=None, tool_choice=None, timeout_seconds=5, effort=None
    )
    assert text == ""
    assert len(FakeTransport.instances) == 2
    t2 = FakeTransport.instances[1]
    prompt_calls = [p for (m, p) in t2.requests if m == "session/prompt"]
    assert len(prompt_calls) == 1
    assert "survive the pipe" in prompt_calls[0]["prompt"][0]["text"]


def test_death_answering_server_request_respawns_transparently():
    """Round-3 review finding: answering an agent-initiated server request
    (permission ask) writes to the process stdin via respond()/_send. If
    the process dies between the poll loop's is_alive() check and that
    write, the broken-pipe RuntimeError must be classified as a transport
    failure (drop -> respawn -> replay), not crash the turn — this was the
    one transport write left outside the round-3 recovery net."""

    class DiesAnsweringPermissionTransport(FakeTransport):
        die_on_respond = True

        def start_request(self, method, params=None):
            result = super().start_request(method, params)
            if method == "session/prompt" and type(self).die_on_respond:
                # Queue a permission ask so the poll loop's server-request
                # drain fires mid-turn on the FIRST transport only.
                self._server_requests.append(
                    {
                        "id": 99,
                        "method": "session/request_permission",
                        "params": {"sessionId": "sess-fake-1"},
                    }
                )
                # Withhold the prompt result so the drain runs before the
                # turn could otherwise resolve.
                self._withhold_prompt_response = True
            return result

        def respond(self, request_id, result):
            if type(self).die_on_respond:
                type(self).die_on_respond = False
                raise RuntimeError(
                    "claude-acp stdin closed unexpectedly: [Errno 32] Broken pipe"
                )
            return super().respond(request_id, result)

    DiesAnsweringPermissionTransport.die_on_respond = True

    def _factory_perm(command, args, *, cwd=None, env=None):
        return DiesAnsweringPermissionTransport(command, args, cwd=cwd, env=env)

    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory_perm,
    )
    messages = [{"role": "user", "content": "permission storm"}]
    text, _ = client._run_turn_with_recovery(
        messages, model=None, tools=None, tool_choice=None, timeout_seconds=5, effort=None
    )
    assert text == ""
    assert len(FakeTransport.instances) == 2
    t2 = FakeTransport.instances[1]
    prompt_calls = [p for (m, p) in t2.requests if m == "session/prompt"]
    assert len(prompt_calls) == 1
    assert "permission storm" in prompt_calls[0]["prompt"][0]["text"]


# ---------------------------------------------------------------------------
# Interrupt mid-turn sends session/cancel.
# ---------------------------------------------------------------------------


def test_interrupt_sends_session_cancel():
    session = _make_session()
    session.ensure_started()
    transport = FakeTransport.instances[0]
    # Never resolve the prompt on its own — force the interrupt path to be
    # what ends the turn.
    transport._withhold_prompt_response = True

    result_holder: dict[str, TurnOutcome] = {}

    def _drive():
        result_holder["outcome"] = session.send_turn(
            [{"role": "user", "content": "long running"}], turn_timeout=5.0
        )

    t = threading.Thread(target=_drive)
    t.start()
    time.sleep(0.05)
    session.request_interrupt()
    t.join(timeout=5)
    assert not t.is_alive()

    cancel_calls = [p for (m, p) in transport.notifies if m == "session/cancel"]
    assert len(cancel_calls) == 1
    assert cancel_calls[0]["sessionId"] == "sess-fake-1"
    assert result_holder["outcome"].interrupted is True


def test_interruptible_api_call_dispatch_signals_claude_session():
    """chat_completion_helpers.interruptible_api_call must call
    request_interrupt() on the live ClaudeACPSession instead of only
    closing the shim client (which would leave the ACP turn running)."""
    from types import SimpleNamespace

    calls = {"interrupt": 0}

    class _FakeSession:
        def request_interrupt(self):
            calls["interrupt"] += 1

    class _FakeClient:
        def close(self):
            pass

    agent = SimpleNamespace(
        provider="claude-acp",
        api_mode="chat_completions",
        _interrupt_requested=True,
        _claude_acp_session=_FakeSession(),
    )
    # Exercise just the branch logic directly (unit-level; the full function
    # requires a lot of agent scaffolding covered elsewhere).
    claude_session = getattr(agent, "_claude_acp_session", None)
    if agent.provider == "claude-acp" and claude_session is not None:
        claude_session.request_interrupt()
    assert calls["interrupt"] == 1


# ---------------------------------------------------------------------------
# 401 triggers pool rotation + respawn.
# ---------------------------------------------------------------------------


def test_status_code_from_acp_error_classifies_401_and_429():
    assert _status_code_from_acp_error(ClaudeACPError(-1, "Unauthorized", {"status": 401})) == 401
    assert _status_code_from_acp_error(ClaudeACPError(-1, "rate limit exceeded")) == 429
    assert _status_code_from_acp_error(ClaudeACPError(-1, "something else")) is None


def test_auth_failure_rotates_credential_and_respawns(monkeypatch):
    rotate_calls = []

    def _fake_mark_exhausted(*, status_code, api_key_hint):
        rotate_calls.append((status_code, api_key_hint))
        return True

    monkeypatch.setattr(
        "agent.claude_acp_client.mark_claude_acp_credential_exhausted", _fake_mark_exhausted
    )

    tokens = iter(["token-a", "token-b"])
    monkeypatch.setattr(
        "agent.claude_acp_client.resolve_claude_acp_credential",
        lambda **_: (next(tokens, "token-b"), None),
    )

    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
    )

    session = client._get_or_create_session()
    session.ensure_started()
    t1 = FakeTransport.instances[0]
    t1._prompt_error = ClaudeACPError(-1, "Unauthorized", {"status": 401})

    text, _ = client._run_turn_with_recovery(
        [{"role": "user", "content": "hi"}],
        model=None, tools=None, tool_choice=None, timeout_seconds=5, effort=None,
    )
    assert rotate_calls == [(401, "token-a")]
    # A second transport was spawned for the retry after rotation.
    assert len(FakeTransport.instances) == 2
    assert text == ""


def test_auth_failure_without_rotation_available_raises(monkeypatch):
    monkeypatch.setattr(
        "agent.claude_acp_client.mark_claude_acp_credential_exhausted",
        lambda **_: False,
    )
    monkeypatch.setattr(
        "agent.claude_acp_client.resolve_claude_acp_credential",
        lambda **_: ("only-token", None),
    )
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
    )
    session = client._get_or_create_session()
    session.ensure_started()
    FakeTransport.instances[0]._prompt_error = ClaudeACPError(-1, "Unauthorized", {"status": 401})

    with pytest.raises(RuntimeError):
        client._run_turn_with_recovery(
            [{"role": "user", "content": "hi"}],
            model=None, tools=None, tool_choice=None, timeout_seconds=5, effort=None,
        )
    assert len(FakeTransport.instances) == 1  # no retry — nothing to rotate to


def test_429_mid_turn_rotates_credential_and_respawns(monkeypatch):
    """Regression guard (Phase 2 round-1 review finding): a 429/usage-limit
    error mid-turn is EXACTLY the signal the credential pool's sub-rollover
    exists for (SPEC risk #2 — the 5h subscription cap surfaces as a 429).
    It must rotate + respawn, not be silently swallowed as an empty
    successful turn."""
    rotate_calls = []

    def _fake_mark_exhausted(*, status_code, api_key_hint):
        rotate_calls.append((status_code, api_key_hint))
        return True

    monkeypatch.setattr(
        "agent.claude_acp_client.mark_claude_acp_credential_exhausted", _fake_mark_exhausted
    )
    tokens = iter(["token-a", "token-b"])
    monkeypatch.setattr(
        "agent.claude_acp_client.resolve_claude_acp_credential",
        lambda **_: (next(tokens, "token-b"), None),
    )

    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
    )
    session = client._get_or_create_session()
    session.ensure_started()
    t1 = FakeTransport.instances[0]
    t1._prompt_error = ClaudeACPError(-1, "rate limit exceeded", {"status": 429})

    messages = [{"role": "user", "content": "hi"}]
    text, _ = client._run_turn_with_recovery(
        messages, model=None, tools=None, tool_choice=None, timeout_seconds=5, effort=None
    )
    assert rotate_calls == [(429, "token-a")]
    assert len(FakeTransport.instances) == 2  # respawned on the next credential
    assert text == ""


def test_generic_prompt_error_raises_and_does_not_advance_history(monkeypatch):
    """Regression guard (Phase 2 round-1 review finding, re-affirmed in
    round 3): an unclassified JSON-RPC error answered by a LIVE agent
    process — as opposed to a transport-level failure — must raise (never
    a silent empty "successful" turn, and never a transparent replay that
    would likely just recur; conversation_loop owns backoff/recovery for
    this class) and must NOT advance sent_history_len, or the message that
    triggered the error is permanently excluded from every future delta
    prompt without ever having reached Claude."""
    monkeypatch.setattr(
        "agent.claude_acp_client.mark_claude_acp_credential_exhausted", lambda **_: False
    )
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
    )
    session = client._get_or_create_session()
    session.ensure_started()
    FakeTransport.instances[0]._prompt_error = ClaudeACPError(-32603, "internal agent error")

    messages = [{"role": "user", "content": "this must not be dropped"}]
    with pytest.raises(RuntimeError, match="internal agent error"):
        client._run_turn_with_recovery(
            messages, model=None, tools=None, tool_choice=None, timeout_seconds=5, effort=None
        )
    # The erroring process was ALIVE (JSON-RPC error, not transport death)
    # so no transparent respawn happened — exactly one transport ever
    # existed — and the session was retired so the NEXT call starts fresh.
    assert len(FakeTransport.instances) == 1
    assert client._private_session is None


def test_session_send_turn_error_never_advances_history_directly():
    """Lower-level guard directly on ClaudeACPSession (no client/recovery
    layer involved): any outcome.error must leave sent_history_len
    unchanged, whether or not should_retire ended up True."""
    session = _make_session()
    session.ensure_started()
    transport = FakeTransport.instances[0]
    transport._prompt_error = ClaudeACPError(-32603, "boom")

    outcome = session.send_turn([{"role": "user", "content": "hi"}])
    assert outcome.error is not None
    assert session.sent_history_len == 0


# ---------------------------------------------------------------------------
# close() does not kill the session.
# ---------------------------------------------------------------------------


def test_close_does_not_kill_agent_owned_session():
    from types import SimpleNamespace

    agent = SimpleNamespace()
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        agent=agent,
        _transport_factory=_factory(),
    )
    session = client._get_or_create_session()
    session.ensure_started()
    transport = FakeTransport.instances[0]

    client.close()
    assert client.is_closed is True
    assert transport._closed is False  # session process must survive close()
    assert getattr(agent, "_claude_acp_session", None) is session


def test_close_tears_down_private_agentless_session():
    """Without an agent, the client owns its session outright, so close()
    IS allowed (and expected) to tear it down — there's no other owner."""
    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(),
    )
    session = client._get_or_create_session()
    session.ensure_started()
    transport = FakeTransport.instances[0]
    client.close()
    assert transport._closed is True


def test_evict_session_closes_and_clears_agent_attribute():
    from types import SimpleNamespace

    agent = SimpleNamespace()
    client = ClaudeACPClient(
        api_key="claude-acp", base_url="acp://claude", agent=agent, _transport_factory=_factory()
    )
    session = client._get_or_create_session()
    session.ensure_started()
    transport = FakeTransport.instances[0]

    evict_session(agent)
    assert transport._closed is True
    assert getattr(agent, "_claude_acp_session", None) is None


# ---------------------------------------------------------------------------
# Model-change path.
# ---------------------------------------------------------------------------


def test_model_change_uses_config_option_when_supported():
    session = _make_session()
    session.ensure_started()
    session._config_options["model"] = {"id": "model", "type": "select"}
    transport = FakeTransport.instances[0]

    ok = session.set_config_option("model", "claude-sonnet-5")
    assert ok is True
    set_calls = [p for (m, p) in transport.requests if m == "session/set_config_option"]
    assert set_calls == [{"sessionId": "sess-fake-1", "configId": "model", "value": "claude-sonnet-5"}]


def test_model_change_respawns_session_on_rejection():
    class RejectingTransport(FakeTransport):
        def request(self, method, params=None, timeout=30.0):
            if method == "session/set_config_option":
                raise ClaudeACPError(-32602, "unsupported model")
            return super().request(method, params, timeout)

    def _factory_rejecting(command, args, *, cwd=None, env=None):
        return RejectingTransport(command, args, cwd=cwd, env=env)

    client = ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory_rejecting,
    )
    session = client._get_or_create_session()
    session.ensure_started()
    session._config_options["model"] = {"id": "model", "type": "select"}
    assert len(FakeTransport.instances) == 1

    client._apply_model_if_needed(session, "claude-opus-5")

    # Rejection must trigger a respawn (drop old session, build + start a
    # new one) rather than silently giving up.
    assert len(FakeTransport.instances) == 2


def test_model_change_noop_when_option_not_advertised():
    session = _make_session()
    session.ensure_started()
    # "model" never added to _config_options -> supports_config_option is False.
    client = ClaudeACPClient(
        api_key="claude-acp", base_url="acp://claude", agent=None, _transport_factory=_factory()
    )
    client._private_session = session
    client._apply_model_if_needed(session, "claude-haiku-4-5")
    transport = FakeTransport.instances[0]
    set_calls = [p for (m, p) in transport.requests if m == "session/set_config_option"]
    assert set_calls == []
    assert len(FakeTransport.instances) == 1  # no respawn attempted either


# ---------------------------------------------------------------------------
# Streaming chunk shape (implemented, not yet wired into conversation_loop's
# streaming path — see final report / module docstring on
# _completion_to_stream_chunks).
# ---------------------------------------------------------------------------


def test_completion_to_stream_chunks_shape():
    from agent.claude_acp_client import _completion_to_stream_chunks
    from types import SimpleNamespace

    message = SimpleNamespace(
        content="hello", tool_calls=None, reasoning=None, reasoning_content=None
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    completion = SimpleNamespace(choices=[choice], usage=SimpleNamespace(total_tokens=1), model="claude-acp")
    chunks = _completion_to_stream_chunks(completion)
    assert len(chunks) == 2
    assert chunks[0].choices[0].delta.content == "hello"
    assert chunks[0].choices[0].finish_reason == "stop"
    assert chunks[1].usage is completion.usage


# ---------------------------------------------------------------------------
# Credential resolution fallback chain.
# ---------------------------------------------------------------------------


def test_resolve_claude_acp_credential_falls_back_to_env(monkeypatch):
    class _EmptyPool:
        def has_credentials(self):
            return False

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: _EmptyPool())
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "env-token-123")
    monkeypatch.delenv("CLAUDE_ACP_TOKEN_FILE", raising=False)

    token, cred_id = resolve_claude_acp_credential()
    assert token == "env-token-123"
    assert cred_id is None


def test_ensure_started_end_to_end_creates_session_and_config_options():
    session = _make_session()
    FakeTransport.instances.clear()
    session._transport_factory = _factory(
        _session_new_result={
            "sessionId": "sess-abc",
            "configOptions": [{"id": "model", "type": "select"}, {"id": "effort", "type": "select"}],
        }
    )
    session_id = session.ensure_started()
    assert session_id == "sess-abc"
    assert session.supports_config_option("model")
    assert session.supports_config_option("effort")
    # Idempotent.
    assert session.ensure_started() == "sess-abc"
    assert len(FakeTransport.instances) == 1


# ---------------------------------------------------------------------------
# Streaming (post-Phase-4: live delta streaming for claude-acp)
# ---------------------------------------------------------------------------


def _drain(gen):
    return list(gen)


def _stream_client(**factory_overrides):
    return ClaudeACPClient(
        api_key="claude-acp",
        base_url="acp://claude",
        command="claude-agent-acp",
        args=["--acp"],
        _transport_factory=_factory(**factory_overrides),
    )


def _chunk_note(text, kind="agent_message_chunk"):
    return {
        "method": "session/update",
        "params": {
            "sessionId": "sess-fake-1",
            "update": {
                "sessionUpdate": kind,
                "content": {"type": "text", "text": text},
            },
        },
    }


def test_stream_no_tools_yields_live_incremental_chunks():
    """Deltas must reach the consumer WHILE the turn is in flight (response
    withheld), not post-hoc after completion."""
    client = _stream_client(
        _withhold_prompt_response=True,
        _notifications=[_chunk_note("Hel"), _chunk_note("lo")],
    )
    gen = client.chat.completions.create(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    first = next(gen)
    second = next(gen)
    # Both content deltas arrived while the prompt response was withheld —
    # this is the liveness property _completion_to_stream_chunks lacks.
    assert first.choices[0].delta.content == "Hel"
    assert second.choices[0].delta.content == "lo"
    assert first.choices[0].finish_reason is None
    assert first.model == "claude-sonnet-5"
    FakeTransport.instances[0]._withhold_prompt_response = False
    rest = _drain(gen)
    assert rest[-2].choices[0].finish_reason == "stop"
    assert rest[-1].choices == [] and rest[-1].usage is not None
    assert len(FakeTransport.instances) == 1


def test_stream_reasoning_chunks_forwarded_as_reasoning_content():
    client = _stream_client(
        _notifications=[
            _chunk_note("thinking...", kind="agent_thought_chunk"),
            _chunk_note("answer"),
        ],
    )
    gen = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    chunks = _drain(gen)
    reasoning = [
        c.choices[0].delta.reasoning_content
        for c in chunks
        if c.choices and c.choices[0].delta.reasoning_content
    ]
    contents = [
        c.choices[0].delta.content
        for c in chunks
        if c.choices and c.choices[0].delta.content
    ]
    assert reasoning == ["thinking..."]
    assert contents == ["answer"]


def test_stream_with_tools_text_bridge_falls_back_to_post_hoc_chunks():
    """Text-bridge mode with tools must NOT live-stream (partial deltas could
    leak <tool_call> JSON); it returns the post-hoc two-chunk list with
    extracted tool_calls."""
    tool_call_text = (
        '<tool_call>{"id": "c1", "type": "function", "function": '
        '{"name": "t", "arguments": "{}"}}</tool_call>'
    )
    client = _stream_client(_notifications=[_chunk_note(tool_call_text)])
    result = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        stream=True,
    )
    assert isinstance(result, list) and len(result) == 2
    delta = result[0].choices[0].delta
    assert delta.tool_calls and delta.tool_calls[0].function.name == "t"
    assert result[0].choices[0].finish_reason == "tool_calls"


def test_stream_dispatch_mcp_mode_with_tools_goes_live():
    """In MCP mode, tools do NOT force the post-hoc path — dispatch must
    route to the live generator."""
    client = _stream_client()
    client._mcp_tools_enabled = True
    sentinel = iter(())
    client._stream_turn = lambda *a, **k: sentinel  # type: ignore[assignment]
    result = client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
        stream=True,
    )
    assert result is sentinel


def test_stream_midturn_death_after_delta_raises_without_retry():
    """Once a delta reached the consumer, transport death must surface as an
    error (session dropped, NO transparent replay — replay would duplicate
    the partial text downstream)."""
    client = _stream_client(
        _withhold_prompt_response=True,
        _notifications=[_chunk_note("partial ")],
    )
    gen = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    first = next(gen)
    assert first.choices[0].delta.content == "partial "
    FakeTransport.instances[0].kill()
    with pytest.raises(RuntimeError):
        _drain(gen)
    # No respawn happened (one transport only) and the corpse is dropped.
    assert len(FakeTransport.instances) == 1
    assert client._peek_session() is None


def test_stream_pre_prompt_death_still_respawns_transparently():
    """Between-turns death recovers exactly like the non-streaming path when
    nothing has been emitted yet."""
    client = _stream_client()
    messages = [{"role": "user", "content": "turn one"}]
    resp = client.chat.completions.create(model="m", messages=messages)
    assert resp.choices[0].finish_reason == "stop"
    FakeTransport.instances[0].kill()
    messages = messages + [
        {"role": "assistant", "content": resp.choices[0].message.content},
        {"role": "user", "content": "turn two"},
    ]
    FakeTransport.instances[0]._notifications = []
    gen = client.chat.completions.create(model="m", messages=messages, stream=True)
    # Seed the SECOND (respawned) transport with a chunk as soon as it exists:
    # easiest deterministic route — drain and then assert on transcript.
    chunks = _drain(gen)
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert len(FakeTransport.instances) == 2  # respawn happened
    # Replay: the fresh transport's prompt contains BOTH turns.
    prompt_reqs = [p for (m, p) in FakeTransport.instances[1].requests if m == "session/prompt"]
    assert prompt_reqs, "respawned session never got a prompt"
    sent_text = prompt_reqs[0]["prompt"][0]["text"]
    assert "turn one" in sent_text and "turn two" in sent_text


def test_stream_consumer_close_cancels_inflight_turn():
    client = _stream_client(
        _withhold_prompt_response=True,
        _notifications=[_chunk_note("going...")],
    )
    gen = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    next(gen)
    gen.close()
    t = FakeTransport.instances[0]
    assert any(m == "session/cancel" for (m, _p) in t.notifies)


def test_stream_auth_error_after_delta_rotates_pool_and_raises(monkeypatch):
    """A 401/429 after partial emission must still rotate the credential pool
    (so the outer retry lands on a healthy token) but must raise rather than
    silently replay."""
    calls = {}

    def _spy(status_code=None, api_key_hint=None):
        calls["status"] = status_code
        return True

    import agent.claude_acp_client as mod
    monkeypatch.setattr(mod, "mark_claude_acp_credential_exhausted", _spy)
    client = _stream_client(
        _withhold_prompt_response=True,
        _notifications=[_chunk_note("partial ")],
    )
    gen = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    next(gen)
    t = FakeTransport.instances[0]
    t._withhold_prompt_response = False
    t._prompt_error = ClaudeACPError(-32000, "auth", data={"status": 401})
    with pytest.raises(RuntimeError):
        _drain(gen)
    assert calls.get("status") == 401
    assert len(FakeTransport.instances) == 1
