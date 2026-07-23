"""Persistent-session OpenAI-compatible client for the `claude-acp` provider.

Phase 2 of the claude-acp provider (see PLAN.md / SPEC.md in
hermes-claude-acp). Replaces the per-turn subprocess spawn in
`agent.copilot_acp_client.CopilotACPClient` (still used by `copilot-acp`,
untouched by this module) with a persistent ACP session whose lifetime
matches the Hermes conversation session, modeled on
`agent.transports.codex_app_server` + `codex_app_server_session` rather than
on `copilot_acp_client`.

Package decision (SPEC §2.1 / risk #4)
---------------------------------------
The pinned `agent-client-protocol==0.9.0` package (importable as `acp`) DOES
expose a usable *client*-side surface: `acp.spawn_agent_process`,
`acp.client.connection.ClientSideConnection`, and a large typed pydantic
schema (`acp.schema`) covering every ACP request/notification shape. But its
transport is asyncio end to end — `asyncio.subprocess`, `asyncio.StreamReader
/StreamWriter`, async method handlers (see `acp/stdio.py`,
`acp/connection.py`). Hermes' `AIAgent.run_conversation()` loop is
synchronous and single-threaded, and `codex_app_server.py` deliberately
avoided asyncio for a stdio child for exactly this reason: "layering asyncio
just to drive a stdio child creates surprising interrupt semantics." Standing
up a background event loop thread just to drive one subprocess would add a
second concurrency model to the codebase and would make `request_interrupt()`
(called from arbitrary Hermes threads — the interrupt-poll loop in
`chat_completion_helpers.interruptible_api_call`) fight the loop's thread
affinity for every cancellation.

Decision: hand-roll the JSON-RPC 2.0 framing with the proven codex
reader-thread architecture (`_Pending` id->queue routing, notification queue,
server-request queue — see `_ACPTransport` below, deliberately structured
like `CodexAppServerClient`), but reuse `acp.schema`'s typed pydantic models
to *build* request params and *parse* `session/update` notifications and
`configOptions`, so we are not hand-parsing untyped dicts for the wire shapes
that matter. This gets the "typed SessionUpdate objects" SPEC §2.1 asks for
without paying for the asyncio transport. Method name strings come from
`acp.meta.AGENT_METHODS` / `CLIENT_METHODS` (generated from the ACP schema),
not hand-typed literals.

Session lifetime & delta prompting
-----------------------------------
One `ClaudeACPSession` per Hermes `AIAgent` (cached as `agent._claude_acp_session`,
mirroring `agent._codex_session` in `agent/codex_runtime.py:305-329`). The ACP
`initialize` + `session/new` handshake happens once; each turn after that
sends only the *new* messages via `session/prompt` (delta prompting) — the
Claude process holds the running conversation. Hermes' `messages` list
remains the source of truth: on session (re)establishment (first turn, or a
respawn after a crash / credential rotation / model change), the client
replays the *entire* history it has not yet represented in the live process
as a single formatted prompt (reusing `copilot_acp_client._format_messages_as_prompt`,
the same formatting the old per-turn client used for every single turn), then
returns to delta sends.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

try:
    import orjson as _json_impl

    def _json_dumps(obj: Any) -> bytes:
        return _json_impl.dumps(obj)

    def _json_loads(data: Any) -> Any:
        return _json_impl.loads(data)
except ImportError:  # pragma: no cover - orjson is an optional accelerant
    import json as _json_impl

    def _json_dumps(obj: Any) -> bytes:
        return _json_impl.dumps(obj).encode("utf-8")

    def _json_loads(data: Any) -> Any:
        return _json_impl.loads(data)

from acp import schema as acp_schema
from acp.meta import AGENT_METHODS, CLIENT_METHODS, PROTOCOL_VERSION

from agent.copilot_acp_client import (
    CLAUDE_ACP_MARKER_BASE_URL,
    _extract_tool_calls_from_text,
    _format_messages_as_prompt,
    _registry_command_for_provider,
    _render_message_content,
)
from tools.environments.local import hermes_subprocess_env

logger = logging.getLogger(__name__)


def _split_loop_tools(
    tools: Optional[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split an OpenAI-schema tool list into (dispatcher_tools, loop_tools).

    A few tools ("todo", "memory", "session_search", "delegate_task") are
    handled INSIDE run_agent's conversation loop — they need agent-instance
    context, and model_tools.handle_function_call (the MCP server's
    execution seam) refuses them with "must be handled by the agent loop".
    In MCP mode those must therefore keep riding the <tool_call> text
    bridge so the loop executes them, while everything else is served
    natively over MCP. Found live: rook's memory tool erroring through the
    MCP path the first day the flag was flipped."""
    try:
        from model_tools import _AGENT_LOOP_TOOLS as loop_names
    except ImportError:  # pragma: no cover - model_tools always importable in-tree
        loop_names = {"todo", "memory", "session_search", "delegate_task"}
    dispatcher: list[dict[str, Any]] = []
    loop: list[dict[str, Any]] = []
    for t in tools or []:
        name = ""
        if isinstance(t, dict):
            fn = t.get("function") or {}
            if isinstance(fn, dict):
                name = str(fn.get("name") or "")
        (loop if name in loop_names else dispatcher).append(t)
    return dispatcher, loop


def _mcp_tools_enabled() -> bool:
    """SPEC §2.4 / PLAN Phase 3 feature flag. Truthy
    `HERMES_CLAUDE_ACP_MCP_TOOLS` switches claude-acp from the `<tool_call>`
    text bridge to native MCP tool exposure via ACP `session/new`'s
    `mcpServers`. Default OFF: today's text bridge is byte-for-byte
    unchanged unless this is explicitly set. Routed through
    `_scoped_get_secret` (defined below) so multiplexed profiles each see
    only their own setting, matching every other claude-acp config read."""
    val = (_scoped_get_secret("HERMES_CLAUDE_ACP_MCP_TOOLS", "") or "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _format_messages_hybrid(
    messages: list[dict[str, Any]],
    model: Optional[str] = None,
    loop_tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Any = None,
) -> str:
    """Hybrid MCP-mode prompt: dispatcher tools are native MCP; ONLY the
    agent-loop tools ride the `<tool_call>` text protocol. Unlike reusing
    copilot's `_format_messages_as_prompt` (whose preamble demands text
    blocks for EVERY tool action while MCP simultaneously advertises the
    same dispatcher tools natively — the both-channels/double-execution
    hazard `_format_messages_plain` documents against), this preamble
    scopes the text protocol to exactly the loop-tool catalog."""
    sections: list[str] = [
        "You are being used as the active ACP agent backend for Hermes.",
        "Hermes tools are exposed to you as native MCP tools for this "
        "session — call those directly with MCP tool calls, never as text.",
        "EXCEPTION: the following tools are NOT available via MCP. To use "
        "one of THESE tools (and only these), output a "
        "<tool_call>{...}</tool_call> block with JSON exactly in OpenAI "
        "function-call shape (one JSON object containing "
        "id/type/function{name,arguments}; arguments must be a JSON "
        "string). Never emit <tool_call> blocks for any other tool.",
    ]
    tool_specs: list[dict[str, Any]] = []
    for t in loop_tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        tool_specs.append(
            {
                "name": name.strip(),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    if tool_specs:
        sections.append(
            "Text-protocol tools (OpenAI function schema):\n"
            + json.dumps(tool_specs, ensure_ascii=False)
        )
    if tool_choice is not None:
        sections.append(f"Tool choice hint: {json.dumps(tool_choice, ensure_ascii=False)}")
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "context"
        rendered = _render_message_content(message.get("content"))
        if not rendered:
            continue
        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())


def _format_messages_plain(messages: list[dict[str, Any]], model: Optional[str] = None) -> str:
    """MCP-mode prompt formatter. Unlike `_format_messages_as_prompt`, this
    never injects the `<tool_call>{...}</tool_call>` instruction text or an
    inline OpenAI-schema tool catalog — in MCP mode Claude discovers and
    invokes Hermes tools natively via the MCP `tools/list` / `tools/call`
    round trip (served by `HermesACPMcpServer`), so prompting it to emit a
    parallel text-based tool-call format would be actively wrong (Claude
    might try to satisfy both channels) and is unnecessary."""
    sections: list[str] = [
        "You are being used as the active ACP agent backend for Hermes.",
        "Use ACP capabilities to complete tasks. Hermes tools available to "
        "you are exposed as native MCP tools for this session — call them "
        "directly with MCP tool calls, not text.",
    ]
    if model:
        sections.append(f"Hermes requested model hint: {model}")

    transcript: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown").strip().lower()
        if role not in {"system", "user", "assistant", "tool"}:
            role = "context"
        rendered = _render_message_content(message.get("content"))
        if not rendered:
            continue
        label = {
            "system": "System",
            "user": "User",
            "assistant": "Assistant",
            "tool": "Tool",
            "context": "Context",
        }.get(role, role.title())
        transcript.append(f"{label}:\n{rendered}")

    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))

    sections.append("Continue the conversation from the latest user request.")
    return "\n\n".join(section.strip() for section in sections if section and section.strip())

_DEFAULT_TURN_TIMEOUT_SECONDS = 900.0
_HANDSHAKE_TIMEOUT_SECONDS = 20.0
_NOTIFICATION_POLL_TIMEOUT = 0.2

# Name advertised for the Phase 3 in-process MCP server in `session/new`'s
# `mcpServers`. Claude's SDK prefixes tools it sources from a named MCP
# server as `mcp__<server-name>__<tool-name>` — both in `tool_call`
# session/update titles (used for reasoning-channel forwarding) and in
# `session/request_permission`'s `toolCall.title` (used by
# `_handle_server_request` to distinguish a Hermes-bridged MCP tool call,
# which must be auto-approved, from one of Claude's own built-in tools,
# which stays denied per the existing native-tools-off contract). Single
# source of truth so the two call sites can't drift.
_MCP_SERVER_NAME = "hermes-tools"
_MCP_TOOL_TITLE_PREFIX = f"mcp__{_MCP_SERVER_NAME}__"

# ACP session/update kinds that carry assistant-visible text. Everything else
# (tool_call_start/update, plan updates, mode/config updates, usage) is
# observed for bookkeeping but not folded into the final text — Hermes'
# tool bridge for claude-acp is still the text-based <tool_call> regex bridge
# (Phase 3 replaces this with native MCP tools), so tool activity Claude's
# own built-ins perform is surfaced via agent_message_chunk text same as any
# other model output.
_TEXT_CHUNK_KINDS = ("agent_message_chunk", "agent_thought_chunk")


class ClaudeACPError(RuntimeError):
    """Raised on a JSON-RPC error response from the claude-agent-acp process."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"claude-acp error {code}: {message}")
        self.code = code
        self.message = message
        self.data = data


@dataclass
class _Pending:
    queue: "queue.Queue"
    method: str
    sent_at: float = field(default_factory=time.time)


class _ACPTransport:
    """Minimal JSON-RPC 2.0 client for `claude-agent-acp` over stdio.

    Threading model mirrors `agent.transports.codex_app_server.CodexAppServerClient`
    exactly: one reader thread dispatches stdout lines to either a pending
    request's queue (replies), the notification queue (session/update etc.),
    or the server-request queue (session/request_permission, fs/*, terminal/*
    — all of which this phase declines, see `ClaudeACPSession._handle_server_request`).
    A second thread drains stderr for diagnostics. The caller drives
    request/response pairs from its own thread with polling reads so
    interrupts and liveness checks can interleave with a pending call
    instead of blocking on it (see `ClaudeACPSession._run_prompt_loop`).
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        *,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        spawn_env = hermes_subprocess_env(inherit_credentials=True)
        if env:
            spawn_env.update(env)
        cmd = [command] + list(args or [])
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                cwd=cwd,
                env=spawn_env,
            )
        except FileNotFoundError as exc:
            raise ClaudeACPError(
                -32000,
                f"Could not start claude-acp command {command!r}. "
                "Install claude-agent-acp or set HERMES_CLAUDE_ACP_COMMAND / "
                "HERMES_CLAUDE_ACP_ARGS.",
            ) from exc
        self._next_id = 1
        self._pending: dict[int, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._notifications: "queue.Queue" = queue.Queue()
        self._server_requests: "queue.Queue" = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()
        self._closed = False

        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid

    # ---------- send/receive ----------

    def request(self, method: str, params: Optional[dict] = None, timeout: float = 30.0) -> dict:
        """Blocking request/response. Only used for quick handshake calls
        (initialize, session/new, session/set_config_option) that resolve
        promptly — NOT for session/prompt, which can run for minutes and
        must stay interruptible (see `start_request` + `poll_response`)."""
        rid, q = self.start_request(method, params)
        try:
            msg = q.get(timeout=timeout)
        except queue.Empty:
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"claude-acp method {method!r} timed out after {timeout}s")
        return self._unwrap(msg)

    def start_request(self, method: str, params: Optional[dict] = None) -> tuple[int, "queue.Queue"]:
        """Send a request without blocking on the reply. Returns (id, queue)
        so the caller can poll `poll_response` on its own cadence."""
        rid = self._take_id()
        q: "queue.Queue" = queue.Queue(maxsize=1)
        with self._pending_lock:
            self._pending[rid] = _Pending(queue=q, method=method)
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        return rid, q

    def poll_response(self, rid: int, q: "queue.Queue", timeout: float = 0.0) -> Optional[dict]:
        """Non-blocking (or short-timeout) poll for a `start_request` reply.
        Returns None if not ready yet; raises ClaudeACPError on a JSON-RPC
        error result."""
        try:
            msg = q.get_nowait() if timeout <= 0 else q.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._pending_lock:
            self._pending.pop(rid, None)
        return self._unwrap(msg)

    @staticmethod
    def _unwrap(msg: dict) -> dict:
        if "error" in msg:
            err = msg["error"] or {}
            raise ClaudeACPError(err.get("code", -1), err.get("message", ""), err.get("data"))
        return msg.get("result") or {}

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def respond(self, request_id: Any, result: Any) -> None:
        self._send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def respond_error(self, request_id: Any, code: int, message: str, data: Any = None) -> None:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        self._send({"jsonrpc": "2.0", "id": request_id, "error": err})

    def take_notification(self, timeout: float = 0.0) -> Optional[dict]:
        try:
            return (
                self._notifications.get_nowait()
                if timeout <= 0
                else self._notifications.get(timeout=timeout)
            )
        except queue.Empty:
            return None

    def take_server_request(self, timeout: float = 0.0) -> Optional[dict]:
        try:
            return (
                self._server_requests.get_nowait()
                if timeout <= 0
                else self._server_requests.get(timeout=timeout)
            )
        except queue.Empty:
            return None

    # ---------- diagnostics ----------

    def stderr_tail(self, n: int = 20) -> list[str]:
        with self._stderr_lock:
            return list(self._stderr_lines[-n:])

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def close(self, timeout: float = 3.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                self._proc.kill()
                self._proc.wait(timeout=1.0)
            except Exception:
                pass
        except Exception:
            pass

    # ---------- internals ----------

    def _take_id(self) -> int:
        rid = self._next_id
        self._next_id += 1
        return rid

    def _send(self, obj: dict) -> None:
        if self._closed:
            raise RuntimeError("claude-acp transport is closed")
        if self._proc.stdin is None:
            raise RuntimeError("claude-acp transport stdin not available")
        try:
            self._proc.stdin.write(_json_dumps(obj) + b"\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            raise RuntimeError(f"claude-acp stdin closed unexpectedly: {exc}") from exc

    def _read_stdout(self) -> None:
        if self._proc.stdout is None:
            return
        try:
            for line in iter(self._proc.stdout.readline, b""):
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = _json_loads(line)
                except Exception:
                    with self._stderr_lock:
                        self._stderr_lines.append(f"<non-json on stdout> {line[:200]!r}")
                    continue
                self._dispatch(msg)
        except Exception as exc:  # pragma: no cover - defensive
            with self._stderr_lock:
                self._stderr_lines.append(f"<stdout reader error> {exc}")

    def _dispatch(self, msg: dict) -> None:
        if not isinstance(msg, dict):
            return
        if "id" in msg and ("result" in msg or "error" in msg):
            with self._pending_lock:
                pending = self._pending.pop(msg["id"], None)
            if pending is not None:
                try:
                    pending.queue.put_nowait(msg)
                except queue.Full:  # pragma: no cover - defensive
                    pass
            return
        if "id" in msg and "method" in msg:
            self._server_requests.put(msg)
            return
        if "method" in msg:
            self._notifications.put(msg)

    def _read_stderr(self) -> None:
        if self._proc.stderr is None:
            return
        try:
            for line in iter(self._proc.stderr.readline, b""):
                if not line:
                    break
                with self._stderr_lock:
                    self._stderr_lines.append(line.decode("utf-8", "replace").rstrip())
                    if len(self._stderr_lines) > 500:
                        self._stderr_lines = self._stderr_lines[-500:]
        except Exception:  # pragma: no cover - defensive
            pass


def _get_hermes_version() -> str:
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:  # pragma: no cover
        return "0.0.0"


@dataclass
class TurnOutcome:
    """Result of one `session/prompt` turn."""

    text: str = ""
    reasoning: str = ""
    stop_reason: Optional[str] = None
    interrupted: bool = False
    error: Optional[str] = None
    # Signals the caller (ClaudeACPClient) that the underlying process is
    # unhealthy and the session should be dropped so the next turn respawns
    # cleanly — mirrors TurnResult.should_retire in codex_app_server_session.
    should_retire: bool = False
    auth_failed: bool = False
    status_code: Optional[int] = None
    # True when the failure is transport-level (process died, stdin closed,
    # the prompt send itself failed, session object already torn down)
    # rather than a JSON-RPC-level error answered by a live agent process.
    # Transport failures are safely retryable on a respawned session with a
    # full history replay — sent_history_len never advanced, so nothing is
    # lost — whereas a JSON-RPC error from a live process (other than the
    # auth/limit ones handled via auth_failed) would likely just recur on
    # replay and must surface to the caller instead (conversation_loop owns
    # backoff/recovery for those).
    transport_failed: bool = False


class ClaudeACPSession:
    """One claude-agent-acp process per Hermes session.

    Not thread-safe — one caller drives `send_turn` at a time (matches
    AIAgent.run_conversation()'s single-threaded turn loop); `request_interrupt()`
    is the one method safe to call from another thread mid-turn.
    """

    def __init__(
        self,
        *,
        command: str,
        args: list[str],
        cwd: Optional[str] = None,
        env_provider: Callable[[], dict[str, str]],
        native_tools_env: Optional[dict[str, str]] = None,
        transport_factory: Optional[Callable[..., _ACPTransport]] = None,
        mcp_tools_enabled: bool = False,
        mcp_server_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._command = command
        self._args = list(args or [])
        self._cwd = cwd or os.getcwd()
        self._env_provider = env_provider
        self._native_tools_env = dict(native_tools_env or {})
        # Injectable for tests (mirrors CodexAppServerSession.client_factory)
        # — a fake standing in for _ACPTransport lets tests drive
        # notifications/server-requests/timing deterministically without a
        # real subprocess.
        self._transport_factory = transport_factory or _ACPTransport
        # Phase 3 (PLAN §3 / SPEC §2.4): when set, `ensure_started()` spawns
        # an in-process MCP server and advertises it in `session/new`'s
        # `mcpServers`, and `send_turn()` swaps its registry to the turn's
        # `tools` instead of injecting `<tool_call>` prompt text.
        # `mcp_server_factory` is injectable for tests (a fake standing in
        # for `HermesACPMcpServer`), mirroring `transport_factory`.
        self._mcp_tools_enabled = bool(mcp_tools_enabled)
        self._mcp_server_factory = mcp_server_factory
        self._mcp_server: Any = None

        self._transport: Optional[_ACPTransport] = None
        self._session_id: Optional[str] = None
        self._config_options: dict[str, dict[str, Any]] = {}
        self._current_model: Optional[str] = None
        self._current_effort: Optional[str] = None
        self._interrupt_event = threading.Event()
        self._closed = False
        # Number of Hermes `messages` entries already represented inside the
        # live claude-agent-acp process. 0 means "nothing sent yet" — the
        # next send_turn call must replay the full history instead of a
        # delta.
        self.sent_history_len = 0
        self.pid: Optional[int] = None

    # ---------- lifecycle ----------

    def is_started(self) -> bool:
        return self._session_id is not None

    def is_process_alive(self) -> bool:
        """Cheap liveness probe on the underlying subprocess (poll(), no
        I/O). Used by the client before reusing a cached session so a
        process that died BETWEEN turns (crash, kill -9, OOM) is detected
        up front instead of surfacing as a broken-pipe RuntimeError from
        the first send on it. Inherently TOCTOU — the process can die
        between this check and the next write — so callers must ALSO catch
        transport-level errors on every pre-response step; this check just
        makes the common between-turns death cheap and clean."""
        return self._transport is not None and self._transport.is_alive()

    def ensure_started(self, *, tools: Optional[list[dict[str, Any]]] = None) -> str:
        """Spawn + handshake if not already running. Idempotent.

        `tools` (MCP mode only — SPEC §2.4 round-1 review finding #1):
        claude-agent-acp's MCP client connects to our server and calls
        `tools/list` exactly ONCE, at `session/new` time — NOT per turn as
        this module originally assumed. A registry that's still empty at
        that moment (populated only later, inside the first `send_turn`)
        means Claude never sees ANY Hermes tool for the life of the
        session, live-probe-verified. So the very first turn's tool
        registry must be populated on the MCP server BEFORE `session/new`
        is sent, not after. `send_turn()` still calls `set_tools()` on
        every turn (harmless, and positions us to pick up a future
        claude-agent-acp version that does honor a live registry change),
        but a known limitation of the current agent behavior is that a
        tool set added ONLY on a later turn (never present on the first
        turn that established the session) will not become visible until
        the session is next respawned — documented in
        hermes-claude-acp/docs/phase3-design.md."""
        if self._session_id is not None:
            return self._session_id
        env = dict(self._env_provider())
        env.update(self._native_tools_env)
        self._transport = self._transport_factory(self._command, self._args, cwd=self._cwd, env=env)
        self.pid = self._transport.pid

        init_params = acp_schema.InitializeRequest(
            protocolVersion=PROTOCOL_VERSION,
            clientCapabilities=acp_schema.ClientCapabilities(
                fs=acp_schema.FileSystemCapabilities(readTextFile=False, writeTextFile=False),
                terminal=False,
                # The schema's own declared default for `auth` is a raw
                # dict ({"terminal": False}) rather than an AuthCapabilities
                # instance, which trips a pydantic serializer warning on
                # every dump if left unset. We don't advertise auth
                # capabilities in Phase 2 (token comes from the credential
                # pool at spawn time, not an ACP authenticate() round-trip)
                # so explicitly clear it instead of inheriting the noisy
                # default.
                auth=None,
            ),
            clientInfo=acp_schema.Implementation(
                name="hermes", title="Hermes Agent", version=_get_hermes_version()
            ),
        ).model_dump(mode="json", by_alias=True, exclude_none=True)
        self._transport.request(
            AGENT_METHODS["initialize"], init_params, timeout=_HANDSHAKE_TIMEOUT_SECONDS
        )

        mcp_servers: list[dict[str, Any]] = []
        if self._mcp_tools_enabled:
            dispatcher_tools, _loop = _split_loop_tools(tools)
            mcp_servers = [self._start_mcp_server(dispatcher_tools)]

        new_session_params = acp_schema.NewSessionRequest(
            cwd=self._cwd, mcpServers=mcp_servers
        ).model_dump(mode="json", by_alias=True, exclude_none=True)
        result = self._transport.request(
            AGENT_METHODS["session_new"], new_session_params, timeout=_HANDSHAKE_TIMEOUT_SECONDS
        )
        session_id = result.get("sessionId")
        if not session_id:
            raise ClaudeACPError(-32603, f"session/new returned no sessionId (keys={sorted(result.keys())})")
        self._session_id = session_id
        for opt in result.get("configOptions") or []:
            opt_id = opt.get("id")
            if opt_id:
                self._config_options[opt_id] = opt
        self.sent_history_len = 0
        logger.info(
            "claude-acp session started: id=%s pid=%s cwd=%s",
            str(session_id)[:8], self.pid, self._cwd,
        )
        return session_id

    def _start_mcp_server(self, tools: Optional[list[dict[str, Any]]]) -> dict[str, Any]:
        """Build + start the per-session `HermesACPMcpServer` and return its
        ACP `HttpMcpServer` entry (typed via `acp_schema`, per the module's
        "typed, not hand-parsed dicts" convention). Bearer token is carried
        in the `headers` field (SPEC §2.4's auth requirement — verified
        supported by claude-agent-acp's http/sse mcpServers branch).

        `tools` is set on the registry BEFORE `start()`'s caller sends
        `session/new` — see `ensure_started`'s docstring for why this
        ordering is load-bearing, not cosmetic."""
        if self._mcp_server_factory is not None:
            self._mcp_server = self._mcp_server_factory()
        else:
            from agent.transports.hermes_acp_mcp_server import HermesACPMcpServer

            self._mcp_server = HermesACPMcpServer()
        self._mcp_server.set_tools(tools)
        self._mcp_server.start()
        logger.info(
            "claude-acp: started in-process MCP tool server at %s", self._mcp_server.url
        )
        return acp_schema.HttpMcpServer(
            name=_MCP_SERVER_NAME,
            type="http",
            url=self._mcp_server.url,
            headers=[
                acp_schema.HttpHeader(
                    name="Authorization", value=f"Bearer {self._mcp_server.token}"
                )
            ],
        ).model_dump(mode="json", by_alias=True, exclude_none=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._mcp_server is not None:
            try:
                self._mcp_server.stop()
            except Exception:  # pragma: no cover - best-effort
                pass
            self._mcp_server = None
        if self._transport is not None:
            try:
                self._transport.close()
            except Exception:  # pragma: no cover - best-effort
                pass
            self._transport = None
        self._session_id = None
        self.sent_history_len = 0

    def __enter__(self) -> "ClaudeACPSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------- interrupt ----------

    def request_interrupt(self) -> None:
        """Idempotent. Signals the in-flight `send_turn` poll loop to send
        ACP `session/cancel` and unwind instead of waiting out the turn
        timeout. Safe to call from any thread."""
        self._interrupt_event.set()

    # ---------- config (model/effort) ----------

    def config_options(self) -> dict[str, dict[str, Any]]:
        return dict(self._config_options)

    def supports_config_option(self, option_id: str) -> bool:
        return option_id in self._config_options

    def set_config_option(self, option_id: str, value_id: str, *, timeout: float = 15.0) -> bool:
        """Set a `configOptions` selector (id in {"model", "effort"} per
        claude-agent-acp 0.56.0) on the live session. Returns True on
        success. Raises ClaudeACPError on a hard rejection so the caller can
        decide whether to fall back to a session respawn (SPEC risk #1)."""
        if self._transport is None or self._session_id is None:
            raise ClaudeACPError(-32000, "claude-acp session not started")
        params = acp_schema.SetSessionConfigOptionSelectRequest(
            sessionId=self._session_id, configId=option_id, value=value_id
        ).model_dump(mode="json", by_alias=True, exclude_none=True)
        self._transport.request(
            AGENT_METHODS["session_set_config_option"], params, timeout=timeout
        )
        return True

    # ---------- per-turn ----------

    def send_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        model: Optional[str] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Any = None,
        turn_timeout: float = _DEFAULT_TURN_TIMEOUT_SECONDS,
        on_delta: Optional[Any] = None,
    ) -> TurnOutcome:
        """Send the not-yet-represented tail of `messages` as a delta prompt
        (or the full formatted history on a fresh session) and block until
        the turn completes, interrupts, or errors.
        """
        outcome = TurnOutcome()
        if self._transport is None or self._session_id is None:
            outcome.error = "claude-acp session not started"
            outcome.should_retire = True
            outcome.transport_failed = True
            return outcome

        loop_tools: list[dict[str, Any]] = []
        if self._mcp_tools_enabled:
            dispatcher_tools, loop_tools = _split_loop_tools(tools)
            if self._mcp_server is not None:
                # Registry swap: serve exactly THIS turn's DISPATCHER tools.
                # Cheap (dict replace under a lock in HermesACPMcpServer, no
                # server restart) — the ACP session and its mcpServers
                # URL/token were fixed once at session/new; only the
                # behavior behind that URL changes turn to turn. Agent-loop
                # tools (memory/todo/...) are excluded: the MCP execution
                # seam can't run them (see _split_loop_tools) — they go via
                # the <tool_call> text protocol below instead.
                self._mcp_server.set_tools(dispatcher_tools)

        is_fresh_session = self.sent_history_len == 0
        if is_fresh_session:
            delta_messages = messages
        else:
            delta_messages = messages[self.sent_history_len :]
            if not delta_messages:
                # Nothing new to send (can happen on a retry with identical
                # history) — resend the last message so the model still gets
                # a turn to respond to, matching the old per-turn client's
                # behavior of always sending *something*.
                delta_messages = messages[-1:]

        if self._mcp_tools_enabled:
            if loop_tools:
                # Hybrid: dispatcher tools are discovered natively via MCP,
                # but agent-loop tools (memory/todo/session_search/
                # delegate_task) can only execute inside run_agent's loop —
                # inject the <tool_call> protocol with ONLY those in the
                # catalog so the loop still gets its round trip.
                prompt_text = _format_messages_hybrid(
                    delta_messages, model=model, loop_tools=loop_tools,
                    tool_choice=tool_choice,
                )
            else:
                # Pure MCP: never inject the `<tool_call>` text protocol or
                # an inline tool catalog (SPEC §2.4).
                prompt_text = _format_messages_plain(delta_messages, model=model)
        else:
            prompt_text = _format_messages_as_prompt(
                delta_messages, model=model, tools=tools, tool_choice=tool_choice
            )

        self._interrupt_event.clear()
        prompt_params = acp_schema.PromptRequest(
            sessionId=self._session_id,
            prompt=[acp_schema.TextContentBlock(type="text", text=prompt_text)],
        ).model_dump(mode="json", by_alias=True, exclude_none=True)

        try:
            rid, q = self._transport.start_request(AGENT_METHODS["session_prompt"], prompt_params)
        except Exception as exc:
            outcome.error = f"claude-acp session/prompt failed to send: {exc}"
            outcome.should_retire = True
            outcome.transport_failed = True
            return outcome

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        deadline = time.monotonic() + turn_timeout
        cancel_sent = False

        while time.monotonic() < deadline:
            if self._interrupt_event.is_set() and not cancel_sent:
                self._issue_cancel()
                cancel_sent = True
                outcome.interrupted = True

            if not self._transport.is_alive():
                outcome.error = self._format_stderr_error("claude-agent-acp process exited unexpectedly")
                outcome.should_retire = True
                outcome.transport_failed = True
                break

            # Drain server-initiated requests (permission asks etc.) so the
            # agent process isn't blocked waiting on us. Answering one
            # writes to the process stdin (respond/respond_error -> _send),
            # so the process dying in the window after the is_alive() check
            # above surfaces here as the broken-pipe RuntimeError — classify
            # it as a transport failure like every other write path (round-3
            # review finding: this drain was the one transport write left
            # outside the recovery net).
            sreq = self._transport.take_server_request(timeout=0)
            if sreq is not None:
                try:
                    self._handle_server_request(sreq)
                except (RuntimeError, OSError) as exc:
                    outcome.error = self._format_stderr_error(
                        f"claude-agent-acp died answering a server request: {exc}"
                    )
                    outcome.should_retire = True
                    outcome.transport_failed = True
                    break
                continue

            note = self._transport.take_notification(timeout=_NOTIFICATION_POLL_TIMEOUT)
            if note is not None:
                self._apply_notification(
                    note, text_parts, reasoning_parts,
                    capture_tool_activity=self._mcp_tools_enabled,
                    on_delta=on_delta,
                )

            try:
                result = self._transport.poll_response(rid, q, timeout=0)
            except ClaudeACPError as exc:
                status_code = _status_code_from_acp_error(exc)
                outcome.status_code = status_code
                if status_code in (401, 403):
                    outcome.auth_failed = True
                    outcome.should_retire = True
                elif status_code == 429:
                    # Usage-limit/rate-limit errors are exactly the signal
                    # the credential pool's sub-rollover exists for (SPEC
                    # §2.2 risk #2 — the 5h subscription cap surfaces as a
                    # 429). Treat like an auth failure for rotation
                    # purposes: drop this session, rotate to the next
                    # credential, and let the caller replay history on the
                    # respawned session — a 429 mid-turn must never be
                    # treated as "nothing happened" (see should_retire /
                    # sent_history_len handling below).
                    outcome.auth_failed = True
                    outcome.should_retire = True
                else:
                    # Any other JSON-RPC error (malformed request, internal
                    # agent error, etc.) is unrecoverable for THIS turn.
                    # Retire the session unconditionally so a broken
                    # process can't be silently reused — falling through
                    # with should_retire=False would let
                    # _run_turn_with_recovery's caller treat this as a
                    # successful empty turn.
                    outcome.should_retire = True
                outcome.error = self._format_stderr_error(f"session/prompt failed: {exc}")
                break
            if result is not None:
                outcome.stop_reason = result.get("stopReason")
                if outcome.stop_reason == "cancelled":
                    outcome.interrupted = True
                break
        else:
            # Loop fell through without break -> deadline hit.
            self._issue_cancel()
            outcome.interrupted = True
            outcome.error = self._format_stderr_error(f"claude-acp turn timed out after {turn_timeout}s")
            outcome.should_retire = True

        # Drain anything left in the notification queue (final chunks that
        # arrived alongside the resolving response) before returning.
        for _ in range(32):
            note = self._transport.take_notification(timeout=0)
            if note is None:
                break
            self._apply_notification(
                note, text_parts, reasoning_parts,
                capture_tool_activity=self._mcp_tools_enabled,
                on_delta=on_delta,
            )

        outcome.text = "".join(text_parts)
        outcome.reasoning = "".join(reasoning_parts)
        if outcome.error is None:
            # Only a genuinely error-free turn (successful completion OR an
            # interrupt on a still-alive session) means the Claude-side
            # conversation now reflects everything in `messages`. Any
            # error path — even one that doesn't retire the session —
            # must NOT advance this counter, or the message that triggered
            # the error is silently excluded from every future delta
            # prompt (never sent to Claude, never retried).
            self.sent_history_len = len(messages)
        return outcome

    # ---------- internals ----------

    def _issue_cancel(self) -> None:
        if self._transport is None or self._session_id is None:
            return
        try:
            params = acp_schema.CancelNotification(sessionId=self._session_id).model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
            self._transport.notify(AGENT_METHODS["session_cancel"], params)
        except Exception:  # pragma: no cover - best-effort
            logger.debug("claude-acp session/cancel notify failed", exc_info=True)

    def _apply_notification(
        self,
        note: dict,
        text_parts: list[str],
        reasoning_parts: list[str],
        *,
        capture_tool_activity: bool = False,
        on_delta: Optional[Any] = None,
    ) -> None:
        method = note.get("method")
        if method != CLIENT_METHODS["session_update"]:
            return
        params = note.get("params") or {}
        update = params.get("update") or {}
        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            chunk_text = _content_block_text(update.get("content"))
            if chunk_text:
                text_parts.append(chunk_text)
                if on_delta is not None:
                    on_delta("text", chunk_text)
        elif kind == "agent_thought_chunk":
            chunk_text = _content_block_text(update.get("content"))
            if chunk_text:
                reasoning_parts.append(chunk_text)
                if on_delta is not None:
                    on_delta("reasoning", chunk_text)
        elif kind == "config_option_update":
            opt = update.get("configOption") or update.get("option")
            if isinstance(opt, dict) and opt.get("id"):
                self._config_options[opt["id"]] = opt
        elif capture_tool_activity and kind in ("tool_call", "tool_call_update"):
            # MCP mode (SPEC §2.4): with tool execution happening natively
            # via MCP inside this ACP turn, Hermes' history for the turn is
            # just user->assistant text (no tool_calls round trip) — but
            # users still deserve visibility into what Claude did. Forward
            # `tool_call`/`tool_call_update` session/update notifications
            # through the same reasoning/thought channel the text bridge
            # already uses for interim updates, rather than silently
            # dropping them.
            title = update.get("title") or update.get("toolCallId") or "tool"
            status = update.get("status")
            line = f"[tool: {title}]" if not status else f"[tool: {title} ({status})]"
            reasoning_parts.append(line + "\n")

    def _handle_server_request(self, req: dict) -> None:
        """Respond to agent-initiated requests. Phase 2 keeps the text-based
        tool bridge and advertises fs/terminal capabilities as False in
        `initialize`, so a well-behaved claude-agent-acp should not send
        fs/terminal requests. Fail closed (decline / method-not-supported)
        on anything unexpected rather than hang the agent process.

        Phase 3 / SPEC §2.4 (round-1 review finding #2): claude-agent-acp
        routes EVERY tool call — including ones served by our own
        Hermes-bridged MCP server — through `session/request_permission`
        before executing it. Blanket-cancelling (Phase 2's behavior,
        correct for denying Claude's own built-in tools) also cancelled
        every Hermes-bridged MCP tool call, live-probe-verified ("aborted
        due to permissions", dispatch never reached). Distinguish by the
        request's `toolCall.title`: Claude's SDK names MCP-sourced tools
        `mcp__<server-name>__<tool-name>` (also visible in `tool_call`
        session/update titles, e.g. `mcp__hermes-tools__skills_list`) — a
        title with our server's prefix is unambiguously a call INTO
        Hermes' own tool registry (already gated by whatever
        `handle_function_call` does internally — approval, redaction,
        etc.), so approve it; anything else (Claude's own built-ins) stays
        denied, unchanged from Phase 2."""
        if self._transport is None:
            return
        method = req.get("method", "")
        rid = req.get("id")
        if method == CLIENT_METHODS["session_request_permission"]:
            if self._mcp_tools_enabled and self._is_hermes_mcp_tool_call(req):
                option_id = self._pick_allow_option_id(req)
                if option_id is not None:
                    self._transport.respond(
                        rid,
                        {"outcome": {"outcome": "selected", "optionId": option_id}},
                    )
                    return
                logger.warning(
                    "claude-acp: MCP tool permission request had no allow-shaped "
                    "option; falling back to cancel: %s",
                    req.get("params"),
                )
            # We never advertised native tool execution (Claude's own
            # built-ins); decline so the agent falls back to whatever it
            # does without permission (typically surfacing the denial in
            # its own output).
            self._transport.respond(
                rid,
                {"outcome": {"outcome": "cancelled"}},
            )
        else:
            self._transport.respond_error(
                rid, code=-32601, message=f"Unsupported method: {method}"
            )

    @staticmethod
    def _is_hermes_mcp_tool_call(req: dict) -> bool:
        """Trust-boundary note: the `mcp__hermes-tools__` title prefix is
        only unambiguous because Hermes controls the session's entire MCP
        server namespace — the wrapper's hermetic CLAUDE_HOME means no
        user-level settings can declare a rogue server also named
        "hermes-tools" whose tools would inherit this prefix and get
        auto-approved. If a future change lets profiles add their own MCP
        servers via CLAUDE_HOME settings, this check must be revisited."""
        params = req.get("params") or {}
        tool_call = params.get("toolCall") or params.get("tool_call") or {}
        title = str(tool_call.get("title") or "")
        return title.startswith(_MCP_TOOL_TITLE_PREFIX)

    @staticmethod
    def _pick_allow_option_id(req: dict) -> Optional[str]:
        """Pick an `allow_once` (preferred) or any `allow_*`-kinded option
        from the request's offered `options` list — never hardcode an
        option_id, since it's the AGENT's ID to assign, not ours."""
        params = req.get("params") or {}
        options = params.get("options") or []
        allow_once = None
        allow_any = None
        for opt in options:
            if not isinstance(opt, dict):
                continue
            kind = str(opt.get("kind") or "")
            opt_id = opt.get("optionId") or opt.get("option_id")
            if not opt_id:
                continue
            if kind == "allow_once":
                allow_once = opt_id
            elif kind.startswith("allow") and allow_any is None:
                allow_any = opt_id
        return allow_once or allow_any

    def _format_stderr_error(self, prefix: str) -> str:
        if self._transport is None:
            return prefix
        tail = self._transport.stderr_tail(20)
        if not tail:
            return prefix
        joined = "\n".join(line.rstrip() for line in tail if line)
        if not joined.strip():
            return prefix
        from agent.redact import redact_sensitive_text

        return f"{prefix}\nclaude-agent-acp stderr (last {len(tail)} lines):\n{redact_sensitive_text(joined, force=True)}"


def _content_block_text(content: Any) -> str:
    """Best-effort text extraction from an ACP ContentBlock (or list of
    them) as found on agent_message_chunk / agent_thought_chunk updates."""
    if content is None:
        return ""
    if isinstance(content, list):
        return "".join(_content_block_text(c) for c in content)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text") or "")
        return ""
    return ""


def _status_code_from_acp_error(exc: ClaudeACPError) -> Optional[int]:
    """Best-effort mapping of a JSON-RPC error onto an HTTP-style status
    code so the caller can drive credential_pool.mark_exhausted_and_rotate
    the same way the rest of Hermes classifies provider auth/limit errors.
    claude-agent-acp surfaces upstream Anthropic API errors either in the
    message text or in `data`."""
    data = exc.data if isinstance(exc.data, dict) else {}
    for key in ("status", "status_code", "statusCode", "http_status"):
        val = data.get(key)
        if isinstance(val, int):
            return val
    haystack = f"{exc.message} {data}".lower()
    if "401" in haystack or "unauthorized" in haystack or "invalid_grant" in haystack:
        return 401
    if "429" in haystack or "rate limit" in haystack or "usage limit" in haystack or "overloaded" in haystack:
        return 429
    if "403" in haystack or "forbidden" in haystack:
        return 403
    return None


# --------------------------------------------------------------------------
# Credential resolution: pool -> CLAUDE_CODE_OAUTH_TOKEN -> CLAUDE_ACP_TOKEN_FILE
# --------------------------------------------------------------------------


def _scoped_get_secret(name: str, default: str = "") -> str:
    """Read a secret through `agent.secret_scope` (fail-closed under an
    active multiplex) with an unscoped os.getenv fallback for contexts
    where secret_scope isn't installed at all — matches the
    `_get_scoped_env` pattern hermes_cli.auth uses (Phase 1 round 2 fix)."""
    try:
        from agent import secret_scope
    except ImportError:
        return os.getenv(name, default)
    return secret_scope.get_secret(name, default)


def resolve_claude_acp_credential(*, api_key_hint: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """Resolve a claude-acp bearer token.

    Returns (token, pool_credential_id). `pool_credential_id` is None when
    the token came from env/file fallback rather than the credential pool,
    so the caller knows whether `mark_exhausted_and_rotate` has anything to
    rotate. Order: credential pool -> CLAUDE_CODE_OAUTH_TOKEN (secret_scope)
    -> CLAUDE_ACP_TOKEN_FILE (secret_scope, default under get_hermes_home()).
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("claude-acp")
        if pool.has_credentials():
            entry = None
            if api_key_hint:
                entry = next(
                    (e for e in pool.entries() if e.runtime_api_key == api_key_hint), None
                )
            if entry is None:
                entry = pool.select()
            if entry is not None and entry.runtime_api_key:
                return entry.runtime_api_key, entry.id
    except Exception:  # pragma: no cover - pool is best-effort, fall through
        logger.debug("claude-acp credential pool lookup failed", exc_info=True)

    token = (_scoped_get_secret("CLAUDE_CODE_OAUTH_TOKEN", "") or "").strip()
    if token:
        return token, None

    tok_file = (_scoped_get_secret("CLAUDE_ACP_TOKEN_FILE", "") or "").strip()
    if not tok_file:
        try:
            from hermes_cli.hermes_home import get_hermes_home

            tok_file = os.path.join(get_hermes_home(), "claude_acp_token")
        except Exception:
            tok_file = os.path.expanduser("~/.hermes/claude_acp_token")
    try:
        token = Path(tok_file).read_text(encoding="utf-8").strip()
        if token:
            return token, None
    except OSError:
        pass
    return None, None


def mark_claude_acp_credential_exhausted(
    *, status_code: Optional[int], api_key_hint: Optional[str]
) -> bool:
    """Rotate the credential pool after an auth/limit error surfaced through
    an ACP turn. Returns True if there is another credential to try."""
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("claude-acp")
        if not pool.has_credentials():
            return False
        next_entry = pool.mark_exhausted_and_rotate(status_code=status_code, api_key_hint=api_key_hint)
        return next_entry is not None
    except Exception:  # pragma: no cover - best-effort
        logger.debug("claude-acp credential rotation failed", exc_info=True)
        return False


# --------------------------------------------------------------------------
# OpenAI-shim client
# --------------------------------------------------------------------------


def _completion_to_stream_chunks(completion: SimpleNamespace) -> list[SimpleNamespace]:
    """Convert a one-shot response into OpenAI-style stream chunks — the
    POST-HOC path, used for text-bridge-with-tools turns where the regex
    extraction needs the complete text before anything reaches the
    consumer. Live turns (MCP mode / no tools) stream real deltas via
    `_stream_turn` instead; conversation_loop's streaming exclusion no
    longer applies to claude-acp."""
    choice = completion.choices[0]
    message = choice.message
    tool_call_deltas = None
    if message.tool_calls:
        tool_call_deltas = []
        for index, tool_call in enumerate(message.tool_calls):
            tool_call_deltas.append(
                SimpleNamespace(
                    index=index,
                    id=getattr(tool_call, "id", None),
                    type=getattr(tool_call, "type", "function"),
                    function=SimpleNamespace(
                        name=getattr(tool_call.function, "name", None),
                        arguments=getattr(tool_call.function, "arguments", None),
                    ),
                )
            )
    delta = SimpleNamespace(
        role="assistant",
        content=message.content or None,
        tool_calls=tool_call_deltas,
        reasoning_content=message.reasoning_content,
        reasoning=message.reasoning,
    )
    data_chunk = SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=delta, finish_reason=choice.finish_reason)],
        model=completion.model,
        usage=None,
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        model=completion.model,
        usage=completion.usage,
    )
    return [data_chunk, usage_chunk]


class _ClaudeACPChatCompletions:
    def __init__(self, client: "ClaudeACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ClaudeACPChatNamespace:
    def __init__(self, client: "ClaudeACPClient"):
        self.completions = _ClaudeACPChatCompletions(client)


class ClaudeACPClient:
    """OpenAI-client-compatible facade over a persistent `ClaudeACPSession`.

    Contract (SPEC §2.1): constructor swallows unknown kwargs;
    `chat.completions.create(**kwargs)` returns
    `choices[0].message.{content,tool_calls}`, `finish_reason`, `usage.*`,
    `model`; `close()` is idempotent and does NOT kill the persistent
    session process — session teardown is the agent lifecycle's job
    (`evict_session` / `agent._claude_acp_session = None`), not this
    per-request wrapper's.

    Session caching: when constructed with `agent=<AIAgent>`, the session is
    cached on `agent._claude_acp_session` (mirrors `agent._codex_session`,
    codex_runtime.py:305-329) so every client instance created for the same
    agent (per-request client recreation in `agent_runtime_helpers`) reuses
    the same underlying process. Without an `agent` (auxiliary/one-off call
    sites — `auxiliary_client.resolve_provider_client`), the client owns a
    private session for its own instance lifetime; multiple `.create()`
    calls on the SAME client instance still reuse one process, but a new
    `ClaudeACPClient()` starts fresh (no agent to hang persistence off of).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        acp_command: str | None = None,
        acp_args: list[str] | None = None,
        acp_cwd: str | None = None,
        agent: Any = None,
        _transport_factory: Any = None,
        _mcp_server_factory: Any = None,
        **_: Any,
    ) -> None:
        self._test_transport_factory = _transport_factory
        self._test_mcp_server_factory = _mcp_server_factory
        self.base_url = base_url or CLAUDE_ACP_MARKER_BASE_URL
        self.api_key = api_key or "claude-acp"
        self._default_headers = dict(default_headers or {})
        explicit_command = acp_command or command
        explicit_args = acp_args or args
        if explicit_command and explicit_args:
            self._acp_command = explicit_command
            self._acp_args = list(explicit_args)
        else:
            registry_command, registry_args = _registry_command_for_provider("claude-acp")
            self._acp_command = explicit_command or registry_command
            self._acp_args = list(explicit_args or registry_args)
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ClaudeACPChatNamespace(self)
        self.is_closed = False
        self._agent = agent
        # Fallback private session for agent-less construction (auxiliary
        # calls). Only used when self._agent is None.
        self._private_session: Optional[ClaudeACPSession] = None
        self._current_credential_id: Optional[str] = None
        self._current_api_key_hint: Optional[str] = None
        # Phase 3 flag snapshot (SPEC §2.4): read once at construction, not
        # per-turn, so a single `chat.completions.create` call always gets
        # a self-consistent view (mode doesn't flip mid-turn); a NEW client
        # instance (created for the next request, per
        # `agent_runtime_helpers.create_openai_client`) re-reads it, so a
        # config change takes effect on the next request without restarting
        # the process. `_build_session` threads this into a fresh session;
        # since sessions are cached on `agent._claude_acp_session`, flipping
        # the flag on a live agent takes effect on the next respawn, not
        # instantly on a cached session — same lifecycle as every other
        # spawn-time setting here (native_tools_env, command/args).
        self._mcp_tools_enabled = _mcp_tools_enabled()

    def close(self) -> None:
        """Idempotent. Does NOT kill the persistent session — that belongs
        to the agent lifecycle. Only tears down a private (agent-less)
        session, since nothing else owns it."""
        self.is_closed = True
        if self._agent is not None:
            return
        if self._private_session is not None:
            try:
                self._private_session.close()
            except Exception:  # pragma: no cover - best-effort
                pass
            self._private_session = None

    # ---------- session access ----------

    def _get_or_create_session(self) -> ClaudeACPSession:
        if self._agent is not None:
            existing = getattr(self._agent, "_claude_acp_session", None)
            if existing is not None:
                return existing
            session = self._build_session()
            self._agent._claude_acp_session = session
            return session
        if self._private_session is None:
            self._private_session = self._build_session()
        return self._private_session

    def _drop_session(self) -> None:
        """Drop the cached session after an unrecoverable failure so the
        NEXT call respawns from scratch — mirrors codex_runtime.py's
        `agent._codex_session = None` on turn failure."""
        if self._agent is not None:
            existing = getattr(self._agent, "_claude_acp_session", None)
            if existing is not None:
                try:
                    existing.close()
                except Exception:
                    pass
                self._agent._claude_acp_session = None
        elif self._private_session is not None:
            try:
                self._private_session.close()
            except Exception:
                pass
            self._private_session = None

    def _build_session(self) -> ClaudeACPSession:
        native_tools_env = {}
        if self._agent is not None:
            native_tools_mode = getattr(self._agent, "claude_acp_native_tools", None)
            if native_tools_mode is not None:
                native_tools_env["CLAUDE_ACP_NATIVE_TOOLS"] = str(native_tools_mode)

        def _env_provider() -> dict[str, str]:
            token, cred_id = resolve_claude_acp_credential()
            self._current_credential_id = cred_id
            self._current_api_key_hint = token
            env: dict[str, str] = {}
            if token:
                env["CLAUDE_CODE_OAUTH_TOKEN"] = token
            return env

        return ClaudeACPSession(
            command=self._acp_command,
            args=self._acp_args,
            cwd=self._acp_cwd,
            env_provider=_env_provider,
            native_tools_env=native_tools_env,
            transport_factory=self._test_transport_factory,
            mcp_tools_enabled=self._mcp_tools_enabled,
            mcp_server_factory=self._test_mcp_server_factory,
        )

    # ---------- chat.completions.create ----------

    def _create_chat_completion(
        self,
        *,
        model: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        **_: Any,
    ) -> Any:
        messages = messages or []
        effective_timeout = _normalize_timeout(timeout)

        if self._mcp_tools_enabled:
            _stream_safe = not _split_loop_tools(tools)[1]
        else:
            _stream_safe = not tools
        if stream and _stream_safe:
            # Real incremental streaming. Safe here because no `<tool_call>`
            # text protocol is in play: in MCP mode tool work happens
            # natively inside the ACP turn (content is pure prose), and
            # with no tools there is nothing to extract. The text-bridge-
            # with-tools case falls through to the blocking turn below and
            # returns post-hoc chunks (`_completion_to_stream_chunks`) —
            # regex extraction needs the complete text, so partial deltas
            # could leak `<tool_call>` JSON to the consumer.
            return self._stream_turn(
                messages,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                timeout_seconds=effective_timeout,
                effort=reasoning_effort,
            )

        response_text, reasoning_text = self._run_turn_with_recovery(
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            timeout_seconds=effective_timeout,
            effort=reasoning_effort,
        )

        # MCP mode (SPEC §2.4 + hybrid amendment): dispatcher-tool work
        # already happened natively inside the ACP turn, but AGENT-LOOP
        # tools (memory/todo/session_search/delegate_task) still ride the
        # <tool_call> text protocol because only run_agent's loop can
        # execute them. Extraction therefore runs iff the text protocol was
        # actually OFFERED this turn (text-bridge mode, or MCP mode with
        # loop tools in the catalog). A pure-dispatcher MCP turn never
        # extracts — <tool_call>-shaped text in the reply there is a quoted
        # red herring, not a call (and an empty tool_calls list must never
        # reach run_agent, which would spin on calls that don't exist).
        _loop_tool_names = {
            str((t.get("function") or {}).get("name") or "")
            for t in _split_loop_tools(tools)[1]
            if isinstance(t, dict)
        } if self._mcp_tools_enabled else None
        if self._mcp_tools_enabled and not _loop_tool_names:
            tool_calls, cleaned_text = None, response_text
        else:
            tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)
            if tool_calls and _loop_tool_names is not None:
                # Hybrid-mode guard: the text protocol is scoped to the
                # loop-tool catalog. A <tool_call> naming a DISPATCHER tool
                # (Claude satisfying both channels, or quoted red-herring
                # text) must not reach run_agent — the native MCP round trip
                # is that tool's only sanctioned path, and executing the
                # text copy would run the same operation twice.
                kept = [
                    tc for tc in tool_calls
                    if getattr(tc.function, "name", None) in _loop_tool_names
                ]
                dropped = len(tool_calls) - len(kept)
                if dropped:
                    logger.warning(
                        "claude-acp hybrid: dropped %d text tool_call(s) "
                        "naming non-loop tools (native MCP is their only "
                        "path)", dropped,
                    )
                tool_calls = kept or None

        usage = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text,
            tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        # Hybrid: extracted loop-tool calls need finish_reason="tool_calls"
        # in MCP mode too, or run_agent treats the turn as final and never
        # executes them.
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        completion = SimpleNamespace(choices=[choice], usage=usage, model=model or "claude-acp")
        if stream:
            return _completion_to_stream_chunks(completion)
        return completion

    def _peek_session(self) -> Optional[ClaudeACPSession]:
        """The currently-cached session, WITHOUT creating one."""
        if self._agent is not None:
            return getattr(self._agent, "_claude_acp_session", None)
        return self._private_session

    def _stream_turn(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        timeout_seconds: float,
        effort: str | None,
    ):
        """Generator of OpenAI-style stream chunks, yielded LIVE as ACP
        `agent_message_chunk`/`agent_thought_chunk` updates arrive (not
        post-hoc like `_completion_to_stream_chunks`). The turn runs in a
        worker thread; deltas cross a queue. Chunk shapes match the
        consumer contract in chat_completion_helpers (delta.content /
        delta.reasoning_content per chunk, finish_reason on the last data
        chunk, usage in a trailing empty-choices chunk, .model on every
        chunk).

        Early consumer exit (interrupt, break) closes the generator; the
        finally block cancels the in-flight ACP turn so the agent process
        isn't left generating into the void."""
        out_q: "queue.Queue[tuple[str, Any]]" = queue.Queue()

        def on_delta(kind: str, text: str) -> None:
            out_q.put((kind, text))

        def _run() -> None:
            try:
                text, reasoning = self._run_turn_with_recovery(
                    messages,
                    model=model,
                    tools=tools,
                    tool_choice=tool_choice,
                    timeout_seconds=timeout_seconds,
                    effort=effort,
                    on_delta=on_delta,
                )
                out_q.put(("done", (text, reasoning)))
            except BaseException as exc:  # noqa: BLE001 - crosses a thread
                out_q.put(("error", exc))

        worker = threading.Thread(
            target=_run, name="claude-acp-stream-turn", daemon=True
        )
        worker.start()
        model_name = model or "claude-acp"
        finished = False
        try:
            while True:
                kind, payload = out_q.get()
                if kind == "text":
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(
                            index=0,
                            delta=SimpleNamespace(
                                role="assistant", content=payload,
                                tool_calls=None, reasoning_content=None,
                                reasoning=None,
                            ),
                            finish_reason=None,
                        )],
                        model=model_name,
                        usage=None,
                    )
                elif kind == "reasoning":
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(
                            index=0,
                            delta=SimpleNamespace(
                                role="assistant", content=None,
                                tool_calls=None, reasoning_content=payload,
                                reasoning=None,
                            ),
                            finish_reason=None,
                        )],
                        model=model_name,
                        usage=None,
                    )
                elif kind == "done":
                    finished = True
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(
                            index=0,
                            delta=SimpleNamespace(
                                role="assistant", content=None,
                                tool_calls=None, reasoning_content=None,
                                reasoning=None,
                            ),
                            finish_reason="stop",
                        )],
                        model=model_name,
                        usage=None,
                    )
                    yield SimpleNamespace(
                        choices=[],
                        model=model_name,
                        usage=SimpleNamespace(
                            prompt_tokens=0,
                            completion_tokens=0,
                            total_tokens=0,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
                        ),
                    )
                    return
                elif kind == "error":
                    finished = True
                    raise payload
        finally:
            if not finished:
                # Consumer bailed mid-stream (interrupt/break/GC). Cancel
                # the in-flight ACP turn; without this the agent process
                # keeps generating a response nobody will read.
                session = self._peek_session()
                if session is not None:
                    try:
                        session.request_interrupt()
                    except Exception:  # pragma: no cover - best-effort
                        pass
            worker.join(timeout=10.0)
            if worker.is_alive():
                # The turn didn't unwind in time (cancel lost, respawn race
                # where _peek_session() was None, agent ignoring cancel).
                # The abandoned daemon worker keeps polling the transport
                # for up to turn_timeout — sharing it with a future turn
                # means two loops stealing each other's chunks. Drop the
                # session so the NEXT turn provably starts on a fresh
                # process; the orphan worker's session object is closed out
                # from under it and its poll loop exits on is_alive().
                logger.warning(
                    "claude-acp: streaming worker still alive after cancel; "
                    "dropping session so the next turn starts fresh"
                )
                try:
                    self._drop_session()
                except Exception:  # pragma: no cover - best-effort
                    pass

    def _run_turn_with_recovery(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: Any,
        timeout_seconds: float,
        effort: str | None,
        _respawn_budget: int = 2,
        on_delta: Optional[Any] = None,
    ) -> tuple[str, str]:
        """Run one turn, transparently respawning (with full history replay)
        on: process death (detected before OR during the turn), model-change
        rejection, or an auth/limit error that rotates the credential pool
        to a new token. `_respawn_budget` bounds retries so a persistently
        broken session can't loop forever; when the budget is exhausted a
        clean RuntimeError is raised.

        Streaming discipline: when `on_delta` is set, transparent
        respawn+replay is only safe while NOTHING has been emitted to the
        consumer — a replayed turn regenerates the whole response, and a
        consumer that already received partial text would see duplicated
        (or diverged) output spliced onto it. Once the first delta is out,
        any failure raises instead of retrying; the conversation loop's
        outer retry then re-requests a clean stream (the session was
        dropped, so any follow-up turn replays history from zero). In
        practice the streaming consumer (chat_completion_helpers) converts
        a post-emission failure into a partial-response stub with
        finish_reason="length" and continues from the delivered text —
        no duplication either way. Pre-emission failures keep the full
        transparent-recovery semantics of the non-streaming path."""

        emitted = {"any": False}
        if on_delta is not None:
            _consumer_on_delta = on_delta

            def _tracked_on_delta(kind: str, text: str) -> None:
                emitted["any"] = True
                _consumer_on_delta(kind, text)

            turn_on_delta: Optional[Any] = _tracked_on_delta
        else:
            turn_on_delta = None

        def _retry() -> tuple[str, str]:
            return self._run_turn_with_recovery(
                messages,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                timeout_seconds=timeout_seconds,
                effort=effort,
                _respawn_budget=_respawn_budget - 1,
                on_delta=on_delta,
            )

        session = self._get_or_create_session()

        # Belt: a cached session whose process died BETWEEN turns (crash,
        # kill -9, OOM-kill) must be detected before we try to write to it,
        # or the first send surfaces as a broken-pipe RuntimeError from
        # _ACPTransport._send that nothing classifies. Cheap poll() check;
        # the braces for the poll->send TOCTOU window are the except blocks
        # below.
        if session.is_started() and not session.is_process_alive():
            logger.info(
                "claude-acp: cached session process died between turns; respawning"
            )
            self._drop_session()
            session = self._get_or_create_session()

        try:
            session.ensure_started(tools=tools)
        except (ClaudeACPError, TimeoutError, RuntimeError, OSError) as exc:
            self._drop_session()
            raise RuntimeError(str(exc)) from exc

        if model and session.is_started():
            # Braces: any transport-level failure on a PRE-RESPONSE step of
            # a previously-cached session (here: the config-apply write) is
            # the same class of failure as a mid-prompt process death and
            # must route through the same drop -> respawn -> replay path —
            # the process can die in the poll->send TOCTOU window above, in
            # which case set_config_option raises a broken-pipe
            # RuntimeError (from _ACPTransport._send) or a ClaudeACPError/
            # TimeoutError. This exact hole crashed the live exit gate's
            # kill -9 scenario (Phase 2 round 3).
            try:
                self._apply_model_if_needed(session, model, tools=tools)
            except (RuntimeError, ClaudeACPError, TimeoutError, OSError) as exc:
                self._drop_session()
                if _respawn_budget > 0 and not emitted["any"]:
                    logger.info(
                        "claude-acp: transport failure applying model config "
                        "(%s); respawning and replaying history", exc,
                    )
                    return _retry()
                raise RuntimeError(
                    f"claude-acp session failed while applying model config "
                    f"and the respawn budget is exhausted: {exc}"
                ) from exc
            # _apply_model_if_needed may have respawned the session
            # internally (model-change rejection path drops and rebuilds) —
            # re-fetch so we never run the turn on a stale, closed session
            # object.
            session = self._get_or_create_session()
            if not session.is_started():
                try:
                    session.ensure_started(tools=tools)
                except (ClaudeACPError, TimeoutError, RuntimeError, OSError) as exc:
                    self._drop_session()
                    raise RuntimeError(str(exc)) from exc

        outcome = session.send_turn(
            messages, model=model, tools=tools, tool_choice=tool_choice,
            turn_timeout=timeout_seconds, on_delta=turn_on_delta,
        )

        if outcome.interrupted and not outcome.should_retire:
            raise InterruptedError("claude-acp turn interrupted")

        if outcome.should_retire:
            self._drop_session()
            if emitted["any"]:
                # Streaming turn already emitted deltas — no transparent
                # replay (see docstring). Surface the failure; the session
                # is dropped, so the outer retry streams clean from zero.
                if outcome.auth_failed:
                    # Still rotate the pool so the NEXT attempt lands on a
                    # healthy credential.
                    mark_claude_acp_credential_exhausted(
                        status_code=outcome.status_code,
                        api_key_hint=self._current_api_key_hint,
                    )
                raise RuntimeError(
                    outcome.error
                    or "claude-acp streaming turn failed after partial output"
                )
            if outcome.auth_failed and _respawn_budget > 0:
                rotated = mark_claude_acp_credential_exhausted(
                    status_code=outcome.status_code,
                    api_key_hint=self._current_api_key_hint,
                )
                if rotated:
                    return _retry()
            elif (
                outcome.transport_failed
                and not outcome.interrupted
                and _respawn_budget > 0
            ):
                # Transport-level retirement: process death mid-turn, stdin
                # closed on the prompt send, session torn down under us.
                # SPEC's crash-recovery path — respawn on the same
                # credential and replay the full history, transparently.
                # Deliberately narrower than "any retirement":
                # unclassified JSON-RPC errors answered by a LIVE process
                # would likely recur on replay and must raise instead
                # (conversation_loop owns backoff/recovery for those —
                # round-2 verified semantics). Interrupted turns (user
                # cancel, turn timeout) are also excluded: auto-replaying a
                # turn the user just cancelled (or one that already burned
                # the full turn timeout) is worse than surfacing the error.
                logger.info(
                    "claude-acp: transport failure mid-turn (%s); respawning "
                    "and replaying history",
                    outcome.error or "no error detail",
                )
                return _retry()
            if outcome.error:
                raise RuntimeError(outcome.error)
            raise RuntimeError("claude-acp session ended without a response and must respawn")

        # Defensive: every error path in ClaudeACPSession.send_turn sets
        # should_retire, so this should be unreachable — but if a future
        # error path forgets to, fail loudly instead of silently returning
        # an empty "successful" response and dropping the user's message
        # from history (the Round-1 review bug this guards against).
        if outcome.error:
            raise RuntimeError(outcome.error)

        return outcome.text, outcome.reasoning

    def _apply_model_if_needed(
        self,
        session: ClaudeACPSession,
        model: str,
        *,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Set the ACP `model` configOption for a per-turn model change. On
        rejection (agent doesn't support live model switching for this
        model, or the option isn't advertised), respawn the session with
        the new model as the spawn-time default and replay history — SPEC
        risk #1's documented fallback."""
        if not session.supports_config_option("model"):
            return
        try:
            session.set_config_option("model", model)
        except ClaudeACPError:
            logger.info(
                "claude-acp: live model switch to %s rejected; respawning session",
                model,
            )
            self._drop_session()
            new_session = self._get_or_create_session()
            new_session.ensure_started(tools=tools)
            try:
                new_session.set_config_option("model", model)
            except ClaudeACPError:
                # Agent doesn't support this model at all via config option —
                # fall through and let the turn itself surface the error.
                logger.warning("claude-acp: model %s rejected even after respawn", model)


def evict_session(agent: Any) -> None:
    """Explicit shutdown hook: close and drop `agent._claude_acp_session` if
    present. Hermes' `AIAgent` objects don't currently register a generic
    per-provider teardown hook when a cached agent is evicted from the
    gateway's session cache — codex sessions (`agent._codex_session`) have
    the same gap today, so this matches existing behavior rather than
    inventing new lifecycle plumbing. Exposed here so a future gateway
    eviction path (or a test, or an explicit `/quit`-style command) has a
    single, correct place to call rather than reaching into the ACP session
    internals directly."""
    session = getattr(agent, "_claude_acp_session", None)
    if session is None:
        return
    try:
        session.close()
    except Exception:  # pragma: no cover - best-effort
        logger.debug("claude-acp evict_session close failed", exc_info=True)
    agent._claude_acp_session = None


def _normalize_timeout(timeout: Any) -> float:
    if timeout is None:
        return _DEFAULT_TURN_TIMEOUT_SECONDS
    if isinstance(timeout, (int, float)):
        return float(timeout)
    candidates = [getattr(timeout, attr, None) for attr in ("read", "write", "connect", "pool", "timeout")]
    numeric = [float(v) for v in candidates if isinstance(v, (int, float))]
    return max(numeric) if numeric else _DEFAULT_TURN_TIMEOUT_SECONDS
