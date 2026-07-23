"""In-process MCP server exposing the CURRENT turn's Hermes tool registry
natively to `claude-agent-acp` via ACP `session/new`'s `mcpServers` param.

Phase 3 of the claude-acp provider (see PLAN.md / SPEC.md §2.4 in
hermes-claude-acp), gated behind `HERMES_CLAUDE_ACP_MCP_TOOLS`. When the
flag is off, `agent/claude_acp_client.py` never imports or starts this
module — the existing `<tool_call>` text bridge (`copilot_acp_client.
_format_messages_as_prompt` / `_extract_tool_calls_from_text`) stays byte-
for-byte unchanged.

Why HTTP, not stdio
--------------------
`claude-agent-acp`'s `createSession()` accepts both stdio and http/sse
`mcpServers` entries (verified directly against the installed package at
node_modules/@agentclientprotocol/claude-agent-acp/dist/acp-agent.js:2725-
2747 — despite `initialize`'s `mcpCapabilities` only advertising
`{http, sse}`, the stdio branch is unconditional, just undeclared). We
still chose HTTP: the architecture requirement is that the tool registry
can change *turn to turn* while the ACP session (and therefore the
`mcpServers` list passed at `session/new`) is only declared **once** at
session establishment. A stdio server is a fixed subprocess argv decided
at that one moment; an HTTP server's *address* is fixed at that moment but
its *behavior* (which tools `tools/list` returns, and what `tools/call`
dispatches to) can be swapped out from under the same URL between turns
with zero ACP-level churn. So: one `HermesACPMcpServer` per
`ClaudeACPSession`, started once in `ensure_started()` before `session/
new`, its registry swapped via `set_tools()` at the top of every
`send_turn()`.

Canonical execution seam
------------------------
`tools/call` dispatches through `model_tools.handle_function_call()` — the
exact function the OpenAI-shim tool-calling loop
(`conversation_loop.py` / `chat_completion_helpers.py`) calls for every
other provider's tool_calls, and the same function
`agent/transports/hermes_tools_mcp_server.py` (the Codex analog) already
reuses for its curated static tool set. This module does NOT reimplement
any tool; it is a thin MCP transport adapter in front of the same
dispatcher, so pre/post-call hooks, the Tool Search bridge, and any
dangerous-command / approval / redaction gating inside
`handle_function_call` apply identically here.

Auth / trust boundary
----------------------
Bound to 127.0.0.1 only (never 0.0.0.0). claude-agent-acp's `HttpMcpServer`
entry supports a `headers` list
(node_modules/.../acp-agent.js:2730-2739), so each server instance mints a
random per-session bearer token and requires
`Authorization: Bearer <token>` on every request; a mismatched or missing
header gets a 401 before the request reaches the MCP session manager. This
is defense in depth on top of the primary boundary, which is the
loopback-only bind: nothing outside this host can reach the port at all.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
import threading
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# (function_name, arguments) -> JSON-ish string result. Matches
# `model_tools.handle_function_call`'s `(function_name, function_args) ->
# str` signature (the extra optional kwargs on the real function all have
# defaults, so a 2-arg positional call is a valid, minimal dispatch).
DispatchFn = Callable[[str, dict[str, Any]], str]


def default_dispatch(function_name: str, function_args: dict[str, Any]) -> str:
    """Lazy-imports and calls the canonical Hermes tool executor so this
    module (and anything that merely imports it, e.g. for tests) never
    requires the full `model_tools` import graph unless a tool is actually
    invoked."""
    from model_tools import handle_function_call

    return handle_function_call(function_name, function_args or {})


class HermesACPMcpServer:
    """Per-session in-process MCP server over Streamable HTTP.

    Lifecycle: `start()` before ACP `session/new`, `stop()` when the owning
    `ClaudeACPSession` is closed/respawned. `set_tools()` swaps the served
    tool registry between turns — cheap (just replaces a dict under a
    lock), no server restart, no URL change.
    """

    def __init__(self, *, dispatch: DispatchFn = default_dispatch, host: str = "127.0.0.1") -> None:
        self._dispatch = dispatch
        self._host = host
        self._token = secrets.token_urlsafe(24)
        self._tools_lock = threading.Lock()
        self._tools: dict[str, dict[str, Any]] = {}

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._uvicorn_server: Any = None
        self._port: Optional[int] = None
        self._started_event = threading.Event()
        self._start_error: Optional[BaseException] = None

    # ---------- identity ----------

    @property
    def token(self) -> str:
        return self._token

    @property
    def port(self) -> Optional[int]:
        return self._port

    @property
    def url(self) -> str:
        if self._port is None:
            raise RuntimeError("HermesACPMcpServer not started")
        return f"http://{self._host}:{self._port}/mcp"

    # ---------- registry ----------

    def set_tools(self, openai_tools: Optional[list[dict[str, Any]]]) -> None:
        """Replace the served tool registry with the OpenAI-schema `tools`
        list Hermes passed for the current turn. Safe to call before
        `start()` (buffers) and any number of times between turns."""
        new_tools: dict[str, dict[str, Any]] = {}
        for entry in openai_tools or []:
            if not isinstance(entry, dict) or entry.get("type") != "function":
                continue
            fn = entry.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            new_tools[name] = {
                "description": fn.get("description") or f"Hermes tool {name}",
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        with self._tools_lock:
            self._tools = new_tools

    def tool_names(self) -> list[str]:
        with self._tools_lock:
            return sorted(self._tools.keys())

    def _snapshot_tools(self) -> dict[str, dict[str, Any]]:
        with self._tools_lock:
            return dict(self._tools)

    # ---------- lifecycle ----------

    def start(self, timeout: float = 10.0) -> None:
        if self._thread is not None:
            return  # already started (idempotent)

        # Bind our own socket up front so we know the real ephemeral port
        # before handing control to uvicorn — avoids a get-port-then-race
        # window and lets callers read `.url` right after `start()` returns.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, 0))
        self._port = sock.getsockname()[1]
        sock.listen(128)

        self._thread = threading.Thread(target=self._run, args=(sock,), daemon=True)
        self._thread.start()
        if not self._started_event.wait(timeout=timeout):
            raise TimeoutError(
                f"hermes-acp-mcp server on 127.0.0.1:{self._port} failed to start "
                f"within {timeout}s"
            )
        if self._start_error is not None:
            err, self._start_error = self._start_error, None
            raise err

    def stop(self, timeout: float = 3.0) -> None:
        server = self._uvicorn_server
        if server is not None:
            server.should_exit = True
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None
        self._uvicorn_server = None
        self._loop = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---------- internals ----------

    def _run(self, sock: socket.socket) -> None:
        try:
            import uvicorn

            app = self._build_asgi_app()
            config = uvicorn.Config(app, log_level="warning")
            server = uvicorn.Server(config)
            self._uvicorn_server = server

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop

            async def _serve() -> None:
                self._started_event.set()
                await server.serve(sockets=[sock])

            loop.run_until_complete(_serve())
        except BaseException as exc:  # pragma: no cover - defensive, surfaced via start()
            self._start_error = exc
            self._started_event.set()
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _build_asgi_app(self) -> Any:
        from mcp.server.lowlevel import Server as MCPServer
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        import mcp.types as types
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Mount
        from starlette.types import Receive, Scope, Send

        server: Any = MCPServer("hermes-acp-tools")

        @server.list_tools()
        async def _list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name=name,
                    description=spec.get("description") or "",
                    inputSchema=spec.get("parameters") or {"type": "object", "properties": {}},
                )
                for name, spec in self._snapshot_tools().items()
            ]

        async def _call_tool(name: str, arguments: dict[str, Any]) -> list[Any]:
            snapshot = self._snapshot_tools()
            if name not in snapshot:
                return [
                    types.TextContent(
                        type="text",
                        text=json.dumps({"error": f"unknown tool {name!r}"}),
                    )
                ]
            try:
                result = await asyncio.to_thread(self._dispatch, name, arguments or {})
            except Exception as exc:  # defensive: dispatch should already catch/format
                logger.exception("hermes-acp-mcp: tool %s raised", name)
                result = json.dumps({"error": str(exc), "tool": name})
            if not isinstance(result, str):
                try:
                    result = json.dumps(result, ensure_ascii=False)
                except TypeError:
                    result = str(result)
            return [types.TextContent(type="text", text=result)]

        server.call_tool()(_call_tool)

        session_manager = StreamableHTTPSessionManager(app=server, stateless=True)
        token = self._token

        async def _mcp_endpoint(scope: Scope, receive: Receive, send: Send) -> None:
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization", b"").decode("latin-1")
            if auth != f"Bearer {token}":
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
            await session_manager.handle_request(scope, receive, send)

        @asynccontextmanager
        async def _lifespan(_app: Any):
            async with session_manager.run():
                yield

        return Starlette(
            routes=[Mount("/mcp", app=_mcp_endpoint)],
            lifespan=_lifespan,
        )
