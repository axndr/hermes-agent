# CLIProxy Claude Core-Tool Deferral Implementation Plan

> **For Hermes:** Implement task-by-task with tests before behavior changes.

**Goal:** Keep full Hermes tool capability while presenting fewer than 23 direct tools to Claude OAuth requests routed through CLIProxy.

**Architecture:** Extend Hermes' existing Tool Search progressive-disclosure bridge with an opt-in, runtime-matched core-deferral policy. The policy activates only when provider/base-URL/model selectors match, retains a configurable small direct-tool set, and defers every other registered tool behind `tool_search`, `tool_describe`, and `tool_call`. Reuse existing bridge dispatch, scope checks, guardrails, and activity unwrapping rather than adding a new transport.

**Tech Stack:** Python, Hermes tool registry/Tool Search, pytest, YAML configuration.

---

### Task 1: Add policy parsing and runtime matching tests

**Objective:** Define a disabled-by-default nested `core_deferral` policy with provider, base URL, and model selectors.

**Files:**
- Modify: `tests/tools/test_tool_search.py`
- Modify: `tools/tool_search.py`

**Steps:**
1. Add tests for defaults, selector matching, globbed Claude model matching, and non-matching Codex routes.
2. Run the focused tests and confirm they fail.
3. Implement validated parsing and `resolve_core_deferral()`.
4. Re-run the focused tests.

### Task 2: Add opt-in core classification tests

**Objective:** Defer core tools only when the resolved policy is active while preserving the legacy never-defer invariant by default.

**Files:**
- Modify: `tests/tools/test_tool_search.py`
- Modify: `tools/tool_search.py`

**Steps:**
1. Add tests showing default assembly keeps core tools direct.
2. Add tests showing forced assembly keeps only the configured direct set and inserts the three bridge tools.
3. Add scope/describe/call tests proving deferred core tools remain discoverable and dispatchable.
4. Implement optional `defer_core` / `keep_visible` arguments through classification and bridge helpers.
5. Run the Tool Search test module.

### Task 3: Thread runtime identity through tool assembly and execution

**Objective:** Resolve the policy against each agent's actual provider, base URL, and model, including session-specific model overrides.

**Files:**
- Modify: `model_tools.py`
- Modify: `agent/agent_init.py`
- Modify: `agent/tool_executor.py`
- Test: `tests/tools/test_tool_search.py`

**Steps:**
1. Extend `get_tool_definitions()` with an optional runtime identity and include it in the cache key.
2. Pass runtime identity from agent initialization.
3. Pass the same identity through bridge catalog/search/describe/call and scope checks.
4. Preserve all existing defaults when runtime identity is omitted.
5. Run Tool Search and agent tool-executor tests.

### Task 4: Configure the Rook profile

**Objective:** Activate the policy only for Claude models on the CLIProxy base URL and normalize client fingerprint headers.

**Files:**
- Modify: `/home/ravenly/.hermes/profiles/rook/config.yaml`

**Steps:**
1. Add `tools.tool_search.core_deferral` selectors for CLIProxy and `claude-*` models.
2. Keep a compact direct set: terminal/file/search/web/clarify.
3. Add `extra_headers` to the `cliproxy` custom provider so CLIProxy receives a consistent JS/Node Stainless tuple.
4. Validate YAML and read back selected non-secret fields.

### Task 5: Verify end to end

**Objective:** Prove the patch avoids current Extra Usage classification and still executes deferred tools.

**Steps:**
1. Run focused pytest modules.
2. Build a temporary Rook-derived profile selecting `custom:cliproxy / claude-opus-4-8`.
3. Inspect the assembled model-facing tool count and confirm it is below 23.
4. Run a Hermes one-shot no-tool response smoke test.
5. Run a Hermes one-shot requiring a deferred tool and verify the underlying tool executes.
6. Inspect the newest CLIProxy error/request logs only if a request fails; do not expose credentials or prompt bodies.
7. Review `git diff` and report files changed, test output, and restart requirement.
