"""Registration-level contract tests for the ``claude-acp`` provider.

hermes-claude-acp PLAN.md Phase 1 ("identity split"): ``claude-acp`` must
exist as its own first-class provider, distinct from ``copilot-acp``, across
every registration surface (registry, canonical list, aliases, model
normalization/metadata buckets, model.dev mapping, runtime resolution, and
the desktop dashboard card). This module is the single place that asserts
the whole checklist from SPEC.md §3 in one pass; individual surfaces also
get focused coverage in their own suites (test_api_key_providers.py,
test_model_normalize.py, test_provider_catalog.py, ...).

The dashboard-card assertion here is LOAD-BEARING for CI parity with
test_provider_parity.py: claude-acp must be configurable from the desktop
Accounts tab, exactly like copilot-acp.
"""

from fastapi.testclient import TestClient

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_auth_status,
    get_external_process_provider_status,
    resolve_external_process_provider_credentials,
    resolve_provider,
)
from agent.model_metadata import _PROVIDER_PREFIXES
from hermes_cli.model_normalize import _DOT_TO_HYPHEN_PROVIDERS, normalize_model_for_provider
from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_ALIASES, _PROVIDER_MODELS, parse_model_input, provider_label
from hermes_cli.provider_catalog import provider_catalog_by_slug
from hermes_cli.providers import HERMES_OVERLAYS, _LABEL_OVERRIDES
from hermes_cli.web_server import _SESSION_TOKEN, app

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


def test_registered_in_provider_registry_distinct_from_copilot_acp():
    assert "claude-acp" in PROVIDER_REGISTRY
    cfg = PROVIDER_REGISTRY["claude-acp"]
    assert cfg.auth_type == "external_process"
    assert cfg.inference_base_url == "acp://claude"
    assert cfg.base_url_env_var == "CLAUDE_ACP_BASE_URL"
    assert cfg.default_command == "claude-agent-acp"
    assert tuple(cfg.default_args) == ()
    assert "HERMES_CLAUDE_ACP_COMMAND" in cfg.command_env_vars
    # Must not share the Copilot command env vars — that's the whole point
    # of the identity split.
    assert "HERMES_COPILOT_ACP_COMMAND" not in cfg.command_env_vars
    assert "COPILOT_CLI_PATH" not in cfg.command_env_vars

    copilot_cfg = PROVIDER_REGISTRY["copilot-acp"]
    assert copilot_cfg.name == "GitHub Copilot ACP"  # reverted to stock
    assert copilot_cfg.inference_base_url == "acp://copilot"
    assert "HERMES_CLAUDE_ACP_COMMAND" not in copilot_cfg.command_env_vars


def test_canonical_providers_entry():
    entry = next((p for p in CANONICAL_PROVIDERS if p.slug == "claude-acp"), None)
    assert entry is not None, "claude-acp missing from CANONICAL_PROVIDERS"
    assert entry.label == "Claude Code · ACP"

    copilot_entry = next((p for p in CANONICAL_PROVIDERS if p.slug == "copilot-acp"), None)
    assert copilot_entry is not None
    assert copilot_entry.label == "GitHub Copilot ACP"  # reverted to stock


def test_curated_fallback_models_are_real_claude_ids():
    curated = _PROVIDER_MODELS.get("claude-acp")
    assert curated, "claude-acp needs a curated fallback model list"
    assert all(m.startswith("claude-") for m in curated)


def test_provider_aliases_resolve_to_claude_acp():
    for alias in ("claude-code-acp", "claudeacp", "claude-acp-agent"):
        assert _PROVIDER_ALIASES.get(alias) == "claude-acp"
        assert resolve_provider(alias) == "claude-acp"


def test_provider_label():
    assert provider_label("claude-acp") == "Claude Code · ACP"


def test_hermes_overlay_registered():
    overlay = HERMES_OVERLAYS.get("claude-acp")
    assert overlay is not None
    assert overlay.auth_type == "external_process"
    assert overlay.base_url_override == "acp://claude"
    assert overlay.base_url_env_var == "CLAUDE_ACP_BASE_URL"


def test_label_overrides_distinct_from_copilot_acp():
    assert _LABEL_OVERRIDES.get("claude-acp") == "Claude Code · ACP"
    assert _LABEL_OVERRIDES.get("copilot-acp") == "GitHub Copilot ACP"


def test_model_normalize_bucket_is_anthropic_style_not_copilot():
    assert "claude-acp" in _DOT_TO_HYPHEN_PROVIDERS
    assert normalize_model_for_provider("claude-sonnet-4.6", "claude-acp") == "claude-sonnet-4-6"


def test_model_metadata_prefix_registered():
    assert "claude-acp" in _PROVIDER_PREFIXES


def test_models_dev_mapping_points_to_anthropic():
    from agent.models_dev import PROVIDER_TO_MODELS_DEV

    assert PROVIDER_TO_MODELS_DEV.get("claude-acp") == "anthropic"


def test_parse_model_input_recognizes_claude_acp_prefix():
    provider, model = parse_model_input("claude-acp:claude-sonnet-5", "auto")
    assert provider == "claude-acp"
    assert model == "claude-sonnet-5"


def test_get_auth_status_and_credentials_round_trip(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: f"/usr/local/bin/{command}")
    monkeypatch.delenv("HERMES_CLAUDE_ACP_COMMAND", raising=False)

    status = get_auth_status("claude-acp")
    assert status["configured"] is True
    assert status["provider"] == "claude-acp"

    ext_status = get_external_process_provider_status("claude-acp")
    assert ext_status["command"] == "claude-agent-acp"

    creds = resolve_external_process_provider_credentials("claude-acp")
    assert creds["provider"] == "claude-acp"
    assert creds["base_url"] == "acp://claude"


def test_provider_catalog_accounts_tab():
    by = provider_catalog_by_slug()
    assert "claude-acp" in by
    d = by["claude-acp"]
    assert d.tab == "accounts"
    assert d.auth_type == "external_process"


def test_dashboard_card_present_and_distinct():
    """LOAD-BEARING for test_provider_parity.py: claude-acp must have its own
    Accounts-tab card, separate from copilot-acp's."""
    resp = client.get("/api/providers/oauth", headers=HEADERS)
    assert resp.status_code == 200, resp.text
    providers = {p["id"]: p for p in resp.json()["providers"]}

    assert "claude-acp" in providers
    assert providers["claude-acp"]["flow"] == "external"
    assert providers["claude-acp"]["disconnectable"] is False
    assert "claude" in providers["claude-acp"]["name"].lower()

    assert "copilot-acp" in providers
    assert providers["copilot-acp"]["name"] != providers["claude-acp"]["name"]
    assert "GitHub Copilot" in providers["copilot-acp"]["name"] or providers["copilot-acp"]["name"] == "GitHub Copilot (ACP)"
