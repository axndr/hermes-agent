"""Hermetic tests for the Infisical integration.

We never hit an Infisical instance in tests — subprocess is mocked so
the suite stays fast and offline-safe.  The "live" pull is exercised
manually by `hermes secrets infisical setup` outside of pytest.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest


# Make the worktree importable without depending on the installed wheel.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources import infisical as inf  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_caches():
    inf._reset_cache_for_tests()
    yield
    inf._reset_cache_for_tests()


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point Hermes at an isolated home directory."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    if hasattr(hermes_constants, "_HERMES_HOME_CACHE"):
        hermes_constants._HERMES_HOME_CACHE = None  # type: ignore[attr-defined]
    return home


def _fake_export_payload(items):
    return json.dumps(items)


def _fake_binary(tmp_path):
    binary = tmp_path / "infisical"
    binary.write_text("")
    return binary


def _run_router(*, login_stdout="tok.abc.def", export_stdout="[]",
                login_rc=0, export_rc=0, login_stderr="", export_stderr="",
                calls=None):
    """Build a subprocess.run stand-in that answers login and export calls.

    ``calls`` (a list, when given) records (cmd, env) tuples per call so
    tests can assert on command shape and credential plumbing.
    """
    def fake_run(cmd, **kwargs):
        if calls is not None:
            calls.append((list(cmd), dict(kwargs.get("env") or {})))
        if "login" in cmd:
            return mock.Mock(
                returncode=login_rc, stdout=login_stdout, stderr=login_stderr
            )
        return mock.Mock(
            returncode=export_rc, stdout=export_stdout, stderr=export_stderr
        )
    return fake_run


# ---------------------------------------------------------------------------
# fetch_infisical_secrets — token auth path
# ---------------------------------------------------------------------------


def test_fetch_happy_path_token(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([
        {"key": "OPENAI_API_KEY", "value": "sk-abc", "type": "shared"},
        {"key": "ANTHROPIC_API_KEY", "value": "sk-ant-xyz", "type": "shared"},
    ])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )

    secrets, warnings = inf.fetch_infisical_secrets(
        project_id="proj-uuid",
        environment="prod",
        token="st.fake.token",
        binary=binary,
        use_cache=False,
    )
    assert secrets == {
        "OPENAI_API_KEY": "sk-abc",
        "ANTHROPIC_API_KEY": "sk-ant-xyz",
    }
    assert warnings == []
    # A direct token skips the login exchange entirely.
    assert len(calls) == 1
    cmd, env = calls[0]
    assert cmd[0] == str(binary)
    assert "export" in cmd
    assert "--format=json" in cmd
    assert "--projectId=proj-uuid" in cmd
    assert "--env=prod" in cmd
    assert "--path=/" in cmd
    assert env["INFISICAL_TOKEN"] == "st.fake.token"


def test_fetch_credentials_never_on_argv(monkeypatch, tmp_path):
    """Security regression: token and client secret must only travel via
    the subprocess environment, never argv (visible in /proc)."""
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess,
        "run",
        _run_router(login_stdout="tok.exchanged", export_stdout=payload,
                    calls=calls),
    )

    inf.fetch_infisical_secrets(
        project_id="p",
        environment="prod",
        client_id="cid-123",
        client_secret="csecret-456",
        binary=binary,
        use_cache=False,
    )
    for cmd, _env in calls:
        joined = " ".join(cmd)
        assert "csecret-456" not in joined
        assert "cid-123" not in joined
        assert "tok.exchanged" not in joined


def test_fetch_universal_auth_exchange(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess,
        "run",
        _run_router(login_stdout="tok.exchanged\n", export_stdout=payload,
                    calls=calls),
    )

    secrets, _ = inf.fetch_infisical_secrets(
        project_id="p",
        environment="prod",
        client_id="cid",
        client_secret="csecret",
        binary=binary,
        use_cache=False,
    )
    assert secrets == {"K": "v"}
    assert len(calls) == 2
    login_cmd, login_env = calls[0]
    assert "login" in login_cmd
    assert "--method=universal-auth" in login_cmd
    assert "--plain" in login_cmd
    assert login_env["INFISICAL_UNIVERSAL_AUTH_CLIENT_ID"] == "cid"
    assert login_env["INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET"] == "csecret"
    export_cmd, export_env = calls[1]
    assert "export" in export_cmd
    assert export_env["INFISICAL_TOKEN"] == "tok.exchanged"


def test_fetch_login_tolerates_update_banner(monkeypatch, tmp_path):
    """Older CLIs mix an update notice into stdout; the token is the last
    non-empty line."""
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    banner = "A new release of infisical is available: 0.43.88 -> 0.43.99\ntok.real\n"
    calls = []
    monkeypatch.setattr(
        inf.subprocess,
        "run",
        _run_router(login_stdout=banner, export_stdout=payload, calls=calls),
    )

    inf.fetch_infisical_secrets(
        project_id="p",
        environment="prod",
        client_id="cid",
        client_secret="cs",
        binary=binary,
        use_cache=False,
    )
    _, export_env = calls[1]
    assert export_env["INFISICAL_TOKEN"] == "tok.real"


def test_fetch_login_garbage_output_raises(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    monkeypatch.setattr(
        inf.subprocess,
        "run",
        _run_router(login_stdout="something went wrong here"),
    )

    with pytest.raises(RuntimeError, match="did not return a token"):
        inf.fetch_infisical_secrets(
            project_id="p",
            environment="prod",
            client_id="cid",
            client_secret="cs",
            binary=binary,
            use_cache=False,
        )


def test_fetch_login_failure(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    monkeypatch.setattr(
        inf.subprocess,
        "run",
        _run_router(login_rc=1, login_stderr="error: Invalid credentials"),
    )

    with pytest.raises(RuntimeError, match="Invalid credentials"):
        inf.fetch_infisical_secrets(
            project_id="p",
            environment="prod",
            client_id="cid",
            client_secret="cs",
            binary=binary,
            use_cache=False,
        )


# ---------------------------------------------------------------------------
# fetch_infisical_secrets — export handling
# ---------------------------------------------------------------------------


def test_fetch_skips_invalid_env_names(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([
        {"key": "VALID_KEY", "value": "v1"},
        {"key": "1BAD_START", "value": "v2"},
        {"key": "has spaces", "value": "v3"},
        {"key": "DASH-KEY", "value": "v4"},
    ])
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload)
    )

    secrets, warnings = inf.fetch_infisical_secrets(
        project_id="p",
        environment="prod",
        token="t",
        binary=binary,
        use_cache=False,
    )
    assert secrets == {"VALID_KEY": "v1"}
    assert len(warnings) == 3


def test_fetch_export_failure(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    monkeypatch.setattr(
        inf.subprocess,
        "run",
        _run_router(export_rc=1, export_stderr="error: project not found"),
    )

    with pytest.raises(RuntimeError, match="project not found"):
        inf.fetch_infisical_secrets(
            project_id="p",
            environment="prod",
            token="t",
            binary=binary,
            use_cache=False,
        )


def test_fetch_timeout(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)

    def fake_run(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="infisical", timeout=30)

    monkeypatch.setattr(inf.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        inf.fetch_infisical_secrets(
            project_id="p",
            environment="prod",
            token="t",
            binary=binary,
            use_cache=False,
        )


def test_fetch_non_json(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout="not json at all")
    )

    with pytest.raises(RuntimeError, match="non-JSON"):
        inf.fetch_infisical_secrets(
            project_id="p",
            environment="prod",
            token="t",
            binary=binary,
            use_cache=False,
        )


def test_fetch_tolerates_banner_before_json(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = (
        "A new release of infisical is available\n"
        + _fake_export_payload([{"key": "K", "value": "v"}])
    )
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload)
    )

    secrets, _ = inf.fetch_infisical_secrets(
        project_id="p",
        environment="prod",
        token="t",
        binary=binary,
        use_cache=False,
    )
    assert secrets == {"K": "v"}


def test_fetch_unexpected_shape(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout='{"key": "v"}')
    )

    with pytest.raises(RuntimeError, match="unexpected shape"):
        inf.fetch_infisical_secrets(
            project_id="p",
            environment="prod",
            token="t",
            binary=binary,
            use_cache=False,
        )


def test_fetch_empty_output_warns(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    monkeypatch.setattr(inf.subprocess, "run", _run_router(export_stdout=""))

    secrets, warnings = inf.fetch_infisical_secrets(
        project_id="p",
        environment="prod",
        token="t",
        binary=binary,
        use_cache=False,
    )
    assert secrets == {}
    assert warnings and "no output" in warnings[0]


def test_fetch_requires_credentials():
    with pytest.raises(RuntimeError, match="credentials are empty"):
        inf.fetch_infisical_secrets(project_id="p", environment="prod")


def test_fetch_requires_environment():
    with pytest.raises(RuntimeError, match="environment is empty"):
        inf.fetch_infisical_secrets(project_id="p", environment="", token="t")


def test_fetch_server_url_sets_env(monkeypatch, tmp_path):
    """server_url must be plumbed into the subprocess as INFISICAL_API_URL."""
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )

    inf.fetch_infisical_secrets(
        project_id="p",
        environment="prod",
        token="t",
        binary=binary,
        use_cache=False,
        server_url="https://infisical.example.com",
    )
    _, env = calls[0]
    assert env["INFISICAL_API_URL"] == "https://infisical.example.com"


def test_fetch_no_server_url_does_not_set_env(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )
    monkeypatch.delenv("INFISICAL_API_URL", raising=False)

    inf.fetch_infisical_secrets(
        project_id="p",
        environment="prod",
        token="t",
        binary=binary,
        use_cache=False,
    )
    _, env = calls[0]
    assert "INFISICAL_API_URL" not in env


def test_fetch_stale_shell_token_does_not_leak_into_exchange(
    monkeypatch, tmp_path
):
    """A stale INFISICAL_TOKEN in the user's shell must not shadow the
    machine-identity exchange result."""
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setenv("INFISICAL_TOKEN", "stale.shell.token")
    monkeypatch.setattr(
        inf.subprocess,
        "run",
        _run_router(login_stdout="tok.fresh", export_stdout=payload,
                    calls=calls),
    )

    inf.fetch_infisical_secrets(
        project_id="p",
        environment="prod",
        client_id="cid",
        client_secret="cs",
        binary=binary,
        use_cache=False,
    )
    _, export_env = calls[1]
    assert export_env["INFISICAL_TOKEN"] == "tok.fresh"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_fetch_cache_hits(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )

    kw = dict(project_id="p", environment="prod", token="t", binary=binary,
              cache_ttl_seconds=60, home_path=tmp_path)
    inf.fetch_infisical_secrets(**kw)
    inf.fetch_infisical_secrets(**kw)
    assert len(calls) == 1  # cached on second call


def test_fetch_environment_part_of_cache_key(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )

    base = dict(project_id="p", token="t", binary=binary,
                cache_ttl_seconds=60, home_path=tmp_path)
    inf.fetch_infisical_secrets(environment="prod", **base)
    inf.fetch_infisical_secrets(environment="dev", **base)
    inf.fetch_infisical_secrets(environment="prod", secret_path="/svc", **base)
    assert len(calls) == 3  # each (env, path) combination fetches fresh


def test_fetch_cache_disabled(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )

    kw = dict(project_id="p", environment="prod", token="t", binary=binary,
              use_cache=False, home_path=tmp_path)
    inf.fetch_infisical_secrets(**kw)
    inf.fetch_infisical_secrets(**kw)
    assert len(calls) == 2


def test_disk_cache_written_0600_without_credentials(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    monkeypatch.setattr(
        inf.subprocess,
        "run",
        _run_router(login_stdout="tok.exchanged", export_stdout=payload),
    )

    inf.fetch_infisical_secrets(
        project_id="p",
        environment="prod",
        client_id="cid-abc",
        client_secret="csecret-xyz",
        binary=binary,
        cache_ttl_seconds=60,
        home_path=tmp_path,
    )
    cache_file = tmp_path / "cache" / "infisical_cache.json"
    assert cache_file.exists()
    assert stat.S_IMODE(cache_file.stat().st_mode) == 0o600
    content = cache_file.read_text()
    assert "csecret-xyz" not in content
    assert "tok.exchanged" not in content
    payload_obj = json.loads(content)
    assert payload_obj["secrets"] == {"K": "v"}


def test_disk_cache_short_circuits_across_processes(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )

    kw = dict(project_id="p", environment="prod", token="t", binary=binary,
              cache_ttl_seconds=60, home_path=tmp_path)
    inf.fetch_infisical_secrets(**kw)
    # Simulate a fresh process: clear only the in-memory layer.
    inf._CACHE.clear()
    secrets, _ = inf.fetch_infisical_secrets(**kw)
    assert secrets == {"K": "v"}
    assert len(calls) == 1  # disk cache answered the second call


def test_disk_cache_expires_with_ttl(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )

    kw = dict(project_id="p", environment="prod", token="t", binary=binary,
              cache_ttl_seconds=60, home_path=tmp_path)
    inf.fetch_infisical_secrets(**kw)
    inf._CACHE.clear()
    # Age the disk entry past the TTL.
    cache_file = tmp_path / "cache" / "infisical_cache.json"
    payload_obj = json.loads(cache_file.read_text())
    payload_obj["fetched_at"] -= 120
    cache_file.write_text(json.dumps(payload_obj))

    inf.fetch_infisical_secrets(**kw)
    assert len(calls) == 2  # expired → refetched


def test_disk_cache_foreign_key_ignored(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )

    kw = dict(project_id="p", environment="prod", binary=binary,
              cache_ttl_seconds=60, home_path=tmp_path)
    inf.fetch_infisical_secrets(token="token-one", **kw)
    inf._CACHE.clear()
    inf.fetch_infisical_secrets(token="token-two", **kw)
    assert len(calls) == 2  # different credential → disk entry ignored


def test_disk_cache_corrupt_file_falls_through(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "infisical_cache.json").write_text("{corrupt json")

    secrets, _ = inf.fetch_infisical_secrets(
        project_id="p", environment="prod", token="t", binary=binary,
        cache_ttl_seconds=60, home_path=tmp_path,
    )
    assert secrets == {"K": "v"}
    assert len(calls) == 1
    # The corrupt file was replaced with a valid one.
    payload_obj = json.loads((cache_dir / "infisical_cache.json").read_text())
    assert payload_obj["secrets"] == {"K": "v"}


def test_reset_cache_for_tests_deletes_disk_file(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "infisical_cache.json"
    cache_file.write_text("{}")
    inf._reset_cache_for_tests(tmp_path)
    assert not cache_file.exists()
    # Idempotent.
    inf._reset_cache_for_tests(tmp_path)


# ---------------------------------------------------------------------------
# apply_infisical_secrets
# ---------------------------------------------------------------------------


def _apply_env(monkeypatch, **env):
    for var in ("INFISICAL_TOKEN", "INFISICAL_CLIENT_ID",
                "INFISICAL_CLIENT_SECRET"):
        monkeypatch.delenv(var, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def test_apply_disabled_returns_empty():
    result = inf.apply_infisical_secrets(enabled=False)
    assert result.ok
    assert result.secrets == {}


def test_apply_missing_credentials(monkeypatch):
    _apply_env(monkeypatch)
    result = inf.apply_infisical_secrets(
        enabled=True, project_id="p", environment="prod"
    )
    assert not result.ok
    assert "INFISICAL_CLIENT_ID" in result.error


def test_apply_missing_project_id(monkeypatch):
    _apply_env(monkeypatch, INFISICAL_TOKEN="t")
    result = inf.apply_infisical_secrets(enabled=True, environment="prod")
    assert not result.ok
    assert "project_id" in result.error


def test_apply_missing_environment(monkeypatch):
    _apply_env(monkeypatch, INFISICAL_TOKEN="t")
    result = inf.apply_infisical_secrets(enabled=True, project_id="p")
    assert not result.ok
    assert "environment" in result.error


def test_apply_missing_binary(monkeypatch):
    _apply_env(monkeypatch, INFISICAL_TOKEN="t")
    monkeypatch.setattr(inf, "find_infisical", lambda: None)
    result = inf.apply_infisical_secrets(
        enabled=True, project_id="p", environment="prod"
    )
    assert not result.ok
    assert "not found on PATH" in result.error


def test_apply_token_wins_over_pair(monkeypatch, tmp_path):
    """When both a token and a client id/secret pair are set, the token is
    used directly — no login exchange."""
    binary = _fake_binary(tmp_path)
    _apply_env(
        monkeypatch,
        INFISICAL_TOKEN="direct-token",
        INFISICAL_CLIENT_ID="cid",
        INFISICAL_CLIENT_SECRET="cs",
    )
    monkeypatch.setattr(inf, "find_infisical", lambda: binary)
    payload = _fake_export_payload([{"key": "K", "value": "v"}])
    calls = []
    monkeypatch.setattr(
        inf.subprocess, "run", _run_router(export_stdout=payload, calls=calls)
    )

    result = inf.apply_infisical_secrets(
        enabled=True, project_id="p", environment="prod",
        cache_ttl_seconds=0, home_path=tmp_path,
    )
    try:
        assert result.ok
        assert len(calls) == 1
        _, env = calls[0]
        assert env["INFISICAL_TOKEN"] == "direct-token"
    finally:
        os.environ.pop("K", None)


def test_apply_does_not_override_existing(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    _apply_env(monkeypatch, INFISICAL_TOKEN="t")
    monkeypatch.setenv("EXISTING_KEY", "original")
    os.environ.pop("NEW_KEY", None)
    monkeypatch.setattr(inf, "find_infisical", lambda: binary)
    payload = _fake_export_payload([
        {"key": "EXISTING_KEY", "value": "from-infisical"},
        {"key": "NEW_KEY", "value": "new-value"},
    ])
    monkeypatch.setattr(inf.subprocess, "run", _run_router(export_stdout=payload))

    result = inf.apply_infisical_secrets(
        enabled=True, project_id="p", environment="prod",
        override_existing=False, cache_ttl_seconds=0, home_path=tmp_path,
    )
    try:
        assert result.ok
        assert os.environ["EXISTING_KEY"] == "original"
        assert os.environ["NEW_KEY"] == "new-value"
        assert "EXISTING_KEY" in result.skipped
        assert "NEW_KEY" in result.applied
    finally:
        # apply set NEW_KEY directly on os.environ (not via monkeypatch),
        # so clean it ourselves to keep later tests hermetic.
        os.environ.pop("NEW_KEY", None)


def test_apply_override_existing(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    _apply_env(monkeypatch, INFISICAL_TOKEN="t")
    monkeypatch.setenv("EXISTING_KEY", "original")
    monkeypatch.setattr(inf, "find_infisical", lambda: binary)
    payload = _fake_export_payload([
        {"key": "EXISTING_KEY", "value": "from-infisical"},
    ])
    monkeypatch.setattr(inf.subprocess, "run", _run_router(export_stdout=payload))

    result = inf.apply_infisical_secrets(
        enabled=True, project_id="p", environment="prod",
        override_existing=True, cache_ttl_seconds=0, home_path=tmp_path,
    )
    assert result.ok
    assert os.environ["EXISTING_KEY"] == "from-infisical"
    assert "EXISTING_KEY" in result.applied


@pytest.mark.parametrize(
    "bootstrap_var",
    ["INFISICAL_TOKEN", "INFISICAL_CLIENT_ID", "INFISICAL_CLIENT_SECRET"],
)
def test_apply_never_overrides_bootstrap_credentials(
    monkeypatch, tmp_path, bootstrap_var
):
    """None of the three bootstrap vars may be clobbered by a fetched
    secret of the same name, even with override_existing=True."""
    binary = _fake_binary(tmp_path)
    _apply_env(
        monkeypatch,
        INFISICAL_TOKEN="real-token",
        INFISICAL_CLIENT_ID="real-cid",
        INFISICAL_CLIENT_SECRET="real-cs",
    )
    monkeypatch.setattr(inf, "find_infisical", lambda: binary)
    payload = _fake_export_payload([
        {"key": bootstrap_var, "value": "evil-replacement"},
    ])
    monkeypatch.setattr(inf.subprocess, "run", _run_router(export_stdout=payload))

    before = os.environ[bootstrap_var]
    result = inf.apply_infisical_secrets(
        enabled=True, project_id="p", environment="prod",
        override_existing=True, cache_ttl_seconds=0, home_path=tmp_path,
    )
    assert result.ok
    assert os.environ[bootstrap_var] == before
    assert bootstrap_var in result.skipped


def test_apply_swallows_fetch_errors(monkeypatch, tmp_path):
    binary = _fake_binary(tmp_path)
    _apply_env(monkeypatch, INFISICAL_TOKEN="t")
    monkeypatch.setattr(inf, "find_infisical", lambda: binary)
    monkeypatch.setattr(
        inf.subprocess,
        "run",
        _run_router(export_rc=1, export_stderr="error: 403 forbidden"),
    )

    result = inf.apply_infisical_secrets(
        enabled=True, project_id="p", environment="prod",
        cache_ttl_seconds=0, home_path=tmp_path,
    )
    assert not result.ok
    assert "403" in result.error


# ---------------------------------------------------------------------------
# env_loader integration
# ---------------------------------------------------------------------------


def test_env_loader_skips_when_disabled(tmp_path, monkeypatch):
    from hermes_cli import env_loader

    (tmp_path / "config.yaml").write_text(
        "secrets:\n  infisical:\n    enabled: false\n"
    )
    called = {"n": 0}
    monkeypatch.setattr(
        inf, "apply_infisical_secrets",
        lambda **kw: called.__setitem__("n", called["n"] + 1),
    )
    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)
    assert called["n"] == 0


def test_env_loader_calls_infisical_when_enabled(tmp_path, monkeypatch):
    from hermes_cli import env_loader

    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  infisical:\n"
        "    enabled: true\n"
        "    project_id: proj-1\n"
        "    environment: prod\n"
        "    secret_path: /svc\n"
        "    server_url: https://infisical.example.com\n"
        "    cache_ttl_seconds: 120\n"
        "    override_existing: false\n"
    )
    captured = {}

    def fake_apply(**kwargs):
        captured.update(kwargs)
        return inf.FetchResult(
            secrets={"K": "v"}, applied=["K"], skipped=[], warnings=[]
        )

    monkeypatch.setattr(
        "agent.secret_sources.infisical.apply_infisical_secrets", fake_apply
    )
    monkeypatch.setenv("K", "")
    env_loader.reset_secret_source_cache()
    env_loader._apply_external_secret_sources(tmp_path)

    assert captured["project_id"] == "proj-1"
    assert captured["environment"] == "prod"
    assert captured["secret_path"] == "/svc"
    assert captured["server_url"] == "https://infisical.example.com"
    assert captured["cache_ttl_seconds"] == 120.0
    assert captured["override_existing"] is False
    assert env_loader.get_secret_source("K") == "infisical"
    assert env_loader.format_secret_source_suffix("K") == " (from Infisical)"
