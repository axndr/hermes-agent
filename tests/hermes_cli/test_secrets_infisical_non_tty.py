"""Tests for hermes secrets infisical setup non-TTY guard.

Mirrors the Bitwarden guard (issue #40274): cmd_setup() must fail early
with a clear error in non-TTY environments instead of crashing on
interactive prompts, and must accept a fully-flagged invocation.
"""
from __future__ import annotations

import argparse

import pytest


def _fake_find(binary="/usr/bin/infisical"):
    return lambda: binary


class TestCmdSetupNonTtyGuard:
    """cmd_setup should fail early with a clear error in non-TTY environments."""

    @staticmethod
    def _make_args(**overrides):
        ns = argparse.Namespace(
            client_id=overrides.get("client_id", ""),
            client_secret=overrides.get("client_secret", ""),
            token=overrides.get("token", ""),
            server_url=overrides.get("server_url", ""),
            project_id=overrides.get("project_id", ""),
            environment=overrides.get("environment", ""),
            path=overrides.get("path", ""),
        )
        return ns

    def _stub_environment(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.secrets_infisical_cli.inf.find_infisical", _fake_find()
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_infisical_cli.inf.infisical_version",
            lambda _: "0.43.88",
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_infisical_cli.load_config", lambda: {}
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_infisical_cli.save_config", lambda cfg: None
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_infisical_cli.save_env_value", lambda *a: None
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_infisical_cli.get_env_path", lambda: "/tmp/.env"
        )
        monkeypatch.setattr(
            "hermes_cli.secrets_infisical_cli.inf.fetch_infisical_secrets",
            lambda **kw: ({"KEY": "val"}, []),
        )

    def test_missing_all_flags_returns_1(self, monkeypatch, capsys):
        """Non-TTY with no flags → exit 1 with missing flags listed."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.delenv("INFISICAL_API_URL", raising=False)
        self._stub_environment(monkeypatch)

        from hermes_cli.secrets_infisical_cli import cmd_setup

        result = cmd_setup(self._make_args())
        assert result == 1
        captured = capsys.readouterr()
        assert "Non-interactive mode" in captured.out
        assert "--client-id" in captured.out
        assert "--server-url" in captured.out
        assert "--project-id" in captured.out
        assert "--environment" in captured.out

    def test_missing_environment_only(self, monkeypatch, capsys):
        """Everything but --environment → the Missing line lists only it."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.delenv("INFISICAL_API_URL", raising=False)
        self._stub_environment(monkeypatch)

        from hermes_cli.secrets_infisical_cli import cmd_setup

        result = cmd_setup(self._make_args(
            client_id="cid",
            client_secret="cs",
            server_url="https://infisical.example.com",
            project_id="aaaa-bbbb",
        ))
        assert result == 1
        captured = capsys.readouterr()
        assert "Missing:" in captured.out
        missing_line = [
            l for l in captured.out.split("\n") if "Missing:" in l
        ][0]
        assert "--environment" in missing_line
        assert "--project-id" not in missing_line
        assert "--client-id" not in missing_line
        assert "--server-url" not in missing_line

    def test_token_substitutes_for_client_pair(self, monkeypatch):
        """--token alone satisfies the credential requirement."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        self._stub_environment(monkeypatch)

        from hermes_cli.secrets_infisical_cli import cmd_setup

        result = cmd_setup(self._make_args(
            token="st.token",
            server_url="https://infisical.example.com",
            project_id="aaaa-bbbb",
            environment="prod",
        ))
        assert result == 0

    def test_missing_server_url_with_env_var_passes(self, monkeypatch):
        """Non-TTY with INFISICAL_API_URL env set → server-url not required."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        monkeypatch.setenv("INFISICAL_API_URL", "https://infisical.example.com")
        self._stub_environment(monkeypatch)

        from hermes_cli.secrets_infisical_cli import cmd_setup

        result = cmd_setup(self._make_args(
            client_id="cid",
            client_secret="cs",
            project_id="aaaa-bbbb",
            environment="prod",
        ))
        assert result == 0

    def test_all_flags_provided_passes_guard(self, monkeypatch):
        """Non-TTY with all flags → guard passes, proceeds to setup."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        self._stub_environment(monkeypatch)

        from hermes_cli.secrets_infisical_cli import cmd_setup

        result = cmd_setup(self._make_args(
            client_id="cid",
            client_secret="cs",
            server_url="https://infisical.example.com",
            project_id="aaaa-bbbb",
            environment="prod",
        ))
        assert result == 0

    def test_missing_binary_non_tty_returns_1(self, monkeypatch, capsys):
        """Non-TTY with no CLI on PATH → exit 1 pointing at install, without
        entering the interactive walkthrough."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        self._stub_environment(monkeypatch)
        monkeypatch.setattr(
            "hermes_cli.secrets_infisical_cli.inf.find_infisical",
            lambda: None,
        )

        from hermes_cli.secrets_infisical_cli import cmd_setup

        result = cmd_setup(self._make_args(
            client_id="cid",
            client_secret="cs",
            server_url="https://infisical.example.com",
            project_id="aaaa-bbbb",
            environment="prod",
        ))
        assert result == 1
        captured = capsys.readouterr()
        assert "not found on PATH" in captured.out

    def test_tty_does_not_trigger_guard(self, monkeypatch):
        """With TTY, the guard should not trigger (interactive mode allowed)."""
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        self._stub_environment(monkeypatch)

        from hermes_cli.secrets_infisical_cli import cmd_setup

        # All values flagged so no interactive prompt is actually reached.
        result = cmd_setup(self._make_args(
            client_id="cid",
            client_secret="cs",
            server_url="https://infisical.example.com",
            project_id="aaaa-bbbb",
            environment="prod",
            path="/",
        ))
        assert result == 0
