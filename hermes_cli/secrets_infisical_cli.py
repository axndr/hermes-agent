"""CLI handlers for ``hermes secrets infisical ...``.

Subcommands:
    setup    — interactive wizard: check/install the CLI, store machine
               identity credentials, pick project + environment, test fetch
    status   — show current config + binary version
    sync     — run a fetch right now and show what would be applied (dry-run friendly)
    disable  — flip ``secrets.infisical.enabled`` to False
    install  — guided install walkthrough for the infisical CLI

Unlike the Bitwarden integration, Hermes never downloads the ``infisical``
binary itself — upstream ships native packages per platform, so setup
detects the platform and walks the user through the right install
command, then re-checks PATH.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from typing import List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.secret_sources import infisical as inf
from hermes_cli.config import (
    get_env_path,
    load_config,
    save_config,
    save_env_value,
)
from hermes_cli.secret_prompt import masked_secret_prompt


# ---------------------------------------------------------------------------
# Argparse wiring — called from hermes_cli.main
# ---------------------------------------------------------------------------


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``infisical`` subcommand tree to a parent parser.

    Called from ``hermes_cli.main`` as part of building the top-level
    ``hermes secrets`` parser.
    """
    sub = parent_parser.add_subparsers(dest="secrets_inf_command")

    setup = sub.add_parser(
        "setup",
        help=(
            "Interactive wizard: check CLI, store machine identity, "
            "pick project + environment"
        ),
    )
    setup.add_argument(
        "--client-id",
        help="Machine identity client ID (universal auth), non-interactive",
    )
    setup.add_argument(
        "--client-secret",
        help=(
            "Machine identity client secret, non-interactive "
            "(will be stored in .env)"
        ),
    )
    setup.add_argument(
        "--token",
        help=(
            "Pre-issued token (service token or machine identity access "
            "token) instead of a client id/secret pair"
        ),
    )
    setup.add_argument(
        "--project-id",
        help="Infisical project UUID, non-interactive",
    )
    setup.add_argument(
        "--environment",
        help="Environment slug to sync (e.g. dev, staging, prod)",
    )
    setup.add_argument(
        "--path",
        help="Folder path within the environment to sync (default: /)",
    )
    setup.add_argument(
        "--server-url",
        help=(
            "Infisical region / self-hosted endpoint. Examples: "
            "https://app.infisical.com (US, default), "
            "https://eu.infisical.com (EU), or your self-hosted URL. "
            "Skips the interactive region prompt."
        ),
    )
    setup.set_defaults(func=cmd_setup)

    status = sub.add_parser("status", help="Show config + binary version")
    status.set_defaults(func=cmd_status)

    sync = sub.add_parser("sync", help="Fetch secrets now and report what changed")
    sync.add_argument(
        "--apply",
        action="store_true",
        help="Actually export the secrets into the current shell's env (default: dry-run)",
    )
    sync.set_defaults(func=cmd_sync)

    disable = sub.add_parser("disable", help="Turn off the Infisical integration")
    disable.set_defaults(func=cmd_disable)

    install = sub.add_parser(
        "install",
        help="Guided install walkthrough for the infisical CLI",
    )
    install.set_defaults(func=cmd_install)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> int:
    console = Console()
    console.print(
        Panel.fit(
            "[bold]Infisical setup[/bold]\n\n"
            "Need a machine identity? In the Infisical web app:\n"
            "  Organization → Access Control → Identities → Create identity\n"
            "  (auth method: [cyan]Universal Auth[/cyan]) → add it to your project\n"
            "  with read access, then create a client secret.\n\n"
            "Copy the client secret when it is shown — it cannot be retrieved later.",
            border_style="cyan",
        )
    )

    # ------------------------------------------------------------------ binary
    console.print()
    console.print("[bold]Step 1[/bold]  Check for the infisical CLI")
    binary = inf.find_infisical()
    if binary is None:
        if not sys.stdin.isatty():
            console.print(
                "  [red]✗ infisical CLI not found on PATH.[/red]  "
                "Install it first — run `hermes secrets infisical install` "
                "for platform instructions."
            )
            return 1
        binary = _install_walkthrough(console)
        if binary is None:
            return 1
    version = inf.infisical_version(binary) or "version unknown"
    console.print(f"  [green]✓[/green] {binary}  ({version})")

    # -- non-interactive guard --
    if not sys.stdin.isatty():
        missing = _missing_non_tty_flags(args)
        if missing:
            console.print(
                f"  [red]Non-interactive mode (no TTY) requires all setup flags.[/red]\n"
                f"  Missing: {', '.join(missing)}\n\n"
                "  Usage:\n"
                "    hermes secrets infisical setup \\\n"
                "      --client-id 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx' \\\n"
                "      --client-secret '...' \\\n"
                "      --server-url 'https://app.infisical.com' \\\n"
                "      --project-id 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx' \\\n"
                "      --environment 'prod'"
            )
            return 1

    # ------------------------------------------------------------- credentials
    console.print()
    console.print("[bold]Step 2[/bold]  Provide your machine identity credentials")
    cfg = load_config()
    secrets_cfg = (cfg.setdefault("secrets", {})
                     .setdefault("infisical", {}))
    client_id_env = secrets_cfg.get("client_id_env", "INFISICAL_CLIENT_ID")
    client_secret_env = secrets_cfg.get(
        "client_secret_env", "INFISICAL_CLIENT_SECRET"
    )
    token_env = secrets_cfg.get("token_env", "INFISICAL_TOKEN")

    token = (args.token or "").strip()
    client_id = (args.client_id or "").strip()
    client_secret = (args.client_secret or "").strip()

    if token:
        save_env_value(token_env, token)
        os.environ[token_env] = token  # so the test fetch below sees it
        console.print(
            f"  [green]✓[/green] token stored in {get_env_path()} as {token_env}"
        )
    else:
        if not client_id:
            client_id = console.input(
                f"  Machine identity client ID ({client_id_env}): "
            ).strip()
        if not client_id:
            console.print("  [red]Empty client ID, aborting.[/red]")
            return 1
        if not client_secret:
            client_secret = masked_secret_prompt(
                f"  Paste client secret ({client_secret_env}): "
            ).strip()
        if not client_secret:
            console.print("  [red]Empty client secret, aborting.[/red]")
            return 1

        save_env_value(client_id_env, client_id)
        save_env_value(client_secret_env, client_secret)
        os.environ[client_id_env] = client_id
        os.environ[client_secret_env] = client_secret
        console.print(
            f"  [green]✓[/green] stored in {get_env_path()} as "
            f"{client_id_env} + {client_secret_env}"
        )

    # ------------------------------------------------------------------ region
    console.print()
    console.print("[bold]Step 3[/bold]  Pick your Infisical instance")
    server_url = _resolve_server_url(args, secrets_cfg, console)
    if server_url is None:
        return 1
    if server_url:
        console.print(f"  [green]✓[/green] using {server_url}")
    else:
        console.print(
            "  [green]✓[/green] using infisical default "
            "(US Cloud, https://app.infisical.com)"
        )

    # ------------------------------------------------ project + environment
    console.print()
    console.print("[bold]Step 4[/bold]  Pick project, environment, and path")
    project_id = (args.project_id or "").strip()
    if not project_id:
        existing_project = str(secrets_cfg.get("project_id", "") or "").strip()
        prompt = "  Project ID"
        if existing_project:
            prompt += f" (Enter to keep {existing_project})"
        prompt += ": "
        console.print(
            "  [dim]Find it in the Infisical web app under "
            "Project → Settings → Project ID.[/dim]"
        )
        project_id = console.input(prompt).strip() or existing_project
    if not project_id:
        console.print("  [red]Empty project ID, aborting.[/red]")
        return 1

    environment = (args.environment or "").strip()
    if not environment:
        existing_env = str(secrets_cfg.get("environment", "") or "").strip()
        prompt = "  Environment slug (e.g. dev, staging, prod)"
        if existing_env:
            prompt += f" (Enter to keep {existing_env})"
        prompt += ": "
        console.print(
            "  [dim]Use the slug, not the display name — check "
            "Project → Settings → Environments.[/dim]"
        )
        environment = console.input(prompt).strip() or existing_env
    if not environment:
        console.print("  [red]Empty environment, aborting.[/red]")
        return 1

    secret_path = (getattr(args, "path", None) or "").strip()
    if not secret_path:
        if sys.stdin.isatty():
            secret_path = console.input(
                "  Folder path [/]: "
            ).strip() or "/"
        else:
            secret_path = str(secrets_cfg.get("secret_path", "/") or "/")

    # ------------------------------------------------------------------- test
    console.print()
    console.print("[bold]Step 5[/bold]  Test fetch")
    try:
        secrets, warnings = inf.fetch_infisical_secrets(
            project_id=project_id,
            environment=environment,
            secret_path=secret_path,
            token=token,
            client_id=client_id,
            client_secret=client_secret,
            binary=binary,
            use_cache=False,
            server_url=server_url,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ Fetch failed: {exc}[/red]")
        _print_fetch_hints(console, str(exc))
        return 1

    bootstrap_vars = {token_env, client_id_env, client_secret_env}
    if not secrets:
        console.print(
            "  [yellow]Fetch succeeded but the environment has no secrets at "
            f"path {secret_path}.[/yellow]"
        )
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Name", style="cyan")
        table.add_column("Status")
        for key in sorted(secrets):
            if key in bootstrap_vars:
                status = "[dim]bootstrap credential — never overrides itself[/dim]"
            elif os.environ.get(key):
                status = "[yellow]already set in env (will be overwritten)[/yellow]"
            else:
                status = "[green]new[/green]"
            table.add_row(key, status)
        console.print(table)
    for w in warnings:
        console.print(f"  [yellow]warning:[/yellow] {w}")

    # ------------------------------------------------------------------- save
    secrets_cfg["enabled"] = True
    secrets_cfg["project_id"] = project_id
    secrets_cfg["environment"] = environment
    secrets_cfg["secret_path"] = secret_path
    secrets_cfg["server_url"] = server_url
    secrets_cfg.setdefault("client_id_env", client_id_env)
    secrets_cfg.setdefault("client_secret_env", client_secret_env)
    secrets_cfg.setdefault("token_env", token_env)
    secrets_cfg.setdefault("cache_ttl_seconds", 300)
    secrets_cfg.setdefault("override_existing", True)
    save_config(cfg)

    console.print()
    console.print(
        "[green]✓ Infisical is enabled.[/green]  "
        "Secrets will be pulled at the start of every Hermes process."
    )
    console.print(
        "  Status:  [cyan]hermes secrets infisical status[/cyan]\n"
        "  Refresh: [cyan]hermes secrets infisical sync[/cyan]\n"
        "  Disable: [cyan]hermes secrets infisical disable[/cyan]"
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    inf_cfg = (cfg.get("secrets") or {}).get("infisical") or {}

    enabled = bool(inf_cfg.get("enabled"))
    client_id_env = inf_cfg.get("client_id_env", "INFISICAL_CLIENT_ID")
    client_secret_env = inf_cfg.get("client_secret_env", "INFISICAL_CLIENT_SECRET")
    token_env = inf_cfg.get("token_env", "INFISICAL_TOKEN")
    project_id = inf_cfg.get("project_id", "")
    environment = str(inf_cfg.get("environment", "") or "").strip()
    secret_path = str(inf_cfg.get("secret_path", "/") or "/")
    server_url = str(inf_cfg.get("server_url", "") or "").strip()
    client_id_set = bool(os.environ.get(client_id_env))
    client_secret_set = bool(os.environ.get(client_secret_env))
    token_set = bool(os.environ.get(token_env))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("", style="bold")
    table.add_column("")
    table.add_row("Enabled",           _yn(enabled))
    table.add_row("Client ID env var", f"{client_id_env}  ({_yn(client_id_set)})")
    table.add_row(
        "Client secret env var", f"{client_secret_env}  ({_yn(client_secret_set)})"
    )
    table.add_row("Token env var",     f"{token_env}  ({_yn(token_set)})")
    table.add_row("Project ID",        project_id or "[dim](unset)[/dim]")
    table.add_row("Environment",       environment or "[red](unset — required)[/red]")
    table.add_row("Secret path",       secret_path)
    table.add_row(
        "Server URL",
        server_url or "[dim]default (US Cloud, https://app.infisical.com)[/dim]",
    )
    table.add_row(
        "Override existing", _yn(bool(inf_cfg.get("override_existing", True)))
    )
    table.add_row("Cache TTL (s)",     str(inf_cfg.get("cache_ttl_seconds", 300)))

    binary = inf.find_infisical()
    if binary:
        version = inf.infisical_version(binary) or "version unknown"
        table.add_row("infisical CLI",  f"{binary} ({version})")
    else:
        table.add_row("infisical CLI",  "[yellow]not installed[/yellow]")

    console.print(Panel(table, title="Infisical", border_style="cyan"))

    if not enabled:
        console.print("\n  Run [cyan]hermes secrets infisical setup[/cyan] to enable.")
        return 0
    if not (token_set or (client_id_set and client_secret_set)):
        console.print(
            f"\n  [yellow]Enabled but neither {token_env} nor "
            f"{client_id_env}+{client_secret_env} are set — Hermes will skip "
            "Infisical and warn on next startup.[/yellow]"
        )
    if not project_id:
        console.print(
            "\n  [yellow]Enabled but no project_id — nothing to fetch.[/yellow]"
        )
    if not environment:
        console.print(
            "\n  [yellow]Enabled but no environment — nothing to fetch.[/yellow]"
        )
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    inf_cfg = (cfg.get("secrets") or {}).get("infisical") or {}
    if not inf_cfg.get("enabled"):
        console.print(
            "[yellow]Infisical integration is disabled.  Run "
            "`hermes secrets infisical setup` first.[/yellow]"
        )
        return 1

    client_id_env = inf_cfg.get("client_id_env", "INFISICAL_CLIENT_ID")
    client_secret_env = inf_cfg.get("client_secret_env", "INFISICAL_CLIENT_SECRET")
    token_env = inf_cfg.get("token_env", "INFISICAL_TOKEN")
    token = os.environ.get(token_env, "").strip()
    client_id = os.environ.get(client_id_env, "").strip()
    client_secret = os.environ.get(client_secret_env, "").strip()
    if not token and not (client_id and client_secret):
        console.print(
            f"[red]Neither {token_env} nor {client_id_env}+{client_secret_env} "
            "are set.[/red]"
        )
        return 1

    project_id = inf_cfg.get("project_id", "")
    if not project_id:
        console.print("[red]No project_id configured.[/red]")
        return 1
    environment = str(inf_cfg.get("environment", "") or "").strip()
    if not environment:
        console.print("[red]No environment configured.[/red]")
        return 1
    secret_path = str(inf_cfg.get("secret_path", "/") or "/")
    server_url = str(inf_cfg.get("server_url", "") or "").strip()

    try:
        secrets, warnings = inf.fetch_infisical_secrets(
            project_id=project_id,
            environment=environment,
            secret_path=secret_path,
            token=token,
            client_id=client_id,
            client_secret=client_secret,
            use_cache=False,
            server_url=server_url,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Fetch failed: {exc}[/red]")
        _print_fetch_hints(console, str(exc))
        return 1

    if not secrets:
        console.print(
            f"[yellow]No secrets in environment {environment!r} at path "
            f"{secret_path!r}.[/yellow]"
        )
        return 0

    bootstrap_vars = {token_env, client_id_env, client_secret_env}
    override = bool(inf_cfg.get("override_existing", True)) or args.apply
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Action")
    applied = 0
    for key in sorted(secrets):
        if key in bootstrap_vars:
            table.add_row(key, "[dim]skip (bootstrap credential)[/dim]")
            continue
        already = bool(os.environ.get(key))
        if already and not override:
            table.add_row(key, "[dim]skip (already set)[/dim]")
            continue
        if args.apply:
            os.environ[key] = secrets[key]
            applied += 1
            table.add_row(key, "[green]exported[/green]" + (" (overrode)" if already else ""))
        else:
            table.add_row(key, "[green]would export[/green]" + (" (overrides)" if already else ""))

    console.print(table)
    for w in warnings:
        console.print(f"[yellow]warning:[/yellow] {w}")

    if not args.apply:
        console.print(
            "\n  This was a dry-run — secrets are picked up automatically on the "
            "next [cyan]hermes[/cyan] invocation.  Re-run with [cyan]--apply[/cyan] "
            "to export into the current shell instead."
        )
    else:
        console.print(f"\n  [green]Exported {applied} secret(s) into current process.[/green]")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    console = Console()
    cfg = load_config()
    inf_cfg = (cfg.setdefault("secrets", {})
                 .setdefault("infisical", {}))
    inf_cfg["enabled"] = False
    save_config(cfg)
    console.print(
        "[green]Disabled.[/green]  Infisical secrets will NOT be pulled on the next "
        "Hermes invocation.\n"
        "  Your machine identity credentials are left in .env — remove them "
        "manually if you also want to revoke the identity."
    )
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    console = Console()
    binary = inf.find_infisical()
    if binary:
        version = inf.infisical_version(binary) or "version unknown"
        console.print(f"[green]✓ already installed:[/green] {binary}  ({version})")
        return 0
    if not sys.stdin.isatty():
        _print_install_instructions(console)
        return 1
    binary = _install_walkthrough(console)
    return 0 if binary else 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yn(b: bool) -> str:
    return "[green]yes[/green]" if b else "[dim]no[/dim]"


def _missing_non_tty_flags(args: argparse.Namespace) -> List[str]:
    """Flags required for a non-interactive (no TTY) setup run."""
    missing = []
    has_token = bool(args.token and args.token.strip())
    has_pair = bool(
        args.client_id and args.client_id.strip()
        and args.client_secret and args.client_secret.strip()
    )
    if not has_token and not has_pair:
        missing.append("--client-id + --client-secret (or --token)")
    if not (args.server_url and args.server_url.strip()):
        # Also accept INFISICAL_API_URL env var as non-interactive substitute
        if not os.environ.get("INFISICAL_API_URL", "").strip():
            missing.append("--server-url")
    if not (args.project_id and args.project_id.strip()):
        missing.append("--project-id")
    if not (args.environment and args.environment.strip()):
        missing.append("--environment")
    return missing


def _print_fetch_hints(console: Console, error: str) -> None:
    """Map common fetch failures to actionable hints."""
    lowered = error.lower()
    if "invalid credentials" in lowered or "401" in lowered:
        console.print(
            "  [yellow]'Invalid credentials' usually means the client secret is "
            "wrong (did you copy the secret's ID instead of its value?), the "
            "identity was created on a different Infisical instance, or you "
            "picked the wrong region/server URL.  Re-run "
            "[cyan]hermes secrets infisical setup[/cyan] and double-check the "
            "instance URL.[/yellow]"
        )
    elif "403" in lowered or "forbidden" in lowered or "permission" in lowered:
        console.print(
            "  [yellow]Permission denied usually means the machine identity is "
            "not attached to this project, or lacks read access to this "
            "environment.  Check Project → Access Control in the Infisical "
            "web app.[/yellow]"
        )
    elif "404" in lowered or "not found" in lowered:
        console.print(
            "  [yellow]Not found usually means a wrong project ID or an "
            "environment slug that doesn't exist in this project (use the "
            "slug, e.g. 'prod', not the display name).[/yellow]"
        )


# Canonical Infisical region endpoints.  Keep in sync with what Infisical
# publishes — these are stable but if a third region appears, add it here
# and to the prompt below.
_REGION_PRESETS = [
    ("US Cloud  (https://app.infisical.com — infisical default)", ""),
    ("EU Cloud  (https://eu.infisical.com)", "https://eu.infisical.com"),
]


def _resolve_server_url(
    args: argparse.Namespace,
    secrets_cfg: dict,
    console: Console,
) -> Optional[str]:
    """Pick an Infisical server URL for setup.

    Resolution order:
      1. ``--server-url`` CLI flag (non-interactive)
      2. ``INFISICAL_API_URL`` env var (so users running with that already
         set in their shell don't have to re-enter it)
      3. Existing ``secrets.infisical.server_url`` value (for re-runs)
      4. Interactive menu: US / EU / self-hosted

    Returns the chosen URL as a string (empty string = infisical default,
    i.e. US Cloud).  Returns None if the user aborted with an empty
    custom URL.
    """
    if args.server_url and args.server_url.strip():
        return args.server_url.strip()

    env_url = os.environ.get("INFISICAL_API_URL", "").strip()
    if env_url:
        console.print(
            f"  Detected [cyan]INFISICAL_API_URL[/cyan]={env_url} in your shell — using it."
        )
        return env_url

    existing = str(secrets_cfg.get("server_url", "") or "").strip()
    if existing:
        console.print(
            f"  Existing config: [cyan]{existing}[/cyan]. "
            "Press Enter to keep, or pick a different option below."
        )

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="cyan", width=4)
    table.add_column("Region / endpoint")
    for i, (label, _url) in enumerate(_REGION_PRESETS, 1):
        table.add_row(str(i), label)
    table.add_row(str(len(_REGION_PRESETS) + 1), "Self-hosted / custom URL")
    console.print(table)

    custom_idx = len(_REGION_PRESETS) + 1
    while True:
        prompt = f"  Select instance [1-{custom_idx}]"
        if existing:
            prompt += " (Enter to keep current)"
        prompt += ": "
        choice = console.input(prompt).strip()
        if not choice:
            if existing:
                return existing
            console.print("  [red]Enter a number.[/red]")
            continue
        try:
            idx = int(choice)
        except ValueError:
            console.print("  [red]Enter a number.[/red]")
            continue
        if 1 <= idx <= len(_REGION_PRESETS):
            return _REGION_PRESETS[idx - 1][1]
        if idx == custom_idx:
            custom = console.input(
                "  Enter your Infisical server URL "
                "(e.g. https://infisical.example.com): "
            ).strip()
            if not custom:
                console.print("  [red]Empty URL, aborting.[/red]")
                return None
            if not custom.startswith(("http://", "https://")):
                console.print(
                    "  [yellow]Warning: URL doesn't start with http:// or "
                    "https:// — the CLI may reject it.[/yellow]"
                )
            return custom
        console.print(f"  [red]Out of range — pick 1-{custom_idx}.[/red]")


# ---------------------------------------------------------------------------
# Install walkthrough — Hermes never downloads the binary itself
# ---------------------------------------------------------------------------


def _install_commands() -> List[Tuple[str, List[str]]]:
    """Return (label, commands) candidates for this platform, best first.

    Sources: https://infisical.com/docs/cli/overview.  We only *suggest*
    these — the user runs them in their own shell (most need sudo), then
    setup re-checks PATH.
    """
    system = platform.system()
    out: List[Tuple[str, List[str]]] = []

    if system == "Darwin":
        out.append((
            "Homebrew",
            ["brew install infisical/get-cli/infisical"],
        ))
    elif system == "Windows":
        out.append((
            "Scoop",
            [
                "scoop bucket add org https://github.com/Infisical/scoop-infisical.git",
                "scoop install infisical",
            ],
        ))
    elif system == "Linux":
        if shutil.which("apt-get"):
            out.append((
                "apt (Debian/Ubuntu)",
                [
                    "curl -1sLf 'https://artifacts-cli.infisical.com/setup.deb.sh' | sudo -E bash",
                    "sudo apt-get update && sudo apt-get install -y infisical",
                ],
            ))
        if shutil.which("dnf") or shutil.which("yum"):
            pkg = "dnf" if shutil.which("dnf") else "yum"
            out.append((
                f"{pkg} (Fedora/RHEL/CentOS)",
                [
                    "curl -1sLf 'https://artifacts-cli.infisical.com/setup.rpm.sh' | sudo -E bash",
                    f"sudo {pkg} install -y infisical",
                ],
            ))
        if shutil.which("apk"):
            out.append((
                "apk (Alpine)",
                [
                    "curl -1sLf 'https://artifacts-cli.infisical.com/setup.alpine.sh' | sudo -E bash",
                    "sudo apk add infisical",
                ],
            ))
        if shutil.which("yay"):
            out.append(("AUR (Arch)", ["yay -S infisical-bin"]))

    # Universal fallback for anyone with node installed.
    if shutil.which("npm"):
        out.append(("npm (any platform)", ["npm install -g @infisical/cli"]))

    if not out:
        out.append((
            "Manual download",
            ["See https://infisical.com/docs/cli/overview for your platform."],
        ))
    return out


def _print_install_instructions(console: Console) -> None:
    console.print(
        "  [yellow]infisical CLI not found on PATH.[/yellow]  "
        "Install options for this machine:"
    )
    for label, commands in _install_commands():
        console.print(f"\n  [bold]{label}[/bold]")
        for c in commands:
            console.print(f"    [cyan]{c}[/cyan]")
    console.print(
        "\n  Full instructions: https://infisical.com/docs/cli/overview"
    )


def _install_walkthrough(console: Console) -> Optional[object]:
    """Interactive loop: show install commands, wait, re-check PATH.

    Returns the binary path once found, or None if the user gives up.
    The commands are run by the *user* in their own terminal — most need
    sudo and we don't want Hermes shelling out with elevated privileges.
    """
    _print_install_instructions(console)
    while True:
        answer = console.input(
            "\n  Run the install in another terminal, then press Enter to "
            "re-check (or type 'q' to abort): "
        ).strip().lower()
        if answer in ("q", "quit", "exit"):
            console.print("  [yellow]Aborted — re-run setup once installed.[/yellow]")
            return None
        binary = inf.find_infisical()
        if binary:
            return binary
        console.print("  [red]Still not found on PATH.[/red]")
