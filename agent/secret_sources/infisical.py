"""Infisical (`infisical` CLI) integration.

Hermes pulls API keys from Infisical at process startup so they don't
have to live in plaintext in ``~/.hermes/.env``.

Design summary
--------------

* The ``infisical`` CLI must already be on PATH (or in
  ``<hermes_home>/bin``).  Unlike ``bws`` we do NOT auto-download it:
  upstream ships tarballs + native packages per distro, so install is
  left to the user and ``hermes secrets infisical setup`` walks them
  through the right command for their platform.
* The bootstrap credentials are a machine identity (universal auth):
  a client id + client secret stored in ``~/.hermes/.env`` as
  ``INFISICAL_CLIENT_ID`` / ``INFISICAL_CLIENT_SECRET`` (or whatever
  names the user picked in ``secrets.infisical.client_id_env`` /
  ``client_secret_env``).  They are exchanged for a short-lived access
  token via ``infisical login --method=universal-auth --plain``.
  Alternatively a pre-issued token (service token or machine identity
  access token) can be supplied directly via ``token_env``.
* Credentials never appear on argv — the CLI reads
  ``INFISICAL_UNIVERSAL_AUTH_CLIENT_ID`` / ``..._CLIENT_SECRET`` and
  ``INFISICAL_TOKEN`` from the subprocess environment, keeping them out
  of ``/proc/<pid>/cmdline``.
* Pulling secrets is a single ``infisical export --format=json`` call.
  Unlike Bitwarden's flat projects, Infisical scopes a fetch by
  (project, environment, folder path), so all three are part of the
  cache key.  ``--env`` has NO safe default upstream (the CLI silently
  assumes ``dev``), so Hermes refuses to fetch until the user picks an
  environment explicitly.
* Failures NEVER block Hermes startup.  Missing binary, no network,
  revoked identity, etc. all emit a one-line warning and continue with
  whatever credentials ``.env`` already had.

The module is intentionally subprocess-driven rather than going through
the ``infisicalsdk`` Python package: the CLI is what self-hosted users
already have installed and authenticated, and one external binary is a
smaller supply-chain surface than another wheel dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.secret_sources.base import (
    ErrorKind,
    FetchResult as SourceFetchResult,
    SecretSource,
)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# How long to wait for infisical subprocesses, in seconds.  An uncached
# fetch makes at most two calls (login exchange + export), each with its
# own budget.
_INFISICAL_RUN_TIMEOUT = 30

# Env var names the Infisical CLI itself understands.  These are the
# canonical spellings from upstream docs — the *user-facing* names of the
# bootstrap vars in .env are configurable (client_id_env etc.) and get
# mapped onto these for the subprocess only.
_CLI_CLIENT_ID_VAR = "INFISICAL_UNIVERSAL_AUTH_CLIENT_ID"
_CLI_CLIENT_SECRET_VAR = "INFISICAL_UNIVERSAL_AUTH_CLIENT_SECRET"
_CLI_TOKEN_VAR = "INFISICAL_TOKEN"
_CLI_API_URL_VAR = "INFISICAL_API_URL"

# In-process cache so repeated load_hermes_dotenv() calls (CLI startup,
# gateway hot-reload, test suites) don't re-fetch from Infisical.
# (credential_fingerprint, project_id, environment, secret_path, server_url)
_CacheKey = Tuple[str, str, str, str, str]
_CACHE: Dict[_CacheKey, "_CachedFetch"] = {}

# Disk-persisted cache so back-to-back CLI invocations (scripts, cron, the
# gateway forking new agents) don't each pay the login + export round trips.
# Same contract as the Bitwarden disk cache: one JSON object per cache key,
# written atomically with mode 0600, holding only the secret VALUES — never
# the client secret or any access token.
_DISK_CACHE_BASENAME = "infisical_cache.json"


def _disk_cache_path(home_path: Optional[Path] = None) -> Path:
    """Return the disk cache path under hermes_home/cache/.

    `home_path` is what `load_hermes_dotenv()` already resolved; falling back
    to `$HERMES_HOME` / `~/.hermes` keeps direct callers working too.
    """
    if home_path is None:
        home_path = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return home_path / "cache" / _DISK_CACHE_BASENAME


def _cache_key_str(cache_key: _CacheKey) -> str:
    """Serialize a cache key to a stable string for JSON storage."""
    return "|".join(cache_key)


def _read_disk_cache(cache_key: _CacheKey, ttl_seconds: float,
                     home_path: Optional[Path] = None) -> Optional["_CachedFetch"]:
    """Return a cached entry from disk if fresh, else None.

    Best-effort: any I/O or parse error returns None and we re-fetch.
    """
    if ttl_seconds <= 0:
        return None
    path = _disk_cache_path(home_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("key") != _cache_key_str(cache_key):
        return None
    secrets = payload.get("secrets")
    fetched_at = payload.get("fetched_at")
    if not isinstance(secrets, dict) or not isinstance(fetched_at, (int, float)):
        return None
    typed_secrets: Dict[str, str] = {
        k: v for k, v in secrets.items() if isinstance(k, str) and isinstance(v, str)
    }
    entry = _CachedFetch(secrets=typed_secrets, fetched_at=float(fetched_at))
    if not entry.is_fresh(ttl_seconds):
        return None
    return entry


def _write_disk_cache(cache_key: _CacheKey, entry: "_CachedFetch",
                      home_path: Optional[Path] = None) -> None:
    """Persist a cache entry to disk atomically with mode 0600.

    Best-effort: any I/O error is swallowed (the next invocation will just
    re-fetch). We never want disk cache failures to break startup.
    """
    path = _disk_cache_path(home_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "key": _cache_key_str(cache_key),
            "secrets": entry.secrets,
            "fetched_at": entry.fetched_at,
        }
        fd, tmp = tempfile.mkstemp(
            prefix=".infisical_cache_", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    except OSError:
        pass  # best-effort — disk cache miss on next invocation is fine


@dataclass
class _CachedFetch:
    secrets: Dict[str, str]
    fetched_at: float

    def is_fresh(self, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Outcome of a single Infisical pull."""

    secrets: Dict[str, str] = field(default_factory=dict)
    applied: List[str] = field(default_factory=list)   # set into os.environ
    skipped: List[str] = field(default_factory=list)   # already set, not overridden
    warnings: List[str] = field(default_factory=list)  # non-fatal issues
    error: Optional[str] = None                        # fatal: nothing was fetched
    binary_path: Optional[Path] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ---------------------------------------------------------------------------
# Binary discovery — deliberately no auto-install (see module docstring)
# ---------------------------------------------------------------------------


def _hermes_bin_dir() -> Path:
    """Where Hermes stores its managed binaries.  Profile-aware."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "bin"


def find_infisical() -> Optional[Path]:
    """Return a path to a usable ``infisical`` binary, or None.

    Resolution order:
      1. ``<hermes_home>/bin/infisical``  (user-managed copy — preferred)
      2. ``shutil.which("infisical")``    (system PATH)

    There is no download fallback — ``hermes secrets infisical setup``
    walks the user through installing the CLI for their platform.
    """
    managed = _hermes_bin_dir() / _platform_binary_name()
    if managed.exists() and os.access(managed, os.X_OK):
        return managed

    system = shutil.which("infisical")
    if system:
        return Path(system)
    return None


def _platform_binary_name() -> str:
    import platform

    return "infisical.exe" if platform.system() == "Windows" else "infisical"


def infisical_version(binary: Path) -> Optional[str]:
    """Return the CLI's version string (e.g. ``0.43.88``), or None."""
    try:
        proc = subprocess.run(  # noqa: S603 — binary path is trusted
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    # Output shape: "infisical version 0.43.88"
    out = (proc.stdout or "").strip()
    return out.rsplit(" ", 1)[-1] if out else None


# ---------------------------------------------------------------------------
# Secret fetch + apply
# ---------------------------------------------------------------------------


def _credential_fingerprint(*parts: str) -> str:
    """SHA-256 prefix used as a cache key — never logged, never displayed."""
    joined = "\x00".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _subprocess_env(server_url: str) -> Dict[str, str]:
    """Base subprocess environment for infisical CLI calls.

    Strips any ambient CLI credential vars so a stale INFISICAL_TOKEN in
    the user's shell can't shadow the machine-identity exchange, then
    plumbs the configured server URL for self-hosted instances.  When
    ``server_url`` is empty, an inherited INFISICAL_API_URL from the
    shell is preserved so manual overrides keep working.
    """
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")
    # --silent still leaves the CLI's telemetry prompt possible on some
    # versions; disabling explicitly keeps stdout parseable.
    env.setdefault("INFISICAL_DISABLE_UPDATE_CHECK", "true")
    if server_url:
        env[_CLI_API_URL_VAR] = server_url
    return env


def _run_infisical(cmd: List[str], env: Dict[str, str],
                   what: str) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(  # noqa: S603 — infisical path is trusted
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=_INFISICAL_RUN_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"infisical timed out after {_INFISICAL_RUN_TIMEOUT}s during {what}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"failed to invoke infisical: {exc}") from exc

    if proc.returncode != 0:
        # The CLI writes auth/network errors to stderr in plain English.
        # Strip ANSI just in case and surface the first 200 chars.
        err = (proc.stderr or proc.stdout or "").strip().replace("\x1b", "")
        raise RuntimeError(
            f"infisical exited {proc.returncode} during {what}: {err[:200]}"
        )
    return proc


def _exchange_machine_identity(
    binary: Path, client_id: str, client_secret: str, server_url: str
) -> str:
    """Exchange universal-auth credentials for a short-lived access token.

    Runs ``infisical login --method=universal-auth --plain --silent`` with
    the credentials in the subprocess environment (the CLI reads
    ``INFISICAL_UNIVERSAL_AUTH_CLIENT_ID`` / ``..._CLIENT_SECRET``), so
    they never appear on argv.
    """
    env = _subprocess_env(server_url)
    env[_CLI_CLIENT_ID_VAR] = client_id
    env[_CLI_CLIENT_SECRET_VAR] = client_secret
    cmd = [str(binary), "login", "--method=universal-auth", "--plain", "--silent"]
    proc = _run_infisical(cmd, env, "universal-auth login")

    # --plain prints just the token, but older CLIs have been seen mixing
    # an update banner into stdout: take the last non-empty line and
    # sanity-check it looks like a bare token.
    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    token = lines[-1] if lines else ""
    if not token or " " in token:
        raise RuntimeError(
            "infisical login --plain did not return a token "
            "(unexpected output shape)"
        )
    return token


def fetch_infisical_secrets(
    *,
    project_id: str,
    environment: str,
    secret_path: str = "/",
    token: str = "",
    client_id: str = "",
    client_secret: str = "",
    binary: Optional[Path] = None,
    cache_ttl_seconds: float = 300,
    use_cache: bool = True,
    server_url: str = "",
    home_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Pull the secrets for (project, environment, path) from Infisical.

    Returns ``(secrets_dict, warnings_list)``.

    Auth precedence: an explicit ``token`` (service token or machine
    identity access token) wins; otherwise ``client_id`` +
    ``client_secret`` are exchanged for a short-lived access token first.

    Set ``server_url`` to point at a self-hosted instance or the EU
    cloud — e.g. ``https://infisical.example.com`` or
    ``https://eu.infisical.com``.  When empty, the CLI uses its built-in
    default (``https://app.infisical.com``, US Cloud).  This is plumbed
    into the subprocess as ``INFISICAL_API_URL``.

    Caching mirrors the Bitwarden backend: an in-process dict plus a
    disk-persisted JSON file under ``<hermes_home>/cache/infisical_cache.json``,
    sharing one TTL.  The (environment, secret_path) pair is part of the
    cache key because each names a different secret set.

    Raises :class:`RuntimeError` for fatal conditions (missing binary,
    auth failure, unparseable output).  Callers in the env_loader path
    catch this and emit a single warning; callers in the user-facing
    setup wizard let it propagate.
    """
    if not token and not (client_id and client_secret):
        raise RuntimeError("Infisical credentials are empty")
    if not project_id:
        raise RuntimeError("Infisical project_id is empty")
    if not environment:
        raise RuntimeError("Infisical environment is empty")
    secret_path = secret_path or "/"

    cred_fp = (
        _credential_fingerprint("token", token)
        if token
        else _credential_fingerprint("universal-auth", client_id, client_secret)
    )
    cache_key = (cred_fp, project_id, environment, secret_path, server_url or "")
    if use_cache:
        cached = _CACHE.get(cache_key)
        if cached and cached.is_fresh(cache_ttl_seconds):
            return cached.secrets, []
        disk_cached = _read_disk_cache(cache_key, cache_ttl_seconds, home_path)
        if disk_cached is not None:
            # Promote into in-process cache so subsequent fetches in the
            # same process skip the disk read too.
            _CACHE[cache_key] = disk_cached
            return disk_cached.secrets, []

    infisical = binary or find_infisical()
    if infisical is None:
        raise RuntimeError(
            "infisical CLI not found on PATH.  Run "
            "`hermes secrets infisical setup` for guided install steps, "
            "or see https://infisical.com/docs/cli/overview."
        )

    access_token = token or _exchange_machine_identity(
        infisical, client_id, client_secret, server_url
    )
    secrets, warnings = _run_infisical_export(
        infisical, access_token, project_id, environment, secret_path, server_url
    )
    entry = _CachedFetch(secrets=secrets, fetched_at=time.time())
    _CACHE[cache_key] = entry
    if use_cache:
        _write_disk_cache(cache_key, entry, home_path)
    return secrets, warnings


def _run_infisical_export(
    infisical: Path,
    access_token: str,
    project_id: str,
    environment: str,
    secret_path: str,
    server_url: str = "",
) -> Tuple[Dict[str, str], List[str]]:
    cmd = [
        str(infisical),
        "export",
        "--format=json",
        f"--projectId={project_id}",
        f"--env={environment}",
        f"--path={secret_path}",
        "--silent",
    ]
    env = _subprocess_env(server_url)
    env[_CLI_TOKEN_VAR] = access_token

    proc = _run_infisical(cmd, env, "export")

    raw = (proc.stdout or "").strip()
    if not raw:
        return {}, ["infisical returned no output (empty environment?)"]

    # `--silent` should keep stdout pure JSON, but be tolerant of a stray
    # banner line before the payload: parse from the first JSON bracket.
    start = raw.find("[")
    if start > 0:
        raw = raw[start:]

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"infisical returned non-JSON output: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError(
            f"infisical returned unexpected shape: {type(payload).__name__}"
        )

    secrets: Dict[str, str] = {}
    warnings: List[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        value = item.get("value")
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if not _is_valid_env_name(key):
            warnings.append(
                f"Skipping secret {key!r}: not a valid env-var name"
            )
            continue
        secrets[key] = value
    return secrets, warnings


def _is_valid_env_name(name: str) -> bool:
    if not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(c.isalnum() or c == "_" for c in name)


# ---------------------------------------------------------------------------
# Quicksilver secret-source registry adapter
# ---------------------------------------------------------------------------


class InfisicalSource(SecretSource):
    """Bulk Infisical backend for the shared secret-source orchestrator."""

    name = "infisical"
    label = "Infisical"
    shape = "bulk"

    def override_existing(self, cfg: dict) -> bool:
        return bool(isinstance(cfg, dict) and cfg.get("override_existing", True))

    def protected_env_vars(self, cfg: dict):
        if not isinstance(cfg, dict):
            cfg = {}
        return frozenset({
            str(cfg.get("token_env") or "INFISICAL_TOKEN"),
            str(cfg.get("client_id_env") or "INFISICAL_CLIENT_ID"),
            str(cfg.get("client_secret_env") or "INFISICAL_CLIENT_SECRET"),
        })

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Enable Infisical startup sync", "default": False},
            "project_id": {"description": "Infisical project UUID", "default": ""},
            "environment": {"description": "Environment slug", "default": ""},
            "secret_path": {"description": "Folder path", "default": "/"},
            "server_url": {"description": "Cloud or self-hosted API URL", "default": ""},
        }

    def remediation(self, kind, cfg: dict) -> str:
        if kind == ErrorKind.BINARY_MISSING:
            return "Run `hermes secrets infisical install`, then retry."
        return "Run `hermes secrets infisical setup` to verify the machine identity and project scope."

    def fetch(self, cfg: dict, home_path: Path) -> SourceFetchResult:
        result = SourceFetchResult()
        cfg = cfg if isinstance(cfg, dict) else {}
        token_env = str(cfg.get("token_env") or "INFISICAL_TOKEN")
        client_id_env = str(cfg.get("client_id_env") or "INFISICAL_CLIENT_ID")
        client_secret_env = str(
            cfg.get("client_secret_env") or "INFISICAL_CLIENT_SECRET"
        )
        token = os.environ.get(token_env, "").strip()
        client_id = os.environ.get(client_id_env, "").strip()
        client_secret = os.environ.get(client_secret_env, "").strip()
        project_id = str(cfg.get("project_id") or "").strip()
        environment = str(cfg.get("environment") or "").strip()

        if not token and not (client_id and client_secret):
            result.error = (
                f"neither {token_env} nor {client_id_env}+{client_secret_env} are set"
            )
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result
        if not project_id or not environment:
            missing = "project_id" if not project_id else "environment"
            result.error = f"secrets.infisical.{missing} is empty"
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result

        binary = find_infisical()
        result.binary_path = binary
        if binary is None:
            result.error = "infisical CLI not found on PATH"
            result.error_kind = ErrorKind.BINARY_MISSING
            return result

        try:
            secrets, warnings = fetch_infisical_secrets(
                project_id=project_id,
                environment=environment,
                secret_path=str(cfg.get("secret_path") or "/").strip() or "/",
                token=token,
                client_id=client_id,
                client_secret=client_secret,
                binary=binary,
                cache_ttl_seconds=float(cfg.get("cache_ttl_seconds", 300)),
                server_url=str(cfg.get("server_url") or "").strip(),
                home_path=home_path,
            )
        except (TypeError, ValueError) as exc:
            result.error = f"invalid Infisical configuration: {exc}"
            result.error_kind = ErrorKind.NOT_CONFIGURED
            return result
        except RuntimeError as exc:
            message = str(exc)
            lower = message.lower()
            result.error = message
            if "timed out" in lower:
                result.error_kind = ErrorKind.TIMEOUT
            elif any(word in lower for word in ("unauthorized", "forbidden", "auth", "token")):
                result.error_kind = ErrorKind.AUTH_FAILED
            elif any(word in lower for word in ("network", "connection", "resolve", "unreachable")):
                result.error_kind = ErrorKind.NETWORK
            else:
                result.error_kind = ErrorKind.INTERNAL
            return result

        result.secrets = secrets
        result.warnings.extend(warnings)
        return result


# ---------------------------------------------------------------------------
# Backward-compatible direct entry point — used by CLI/setup and older callers
# ---------------------------------------------------------------------------


def apply_infisical_secrets(
    *,
    enabled: bool,
    client_id_env: str = "INFISICAL_CLIENT_ID",
    client_secret_env: str = "INFISICAL_CLIENT_SECRET",
    token_env: str = "INFISICAL_TOKEN",
    project_id: str = "",
    environment: str = "",
    secret_path: str = "/",
    override_existing: bool = True,
    cache_ttl_seconds: float = 300,
    server_url: str = "",
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Pull secrets from Infisical and set them on ``os.environ``.

    This is the function ``load_hermes_dotenv()`` calls after the .env
    files have loaded.  It is intentionally defensive — any failure
    returns a :class:`FetchResult` with ``error`` set; it never raises.

    Auth: reads a machine identity from ``client_id_env`` +
    ``client_secret_env`` (primary), or a pre-issued token from
    ``token_env`` (which wins when set, covering service tokens and
    externally-managed identity tokens).

    ``server_url`` selects a self-hosted instance or non-US cloud
    (e.g. ``https://infisical.example.com``).  Empty string means the
    CLI's default (US Cloud).

    Parameters mirror the ``secrets.infisical.*`` config keys so the
    caller can just splat the dict in.
    """
    result = FetchResult()

    if not enabled:
        return result

    token = os.environ.get(token_env, "").strip()
    client_id = os.environ.get(client_id_env, "").strip()
    client_secret = os.environ.get(client_secret_env, "").strip()

    if not token and not (client_id and client_secret):
        result.error = (
            f"secrets.infisical.enabled is true but neither {token_env} nor "
            f"{client_id_env}+{client_secret_env} are set.  "
            "Run `hermes secrets infisical setup`."
        )
        return result

    if not project_id:
        result.error = (
            "secrets.infisical.project_id is empty.  "
            "Run `hermes secrets infisical setup`."
        )
        return result

    if not environment:
        # The CLI would silently default to `dev` here — refusing is
        # safer than fetching the wrong environment's secrets.
        result.error = (
            "secrets.infisical.environment is empty.  Set it to the "
            "environment slug to sync (e.g. prod) or run "
            "`hermes secrets infisical setup`."
        )
        return result

    binary = find_infisical()
    result.binary_path = binary
    if binary is None:
        result.error = (
            "infisical CLI not found on PATH.  "
            "Run `hermes secrets infisical setup` for guided install steps."
        )
        return result

    try:
        secrets, warnings = fetch_infisical_secrets(
            project_id=project_id,
            environment=environment,
            secret_path=secret_path,
            token=token,
            client_id=client_id,
            client_secret=client_secret,
            binary=binary,
            cache_ttl_seconds=cache_ttl_seconds,
            server_url=server_url,
            home_path=home_path,
        )
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(warnings)

    # Never let a fetched secret clobber the bootstrap credentials we
    # used to fetch it — same footgun guard as the Bitwarden backend,
    # extended to all three credential vars.
    bootstrap_vars = {token_env, client_id_env, client_secret_env}

    for key, value in secrets.items():
        if key in bootstrap_vars:
            result.skipped.append(key)
            continue
        if not override_existing and os.environ.get(key):
            result.skipped.append(key)
            continue
        os.environ[key] = value
        result.applied.append(key)

    return result


# ---------------------------------------------------------------------------
# Test hook — used by hermetic tests to flush the cache between cases.
# ---------------------------------------------------------------------------


def _reset_cache_for_tests(home_path: Optional[Path] = None) -> None:
    """Clear in-process AND disk caches.

    Tests can pass ``home_path`` to scope the disk cleanup to a tmpdir.
    Without it we fall back to the same default resolution as the cache
    writer itself.
    """
    _CACHE.clear()
    try:
        _disk_cache_path(home_path).unlink()
    except (FileNotFoundError, OSError):
        pass
