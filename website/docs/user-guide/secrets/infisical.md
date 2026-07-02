# Infisical

Pull API keys from [Infisical](https://infisical.com/) at process startup instead of storing them in plaintext inside `~/.hermes/.env`. One bootstrap credential (a machine identity) replaces N per-provider keys, and rotating a credential becomes a single change in the Infisical dashboard. Infisical is open source and self-hostable, so this also works when your secrets never leave your own infrastructure.

## How it works

1. You create a **machine identity** in Infisical (auth method: Universal Auth), give it read access to a project, and generate a **client secret**.
2. Hermes stores the identity's client id + client secret in `~/.hermes/.env` as `INFISICAL_CLIENT_ID` / `INFISICAL_CLIENT_SECRET`.
3. Every time `hermes` (or the gateway, or a cron job) starts, after `~/.hermes/.env` has loaded, Hermes exchanges those credentials for a short-lived access token and calls `infisical export` for your configured project + environment + folder path, setting the returned keys into `os.environ`. Credentials only ever travel via the subprocess environment — never on the command line.
4. By default Hermes **overrides** values already in your environment, so Infisical is the source of truth — rotate a key once in the dashboard and every Hermes process picks it up on next start. Flip `override_existing: false` in config if you want `.env` to win instead.

Unlike the Bitwarden integration, Hermes does **not** download the `infisical` CLI itself — Infisical ships native packages per platform, and the setup wizard walks you through the right install command for your machine, then re-checks.

## Why machine identities (and why not `infisical login`)

Machine identities are Infisical's credential for non-interactive workloads: no human, no 2FA prompt, scoped to exactly the projects and environments you attach them to. The client secret is the credential — anyone with the id + secret pair can read every secret the identity has access to, so treat it like a high-value bearer token: store it in `.env` (not `config.yaml`), and revoke + regenerate from the dashboard if it ever leaks.

An interactive `infisical login` session also works with the CLI, but it's tied to your user account and its keyring session — fine on a laptop, wrong for gateways, cron, and fleets. Hermes therefore treats machine identities as the primary path. If you already have a **service token** or an externally-managed identity access token, you can supply it via `INFISICAL_TOKEN` instead and it wins over the id/secret pair.

## Setup

### 1. Create a machine identity and client secret

In the Infisical dashboard (cloud or your self-hosted instance):

1. **Organization → Access Control → Identities → Create identity**, auth method **Universal Auth**.
2. Open your **Project → Access Control → Machine identities** and add the identity with read access.
3. Add your provider keys as secrets in the environment you'll sync (e.g. `prod`). The secret **key** becomes the environment variable name — use `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, etc.
4. On the identity's **Universal Auth** page, create a **client secret** and copy it when it is shown — it cannot be retrieved later. The **client ID** is on the same page.

### 2. Run the wizard

```bash
hermes secrets infisical setup
```

It will:

1. Check for the `infisical` CLI on PATH. If missing, it prints the install commands for your platform (apt/dnf/apk/brew/scoop/npm) and waits while you run them in another terminal.
2. Prompt for the client id and client secret (input is hidden). Stored in `~/.hermes/.env` as `INFISICAL_CLIENT_ID` / `INFISICAL_CLIENT_SECRET`.
3. Ask which Infisical instance you use — **US Cloud**, **EU Cloud**, or **self-hosted / custom URL**. Stored in `config.yaml` as `secrets.infisical.server_url` and passed to the CLI as `INFISICAL_API_URL`.
4. Prompt for the project ID (Project → Settings → Project ID), the **environment slug** (e.g. `prod` — the slug, not the display name), and optionally a folder path (default `/`).
5. Test-fetch the secrets and show you which env vars will resolve.
6. Flip `secrets.infisical.enabled: true`.

Non-interactive setup is also supported via flags:

```bash
hermes secrets infisical setup \
  --client-id <identity-client-id> \
  --client-secret "$INFISICAL_CLIENT_SECRET" \
  --server-url https://infisical.example.com \
  --project-id <project-uuid> \
  --environment prod
```

### 3. Confirm

```bash
hermes secrets infisical status
```

From now on, every `hermes` invocation pulls fresh secrets at startup. You'll see a one-line summary in stderr the first time secrets are applied in a process.

## CLI

| Command | What it does |
|---|---|
| `hermes secrets infisical setup` | Interactive wizard (check/install CLI, store credentials, pick project + environment, test fetch) |
| `hermes secrets infisical status` | Show config + CLI version + credential presence |
| `hermes secrets infisical sync` | Dry-run: pull secrets now and show what would be applied |
| `hermes secrets infisical sync --apply` | Pull and export into the current shell's environment |
| `hermes secrets infisical install` | Guided install walkthrough for the `infisical` CLI (no auth required) |
| `hermes secrets infisical disable` | Flip `enabled: false`; leaves credentials + project config in place |

`inf` works as a short alias everywhere: `hermes secrets inf status`.

## Configuration

Defaults in `~/.hermes/config.yaml`:

```yaml
secrets:
  infisical:
    enabled: false
    client_id_env: INFISICAL_CLIENT_ID
    client_secret_env: INFISICAL_CLIENT_SECRET
    token_env: INFISICAL_TOKEN
    project_id: ""
    environment: ""
    secret_path: "/"
    server_url: ""
    cache_ttl_seconds: 300
    override_existing: true
```

| Key | Default | What it does |
|---|---|---|
| `enabled` | `false` | Master switch. When false, Infisical is never contacted. |
| `client_id_env` | `INFISICAL_CLIENT_ID` | Env var name that holds the machine identity client ID. |
| `client_secret_env` | `INFISICAL_CLIENT_SECRET` | Env var name that holds the machine identity client secret. |
| `token_env` | `INFISICAL_TOKEN` | Env var name for a pre-issued token (service token or identity access token). When set, it wins over the id/secret pair. |
| `project_id` | `""` | UUID of the project to sync from. |
| `environment` | `""` | Environment slug to sync (e.g. `dev`, `staging`, `prod`). **Required** — the `infisical` CLI silently defaults to `dev`, which is the wrong guess for most deployments, so Hermes refuses to fetch until you set it explicitly. |
| `secret_path` | `"/"` | Folder path within the environment. `/` is the project root; imported/linked secrets are included. |
| `server_url` | `""` | Infisical region or self-hosted endpoint. Empty = CLI default (US Cloud, `https://app.infisical.com`). Set to `https://eu.infisical.com` for EU Cloud, or your own URL for self-hosted. Plumbed into the subprocess as `INFISICAL_API_URL`. |
| `cache_ttl_seconds` | `300` | How long a fetch result is reused. Two layers share this TTL: an in-process cache and a disk cache at `~/.hermes/cache/infisical_cache.json` (mode 0600, holds secret values but never the credentials), so back-to-back CLI invocations skip the network too. Set to `0` to disable caching. |
| `override_existing` | `true` | When true, Infisical values overwrite anything already in env (so rotation in the dashboard actually takes effect). Flip to `false` if you want `.env` / shell exports to win locally. |

## Failure modes

Infisical never blocks Hermes startup. If anything goes wrong, you'll see a one-line warning in stderr and Hermes continues with whatever credentials `.env` already had:

| Symptom | Cause | Fix |
|---|---|---|
| `neither INFISICAL_TOKEN nor INFISICAL_CLIENT_ID+INFISICAL_CLIENT_SECRET are set` | Enabled in config but credentials cleared from `.env` | Re-run `hermes secrets infisical setup` |
| `infisical exited 1 during universal-auth login: … Invalid credentials` | Wrong client secret (a classic: copying the secret's *ID* instead of its value), identity created on a different instance, or wrong `server_url` | Create a fresh client secret, double-check the instance URL, re-run setup |
| `403` / permission errors during export | Identity not attached to the project, or lacks read access to this environment | Project → Access Control in the dashboard |
| `404` / not found during export | Wrong project ID, or an environment slug that doesn't exist in this project | Use the slug (e.g. `prod`), not the display name |
| `secrets.infisical.environment is empty` | Config half-filled | Set `environment` or re-run setup |
| `infisical timed out` | Network blocked or instance unreachable | Check connectivity to `app.infisical.com` (or your `server_url`) |
| `infisical CLI not found on PATH` | CLI not installed | Run `hermes secrets infisical install` for platform instructions |

## Security notes

- The machine identity credentials (`INFISICAL_CLIENT_ID` + `INFISICAL_CLIENT_SECRET`) are themselves sensitive — anyone with the pair can read every secret the identity has access to. Treat them the same as any other API key.
- Credentials and access tokens are passed to the `infisical` subprocess via its environment, never on the command line, so they don't show up in `/proc/<pid>/cmdline` or process listings.
- Hermes will refuse to let Infisical overwrite its own bootstrap credentials, even with `override_existing: true`. If you store `INFISICAL_CLIENT_SECRET` (or `INFISICAL_TOKEN`, or `INFISICAL_CLIENT_ID`) as a secret inside the project, it's silently skipped during apply.
- The disk cache (`~/.hermes/cache/infisical_cache.json`) holds fetched secret *values* with mode 0600 — plaintext-equivalent to `~/.hermes/.env`, which you already accept — but never the bootstrap credentials or any access token.
- Scope the identity tightly: attach it only to the project it needs, with read-only access to one environment. A leaked `prod`-only identity can't read `dev`, and vice versa.

## When NOT to use this

- **Single-machine personal setups** where `~/.hermes/.env` is fine. You're trading one credential for another and adding a network dependency at startup.
- **Air-gapped environments** — unless you self-host Infisical inside the same network, which is exactly what it's for.
- **CI/CD** where the existing secrets-injection mechanism (GitHub Actions secrets, Vault, etc.) is already set up — pick one path, not two.

The good case for this is multi-machine fleets, shared dev boxes, gateway VPSes, or any setup where you want centralized rotation and revocation across multiple Hermes installations — especially when you want that on your own hardware.
