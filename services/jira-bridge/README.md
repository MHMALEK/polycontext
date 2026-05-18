# jira-bridge

Small Node service that takes a Jira ticket key, runs it through your
[tech-decomposition](../../README.md) API, and posts the result back as
a comment on the ticket.

Self-contained on purpose — designed to lift into its own repo whenever
the integration outgrows being a sibling of tech-decomposition.

```
[Jira ticket TRT-123]
      │
      │  CLI:    npm run cli -- TRT-123
      │  HTTP:   POST /run {"key":"TRT-123"}
      ▼
[ jira-bridge ]
  ├─ GET  Jira /rest/api/3/issue/TRT-123  ← summary + description
  ├─ POST tech-decomposition /v1/adapters/{adapter}/decompose
  └─ POST Jira /rest/api/3/issue/TRT-123/comment  → ADF body
```

## Two ways to drive it

### 1. CLI — manual / scripted use

```bash
cd services/jira-bridge
cp .env.example .env
# fill in JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN
# point TECH_DECOMP_URL at your running tech-decomposition API

npm install
npm run cli -- TRT-123 TRT-456    # one or more ticket keys
```

Output per ticket:

```
→ TRT-123: fetching + decomposing…
  ok: 5 subtasks across 3 repos, cursor/composer-2, 42.1s
  comment: https://yourcompany.atlassian.net/browse/TRT-123?focusedCommentId=…
```

Exit code is non-zero if any ticket failed.

### 2. HTTP server — automation hook

```bash
npm run dev          # tsx watch, hot reload
# or
npm run build && npm start
```

```bash
curl -sS http://localhost:13200/run \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $JIRA_BRIDGE_SHARED_SECRET" \
  -d '{"key":"TRT-123"}'
```

The HTTP entry is what you'd point a Jira automation rule or a Slack /shortcut
handler at — anything that can POST a JSON body when a ticket reaches a
certain state.

## Environment

See [`.env.example`](.env.example). Required:

| Variable | What |
|---|---|
| `JIRA_BASE_URL` | `https://yourcompany.atlassian.net` |
| `JIRA_EMAIL` | The Atlassian account email |
| `JIRA_API_TOKEN` | Token from [id.atlassian.com](https://id.atlassian.com/manage-profile/security/api-tokens) |
| `TECH_DECOMP_URL` | Where the tech-decomposition FastAPI is reachable |

Optional but commonly set:

- `TECH_DECOMP_ADAPTER` — which adapter to call. Defaults to `cursor`
  (per [the bake-off](../../eval/outputs/bakeoff-20260518T134739Z/comparison.md)
  this is the best-quality adapter).
- `TECH_DECOMP_REPOS` — comma-separated list of repo names to scope the
  decompose to. Leave unset to use the server default.
- `JIRA_BRIDGE_SHARED_SECRET` — if set, `/run` requires
  `Authorization: Bearer <secret>`. The CLI ignores this.

## What the comment looks like

Atlassian Document Format (ADF), structured as:

- Header line (which adapter + model + wall time)
- Horizontal rule
- `## Overview` — the model's framing of the work
- `**Affected repos:**` line
- `## Subtasks (N)` — each as a `### NN — Title (repo, complexity)` block
  with description, files (linked when GitLab/GitHub URLs are known), and
  acceptance criteria
- `## Risks (N)` and `## Open questions (N)` when present

ADF renders properly in the Jira UI — paragraphs, bullets, monospace for file
paths, links you can click. No "code block of raw markdown" affordance.

## Files

| File | What |
|---|---|
| `src/config.ts` | Env loader, no schema lib |
| `src/jira.ts` | Minimal Jira Cloud v3 REST client (GET issue, POST comment) |
| `src/tech_decomp.ts` | Client for `/v1/adapters/{name}/decompose` |
| `src/adf.ts` | `Decomposition` → ADF document |
| `src/decompose_ticket.ts` | The flow: fetch → decompose → comment |
| `src/cli.ts` | CLI entry |
| `src/server.ts` | Fastify HTTP entry |
| `Dockerfile` | Node 22 slim, builds and runs the HTTP server |

## Moving this into its own repo later

Nothing in `src/` imports from outside this directory. To split it out:

```bash
cd services/jira-bridge
git init
git add .
git commit -m "init from tech-decomposition/services/jira-bridge"
git remote add origin <your new repo url>
git push -u origin main
```

The only thing you'd want to update on its way out is the README's
"see tech-decomposition" cross-link, and the bake-off references.

## What's intentionally not here

- **Slack.** The user requested Jira-only for now. Add a separate
  `src/slack.ts` later if a Slack message is wanted on success/failure.
- **Webhook receiver for Jira's "Issue Created" events.** `/run` is
  driven by an explicit POST; the simpler trigger is a Jira automation
  rule that POSTs to `/run` on the desired transition. Wiring is up to
  you.
- **OAuth / forge app.** Basic auth with an API token is enough for a
  single-team service. Move to OAuth only when this needs to act on
  behalf of multiple humans.
