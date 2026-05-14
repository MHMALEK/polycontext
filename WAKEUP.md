# Wake-up summary — adapter bake-off

Ran the eval autonomously. Below is what happened, what I found, and what's
ready for you to look at over coffee.

## TL;DR

- **Built**: full eval harness in `eval/bakeoff/` — separate from the runtime app, has its own tests, CLI, and report writer
- **Installed**: `claude-agent-sdk`, `aider-chat`, and the `audioop-lts` Python-3.13 shim (aider's voice import chain breaks without it on Py3.13)
- **Ran**: ask bake-off (6 questions × 2 adapters = 12 calls) and decompose bake-off (2 tickets × 2 adapters = 4 calls)
- **Did NOT run**: implement bake-off (would push real branches to `tract1/*` on GitLab — needs your sign-off)
- **Cost**: ~$0.08 total across all runs
- **3 real bugs surfaced** that I fixed during the night (details below)

## Final results

### ask (eval-20260514T093934Z)

| Adapter | runs | success | avg score | avg ms | total $ |
|---|---|---|---|---|---|
| `aider` | 6 | 6/6 (100%) | 0.84 | 15 s | $0.011 |
| `baseline` | 6 | 5/6 (83%) | 0.63 | 121 s | $0.011 |

[`report`](eval/outputs/eval-20260514T093934Z/report.md) | [`summary.json`](eval/outputs/eval-20260514T093934Z/summary.json)

### decompose (eval-20260514T095521Z)

| Adapter | runs | success | avg score | avg ms | total $ |
|---|---|---|---|---|---|
| `baseline` | 2 | 1/2 (50%) | 1.00 | 80 s | $0.053 |
| `aider` | 2 | 1/2 (50%) | 0.55 | 25 s | $0.013 |

[`report`](eval/outputs/eval-20260514T095521Z/report.md) | [`summary.json`](eval/outputs/eval-20260514T095521Z/summary.json)

## What the numbers actually mean

**Aider's 0.84 ask score is misleading** — the rubric mostly checks
"non-empty answer of reasonable length." Aider returned 6/6 *non-empty
answers*, but most of them are variations of:

> "To answer your question, I need to see the code. Please provide the
> relevant files."

So Aider is technically passing the formal-shape checks but failing the
substance test (the `mentions[regex]` checks fail almost everywhere).
**Root cause**: Aider's working dir is `REPOS_ROOT`, which contains 4
sibling repos. Aider is a single-repo tool — its repo map gets confused
when pointed at a directory that contains multiple unrelated checkouts,
and falls back to asking the user to add files manually. This is a real
limitation, not a bug.

**Baseline 0.63 is more honest** — actual Sourcebot answers grounded in
real code, but with one Sourcebot timeout (q3) and a few cases where the
`mentions[regex]` rubric was too literal (the answer cited the validator
without the word "regex").

**Decompose: baseline wins on substance.** The case it succeeded on
(d2-farm-name-validation, single-repo) scored 1.00 — 4 well-formed subtasks
all in `frontend`. The one it failed (d1-supplier-prefill, cross-repo) hit
the deep-decompose 20-request limit before finishing.

**Aider decompose** succeeded on d1 but the output had `files=[]` and
`affected_repos=[]` (no real grounding), and failed JSON schema validation
on d2 — its model didn't return all required fields.

## Bugs I found and fixed during the night

1. **Aider on Python 3.13** — aider 0.86 imports pydub which imports
   audioop, which was removed in Python 3.13. `health()` shouted a
   misleading "not installed" because the import error originated deep
   in the dependency chain. **Fix**: `uv pip install audioop-lts`.

2. **TestClient event-loop bug** in the harness. After one call,
   subsequent calls failed with `RuntimeError: Event loop is closed`.
   **Fix**: replaced `fastapi.testclient.TestClient` with
   `httpx.AsyncClient(transport=httpx.ASGITransport(app=app))` in
   [`eval/bakeoff/client.py`](eval/bakeoff/client.py).

3. **env var propagation gap**. `pydantic-settings` reads `.env` into
   the `Settings` model fields but does not populate `os.environ`.
   libraries like litellm (used by aider) read `os.environ` directly,
   so they couldn't see your `GEMINI_API_KEY`. **Fix**: shim at the top
   of [`eval/bakeoff/cli.py`](eval/bakeoff/cli.py) that loads `.env`
   into `os.environ` before any other imports.

Smaller fix: **Aider model spec resolution** — your `.env` has bare
`DECOMPOSE_MODEL=gemini-2.5-pro` (no `provider:` prefix), so my adapter
was defaulting to Sonnet. Updated `_aider.py` to recognize bare names
like `gemini-*`, `gpt-*`, `claude-*`.

## What I did NOT run (deliberately)

- **Implement bake-off** — your `traceability` repo is at
  `tract1/application/api/traceability`, a real shared repo. Pushing
  branches and opening MRs there while you slept didn't seem right.
  Ready to run when you give the word; see "Running implement" below.

- **OpenCode and Goose** — both need `curl | bash` installers. I'm not
  piping arbitrary scripts from the web without explicit go-ahead. To
  install: `brew install opencode` and `brew install block/tap/goose`.

- **Cursor** — `cursor-agent` is on your PATH but needs `cursor-agent
  login` (interactive). It ran 6/6 cases in the first attempt and 6/6
  returned exit-1 "Authentication required". Once you log in, the next
  bake-off will pick it up automatically.

- **claude_sdk** — your shell session that launched Claude Code had
  `ANTHROPIC_API_KEY` set, but my bash subprocess sessions don't see it.
  Add `ANTHROPIC_API_KEY=...` to `.env` to enable this adapter.

## State of the world when you wake up

- Docker stack is **up** (sourcebot, postgres, redis). `make down` to stop.
- `.env` is symlinked into this worktree from the main project dir.
- All 44 + 19 = 63 tests pass: `uv run pytest`.
- New file: this `WAKEUP.md`. Nothing else committed — all the
  bug-fix edits are uncommitted changes you can review with `git diff`.

## To re-run today

```bash
# Make sure Docker is up (for baseline ask)
make eval-adapters                # confirm health states

# Same ask bake-off you'd run from cold
make eval-run ADAPTERS=baseline,aider JOB=ask

# Decompose
make eval-run ADAPTERS=baseline,aider JOB=decompose

# After installing more tools:
brew install block/tap/goose      # install Goose
# brew install opencode           # OpenCode (homebrew or direct download)
cursor-agent login                # one-time interactive auth
echo 'ANTHROPIC_API_KEY=...' >> .env  # enable claude_sdk
```

## Running implement (when you're ready)

```bash
# Single trivial implement task to validate the path end-to-end:
make eval-run ADAPTERS=aider JOB=implement IDS=i1-trivial-docstring

# What it'll do:
#   1. Create worktree at outputs/worktrees/<run_id>
#   2. Run aider against a fresh branch off origin/main
#   3. Commit any changes, push the branch
#   4. Open a draft MR (only if GITLAB_TOKEN is set in .env)
#
# Without GITLAB_TOKEN: branch gets pushed, no MR. Safe — you can
# inspect the branch and decide whether to open the MR manually.
```

## Open questions / next moves

1. The Aider Q&A weakness is the kind of thing a fix could address —
   point Aider at *one* repo (e.g. `traceability`) per ask request
   rather than at `REPOS_ROOT`. Want me to add a repo-aware mode?

2. The `mentions[regex]` rubric was too literal — answers that cited
   the actual validator pattern but said "the validation" instead of
   "the regex" failed. Either loosen the rubric, or add LLM-as-judge
   scoring on top of the rule checks.

3. Baseline deep-decompose hit the 20-request limit on the cross-repo
   ticket. Worth either raising the limit or having the auto-mode
   notice the limit was the failure cause and not the model quality.

4. Want me to push into Phase 3 (Tabby/OpenHands/Cline/Roo sidecars)
   next, or polish the existing 6 adapters first?
