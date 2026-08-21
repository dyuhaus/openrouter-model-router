# Agent Instructions

This repository is a lightweight, provider-neutral Python package for routing
OpenRouter model selection. Keep it small and dependency-free unless a new
dependency clearly pays for itself.

## Commands

- Run tests: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests`
- Live catalog smoke test: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m openrouter_model_router.cli research --catalog /tmp/openrouter-model-router-catalog.json --timeout 30`
- Catalog freshness gate: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m openrouter_model_router.cli catalog-status --catalog /tmp/openrouter-model-router-catalog.json`
- Cost demo (no key, no network): `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 examples/cogs_demo.py`

## Rules

- Do not commit secrets, API keys, refreshed user catalogs, or `.env` files.
- Keep OpenRouter credentials in `OPENROUTER_API_KEY` or explicit caller config.
- Preserve standard-library-only runtime behavior unless the user explicitly wants a heavier integration.
- Add focused tests for routing, catalog parsing, CLI behavior, or client behavior when changing those areas.
- Do not commit a refreshed catalog or a run ledger; both are gitignored runtime state.

## Cost Accounting Rules

These exist because this package once estimated $0.00 for a 120K-token request
and nothing complained.

- **A missing number is `None`, never `0.0`.** Absent usage, absent cost, and
  absent pricing must all be representable as "unknown". Anything that collapses
  unknown into zero silently understates spend.
- **Unpriced is not free.** `pricing_known=False` marks models whose price the
  provider does not publish (the `-1` sentinel on the meta-routers, or no
  pricing block). Their `$0.00` estimate is not a measurement.
- **A flag is not evidence; derive it from the data.** `pricing_known` is
  computed from the prices actually present every time a `ModelInfo` is built,
  including on load. Never default it to `True`, and never let a stored `true`
  outrank absent, null, negative or non-numeric prices beside it. When the
  record is ambiguous the answer is UNKNOWN: refusing to quote is always safer
  than quoting zero, because zero is the number that wins every cost comparison.
- **Money nobody could estimate is not $0.00 of spend.** Reported charges on
  runs with no estimate are summed and reported on their own
  (`unreconciled_reported_cost_usd`); a reconciliation report may not say `ok`
  while real dollars sit outside the comparison.
- **Record every attempt.** The ledger takes gate failures and errors as well as
  successes. The retry multiplier is attempts-over-accepted-outputs; a
  success-only ledger can only ever report 1.0, and the cost model stays a guess.
- **A gate that raises has failed.** Never let an exception in a validator read
  as a pass.
- **A check that examined nothing has not passed.** Zero comparable runs, zero
  models, an unreadable timestamp: all of these exit non-zero. "I could not tell"
  and "it was fine" must never share an exit code.
- **The fetch time belongs to the fetch.** `ModelCatalog.fetched_at` is written
  by a refresh and by nothing else. `updated_at` moves on any local edit, so
  deriving freshness from it means a 2019 catalog reports as fetched today - the
  bug that made the staleness advice in the reconciliation report unactionable.
  Anything claiming a catalog is current must be able to fail on `catalog-status`.
- **Every new checker ships with a negative control.** Break it deliberately,
  watch the test fail, and say so in the PR. A validator that has never been
  seen to fail is indistinguishable from one that does nothing.
- **No credential may be required to test.** Model calls go through
  `HttpTransport`; `FakeTransport` is shipped in the package so the whole path
  is exercisable with no key. The live path must fail loudly and send nothing
  when `OPENROUTER_API_KEY` is unset.

## Git Workflow (machine standard)
This repo follows /home/dyadmin/AGENTS.md "Git Workflow Standard".
- Default branch: main (protected, PR-only, squash merge)
- Branches: feat/ fix/ chore/ docs/ exp/ (+ agent/<harness>/ optional)
- Commits: Conventional Commits; hooks must pass; never --no-verify
- Review: run `/code-reviewer` on the branch BEFORE opening the PR; address all
  findings, then request David's approval (agent PRs require it)
- Deploy coupling: <none | "merging main deploys to X — humans merge">
- Long-lived branch exceptions: <none | list + purpose>
