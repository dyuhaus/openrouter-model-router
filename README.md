# OpenRouter Model Router

A small, dependency-free Python package for choosing an OpenRouter model for the task at hand. It is meant to be dropped into projects that already call LLMs and need a stable way to trade off cost, speed, context length, modality support, and quality.

## What It Does

- Selects the best compatible model for a `TaskSpec`.
- Estimates request cost from OpenRouter pricing metadata.
- Supports cheap, balanced, fast, and quality-first routing preferences.
- Filters by context window, modality, required capabilities, allowlists, blocklists, and max cost.
- Refreshes a local model catalog from OpenRouter's `/models` endpoint.
- Infers practical capabilities such as `vision`, `tool_use`, `json_mode`, `coding`, `reasoning`, `fast`, `cheap`, and `long_context`.
- Records lightweight outcome feedback so quality, reliability, and speed scores can improve over time.
- Provides an optional OpenAI-compatible chat-completion wrapper.
- **Reads `usage` back off every response** and returns reported tokens and cost alongside the content.
- **Appends a run ledger** (JSONL) with one row per call - successes, gate failures, and errors alike.
- **Reconciles estimated against reported cost** and flags drift, which is how a stale catalog gets caught.
- **Knows how old its own prices are.** The catalog records the time of the *fetch*, and `catalog-status` exits non-zero on anything it cannot prove is fresh.

The package stores state in normal JSON files and reads secrets only from environment variables or explicit constructor arguments.

## Install

From this checkout:

```bash
pip install -e /home/dyadmin/openrouter-model-router
```

No runtime dependencies are installed.

## Quick Start

```python
from openrouter_model_router import ModelRouter, TaskSpec

router = ModelRouter.from_file()

selection = router.select(TaskSpec(
    task_type="coding",
    input_tokens=6000,
    output_tokens=1500,
    required_capabilities=("coding", "tool_use"),
    preference="balanced",
    max_cost_usd=0.08,
))

print(selection.model.id, selection.estimated_cost_usd, selection.reasons)
```

To use the selected model for a chat completion:

```python
from openrouter_model_router import ModelRouter, OpenRouterClient, TaskSpec

router = ModelRouter.from_file()
client = OpenRouterClient.from_env()

result = router.chat_completion(
    client=client,
    task=TaskSpec(task_type="summarization", preference="cheap"),
    messages=[{"role": "user", "content": "Summarize this release note..."}],
)

print(result["selection"].model.id)
print(result["response"]["choices"][0]["message"]["content"])
```

## Cost Instrumentation

### The catalog must be populated or every estimate is $0.00

With no catalog on disk, `ModelRouter.from_file()` falls back to a single
bootstrap entry (`openrouter/auto`) whose price OpenRouter publishes as `-1`,
meaning "depends on where I route this". The router used to clamp that to `0.0`
and quote it as a real price:

```text
BEFORE  120K-token request -> openrouter/auto  estimated_cost_usd=$0.000000
```

`GET /api/v1/models` is **public** - verified 2026-08-21 returning HTTP 200 and
420 models with no credential - so there is no reason to run unpriced:

```bash
openrouter-model-router refresh
```

```json
{ "models": 420, "added": 420, "authenticated": false,
  "pricing": { "total": 420, "priced": 394, "free": 21, "pricing_unknown": 5 } }
```

```bash
openrouter-model-router estimate --input-tokens 40000 --output-tokens 120000 \
  --model anthropic/claude-opus-4.5 --model openai/gpt-4.1-mini --model deepseek/deepseek-v3.2
```

| model | $/M in | $/M out | 40K in + 120K out |
| --- | --- | --- | --- |
| `anthropic/claude-opus-4.5` | 5.00 | 25.00 | **$3.2000** |
| `google/gemini-2.5-flash` | 0.30 | 2.50 | **$0.3120** |
| `openai/gpt-4.1-mini` | 0.40 | 1.60 | **$0.2080** |
| `deepseek/deepseek-v3.2` | 0.269 | 0.40 | **$0.0588** |

Refresh prints pricing coverage on purpose. A catalog that *loads* is not a
catalog that *prices*, and `refresh` exits non-zero if it wrote one in which no
model carries a usable price.

**Unknown pricing is not free pricing.** `ModelInfo.pricing_known` is `False`
for the five meta-routers that publish a `-1` sentinel and for any model with no
pricing block. Zero is the *best possible* number in every cost comparison, so an
unknown price left as `0.0` wins the cheap route, slips under every budget
ceiling, and adds nothing to the spend estimate while the bill arrives anyway.
The package therefore keeps the two apart everywhere the number is used:

- `ModelInfo.cost_estimate()` returns `None`, not `0.0`, for an unpriced model.
  `estimated_cost_usd()` still returns a number, because scoring needs one.
- `estimate` prints `"estimated_cost_usd": null` with `"pricing_status": "UNKNOWN"`,
  and **exits non-zero when none of the requested models could be priced** - a
  command that produced no cost estimate has not succeeded.
- The router scores an unknown price as the **worst** case, not the best, and
  prefers a known price over an unknown one when scores tie.
- `max_cost_usd` **excludes** unpriced models. A ceiling that admits a model of
  unbounded price is not a ceiling.
- The ledger records `estimated_cost_usd: null` rather than `0.0`, and
  `ledger` reports `runs_missing_estimate` so a total summed over an unknown
  share of the runs cannot pass for the whole bill.
- `reconcile` sums real charges on runs with no estimate into
  `unreconciled_reported_cost_usd`, names the unpriced models, and refuses to
  report `ok` while any of that money is outstanding.

**`pricing_known` is derived from the prices, never read as a flag.** The flag
used to be *defaulted* on load - `bool(data.get("pricing_known", True))` - while
the prices beside it were read as `float(x or 0.0)`. A record with no pricing at
all therefore came back off disk as "price known: $0.00", and every hardened
consumer downstream did exactly what it was told about a number nobody measured.
`ModelInfo.from_dict` now re-derives the answer from the data:

| record on disk | verdict |
| --- | --- |
| no price keys at all | **UNKNOWN** |
| `null` on either side | **UNKNOWN** |
| `-1` on either side (the sentinel, written through) | **UNKNOWN** |
| one side published, the other absent | **UNKNOWN** |
| non-numeric junk | **UNKNOWN** |
| `pricing_known: true` beside absent or negative prices | **UNKNOWN** - a flag is not evidence of a price |
| `pricing_known: false` beside real prices | **UNKNOWN** - the verdict is never overridden upward |
| `0.0` / `0.0` with no `pricing_known` | **UNKNOWN** - indistinguishable from the old clamped sentinel, and every schema_version 1 catalog is full of that shape |
| `0.0` / `0.0` with `pricing_known: true` | known, genuinely **free** |
| real prices, flag or no flag | known |

The same derivation runs in `ModelInfo.__post_init__`, so `ModelInfo(id="x")`
with no prices is UNKNOWN rather than a confident $0.00 as well.

Verified against the live catalog on 2026-08-21, before and after the
derivation change, with identical results: 420 models, 415 with a known price
(394 charging something, 21 published at exactly `$0.00`), 5 carrying the `-1`
sentinel, 0 unaccounted for. At 34K in / 81K out the five sentinel models report
`UNKNOWN` and no priced model reports `$0.00`. The priced tiers are unchanged to
the cent, run against the same catalog on both sides of the change:
`mistralai/ministral-8b` **$0.0126**, `openai/gpt-4.1-mini` **$0.1432**,
`google/gemini-2.5-flash` **$0.2127**, `openai/gpt-5` **$0.8525**,
`anthropic/claude-sonnet-4` **$1.3170**. Deriving the flag costs a real catalog
nothing, because a real catalog states its prices - it only stops a *record
missing them* from claiming one.

### Catalog staleness

The fetch time is a property of the **fetch**, not of the load. `fetched_at` is
written by `refresh` and by nothing else; `updated_at` moves on any local edit,
including `record-outcome`. Keeping them apart is what makes staleness
detectable at all - an earlier version stamped `updated_at` on every `load()`,
so a catalog written in 2019 reported as fetched today and one load-and-save
cycle overwrote the real date on disk.

```bash
openrouter-model-router catalog-status --catalog ~/.cache/openrouter-model-router/catalog.json
```

```text
catalog: STALE
  path            /path/to/catalog.json
  models          420
  fetched_at      2026-05-23T14:26:13Z
  updated_at      2026-08-21T14:25:28Z
  age             90.00 days
  limit           7 days
  catalog was fetched 90.0 days ago (limit 7) - prices may have changed; run `openrouter-model-router refresh`
```

Exit code is 0 **only** for a catalog proven fresh. Every other answer exits 1,
including the ones that are not literally "old":

| condition | status | exits |
| --- | --- | --- |
| fetched inside the limit (default 7 days) | `fresh` | 0 |
| fetched longer ago than the limit | `stale` | 1 |
| never fetched (`fetched_at: null`, incl. the bootstrap catalog) | `never_fetched` | 1 |
| a `schema_version: 1` catalog inside the limit | `unverifiable_legacy_timestamp` | 1 |
| timestamp cannot be parsed | `unparseable_timestamp` | 1 |
| timestamp is in the future beyond clock skew | `clock_skew` | 1 |
| catalog holds 0 models | `empty_catalog` | 1 |

"I cannot tell how old this is" is never scored as new, and a catalog holding
nothing is a configuration failure rather than a pass.

A `schema_version: 1` catalog written before this change has no fetch timestamp
at all - the version that wrote it restamped `updated_at` on every load, so that
value is an *upper bound* on the real fetch time and such a catalog can only ever
look fresher than it is. It is therefore reported as `unverifiable_legacy_timestamp`
while it is inside the limit, and plainly `stale` once its upper bound is outside
it (which is sound, since the real fetch was at or before that). One `refresh`
records a real fetch time and clears it.

### Reading usage back

`usage` is optional in OpenRouter's response type and cost is optional inside
it, so a missing number is reported as `None`, never as `0.0`:

```python
result = client.chat("openai/gpt-4.1-mini", messages)

result.content            # the completion
result.prompt_tokens      # 194
result.completion_tokens  # 88
result.reported_cost_usd  # 7.83e-05, or None if the provider did not say
result.usage.source_fields  # {"prompt_tokens": "prompt_tokens", "cost": "cost"}
result.usage.missing_fields # names of anything the response did not carry
```

Field names are resolved through alias lists (`input_tokens`, `tokens_prompt`,
`total_cost`, ...) so other OpenAI-compatible gateways parse too, and the key
that actually matched is recorded on the ledger row.

### The run ledger

One JSONL row per call. **Failures are recorded**, because the retry multiplier
is attempts-over-accepted-outputs and a success-only ledger can only ever report
1.0:

```python
from openrouter_model_router import ModelRouter, OpenRouterClient, RunLedger, TaskSpec

ledger = RunLedger()  # ~/.cache/openrouter-model-router/runs.jsonl
outcome = ModelRouter.from_file().run(
    client=OpenRouterClient.from_env(),
    messages=messages,
    task=TaskSpec(task_type="coding"),
    ledger=ledger,
    task_label="lesson-3",
    gate=my_content_gate,   # -> [] to accept, or a list of rejection reasons
)
```

```bash
openrouter-model-router ledger
```

```json
{ "runs": 3, "completed": 1, "gate_failed": 1, "errored": 1,
  "estimated_cost_usd": 0.0192, "reported_cost_usd": 0.0324,
  "cost_of_failed_runs_usd": 0.0128,
  "retry_multiplier": 3.0, "retry_multiplier_measured": true }
```

With no accepted run yet, `retry_multiplier` is `null` and
`retry_multiplier_measured` is `false`. It says so rather than inventing a
number.

A gate that raises is a **failed** gate, not a passed one.

### Reconciliation

```bash
openrouter-model-router reconcile --catalog ~/.cache/openrouter-model-router/catalog.json --fail-on-drift
```

```text
reconciliation: DRIFT
  runs                 3 (2 comparable)
  estimated            $0.012800
  reported             $0.032400
  drift                $+0.019600 (+153.1%)
  missing reported     1
  catalog age         [ok] 0.4d (fetched 2026-08-21T14:25:28Z, 420 models)
                      catalog fetched 0.4 days ago (limit 7)
  [FLAG] openai/gpt-4.1-mini: cost off by 153.1% (tolerance 10.0%) while token counts agree within 0.0% - the catalog price is stale, run refresh
```

Pass `--catalog` and the age of the prices is checked alongside the drift maths.
Without it the report says `catalog age NOT CHECKED` - never "fresh", because a
check that never ran has not passed.

`--fail-on-drift` exits non-zero on three things, not one:

1. any model drifting past the tolerance;
2. a **stale catalog**, even when every drift number is inside tolerance - a
   drift comparison against prices nobody has refreshed is two unverified
   numbers agreeing with each other;
3. **zero comparable runs**. An empty ledger, or one where no row carried both an
   estimate and a reported cost, verified nothing. Exiting 0 there is a gate
   printing a pass over nothing.

The header line reports the whole verdict (`STALE_CATALOG (cost drift: ok)`), not
just the drift status, so it can never read `OK` on a run that exits 1.

Runs carrying only one of the two numbers are counted separately and excluded
from the drift maths - treating a missing reported cost as `$0.00` would
manufacture a 100% drift, and treating it as matching would hide a real one.

Drift has two causes and the report tells them apart rather than guessing. It
compares the ledger's `estimated_input_tokens`/`estimated_output_tokens` against
the reported `prompt_tokens`/`completion_tokens`:

| token counts | cost | `cause` |
| --- | --- | --- |
| agree | diverges | `stale_catalog_price` - run `refresh` |
| diverge | diverges | `wrong_token_estimate` - the `TaskSpec` sizes are wrong, the catalog is fine |
| incomplete | diverges | `undetermined` - not enough evidence to say |
| n/a | estimate is `$0.00`, provider charged | `no_catalog_price` |

Blaming the catalog for a token-estimate error would send someone to refresh a
catalog that is already correct. A cause is named only when *every* run in the
cost comparison also carried token counts - otherwise the token sums and the
cost sums cover different subsets of runs and cannot be divided against each
other, so the report says `undetermined` rather than guessing.

### Running it with no API key

Everything above except the actual model call works with no credential.
`FakeTransport` is a shipped implementation, not a test fixture, so the full
chain is exercisable today:

```bash
PYTHONPATH=src python3 examples/cogs_demo.py
```

The live path fails loudly and **sends nothing** when `OPENROUTER_API_KEY` is
unset:

```text
OpenRouterError: OPENROUTER_API_KEY is not set. The live OpenRouter path refuses
to run without a credential; no request was sent.
```

## Refresh The Catalog

Refresh the local catalog whenever you want the router to learn about newly available OpenRouter models:

```bash
openrouter-model-router refresh
```

Aliases are also available:

```bash
openrouter-model-router research
openrouter-model-router update
```

By default the catalog is stored at:

```text
~/.cache/openrouter-model-router/catalog.json
```

Override it with:

```bash
export OPENROUTER_MODEL_ROUTER_CATALOG=/path/to/catalog.json
```

The refresh command does not require a key for public model metadata, but if you need authenticated access:

```bash
export OPENROUTER_API_KEY=...
openrouter-model-router refresh
```

`refresh` fails rather than writing nothing over something: a response carrying
zero models exits non-zero and leaves the existing catalog alone, and a refresh
in which no model carries a usable price also exits non-zero. Both would
otherwise leave every downstream estimate at `$0.00` with a successful-looking
run behind it.

Check the age of what you have without fetching anything:

```bash
openrouter-model-router catalog-status          # exit 0 only when it is fresh
openrouter-model-router catalog-status --json --max-age-days 3
```

## CLI Selection

```bash
openrouter-model-router select \
  --task coding \
  --input-tokens 12000 \
  --output-tokens 2000 \
  --required-capability coding \
  --required-capability tool_use \
  --preference balanced \
  --max-cost 0.15
```

The command prints JSON with the selected model, estimated cost, score, and reasons.

## Configuration

Environment variables:

- `OPENROUTER_API_KEY` - used only for API calls, never written to disk.
- `OPENROUTER_BASE_URL` - defaults to `https://openrouter.ai/api/v1`.
- `OPENROUTER_HTTP_REFERER` - optional OpenRouter attribution header.
- `OPENROUTER_APP_TITLE` - optional OpenRouter attribution header.
- `OPENROUTER_MODEL_ROUTER_CATALOG` - path to the JSON model catalog.
- `OPENROUTER_MODEL_ROUTER_LEDGER` - path to the JSONL run ledger.

## Plug-In Pattern

Existing projects usually have one hardcoded model:

```python
model = os.environ.get("OPENROUTER_MODEL", "openrouter/auto")
```

Replace that with a router call:

```python
from openrouter_model_router import ModelRouter, TaskSpec

router = ModelRouter.from_file()
selection = router.select(TaskSpec(task_type="coding", preference="balanced"))
model = selection.model.id
```

Then pass `model` to the same OpenRouter-compatible client you already use.

## Production Notes

- Run `openrouter-model-router refresh` on a schedule, during deploy, or behind an admin action.
- Keep model allowlists/blocklists in project config when compliance or cost control matters.
- Use `max_cost_usd` for user-facing or batch workflows.
- Use `record-outcome` or `ModelCatalog.record_outcome()` after calls to let observed quality, success, and latency tune future routing.
- Commit a curated catalog for fully deterministic deployments, or refresh into an uncommitted runtime cache for constantly updated routing.
- Run `openrouter-model-router reconcile --catalog <path> --fail-on-drift` in CI or after a batch. Drift past the tolerance means the catalog is stale and every cost number downstream is wrong.
- Run `openrouter-model-router catalog-status` as its own gate before a costed batch. It is cheap, needs no ledger, and fails on a catalog nobody has refreshed.
- Never treat `estimated_cost_usd == 0.0` as free without checking `estimated_cost_is_known`. Prefer `cost_estimate()`, which returns `None`.
