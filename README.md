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
pricing block. Those models estimate `$0.00`, so `Selection.estimated_cost_is_known`
tells you whether that zero is a measurement or the absence of one, and the
ledger records `estimated_cost_usd: null` rather than `0.0` for them.

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
openrouter-model-router reconcile --fail-on-drift
```

```text
reconciliation: DRIFT
  runs                 3 (2 comparable)
  estimated            $0.012800
  reported             $0.032400
  drift                $+0.019600 (+153.1%)
  missing reported     1
  [FLAG] openai/gpt-4.1-mini: cost off by 153.1% (tolerance 10.0%) while token counts agree within 0.0% - the catalog price is stale, run refresh
```

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
| n/a | estimate is `$0.00`, provider charged | `no_catalog_price` |

Blaming the catalog for a token-estimate error would send someone to refresh a
catalog that is already correct.

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
- Run `openrouter-model-router reconcile --fail-on-drift` in CI or after a batch. Drift past the tolerance means the catalog is stale and every cost number downstream is wrong.
- Never treat `estimated_cost_usd == 0.0` as free without checking `estimated_cost_is_known`.
