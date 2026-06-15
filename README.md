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
