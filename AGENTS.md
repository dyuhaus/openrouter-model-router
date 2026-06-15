# Agent Instructions

This repository is a lightweight, provider-neutral Python package for routing
OpenRouter model selection. Keep it small and dependency-free unless a new
dependency clearly pays for itself.

## Commands

- Run tests: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests`
- Live catalog smoke test: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m openrouter_model_router.cli research --catalog /tmp/openrouter-model-router-catalog.json --timeout 30`

## Rules

- Do not commit secrets, API keys, refreshed user catalogs, or `.env` files.
- Keep OpenRouter credentials in `OPENROUTER_API_KEY` or explicit caller config.
- Preserve standard-library-only runtime behavior unless the user explicitly wants a heavier integration.
- Add focused tests for routing, catalog parsing, CLI behavior, or client behavior when changing those areas.
