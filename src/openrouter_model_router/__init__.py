"""Lightweight OpenRouter model routing."""

from .catalog import ModelCatalog, default_catalog_path
from .openrouter import OpenRouterClient, OpenRouterError
from .router import ModelRouter
from .types import ModelInfo, Selection, TaskSpec

__all__ = [
    "ModelCatalog",
    "ModelInfo",
    "ModelRouter",
    "OpenRouterClient",
    "OpenRouterError",
    "Selection",
    "TaskSpec",
    "default_catalog_path",
]
