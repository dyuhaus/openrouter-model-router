"""Lightweight OpenRouter model routing with cost instrumentation."""

from .catalog import ModelCatalog, default_catalog_path
from .ledger import RunLedger, RunRecord, default_ledger_path
from .openrouter import ChatResult, OpenRouterClient, OpenRouterError
from .reconcile import ReconciliationReport, format_report, reconcile
from .router import ModelRouter
from .transport import FakeTransport, HttpRequest, HttpResponse, HttpTransport, UrllibTransport
from .types import ModelInfo, Selection, TaskSpec
from .usage import TokenUsage, parse_usage

__all__ = [
    "ChatResult",
    "FakeTransport",
    "HttpRequest",
    "HttpResponse",
    "HttpTransport",
    "ModelCatalog",
    "ModelInfo",
    "ModelRouter",
    "OpenRouterClient",
    "OpenRouterError",
    "ReconciliationReport",
    "RunLedger",
    "RunRecord",
    "Selection",
    "TaskSpec",
    "TokenUsage",
    "UrllibTransport",
    "default_catalog_path",
    "default_ledger_path",
    "format_report",
    "parse_usage",
    "reconcile",
]
