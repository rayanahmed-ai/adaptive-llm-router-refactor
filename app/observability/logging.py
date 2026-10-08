import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra_fields = getattr(record, "telemetry", None)

        if extra_fields:
            payload.update(extra_fields)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def get_logger(name: str = "adaptive-llm-router") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    logger.addHandler(handler)
    logger.propagate = False

    return logger


logger = get_logger()


def log_routing_event(
    *,
    prompt: str,
    complexity: Optional[str],
    confidence: Optional[float],
    selected_model: Optional[str],
    decision_type: Optional[str],
    success: bool,
    latency_ms: Optional[float],
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    error: Optional[str] = None,
) -> None:

    telemetry = {
        "event_type": "llm_inference",
        "version": "V7.2",

        "prompt": prompt,

        "complexity": complexity,
        "confidence": confidence,

        "selected_model": selected_model,
        "decision_type": decision_type,

        "success": success,
        "latency_ms": latency_ms,

        "input_tokens": input_tokens,
        "output_tokens": output_tokens,

        "error": error,
    }

    logger.info(
        "LLM inference completed",
        extra={"telemetry": telemetry},
    )
def log_routing_decision(
    *,
    prompt: str,
    complexity: str,
    confidence: float,
    selected_model: str,
) -> None:

    telemetry = {
        "event_type": "routing_decision",
        "version": "V7.2",

        "prompt": prompt,
        "complexity": complexity,
        "confidence": confidence,
        "selected_model": selected_model,
    }

    logger.info(
        "Routing decision created",
        extra={"telemetry": telemetry},
    )