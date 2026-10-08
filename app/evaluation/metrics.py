import json
import os
import time
from datetime import datetime, timezone


class TelemetryLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = os.path.join(
            self.log_dir,
            "v7_2_telemetry.jsonl"
        )

    def log_request(
        self,
        prompt,
        complexity,
        confidence,
        selected_model,
        decision_type,
        latency_ms,
        success,
        response=None,
        error=None,
    ):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),

            "version": "V7.2",

            "prompt": prompt,

            "complexity": complexity,

            "confidence": round(float(confidence), 4)
            if confidence is not None else None,

            "selected_model": selected_model,

            "decision_type": decision_type,

            "latency_ms": round(float(latency_ms), 2),

            "success": bool(success),

            "response_length": len(response)
            if response else 0,

            "error": str(error) if error else None,
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        return event

    def log_error(
        self,
        prompt,
        complexity,
        selected_model,
        error,
        latency_ms=None,
    ):
        return self.log_request(
            prompt=prompt,
            complexity=complexity,
            confidence=None,
            selected_model=selected_model,
            decision_type="ERROR",
            latency_ms=latency_ms or 0,
            success=False,
            response=None,
            error=error,
        )