import time
from typing import Any, Dict, Optional

import boto3

from app.observability.logging import log_routing_event
from app.observability.metrics import metrics


class BedrockClient:

    def __init__(
        self,
        region_name: str = "eu-central-1",
    ):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
        )

    def invoke(
        self,
        *,
        model_id: str,
        prompt: str,
        complexity: Optional[str] = None,
        confidence: Optional[float] = None,
        decision_type: Optional[str] = None,
    ) -> Dict[str, Any]:

        start_time = time.perf_counter()

        try:

            response = self.client.converse(
                modelId=model_id,

                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt
                            }
                        ],
                    }
                ],
            )

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            output_message = (
                response
                .get("output", {})
                .get("message", {})
                .get("content", [])
            )

            generated_text = ""

            if output_message:

                generated_text = (
                    output_message[0]
                    .get("text", "")
                )

            usage = response.get("usage", {})

            input_tokens = usage.get(
                "inputTokens",
                0,
            )

            output_tokens = usage.get(
                "outputTokens",
                0,
            )

            metrics.record(
                model=model_id,
                complexity=complexity or "unknown",
                decision_type=decision_type or "unknown",
                latency_ms=latency_ms,
                success=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            log_routing_event(
                prompt=prompt,
                complexity=complexity,
                confidence=confidence,
                selected_model=model_id,
                decision_type=decision_type,
                success=True,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            return {
                "success": True,
                "response": generated_text,

                "model_id": model_id,

                "latency_ms": latency_ms,

                "input_tokens": input_tokens,
                "output_tokens": output_tokens,

                "error": None,
            }

        except Exception as exc:

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            metrics.record(
                model=model_id,
                complexity=complexity or "unknown",
                decision_type=decision_type or "unknown",
                latency_ms=latency_ms,
                success=False,
            )

            log_routing_event(
                prompt=prompt,
                complexity=complexity,
                confidence=confidence,
                selected_model=model_id,
                decision_type=decision_type,
                success=False,
                latency_ms=latency_ms,
                error=str(exc),
            )

            return {
                "success": False,
                "response": None,

                "model_id": model_id,

                "latency_ms": latency_ms,

                "input_tokens": 0,
                "output_tokens": 0,

                "error": str(exc),
            }