from __future__ import annotations

import time
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError


class V7BedrockClient:
    """
    V7.1 Bedrock inference adapter.

    This file is intentionally separate from the existing
    app/models/bedrock.py so V1-V6 are not broken.

    V7.1:
        logical model
             ↓
        Bedrock model ID
             ↓
        Converse API
             ↓
        real response
    """

    def __init__(
        self,
        region: str = "eu-north-1",
    ):

        self.region = region

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
        )

        self.model_ids = {
            "small_model":
                "eu.amazon.nova-micro-v1:0",

            "medium_model":
                "eu.amazon.nova-lite-v1:0",

            "large_model":
                "eu.amazon.nova-pro-v1:0",
        }


    def get_model_id(
        self,
        logical_model: str,
    ) -> str:

        if logical_model not in self.model_ids:

            raise ValueError(
                f"Unknown logical model: "
                f"{logical_model}"
            )

        return self.model_ids[
            logical_model
        ]


    def invoke(
        self,
        logical_model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:

        if not prompt.strip():
  
            raise ValueError(
                "Prompt cannot be empty."
            )

        model_id = self.get_model_id(
            logical_model
        )

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

                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                },
            )

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000


            # ------------------------------------------------
            # RESPONSE TEXT
            # ------------------------------------------------

            content = (
                response
                .get("output", {})
                .get("message", {})
                .get("content", [])
            )

            response_text = ""

            for block in content:

                if (
                    isinstance(block, dict)
                    and "text" in block
                ):

                    response_text += (
                        block["text"]
                    )

            if not response_text.strip():
                stop_reason = response.get(
                    "stopReason",
                    "unknown",
                )

                raise RuntimeError(
                    "Bedrock returned no text content. "
                    f"model_id={model_id}, "
                    f"content_blocks={len(content)}, "
                    f"stop_reason={stop_reason}"
                )


            # ------------------------------------------------
            # USAGE
            # ------------------------------------------------

            usage = response.get(
                "usage",
                {},
            )

            metrics = response.get(
                "metrics",
                {},
            )


            return {

                "success": True,

                "response":
                    response_text,

                "model_id":
                    model_id,

                "latency_ms":
                    float(
                        metrics.get(
                            "latencyMs",
                            elapsed_ms,
                        )
                    ),

                "input_tokens":
                    int(
                        usage.get(
                            "inputTokens",
                            0,
                        )
                    ),

                "output_tokens":
                    int(
                        usage.get(
                            "outputTokens",
                            0,
                        )
                    ),

                "total_tokens":
                    int(
                        usage.get(
                            "totalTokens",
                            0,
                        )
                    ),

                "error": None,
            }


        except ClientError as error:

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            error_data = (
                error.response.get(
                    "Error",
                    {},
                )
            )

            return {

                "success": False,

                "response": "",

                "model_id":
                    model_id,

                "latency_ms":
                    elapsed_ms,

                "input_tokens": 0,

                "output_tokens": 0,

                "total_tokens": 0,

                "error":
                    error_data.get(
                        "Message",
                        str(error),
                    ),
            }


        except Exception as error:

            elapsed_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            return {

                "success": False,

                "response": "",

                "model_id":
                    model_id,

                "latency_ms":
                    elapsed_ms,

                "input_tokens": 0,

                "output_tokens": 0,

                "total_tokens": 0,

                "error":
                    str(error),
            }