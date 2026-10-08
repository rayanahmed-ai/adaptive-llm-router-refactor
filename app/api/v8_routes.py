from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config.settings import (
    Settings,
    get_enabled_models,
)
from app.models.v7_bedrock import (
    V7BedrockClient,
)
from app.router.v7_7 import (
    V77FinalAdaptiveRouter,
)


router = APIRouter()


# ============================================================
# MODEL STATISTICS
# ============================================================

DEFAULT_STATS = {
    "small_model": {
        "sample_count": 50,
        "avg_reward": 0.72,
        "success_rate": 0.94,
        "avg_latency_ms": 1100.0,
        "avg_output_tokens": 250,
        "evidence_confidence": 0.918,
        "evidence_adjusted_reward": 0.661,
    },

    "medium_model": {
        "sample_count": 35,
        "avg_reward": 0.81,
        "success_rate": 0.97,
        "avg_latency_ms": 1800.0,
        "avg_output_tokens": 320,
        "evidence_confidence": 0.826,
        "evidence_adjusted_reward": 0.670,
    },

    "large_model": {
        "sample_count": 5,
        "avg_reward": 0.94,
        "success_rate": 1.00,
        "avg_latency_ms": 3000.0,
        "avg_output_tokens": 500,
        "evidence_confidence": 0.221,
        "evidence_adjusted_reward": 0.208,
    },
}


# ============================================================
# REQUEST / RESPONSE
# ============================================================

class GenerateRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=1,
    )

    max_tokens: int = Field(
        default=512,
        ge=1,
        le=4096,
    )

    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
    )


class GenerateResponse(BaseModel):

    success: bool

    selected_model: str

    decision_type: str

    v75_candidate: str

    v76_best_model: str

    final_score: float

    latency_ms: float

    response: Any

    model_id: str | None = None


# ============================================================
# SERVICES
# ============================================================

bedrock = V7BedrockClient(
    region=Settings.AWS_REGION
)

adaptive_router = V77FinalAdaptiveRouter(
    epsilon=0.05,
    epsilon_min=0.02,
    epsilon_decay=0.995,
    quality_weight=0.45,
    reliability_weight=0.20,
    latency_weight=0.20,
    cost_weight=0.15,
    seed=42,
)


def get_model_stats() -> list[dict[str, Any]]:

    enabled_models = get_enabled_models()

    stats = []

    for logical_model in enabled_models:

        if logical_model in DEFAULT_STATS:

            row = dict(
                DEFAULT_STATS[
                    logical_model
                ]
            )

            row["model"] = (
                logical_model
            )

            stats.append(row)

    if not stats:

        raise RuntimeError(
            "No enabled models have "
            "routing statistics."
        )

    return stats


# ============================================================
# HEALTH
# ============================================================

@router.get("/health")
def health():

    return {
        "status": "ok",
        "service":
            "adaptive-llm-routing",
        "version": "V8.3",
        "routing_version": "V7.7",
        "aws_region":
            Settings.AWS_REGION,
    }


# ============================================================
# ROUTE ONLY
# ============================================================

@router.post("/route")
def route_request(
    request: GenerateRequest,
):

    try:

        stats = get_model_stats()

        decision = adaptive_router.decide(
            model_stats=stats,
            cost_data=None,
        )

        return {
            "version": "V8.3",
            "selected_model":
                decision.selected_model,
            "decision_type":
                decision.decision_type,
            "v75_candidate":
                decision.v75_candidate,
            "v76_best_model":
                decision.v76_best_model,
            "final_score":
                decision.final_score,
            "candidate_models":
                decision.candidate_models,
            "cost_available":
                decision.cost_available,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# GENERATE
# ============================================================

@router.post(
    "/generate",
    response_model=GenerateResponse,
)
def generate(
    request: GenerateRequest,
):

    start = time.perf_counter()

    try:

        stats = get_model_stats()

        decision = adaptive_router.decide(
            model_stats=stats,
            cost_data=None,
        )

        result = bedrock.invoke(
            logical_model=
                decision.selected_model,

            prompt=request.prompt,

            max_tokens=
                request.max_tokens,

            temperature=
                request.temperature,
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        return GenerateResponse(
            success=bool(
                result.get(
                    "success",
                    False,
                )
            ),

            selected_model=
                decision.selected_model,

            decision_type=
                decision.decision_type,

            v75_candidate=
                decision.v75_candidate,

            v76_best_model=
                decision.v76_best_model,

            final_score=float(
                decision.final_score
            ),

            latency_ms=latency_ms,

            response=result.get(
                "response"
            ),

            model_id=result.get(
                "model_id"
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )