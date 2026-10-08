# ============================================================
# V7.7 — FINAL LIVE ROUTER RUNNER
# ============================================================

from __future__ import annotations

from typing import Any, Callable

from app.router.v7_7 import V77FinalAdaptiveRouter


class V77LiveRunner:

    VERSION = "V7.7"

    def __init__(
        self,
        bedrock_invoke: Callable[..., Any],
        epsilon: float = 0.10,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        quality_weight: float = 0.45,
        reliability_weight: float = 0.20,
        latency_weight: float = 0.20,
        cost_weight: float = 0.15,
        seed: int | None = None,
    ):

        if not callable(bedrock_invoke):
            raise TypeError(
                "bedrock_invoke must be callable"
            )

        self.bedrock_invoke = bedrock_invoke

        self.router = V77FinalAdaptiveRouter(
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            quality_weight=quality_weight,
            reliability_weight=reliability_weight,
            latency_weight=latency_weight,
            cost_weight=cost_weight,
            seed=seed,
        )

    # ========================================================
    # RUN ONE REQUEST
    # ========================================================

    def run(
        self,
        *,
        prompt: str,
        model_stats: list[dict[str, Any]],
        cost_data: dict[str, float] | None = None,
        **bedrock_kwargs: Any,
    ) -> dict[str, Any]:

        if not prompt:
            raise ValueError(
                "prompt cannot be empty"
            )

        # ----------------------------------------------------
        # V7.7 DECISION
        # ----------------------------------------------------

        decision = self.router.decide(
            model_stats=model_stats,
            cost_data=cost_data,
        )

        # ----------------------------------------------------
        # EXISTING BEDROCK LAYER
        # ----------------------------------------------------

        response = self.bedrock_invoke(
            model=decision.selected_model,
            prompt=prompt,
            **bedrock_kwargs,
        )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        return {
            "version": self.VERSION,
            "prompt": prompt,

            "selected_model":
                decision.selected_model,

            "decision_type":
                decision.decision_type,

            "epsilon":
                decision.epsilon,

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

            "routing_scores":
                decision.all_scores,

            "response":
                response,
        }