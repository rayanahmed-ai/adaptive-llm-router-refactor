# ============================================================
# V7.7 — FINAL ADAPTIVE ROUTER
# ============================================================

"""
V7.7 integrates:

V7.4
    Historical model performance

V7.5
    Exploration / exploitation

V7.6
    Quality + reliability + latency + optional cost

V7.7 is the final decision/orchestration layer.

It does NOT:
    - call Bedrock
    - write to DynamoDB
    - create AWS resources

It returns the final model decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.router.v7_5 import (
    V75ExplorationPolicy,
)

from app.router.v7_6 import (
    V76MultiObjectiveRouter,
)


# ============================================================
# DECISION RESULT
# ============================================================

@dataclass
class V77Decision:

    selected_model: str

    decision_type: str

    epsilon: float

    v75_candidate: str

    v76_best_model: str

    final_score: float

    all_scores: dict[str, float]

    quality_scores: dict[str, float]

    reliability_scores: dict[str, float]

    latency_scores: dict[str, float]

    cost_scores: dict[str, float]

    cost_available: bool

    candidate_models: list[str]

    decision_version: str = "V7.7"


# ============================================================
# FINAL ADAPTIVE ROUTER
# ============================================================

class V77FinalAdaptiveRouter:

    VERSION = "V7.7"

    def __init__(
        self,
        epsilon: float = 0.10,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,

        quality_weight: float = 0.45,
        reliability_weight: float = 0.20,
        latency_weight: float = 0.20,
        cost_weight: float = 0.15,

        seed: int | None = None,
    ):

        # ----------------------------------------------------
        # V7.5
        # ----------------------------------------------------

        self.v75 = V75ExplorationPolicy(

            epsilon=epsilon,

            epsilon_min=epsilon_min,

            epsilon_decay=epsilon_decay,

            seed=seed,
        )

        # ----------------------------------------------------
        # V7.6
        # ----------------------------------------------------

        self.v76 = V76MultiObjectiveRouter(

            quality_weight=
                quality_weight,

            reliability_weight=
                reliability_weight,

            latency_weight=
                latency_weight,

            cost_weight=
                cost_weight,
        )

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    @staticmethod
    def _validate_model_stats(
        model_stats: list[dict[str, Any]],
    ) -> None:

        if not model_stats:
            raise ValueError(
                "model_stats cannot be empty"
            )

        models = []

        for item in model_stats:

            model = item.get("model")

            if not model:
                raise ValueError(
                    "Every candidate must contain 'model'"
                )

            models.append(model)

        if len(models) != len(set(models)):
            raise ValueError(
                "Duplicate model names found"
            )

    # ========================================================
    # LEAST OBSERVED MODELS
    # ========================================================

    @staticmethod
    def _least_observed_models(
        model_stats: list[dict[str, Any]],
    ) -> list[str]:

        minimum_samples = min(

            int(
                item.get(
                    "sample_count",
                    0,
                )
            )

            for item in model_stats
        )

        return [

            item["model"]

            for item in model_stats

            if int(
                item.get(
                    "sample_count",
                    0,
                )
            )
            == minimum_samples
        ]

    # ========================================================
    # FINAL DECISION
    # ========================================================

    def decide(
        self,
        model_stats: list[dict[str, Any]],
        cost_data: dict[str, float] | None = None,
    ) -> V77Decision:

        self._validate_model_stats(
            model_stats
        )

        # ----------------------------------------------------
        # V7.5
        # ----------------------------------------------------

        v75_decision = self.v75.decide(
            model_stats
        )

        # ----------------------------------------------------
        # V7.6
        # ----------------------------------------------------

        v76_decision = self.v76.decide(

            model_stats,

            cost_data=cost_data,
        )

        v76_best_model = (
            v76_decision.selected_model
        )

        # ----------------------------------------------------
        # FINAL INTEGRATION
        # ----------------------------------------------------

        if (
            v75_decision.decision_type
            == "EXPLOITATION"
        ):

            # Exploitation:
            # choose the best multi-objective model.

            final_model = (
                v76_best_model
            )

        else:

            # Exploration:
            # only explore the least-observed models.

            exploration_models = set(
                self._least_observed_models(
                    model_stats
                )
            )

            exploration_scores = {

                model: score

                for model, score
                in v76_decision.scores.items()

                if model in exploration_models
            }

            # Safety fallback.

            if not exploration_scores:

                final_model = (
                    v75_decision.selected_model
                )

            else:

                final_model = max(
                    exploration_scores,
                    key=exploration_scores.get,
                )

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = float(
            v76_decision.scores[
                final_model
            ]
        )

        # ----------------------------------------------------
        # RETURN
        # ----------------------------------------------------

        return V77Decision(

            selected_model=
                final_model,

            decision_type=
                v75_decision.decision_type,

            epsilon=
                v75_decision.epsilon,

            v75_candidate=
                v75_decision.selected_model,

            v76_best_model=
                v76_best_model,

            final_score=
                final_score,

            all_scores=
                v76_decision.scores,

            quality_scores=
                v76_decision.quality_scores,

            reliability_scores=
                v76_decision.reliability_scores,

            latency_scores=
                v76_decision.latency_scores,

            cost_scores=
                v76_decision.cost_scores,

            cost_available=
                v76_decision.cost_available,

            candidate_models=[
                item["model"]
                for item in model_stats
            ],
        )

    # ========================================================
    # STATE
    # ========================================================

    def get_state(self) -> dict[str, Any]:

        return {

            "version":
                self.VERSION,

            "v75":
                self.v75.get_state(),

            "v76":
                {
                    "quality_weight":
                        self.v76.quality_weight,

                    "reliability_weight":
                        self.v76.reliability_weight,

                    "latency_weight":
                        self.v76.latency_weight,

                    "cost_weight":
                        self.v76.cost_weight,
                },
        }

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        epsilon: float | None = None,
    ) -> None:

        self.v75.reset(
            epsilon=epsilon
        )