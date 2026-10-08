# ============================================================
# V7.6 — COST + QUALITY + LATENCY ROUTING
# ============================================================

"""
V7.6 adds multi-objective model scoring.

Inputs from V7.4:
    - evidence_adjusted_reward
    - avg_reward
    - success_rate
    - avg_latency_ms
    - sample_count

Optional:
    - cost_usd

V7.6 objective:

    quality
    + reliability
    - latency
    - cost

Important:
    - No fake cost is generated.
    - If cost data is unavailable, cost is excluded and
      the remaining weights are renormalized.
    - V7.6 does not modify DynamoDB.
    - V7.6 does not modify V7.3/V7.4 state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class V76Decision:
    """Result of a V7.6 multi-objective decision."""

    selected_model: str
    scores: dict[str, float]
    quality_scores: dict[str, float]
    reliability_scores: dict[str, float]
    latency_scores: dict[str, float]
    cost_scores: dict[str, float]
    cost_available: bool


class V76MultiObjectiveRouter:

    VERSION = "V7.6"

    def __init__(
        self,
        quality_weight: float = 0.45,
        reliability_weight: float = 0.20,
        latency_weight: float = 0.20,
        cost_weight: float = 0.15,
    ) -> None:

        weights = {
            "quality": quality_weight,
            "reliability": reliability_weight,
            "latency": latency_weight,
            "cost": cost_weight,
        }

        for name, value in weights.items():

            if value < 0:
                raise ValueError(
                    f"{name}_weight cannot be negative"
                )

        if sum(weights.values()) <= 0:
            raise ValueError(
                "At least one weight must be greater than zero"
            )

        self.quality_weight = float(
            quality_weight
        )

        self.reliability_weight = float(
            reliability_weight
        )

        self.latency_weight = float(
            latency_weight
        )

        self.cost_weight = float(
            cost_weight
        )

    # ========================================================
    # MIN-MAX NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_high_is_good(
        values: dict[str, float],
    ) -> dict[str, float]:

        if not values:
            return {}

        minimum = min(
            values.values()
        )

        maximum = max(
            values.values()
        )

        if maximum == minimum:

            return {
                key: 1.0
                for key in values
            }

        return {
            key: (
                (value - minimum)
                /
                (maximum - minimum)
            )
            for key, value in values.items()
        }

    @staticmethod
    def _normalize_low_is_good(
        values: dict[str, float],
    ) -> dict[str, float]:

        if not values:
            return {}

        high_is_good = (
            V76MultiObjectiveRouter
            ._normalize_high_is_good(
                values
            )
        )

        return {
            key: 1.0 - value
            for key, value in high_is_good.items()
        }

    # ========================================================
    # EXTRACT QUALITY
    # ========================================================

    @staticmethod
    def _quality_value(
        item: dict[str, Any],
    ) -> float:

        # Prefer V7.4 evidence-adjusted reward.
        if (
            item.get(
                "evidence_adjusted_reward"
            )
            is not None
        ):
            return float(
                item[
                    "evidence_adjusted_reward"
                ]
            )

        return float(
            item.get(
                "avg_reward",
                0.0,
            )
        )

    # ========================================================
    # EXTRACT RELIABILITY
    # ========================================================

    @staticmethod
    def _reliability_value(
        item: dict[str, Any],
    ) -> float:

        return float(
            item.get(
                "success_rate",
                0.0,
            )
        )

    # ========================================================
    # EXTRACT LATENCY
    # ========================================================

    @staticmethod
    def _latency_value(
        item: dict[str, Any],
    ) -> float:

        return float(
            item.get(
                "avg_latency_ms",
                0.0,
            )
        )

    # ========================================================
    # DETECT COST
    # ========================================================

    @staticmethod
    def _extract_costs(
        model_stats: list[dict[str, Any]],
        cost_data: dict[str, float] | None,
    ) -> dict[str, float]:

        if cost_data is not None:

            return {
                model: float(cost)
                for model, cost in cost_data.items()
            }

        # Accept cost if it already exists in model_stats.
        extracted = {}

        for item in model_stats:

            model = item.get(
                "model"
            )

            cost = item.get(
                "avg_cost_usd"
            )

            if (
                model
                and cost is not None
            ):
                extracted[model] = float(
                    cost
                )

        return extracted

    # ========================================================
    # WEIGHT RENORMALIZATION
    # ========================================================

    def _effective_weights(
        self,
        cost_available: bool,
    ) -> dict[str, float]:

        weights = {

            "quality":
                self.quality_weight,

            "reliability":
                self.reliability_weight,

            "latency":
                self.latency_weight,

            "cost":
                self.cost_weight
                if cost_available
                else 0.0,
        }

        total = sum(
            weights.values()
        )

        if total <= 0:
            raise ValueError(
                "Effective V7.6 weights sum to zero"
            )

        return {
            key: value / total
            for key, value in weights.items()
        }

    # ========================================================
    # SCORE MODELS
    # ========================================================

    def score_models(
        self,
        model_stats: list[dict[str, Any]],
        cost_data: dict[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:

        if not model_stats:
            raise ValueError(
                "model_stats cannot be empty"
            )

        # ----------------------------------------------
        # QUALITY
        # ----------------------------------------------

        quality_raw = {
            item["model"]:
                self._quality_value(item)
            for item in model_stats
            if item.get("model")
        }

        quality_scores = (
            self._normalize_high_is_good(
                quality_raw
            )
        )

        # ----------------------------------------------
        # RELIABILITY
        # ----------------------------------------------

        reliability_raw = {
            item["model"]:
                self._reliability_value(item)
            for item in model_stats
            if item.get("model")
        }

        reliability_scores = (
            self._normalize_high_is_good(
                reliability_raw
            )
        )

        # ----------------------------------------------
        # LATENCY
        # LOWER = BETTER
        # ----------------------------------------------

        latency_raw = {
            item["model"]:
                self._latency_value(item)
            for item in model_stats
            if item.get("model")
        }

        latency_scores = (
            self._normalize_low_is_good(
                latency_raw
            )
        )

        # ----------------------------------------------
        # COST
        # LOWER = BETTER
        # ----------------------------------------------

        costs = self._extract_costs(
            model_stats,
            cost_data,
        )

        cost_available = (
            len(costs)
            == len(model_stats)
            and len(costs) > 0
        )

        if cost_available:

            cost_scores = (
                self._normalize_low_is_good(
                    costs
                )
            )

        else:

            cost_scores = {
                item["model"]: 0.0
                for item in model_stats
                if item.get("model")
            }

        # ----------------------------------------------
        # EFFECTIVE WEIGHTS
        # ----------------------------------------------

        weights = self._effective_weights(
            cost_available
        )

        # ----------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------

        scores = {}

        for item in model_stats:

            model = item.get(
                "model"
            )

            if not model:
                continue

            score = (

                weights["quality"]
                * quality_scores.get(
                    model,
                    0.0
                )

                +

                weights["reliability"]
                * reliability_scores.get(
                    model,
                    0.0
                )

                +

                weights["latency"]
                * latency_scores.get(
                    model,
                    0.0
                )

                +

                weights["cost"]
                * cost_scores.get(
                    model,
                    0.0
                )
            )

            scores[model] = float(
                score
            )

        return {
            model: {
                "final_score":
                    scores[model],

                "quality_score":
                    quality_scores.get(
                        model,
                        0.0
                    ),

                "reliability_score":
                    reliability_scores.get(
                        model,
                        0.0
                    ),

                "latency_score":
                    latency_scores.get(
                        model,
                        0.0
                    ),

                "cost_score":
                    cost_scores.get(
                        model,
                        0.0
                    ),
            }
            for model in scores
        }

    # ========================================================
    # DECIDE
    # ========================================================

    def decide(
        self,
        model_stats: list[dict[str, Any]],
        cost_data: dict[str, float] | None = None,
    ) -> V76Decision:

        scored = self.score_models(
            model_stats,
            cost_data=cost_data,
        )

        if not scored:
            raise ValueError(
                "No models could be scored"
            )

        selected_model = max(
            scored,
            key=lambda model:
                scored[model][
                    "final_score"
                ],
        )

        return V76Decision(

            selected_model=
                selected_model,

            scores={
                model:
                    values["final_score"]
                for model, values
                in scored.items()
            },

            quality_scores={
                model:
                    values["quality_score"]
                for model, values
                in scored.items()
            },

            reliability_scores={
                model:
                    values[
                        "reliability_score"
                    ]
                for model, values
                in scored.items()
            },

            latency_scores={
                model:
                    values[
                        "latency_score"
                    ]
                for model, values
                in scored.items()
            },

            cost_scores={
                model:
                    values["cost_score"]
                for model, values
                in scored.items()
            },

            cost_available=
                any(
                    values["cost_score"] > 0
                    for values in scored.values()
                ),
        )