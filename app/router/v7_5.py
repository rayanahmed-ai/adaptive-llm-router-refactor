# ============================================================
# V7.5 — EXPLORATION / EXPLOITATION ROUTER
# ============================================================

"""
V7.5 introduces adaptive exploration vs exploitation.

V7.4 provides historical evidence:

    v74_evidence_adjusted_reward
    v74_evidence_confidence
    v74_success_rate
    v74_avg_latency_ms

V7.5 decides:

    EXPLORE
        ->
        try a candidate that may have less evidence

    EXPLOIT
        ->
        choose the best-supported candidate

Important:
    V7.5 does not yet optimize cost.
    V7.6 will add cost/quality/latency trade-offs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class V75Decision:
    """Result of one V7.5 routing decision."""

    selected_model: str
    decision_type: str
    epsilon: float
    candidates: list[str]
    scores: dict[str, float]


class V75ExplorationPolicy:

    VERSION = "V7.5"

    def __init__(
        self,
        epsilon: float = 0.10,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        """
        Parameters
        ----------
        epsilon:
            Probability of exploration.

        epsilon_min:
            Minimum exploration probability.

        epsilon_decay:
            Multiplicative decay after each decision.

        seed:
            Optional random seed for deterministic tests.
        """

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be between 0 and 1"
            )

        if not 0.0 <= epsilon_min <= 1.0:
            raise ValueError(
                "epsilon_min must be between 0 and 1"
            )

        if epsilon_min > epsilon:
            raise ValueError(
                "epsilon_min cannot exceed epsilon"
            )

        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError(
                "epsilon_decay must be in (0, 1]"
            )

        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.random = random.Random(seed)

        self.decision_count = 0

    # ========================================================
    # NORMALIZE CANDIDATES
    # ========================================================

    @staticmethod
    def _normalize_candidates(
        model_stats: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        candidates = []

        for item in model_stats:

            model = item.get("model")

            if not model:
                continue

            candidates.append(item)

        return candidates

    # ========================================================
    # EXTRACT V7.4 SCORE
    # ========================================================

    @staticmethod
    def _score(
        item: dict[str, Any],
    ) -> float:

        value = item.get(
            "evidence_adjusted_reward",
            0.0,
        )

        return float(value)

    # ========================================================
    # EXPLOIT
    # ========================================================

    def _exploit(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:

        return max(
            candidates,
            key=self._score,
        )

    # ========================================================
    # EXPLORE
    # ========================================================

    def _explore(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:

        """
        Exploration intentionally favors candidates with
        less evidence.

        We first choose the candidates with the minimum
        sample count, then randomly select among them.
        """

        minimum_samples = min(
            int(
                item.get(
                    "sample_count",
                    0,
                )
            )
            for item in candidates
        )

        least_observed = [
            item
            for item in candidates
            if int(
                item.get(
                    "sample_count",
                    0,
                )
            )
            == minimum_samples
        ]

        return self.random.choice(
            least_observed
        )

    # ========================================================
    # DECIDE
    # ========================================================

    def decide(
        self,
        model_stats: list[dict[str, Any]],
    ) -> V75Decision:

        candidates = (
            self._normalize_candidates(
                model_stats
            )
        )

        if not candidates:
            raise ValueError(
                "No valid model candidates provided"
            )

        self.decision_count += 1

        candidate_names = [
            item["model"]
            for item in candidates
        ]

        scores = {
            item["model"]:
                self._score(item)
            for item in candidates
        }

        # ----------------------------------------------------
        # EXPLORATION
        # ----------------------------------------------------

        explore = (
            self.random.random()
            < self.epsilon
        )

        if explore:

            selected = self._explore(
                candidates
            )

            decision_type = (
                "EXPLORATION"
            )

        # ----------------------------------------------------
        # EXPLOITATION
        # ----------------------------------------------------

        else:

            selected = self._exploit(
                candidates
            )

            decision_type = (
                "EXPLOITATION"
            )

        current_epsilon = (
            self.epsilon
        )

        # ----------------------------------------------------
        # EPSILON DECAY
        # ----------------------------------------------------

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon
            * self.epsilon_decay,
        )

        return V75Decision(
            selected_model=
                selected["model"],

            decision_type=
                decision_type,

            epsilon=
                current_epsilon,

            candidates=
                candidate_names,

            scores=
                scores,
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        epsilon: float | None = None,
    ) -> None:

        if epsilon is not None:

            if not 0.0 <= epsilon <= 1.0:
                raise ValueError(
                    "epsilon must be between 0 and 1"
                )

            self.epsilon = (
                float(epsilon)
            )

        self.decision_count = 0

    # ========================================================
    # STATE
    # ========================================================

    def get_state(self) -> dict[str, Any]:

        return {

            "version":
                self.VERSION,

            "epsilon":
                self.epsilon,

            "epsilon_min":
                self.epsilon_min,

            "epsilon_decay":
                self.epsilon_decay,

            "decision_count":
                self.decision_count,
        }