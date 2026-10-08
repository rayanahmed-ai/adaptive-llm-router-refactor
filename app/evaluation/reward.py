from dataclasses import dataclass
from math import exp
from typing import Optional


@dataclass
class RewardResult:

    reward: float

    success_score: float
    latency_score: float
    efficiency_score: float

    quality_score: Optional[float]

    latency_ms: float
    output_tokens: int


class RewardCalculator:

    def __init__(
        self,
        target_latency_ms: float = 2500.0,
        target_output_tokens: int = 300,

        success_weight: float = 0.50,
        latency_weight: float = 0.30,
        efficiency_weight: float = 0.20,

        quality_weight: float = 0.0,
    ):

        self.target_latency_ms = (
            target_latency_ms
        )

        self.target_output_tokens = (
            target_output_tokens
        )

        self.success_weight = (
            success_weight
        )

        self.latency_weight = (
            latency_weight
        )

        self.efficiency_weight = (
            efficiency_weight
        )

        self.quality_weight = (
            quality_weight
        )

    def calculate(
        self,
        *,
        success: bool,
        latency_ms: float,
        output_tokens: int = 0,
        quality_score: Optional[float] = None,
    ) -> RewardResult:

        # --------------------------------------------------
        # SUCCESS
        # --------------------------------------------------

        success_score = (
            1.0 if success else 0.0
        )

        # --------------------------------------------------
        # LATENCY
        # --------------------------------------------------

        if latency_ms <= 0:

            latency_score = 0.0

        else:

            latency_score = exp(
                -latency_ms
                / self.target_latency_ms
            )

        # --------------------------------------------------
        # OUTPUT EFFICIENCY
        # --------------------------------------------------

        if output_tokens <= 0:

            efficiency_score = 0.0

        else:

            efficiency_score = min(
                1.0,
                self.target_output_tokens
                / output_tokens,
            )

        # --------------------------------------------------
        # BASE RUNTIME REWARD
        # --------------------------------------------------

        runtime_reward = (
            self.success_weight
            * success_score

            + self.latency_weight
            * latency_score

            + self.efficiency_weight
            * efficiency_score
        )

        # --------------------------------------------------
        # OPTIONAL QUALITY
        # --------------------------------------------------

        if quality_score is not None:

            quality_score = max(
                0.0,
                min(
                    1.0,
                    float(quality_score)
                )
            )

            runtime_weight = (
                1.0 - self.quality_weight
            )

            reward = (
                runtime_weight
                * runtime_reward

                + self.quality_weight
                * quality_score
            )

        else:

            reward = runtime_reward

        # --------------------------------------------------
        # FAILED REQUESTS
        # --------------------------------------------------

        if not success:

            reward *= 0.25

        # --------------------------------------------------
        # SAFETY CLAMP
        # --------------------------------------------------

        reward = max(
            0.0,
            min(
                1.0,
                reward
            )
        )

        return RewardResult(
            reward=reward,

            success_score=success_score,
            latency_score=latency_score,
            efficiency_score=efficiency_score,

            quality_score=quality_score,

            latency_ms=latency_ms,
            output_tokens=output_tokens,
        )