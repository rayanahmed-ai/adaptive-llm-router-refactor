"""
V7.4 — Adaptive Model Performance Statistics

V7.4 works on top of the existing V7.3 DynamoDB state.

Existing V7.3 state contains:
    reward_sum
    reward_count
    success_count
    failure_count
    total_latency_ms
    total_output_tokens
    reward

V7.4 derives:
    avg_reward
    success_rate
    failure_rate
    avg_latency_ms
    avg_output_tokens
    evidence_confidence
    evidence_adjusted_reward

Important:
    V7.4 does NOT create a new DynamoDB table.
    V7.4 does NOT call DescribeTable.
    V7.4 does NOT select a model.

V7.5 will handle exploration/exploitation.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


class ModelPerformanceStatistics:
    """
    Pure V7.4 statistics calculator.

    It operates on the attributes returned by the existing
    V7.3 UpdateItem call.
    """

    VERSION = "V7.4"

    # ----------------------------------------------------------
    # CALCULATE STATISTICS
    # ----------------------------------------------------------

    @classmethod
    def calculate(
        cls,
        attributes: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert existing V7.3 state into V7.4 statistics.

        No AWS call happens here.
        """

        reward_count = int(
            attributes.get(
                "reward_count",
                0,
            )
        )

        success_count = int(
            attributes.get(
                "success_count",
                0,
            )
        )

        failure_count = int(
            attributes.get(
                "failure_count",
                0,
            )
        )

        reward_sum = Decimal(
            str(
                attributes.get(
                    "reward_sum",
                    "0",
                )
            )
        )

        total_latency_ms = Decimal(
            str(
                attributes.get(
                    "total_latency_ms",
                    "0",
                )
            )
        )

        total_output_tokens = int(
            attributes.get(
                "total_output_tokens",
                0,
            )
        )

        # ------------------------------------------------------
        # AVERAGE REWARD
        # ------------------------------------------------------

        if reward_count > 0:

            average_reward = (
                reward_sum
                / Decimal(reward_count)
            )

        else:

            average_reward = Decimal("0")

        # ------------------------------------------------------
        # SUCCESS RATE
        # ------------------------------------------------------

        if reward_count > 0:

            success_rate = (
                Decimal(success_count)
                / Decimal(reward_count)
            )

        else:

            success_rate = Decimal("0")

        # ------------------------------------------------------
        # FAILURE RATE
        # ------------------------------------------------------

        if reward_count > 0:

            failure_rate = (
                Decimal(failure_count)
                / Decimal(reward_count)
            )

        else:

            failure_rate = Decimal("0")

        # ------------------------------------------------------
        # AVERAGE LATENCY
        # ------------------------------------------------------

        if reward_count > 0:

            average_latency_ms = (
                total_latency_ms
                / Decimal(reward_count)
            )

        else:

            average_latency_ms = Decimal("0")

        # ------------------------------------------------------
        # AVERAGE OUTPUT TOKENS
        # ------------------------------------------------------

        if reward_count > 0:

            average_output_tokens = (
                Decimal(total_output_tokens)
                / Decimal(reward_count)
            )

        else:

            average_output_tokens = Decimal("0")

        # ------------------------------------------------------
        # EVIDENCE CONFIDENCE
        #
        # This measures how much historical evidence exists.
        # It is NOT LLM response confidence.
        # ------------------------------------------------------

        evidence_confidence = (
            1.0
            -
            math.exp(
                -reward_count / 20.0
            )
        )

        evidence_confidence_decimal = Decimal(
            str(evidence_confidence)
        )

        # ------------------------------------------------------
        # EVIDENCE-ADJUSTED REWARD
        #
        # Prevents tiny sample sizes from dominating.
        # ------------------------------------------------------

        evidence_adjusted_reward = (
            average_reward
            *
            evidence_confidence_decimal
        )

        return {

            # Existing identity
            "state_key":
                attributes.get(
                    "state_key"
                ),

            "complexity":
                attributes.get(
                    "last_complexity"
                ),

            "model":
                attributes.get(
                    "last_model"
                ),

            # Historical counts
            "sample_count":
                reward_count,

            "success_count":
                success_count,

            "failure_count":
                failure_count,

            # Historical reward
            "reward_sum":
                reward_sum,

            "avg_reward":
                average_reward,

            # Reliability
            "success_rate":
                success_rate,

            "failure_rate":
                failure_rate,

            # Runtime
            "total_latency_ms":
                total_latency_ms,

            "avg_latency_ms":
                average_latency_ms,

            "total_output_tokens":
                total_output_tokens,

            "avg_output_tokens":
                average_output_tokens,

            # Adaptive evidence
            "evidence_confidence":
                evidence_confidence_decimal,

            "evidence_adjusted_reward":
                evidence_adjusted_reward,

            # Metadata
            "v7_4_version":
                cls.VERSION,

            "v7_4_updated_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

    # ----------------------------------------------------------
    # DYNAMODB VALUES
    # ----------------------------------------------------------

    @classmethod
    def dynamodb_values(
        cls,
        statistics: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert calculated V7.4 statistics into DynamoDB-safe
        ExpressionAttributeValues.
        """

        return {

            ":v74_avg_reward":
                Decimal(
                    str(
                        statistics[
                            "avg_reward"
                        ]
                    )
                ),

            ":v74_success_rate":
                Decimal(
                    str(
                        statistics[
                            "success_rate"
                        ]
                    )
                ),

            ":v74_failure_rate":
                Decimal(
                    str(
                        statistics[
                            "failure_rate"
                        ]
                    )
                ),

            ":v74_avg_latency":
                Decimal(
                    str(
                        statistics[
                            "avg_latency_ms"
                        ]
                    )
                ),

            ":v74_avg_tokens":
                Decimal(
                    str(
                        statistics[
                            "avg_output_tokens"
                        ]
                    )
                ),

            ":v74_evidence_confidence":
                Decimal(
                    str(
                        statistics[
                            "evidence_confidence"
                        ]
                    )
                ),

            ":v74_adjusted_reward":
                Decimal(
                    str(
                        statistics[
                            "evidence_adjusted_reward"
                        ]
                    )
                ),

            ":v74_version":
                statistics[
                    "v7_4_version"
                ],

            ":v74_updated_at":
                statistics[
                    "v7_4_updated_at"
                ],
        }