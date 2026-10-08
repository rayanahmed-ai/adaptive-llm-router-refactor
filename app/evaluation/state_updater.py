# import os
# from datetime import datetime, timezone
# from decimal import Decimal
# from typing import Dict, Any

# import boto3


# class DynamoDBRewardUpdater:

#     def __init__(
#         self,
#         table_name: str | None = None,
#         region_name: str = "eu-north-1",
#     ):
#         self.table_name = (
#             table_name
#             or os.getenv(
#                 "V6_DYNAMODB_TABLE",
#                 "adaptive-llm-router-v6-state-753176172779",
#             )
#         )

#         self.dynamodb = boto3.resource(
#             "dynamodb",
#             region_name=region_name,
#         )

#         self.table = self.dynamodb.Table(
#             self.table_name
#         )

#     def update_model(
#         self,
#         *,
#         complexity: str,
#         model: str,
#         reward: float,
#         success: bool,
#         latency_ms: float,
#         output_tokens: int,
#     ) -> Dict[str, Any]:

#         # --------------------------------------------------
#         # V6 STATE KEY
#         # STATE#<complexity>#<model>
#         # --------------------------------------------------

#         state_key = (
#             f"STATE#{complexity}#{model}"
#         )

#         # --------------------------------------------------
#         # FIRST UPDATE
#         #
#         # Only UpdateItem is used.
#         # No GetItem permission is required.
#         # --------------------------------------------------

#         first_response = self.table.update_item(

#             Key={
#                 "state_key": state_key
#             },

#             UpdateExpression="""
#                 SET
#                     reward_sum =
#                         if_not_exists(
#                             reward_sum,
#                             :zero_decimal
#                         ) + :reward,

#                     reward_count =
#                         if_not_exists(
#                             reward_count,
#                             :zero
#                         ) + :one,

#                     success_count =
#                         if_not_exists(
#                             success_count,
#                             :zero
#                         ) + :success,

#                     failure_count =
#                         if_not_exists(
#                             failure_count,
#                             :zero
#                         ) + :failure,

#                     total_latency_ms =
#                         if_not_exists(
#                             total_latency_ms,
#                             :zero_decimal
#                         ) + :latency,

#                     total_output_tokens =
#                         if_not_exists(
#                             total_output_tokens,
#                             :zero
#                         ) + :tokens,

#                     updated_at = :timestamp,

#                     last_reward = :last_reward,

#                     last_success = :last_success,

#                     last_latency_ms = :last_latency,

#                     last_output_tokens = :last_tokens,

#                     last_model = :model,

#                     last_complexity = :complexity,

#                     last_update_version = :version
#             """,

#             ExpressionAttributeValues={

#                 ":zero_decimal":
#                     Decimal("0"),

#                 ":zero":
#                     0,

#                 ":one":
#                     1,

#                 ":reward":
#                     Decimal(
#                         str(reward)
#                     ),

#                 ":success":
#                     1 if success else 0,

#                 ":failure":
#                     0 if success else 1,

#                 ":latency":
#                     Decimal(
#                         str(latency_ms)
#                     ),

#                 ":tokens":
#                     int(output_tokens),

#                 ":timestamp":
#                     datetime.now(
#                         timezone.utc
#                     ).isoformat(),

#                 ":last_reward":
#                     Decimal(
#                         str(reward)
#                     ),

#                 ":last_success":
#                     success,

#                 ":last_latency":
#                     Decimal(
#                         str(latency_ms)
#                     ),

#                 ":last_tokens":
#                     int(output_tokens),

#                 ":model":
#                     model,

#                 ":complexity":
#                     complexity,

#                 ":version":
#                     "V7.3",
#             },

#             ReturnValues="ALL_NEW",
#         )

#         # --------------------------------------------------
#         # GET UPDATED VALUES FROM UPDATE RESPONSE
#         # --------------------------------------------------

#         attributes = first_response.get(
#             "Attributes",
#             {}
#         )

#         reward_sum = Decimal(
#             str(
#                 attributes.get(
#                     "reward_sum",
#                     "0"
#                 )
#             )
#         )

#         reward_count = int(
#             attributes.get(
#                 "reward_count",
#                 1
#             )
#         )

#         # --------------------------------------------------
#         # CALCULATE RUNNING AVERAGE
#         # --------------------------------------------------

#         average_reward = (
#             reward_sum /
#             Decimal(reward_count)
#         )

#         # --------------------------------------------------
#         # SECOND UPDATE
#         #
#         # Still only UpdateItem.
#         # No GetItem permission required.
#         # --------------------------------------------------

#         second_response = self.table.update_item(

#             Key={
#                 "state_key": state_key
#             },

#             UpdateExpression="""
#                 SET
#                     reward = :average_reward
#             """,

#             ExpressionAttributeValues={
#                 ":average_reward":
#                     average_reward
#             },

#             ReturnValues="ALL_NEW",
#         )

#         return second_response.get(
#             "Attributes",
#             attributes
#         )
# ============================================================
# V7.3 + V7.4 — DYNAMODB STATE UPDATER
# ============================================================

import os
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any

import boto3

from app.evaluation.model_stats import (
    ModelPerformanceStatistics
)


class DynamoDBRewardUpdater:

    def __init__(
        self,
        table_name: str | None = None,
        region_name: str = "eu-north-1",
    ):

        # --------------------------------------------------
        # EXISTING V7.3 TABLE
        # --------------------------------------------------

        self.table_name = (
            table_name
            or os.getenv(
                "V6_DYNAMODB_TABLE",
                "adaptive-llm-router-v6-state-753176172779",
            )
        )

        self.dynamodb = boto3.resource(
            "dynamodb",
            region_name=region_name,
        )

        self.table = self.dynamodb.Table(
            self.table_name
        )

    # ============================================================
    # UPDATE MODEL
    # ============================================================

    def update_model(
        self,
        *,
        complexity: str,
        model: str,
        reward: float,
        success: bool,
        latency_ms: float,
        output_tokens: int,
    ) -> Dict[str, Any]:

        # --------------------------------------------------
        # STATE KEY
        #
        # STATE#<complexity>#<model>
        # --------------------------------------------------

        state_key = (
            f"STATE#{complexity}#{model}"
        )

        # --------------------------------------------------
        # TIMESTAMP
        # --------------------------------------------------

        timestamp = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        # --------------------------------------------------
        # V7.3 — PRIMARY AGGREGATE UPDATE
        #
        # IMPORTANT:
        # Only UpdateItem is used.
        # No GetItem permission required.
        # --------------------------------------------------

        first_response = (
            self.table.update_item(

                Key={
                    "state_key": state_key
                },

                UpdateExpression="""
                    SET
                        reward_sum =
                            if_not_exists(
                                reward_sum,
                                :zero_decimal
                            ) + :reward,

                        reward_count =
                            if_not_exists(
                                reward_count,
                                :zero
                            ) + :one,

                        success_count =
                            if_not_exists(
                                success_count,
                                :zero
                            ) + :success,

                        failure_count =
                            if_not_exists(
                                failure_count,
                                :zero
                            ) + :failure,

                        total_latency_ms =
                            if_not_exists(
                                total_latency_ms,
                                :zero_decimal
                            ) + :latency,

                        total_output_tokens =
                            if_not_exists(
                                total_output_tokens,
                                :zero
                            ) + :tokens,

                        updated_at =
                            :timestamp,

                        last_reward =
                            :last_reward,

                        last_success =
                            :last_success,

                        last_latency_ms =
                            :last_latency,

                        last_output_tokens =
                            :last_tokens,

                        last_model =
                            :model,

                        last_complexity =
                            :complexity,

                        last_update_version =
                            :version
                """,

                ExpressionAttributeValues={

                    ":zero_decimal":
                        Decimal("0"),

                    ":zero":
                        0,

                    ":one":
                        1,

                    ":reward":
                        Decimal(
                            str(reward)
                        ),

                    ":success":
                        (
                            1
                            if success
                            else 0
                        ),

                    ":failure":
                        (
                            0
                            if success
                            else 1
                        ),

                    ":latency":
                        Decimal(
                            str(
                                latency_ms
                            )
                        ),

                    ":tokens":
                        int(
                            output_tokens
                        ),

                    ":timestamp":
                        timestamp,

                    ":last_reward":
                        Decimal(
                            str(reward)
                        ),

                    ":last_success":
                        bool(success),

                    ":last_latency":
                        Decimal(
                            str(
                                latency_ms
                            )
                        ),

                    ":last_tokens":
                        int(
                            output_tokens
                        ),

                    ":model":
                        model,

                    ":complexity":
                        complexity,

                    ":version":
                        "V7.3",
                },

                ReturnValues="ALL_NEW",
            )
        )

        # --------------------------------------------------
        # GET ATTRIBUTES FROM UPDATE RESPONSE
        #
        # No GetItem required.
        # --------------------------------------------------

        attributes = (
            first_response.get(
                "Attributes",
                {}
            )
        )

        # --------------------------------------------------
        # V7.3 — CALCULATE RUNNING AVERAGE REWARD
        # --------------------------------------------------

        reward_sum = Decimal(
            str(
                attributes.get(
                    "reward_sum",
                    "0"
                )
            )
        )

        reward_count = int(
            attributes.get(
                "reward_count",
                1
            )
        )

        if reward_count > 0:

            average_reward = (
                reward_sum
                /
                Decimal(
                    reward_count
                )
            )

        else:

            average_reward = Decimal(
                "0"
            )

        # --------------------------------------------------
        # V7.4 — CALCULATE MODEL STATISTICS
        #
        # This uses the already-updated V7.3 attributes.
        # No additional DynamoDB read.
        # --------------------------------------------------

        v74_stats = (
            ModelPerformanceStatistics.calculate(
                attributes
            )
        )

        # --------------------------------------------------
        # V7.4 — CONVERT TO DYNAMODB VALUES
        # --------------------------------------------------

        v74_values = (
            ModelPerformanceStatistics.dynamodb_values(
                v74_stats
            )
        )

        # --------------------------------------------------
        # V7.3 + V7.4 — SECOND UPDATE
        #
        # Still only UpdateItem.
        # --------------------------------------------------

        second_response = (
            self.table.update_item(

                Key={
                    "state_key": state_key
                },

                UpdateExpression="""
                    SET

                        reward =
                            :average_reward,

                        v74_avg_reward =
                            :v74_avg_reward,

                        v74_success_rate =
                            :v74_success_rate,

                        v74_failure_rate =
                            :v74_failure_rate,

                        v74_avg_latency_ms =
                            :v74_avg_latency,

                        v74_avg_output_tokens =
                            :v74_avg_tokens,

                        v74_evidence_confidence =
                            :v74_evidence_confidence,

                        v74_evidence_adjusted_reward =
                            :v74_adjusted_reward,

                        v74_version =
                            :v74_version,

                        v74_updated_at =
                            :v74_updated_at
                """,

                ExpressionAttributeValues={

                    # Existing V7.3 field
                    ":average_reward":
                        average_reward,

                    # V7.4 fields
                    **v74_values,
                },

                ReturnValues="ALL_NEW",
            )
        )

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        final_attributes = (
            second_response.get(
                "Attributes",
                attributes
            )
        )

        return final_attributes