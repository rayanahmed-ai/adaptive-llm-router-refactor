from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import boto3
from botocore.exceptions import ClientError

from app.evaluation.v8_benchmark import (
    BenchmarkResult,
)


# ============================================================
# PATHS / CONFIG
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "notebooks"
    / "data"
    / "evaluation"
    / "v8_2"
)


# ============================================================
# HELPERS
# ============================================================

def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    if not values:
        return 0.0

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * percentile_value
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower]
        * (1.0 - weight)
        + ordered[upper]
        * weight
    )


def safe_mean(
    values: list[float],
) -> float:

    return (
        mean(values)
        if values
        else 0.0
    )


# ============================================================
# V8.2 EVALUATOR
# ============================================================

class V82Evaluator:

    VERSION = "V8.2"

    def __init__(
        self,
        output_dir: str | Path | None = None,
    ):

        self.output_dir = Path(
            output_dir
            if output_dir is not None
            else DEFAULT_OUTPUT_DIR
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.s3_bucket = os.getenv(
            "S3_BUCKET"
        )

        self.s3_prefix = os.getenv(
            "S3_PREFIX",
            "adaptive-llm-router",
        )

        self.cloudwatch_namespace = os.getenv(
            "CLOUDWATCH_NAMESPACE"
        )

    # ========================================================
    # GROUP RESULTS
    # ========================================================

    @staticmethod
    def _group_results(
        results: list[BenchmarkResult],
    ) -> dict[str, list[BenchmarkResult]]:

        grouped = {}

        for result in results:

            grouped.setdefault(
                result.strategy,
                [],
            ).append(result)

        return grouped

    # ========================================================
    # STRATEGY SUMMARY
    # ========================================================

    def _summarize_strategy(
        self,
        strategy: str,
        rows: list[BenchmarkResult],
    ) -> dict[str, Any]:

        latencies = [
            float(row.latency_ms)
            for row in rows
            if row.success
        ]

        qualities = [
            float(row.quality_score)
            for row in rows
        ]

        tokens = [
            int(row.output_tokens)
            for row in rows
        ]

        success_values = [
            1 if row.success else 0
            for row in rows
        ]

        model_counts = Counter(
            row.model
            for row in rows
        )

        return {
            "strategy": strategy,
            "requests": len(rows),

            "successful_requests":
                sum(success_values),

            "failed_requests":
                len(rows)
                - sum(success_values),

            "success_rate":
                safe_mean(
                    success_values
                ),

            "average_quality":
                safe_mean(
                    qualities
                ),

            "median_quality":
                median(qualities)
                if qualities
                else 0.0,

            "average_latency_ms":
                safe_mean(
                    latencies
                ),

            "median_latency_ms":
                median(latencies)
                if latencies
                else 0.0,

            "p95_latency_ms":
                percentile(
                    latencies,
                    0.95,
                ),

            "average_output_tokens":
                safe_mean(
                    tokens
                ),

            "models_used":
                sorted(
                    model_counts.keys()
                ),

            "model_distribution":
                dict(model_counts),

            # IMPORTANT:
            # No fake cost number.
            "cost_available":
                False,

            "average_cost_usd":
                None,
        }

    # ========================================================
    # RELATIVE METRICS
    # ========================================================

    @staticmethod
    def _add_relative_metrics(
        summaries: list[dict[str, Any]],
    ) -> None:

        strongest = next(
            (
                row
                for row in summaries
                if row["strategy"]
                == "strongest"
            ),
            None,
        )

        if strongest is None:
            return

        strongest_quality = float(
            strongest[
                "average_quality"
            ]
        )

        strongest_latency = float(
            strongest[
                "average_latency_ms"
            ]
        )

        strongest_p95 = float(
            strongest[
                "p95_latency_ms"
            ]
        )

        for row in summaries:

            quality = float(
                row[
                    "average_quality"
                ]
            )

            latency = float(
                row[
                    "average_latency_ms"
                ]
            )

            p95 = float(
                row[
                    "p95_latency_ms"
                ]
            )

            if strongest_quality > 0:

                row[
                    "quality_retention_vs_strongest"
                ] = (
                    quality
                    / strongest_quality
                )

                row[
                    "quality_gap_vs_strongest"
                ] = (
                    quality
                    - strongest_quality
                )

            else:

                row[
                    "quality_retention_vs_strongest"
                ] = None

                row[
                    "quality_gap_vs_strongest"
                ] = None

            if strongest_latency > 0:

                row[
                    "latency_change_vs_strongest_ms"
                ] = (
                    latency
                    - strongest_latency
                )

                row[
                    "latency_reduction_vs_strongest"
                ] = (
                    1.0
                    - (
                        latency
                        / strongest_latency
                    )
                )

            else:

                row[
                    "latency_change_vs_strongest_ms"
                ] = None

                row[
                    "latency_reduction_vs_strongest"
                ] = None

            if strongest_p95 > 0:

                row[
                    "p95_reduction_vs_strongest"
                ] = (
                    1.0
                    - (
                        p95
                        / strongest_p95
                    )
                )

            else:

                row[
                    "p95_reduction_vs_strongest"
                ] = None

    # ========================================================
    # QUALITY-LATENCY FRONTIER
    # ========================================================

    @staticmethod
    def _add_frontier_flags(
        summaries: list[dict[str, Any]],
    ) -> None:

        for candidate in summaries:

            candidate_quality = float(
                candidate[
                    "average_quality"
                ]
            )

            candidate_latency = float(
                candidate[
                    "average_latency_ms"
                ]
            )

            dominated = False

            for other in summaries:

                if other is candidate:
                    continue

                other_quality = float(
                    other[
                        "average_quality"
                    ]
                )

                other_latency = float(
                    other[
                        "average_latency_ms"
                    ]
                )

                quality_better_or_equal = (
                    other_quality
                    >= candidate_quality
                )

                latency_better_or_equal = (
                    other_latency
                    <= candidate_latency
                )

                strictly_better = (
                    other_quality
                    > candidate_quality
                    or
                    other_latency
                    < candidate_latency
                )

                if (
                    quality_better_or_equal
                    and latency_better_or_equal
                    and strictly_better
                ):

                    dominated = True
                    break

            candidate[
                "pareto_quality_latency"
            ] = not dominated

    # ========================================================
    # FULL EVALUATION
    # ========================================================

    def evaluate(
        self,
        results: list[BenchmarkResult],
    ) -> dict[str, Any]:

        if not results:

            raise ValueError(
                "No benchmark results supplied."
            )

        grouped = self._group_results(
            results
        )

        summaries = []

        for strategy, rows in grouped.items():

            summaries.append(
                self._summarize_strategy(
                    strategy,
                    rows,
                )
            )

        summaries.sort(
            key=lambda row:
                row["average_quality"],
            reverse=True,
        )

        self._add_relative_metrics(
            summaries
        )

        self._add_frontier_flags(
            summaries
        )

        return {
            "evaluation_version":
                self.VERSION,

            "generated_at":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            "benchmark_requests":
                len(results),

            "strategies_evaluated":
                len(summaries),

            "cost_data_available":
                False,

            "cost_note":
                (
                    "Real inference cost was not "
                    "available in V8.1. Cost metrics "
                    "are intentionally omitted rather "
                    "than estimated."
                ),

            "strategies":
                summaries,

            "raw_results":
                [
                    {
                        "strategy":
                            row.strategy,

                        "prompt_id":
                            row.prompt_id,

                        "model":
                            row.model,

                        "success":
                            row.success,

                        "latency_ms":
                            row.latency_ms,

                        "output_tokens":
                            row.output_tokens,

                        "quality_score":
                            row.quality_score,
                    }
                    for row in results
                ],
        }

    # ========================================================
    # SAVE JSON
    # ========================================================

    def save_json(
        self,
        report: dict[str, Any],
    ) -> Path:

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        path = (
            self.output_dir
            / f"v8_2_evaluation_{timestamp}.json"
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=2,
            )

        return path

    # ========================================================
    # SAVE CSV
    # ========================================================

    def save_csv(
        self,
        report: dict[str, Any],
    ) -> Path:

        timestamp = datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )

        path = (
            self.output_dir
            / f"v8_2_summary_{timestamp}.csv"
        )

        rows = report[
            "strategies"
        ]

        if not rows:
            raise ValueError(
                "No strategy summaries."
            )

        fieldnames = list(
            rows[0].keys()
        )

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(rows)

        return path

    # ========================================================
    # OPTIONAL S3 UPLOAD
    # ========================================================

    def upload_to_s3(
        self,
        local_path: Path,
    ) -> str | None:

        if not self.s3_bucket:

            return None

        s3 = boto3.client(
            "s3"
        )

        key = (
            f"{self.s3_prefix}/"
            f"evaluation/v8_2/"
            f"{local_path.name}"
        )

        s3.upload_file(
            str(local_path),
            self.s3_bucket,
            key,
        )

        return (
            f"s3://{self.s3_bucket}/{key}"
        )

    # ========================================================
    # OPTIONAL CLOUDWATCH
    # ========================================================

    def publish_cloudwatch(
        self,
        report: dict[str, Any],
    ) -> bool:

        if not self.cloudwatch_namespace:

            return False

        cloudwatch = boto3.client(
            "cloudwatch"
        )

        metrics = []

        for row in report[
            "strategies"
        ]:

            strategy = row[
                "strategy"
            ]

            metrics.extend(
                [
                    {
                        "MetricName":
                            "AverageQuality",

                        "Dimensions": [
                            {
                                "Name":
                                    "Strategy",

                                "Value":
                                    strategy,
                            }
                        ],

                        "Value":
                            row[
                                "average_quality"
                            ],

                        "Unit":
                            "None",
                    },
                    {
                        "MetricName":
                            "AverageLatencyMs",

                        "Dimensions": [
                            {
                                "Name":
                                    "Strategy",

                                "Value":
                                    strategy,
                            }
                        ],

                        "Value":
                            row[
                                "average_latency_ms"
                            ],

                        "Unit":
                            "Milliseconds",
                    },
                    {
                        "MetricName":
                            "SuccessRate",

                        "Dimensions": [
                            {
                                "Name":
                                    "Strategy",

                                "Value":
                                    strategy,
                            }
                        ],

                        "Value":
                            row[
                                "success_rate"
                            ],

                        "Unit":
                            "None",
                    },
                ]
            )

        if metrics:

            cloudwatch.put_metric_data(
                Namespace=
                    self.cloudwatch_namespace,
                MetricData=metrics,
            )

        return True

    # ========================================================
    # PERSIST EVERYTHING
    # ========================================================

    def persist(
        self,
        report: dict[str, Any],
    ) -> dict[str, Any]:

        json_path = self.save_json(
            report
        )

        csv_path = self.save_csv(
            report
        )

        s3_locations = []

        for path in [
            json_path,
            csv_path,
        ]:

            try:

                location = (
                    self.upload_to_s3(
                        path
                    )
                )

                if location:
                    s3_locations.append(
                        location
                    )

            except (
                ClientError,
                Exception,
            ) as exc:

                print(
                    f"S3 upload warning: {exc}"
                )

        cloudwatch_published = False

        try:

            cloudwatch_published = (
                self.publish_cloudwatch(
                    report
                )
            )

        except (
            ClientError,
            Exception,
        ) as exc:

            print(
                f"CloudWatch warning: {exc}"
            )

        return {
            "json_path":
                str(json_path),

            "csv_path":
                str(csv_path),

            "s3_locations":
                s3_locations,

            "cloudwatch_published":
                cloudwatch_published,
        }