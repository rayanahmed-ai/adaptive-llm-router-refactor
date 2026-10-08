from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

from app.models.v7_bedrock import V7BedrockClient
from app.router.v7_7 import V77FinalAdaptiveRouter


# ============================================================
# BENCHMARK DATA
# ============================================================

BENCHMARK_PROMPTS = [
    {
        "id": "b1",
        "prompt": "What is Python?",
        "expected_complexity": "simple",
        "keywords": [
            "python",
            "programming",
        ],
    },
    {
        "id": "b2",
        "prompt": (
            "Explain gradient descent and show "
            "a simple Python example."
        ),
        "expected_complexity": "medium",
        "keywords": [
            "gradient",
            "learning",
            "python",
        ],
    },
    {
        "id": "b3",
        "prompt": (
            "Write a Python function to reverse "
            "a singly linked list and explain "
            "the time complexity."
        ),
        "expected_complexity": "medium",
        "keywords": [
            "linked",
            "list",
            "time",
            "complexity",
        ],
    },
    {
        "id": "b4",
        "prompt": (
            "Explain the CAP theorem in distributed "
            "systems with an example."
        ),
        "expected_complexity": "medium",
        "keywords": [
            "consistency",
            "availability",
            "partition",
        ],
    },
    {
        "id": "b5",
        "prompt": (
            "Design an end-to-end machine learning "
            "pipeline for financial fraud detection "
            "including data preparation, training, "
            "evaluation, deployment and monitoring."
        ),
        "expected_complexity": "complex",
        "keywords": [
            "data",
            "training",
            "evaluation",
            "deployment",
            "monitoring",
        ],
    },
    {
        "id": "b6",
        "prompt": (
            "Design a production architecture for "
            "an adaptive multi-model LLM router that "
            "balances response quality, latency, "
            "reliability and inference cost."
        ),
        "expected_complexity": "complex",
        "keywords": [
            "model",
            "quality",
            "latency",
            "reliability",
            "cost",
        ],
    },
]


# ============================================================
# MODEL STATISTICS
# ============================================================

DEFAULT_MODEL_STATS = [
    {
        "model": "small_model",
        "sample_count": 50,
        "avg_reward": 0.72,
        "success_rate": 0.94,
        "avg_latency_ms": 1100.0,
        "avg_output_tokens": 250,
        "evidence_confidence": 0.918,
        "evidence_adjusted_reward": 0.661,
    },
    {
        "model": "medium_model",
        "sample_count": 35,
        "avg_reward": 0.81,
        "success_rate": 0.97,
        "avg_latency_ms": 1800.0,
        "avg_output_tokens": 320,
        "evidence_confidence": 0.826,
        "evidence_adjusted_reward": 0.670,
    },
    {
        "model": "large_model",
        "sample_count": 5,
        "avg_reward": 0.94,
        "success_rate": 1.00,
        "avg_latency_ms": 3000.0,
        "avg_output_tokens": 500,
        "evidence_confidence": 0.221,
        "evidence_adjusted_reward": 0.208,
    },
]


# ============================================================
# RESULT
# ============================================================

@dataclass
class BenchmarkResult:

    strategy: str
    prompt_id: str
    model: str
    success: bool
    latency_ms: float
    output_tokens: int
    quality_score: float


# ============================================================
# BENCHMARK RUNNER
# ============================================================

class V81Benchmark:

    def __init__(
        self,
        model_stats: list[dict[str, Any]] | None = None,
        seed: int = 42,
    ):

        self.model_stats = (
            model_stats
            if model_stats is not None
            else DEFAULT_MODEL_STATS
        )

        self.bedrock = V7BedrockClient(
            region="eu-north-1"
        )

        self.random = random.Random(
            seed
        )

        self.v77 = V77FinalAdaptiveRouter(
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            quality_weight=0.45,
            reliability_weight=0.20,
            latency_weight=0.20,
            cost_weight=0.15,
            seed=seed,
        )

    # ========================================================
    # QUALITY SCORING
    # ========================================================

    @staticmethod
    def quality_score(
        response: str,
        keywords: list[str],
    ) -> float:

        if not response:
            return 0.0

        text = response.lower()

        matches = 0

        for keyword in keywords:

            if keyword.lower() in text:
                matches += 1

        if not keywords:
            return 1.0

        return matches / len(
            keywords
        )

    # ========================================================
    # MODEL SELECTION STRATEGIES
    # ========================================================

    def select_model(
        self,
        strategy: str,
        prompt_data: dict[str, Any],
    ) -> str:

        if strategy == "strongest":

            return "large_model"

        if strategy == "cheapest":

            return "small_model"

        if strategy == "random":

            return self.random.choice(
                [
                    "small_model",
                    "medium_model",
                    "large_model",
                ]
            )

        if strategy == "static_complexity":

            mapping = {
                "simple": "small_model",
                "medium": "medium_model",
                "complex": "large_model",
            }

            return mapping[
                prompt_data[
                    "expected_complexity"
                ]
            ]

        if strategy == "v7_adaptive":

            decision = self.v77.decide(
                model_stats=self.model_stats,
                cost_data=None,
            )

            return decision.selected_model

        raise ValueError(
            f"Unknown strategy: {strategy}"
        )

    # ========================================================
    # LIVE INFERENCE
    # ========================================================

    def invoke(
        self,
        model: str,
        prompt: str,
        keywords: list[str],
    ) -> tuple[
        bool,
        float,
        int,
        float,
    ]:

        start = time.perf_counter()

        response = self.bedrock.invoke(
            logical_model=model,
            prompt=prompt,
            max_tokens=256,
            temperature=0.2,
        )

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        success = bool(
            response.get(
                "success",
                False,
            )
        )

        raw_response = response.get(
            "response",
            "",
        )

        if not isinstance(
            raw_response,
            str,
        ):

            raw_response = str(
                raw_response
            )

        output_tokens = int(
            response.get(
                "output_tokens",
                0,
            )
            or 0
        )

        quality = (
            self.quality_score(
                raw_response,
                keywords,
            )
            if success
            else 0.0
        )

        return (
            success,
            elapsed,
            output_tokens,
            quality,
        )

    # ========================================================
    # RUN
    # ========================================================

    def run(self) -> list[BenchmarkResult]:

        strategies = [
            "strongest",
            "cheapest",
            "random",
            "static_complexity",
            "v7_adaptive",
        ]

        results = []

        for strategy in strategies:

            print()
            print(
                "=" * 100
            )

            print(
                f"STRATEGY: {strategy}"
            )

            print(
                "=" * 100
            )

            for prompt_data in BENCHMARK_PROMPTS:

                model = self.select_model(
                    strategy,
                    prompt_data,
                )

                print()
                print(
                    f"{prompt_data['id']} "
                    f"→ {model}"
                )

                (
                    success,
                    latency,
                    output_tokens,
                    quality,
                ) = self.invoke(
                    model=model,
                    prompt=prompt_data[
                        "prompt"
                    ],
                    keywords=prompt_data[
                        "keywords"
                    ],
                )

                print(
                    f"success={success} "
                    f"latency={latency:.2f} ms "
                    f"tokens={output_tokens} "
                    f"quality={quality:.3f}"
                )

                results.append(
                    BenchmarkResult(
                        strategy=strategy,
                        prompt_id=prompt_data[
                            "id"
                        ],
                        model=model,
                        success=success,
                        latency_ms=latency,
                        output_tokens=output_tokens,
                        quality_score=quality,
                    )
                )

        return results

    # ========================================================
    # SUMMARY
    # ========================================================

    @staticmethod
    def summarize(
        results: list[BenchmarkResult],
    ) -> list[dict[str, Any]]:

        strategies = sorted(
            {
                result.strategy
                for result in results
            }
        )

        summary = []

        for strategy in strategies:

            rows = [
                result
                for result in results
                if result.strategy == strategy
            ]

            latencies = [
                result.latency_ms
                for result in rows
                if result.success
            ]

            qualities = [
                result.quality_score
                for result in rows
            ]

            tokens = [
                result.output_tokens
                for result in rows
            ]

            successes = [
                result.success
                for result in rows
            ]

            summary.append(
                {
                    "strategy": strategy,
                    "requests": len(rows),
                    "success_rate":
                        sum(successes)
                        / len(successes),
                    "avg_quality":
                        statistics.mean(
                            qualities
                        ),
                    "avg_latency_ms":
                        statistics.mean(
                            latencies
                        )
                        if latencies
                        else 0.0,
                    "p95_latency_ms":
                        (
                            sorted(
                                latencies
                            )[
                                max(
                                    0,
                                    int(
                                        len(latencies)
                                        * 0.95
                                    )
                                    - 1,
                                )
                            ]
                            if latencies
                            else 0.0
                        ),
                    "avg_output_tokens":
                        statistics.mean(
                            tokens
                        )
                        if tokens
                        else 0.0,
                    "models_used":
                        sorted(
                            {
                                result.model
                                for result in rows
                            }
                        ),
                }
            )

        return summary