import statistics
from collections import defaultdict
from threading import Lock
from typing import Dict, Any


class MetricsCollector:

    def __init__(self):
        self.lock = Lock()

        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        self.latencies = []

        self.model_usage = defaultdict(int)
        self.model_latencies = defaultdict(list)

        self.complexity_usage = defaultdict(int)
        self.decision_usage = defaultdict(int)

        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def record(
        self,
        *,
        model: str,
        complexity: str,
        decision_type: str,
        latency_ms: float,
        success: bool,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:

        with self.lock:

            self.total_requests += 1

            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            self.latencies.append(latency_ms)

            self.model_usage[model] += 1
            self.model_latencies[model].append(latency_ms)

            self.complexity_usage[complexity] += 1
            self.decision_usage[decision_type] += 1

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

    def _percentile(self, values, percentile):
        if not values:
            return 0.0

        values = sorted(values)

        index = int(
            (percentile / 100) * (len(values) - 1)
        )

        return values[index]

    def snapshot(self) -> Dict[str, Any]:

        with self.lock:

            average_latency = (
                statistics.mean(self.latencies)
                if self.latencies
                else 0.0
            )

            p95_latency = self._percentile(
                self.latencies,
                95
            )

            success_rate = (
                self.successful_requests /
                self.total_requests
                if self.total_requests
                else 0.0
            )

            model_latency = {}

            for model, latencies in self.model_latencies.items():

                model_latency[model] = {
                    "requests": len(latencies),
                    "avg_latency_ms": (
                        statistics.mean(latencies)
                        if latencies
                        else 0.0
                    ),
                    "p95_latency_ms": self._percentile(
                        latencies,
                        95
                    ),
                }

            return {
                "version": "V7.2",

                "total_requests":
                    self.total_requests,

                "successful_requests":
                    self.successful_requests,

                "failed_requests":
                    self.failed_requests,

                "success_rate":
                    success_rate,

                "avg_latency_ms":
                    average_latency,

                "p95_latency_ms":
                    p95_latency,

                "model_usage":
                    dict(self.model_usage),

                "model_latency":
                    model_latency,

                "complexity_usage":
                    dict(self.complexity_usage),

                "decision_usage":
                    dict(self.decision_usage),

                "total_input_tokens":
                    self.total_input_tokens,

                "total_output_tokens":
                    self.total_output_tokens,
            }


metrics = MetricsCollector()