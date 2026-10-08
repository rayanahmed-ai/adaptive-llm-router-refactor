from app.observability.metrics import MetricsCollector


def test_metrics_recording():

    collector = MetricsCollector()

    collector.record(
        model="eu.amazon.nova-lite-v1:0",
        complexity="simple",
        decision_type="EXPLOITATION",
        latency_ms=500,
        success=True,
        input_tokens=20,
        output_tokens=50,
    )

    collector.record(
        model="eu.amazon.nova-pro-v1:0",
        complexity="complex",
        decision_type="EXPLOITATION",
        latency_ms=1000,
        success=True,
        input_tokens=30,
        output_tokens=100,
    )

    snapshot = collector.snapshot()

    assert snapshot["total_requests"] == 2

    assert snapshot["successful_requests"] == 2

    assert snapshot["failed_requests"] == 0

    assert snapshot["success_rate"] == 1.0

    assert snapshot["total_input_tokens"] == 50

    assert snapshot["total_output_tokens"] == 150