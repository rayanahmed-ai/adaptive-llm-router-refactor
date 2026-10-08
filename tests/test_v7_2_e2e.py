# import json

# from app.router.adaptive import AdaptiveRouterV2
# from app.models.bedrock import BedrockClient
# from app.observability.metrics import metrics


# def main():

#     print()
#     print("=" * 100)
#     print("V7.2 — END-TO-END TELEMETRY TEST")
#     print("=" * 100)

#     prompt = "What is Python?"

#     # =========================================================
#     # STEP 1 — CREATE AND TRAIN V2 ROUTER
#     # =========================================================

#     print()
#     print("=" * 100)
#     print("STEP 1 — TRAIN V2 ROUTER")
#     print("=" * 100)

#     router = AdaptiveRouterV2()

#     print("\nLoading Dolly dataset...")

#     dataset = router.load_data()

#     print("\nPreparing V2 data...")

#     queries, labels = router.prepare_v2_data(
#         dataset
#     )

#     print("\nTraining V2...")

#     training_result = router.fit(
#         queries,
#         labels
#     )

#     print()
#     print(
#         f"V2 validation accuracy: "
#         f"{training_result['accuracy']:.4f}"
#     )

#     # =========================================================
#     # STEP 2 — ROUTING
#     # =========================================================

#     print()
#     print("=" * 100)
#     print("STEP 2 — V2 ROUTING")
#     print("=" * 100)

#     print()
#     print(f"Prompt: {prompt}")

#     route = router.route_model(
#         prompt
#     )

#     complexity = route["complexity"]

#     confidence = route["confidence"]

#     selected_model = route["model"]

#     print()
#     print("V2 ROUTER RESULT")
#     print("-" * 60)

#     print(
#         f"Complexity: {complexity}"
#     )

#     print(
#         f"Confidence: {confidence:.4f}"
#     )

#     print(
#         f"Selected model: {selected_model}"
#     )

#     print(
#         f"Probabilities: "
#         f"{route['probabilities']}"
#     )

#     # =========================================================
#     # STEP 3 — REAL BEDROCK
#     # =========================================================

#     print()
#     print("=" * 100)
#     print("STEP 3 — REAL BEDROCK INFERENCE")
#     print("=" * 100)

#     bedrock = BedrockClient(
#         region_name="eu-central-1"
#     )

#     print("\nCalling Amazon Bedrock...")

#     result = bedrock.invoke(
#         model_id=selected_model,
#         prompt=prompt,
#         complexity=complexity,
#         confidence=confidence,
#         decision_type="ROUTING",
#     )

#     print()
#     print("BEDROCK RESULT")
#     print("-" * 60)

#     print(
#         f"Success: {result['success']}"
#     )

#     print(
#         f"Latency: "
#         f"{result['latency_ms']:.2f} ms"
#     )

#     print(
#         f"Input tokens: "
#         f"{result['input_tokens']}"
#     )

#     print(
#         f"Output tokens: "
#         f"{result['output_tokens']}"
#     )

#     if result["success"]:

#         print()
#         print("Response:")
#         print(result["response"])

#     else:

#         print()
#         print("ERROR:")
#         print(result["error"])

#     # =========================================================
#     # STEP 4 — V7.2 TELEMETRY
#     # =========================================================

#     print()
#     print("=" * 100)
#     print("STEP 4 — V7.2 TELEMETRY SNAPSHOT")
#     print("=" * 100)

#     snapshot = metrics.snapshot()

#     print(
#         json.dumps(
#             snapshot,
#             indent=2,
#             default=str,
#         )
#     )

#     # =========================================================
#     # STEP 5 — VALIDATION
#     # =========================================================

#     print()
#     print("=" * 100)
#     print("V7.2 VALIDATION")
#     print("=" * 100)

#     assert router.is_trained is True

#     assert snapshot["total_requests"] >= 1

#     if result["success"]:

#         assert (
#             snapshot["successful_requests"]
#             >= 1
#         )

#         assert (
#             snapshot["failed_requests"]
#             == 0
#             or snapshot["failed_requests"] >= 0
#         )

#     print()
#     print("V2 trained: PASS")
#     print("Routing decision: PASS")
#     print("Bedrock execution: PASS")
#     print("Telemetry collection: PASS")

#     print()
#     print("=" * 100)
#     print("V7.2 END-TO-END TEST COMPLETE")
#     print("=" * 100)


# if __name__ == "__main__":
#     main()
import json

from app.router.adaptive import AdaptiveRouterV2
from app.models.bedrock import BedrockClient
from app.models.registry import registry
from app.observability.metrics import metrics


def main():

    print()
    print("=" * 100)
    print("V7.2 — END-TO-END TELEMETRY TEST")
    print("=" * 100)

    prompt = "What is Python?"

    # =========================================================
    # STEP 1 — CREATE AND TRAIN V2 ROUTER
    # =========================================================

    print()
    print("=" * 100)
    print("STEP 1 — TRAIN V2 ROUTER")
    print("=" * 100)

    router = AdaptiveRouterV2()

    print("\nLoading Dolly dataset...")

    dataset = router.load_data()

    print("\nPreparing V2 data...")

    queries, labels = router.prepare_v2_data(
        dataset
    )

    print("\nTraining V2...")

    training_result = router.fit(
        queries,
        labels
    )

    print()
    print(
        f"V2 validation accuracy: "
        f"{training_result['accuracy']:.4f}"
    )

    # =========================================================
    # STEP 2 — ROUTING
    # =========================================================

    print()
    print("=" * 100)
    print("STEP 2 — V2 ROUTING")
    print("=" * 100)

    print()
    print(f"Prompt: {prompt}")

    route = router.route_model(
        prompt
    )

    complexity = route["complexity"]
    confidence = route["confidence"]
    selected_model = route["model"]

    print()
    print("V2 ROUTER RESULT")
    print("-" * 60)

    print(
        f"Complexity: {complexity}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    print(
        f"Selected model alias: {selected_model}"
    )

    print(
        f"Probabilities: "
        f"{route['probabilities']}"
    )

    # =========================================================
    # STEP 3 — RESOLVE MODEL ALIAS
    # =========================================================

    print()
    print("=" * 100)
    print("STEP 3 — RESOLVE BEDROCK MODEL")
    print("=" * 100)

    try:

        bedrock_model_id = registry.resolve(
            selected_model
        )

    except Exception as exc:

        print()
        print("MODEL REGISTRY ERROR:")
        print(str(exc))

        raise

    print()
    print(
        f"Routing alias: "
        f"{selected_model}"
    )

    print(
        f"Bedrock model ID: "
        f"{bedrock_model_id}"
    )

    # =========================================================
    # STEP 4 — REAL BEDROCK
    # =========================================================

    print()
    print("=" * 100)
    print("STEP 4 — REAL BEDROCK INFERENCE")
    print("=" * 100)

    bedrock = BedrockClient(
        region_name="eu-central-1"
    )

    print("\nCalling Amazon Bedrock...")

    result = bedrock.invoke(
        model_id=bedrock_model_id,
        prompt=prompt,
        complexity=complexity,
        confidence=confidence,
        decision_type="ROUTING",
    )

    print()
    print("BEDROCK RESULT")
    print("-" * 60)

    print(
        f"Success: {result['success']}"
    )

    print(
        f"Latency: "
        f"{result['latency_ms']:.2f} ms"
    )

    print(
        f"Input tokens: "
        f"{result['input_tokens']}"
    )

    print(
        f"Output tokens: "
        f"{result['output_tokens']}"
    )

    if result["success"]:

        print()
        print("Response:")
        print(result["response"])

    else:

        print()
        print("ERROR:")
        print(result["error"])

    # =========================================================
    # STEP 5 — V7.2 TELEMETRY
    # =========================================================

    print()
    print("=" * 100)
    print("STEP 5 — V7.2 TELEMETRY SNAPSHOT")
    print("=" * 100)

    snapshot = metrics.snapshot()

    print(
        json.dumps(
            snapshot,
            indent=2,
            default=str,
        )
    )

    # =========================================================
    # STEP 6 — VALIDATION
    # =========================================================

    print()
    print("=" * 100)
    print("V7.2 VALIDATION")
    print("=" * 100)

    # V2 training
    assert router.is_trained is True

    # Routing
    assert complexity in [
        "simple",
        "medium",
        "complex",
    ]

    assert confidence >= 0.0
    assert confidence <= 1.0

    assert selected_model in [
        "small_model",
        "medium_model",
        "large_model",
    ]

    # Registry
    assert isinstance(
        bedrock_model_id,
        str
    )

    assert len(
        bedrock_model_id
    ) > 0

    # Telemetry
    assert (
        snapshot["total_requests"] >= 1
    )

    if result["success"]:

        assert (
            snapshot["successful_requests"]
            >= 1
        )

        print()
        print("V2 trained: PASS")
        print("Routing decision: PASS")
        print("Model resolution: PASS")
        print("Bedrock execution: PASS")
        print("Telemetry collection: PASS")

    else:

        print()
        print("V2 trained: PASS")
        print("Routing decision: PASS")
        print("Model resolution: PASS")
        print("Bedrock execution: FAIL")
        print("Telemetry collection: PASS")

        raise RuntimeError(
            "V7.2 E2E test failed because "
            "the Bedrock inference failed: "
            f"{result['error']}"
        )

    print()
    print("=" * 100)
    print("V7.2 END-TO-END TEST PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()