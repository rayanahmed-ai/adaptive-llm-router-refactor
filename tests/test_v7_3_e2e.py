import os
import sys

sys.path.append(
    "/home/sagemaker-user/adaptive-llm-routing"
)

from app.router.adaptive import AdaptiveRouterV2
from app.models.bedrock import BedrockClient
from app.models.registry import registry
from app.evaluation.reward import RewardCalculator
from app.evaluation.state_updater import (
    DynamoDBRewardUpdater,
)


def main():

    print()
    print("=" * 80)
    print("V7.3 — REWARD + DYNAMODB E2E TEST")
    print("=" * 80)

    # ======================================================
    # QUERY
    # ======================================================

    query = "What is Python?"

    print()
    print("Prompt:")
    print(query)

    # ======================================================
    # V2 TRAINING
    # ======================================================

    router = AdaptiveRouterV2(
        label_cache_path=(
            "notebooks/data/evaluation/"
            "v1_corrected_complexity_labels.json"
        )
    )

    dataset = router.load_data()

    queries, labels = (
        router.prepare_v2_data(
            dataset
        )
    )

    router.fit(
        queries,
        labels
    )

    print()
    print("V2 training: PASS")

    # ======================================================
    # ROUTING
    # ======================================================

    route_result = router.route_model(
        query
    )

    complexity = route_result[
        "complexity"
    ]

    confidence = float(
        route_result[
            "confidence"
        ]
    )

    selected_model = route_result[
        "model"
    ]

    print()
    print(
        f"Complexity: {complexity}"
    )
    print(
        f"Confidence: {confidence:.4f}"
    )
    print(
        f"Selected model: {selected_model}"
    )

    # ======================================================
    # MODEL REGISTRY
    # ======================================================

    actual_model_id = registry.resolve(
        selected_model
    )

    print(
        f"Bedrock model: {actual_model_id}"
    )

    print()
    print("Model resolution: PASS")

    # ======================================================
    # BEDROCK
    # ======================================================

    client = BedrockClient(
        region_name="eu-central-1"
    )

    result = client.invoke(
        model_id=actual_model_id,
        prompt=query,
        complexity=complexity,
        confidence=confidence,
        decision_type="ROUTING",
    )

    if not result["success"]:
        raise RuntimeError(
            "Bedrock inference failed: "
            f"{result.get('error')}"
        )

    latency_ms = float(
        result["latency_ms"]
    )

    output_tokens = int(
        result.get(
            "output_tokens",
            0
        )
    )

    print()
    print("Bedrock execution: PASS")
    print(
        f"Latency: {latency_ms:.2f} ms"
    )
    print(
        f"Output tokens: {output_tokens}"
    )

    # ======================================================
    # REWARD
    # ======================================================

    reward_calculator = RewardCalculator()

    reward_result = (
        reward_calculator.calculate(
            success=True,
            latency_ms=latency_ms,
            output_tokens=output_tokens,
        )
    )

    reward = float(
        reward_result.reward
    )

    print()
    print("Reward calculation: PASS")
    print(
        f"Reward: {reward:.4f}"
    )
    print(
        f"Success score: "
        f"{reward_result.success_score:.4f}"
    )
    print(
        f"Latency score: "
        f"{reward_result.latency_score:.4f}"
    )
    print(
        f"Efficiency score: "
        f"{reward_result.efficiency_score:.4f}"
    )

    # ======================================================
    # DYNAMODB KEY
    # ======================================================

    state_key = (
        f"STATE#{complexity}#"
        f"{selected_model}"
    )

    print()
    print(
        f"DynamoDB state key: {state_key}"
    )

    # ======================================================
    # DYNAMODB UPDATE
    # ======================================================

    updater = DynamoDBRewardUpdater()

    attributes = updater.update_model(
        complexity=complexity,
        model=selected_model,
        reward=reward,
        success=True,
        latency_ms=latency_ms,
        output_tokens=output_tokens,
    )

    print()
    print("DynamoDB update: PASS")

    print()
    print("Updated V6 state:")

    for key in sorted(attributes):
        print(
            f"  {key}: {attributes[key]}"
        )

    # ======================================================
    # FINAL
    # ======================================================

    print()
    print("=" * 80)
    print("V7.3 END-TO-END TEST PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
