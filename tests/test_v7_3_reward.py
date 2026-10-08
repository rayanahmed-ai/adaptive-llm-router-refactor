from app.evaluation.reward import RewardCalculator


def main():

    print()
    print("=" * 100)
    print("V7.3 — REWARD CALCULATION TEST")
    print("=" * 100)

    calculator = RewardCalculator()

    # --------------------------------------------------
    # SUCCESSFUL REQUEST
    # --------------------------------------------------

    result = calculator.calculate(

        success=True,

        latency_ms=2208.87,

        output_tokens=322,
    )

    print()
    print("SUCCESSFUL REQUEST")
    print("-" * 60)

    print(
        f"Reward: "
        f"{result.reward:.4f}"
    )

    print(
        f"Success score: "
        f"{result.success_score:.4f}"
    )

    print(
        f"Latency score: "
        f"{result.latency_score:.4f}"
    )

    print(
        f"Efficiency score: "
        f"{result.efficiency_score:.4f}"
    )

    # --------------------------------------------------
    # FAILED REQUEST
    # --------------------------------------------------

    failed = calculator.calculate(

        success=False,

        latency_ms=173.24,

        output_tokens=0,
    )

    print()
    print("FAILED REQUEST")
    print("-" * 60)

    print(
        f"Reward: "
        f"{failed.reward:.4f}"
    )

    print(
        f"Success score: "
        f"{failed.success_score:.4f}"
    )

    print(
        f"Latency score: "
        f"{failed.latency_score:.4f}"
    )

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    assert 0.0 <= result.reward <= 1.0

    assert 0.0 <= failed.reward <= 1.0

    assert result.reward > failed.reward

    print()
    print("=" * 100)
    print("V7.3 REWARD TEST PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()
    