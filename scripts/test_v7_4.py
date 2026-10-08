# ============================================================
# V7.4 — MODEL STATISTICS INSPECTION
# ============================================================

from app.evaluation.model_stats import ModelStatsStore


def main() -> None:

    store = ModelStatsStore(
        create_table=True
    )

    print()
    print("=" * 100)
    print("V7.4 — ADAPTIVE MODEL PERFORMANCE STATISTICS")
    print("=" * 100)

    statistics = (
        store.get_all_model_stats()
    )

    if not statistics:

        print()
        print(
            "No V7.4 statistics found."
        )

        print(
            "Run V7.3/V7.4 observations first."
        )

        return

    # ========================================================
    # FULL STATISTICS
    # ========================================================

    for item in statistics:

        print()
        print("-" * 100)

        print(
            f"MODEL: "
            f"{item['model']}"
        )

        print(
            f"Samples: "
            f"{item['sample_count']}"
        )

        print(
            f"Success Count: "
            f"{item['success_count']}"
        )

        print(
            f"Success Rate: "
            f"{item['success_rate']:.4f}"
        )

        print(
            f"Reward Sum: "
            f"{item['reward_sum']:.6f}"
        )

        print(
            f"Average Reward: "
            f"{item['avg_reward']:.6f}"
        )

        print(
            f"Average Latency: "
            f"{item['avg_latency_ms']:.2f} ms"
        )

        print(
            f"Average Cost: "
            f"${item['avg_cost_usd']:.8f}"
        )

        print(
            f"Evidence Confidence: "
            f"{item['evidence_confidence']:.6f}"
        )

        print(
            f"Evidence-Adjusted Reward: "
            f"{item['evidence_adjusted_reward']:.6f}"
        )

    # ========================================================
    # RANKING
    # ========================================================

    print()
    print("=" * 100)
    print("V7.4 — DESCRIPTIVE MODEL RANKING")
    print("=" * 100)

    ranked = (
        store.rank_models()
    )

    for position, item in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"{position}. "
            f"{item['model']:<20} "
            f"samples="
            f"{item['sample_count']:<5} "
            f"reward="
            f"{item['avg_reward']:.4f} "
            f"confidence="
            f"{item['evidence_confidence']:.4f} "
            f"adjusted="
            f"{item['evidence_adjusted_reward']:.4f}"
        )

    print()
    print("=" * 100)
    print("V7.4 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()